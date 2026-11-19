"""A shared box, one Modal VM holding several tasks' slots, is ended only by ``terminate_shared_box``: every other Modal
terminate reads whether its sandbox is one, unless this process made it, and refuses to end one."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import modal
import pytest

from agent_env.attribution import PIPELINE_STEP_KEY
from agent_env.config import get_config, reset_config
from agent_env.providers.sandbox_providers import modal_sandbox
from agent_env.providers.sandbox_providers.modal_sandbox import (
    SHARED_BOX_TAG,
    SHARED_BOX_TAG_READ_FAILED_EVENT,
    ModalSandboxProvider,
    SharedBoxTerminateRefused,
)
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandboxProvider
from tst.unit.store.fakes import FakeImageStore

BOX = {SHARED_BOX_TAG: "box-1"}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    get_config().set_image_store(FakeImageStore())
    monkeypatch.setattr(modal_sandbox, "_TAG_READ_BACKOFF_SECONDS", 0)
    yield
    reset_config()


def modal_sb(tags=None, *, tag_reads=None, sandbox_id="sb-1") -> MagicMock:
    """A Modal sandbox whose tags are ``tags``, or whose reads go as ``tag_reads``, recording its terminate."""
    sb = MagicMock(object_id=sandbox_id)
    sb.get_tags.aio = AsyncMock(side_effect=tag_reads) if tag_reads else AsyncMock(return_value=dict(tags or {}))
    sb.terminate.aio = AsyncMock()
    sb.tunnels.aio = AsyncMock(return_value={})
    sb.wait_until_ready.aio = AsyncMock()
    sb._get_task_id.aio = AsyncMock(return_value="ta-1")
    sb.exec.aio = AsyncMock(side_effect=RuntimeError("no exec here"))
    return sb


def provider(cls):
    p = cls(app_name="agent-env-test")
    p._get_app = AsyncMock(return_value=MagicMock())
    p._get_client = AsyncMock(return_value=MagicMock())
    return p


@pytest.fixture
def modal_api(monkeypatch):
    """Patch Modal's create and from_id to hand out the sandbox a test sets as ``modal_api.sb``."""
    api = MagicMock()
    api.create = AsyncMock(side_effect=lambda *a, **kw: api.sb)
    api.from_id = AsyncMock(side_effect=lambda *a, **kw: api.sb)
    monkeypatch.setattr(modal.Sandbox, "create", MagicMock(aio=api.create))
    monkeypatch.setattr(modal.Sandbox, "from_id", MagicMock(aio=api.from_id))
    return api


async def created_vm(p):
    return await p.create_vm(setup_for_gateway=False)


async def created_container(p):
    return await p.create_container(image_name="img:latest", port=8000, env={}, expose_externally=False)


PROVIDERS = pytest.mark.parametrize("cls", [ModalVmSandboxProvider, ModalSandboxProvider], ids=["vm", "container"])


@pytest.mark.parametrize("create", [(ModalVmSandboxProvider, created_vm), (ModalSandboxProvider, created_container)],
                         ids=["vm", "container"])
@pytest.mark.asyncio
async def test_a_handle_this_process_created_terminates_with_no_tag_read(modal_api, create):
    cls, make = create
    modal_api.sb = modal_sb()
    handle = await make(provider(cls))

    await handle.terminate()

    modal_api.sb.terminate.aio.assert_awaited_once()
    modal_api.sb.get_tags.aio.assert_not_called()


@PROVIDERS
@pytest.mark.asyncio
async def test_terminate_sandbox_reads_the_tags_of_a_sandbox_that_is_no_box_then_terminates_it(modal_api, cls):
    modal_api.sb = modal_sb({PIPELINE_STEP_KEY: "t_s"})

    await provider(cls).terminate_sandbox("sb-1")

    modal_api.sb.get_tags.aio.assert_awaited_once()
    modal_api.sb.terminate.aio.assert_awaited_once()


@pytest.mark.parametrize("gone_at", ["from_id", "get_tags"])
@PROVIDERS
@pytest.mark.asyncio
async def test_terminate_sandbox_counts_a_sandbox_modal_no_longer_knows_as_gone(modal_api, cls, gone_at):
    modal_api.sb = modal_sb(tag_reads=[modal.exception.NotFoundError("Sandbox not found")])
    if gone_at == "from_id":
        modal_api.from_id.side_effect = modal.exception.NotFoundError("Sandbox not found")

    await provider(cls).terminate_sandbox("sb-1")

    modal_api.sb.terminate.aio.assert_not_called()


@PROVIDERS
@pytest.mark.asyncio
async def test_a_tag_read_that_fails_then_succeeds_terminates(modal_api, cls):
    modal_api.sb = modal_sb(tag_reads=[modal.exception.ConnectionError("reset"), {}])

    await provider(cls).terminate_sandbox("sb-1")

    modal_api.sb.terminate.aio.assert_awaited_once()


@PROVIDERS
@pytest.mark.asyncio
async def test_a_tag_read_that_keeps_failing_raises_terminates_nothing_and_says_so(modal_api, cls, caplog):
    modal_api.sb = modal_sb(tag_reads=modal.exception.ConnectionError("reset"))

    with caplog.at_level(logging.WARNING, logger="agent_env"), pytest.raises(modal.exception.ConnectionError):
        await provider(cls).terminate_sandbox("sb-1")

    modal_api.sb.terminate.aio.assert_not_called()
    [warning] = [r for r in caplog.records if getattr(r, "event", None) == SHARED_BOX_TAG_READ_FAILED_EVENT]
    assert warning.modal_sandbox_id == "sb-1"


@pytest.mark.asyncio
async def test_a_created_shared_box_handle_refuses_to_terminate(modal_api):
    modal_api.sb = modal_sb(BOX)
    handle = await provider(ModalVmSandboxProvider).create_shared_box("box-1", setup_for_gateway=False)

    with pytest.raises(SharedBoxTerminateRefused) as refused:
        await handle.terminate()

    assert (refused.value.sandbox_id, refused.value.box_id) == ("sb-1", "box-1")
    modal_api.sb.terminate.aio.assert_not_called()


@PROVIDERS
@pytest.mark.asyncio
async def test_a_reattached_handle_to_a_shared_box_refuses_to_terminate(modal_api, cls):
    modal_api.sb = modal_sb(BOX)
    handle = await provider(cls).get_sandbox("sb-1")

    for _ in range(2):
        with pytest.raises(SharedBoxTerminateRefused):
            await handle.terminate()

    assert handle.shared_box_id == "box-1"
    modal_api.sb.terminate.aio.assert_not_called()


@PROVIDERS
@pytest.mark.asyncio
async def test_terminate_sandbox_refuses_a_shared_box(modal_api, cls):
    modal_api.sb = modal_sb(BOX)

    with pytest.raises(SharedBoxTerminateRefused):
        await provider(cls).terminate_sandbox("sb-1")

    modal_api.sb.terminate.aio.assert_not_called()


@pytest.mark.asyncio
async def test_terminate_shared_box_ends_the_box_it_names(modal_api):
    modal_api.sb = modal_sb(BOX)

    await provider(ModalVmSandboxProvider).terminate_shared_box("sb-1", "box-1")

    modal_api.sb.terminate.aio.assert_awaited_once()


@pytest.mark.parametrize("sb", [
    pytest.param(lambda: modal_sb({SHARED_BOX_TAG: "box-2"}), id="another box"),
    pytest.param(lambda: modal_sb({PIPELINE_STEP_KEY: "t_s"}), id="no box"),
])
@pytest.mark.asyncio
async def test_terminate_shared_box_refuses_a_sandbox_that_isnt_that_box(modal_api, sb):
    modal_api.sb = sb()

    with pytest.raises(SharedBoxTerminateRefused):
        await provider(ModalVmSandboxProvider).terminate_shared_box("sb-1", "box-1")

    modal_api.sb.terminate.aio.assert_not_called()


@pytest.mark.asyncio
async def test_terminate_shared_box_raises_when_the_tags_cant_be_read(modal_api):
    modal_api.sb = modal_sb(tag_reads=modal.exception.ConnectionError("reset"))

    with pytest.raises(modal.exception.ConnectionError):
        await provider(ModalVmSandboxProvider).terminate_shared_box("sb-1", "box-1")

    modal_api.sb.terminate.aio.assert_not_called()


@pytest.mark.asyncio
async def test_create_shared_box_tags_the_vm_with_its_box_beside_its_attribution(modal_api):
    modal_api.sb = modal_sb()

    handle = await provider(ModalVmSandboxProvider).create_shared_box(
        "box-1", setup_for_gateway=False, attribution={PIPELINE_STEP_KEY: "t_s"})

    assert modal_api.create.call_args.kwargs["tags"] == {PIPELINE_STEP_KEY: "t_s", SHARED_BOX_TAG: "box-1"}
    assert handle.shared_box_id == "box-1"


@pytest.mark.parametrize("box_id, attribution", [
    pytest.param("box-1", {SHARED_BOX_TAG: "forged"}, id="attribution names the box tag"),
    pytest.param("", {}, id="empty id"),
    pytest.param("box 1", {}, id="id outside a tag's characters"),
    pytest.param("b" * 64, {}, id="id longer than a tag"),
])
@pytest.mark.asyncio
async def test_create_shared_box_refuses_before_creating(modal_api, box_id, attribution):
    with pytest.raises(ValueError):
        await provider(ModalVmSandboxProvider).create_shared_box(box_id, attribution=attribution)

    modal_api.create.assert_not_called()


@pytest.mark.asyncio
async def test_an_ordinary_vm_cant_be_given_the_box_tag_through_its_attribution(modal_api):
    with pytest.raises(ValueError):
        await provider(ModalVmSandboxProvider).create_vm(attribution={SHARED_BOX_TAG: "forged"})

    modal_api.create.assert_not_called()


@pytest.mark.asyncio
async def test_providers_other_than_modal_vm_make_no_shared_box():
    with pytest.raises(NotImplementedError):
        await ModalSandboxProvider().create_shared_box("box-1")


_ALLOWED = {"_terminate", "_terminate_created"}
_CREATE_CLEANUPS = {"create_container", "_created"}


def test_every_modal_terminate_goes_through_the_check_point_but_a_failed_creates_cleanup():
    """``terminate.aio(`` is called only in ``_terminate``, or where a create that failed ends its sandbox: the cancelled
    create's cleanup, or an ``except`` in a create."""
    stray = []
    sources = Path(modal_sandbox.__file__).parent.glob("modal_*.py")
    for source in sources:
        tree = ast.parse(source.read_text())
        for function in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            in_handler = {id(c) for h in ast.walk(function) if isinstance(h, ast.ExceptHandler) for c in ast.walk(h)}
            for call in (n for n in ast.walk(function) if isinstance(n, ast.Call)):
                func = call.func
                if not (isinstance(func, ast.Attribute) and func.attr == "aio"
                        and isinstance(func.value, ast.Attribute) and func.value.attr == "terminate"):
                    continue
                if function.name in _ALLOWED or (function.name in _CREATE_CLEANUPS and id(call) in in_handler):
                    continue
                stray.append(f"{source.name}:{call.lineno} in {function.name}")
    assert stray == []
