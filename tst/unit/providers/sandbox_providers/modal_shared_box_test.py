"""A shared box, one Modal VM holding several tasks' slots, is ended only by ``terminate_shared_box``: every other Modal
terminate reads whether its sandbox is one, unless this process made it, and refuses to end one."""

from __future__ import annotations

import ast
import json
import re
import logging
import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import modal
import pytest

from agent_env.attribution import PIPELINE_STEP_KEY
from agent_env.config import get_config, reset_config
from agent_env.providers.sandbox_providers import modal_sandbox
from agent_env.providers.sandbox_providers import modal_vm_sandbox
from agent_env.providers.sandbox_providers.modal_sandbox import (
    SHARED_BOX_TAG,
    SHARED_BOX_TAG_READ_FAILED_EVENT,
    UNREAD,
    WALLS_TAG,
    WALLS_VERSION,
    ModalSandboxProvider,
    SharedBoxMismatch,
    SharedBoxTerminateRefused,
)
from agent_env.providers.sandbox_providers.modal_vm_sandbox import (
    ModalVmSandbox,
    ModalVmSandboxProvider,
    _mark_walls_verified,
)
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
    sb.set_tags.aio = AsyncMock()
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
    create's cleanup, or an ``except`` in a create. Every module of agent_env that uses the Modal SDK is checked, and no
    module calls Modal's terminate RPC directly."""
    stray = []
    package = Path(modal_sandbox.__file__).parents[2]
    sources = [p for p in package.rglob("*.py") if re.search(r"^\s*(import modal|from modal)", p.read_text(), re.M)]
    assert Path(modal_sandbox.__file__) in sources
    for source in package.rglob("*.py"):
        assert "SandboxTerminate" not in source.read_text(), f"{source} calls Modal's terminate RPC directly"
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


@pytest.mark.asyncio
async def test_a_shared_box_is_never_rebooted_under_its_slots(modal_api):
    modal_api.sb = modal_sb()
    vm_provider = provider(ModalVmSandboxProvider)
    box = await vm_provider.create_shared_box("box-1", setup_for_gateway=False)
    box._exec_with_stdin = AsyncMock()

    with pytest.raises(SharedBoxTerminateRefused):
        await vm_provider.reset_vm(box, image="disk", env={})
    box._exec_with_stdin.assert_not_called()


WALLED = {**BOX, WALLS_TAG: WALLS_VERSION}


def test_a_shared_box_runs_from_the_vm_image_plus_a_pinned_checksummed_runsc_and_its_dockerd_config():
    [install, configure] = modal_vm_sandbox._SHARED_BOX_BUILD
    tarball = (f"https://storage.googleapis.com/gvisor/releases/release/{modal_vm_sandbox._GVISOR_RELEASE}"
               "/x86_64/gvisor.tar.bz2")

    assert re.fullmatch(r"\d{8}\.\d+", modal_vm_sandbox._GVISOR_RELEASE)
    assert re.fullmatch(r"[0-9a-f]{128}", modal_vm_sandbox._GVISOR_TARBALL_SHA512)
    assert install.index(tarball) < install.index(f"{modal_vm_sandbox._GVISOR_TARBALL_SHA512}  /tmp/gvisor.tar.bz2' "
                                                  "| sha512sum -c -") < install.index("tar -xjf")
    written = shlex.split(configure.split("printf '%s\\n' ", 1)[1].split(" > ")[0])[0]
    assert configure.endswith("> /etc/docker/daemon.json")
    assert json.loads(written) == {
        "default-runtime": "runsc",
        "runtimes": {"runsc": {"path": "/usr/local/bin/runsc"}},
        "userns-remap": "default",
        "no-new-privileges": True,
        "userland-proxy": False,
        "default-address-pools": [{"base": "10.251.0.0/16", "size": 24}],
    }
    assert modal_vm_sandbox._SHARED_BOX_IMAGE is not modal_vm_sandbox._VM_IMAGE


@pytest.mark.asyncio
async def test_only_a_shared_box_gets_the_box_image(modal_api):
    modal_api.sb = modal_sb()
    p = provider(ModalVmSandboxProvider)

    await p.create_shared_box("box-1", setup_for_gateway=False)
    await p.create_vm(setup_for_gateway=False)

    box, vm = (c.kwargs["image"] for c in modal_api.create.call_args_list)
    assert (box, vm) == (modal_vm_sandbox._SHARED_BOX_IMAGE, modal_vm_sandbox._VM_IMAGE)


@pytest.mark.parametrize("shared_box_id, walled", [("box-1", True), (None, False), (UNREAD, False)])
@pytest.mark.asyncio
async def test_setting_up_a_shared_box_walls_its_slots_once_dockerd_is_up(monkeypatch, shared_box_id, walled):
    calls = []
    monkeypatch.setattr(modal_vm_sandbox, "apply_walls_firewall", AsyncMock(side_effect=lambda s: calls.append("walls")))
    handle = ModalVmSandbox(modal_sb(), {}, shared_box_id=shared_box_id)
    handle._ensure_dockerd_running = AsyncMock(side_effect=lambda: calls.append("dockerd"))

    await handle.setup_vm_for_gateway([])

    assert calls == (["dockerd", "walls"] if walled else ["dockerd"])


@pytest.mark.parametrize("shared_box_id, walls, verified", [
    pytest.param("box-1", WALLS_VERSION, True, id="box with current walls"),
    pytest.param("box-1", None, False, id="box, walls unread or untagged"),
    pytest.param("box-1", "walls-v0", False, id="box with an older walls version"),
    pytest.param(None, WALLS_VERSION, False, id="no box"),
    pytest.param(UNREAD, WALLS_VERSION, False, id="box unread"),
])
def test_walls_are_verified_only_on_a_shared_box_tagged_with_the_current_version(shared_box_id, walls, verified):
    assert ModalVmSandbox(modal_sb(), {}, shared_box_id=shared_box_id, walls=walls).walls_verified is verified


@pytest.mark.asyncio
async def test_a_created_shared_box_has_walls_not_yet_verified(modal_api):
    modal_api.sb = modal_sb(WALLED)

    handle = await provider(ModalVmSandboxProvider).create_shared_box("box-1", setup_for_gateway=False)

    assert handle.walls_verified is False


@pytest.mark.asyncio
async def test_get_shared_box_reads_the_tags_once_for_both_the_box_and_its_walls(modal_api):
    modal_api.sb = modal_sb(WALLED)

    handle = await provider(ModalVmSandboxProvider).get_shared_box("sb-1", "box-1")

    assert (handle.shared_box_id, handle.walls_verified) == ("box-1", True)
    modal_api.sb.get_tags.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_shared_box_of_a_box_with_older_walls_is_unverified(modal_api):
    modal_api.sb = modal_sb({**BOX, WALLS_TAG: "walls-v0"})

    handle = await provider(ModalVmSandboxProvider).get_shared_box("sb-1", "box-1")

    assert (handle.shared_box_id, handle.walls_verified) == ("box-1", False)


@pytest.mark.parametrize("tags, found", [
    pytest.param({SHARED_BOX_TAG: "box-2", WALLS_TAG: WALLS_VERSION}, "box-2", id="another box"),
    pytest.param({WALLS_TAG: WALLS_VERSION}, None, id="no box"),
])
@pytest.mark.asyncio
async def test_get_shared_box_refuses_a_sandbox_that_isnt_that_box(modal_api, tags, found):
    modal_api.sb = modal_sb(tags)

    with pytest.raises(SharedBoxMismatch) as mismatch:
        await provider(ModalVmSandboxProvider).get_shared_box("sb-1", "box-1")

    assert (mismatch.value.box_id, mismatch.value.expected) == (found, "box-1")


@pytest.mark.asyncio
async def test_get_shared_box_raises_when_the_tags_cant_be_read(modal_api):
    modal_api.sb = modal_sb(tag_reads=modal.exception.ConnectionError("reset"))

    with pytest.raises(modal.exception.ConnectionError):
        await provider(ModalVmSandboxProvider).get_shared_box("sb-1", "box-1")


@pytest.mark.asyncio
async def test_a_box_reached_through_get_sandbox_never_claims_verified_walls(modal_api):
    modal_api.sb = modal_sb(WALLED)

    handle = await provider(ModalVmSandboxProvider).get_sandbox("sb-1")

    assert handle.walls_verified is False
    modal_api.sb.get_tags.aio.assert_not_called()


@pytest.mark.parametrize("create", [
    pytest.param(lambda p: p.create_vm(attribution={WALLS_TAG: WALLS_VERSION}), id="vm"),
    pytest.param(lambda p: p.create_shared_box("box-1", attribution={WALLS_TAG: WALLS_VERSION}), id="shared box"),
])
@pytest.mark.asyncio
async def test_no_sandbox_can_be_given_the_walls_tag_through_its_attribution(modal_api, create):
    with pytest.raises(ValueError, match="walls"):
        await create(provider(ModalVmSandboxProvider))

    modal_api.create.assert_not_called()


@pytest.mark.asyncio
async def test_marking_the_walls_verified_keeps_every_other_tag(modal_api):
    modal_api.sb = modal_sb({**BOX, PIPELINE_STEP_KEY: "t_s", WALLS_TAG: "walls-v0"})
    box = ModalVmSandbox(modal_api.sb, {}, shared_box_id="box-1")

    await _mark_walls_verified(box)

    modal_api.sb.set_tags.aio.assert_awaited_once_with({**BOX, PIPELINE_STEP_KEY: "t_s", WALLS_TAG: WALLS_VERSION})
    assert box.walls_verified is True


@pytest.mark.parametrize("shared_box_id, tags, error", [
    pytest.param(None, {}, ValueError, id="handle is no box"),
    pytest.param(UNREAD, BOX, ValueError, id="handle's box unread"),
    pytest.param("box-1", {SHARED_BOX_TAG: "box-2"}, SharedBoxMismatch, id="sandbox is another box"),
    pytest.param("box-1", None, modal.exception.ConnectionError, id="tags can't be read"),
])
@pytest.mark.asyncio
async def test_marking_the_walls_verified_writes_nothing_unless_the_sandbox_is_that_box(shared_box_id, tags, error):
    sb = modal_sb(tags) if tags is not None else modal_sb(tag_reads=modal.exception.ConnectionError("reset"))
    box = ModalVmSandbox(sb, {}, shared_box_id=shared_box_id)

    with pytest.raises(error):
        await _mark_walls_verified(box)

    sb.set_tags.aio.assert_not_called()
    assert box.walls_verified is False


_WALLS_WRITER = "_mark_walls_verified"
#: Functions that may name WALLS_TAG without writing it: its refusal in attribution and the reattach that reads it,
#: which alone passes what it read to a handle's ``walls``.
_WALLS_READERS = {"_attribution_tags", "get_shared_box"}
#: Names whose spelling as a string would reach the walls tag or the tags around getattr and setattr.
_WALLS_NAMES = {"WALLS_TAG", "set_tags", "_walls"}


def _walls_tag_strays(name: str, text: str) -> list[str]:
    """Each place in the module ``text`` that writes the walls tag, or a handle's record of it, outside its one writer.

    WALLS_TAG, bare or as an attribute, is named only in its definition, by the functions that read or refuse it, and
    by the writer, the only one to put it in a dict or assign it. ``set_tags`` is reached, called or not, only on a
    Modal App, named ``app``, or by the writer. A handle's ``_walls`` is assigned only by the writer, or from the
    ``walls`` argument in ``__init__``, and ``walls=`` is passed only by the readers. None of these names is spelled
    as a string, as for getattr, except by the writer."""
    tree = ast.parse(text)
    owner = {}
    for function in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for node in ast.walk(function):
            owner.setdefault(id(node), function.name)
    writes, passthrough = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            writes |= {id(k) for k in node.keys if k is not None}
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            writes.add(id(node.slice))
        if isinstance(node, ast.Assign) and owner.get(id(node)) == "__init__" \
                and isinstance(node.value, ast.Name) and node.value.id == "walls":
            passthrough |= {id(t) for t in node.targets}
    stray = []
    for node in ast.walk(tree):
        function = owner.get(id(node))
        if function == _WALLS_WRITER:
            continue
        where = f"{name}:{getattr(node, 'lineno', '?')} in {function or 'module scope'}"
        if (isinstance(node, ast.Name) and node.id == "WALLS_TAG") or \
                (isinstance(node, ast.Attribute) and node.attr == "WALLS_TAG"):
            if function is None and isinstance(node.ctx, ast.Store):
                continue
            if function not in _WALLS_READERS or id(node) in writes:
                stray.append(f"{where} names WALLS_TAG")
        if isinstance(node, ast.Attribute) and node.attr == "set_tags" \
                and not (isinstance(node.value, ast.Name) and node.value.id == "app"):
            stray.append(f"{where} sets tags")
        if isinstance(node, ast.Attribute) and node.attr == "_walls" and not isinstance(node.ctx, ast.Load) \
                and id(node) not in passthrough:
            stray.append(f"{where} assigns _walls")
        if isinstance(node, ast.keyword) and node.arg == "walls" and function not in _WALLS_READERS:
            stray.append(f"{where} passes walls=")
        callee = getattr(node, "func", None)
        if isinstance(node, ast.Call) and getattr(callee, "id", getattr(callee, "attr", None)) == "ModalVmSandbox" \
                and (len(node.args) > 4 or any(k.arg is None for k in node.keywords)):
            stray.append(f"{where} may pass walls to ModalVmSandbox unnamed")
        if isinstance(node, ast.Constant) and node.value in _WALLS_NAMES:
            stray.append(f"{where} spells {node.value!r}")
    return stray


def test_only_the_walls_writer_writes_the_walls_tag_or_sets_a_sandboxs_tags():
    package = Path(modal_sandbox.__file__).parents[2]
    stray = []
    for source in package.rglob("*.py"):
        text = source.read_text()
        if f'"{WALLS_TAG}"' in text and source != Path(modal_sandbox.__file__):
            stray.append(f"{source.name} spells the walls tag's value")
        stray += _walls_tag_strays(source.name, text)
    assert stray == []


_PLANTED_WALLS_WRITES = {
    "tag in a dict, async set_tags": ("def f(sb):\n    sb.set_tags.aio({WALLS_TAG: WALLS_VERSION})", "names WALLS_TAG"),
    "sync set_tags": ("def f(sb, tags):\n    sb.set_tags(tags)", "sets tags"),
    "getattr set_tags": ("def f(sb, tags):\n    getattr(sb, 'set_tags').aio(tags)", "spells 'set_tags'"),
    "set_tags held for later": ("def f(sb):\n    write = sb.set_tags\n    return write", "sets tags"),
    "WALLS_TAG as an attribute": ("def f(m, tags):\n    tags[m.WALLS_TAG] = 'walls-v1'", "names WALLS_TAG"),
    "reader writes the tag": ("def get_shared_box(m, tags):\n    tags[m.WALLS_TAG] = 'v'", "names WALLS_TAG"),
    "getattr WALLS_TAG": ("def f(m):\n    return getattr(m, 'WALLS_TAG')", "spells 'WALLS_TAG'"),
    "constructor walls=": ("def f(sb):\n    return ModalVmSandbox(sb, {}, walls=WALLS_VERSION)", "passes walls="),
    "constructor positional": ("def f(sb):\n    return ModalVmSandbox(sb, {}, None, 'b', WALLS_VERSION)", "unnamed"),
    "constructor **kwargs": ("def f(sb, kw):\n    return ModalVmSandbox(sb, {}, **kw)", "unnamed"),
    "_walls assigned": ("def f(box):\n    box._walls = WALLS_VERSION", "assigns _walls"),
    "_walls set in __init__ to a constant": ("def __init__(self):\n    self._walls = WALLS_VERSION", "assigns _walls"),
    "setattr _walls": ("def f(box):\n    setattr(box, '_walls', WALLS_VERSION)", "spells '_walls'"),
}


@pytest.mark.parametrize("snippet, reason", _PLANTED_WALLS_WRITES.values(), ids=_PLANTED_WALLS_WRITES.keys())
def test_the_walls_guard_catches_each_way_to_write_the_walls_outside_its_writer(snippet, reason):
    assert any(reason in stray for stray in _walls_tag_strays("planted.py", snippet))


def test_the_walls_guard_lets_the_writer_readers_and_handle_do_their_part():
    clean = (
        "WALLS_TAG = 'x'\n"
        "def _mark_walls_verified(box, tags):\n"
        "    box._sb.set_tags.aio({**tags, WALLS_TAG: WALLS_VERSION})\n    box._walls = WALLS_VERSION\n"
        "def get_shared_box(sb, tags):\n    return ModalVmSandbox(sb, {}, walls=tags.get(WALLS_TAG))\n"
        "def _attribution_tags(key):\n    return key == WALLS_TAG\n"
        "def __init__(self, walls):\n    self._walls = walls\n"
        "def f(app, tags):\n    app.set_tags.aio(tags)\n"
    )
    assert _walls_tag_strays("clean.py", clean) == []
