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


_MODAL_SANDBOX, _MODAL_VM_SANDBOX = modal_sandbox.__name__, modal_vm_sandbox.__name__
#: The places, each a module and a qualified name in it, with a part in the walls tag: its one writer; the functions
#: that may name WALLS_TAG without writing it, its refusal in attribution and the reattach, which alone passes a
#: handle the ``walls`` it read from the tags; and the handle's ``__init__``, which alone stores that ``walls``.
_WALLS_WRITER = (_MODAL_VM_SANDBOX, "_mark_walls_verified")
_WALLS_REATTACH = (_MODAL_VM_SANDBOX, "ModalVmSandboxProvider.get_shared_box")
_WALLS_READERS = {(_MODAL_SANDBOX, "_attribution_tags"), _WALLS_REATTACH}
_HANDLE_INIT = (_MODAL_VM_SANDBOX, "ModalVmSandbox.__init__")
_TAG_DEFINITION = (_MODAL_SANDBOX, "")
#: The strings that would reach the walls tag, or the tags and the handle's record around getattr and setattr.
_WALLS_NAMES = {"WALLS_TAG", "set_tags", "_walls", WALLS_TAG}


def _places(tree: ast.Module) -> tuple[dict[int, str], dict[str, ast.AST]]:
    """The qualified name, as ``Class.method``, of the class or function holding each node of ``tree``, "" at module
    scope; and each function by its qualified name."""
    place, functions = {}, {}

    def visit(node: ast.AST, path: str) -> None:
        for child in ast.iter_child_nodes(node):
            place[id(child)] = path
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = f"{path}.{child.name}" if path else child.name
                if not isinstance(child, ast.ClassDef):
                    functions[inner] = child
                visit(child, inner)
            else:
                visit(child, path)

    visit(tree, "")
    return place, functions


def _folded(node: ast.AST) -> str | None:
    """The string ``node`` makes from literals alone, joined by ``+`` or in an f-string; None for anything else."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _folded(node.left), _folded(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.FormattedValue):
        return _folded(node.value) if node.conversion == -1 and node.format_spec is None else None
    if isinstance(node, ast.JoinedStr):
        parts = [_folded(value) for value in node.values]
        return None if None in parts else "".join(parts)
    return None


def _walls_read_from_tags(function: ast.AST, value: ast.expr) -> bool:
    """Whether ``value`` is ``tags.get(WALLS_TAG)``, for a ``tags`` that ``function`` assigns once, from
    ``await _read_tags(...)``."""
    read = re.fullmatch(r"(\w+)\.get\(WALLS_TAG\)", ast.unparse(value))
    if read is None:
        return False
    stores = [n for n in ast.walk(function) if isinstance(n, ast.Name) and n.id == read[1] and isinstance(n.ctx, ast.Store)]
    return len(stores) == 1 and any(
        isinstance(n, ast.Assign) and n.targets == stores and re.fullmatch(r"await _read_tags\(.*\)", ast.unparse(n.value))
        for n in ast.walk(function))


def _walls_tag_strays(module: str, text: str) -> list[str]:
    """Each place in ``text``, the source of ``module``, that writes the walls tag, or a handle's record of it, outside
    its one writer, each place matched by module and qualified name, never by name alone.

    WALLS_TAG, bare, as an attribute or imported under another name, is named only in its definition, by the readers,
    and by the writer, the only one to put it in a dict or assign it. ``set_tags`` is reached, called or not, only on a
    Modal App, named ``app``, or by the writer. A handle's ``_walls`` is assigned only by the writer, or from the
    ``walls`` argument in the handle's ``__init__``; no call passes ``_walls=``, and ``walls=`` is passed only by the
    reattach, with what it read from the tags. None of these names, nor the tag's value, is made as a string, quoted
    any way, joined with ``+`` or in an f-string, outside the tag's definition. A module that mentions walls reaches
    no object's ``__dict__`` or ``vars``, which would write ``_walls`` unnamed."""
    tree = ast.parse(text)
    place, functions = _places(tree)
    writes, passthrough, definition = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            writes |= {id(k) for k in node.keys if k is not None}
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            writes.add(id(node.slice))
        if isinstance(node, ast.Assign) and (module, place[id(node)]) == _HANDLE_INIT \
                and isinstance(node.value, ast.Name) and node.value.id == "walls":
            passthrough |= {id(t) for t in node.targets}
        if isinstance(node, ast.Assign) and (module, place[id(node)]) == _TAG_DEFINITION \
                and [ast.unparse(t) for t in node.targets] == ["WALLS_TAG"]:
            definition |= {id(node.targets[0]), id(node.value)}
    mentions_walls = "walls" in text.lower()
    stray = []
    for node in ast.walk(tree):
        where_in = place.get(id(node), "")
        if (module, where_in) == _WALLS_WRITER:
            continue
        where = f"{module}:{getattr(node, 'lineno', '?')} in {where_in or 'module scope'}"
        if (isinstance(node, ast.Name) and node.id == "WALLS_TAG") or \
                (isinstance(node, ast.Attribute) and node.attr == "WALLS_TAG"):
            if id(node) not in definition and ((module, where_in) not in _WALLS_READERS or id(node) in writes):
                stray.append(f"{where} names WALLS_TAG")
        if isinstance(node, ast.alias) and node.name == "WALLS_TAG" and node.asname not in (None, "WALLS_TAG"):
            stray.append(f"{where} imports WALLS_TAG as {node.asname}")
        if isinstance(node, ast.Attribute) and node.attr == "set_tags" \
                and not (isinstance(node.value, ast.Name) and node.value.id == "app"):
            stray.append(f"{where} sets tags")
        if isinstance(node, ast.Attribute) and node.attr == "_walls" and not isinstance(node.ctx, ast.Load) \
                and id(node) not in passthrough:
            stray.append(f"{where} assigns _walls")
        if isinstance(node, ast.keyword) and node.arg == "_walls":
            stray.append(f"{where} passes _walls=")
        if isinstance(node, ast.keyword) and node.arg == "walls" and not (
                (module, where_in) == _WALLS_REATTACH and _walls_read_from_tags(functions[where_in], node.value)):
            stray.append(f"{where} passes walls=")
        callee = getattr(node, "func", None)
        callee_name = getattr(callee, "id", getattr(callee, "attr", None))
        if isinstance(node, ast.Call) and callee_name == "ModalVmSandbox" \
                and (len(node.args) > 4 or any(k.arg is None for k in node.keywords)):
            stray.append(f"{where} may pass walls to ModalVmSandbox unnamed")
        if mentions_walls and ((isinstance(node, ast.Attribute) and node.attr == "__dict__")
                               or (isinstance(node, ast.Call) and callee_name == "vars")):
            stray.append(f"{where} reaches an object's __dict__")
        if (spelled := _folded(node)) in _WALLS_NAMES and id(node) not in definition:
            stray.append(f"{where} spells {spelled!r}")
    return stray


def _module_name(source: Path, package: Path) -> str:
    return ".".join(source.relative_to(package.parent).with_suffix("").parts)


def test_only_the_walls_writer_writes_the_walls_tag_or_sets_a_sandboxs_tags():
    package = Path(modal_sandbox.__file__).parents[2]
    assert _module_name(Path(modal_vm_sandbox.__file__), package) == _MODAL_VM_SANDBOX
    stray = []
    for source in package.rglob("*.py"):
        stray += _walls_tag_strays(_module_name(source, package), source.read_text())
    assert stray == []


_REATTACH = ("class ModalVmSandboxProvider:\n    async def get_shared_box(self, sb):\n"
             "        tags = await _read_tags(sb)\n        return ModalVmSandbox(sb, {{}}, shared_box_id='b', walls={})")
_PLANTED = "agent_env.planted"
_PLANTED_WALLS_WRITES = {
    "tag in a dict, async set_tags": ("def f(sb):\n    sb.set_tags.aio({WALLS_TAG: WALLS_VERSION})", "names WALLS_TAG"),
    "sync set_tags": ("def f(sb, tags):\n    sb.set_tags(tags)", "sets tags"),
    "getattr set_tags": ("def f(sb, tags):\n    getattr(sb, 'set_tags').aio(tags)", "spells 'set_tags'"),
    "set_tags held for later": ("def f(sb):\n    write = sb.set_tags\n    return write", "sets tags"),
    "WALLS_TAG as an attribute": ("def f(m, tags):\n    tags[m.WALLS_TAG] = 'walls-v1'", "names WALLS_TAG"),
    "reader writes the tag": (f"{_REATTACH.format('tags.get(WALLS_TAG)')}\n        tags[WALLS_TAG] = 'v'",
                              "names WALLS_TAG", _MODAL_VM_SANDBOX),
    "getattr WALLS_TAG": ("def f(m):\n    return getattr(m, 'WALLS_TAG')", "spells 'WALLS_TAG'"),
    "WALLS_TAG imported under another name": (f"from {_MODAL_SANDBOX} import WALLS_TAG as T\n"
                                              "def f(sb, tags):\n    sb.x({**tags, T: 'v'})", "imports WALLS_TAG as T"),
    "WALLS_TAG defined again": ("WALLS_TAG = 'x'", "names WALLS_TAG"),
    "tag value in single quotes": (f"def f(sb):\n    sb.x({{'{WALLS_TAG}': 'v'}})", f"spells {WALLS_TAG!r}"),
    "tag value in triple quotes": (f'def f(sb):\n    sb.x({{"""{WALLS_TAG}""": "v"}})', f"spells {WALLS_TAG!r}"),
    "tag value joined with +": (f"def f(sb):\n    sb.x({{'{WALLS_TAG[:4]}' + '{WALLS_TAG[4:]}': 'v'}})",
                                f"spells {WALLS_TAG!r}"),
    "tag value in an f-string": (f"def f(sb):\n    sb.x({{f'{WALLS_TAG[:4]}{{\"{WALLS_TAG[4:]}\"}}': 'v'}})",
                                 f"spells {WALLS_TAG!r}"),
    "_walls joined with +": ("def f(box):\n    setattr(box, '_wa' + 'lls', 1)", "spells '_walls'"),
    "constructor walls=": ("def f(sb):\n    return ModalVmSandbox(sb, {}, walls=WALLS_VERSION)", "passes walls="),
    "reattach passes a constant": (_REATTACH.format("WALLS_VERSION"), "passes walls=", _MODAL_VM_SANDBOX),
    "reattach passes another tag": (_REATTACH.format("tags.get(SHARED_BOX_TAG)"), "passes walls=", _MODAL_VM_SANDBOX),
    "reattach rewrites what it read": (f"{_REATTACH.format('tags.get(WALLS_TAG)')}\n        tags = {{}}",
                                       "passes walls=", _MODAL_VM_SANDBOX),
    "reattach of another class": (_REATTACH.replace("ModalVmSandboxProvider", "Other").format("tags.get(WALLS_TAG)"),
                                  "passes walls=", _MODAL_VM_SANDBOX),
    "reattach in another module": (_REATTACH.format("tags.get(WALLS_TAG)"), "passes walls="),
    "constructor positional": ("def f(sb):\n    return ModalVmSandbox(sb, {}, None, 'b', WALLS_VERSION)", "unnamed"),
    "constructor **kwargs": ("def f(sb, kw):\n    return ModalVmSandbox(sb, {}, **kw)", "unnamed"),
    "_walls assigned": ("def f(box):\n    box._walls = WALLS_VERSION", "assigns _walls"),
    "_walls set in __init__ to a constant": ("def __init__(self):\n    self._walls = WALLS_VERSION", "assigns _walls"),
    "_walls set in another class's __init__": ("class Other:\n    def __init__(self, walls):\n        self._walls = walls",
                                               "assigns _walls", _MODAL_VM_SANDBOX),
    "setattr _walls": ("def f(box):\n    setattr(box, '_walls', WALLS_VERSION)", "spells '_walls'"),
    "vars(...).update(_walls=)": ("def f(box):\n    vars(box).update(_walls=WALLS_VERSION)", "passes _walls="),
    "vars in a module that mentions walls": ("def f(box, d):\n    vars(box).update(d)\n    return box.walls_verified",
                                             "reaches an object's __dict__"),
    "__dict__ in a module that mentions walls": ("def f(box, d):\n    box.__dict__.update(d)  # walls",
                                                 "reaches an object's __dict__"),
    "writer's name in another module": ("def _mark_walls_verified(box):\n    box._walls = WALLS_VERSION",
                                        "assigns _walls"),
    "writer's name as a method": ("class X:\n    def _mark_walls_verified(self):\n        self._walls = WALLS_VERSION",
                                  "assigns _walls", _MODAL_VM_SANDBOX),
    "inside the writer, a nested function": ("def _mark_walls_verified(box):\n    def inner():\n"
                                             "        box._walls = WALLS_VERSION\n    inner()",
                                             "assigns _walls", _MODAL_VM_SANDBOX),
}


@pytest.mark.parametrize("planted", _PLANTED_WALLS_WRITES.values(), ids=_PLANTED_WALLS_WRITES.keys())
def test_the_walls_guard_catches_each_way_to_write_the_walls_outside_its_writer(planted):
    snippet, reason, module = (*planted, _PLANTED)[:3]
    assert any(reason in stray for stray in _walls_tag_strays(module, snippet))


def test_the_walls_guard_lets_the_writer_readers_and_handle_do_their_part():
    clean = {
        _MODAL_SANDBOX: (
            f"WALLS_TAG = '{WALLS_TAG}'\n"
            "def _attribution_tags(key):\n    return key == WALLS_TAG\n"
            "def f(app, tags):\n    app.set_tags.aio(tags)\n"
        ),
        _MODAL_VM_SANDBOX: (
            f"from {_MODAL_SANDBOX} import WALLS_TAG\n"
            "async def _mark_walls_verified(box):\n"
            "    tags = await _read_tags(box._sb)\n"
            "    await box._sb.set_tags.aio({**tags, WALLS_TAG: WALLS_VERSION})\n    box._walls = WALLS_VERSION\n"
            f"{_REATTACH.format('tags.get(WALLS_TAG)')}\n"
            "class ModalVmSandbox:\n    def __init__(self, walls):\n        self._walls = walls\n"
        ),
    }
    assert [stray for module, text in clean.items() for stray in _walls_tag_strays(module, text)] == []
