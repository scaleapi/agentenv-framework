"""An object a store can't sign reaches a VM host over exec: a chunk per exec, or segments over stdin, each checked."""

import asyncio
import gzip
import io
import os
import re

import pytest

from agent_env.providers.sandbox_providers import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.sandbox import push_object_over_exec, push_object_over_stdin
from agent_env.store.object_store.local import LocalFilesystemObjectStore

BLOCK = sandbox_module._PUSH_BLOCK


class _ShellVm:
    """Runs each script in a real shell on this machine, as the VM host would, stdin included."""

    _WFT_CHUNK_BYTES = BLOCK // 3 * 4
    _PUSHES_IN_FLIGHT = 3

    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.stdin_scripts: list[str] = []

    async def exec_script(self, script: str, *, max_retries: int = 0) -> str:
        self.scripts.append(script)
        code, out, err = await self._run(script, None)
        if code:
            raise RuntimeError(f"Script failed (exit {code}): {err}")
        return out

    async def _exec_with_stdin(self, script, stdin):
        self.stdin_scripts.append(script)
        return await self._run(script, stdin)

    async def _run(self, script, stdin):
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", script, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        if stdin is not None:
            async for piece in stdin:
                process.stdin.write(piece)
                await process.stdin.drain()
        process.stdin.close()
        out, err = await process.communicate()
        return process.returncode, out.decode(), err.decode()


class _GarblingVm(_ShellVm):
    """Flips a byte of what it is sent for the chunk or segment at ``offset``."""

    def __init__(self, offset: int) -> None:
        super().__init__()
        self._seek = f"seek={offset // 4096} "

    async def exec_script(self, script: str, *, max_retries: int = 0) -> str:
        if self._seek in script:
            script = re.sub(r"printf %s (\S)", lambda m: "printf %s " + ("B" if m[1] == "A" else "A"), script)
        return await super().exec_script(script, max_retries=max_retries)

    async def _exec_with_stdin(self, script, stdin):
        if self._seek not in script:
            return await super()._exec_with_stdin(script, stdin)

        async def garbled():
            first = True
            async for piece in stdin:
                yield (b"B" if piece[:1] == b"A" else b"A") + piece[1:] if first else piece
                first = False

        return await super()._exec_with_stdin(script, garbled())


class _ChunkFailingVm(_ShellVm):
    def __init__(self, offset: int) -> None:
        super().__init__()
        self._seek = f"seek={offset // 4096} "

    async def exec_script(self, script: str, *, max_retries: int = 0) -> str:
        if self._seek in script:
            self.scripts.append(script)
            raise RuntimeError("Script failed (exit 1): the VM went away")
        return await super().exec_script(script, max_retries=max_retries)


class _SilentVm(_ShellVm):
    """Answers a stdin exec with success and no output, as one provider does for an exec its VM's termination cut."""

    async def _exec_with_stdin(self, script, stdin):
        async for _ in stdin:
            pass
        return 0, "", ""


class _StreamStore:
    """Opens its object as a stream that isn't a file, as a hosted store's reader is."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def open(self, object_url: str):
        return io.BytesIO(self.data)


class _GzipStore:
    """Opens its object through a decompressing reader, which still names the compressed file's descriptor."""

    def __init__(self, path) -> None:
        self.path = path

    def open(self, object_url: str):
        return gzip.open(self.path, "rb")


class _ChangingVm(_ShellVm):
    """Changes the object's file on this machine as the first segment starts, before any of it is read."""

    def __init__(self, change) -> None:
        super().__init__()
        self._change = change

    async def _exec_with_stdin(self, script, stdin):
        if self._change:
            self._change()
            self._change = None
        return await super()._exec_with_stdin(script, stdin)


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(sandbox_module, "_MIN_SEGMENT_BYTES", BLOCK)
    monkeypatch.setattr(sandbox_module, "_STDIN_PIECE_BYTES", BLOCK)


def _stored(tmp_path, data: bytes):
    source = tmp_path / "object.bin"
    source.write_bytes(data)
    store = LocalFilesystemObjectStore(str(tmp_path / "objects"))
    return store, store.put_file("images/object.bin", str(source))


SIZES = [0, 1, BLOCK - 1, BLOCK, BLOCK + 1, 10 * BLOCK + 123]
SIZE_IDS = ["empty", "a-byte", "under-a-block", "one-block", "over-a-block", "many-blocks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
async def test_an_object_pushed_over_exec_lands_byte_for_byte(small, tmp_path, size):
    data = os.urandom(size)
    store, url = _stored(tmp_path, data)
    target = tmp_path / "on the vm.bin"

    await push_object_over_exec(_ShellVm(), store, url, str(target))

    assert target.read_bytes() == data


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
async def test_an_object_pushed_over_stdin_lands_byte_for_byte(small, tmp_path, size):
    data = os.urandom(size)
    store, url = _stored(tmp_path, data)
    target = tmp_path / "on the vm.bin"

    await push_object_over_stdin(_ShellVm(), store, url, str(target))

    assert target.read_bytes() == data


@pytest.mark.asyncio
async def test_an_object_goes_over_stdin_as_that_many_segments_at_once(small, tmp_path):
    store, url = _stored(tmp_path, os.urandom(10 * BLOCK + 123))
    vm = _ShellVm()

    await push_object_over_stdin(vm, store, url, str(tmp_path / "on the vm.bin"))

    assert sorted(int(re.search(r"seek=(\d+) ", script)[1]) for script in vm.stdin_scripts) == [0, 12, 24]


@pytest.mark.asyncio
async def test_a_reader_that_isnt_a_file_goes_over_stdin_as_one_stream(small, tmp_path):
    data = os.urandom(5 * BLOCK + 7)
    target = tmp_path / "on the vm.bin"
    vm = _ShellVm()

    await push_object_over_stdin(vm, _StreamStore(data), "mem://object", str(target))

    assert target.read_bytes() == data
    assert len(vm.stdin_scripts) == 1


@pytest.mark.asyncio
async def test_a_decompressing_reader_isnt_taken_for_the_file_under_it(small, tmp_path):
    data = os.urandom(5 * BLOCK)
    compressed = tmp_path / "object.gz"
    with gzip.open(compressed, "wb") as out:
        out.write(data)
    target = tmp_path / "on the vm.bin"
    vm = _ShellVm()

    await push_object_over_stdin(vm, _GzipStore(compressed), "mem://object", str(target))

    assert target.read_bytes() == data
    assert len(vm.stdin_scripts) == 1


@pytest.mark.asyncio
async def test_an_object_of_one_chunk_goes_over_exec_in_a_single_exec(small, tmp_path):
    data = os.urandom(BLOCK - 1)
    store, url = _stored(tmp_path, data)
    target = tmp_path / "on the vm.bin"
    vm = _ShellVm()

    await push_object_over_exec(vm, store, url, str(target))

    assert target.read_bytes() == data
    assert len(vm.scripts) == 1


@pytest.mark.asyncio
async def test_an_object_under_a_chunk_goes_in_one_exec_even_where_stdin_is_taken(small, tmp_path):
    data = os.urandom(BLOCK - 1)
    store, url = _stored(tmp_path, data)
    target = tmp_path / "on the vm.bin"
    vm = _ShellVm()

    await push_object_over_stdin(vm, store, url, str(target))

    assert target.read_bytes() == data
    assert len(vm.scripts) == 1 and vm.stdin_scripts == []


@pytest.mark.asyncio
async def test_an_object_of_one_segment_goes_over_stdin_in_a_single_exec(small, tmp_path):
    data = os.urandom(BLOCK)
    store, url = _stored(tmp_path, data)
    target = tmp_path / "on the vm.bin"
    target.write_bytes(os.urandom(3 * BLOCK))
    vm = _ShellVm()

    await push_object_over_stdin(vm, store, url, str(target))

    assert target.read_bytes() == data
    assert vm.scripts == [] and len(vm.stdin_scripts) == 1


@pytest.mark.asyncio
async def test_every_segment_reads_the_version_the_push_opened(small, tmp_path):
    data = os.urandom(10 * BLOCK)
    store, url = _stored(tmp_path, data)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(os.urandom(10 * BLOCK))
    target = tmp_path / "on the vm.bin"

    await push_object_over_stdin(
        _ChangingVm(lambda: os.replace(replacement, store.root / "images" / "object.bin")), store, url, str(target))

    assert target.read_bytes() == data


@pytest.mark.asyncio
async def test_an_object_cut_short_during_the_push_fails_it(small, tmp_path):
    store, url = _stored(tmp_path, os.urandom(10 * BLOCK))

    with pytest.raises(RuntimeError, match="bytes short of the segment at offset 98304"):
        await push_object_over_stdin(
            _ChangingVm(lambda: os.truncate(store.root / "images" / "object.bin", 9 * BLOCK)), store, url,
            str(tmp_path / "on the vm.bin"))


@pytest.mark.asyncio
async def test_a_command_limit_under_a_block_still_pushes_the_object(small, tmp_path):
    data = os.urandom(3 * BLOCK + 5)
    store, url = _stored(tmp_path, data)
    target = tmp_path / "on the vm.bin"
    vm = _ShellVm()
    vm._WFT_CHUNK_BYTES = 1024

    await push_object_over_exec(vm, store, url, str(target))

    assert target.read_bytes() == data


@pytest.mark.asyncio
async def test_a_chunk_the_vm_garbles_fails_the_push(small, tmp_path):
    store, url = _stored(tmp_path, os.urandom(4 * BLOCK))

    with pytest.raises(RuntimeError, match="on the VM doesn't match"):
        await push_object_over_exec(_GarblingVm(2 * BLOCK), store, url, str(tmp_path / "on the vm.bin"))


@pytest.mark.asyncio
async def test_a_segment_the_vm_garbles_fails_the_push(small, tmp_path):
    store, url = _stored(tmp_path, os.urandom(10 * BLOCK))

    with pytest.raises(RuntimeError, match=f"doesn't match at offset {4 * BLOCK}"):
        await push_object_over_stdin(_GarblingVm(4 * BLOCK), store, url, str(tmp_path / "on the vm.bin"))


@pytest.mark.asyncio
async def test_a_stdin_exec_that_reports_success_and_nothing_else_fails_the_push(small, tmp_path):
    store, url = _stored(tmp_path, os.urandom(2 * BLOCK))

    with pytest.raises(RuntimeError, match=r"its sha256 is \[\]"):
        await push_object_over_stdin(_SilentVm(), store, url, str(tmp_path / "on the vm.bin"))


@pytest.mark.asyncio
async def test_a_failed_chunk_stops_the_push(small, tmp_path):
    store, url = _stored(tmp_path, os.urandom(10 * BLOCK))
    vm = _ChunkFailingVm(3 * BLOCK)
    vm._PUSHES_IN_FLIGHT = 1

    with pytest.raises(RuntimeError, match="the VM went away"):
        await push_object_over_exec(vm, store, url, str(tmp_path / "on the vm.bin"))

    assert sum("| dd of=" in script for script in vm.scripts) == 4
