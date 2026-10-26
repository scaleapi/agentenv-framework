"""`_collect_file` reads a large file in bounded chunks so no single exec streams more than a
transport's output buffer holds (a sandbox exec channel can drop the oldest output silently)."""
import base64
import hashlib
import os
import re

import pytest

from agent_env.task_step.task_steps import collect_artifacts
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep

CHUNK = 1000
RANGED = re.compile(r"tail -c \+(\d+) \S+ \| head -c (\d+) \| base64")


class _Stdout:
    def __init__(self, data: bytes):
        self._data = data

    def __aiter__(self):
        self._it = iter([self._data[i:i + 4096] for i in range(0, len(self._data), 4096)])
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Stderr:
    async def read(self):
        return b""


class _Proc:
    def __init__(self, data: bytes):
        self.stdout = _Stdout(data)
        self.stderr = _Stderr()

    async def wait(self):
        return 0


class _Sandbox:
    """Serves `base64 < file` whole and `tail | head | base64` ranges, recording each command."""
    mode = "container"

    def __init__(self, content: bytes, serve_short: int = 0, append_after_first_read: bytes = b""):
        self.content = content
        self.serve_short = serve_short
        self.append_after_first_read = append_after_first_read
        self.commands: list[str] = []

    async def exec(self, *args):
        cmd = args[-1]
        self.commands.append(cmd)
        match = RANGED.search(cmd)
        if match:
            start, length = int(match.group(1)), int(match.group(2))
            data = self.content[start - 1:start - 1 + length]
            if self.serve_short:
                data = data[:-self.serve_short]
        else:
            data = self.content
        if self.append_after_first_read:
            self.content += self.append_after_first_read
            self.append_after_first_read = b""
        return _Proc(base64.encodebytes(data))

    async def exec_with_output(self, *args):
        return (0, f"{len(self.content)}\n", "")


class _Store:
    def __init__(self):
        self.uploaded = None

    def put_object_file(self, **kw):
        with open(kw["file_path"], "rb") as fh:
            self.uploaded = fh.read()
        return "s3://bucket/ok"


@pytest.fixture(autouse=True)
def _small_chunks(monkeypatch):
    monkeypatch.setattr(collect_artifacts, "_COLLECT_CHUNK_BYTES", CHUNK)


async def _collect(sandbox):
    store = _Store()
    step = CollectArtifactsTaskStep(id="collect", version=None)
    await step._collect_file(sandbox, None, "/app/artifact/a.bin", "a.bin", store, "uid", 1,
                             "application/octet-stream")
    return store.uploaded


def _ranged(sandbox):
    return [c for c in sandbox.commands if RANGED.search(c)]


@pytest.mark.asyncio
async def test_multi_chunk_file_is_reassembled_exactly():
    content = os.urandom(CHUNK * 3 + 123)
    sandbox = _Sandbox(content)
    got = await _collect(sandbox)
    assert hashlib.sha256(got).digest() == hashlib.sha256(content).digest()
    assert len(_ranged(sandbox)) == 4


@pytest.mark.asyncio
async def test_exact_multiple_of_the_chunk_size_reads_no_extra_chunk():
    content = os.urandom(CHUNK * 2)
    sandbox = _Sandbox(content)
    assert await _collect(sandbox) == content
    assert len(_ranged(sandbox)) == 2


@pytest.mark.asyncio
async def test_small_file_takes_a_single_unranged_exec():
    content = os.urandom(CHUNK)
    sandbox = _Sandbox(content)
    assert await _collect(sandbox) == content
    assert len(sandbox.commands) == 1 and not _ranged(sandbox)


@pytest.mark.asyncio
async def test_empty_file_collects_as_zero_bytes():
    sandbox = _Sandbox(b"")
    assert await _collect(sandbox) == b""


@pytest.mark.asyncio
async def test_short_chunk_read_raises_and_uploads_nothing():
    sandbox = _Sandbox(os.urandom(CHUNK * 3), serve_short=1)
    with pytest.raises(RuntimeError, match="short read"):
        await _collect(sandbox)


@pytest.mark.asyncio
async def test_a_file_that_grows_during_collection_fails_the_integrity_check():
    sandbox = _Sandbox(os.urandom(CHUNK * 3), append_after_first_read=b"late bytes")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        await _collect(sandbox)
