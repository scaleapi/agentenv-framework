"""The collect integrity check must FAIL loudly on a truncated base64-over-exec transfer
(a reproduced WebSocket silent-truncation failure) instead of uploading a short artifact."""
import base64
import pytest
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep


class _TruncStdout:
    """async-iterable stdout yielding only PART of the base64 (a dropped-chunk truncation)."""
    def __init__(self, full_b64: bytes, keep: int):
        self._chunks = [full_b64[:keep]]
    def __aiter__(self):
        self._it = iter(self._chunks); return self
    async def __anext__(self):
        try: return next(self._it)
        except StopIteration: raise StopAsyncIteration


class _Stderr:
    async def read(self): return b""


class _Proc:
    def __init__(self, stdout): self.stdout = stdout; self.stderr = _Stderr()
    async def wait(self): return 0


class _Sandbox:
    """Non-VM sandbox -> base64 path; exec returns a TRUNCATED stream; exec_with_output
    (used by _get_file_size) reports the FULL source size."""
    mode = "container"
    def __init__(self, full_b64, keep, src_size):
        self._full, self._keep, self._src = full_b64, keep, src_size
    async def exec(self, *args):
        return _Proc(_TruncStdout(self._full, self._keep))
    async def exec_with_output(self, *args):
        return (0, f"{self._src}\n", "")


@pytest.mark.asyncio
async def test_integrity_check_fails_on_truncated_transfer():
    src = b"A" * 9000
    full_b64 = base64.b64encode(src)          # 12000 chars
    keep = 8000                                # drop 4000 -> decodes to 6000 bytes, %4==0 (no decode error)
    sb = _Sandbox(full_b64, keep, len(src))

    class _Store:
        def put_object_file(self, **kw):
            raise AssertionError("uploaded a truncated artifact — integrity check missed it")

    step = CollectArtifactsTaskStep(id="collect", version=None)
    with pytest.raises(RuntimeError, match="integrity check failed"):
        await step._collect_file(sb, None, "/app/artifact/corpus.warc.gz", "corpus.warc.gz",
                                 _Store(), "uid", 1, "application/octet-stream")


@pytest.mark.asyncio
async def test_integrity_check_passes_on_complete_transfer():
    src = b"B" * 9000
    full_b64 = base64.b64encode(src)
    sb = _Sandbox(full_b64, len(full_b64), len(src))  # keep all -> complete

    class _OkStore:
        def put_object_file(self, **kw):
            import os
            assert os.path.getsize(kw["file_path"]) == 9000
            return "s3://bucket/ok"

    step = CollectArtifactsTaskStep(id="collect", version=None)
    out = await step._collect_file(sb, None, "/app/artifact/corpus.warc.gz", "corpus.warc.gz",
                                   _OkStore(), "uid", 1, "application/octet-stream")
    assert out == "s3://bucket/ok"
