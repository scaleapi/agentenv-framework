"""The collect integrity check must FAIL loudly on a truncated base64-over-exec transfer
(a reproduced WebSocket silent-truncation failure) instead of uploading a short artifact."""
import asyncio
import base64
import tempfile
import threading

import pytest
from agent_env.task_step.task_steps import collect_artifacts
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
from tst.unit.event_loop_probe import on_event_loop


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
        return (0, f"regular file:{self._src}\n", "")


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
        await step._collect_file(sb, None, "/app/artifact/corpus.warc.gz", len(src), "corpus.warc.gz",
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
    out = await step._collect_file(sb, None, "/app/artifact/corpus.warc.gz", len(src), "corpus.warc.gz",
                                   _OkStore(), "uid", 1, "application/octet-stream")
    assert out == "s3://bucket/ok"


@pytest.mark.asyncio
async def test_the_upload_runs_off_the_event_loop():
    src = b"C" * 900
    full_b64 = base64.b64encode(src)
    sb = _Sandbox(full_b64, len(full_b64), len(src))
    on_loop: list[bool] = []

    class _Store:
        def put_object_file(self, **kw):
            on_loop.append(on_event_loop())
            return "s3://bucket/ok"

    step = CollectArtifactsTaskStep(id="collect", version=None)
    await step._collect_file(sb, None, "/app/artifact/a.bin", len(src), "a.bin", _Store(), "uid", 1, "application/octet-stream")
    assert on_loop == [False]


@pytest.mark.asyncio
async def test_a_cancel_mid_upload_leaves_the_file_to_the_upload(caplog, monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    src = b"D" * 900
    full_b64 = base64.b64encode(src)
    sb = _Sandbox(full_b64, len(full_b64), len(src))
    uploading, release, uploaded = threading.Event(), threading.Event(), []

    class _Store:
        def put_object_file(self, **kw):
            uploading.set()
            release.wait(5)
            with open(kw["file_path"], "rb") as fh:
                uploaded.append(fh.read())
            return "s3://bucket/ok"

    step = CollectArtifactsTaskStep(id="collect", version=None)
    collect = asyncio.create_task(step._collect_file(sb, None, "/app/artifact/a.bin", len(src), "a.bin", _Store(), "uid", 1, "application/octet-stream"))
    await asyncio.to_thread(uploading.wait, 5)
    collect.cancel()
    with pytest.raises(asyncio.CancelledError):
        await collect
    with caplog.at_level("INFO", logger="agent_env.task_step.thread_work"):
        release.set()
        for _ in range(100):
            if "finished after its caller was cancelled" in caplog.text:
                break
            await asyncio.sleep(0.02)
    assert uploaded == [src]
    assert "Uploading collected /app/artifact/a.bin finished after its caller was cancelled" in caplog.text
    assert list(tmp_path.iterdir()) == []


class _HangingStdout:
    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()


class _RecordingSandbox(_Sandbox):
    def __init__(self, *args, stdout=None):
        super().__init__(*args)
        self.commands, self._stdout = [], stdout

    async def exec(self, *args):
        self.commands.append(args)
        return _Proc(self._stdout or _TruncStdout(self._full, self._keep))


class _UploadingStore:
    def put_object_file(self, **kw):
        return "s3://bucket/ok"


class _RefusingStore:
    def put_object_file(self, **kw):
        raise AssertionError("uploaded an artifact the read should have refused")


@pytest.mark.asyncio
async def test_the_sandbox_reads_no_more_than_the_size_it_reported():
    src = b"F" * 900
    sb = _RecordingSandbox(base64.b64encode(src), len(base64.b64encode(src)), len(src))
    step = CollectArtifactsTaskStep(id="collect", version=None)
    await step._collect_file(sb, None, "/app/artifact/a b.bin", len(src), "a b.bin", _UploadingStore(), "uid", 1,
                             "application/octet-stream")
    assert sb.commands == [("bash", "-c", "head -c 900 < '/app/artifact/a b.bin' | base64")]


@pytest.mark.asyncio
async def test_a_stream_longer_than_the_reported_size_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    src = b"G" * 900
    more = base64.b64encode(src * 3)
    sb = _Sandbox(more, len(more), len(src))
    step = CollectArtifactsTaskStep(id="collect", version=None)
    with pytest.raises(RuntimeError, match="more than the 900 bytes it reported"):
        await step._collect_file(sb, None, "/app/artifact/zero", len(src), "zero", _RefusingStore(), "uid", 1,
                                 "application/octet-stream")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_a_read_past_its_deadline_is_abandoned(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(collect_artifacts, "_READ_DEADLINE_FLOOR_S", 0.05)
    sb = _RecordingSandbox(b"", 0, 10, stdout=_HangingStdout())
    step = CollectArtifactsTaskStep(id="collect", version=None)
    with pytest.raises(RuntimeError, match="took more than 0s; abandoned it"):
        await step._collect_file(sb, None, "/app/artifact/slow", 10, "slow", _RefusingStore(), "uid", 1,
                                 "application/octet-stream")
    assert list(tmp_path.iterdir()) == []

