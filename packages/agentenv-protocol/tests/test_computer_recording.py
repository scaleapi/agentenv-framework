"""computer-recording/v1: a driver with ``recording`` serves it, from its ``@record`` stream or by polling ``@capture``;
segments roll over and upload as they close, the manifest matches the files, and the limits and retries hold."""
from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")
av = pytest.importorskip("av")

from computer_fake import FakeDriver, MinimalDriver, http_client  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from agentenv_protocol.computer import (  # noqa: E402
    COMPUTER_RECORDING,
    DeviceUnavailable,
    RecordedFrame,
    RecordedSegment,
    Recording,
    frames_in_thread,
    record,
)
from agentenv_protocol.computer import recording as recording_module  # noqa: E402
from agentenv_protocol.transfers import TransferError  # noqa: E402

START, END = "/agentenv/ext/computer-recording/start", "/agentenv/ext/computer-recording/end"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class Recorded(FakeDriver):
    recording = Recording(max_segment_s=1)


class Streamed(FakeDriver):
    """A stream of the fake screen's frames as a decoder hands them over, each marked a keyframe, as x11grab's are."""
    recording = Recording(max_fps=30, max_segment_s=60)

    @record
    async def frames(self, hints):
        while True:
            frame = av.VideoFrame.from_image(self.screen.image.copy())
            frame.pict_type = av.video.frame.PictureType.I
            yield RecordedFrame(frame, datetime.now(timezone.utc))
            await asyncio.sleep(1 / hints.fps)


class Segmented(FakeDriver):
    """A native recorder that writes whole segments itself."""
    recording = Recording(max_segment_s=60)

    @record
    async def segments(self, hints):
        for n in range(2):
            path = hints.directory / ("0000.mp4" if n == 0 else f"native-{n}.mp4")
            start = datetime.now(timezone.utc)
            await asyncio.to_thread(_mp4, path, 5, (160, 100))
            yield RecordedSegment(path, start, start + timedelta(seconds=0.4), 160, 100, 5)
        await asyncio.sleep(3600)


def _mp4(path, frames, size):
    out = av.open(str(path), "w", format="mp4")
    stream = out.add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = *size, "yuv420p"
    for i in range(frames):
        frame = av.VideoFrame.from_image(PILImage.new("RGB", size, (i * 40, 0, 0))).reformat(format="yuv420p")
        frame.pts, frame.time_base = i * 100, Fraction(1, 1000)
        for p in stream.encode(frame):
            out.mux(p)
    for p in stream.encode():
        out.mux(p)
    out.close()


def _grant() -> dict:
    return {"root_path": "rec/run-1", "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "max_objects": 50, "max_object_bytes": 10 << 20, "max_total_bytes": 100 << 20,
            "write": {"kind": "http-post-policy", "url": "https://store.test/", "fields": {}, "path_field": "key",
                      "file_field": "file"}}


async def _record(driver, seconds: float, start: dict | None = None, end: dict | None = None) -> tuple[dict, dict]:
    """A recording that runs ``seconds`` from its first frame, however long that took to come, and then ends."""
    async with http_client(driver) as http:
        started = (await http.post(START, json={"request_id": "r", "fps": 10, **(start or {})})).json()
        await _first_frame(driver, started)
        await asyncio.sleep(seconds)
        manifest = (await http.post(END, json={"recording_id": started["recording_id"], **(end or {})})).json()
    return started, manifest


async def _first_frame(driver, started: dict) -> None:
    recording = driver._computer.recordings.running[started["recording_id"]]
    for _ in range(1000):
        if recording.segments or (recording.encoder and recording.encoder.frames) or recording.done:
            return
        await asyncio.sleep(0.01)


def _decoded(path) -> list:
    with av.open(str(path)) as container:
        return list(container.decode(container.streams.video[0]))


def _at(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


@pytest.mark.asyncio
async def test_every_driver_serves_recording_unless_it_turns_it_off():
    class Off(FakeDriver):
        recording = None

    async with http_client(FakeDriver()) as http:
        card = (await http.get("/.well-known/agent-env.json")).json()
    async with http_client(Off()) as http:
        off = (await http.get("/.well-known/agent-env.json")).json()
    extension = next(e for e in card["capabilities"]["extensions"] if e["uri"] == COMPUTER_RECORDING)
    assert set(extension["params"]["methods"]) == {"start", "end"}
    assert {k: v for k, v in extension["params"].items() if k not in ("methods", "endpoint")} == {
        "formats": ["video/mp4; codecs=avc1"], "scopes": ["display"], "fps": {"default": 10, "max": 15},
        "cursor": True, "source": "capture", "max_segment_s": 15, "max_concurrent": 2, "max_duration_s": 86_400}
    assert COMPUTER_RECORDING not in [e["uri"] for e in off["capabilities"]["extensions"]]


def test_recording_features_and_record_are_checked_when_the_class_is_defined():
    with pytest.raises(TypeError, match="recording.max_fps is an integer from 1 to 60"):
        class TooFast(FakeDriver):
            recording = Recording(max_fps=120)
    with pytest.raises(TypeError, match="@record needs recording, which recording = None turns off"):
        class Off(FakeDriver):
            recording = None

            @record
            async def frames(self, hints):
                yield None
    with pytest.raises(TypeError, match="recording is a Recording"):
        class Wrong(FakeDriver):
            recording = {"max_fps": 10}


@pytest.mark.asyncio
async def test_the_fallback_polls_capture_and_draws_the_pointer(home):
    driver = Recorded()
    driver.screen.pointer = (200, 120)
    started, manifest = await _record(driver, 0.6)
    assert started["recording_id"].startswith("rec-") and started["format"] == "video/mp4; codecs=avc1"
    assert manifest["source"] == "capture" and manifest["cursor"] is True and manifest["ended_by"] == "end"
    (segment,) = manifest["segments"]
    file = home / ".agentenv-recordings" / segment["path"]
    assert segment["uploaded"] is False and segment["bytes"] == file.stat().st_size
    assert segment["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    frames = _decoded(file)
    assert len(frames) == segment["frames"] >= 4 and (frames[0].width, frames[0].height) == (1280, 800)
    pixel = frames[-1].to_image().getpixel((202, 128))
    assert sum(pixel) < 200, f"no pointer drawn at the pointer: {pixel}"
    assert _at(manifest["started_at"]) <= _at(manifest["ended_at"])


@pytest.mark.asyncio
async def test_a_driver_without_the_pointer_group_records_without_a_cursor(home):
    _, manifest = await _record(MinimalDriver(), 0.4)
    assert manifest["cursor"] is False and manifest["segments"][0]["width"] == 640


@pytest.mark.asyncio
async def test_segments_roll_over_with_no_gap_between_them(home):
    _, manifest = await _record(Recorded(), 2.6)
    segments = manifest["segments"]
    assert len(segments) >= 2 and [s["index"] for s in segments] == list(range(len(segments)))
    for before, after in zip(segments, segments[1:]):
        gap = (_at(after["started_at"]) - _at(before["ended_at"])).total_seconds()
        assert 0 < gap < 0.25, gap
    for segment in segments:
        assert (_at(segment["ended_at"]) - _at(segment["started_at"])).total_seconds() <= 1.0


@pytest.mark.asyncio
async def test_a_size_change_starts_a_segment_of_the_new_size(home):
    driver = Recorded()
    driver.recording = Recording(max_segment_s=60)
    async with http_client(driver) as http:
        started = (await http.post(START, json={"request_id": "r", "fps": 10})).json()
        await _first_frame(driver, started)
        await asyncio.sleep(0.4)
        driver.screen.resize(640, 400, 1.0)
        await asyncio.sleep(0.4)
        manifest = (await http.post(END, json={"recording_id": started["recording_id"]})).json()
    assert [(s["width"], s["height"]) for s in manifest["segments"]] == [(1280, 800), (640, 400)]


@pytest.mark.asyncio
async def test_captures_that_fail_leave_a_gap_and_recording_resumes(home):
    driver = Recorded()
    driver.recording = Recording(max_segment_s=60)
    async with http_client(driver) as http:
        started = (await http.post(START, json={"request_id": "r", "fps": 10})).json()
        await _first_frame(driver, started)
        await asyncio.sleep(0.4)
        driver.fail["capture"] = DeviceUnavailable("the X server went away")
        await asyncio.sleep(1.6)
        manifest = (await http.post(END, json={"recording_id": started["recording_id"]})).json()
    assert len(manifest["segments"]) == 2 and manifest["ended_by"] == "end"
    gap = (_at(manifest["segments"][1]["started_at"]) - _at(manifest["segments"][0]["ended_at"])).total_seconds()
    assert gap >= 0.9


@pytest.mark.asyncio
async def test_a_stream_that_keeps_failing_ends_the_recording_with_its_error(home, monkeypatch):
    monkeypatch.setattr(recording_module, "FAILING_S", 0.5)

    class Broken(FakeDriver):
        @record
        async def frames(self, hints):
            raise RuntimeError("no display")
            yield

    _, manifest = await _record(Broken(), 2.5)
    assert manifest["ended_by"] == "error" and manifest["error"] == "RuntimeError: no display"
    assert manifest["segments"] == [] and "started_at" not in manifest


@pytest.mark.asyncio
async def test_a_stream_of_frames_keeps_each_frames_capture_time(home):
    _, manifest = await _record(Streamed(), 0.8, start={"fps": 20, "max_edge": 640})
    (segment,) = manifest["segments"]
    assert manifest["source"] == "stream" and (segment["width"], segment["height"]) == (640, 400)
    frames = _decoded(home / ".agentenv-recordings" / segment["path"])
    span = (_at(segment["ended_at"]) - _at(segment["started_at"])).total_seconds()
    assert len(frames) == segment["frames"] >= 10 and abs(frames[-1].time - span) < 0.002
    with av.open(str(home / ".agentenv-recordings" / segment["path"])) as container:
        keyframes = sum(packet.is_keyframe for packet in container.demux(video=0) if packet.size)
    assert keyframes <= 2, f"{keyframes} of {len(frames)} frames were encoded as keyframes"


@pytest.mark.asyncio
async def test_a_native_recorders_segments_are_kept_as_they_come(home):
    _, manifest = await _record(Segmented(), 1.0)
    assert [(s["index"], s["path"].rsplit("/", 1)[1], s["frames"]) for s in manifest["segments"]] == [
        (0, "0000.mp4", 5), (1, "0001.mp4", 5)]
    assert len(_decoded(home / ".agentenv-recordings" / manifest["segments"][1]["path"])) == 5


@pytest.mark.asyncio
async def test_segments_upload_as_they_close_and_end_retries_with_a_fresh_grant(home):
    calls = []

    def uploader(grant):
        fail = grant.root_path == "rec/run-1" and not calls
        upload = AsyncMock(side_effect=TransferError("transfer_unavailable", "down") if fail else None)
        calls.append((grant.root_path, upload))
        return type("Uploader", (), {"upload": upload})()

    with patch.object(recording_module, "NamespaceUploader", side_effect=uploader):
        _, manifest = await _record(Recorded(), 0.5, start={"upload": _grant()},
                                    end={"upload": {**_grant(), "root_path": "rec/run-2"}})
    (segment,) = manifest["segments"]
    assert segment["uploaded"] is True and "upload_error" not in segment
    assert [root for root, _ in calls] == ["rec/run-1", "rec/run-2"]
    assert calls[1][1].await_args.args[0] == segment["path"]
    assert not (home / ".agentenv-recordings" / segment["path"]).exists()


@pytest.mark.asyncio
async def test_an_upload_that_fails_with_no_fresh_grant_is_reported_and_the_file_kept(home):
    failing = AsyncMock(side_effect=TransferError("grant_expired", "expired"))
    with patch.object(recording_module, "NamespaceUploader") as uploader:
        uploader.return_value.upload = failing
        _, manifest = await _record(Recorded(), 0.4, start={"upload": _grant()})
    (segment,) = manifest["segments"]
    assert segment["uploaded"] is False and "grant_expired" in segment["upload_error"]
    assert (home / ".agentenv-recordings" / segment["path"]).exists()


@pytest.mark.asyncio
async def test_retries_join_the_first_start_and_answer_end_again(home):
    async with http_client(Recorded()) as http:
        first = (await http.post(START, json={"request_id": "same"})).json()
        again = (await http.post(START, json={"request_id": "same", "fps": 5})).json()
        await asyncio.sleep(0.3)
        ended = (await http.post(END, json={"recording_id": first["recording_id"]})).json()
        repeat = (await http.post(END, json={"recording_id": first["recording_id"], "reason": "teardown"})).json()
    assert again == first and repeat == ended and ended["ended_by"] == "end"


@pytest.mark.asyncio
async def test_a_start_past_max_concurrent_is_refused_and_a_reason_is_kept(home):
    driver = FakeDriver()
    driver.recording = Recording(max_concurrent=1)
    async with http_client(driver) as http:
        first = (await http.post(START, json={"request_id": "a"})).json()
        refused = await http.post(START, json={"request_id": "b"})
        ended = (await http.post(END, json={"recording_id": first["recording_id"], "reason": "reset"})).json()
        after = await http.post(START, json={"request_id": "c"})
        await http.post(END, json={"recording_id": after.json()["recording_id"]})
    assert refused.status_code == 429 and refused.json()["error"]["code"] == "limit_exceeded"
    assert ended["ended_by"] == "reset" and after.status_code == 200


@pytest.mark.asyncio
async def test_max_duration_and_a_full_disk_end_a_recording_as_limit(home, monkeypatch):
    driver = FakeDriver()
    driver.recording = Recording(max_duration_s=1)
    _, timed = await _record(driver, 1.6)
    monkeypatch.setattr(recording_module, "MIN_FREE_BYTES", 1 << 62)
    _, full = await _record(Recorded(), 0.4, start={"request_id": "full"})
    assert timed["ended_by"] == "limit" and (_at(timed["ended_at"]) - _at(timed["started_at"])).total_seconds() < 1.2
    assert full["ended_by"] == "limit"


@pytest.mark.asyncio
@pytest.mark.parametrize("path, body, status, code", [
    (START, {"request_id": "r", "fps": 0}, 400, "invalid_argument"),
    (START, {"request_id": "r", "fps": 30}, 400, "invalid_argument"),
    (START, {"request_id": "r", "max_edge": 10}, 400, "invalid_argument"),
    (START, {"request_id": "r", "scope": "region"}, 400, "invalid_argument"),
    (START, {"request_id": "r", "scope": "window", "window": "w1"}, 501, "unsupported"),
    (START, {"request_id": "r", "upload": {"root_path": "x"}}, 400, "invalid_argument"),
    (START, {"request_id": ""}, 400, "invalid_argument"),
    (END, {"recording_id": "rec-nope"}, 400, "invalid_argument"),
    (END, {"recording_id": "rec-nope", "reason": "later"}, 400, "invalid_argument"),
])
async def test_bad_calls_are_refused_with_a_code(path, body, status, code):
    async with http_client(Recorded()) as http:
        refused = await http.post(path, json=body)
    assert refused.status_code == status and refused.json()["error"]["code"] == code


@pytest.mark.asyncio
async def test_uploaded_segments_are_not_sent_again_at_end(home):
    upload = AsyncMock()
    with patch.object(recording_module, "NamespaceUploader") as uploader:
        uploader.return_value.upload = upload
        _, manifest = await _record(Recorded(), 1.4, start={"upload": _grant()})
    assert all(s["uploaded"] for s in manifest["segments"]) and upload.await_count == len(manifest["segments"]) >= 2


@pytest.mark.asyncio
async def test_an_encoder_that_fails_on_its_first_frame_leaves_no_empty_segment(home, monkeypatch):
    real, failed = recording_module._Encoder.add, []

    def add(self, frame, at):
        if not failed:
            failed.append(at)
            raise RuntimeError("encoder failed")
        return real(self, frame, at)

    monkeypatch.setattr(recording_module._Encoder, "add", add)
    _, manifest = await _record(Recorded(), 1.6)
    assert failed and manifest["segments"] and all(s["frames"] > 0 for s in manifest["segments"])
    assert sorted(p.name for p in (home / ".agentenv-recordings" / manifest["recording_id"]).iterdir()) == [
        s["path"].rsplit("/", 1)[1] for s in manifest["segments"]]


@pytest.mark.asyncio
async def test_only_the_latest_starts_and_ends_are_kept_for_retries(home, monkeypatch):
    monkeypatch.setattr(recording_module, "KEEP_ENDED", 1)
    monkeypatch.setattr(recording_module, "KEEP_STARTED", 4)
    driver = Recorded()
    async with http_client(driver) as http:
        for name in "abcde":
            started = (await http.post(START, json={"request_id": name})).json()
            await http.post(END, json={"recording_id": started["recording_id"]})
    kept = driver._computer.recordings
    assert list(kept.started) == list("bcde") and len(kept.ended) == 1


class Stalls(FakeDriver):
    """A stream that hands over one frame and then waits forever."""
    recording = Recording(max_duration_s=1, max_concurrent=1)

    @record
    async def frames(self, hints):
        yield RecordedFrame(self.screen.image.copy(), datetime.now(timezone.utc))
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_a_stalled_stream_still_ends_at_max_duration_and_frees_its_slot(home):
    driver = Stalls()
    async with http_client(driver) as http:
        first = (await http.post(START, json={"request_id": "a"})).json()
        await asyncio.sleep(1.4)
        second = await http.post(START, json={"request_id": "b"})
        ended = (await http.post(END, json={"recording_id": first["recording_id"]})).json()
        await http.post(END, json={"recording_id": second.json()["recording_id"]})
    assert second.status_code == 200 and ended["ended_by"] == "limit" and len(ended["segments"]) == 1


@pytest.mark.asyncio
async def test_an_encoder_that_fails_every_frame_ends_the_recording_with_its_error(home, monkeypatch):
    monkeypatch.setattr(recording_module, "FAILING_S", 0.5)
    monkeypatch.setattr(recording_module, "RETRY_S", 0.1)

    def fail(self, frame, at):
        raise RuntimeError("the encoder is out of memory")

    monkeypatch.setattr(recording_module._Encoder, "add", fail)
    _, manifest = await _record(Recorded(), 1.5)
    assert manifest["ended_by"] == "error" and manifest["error"] == "RuntimeError: the encoder is out of memory"


@pytest.mark.asyncio
async def test_an_end_that_comes_while_a_segment_is_sealing_keeps_it(home, monkeypatch):
    real_close = recording_module._Encoder.close

    def slow_close(self):
        time.sleep(0.6)
        real_close(self)

    monkeypatch.setattr(recording_module._Encoder, "close", slow_close)
    _, manifest = await _record(Recorded(), 1.2)
    (segment,) = manifest["segments"]  # the end came while the rolled-over segment was still sealing
    assert segment["frames"] >= 8
    for segment in manifest["segments"]:
        assert len(_decoded(home / ".agentenv-recordings" / segment["path"])) == segment["frames"]


@pytest.mark.asyncio
async def test_recordings_that_stop_themselves_release_their_thread_and_only_the_latest_are_kept(home, monkeypatch):
    monkeypatch.setattr(recording_module, "KEEP_ENDED", 1)
    driver = FakeDriver()
    driver.recording = Recording(max_duration_s=1, max_concurrent=4)
    async with http_client(driver) as http:
        for name in "abc":
            await http.post(START, json={"request_id": name})
        await asyncio.sleep(1.6)
        stopped = list(driver._computer.recordings.running.values())
        await http.post(START, json={"request_id": "d"})
        running = driver._computer.recordings.running
        assert all(r.done and r.executor._shutdown for r in stopped)
        assert len(running) == 2 and sum(not r.done for r in running.values()) == 1
        for recording_id in list(running):
            await http.post(END, json={"recording_id": recording_id})


def _counted(n: int, closed: threading.Event, fail: Exception | None = None):
    """A blocking source of ``n`` frames numbered from 0, a millisecond apart, that may fail after them."""
    def frames():
        try:
            for i in range(n):
                yield RecordedFrame(i, datetime.now(timezone.utc))
                time.sleep(0.001)
            if fail is not None:
                raise fail
        finally:
            closed.set()
    return frames


@pytest.mark.asyncio
async def test_frames_in_thread_hands_over_a_blocking_sources_frames_and_what_it_raises():
    closed = threading.Event()
    with pytest.raises(RuntimeError, match="the X server went away"):
        got = [f.image async for f in frames_in_thread(_counted(5, closed, RuntimeError("the X server went away")))]
    got = [f.image async for f in frames_in_thread(_counted(5, closed))]
    assert got == [0, 1, 2, 3, 4] and closed.is_set()
    late = frames_in_thread(_counted(5, threading.Event(), RuntimeError("gone")), keep=1)
    await anext(late)
    await asyncio.sleep(0.2)
    with pytest.raises(RuntimeError, match="gone"):
        [f async for f in late]


@pytest.mark.asyncio
async def test_frames_in_thread_keeps_the_latest_frames_for_a_slow_reader_and_closes_its_source_when_stopped():
    done, stopped = threading.Event(), threading.Event()
    stream = frames_in_thread(_counted(30, done), keep=4)
    first = (await anext(stream)).image
    await asyncio.to_thread(done.wait, 5)
    await asyncio.sleep(0.05)
    rest = [f.image async for f in stream]
    endless = frames_in_thread(_counted(1_000_000, stopped))
    await anext(endless)
    await endless.aclose()
    assert first == 0 and rest == [26, 27, 28, 29]
    assert await asyncio.to_thread(stopped.wait, 5)
