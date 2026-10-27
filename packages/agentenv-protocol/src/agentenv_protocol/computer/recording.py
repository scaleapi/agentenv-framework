"""Recording a computer's display, computer-recording/v1: the base class runs each recording from the driver's
``@record`` stream, or by polling ``@capture`` where the driver has none, encodes frames as H.264 segments, uploads
each segment as it closes and seals the recording at ``end`` with its manifest."""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import shutil
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import av
from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import ValidationError

from ..transfers import NamespaceUploader, WriteNamespaceGrant
from .errors import InvalidArgument, LimitExceeded, Unsupported
from .frames import to_pil
from .manifest import ManifestSegment, RecordingManifest

if TYPE_CHECKING:
    from .runtime import ComputerRuntime

MP4 = "video/mp4; codecs=avc1"
#: Where segments are written, relative to the session user's home, and kept when there is no grant.
RECORDINGS_DIR = ".agentenv-recordings"
SCOPES = ("display", "window")
REASONS = ("end", "reset", "teardown")
DEFAULT_FPS = 10
#: The fallback polls ``@capture``, a full read of the screen each pass, so it is capped lower than streams.
MAX_POLLED_FPS = 15
#: Free disk below this ends a recording (``limit``).
MIN_FREE_BYTES = 512 << 20
#: A stream that keeps failing this long ends its recording (``error``).
FAILING_S = 60.0
#: Frames further apart than this start a new segment.
GAP_S = 2.0
#: How long a stream that stopped or failed waits before it is started again.
RETRY_S = 1.0
#: How often free disk is checked while frames arrive.
DISK_CHECK_S = 5.0
KEEP_ENDED = 64
KEEP_STARTED = 4 * KEEP_ENDED
MIN_EDGE, MAX_EDGE = 64, 8192
#: libx264 tuned for live capture: quick to encode, at a quality screen text survives.
H264_OPTIONS = {"preset": "ultrafast", "tune": "zerolatency", "crf": "28"}
KEYFRAME_S = 2
#: The frames ``frames_in_thread`` holds for an encoder that falls behind; past it the oldest is dropped.
THREAD_QUEUE = 8


@dataclass(frozen=True)
class Recording:
    """A driver's recording limits, as its ``recording`` class attribute; every driver has the defaults."""
    max_fps: int = 15
    max_concurrent: int = 2
    max_duration_s: int = 86_400
    max_segment_s: int = 15


@dataclass(frozen=True)
class RecordHints:
    """What ``@record`` is asked for. A stream that yields segments writes its files under ``directory`` and rolls
    them over at ``max_segment_s``; one that yields frames leaves both to the base class."""
    scope: str
    fps: int
    max_edge: Optional[int]
    cursor: bool
    max_segment_s: int
    directory: Path


@dataclass(frozen=True)
class RecordedFrame:
    """A frame of a stream: a Pillow image or a PyAV ``VideoFrame``, and when it was captured."""
    image: Any
    captured_at: datetime


@dataclass(frozen=True)
class RecordedSegment:
    """A finished segment a native recorder wrote, and the times of its first and last frames."""
    path: Path
    started_at: datetime
    ended_at: datetime
    width: int
    height: int
    frames: int
    format: str = MP4


def check_recording(recording: object) -> list[str]:
    if recording is None:
        return []
    if not isinstance(recording, Recording):
        return [f"recording is a Recording, got {type(recording).__name__}"]
    problems = []
    for name, low, high in (("max_fps", 1, 60), ("max_concurrent", 1, 16), ("max_duration_s", 1, 86_400),
                            ("max_segment_s", 1, 3600)):
        value = getattr(recording, name)
        if not isinstance(value, int) or not low <= value <= high:
            problems.append(f"recording.{name} is an integer from {low} to {high}, got {value!r}")
    return problems


def recording_params(recording: Recording, streams: bool) -> dict:
    """The extension's params on the card: what ``start`` can be asked for, and the limits."""
    most = recording.max_fps if streams else min(recording.max_fps, MAX_POLLED_FPS)
    return {"formats": [MP4], "scopes": ["display"], "fps": {"default": min(DEFAULT_FPS, most), "max": most},
            "cursor": True, "source": "stream" if streams else "capture", "max_segment_s": recording.max_segment_s,
            "max_concurrent": recording.max_concurrent, "max_duration_s": recording.max_duration_s}


class Recordings:
    """The recordings of one driver: started by request id, open ones counted against ``max_concurrent``, ended
    ones kept with their manifests so a repeated ``end`` answers the same."""

    def __init__(self, runtime: ComputerRuntime) -> None:
        self.runtime = runtime
        self.running: dict[str, _Recording] = {}
        self.ended: OrderedDict[str, dict] = OrderedDict()
        self.started: OrderedDict[str, dict] = OrderedDict()
        self._lock = asyncio.Lock()

    async def start(self, request_id: str, scope: str, window: Optional[str], fps: Optional[int],
                    max_edge: Optional[int], cursor: bool, upload: Optional[dict]) -> dict:
        features: Recording = self.runtime.driver.recording
        if not (isinstance(request_id, str) and request_id):
            raise InvalidArgument("request_id is a non-empty string")
        async with self._lock:
            if request_id in self.started:
                return self.started[request_id]
            hints, grant = self._hints(features, scope, window, fps, max_edge, cursor, upload)
            stopped = [rid for rid, r in self.running.items() if r.done]
            for rid in stopped[:max(0, len(stopped) - KEEP_ENDED)]:  # stopped by itself and never ended
                self.running.pop(rid)
            if sum(not r.done for r in self.running.values()) >= features.max_concurrent:
                raise LimitExceeded(f"{features.max_concurrent} recordings are running, the most this computer "
                                    "records at once; end one first")
            recording_id = f"rec-{uuid.uuid4().hex[:12]}"
            root = Path.home() / RECORDINGS_DIR
            hints = dataclasses.replace(hints, directory=root / recording_id)
            hints.directory.mkdir(parents=True, exist_ok=True)
            recording = _Recording(recording_id, hints, features, root, grant, self.runtime)
            self.running[recording_id] = recording
            recording.task = asyncio.create_task(recording.run())
            recording.timer = asyncio.get_running_loop().call_later(features.max_duration_s, recording.expire)
            answer = {"recording_id": recording_id, "started_at": _iso(recording.requested_at), "format": MP4}
            self.started[request_id] = answer
            while len(self.started) > KEEP_STARTED:
                self.started.popitem(last=False)
            return answer

    async def end(self, recording_id: str, upload: Optional[dict], reason: str) -> dict:
        if reason not in REASONS:
            raise InvalidArgument(f"reason is one of {list(REASONS)}, got {reason!r}")
        if recording_id in self.ended:
            return self.ended[recording_id]
        recording = self.running.get(recording_id)
        if recording is None:
            raise InvalidArgument(f"there is no recording {recording_id!r} on this computer")
        grant = _grant(upload) if upload is not None else None
        manifest = await recording.end(reason, grant)
        self.running.pop(recording_id, None)
        self.ended[recording_id] = manifest
        while len(self.ended) > KEEP_ENDED:
            self.ended.popitem(last=False)
        return manifest

    def _hints(self, features: Recording, scope: str, window: Optional[str], fps: Optional[int],
               max_edge: Optional[int], cursor: bool, upload: Optional[dict]) -> tuple[RecordHints, Any]:
        if scope not in SCOPES:
            raise InvalidArgument(f"scope is 'display' or 'window', got {scope!r}")
        if scope == "window" or window is not None:
            raise Unsupported("this computer records the whole display only; window recordings need computer-window/v1")
        params = recording_params(features, self.runtime.has("record"))
        fps = params["fps"]["default"] if fps is None else fps
        if not (isinstance(fps, int) and 1 <= fps <= params["fps"]["max"]):
            raise InvalidArgument(f"fps is an integer from 1 to {params['fps']['max']}, got {fps!r}")
        if max_edge is not None and not (isinstance(max_edge, int) and MIN_EDGE <= max_edge <= MAX_EDGE):
            raise InvalidArgument(f"max_edge is an integer from {MIN_EDGE} to {MAX_EDGE}, got {max_edge!r}")
        grant = _grant(upload) if upload is not None else None
        return RecordHints(scope, fps, max_edge, bool(cursor), features.max_segment_s, Path()), grant


class _Recording:
    def __init__(self, recording_id: str, hints: RecordHints, features: Recording, root: Path,
                 grant: Optional[WriteNamespaceGrant], runtime: ComputerRuntime) -> None:
        self.id = recording_id
        self.hints = hints
        self.features = features
        self.root = root
        self.runtime = runtime
        self.uploader = NamespaceUploader(grant) if grant is not None else None
        self.requested_at = datetime.now(timezone.utc)
        self.streams = runtime.has("record")
        self.cursor = hints.cursor if self.streams else hints.cursor and runtime.has("pointer")
        self.segments: list[_Segment] = []
        self.encoder: Optional[_Encoder] = None
        self.ended_by: Optional[str] = None
        self.error: Optional[str] = None
        self.done = False
        self.task: Optional[asyncio.Task] = None
        self.timer: Optional[asyncio.TimerHandle] = None
        self.released: Optional[asyncio.Future] = None
        self.uploads: set[asyncio.Task] = set()
        self.sealing: set[asyncio.Future] = set()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"computer-{recording_id}")
        self._last_disk_check = 0.0

    async def run(self) -> None:
        failing_since: Optional[float] = None
        try:
            while True:
                try:
                    async for item in self._source():
                        if isinstance(item, RecordedSegment):
                            await self._add_segment(item)
                        else:
                            await self._add_frame(item)
                        failing_since = None
                        if self.ended_by is not None:
                            return
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    failing_since = failing_since or time.monotonic()
                    self.error = f"{type(e).__name__}: {e}"
                    if time.monotonic() - failing_since > FAILING_S:
                        self.ended_by = "error"
                        return
                await self._close()
                await asyncio.sleep(RETRY_S)
        finally:
            try:
                await self._close()
            finally:
                self.done = True
                self.released = asyncio.ensure_future(self._release())

    def expire(self) -> None:
        """``max_duration_s`` is up, whether or not frames are arriving."""
        if not self.done:
            self.ended_by = self.ended_by or "limit"
            self.task.cancel()

    async def _release(self) -> None:
        """Once every closing segment is sealed, the encoder's thread goes."""
        if self.sealing:
            await asyncio.gather(*self.sealing, return_exceptions=True)
        self.executor.shutdown(wait=False)

    def _source(self) -> AsyncIterator:
        if self.streams:
            return self.runtime.stream(self.hints)
        return _poll(self.runtime, self.hints, self.cursor)

    async def end(self, reason: str, grant: Optional[WriteNamespaceGrant]) -> dict:
        if self.timer is not None:
            self.timer.cancel()
        if self.task is not None and not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        if self.ended_by is None:
            self.ended_by = reason
        await (self.released or asyncio.ensure_future(self._release()))
        if self.uploads:
            await asyncio.gather(*self.uploads, return_exceptions=True)
        uploader = NamespaceUploader(grant) if grant is not None else self.uploader
        if uploader is not None:
            for segment in self.segments:
                if not segment.entry.uploaded:
                    await self._upload(uploader, segment)
        with contextlib.suppress(OSError):
            self.hints.directory.rmdir()
        entries = [s.entry for s in self.segments]
        return RecordingManifest(
            recording_id=self.id, scope=self.hints.scope, format=MP4, fps=self.hints.fps, cursor=self.cursor,
            source="stream" if self.streams else "capture", started_at=entries[0].started_at if entries else None,
            ended_at=entries[-1].ended_at if entries else None, ended_by=self.ended_by, segments=entries,
            error=self.error if self.ended_by == "error" else None).wire()

    async def _add_frame(self, item: RecordedFrame) -> None:
        frame = await self._in_thread(_video_frame, item.image, self.hints.max_edge)
        at = item.captured_at
        if self.encoder is not None and ((frame.width, frame.height) != self.encoder.size
                                         or (at - self.encoder.last_at).total_seconds() > GAP_S
                                         or (at - self.encoder.started_at).total_seconds() >= self.hints.max_segment_s):
            await self._close()
        if self.encoder is None:
            path = self.hints.directory / f"{len(self.segments):04d}.mp4"
            self.encoder = await self._in_thread(_Encoder, path, frame.width, frame.height, self.hints.fps, at)
        await self._in_thread(self.encoder.add, frame, at)
        self._check_disk()

    async def _add_segment(self, item: RecordedSegment) -> None:
        index = len(self.segments)
        target = self.hints.directory / f"{index:04d}.mp4"
        if Path(item.path) != target:
            await asyncio.to_thread(shutil.move, str(item.path), target)
        await self._finish(target, item.started_at, item.ended_at, item.width, item.height, item.frames)
        self._check_disk()

    def _check_disk(self) -> None:
        if time.monotonic() - self._last_disk_check > DISK_CHECK_S:
            self._last_disk_check = time.monotonic()
            if shutil.disk_usage(self.root).free < MIN_FREE_BYTES:
                self.ended_by = "limit"

    async def _close(self) -> None:
        """Seal the open segment. Sealing runs on even when ``end`` cancels the recording meanwhile, and ``end``
        waits for it, so a segment is never lost between its encoder and the manifest."""
        encoder, self.encoder = self.encoder, None
        if encoder is None:
            return
        sealing = asyncio.ensure_future(self._seal(encoder))
        self.sealing.add(sealing)
        sealing.add_done_callback(self.sealing.discard)
        await asyncio.shield(sealing)

    async def _seal(self, encoder: _Encoder) -> None:
        await self._in_thread(encoder.close)
        if encoder.frames:
            await self._finish(encoder.path, encoder.started_at, encoder.last_at, *encoder.size, encoder.frames)
        else:
            encoder.path.unlink(missing_ok=True)

    async def _finish(self, path: Path, started_at: datetime, ended_at: datetime, width: int, height: int,
                      frames: int) -> None:
        size, digest = await asyncio.to_thread(_digest, path)
        segment = _Segment(ManifestSegment(
            index=len(self.segments), path=f"{self.id}/{path.name}", started_at=_iso(started_at),
            ended_at=_iso(ended_at), width=width, height=height, frames=frames, bytes=size, sha256=digest,
            uploaded=False), path)
        self.segments.append(segment)
        if self.uploader is not None:
            task = asyncio.create_task(self._upload(self.uploader, segment))
            self.uploads.add(task)
            task.add_done_callback(self.uploads.discard)

    async def _upload(self, uploader: NamespaceUploader, segment: _Segment) -> None:
        try:
            await uploader.upload(segment.entry.path, segment.file)
        except Exception as e:
            segment.entry.upload_error = f"{getattr(e, 'code', type(e).__name__)}: {e}"
            return
        segment.entry.uploaded, segment.entry.upload_error = True, None
        segment.file.unlink(missing_ok=True)

    async def _in_thread(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)


@dataclass
class _Segment:
    """A closed segment: its manifest entry, and its file until it uploads."""
    entry: ManifestSegment
    file: Path


class _Encoder:
    """One segment's H.264 encoder: each frame's timestamp is its ``captured_at`` from the segment's first frame, so
    the video's timeline is the computer's clock."""

    def __init__(self, path: Path, width: int, height: int, fps: int, started_at: datetime) -> None:
        self.path = path
        self.size = (width, height)
        self.started_at = self.last_at = started_at
        self.frames = 0
        self._last_pts = -1
        self.container = av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=fps,
                                                options={**H264_OPTIONS, "g": str(KEYFRAME_S * fps)})
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, "yuv420p"
        self.stream.codec_context.time_base = Fraction(1, 1000)

    def add(self, frame: Any, at: datetime) -> None:
        pts = max(int((at - self.started_at).total_seconds() * 1000), self._last_pts + 1)
        frame.pts, frame.time_base = pts, Fraction(1, 1000)
        frame.pict_type = av.video.frame.PictureType.NONE  # a decoded stream marks every frame a keyframe
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self._last_pts, self.last_at = pts, at
        self.frames += 1

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


async def frames_in_thread(frames: Callable[[], Iterator[RecordedFrame]],
                           keep: int = THREAD_QUEUE) -> AsyncIterator[RecordedFrame]:
    """A ``@record`` stream from a blocking source: the generator ``frames()`` runs in a thread of its own, the
    latest ``keep`` of its frames wait for the encoder, what it raises is raised here after them, and it is closed
    when the recording stops."""
    loop, stop, wake = asyncio.get_running_loop(), threading.Event(), asyncio.Event()
    held: deque[RecordedFrame] = deque(maxlen=keep)
    ended: list[Optional[Exception]] = []  # the source's end, and what it raised: never dropped like a frame

    def hold(frame: RecordedFrame) -> None:
        held.append(frame)
        wake.set()

    def end(error: Optional[Exception]) -> None:
        ended.append(error)
        wake.set()

    def post(fn: Callable, item: Any) -> None:
        with contextlib.suppress(RuntimeError):  # the loop has closed: nobody is reading
            loop.call_soon_threadsafe(fn, item)

    def pump() -> None:
        error = None
        try:
            with contextlib.closing(frames()) as source:
                for frame in source:
                    if stop.is_set():
                        return
                    post(hold, frame)
        except Exception as e:
            error = e
        finally:
            post(end, error)

    threading.Thread(target=pump, name="computer-record", daemon=True).start()
    try:
        while True:
            if held:
                yield held.popleft()
            elif ended:
                if ended[0] is not None:
                    raise ended[0]
                return
            else:
                wake.clear()
                await wake.wait()
    finally:
        stop.set()


async def _poll(runtime: ComputerRuntime, hints: RecordHints, cursor: bool) -> AsyncIterator[RecordedFrame]:
    """The fallback stream: ``@capture`` every ``1/fps`` seconds, with a pointer drawn where the driver says it is."""
    interval, due = 1 / hints.fps, time.monotonic()
    while True:
        display = await runtime.display_now()
        raw = await runtime.call("capture")
        captured_at = datetime.now(timezone.utc)
        image = to_pil(raw, display)
        if cursor:
            x, y = await runtime.pointer()
            _draw_pointer(image, x * display.scale, y * display.scale)
        yield RecordedFrame(image, captured_at)
        due = max(due + interval, time.monotonic())
        await asyncio.sleep(due - time.monotonic())


def _draw_pointer(image: PILImage.Image, x: float, y: float) -> None:
    arrow = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
    ImageDraw.Draw(image).polygon([(x + dx, y + dy) for dx, dy in arrow], fill=(0, 0, 0), outline=(255, 255, 255))


def _video_frame(image: Any, max_edge: Optional[int]) -> Any:
    """``image`` as a ``yuv420p`` frame of even size, its longer edge at most ``max_edge``."""
    frame = image if isinstance(image, av.VideoFrame) else av.VideoFrame.from_image(image.convert("RGB"))
    width, height = frame.width, frame.height
    if max_edge is not None and max(width, height) > max_edge:
        ratio = max_edge / max(width, height)
        width, height = round(width * ratio), round(height * ratio)
    return frame.reformat(width=max(2, width - width % 2), height=max(2, height - height % 2), format="yuv420p")


def _grant(upload: object) -> WriteNamespaceGrant:
    try:
        return WriteNamespaceGrant.model_validate(upload)
    except ValidationError:
        raise InvalidArgument("upload is a write_namespace grant") from None


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return path.stat().st_size, digest.hexdigest()


def _iso(at: datetime) -> str:
    return at.isoformat(timespec="milliseconds").replace("+00:00", "Z")
