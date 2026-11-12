"""Recording a website browser, browser-recording/v1: frames of the tab an agent last acted in, one at the start, one
after each burst of its actions settles and one at the end, each keyed to the log entries it follows and uploaded under
the recording's grant as it's taken; and, at the end, the log's lines since the start, its actions and tab events, and
a manifest of it all. A frame the browser couldn't take is in the manifest with why, and counts as missing."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, ValidationError

from ..computer.errors import InvalidArgument
from ..transfers import TRANSFER_STALL_BUDGET_SECONDS, NamespaceUploader, WriteNamespaceGrant
from .control import FRAME_CAPTURE_S
from .log import ERROR_CHARS, BrowserLog

if TYPE_CHECKING:
    from .control import BrowserControl

logger = logging.getLogger(__name__)

BROWSER_RECORDING = "urn:agentenv:browser-recording/v1"
MANIFEST = "manifest.json"
LOG = "log.jsonl"
FRAMES = "frames"
REASONS = ("end", "reset", "teardown")
#: How long a burst of an agent's actions must go quiet before its frame is taken, so typing a word is one frame.
SETTLE_S = 0.5
#: How long a frame waits for its tab's page to load after a navigation.
LOAD_WAIT_S = 5.0
#: How long a recording may run, which its grant lasts, and how many frames it takes at most.
MAX_DURATION_S = 4 * 3600
MAX_FRAMES = 5000
#: The ways in whose actions are an agent's: CDP commands and MCP tool calls, not agentenv's own.
AGENTS = ("cdp", "mcp")
CONTENT_TYPES = {"jpg": "image/jpeg", "png": "image/png"}
#: How long the recordings a server ended are kept for an end that comes again.
KEEP_ENDED = 64
#: How long an end waits for the entries still in flight to be written, so its log covers them, and how often it
#: looks again.
LOG_WAIT_S = 10.0
LOG_POLL_S = 0.2
#: The longest a start takes, its first frame taken and uploaded, and an end, a frame still being taken, its last two,
#: the wait for its log and the log's and the manifest's uploads, with each upload as long as the transfer helpers
#: allow one that stalls and a reply's margin: what a client of the extension waits on each.
_FRAME_S = LOAD_WAIT_S + FRAME_CAPTURE_S + TRANSFER_STALL_BUDGET_SECONDS
_REPLY_S = 30.0
START_S = _FRAME_S + _REPLY_S
END_S = 3 * _FRAME_S + LOG_WAIT_S + 2 * TRANSFER_STALL_BUDGET_SECONDS + _REPLY_S


class RecordedFrame(BaseModel):
    """One frame: the first, one after a burst of actions, whose entries it names in ``after``, or the last; its tab
    and URL, and its file under the recording's folder, or why there is none. Its ``kind`` and ``at`` are None where
    they aren't known, in a manifest rebuilt from the files a recording uploaded."""

    index: int
    kind: Optional[Literal["first", "action", "last"]] = None
    at: Optional[str] = None
    after: list[int] = []
    tab: Optional[str] = None
    url: Optional[str] = None
    path: Optional[str] = None
    content_type: Optional[str] = None
    uploaded: bool = False
    error: Optional[str] = None


class BrowserRecordingManifest(BaseModel):
    """A recording as it ended: its frames, and its log's lines, entries ``log_after`` + 1 to ``log_last``, as
    ``log``, the actions and tab events between its start and its end, with any of those entries it doesn't hold, still
    unwritten or unreadable at its end, in ``log_missing``."""

    recording_id: str
    started_at: str
    ended_at: Optional[str] = None
    ended_by: Optional[Literal["end", "reset", "teardown", "error"]] = None
    error: Optional[str] = None
    frames_on: bool = True
    frames: list[RecordedFrame] = []
    log: Optional[str] = None
    log_after: int = 0
    log_last: int = 0
    log_uploaded: bool = False
    log_missing: list[int] = []

    @property
    def missing_frames(self) -> int:
        return sum(1 for frame in self.frames if not frame.uploaded)


def frame_path(index: int, extension: str) -> str:
    """Where frame ``index`` uploads, under its recording's folder."""
    return f"{FRAMES}/{index:04d}.{extension}"


def frame_of(path: str) -> Optional[tuple[int, str]]:
    """The index and content type of the frame ``frame_path`` put at ``path``, or None for another file."""
    if (found := re.fullmatch(rf"{FRAMES}/(\d+)\.(\w+)", path)) and found[2] in CONTENT_TYPES:
        return int(found[1]), CONTENT_TYPES[found[2]]
    return None


class BrowserRecordings:
    """The recordings of the browser that ``log`` and ``control`` are, followed from the log as its actions are
    written."""

    def __init__(self, log: BrowserLog, control: BrowserControl) -> None:
        self.log, self.control = log, control
        self._open: dict[str, _Recording] = {}
        self._by_request: dict[str, str] = {}
        self._ended: dict[str, dict] = {}
        log.watchers.append(self.acted)

    async def start(self, request_id: str, upload: dict, frames: bool = True) -> dict:
        """A recording under the grant ``upload``, or the one a start with the same ``request_id`` began, which a
        retried start joins: its ``recording_id`` and ``started_at``."""
        if (known := self._by_request.get(request_id)) in self._open:
            recording = self._open[known]
            return {"recording_id": recording.manifest.recording_id, "started_at": recording.manifest.started_at}
        recording = _Recording(self, _grant(upload), frames)
        self._open[recording.manifest.recording_id] = recording
        self._by_request[request_id] = recording.manifest.recording_id
        if frames:
            await recording.take("first")
        return {"recording_id": recording.manifest.recording_id, "started_at": recording.manifest.started_at}

    async def end(self, recording_id: str, upload: Optional[dict] = None, reason: str = "end") -> dict:
        """The recording ended, its last frames taken and its log and manifest uploaded, under ``upload`` when one is
        given, else under its start's grant: its manifest. One already ended answers its manifest again."""
        if reason not in REASONS:
            raise InvalidArgument(f"reason is one of {', '.join(REASONS)}, not {reason!r}")
        if recording_id in self._ended:
            return self._ended[recording_id]
        recording = self._open.get(recording_id)
        if recording is None:
            raise InvalidArgument(f"no recording {recording_id} is open here")
        manifest = await recording.end(_grant(upload) if upload is not None else None, reason)
        self._open.pop(recording_id, None)
        self._ended[recording_id] = manifest
        while len(self._ended) > KEEP_ENDED:
            self._ended.pop(next(iter(self._ended)))
        return manifest

    async def end_all(self, reason: str = "teardown") -> list[dict]:
        """Every open recording ended, as its env closes or resets."""
        return [await self.end(recording_id, None, reason) for recording_id in list(self._open)]

    def acted(self, seq: int, action: dict) -> None:
        """An action written to the log: an agent's acting one starts, or holds back, the frame of its burst."""
        if action.get("acts") and action.get("via") in AGENTS:
            for recording in self._open.values():
                recording.acted(seq, action.get("tab"))


class _Recording:
    def __init__(self, recordings: BrowserRecordings, grant: WriteNamespaceGrant, frames: bool) -> None:
        self.recordings, self.uploader, self.frames_on = recordings, NamespaceUploader(grant), frames
        recording_id = uuid.uuid4().hex
        self.manifest = BrowserRecordingManifest(recording_id=recording_id, started_at=_iso(), frames_on=frames,
                                                 log_after=recordings.log.next_seq - 1)
        #: The acting entries since the last frame, and the tab the latest of them was on.
        self.pending: list[int] = []
        self.tab: Optional[str] = None
        self._timer: Optional[asyncio.TimerHandle] = None
        self._taking: set[asyncio.Future] = set()
        self._lock = asyncio.Lock()
        self._ending = False

    def acted(self, seq: int, tab: Optional[str]) -> None:
        if not self.frames_on or self._ending:
            return
        self.pending.append(seq)
        if tab:
            self.tab = tab
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.get_running_loop().call_later(SETTLE_S, self._settled)

    def _settled(self) -> None:
        self._timer = None
        taking = asyncio.ensure_future(self.take("action"))
        self._taking.add(taking)
        taking.add_done_callback(self._taking.discard)

    async def take(self, kind: str) -> None:
        """The frame of ``kind``, of the tab the agent last acted in, uploaded as it's taken; one an action burst
        brings only once that burst has entries no frame followed yet."""
        async with self._lock:
            after, self.pending = self.pending, []
            if kind == "action" and not after:
                return
            index = len(self.manifest.frames)
            frame = RecordedFrame(index=index, kind=kind, at=_iso(), after=after, tab=self.tab)
            self.manifest.frames.append(frame)
            if index >= MAX_FRAMES:
                frame.error = f"the recording reached its {MAX_FRAMES} frames"
                return
            try:
                data, extension, frame.tab, frame.url = await self.recordings.control.frame(self.tab, LOAD_WAIT_S)
                frame.at = _iso()
                frame.path, frame.content_type = frame_path(index, extension), CONTENT_TYPES[extension]
                await self.uploader.upload(self._path(frame.path), base64.b64decode(data))
                frame.uploaded = True
            except Exception as e:
                frame.error = f"{type(e).__name__}: {e}"[:ERROR_CHARS]
                logger.warning("browser recording %s: frame %d: %s", self.manifest.recording_id, index, frame.error)

    async def end(self, grant: Optional[WriteNamespaceGrant], reason: str) -> dict:
        self._ending = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        await asyncio.gather(*self._taking, return_exceptions=True)
        if grant is not None:
            self.uploader = NamespaceUploader(grant)
        if self.frames_on:
            await self.take("action")
            await self.take("last")
        manifest = self.manifest
        manifest.log_last = self.recordings.log.next_seq - 1
        lines = await self._log_lines()
        manifest.log_missing = sorted(set(range(manifest.log_after + 1, manifest.log_last + 1))
                                      - {line.get("seq") for line in lines})
        text = "".join(json.dumps(line, default=str) + "\n" for line in lines)
        manifest.log = LOG
        try:
            await self.uploader.upload(self._path(LOG), text.encode())
            manifest.log_uploaded = True
        except Exception as e:
            manifest.error = f"the log didn't upload: {type(e).__name__}: {e}"[:ERROR_CHARS]
        manifest.ended_at, manifest.ended_by = _iso(), reason
        wire = manifest.model_dump(mode="json")
        try:
            await self.uploader.upload(self._path(MANIFEST), json.dumps(wire, indent=1).encode())
        except Exception as e:
            logger.warning("browser recording %s: its manifest didn't upload: %s", manifest.recording_id, e)
        return wire

    async def _log_lines(self) -> list[dict]:
        """The log's lines from the recording's start to its end, once every one of them is written, or what's
        written of them after ``LOG_WAIT_S``."""
        manifest, log = self.manifest, self.recordings.log
        deadline = asyncio.get_running_loop().time() + LOG_WAIT_S
        wanted = manifest.log_last - manifest.log_after
        while True:
            lines = [line for line in (await log.read_async(manifest.log_after, limit=wanted))["lines"]
                     if line.get("seq", 0) <= manifest.log_last]
            if len(lines) >= wanted or asyncio.get_running_loop().time() >= deadline:
                return lines
            await asyncio.sleep(LOG_POLL_S)

    def _path(self, name: str) -> str:
        return f"{self.manifest.recording_id}/{name}"


def _grant(upload: object) -> WriteNamespaceGrant:
    try:
        return WriteNamespaceGrant.model_validate(upload)
    except ValidationError as e:
        raise InvalidArgument(f"upload must be a write namespace grant: {e.errors()[0]['msg']}") from None


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
