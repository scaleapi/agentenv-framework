"""Recording a website browser, browser-recording/v1: one recording per context, a lease's or the default one's. A
recording has a video of the tab its context's agent is on (``stream.TabStream``), encoded as computer-recording/v1's
is (``computer.recording.Video``), each segment uploaded as it closes; at its end the log's lines of its context since
its start, its actions and tab events, and the browser's whole network log as HAR unless it's asked not to; and, when
asked, frames of that tab, one at the start, one after each burst of an agent's actions settles and one at the end,
each keyed to the log entries it follows. A context an agent makes over an address scoped to the recorded context is
recorded with it, in ``contexts/<label>/``, with the same settings. Everything uploads under the recording's grant, the
manifest last; what couldn't be taken or uploaded is in the manifest with why, and counts as missing."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, ValidationError

from ..computer.errors import InvalidArgument, LimitExceeded, TargetUnavailable, Unsupported
from ..computer.manifest import ManifestSegment
from ..transfers import TRANSFER_STALL_BUDGET_SECONDS, NamespaceUploader, WriteNamespaceGrant
from .control import FRAME_CAPTURE_S
from .log import ERROR_CHARS, BrowserLog
from .network import HAR_FILE, NetworkLog, har

try:  # av and Pillow, which a website browser app installs with its own requirements
    from ..computer.recording import (FIRST_FRAME_S, MAX_EDGE, MIN_EDGE, MP4, RECORDINGS_DIR, Recording, RecordHints,
                                      Video)
    from .stream import BLANK_SIZE, TabStream
except ImportError:  # an app installed without them records no video, so it starts no recording
    Video = None
    MP4, FIRST_FRAME_S, MIN_EDGE, MAX_EDGE, BLANK_SIZE = "video/mp4; codecs=avc1", 5.0, 64, 8192, (1280, 800)

if TYPE_CHECKING:
    from .contexts import Contexts, Lease
    from .control import BrowserControl

logger = logging.getLogger(__name__)

BROWSER_RECORDING = "urn:agentenv:browser-recording/v1"
MANIFEST = "manifest.json"
LOG = "log.jsonl"
FRAMES = "frames"
VIDEO = "video"
CONTEXTS = "contexts"
REASONS = ("end", "reset", "teardown")
#: How a recording ended: as its end asked, its context's end, its limit or an error.
EndedBy = Literal["end", "reset", "teardown", "context_ended", "limit", "error"]
#: How long a burst of an agent's actions must go quiet before its frame is taken, so typing a word is one frame.
SETTLE_S = 0.5
#: How long a frame waits for its tab's page to load after a navigation.
LOAD_WAIT_S = 5.0
#: How long a recording may run, which its grant lasts, and how many frames it takes at most.
MAX_DURATION_S = 4 * 3600
MAX_FRAMES = 5000
#: Its video's frames a second by default and at most, and how long each segment runs.
DEFAULT_FPS, MAX_FPS, SEGMENT_S = 10, 15, 15
#: How many recordings run at once at most, and how many contexts an agent makes each records with it.
MAX_OPEN, MAX_CHILDREN = 8, 4
#: The objects a recording uploads at most, children's among them: frames, segments, its log, HAR and manifest.
MAX_SEGMENTS = 2 * MAX_DURATION_S // SEGMENT_S
MAX_OBJECTS = (1 + MAX_CHILDREN) * (MAX_FRAMES + MAX_SEGMENTS + 3)
#: The ways in whose actions are an agent's: CDP commands and MCP tool calls, not agentenv's own.
AGENTS = ("cdp", "mcp")
#: The CDP commands that move a video to the tab they're on, besides every ``Input`` one: those that show an agent at
#: work there, not the setup a client sends every page it attaches to, its scripts, bindings and emulation.
FOLLOWED = frozenset({"Page.navigate", "Page.reload", "Page.navigateToHistoryEntry", "Page.bringToFront",
                      "Page.handleJavaScriptDialog", "Target.activateTarget", "DOM.setFileInputFiles"})
CONTENT_TYPES = {"jpg": "image/jpeg", "png": "image/png"}
#: How long the recordings a server ended are kept for an end that comes again.
KEEP_ENDED = 64
#: How long an end waits for the entries still in flight to be written, so its log covers them, and how often it
#: looks again.
LOG_WAIT_S = 10.0
LOG_POLL_S = 0.2
#: Why a browser records no video.
NO_VIDEO = "this browser app records no video: its server needs av and Pillow, which its own requirements install"
#: The longest a start takes, its first frame taken and uploaded, and an end, a frame still being taken, its last two,
#: the wait for its log, the video's last segment and the log's, HAR's and manifest's uploads, with each upload as
#: long as the transfer helpers allow one that stalls and a reply's margin: what a client of the extension waits on
#: each. Children end beside their parent.
_FRAME_S = LOAD_WAIT_S + FRAME_CAPTURE_S + TRANSFER_STALL_BUDGET_SECONDS
_REPLY_S = 30.0
START_S = FIRST_FRAME_S + _FRAME_S + _REPLY_S
END_S = 3 * _FRAME_S + LOG_WAIT_S + 4 * TRANSFER_STALL_BUDGET_SECONDS + _REPLY_S
#: Who makes a lease an agent doesn't: the task, or nobody the relay saw.
_NOT_AGENTS = ("task", "unknown")


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


class FollowedTab(BaseModel):
    """What the video shows from ``since``: the tab ``tab`` by screencast or by screenshots, or a blank frame while the
    context has no tab."""

    tab: Optional[str] = None
    since: str
    via: Literal["screencast", "screenshots", "blank"]


class RecordedVideo(BaseModel):
    """The video: its segments, each path under the recording's folder, and which tab it showed when."""

    format: str = MP4
    fps: Optional[int] = None
    segments: list[ManifestSegment] = []
    followed: list[FollowedTab] = []
    ended_by: Optional[str] = None
    error: Optional[str] = None

    @property
    def missing_segments(self) -> int:
        return sum(1 for segment in self.segments if not segment.uploaded)


class RecordedNetwork(BaseModel):
    """The browser's network log as HAR at the recording's end, every context's requests in it: ``during`` the first
    and last of its lines written while the recording ran, and ``contexts_open`` the leases open meanwhile, beside the
    default context."""

    file: str = HAR_FILE
    scope: Literal["browser"] = "browser"
    uploaded: bool = False
    entries: Optional[int] = None
    unreadable: int = 0
    during: Optional[list[int]] = None
    contexts_open: list[str] = []
    error: Optional[str] = None


class ChildRecording(BaseModel):
    """A context an agent made while the recording ran, recorded with it under ``folder``, and what's missing of it
    once it ended, None for nothing."""

    context: str
    made_by: str
    folder: str
    started_at: str
    ended_at: Optional[str] = None
    ended_by: Optional[EndedBy] = None
    missing: Optional[str] = None


class BrowserRecordingManifest(BaseModel):
    """A recording as it ended: of the lease ``context``, None for the default context; its video, its frames, and its
    log's lines of its context, entries ``log_after`` + 1 to ``log_last``, as ``log``, with any entry between them it
    doesn't hold, still unwritten or unreadable at its end, in ``log_missing``; the network log; and the contexts an
    agent made that were recorded with it."""

    recording_id: str
    started_at: str
    ended_at: Optional[str] = None
    ended_by: Optional[EndedBy] = None
    error: Optional[str] = None
    context: Optional[str] = None
    video: Optional[RecordedVideo] = None
    frames_on: bool = True
    frames: list[RecordedFrame] = []
    log: Optional[str] = None
    log_after: int = 0
    log_last: int = 0
    log_uploaded: bool = False
    log_missing: list[int] = []
    network: Optional[RecordedNetwork] = None
    children: list[ChildRecording] = []

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
    """The recordings of the browser that ``log`` and ``control`` are, following its log, its tabs and the leases of
    ``contexts``; ``network``, when it keeps one, its network log; ``window`` the size of a blank frame."""

    def __init__(self, log: BrowserLog, control: BrowserControl, contexts: Optional[Contexts] = None,
                 network: Optional[NetworkLog] = None, window: tuple[int, int] = BLANK_SIZE) -> None:
        self.log, self.control, self.contexts, self.network, self.window = log, control, contexts, network, window
        self._open: dict[str, _Recording] = {}
        self._by_request: dict[str, str] = {}
        self._ended: dict[str, dict] = {}
        self._tasks: set[asyncio.Future] = set()
        #: The context of every tab the browser opened since this server started, kept past a reset.
        self.tab_contexts: dict[str, Optional[str]] = {}
        log.watchers.append(self.acted)
        control.on_tab.append(self.tab_event)
        if contexts is not None:
            contexts.on_made.append(self.context_made)
            contexts.on_end.append(self.context_ended)

    async def start(self, request_id: str, upload: dict, frames: bool = False, context: Optional[str] = None,
                    fps: Optional[int] = None, network: bool = True, max_edge: Optional[int] = None) -> dict:
        """A recording of the lease ``context``, else of the default context, under the grant ``upload``, or the one a
        start with the same ``request_id`` began, which a retried start joins: its ``recording_id``, ``started_at``
        and video ``format``."""
        if (known := self._by_request.get(request_id)) in self._open:
            recording = self._open[known]
            return {"recording_id": known, "started_at": recording.manifest.started_at, "format": MP4}
        if Video is None:
            raise Unsupported(NO_VIDEO)
        fps = DEFAULT_FPS if fps is None else fps
        if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= MAX_FPS:
            raise InvalidArgument(f"fps is an integer from 1 to {MAX_FPS}, got {fps!r}")
        if max_edge is not None and (isinstance(max_edge, bool) or not isinstance(max_edge, int)
                                     or not MIN_EDGE <= max_edge <= MAX_EDGE):
            raise InvalidArgument(f"max_edge is an integer from {MIN_EDGE} to {MAX_EDGE}, got {max_edge!r}")
        if network and self.network is None:
            raise Unsupported("this browser keeps no network log, so a recording of it has no network.har: start it "
                              "with network false")
        if context is not None and self.contexts is None:
            raise TargetUnavailable(f"no context {context!r} is leased: this browser leases none")
        context_id = self.contexts.id_of(context) if context is not None else None
        if len(self._open) >= MAX_OPEN:
            raise LimitExceeded(f"{MAX_OPEN} recordings are open, the most this browser records at once; end one "
                                "first")
        recording_id = uuid.uuid4().hex
        recording = _Recording(self, _grant(upload), recording_id, recording_id, context, context_id, frames, fps,
                               network, max_edge)
        self._open[recording_id] = recording
        self._by_request[request_id] = recording_id
        try:
            await recording.begin()
        except BaseException:
            self._open.pop(recording_id, None)
            await recording.abandon()
            raise
        return {"recording_id": recording_id, "started_at": recording.manifest.started_at, "format": MP4}

    async def end(self, recording_id: str, upload: Optional[dict] = None, reason: str = "end") -> dict:
        """The recording ended, its last frames taken, its video sealed and its log, HAR and manifest uploaded, under
        ``upload`` when one is given, else under its start's grant: its manifest. One already ended answers its
        manifest again."""
        if reason not in REASONS:
            raise InvalidArgument(f"reason is one of {', '.join(REASONS)}, not {reason!r}")
        return await self._end(recording_id, _grant(upload) if upload is not None else None, reason)

    async def end_all(self, reason: str = "teardown") -> list[dict]:
        """Every open recording ended, as its env closes or resets."""
        return [await self.end(recording_id, None, reason) for recording_id in list(self._open)]

    async def _end(self, recording_id: str, grant: Optional[WriteNamespaceGrant], reason: str) -> dict:
        if recording_id in self._ended:
            return self._ended[recording_id]
        recording = self._open.get(recording_id)
        if recording is None:
            raise InvalidArgument(f"no recording {recording_id} is open here")
        manifest = await recording.end(grant, reason)
        self._open.pop(recording_id, None)
        self._ended[recording_id] = manifest
        while len(self._ended) > KEEP_ENDED:
            self._ended.pop(next(iter(self._ended)))
        return manifest

    def every(self) -> list[_Recording]:
        """Every recording still running, children among them."""
        found, waiting = [], list(self._open.values())
        while waiting:
            recording = waiting.pop()
            if not recording.ending:
                found.append(recording)
                waiting += recording.children.values()
        return found

    def acted(self, seq: int, action: dict) -> None:
        """An action written to the log: an agent's acting one on a tab starts, or holds back, the frame of its burst,
        and one that shows it at work there, an MCP tool call or such a command, moves the video of that tab's context
        there."""
        if action.get("acts") and action.get("via") in AGENTS:
            method = action.get("method") or ""
            moves = action.get("via") == "mcp" or method.startswith("Input.") or method in FOLLOWED
            for recording in self.every():
                recording.acted(seq, action.get("tab"), moves)

    def tab_event(self, event: str, tab: dict) -> None:
        """A tab opened or closed: its context kept, and every recording told."""
        if isinstance(target := tab.get("id"), str):
            self.tab_contexts[target] = tab.get("context_id")
        for recording in self.every():
            recording.stream.tab_event(event, tab)

    def context_of_tab(self, target: object) -> Optional[str]:
        """The context of the tab ``target``, as it opened in, or as the browser's control has it."""
        if isinstance(target, str) and target in self.tab_contexts:
            return self.tab_contexts[target]
        return self.control.context_of_tab(target)

    def context_made(self, lease: Lease, scope: Optional[str]) -> None:
        """A lease made: open alongside every open recording; and one an agent made over an address scoped to
        ``scope``, a lease's id or None for the default context, recorded with each recording of that context."""
        from_agent = lease.made_by not in _NOT_AGENTS
        for recording in self.every():
            if recording.network_on and lease.label not in recording.contexts_open:
                recording.contexts_open.append(lease.label)
            if from_agent and recording.context_id == scope:
                recording.made(lease)

    def context_ended(self, lease: Lease) -> None:
        """A lease ended: the recordings of its context end with it."""
        for recording in self.every():
            if recording.context_id == lease.id:
                self._spawn(self._context_ended(recording))

    async def _context_ended(self, recording: _Recording) -> None:
        await self._stopped(recording, "context_ended")

    async def _stopped(self, recording: _Recording, reason: str) -> None:
        """``recording`` ended for ``reason``, not by its end: its context's, or its video's limit or failure."""
        if recording.parent is None:
            await self._end(recording.manifest.recording_id, None, reason)
        else:
            await recording.end(None, reason)

    def _spawn(self, work) -> None:
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Future) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.warning(f"a browser recording: {type(error).__name__}: {error}")


class _Recording:
    """One recording of the context ``context_id``, the lease ``context``'s or the default one's, uploading under
    ``prefix``: a recording's id, or for a child its parent's prefix and ``contexts/<label>``."""

    def __init__(self, recordings: BrowserRecordings, grant: WriteNamespaceGrant, recording_id: str, prefix: str,
                 context: Optional[str], context_id: Optional[str], frames: bool, fps: int, network: bool,
                 max_edge: Optional[int], parent: Optional[_Recording] = None) -> None:
        self.recordings, self.grant, self.uploader = recordings, grant, NamespaceUploader(grant)
        self.prefix, self.context, self.context_id, self.parent = prefix, context, context_id, parent
        self.frames_on, self.fps, self.network_on, self.max_edge = frames, fps, network, max_edge
        self.manifest = BrowserRecordingManifest(recording_id=recording_id, started_at=_iso(), frames_on=frames,
                                                 context=context, log_after=recordings.log.next_seq - 1)
        self.stream = TabStream(recordings.control, context_id, recordings.window)
        root = Path.home() / RECORDINGS_DIR
        hints = RecordHints(scope="tab", fps=fps, max_edge=max_edge, cursor=False, max_segment_s=SEGMENT_S,
                            directory=root / f"browser-{recording_id}")
        features = Recording(max_fps=MAX_FPS, max_concurrent=MAX_OPEN, max_duration_s=MAX_DURATION_S,
                             max_segment_s=SEGMENT_S)
        self.video = Video(f"browser-{recording_id}", hints, features, root, grant, self.stream.frames, streams=True,
                           cursor=False, prefix=f"{prefix}/{VIDEO}", manifest=False)
        self.children: dict[str, _Recording] = {}
        self.contexts_open: list[str] = [lease.label for lease in recordings.contexts or ()] if network else []
        self.network_after = 0
        self.ending = False
        #: The acting entries since the last frame, and the tab the latest was on, which a frame is of.
        self.pending: list[int] = []
        self.acted_tab: Optional[str] = None
        self._timer: Optional[asyncio.TimerHandle] = None
        self._taking: set[asyncio.Future] = set()
        self._beginning: Optional[asyncio.Future] = None
        self._lock = asyncio.Lock()

    async def begin(self) -> None:
        """Recording from now: the context's active tab followed, the video started and, with frames, the first."""
        await self.stream.start()
        self.video.hints.directory.mkdir(parents=True, exist_ok=True)
        if self.network_on:
            self.network_after = await asyncio.to_thread(self.recordings.network.last_seq)
        self.video.begin()
        self.video.task.add_done_callback(self._video_stopped)
        await self.video.until_live()
        if self.frames_on:
            await self.take("first")

    def _video_stopped(self, _: asyncio.Future) -> None:
        """Its video stopped on its own, at its limit or after failing, which ends the recording: one still running
        can't take frames or children without it."""
        if not self.ending:
            ended_by = "limit" if self.video.ended_by == "limit" else "error"
            self.recordings._spawn(self.recordings._stopped(self, ended_by))

    async def abandon(self) -> None:
        """A recording whose start failed: nothing of it left running."""
        self.ending = True
        with contextlib.suppress(Exception):
            await self.video.end("end", None)

    def acted(self, seq: int, tab: Optional[str], moves: bool = True) -> None:
        if self.ending or not self.stream.sees(tab):
            return
        if moves:
            self.stream.follow(tab)
        if not self.frames_on:
            return
        self.pending.append(seq)
        self.acted_tab = tab
        if self._timer is not None:
            self._timer.cancel()
        self._timer = asyncio.get_running_loop().call_later(SETTLE_S, self._settled)

    def made(self, lease: Lease) -> None:
        """A context ``lease`` an agent made over an address scoped to this one's, recorded with it from now."""
        if self.ending or lease.label in self.children or len(self.children) >= MAX_CHILDREN:
            return
        folder = f"{CONTEXTS}/{lease.label}"
        child = _Recording(self.recordings, self.grant, uuid.uuid4().hex, f"{self.prefix}/{folder}", lease.label,
                           lease.id, self.frames_on, self.fps, False, self.max_edge, parent=self)
        self.children[lease.label] = child
        self.manifest.children.append(ChildRecording(context=lease.label, made_by=lease.made_by, folder=f"{folder}/",
                                                     started_at=child.manifest.started_at))
        child._beginning = asyncio.ensure_future(child.begin())
        child._beginning.add_done_callback(lambda done: done.cancelled() or done.exception())

    def _settled(self) -> None:
        self._timer = None
        taking = asyncio.ensure_future(self.take("action"))
        self._taking.add(taking)
        taking.add_done_callback(self._taking.discard)

    async def take(self, kind: str) -> None:
        """The frame of ``kind``, of the tab its context's agent last acted on, else the tab the video follows, uploaded
        as it's taken; one an action burst brings only once that burst has entries no frame followed yet."""
        async with self._lock:
            after, self.pending = self.pending, []
            if kind == "action" and not after:
                return
            tab = self.acted_tab if self.acted_tab in self.recordings.control.tabs else self.stream.tab
            if tab is None and not any(self.stream.sees(open_tab) for open_tab in self.recordings.control.tabs):
                return
            index = len(self.manifest.frames)
            frame = RecordedFrame(index=index, kind=kind, at=_iso(), after=after, tab=tab)
            self.manifest.frames.append(frame)
            if index >= MAX_FRAMES:
                frame.error = f"the recording reached its {MAX_FRAMES} frames"
                return
            try:
                data, extension, frame.tab, frame.url = await self.recordings.control.frame(
                    tab, LOAD_WAIT_S, self.context_id, within=True)
                frame.at = _iso()
                frame.path, frame.content_type = frame_path(index, extension), CONTENT_TYPES[extension]
                await self.uploader.upload(self._path(frame.path), base64.b64decode(data))
                frame.uploaded = True
            except Exception as e:
                frame.error = f"{type(e).__name__}: {e}"[:ERROR_CHARS]
                logger.warning("browser recording %s: frame %d: %s", self.manifest.recording_id, index, frame.error)

    async def end(self, grant: Optional[WriteNamespaceGrant], reason: str) -> dict:
        """Ended for ``reason``, its children with it: its last frames, its video sealed, its log, HAR and manifest
        uploaded, under ``grant`` when one is given; its manifest."""
        if self.manifest.ended_at is not None:
            return self.manifest.model_dump(mode="json")
        self.ending = True
        if self._beginning is not None:
            await asyncio.gather(self._beginning, return_exceptions=True)
            self._beginning = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        await asyncio.gather(*self._taking, return_exceptions=True)
        if grant is not None:
            self.grant, self.uploader = grant, NamespaceUploader(grant)
        await asyncio.gather(*(child.end(grant, reason) for child in self.children.values()), return_exceptions=True)
        if self.frames_on and reason != "context_ended":  # a context that's gone has no tab left to frame
            await self.take("action")
            await self.take("last")
        manifest = self.manifest
        video = await self.video.end(reason if reason in REASONS else "end", grant)
        await self.stream.stop()
        manifest.video = RecordedVideo(fps=self.fps, error=video.get("error"),
                                       followed=[FollowedTab(**seen) for seen in self.stream.followed],
                                       segments=[ManifestSegment(**{**segment, "path": self._within(segment["path"])})
                                                 for segment in video.get("segments", [])])
        if video.get("ended_by") in ("limit", "error") and reason in REASONS:
            reason = video["ended_by"]
        manifest.video.ended_by = video.get("ended_by")
        manifest.log_last = self.recordings.log.next_seq - 1
        lines = await self._log_lines()
        manifest.log_missing = sorted(set(range(manifest.log_after + 1, manifest.log_last + 1))
                                      - {line.get("seq") for line in lines})
        text = "".join(json.dumps(line, default=str) + "\n" for line in self._own(lines))
        manifest.log = LOG
        try:
            await self.uploader.upload(self._path(LOG), text.encode())
            manifest.log_uploaded = True
        except Exception as e:
            manifest.error = f"the log didn't upload: {type(e).__name__}: {e}"[:ERROR_CHARS]
        if self.network_on:
            manifest.network = await self._network()
        manifest.ended_at, manifest.ended_by = _iso(), reason
        wire = manifest.model_dump(mode="json")
        uploaded = True
        try:
            await self.uploader.upload(self._path(MANIFEST), json.dumps(wire, indent=1).encode())
        except Exception as e:
            uploaded = False
            logger.warning("browser recording %s: its manifest didn't upload: %s", manifest.recording_id, e)
        if self.parent is not None and (entry := next((child for child in self.parent.manifest.children
                                                        if child.context == self.context), None)) is not None:
            entry.ended_at, entry.ended_by = manifest.ended_at, reason
            entry.missing = _missing(manifest) or (None if uploaded else "its manifest")
        return wire

    async def _network(self) -> RecordedNetwork:
        """The browser's network log as HAR, uploaded: every line so far, so the start tabs' loads are in it, and
        which of them it wrote while this recording ran."""
        network = RecordedNetwork(contexts_open=list(self.contexts_open))
        try:
            lines = await asyncio.to_thread(self.recordings.network.every_line)
            document = har(lines)
            network.entries = len(document["log"]["entries"])
            network.unreadable = document["log"].get("_unreadable", 0)
            last = max((line["seq"] for line in lines), default=0)
            network.during = [self.network_after + 1, last] if last > self.network_after else None
            await self.uploader.upload(self._path(HAR_FILE), json.dumps(document, indent=1).encode())
            network.uploaded = True
        except Exception as e:
            network.error = f"{type(e).__name__}: {e}"[:ERROR_CHARS]
            logger.warning("browser recording %s: its network log: %s", self.manifest.recording_id, network.error)
        return network

    def _own(self, lines: list[dict]) -> list[dict]:
        """The lines of its context: actions and tab events on its tabs, actions with no tab from a connection
        scoped to it, or for the default context from one scoped to none; an outcome of an action it keeps; and a
        reset, which every recording's log has."""
        kept, seqs = [], set()
        for line in lines:
            if (outcome := line.get("outcome")) is not None:
                own = isinstance(outcome, dict) and outcome.get("entry") in seqs
            elif isinstance(action := line.get("action"), dict):
                own = self._ours(action["tab"]) if action.get("tab") else action.get("context") == self.context
            elif isinstance(tab := line.get("tab"), dict):
                own = self._ours(tab.get("id"))
            else:
                own = True
            if own:
                kept.append(line)
                seqs.add(line.get("seq"))
        return kept

    def _ours(self, target: object) -> bool:
        return self.recordings.control.in_context(self.recordings.context_of_tab(target), self.context_id)

    async def _log_lines(self) -> list[dict]:
        """The log's lines from the recording's start to its end, once every one of them is written; after
        ``LOG_WAIT_S``, an action still going is written as unfinished, as it was at the end, and what's written of
        them is what there is."""
        manifest, log = self.manifest, self.recordings.log
        deadline = asyncio.get_running_loop().time() + LOG_WAIT_S
        wanted = manifest.log_last - manifest.log_after
        while True:
            late = asyncio.get_running_loop().time() >= deadline
            read = await log.read_async(manifest.log_after, limit=wanted,
                                        settle_through=manifest.log_last if late else None)
            lines = [line for line in read["lines"] if line.get("seq", 0) <= manifest.log_last]
            if len(lines) >= wanted or late:
                return lines
            await asyncio.sleep(LOG_POLL_S)

    def _path(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def _within(self, path: str) -> str:
        """A path under the grant's root as one under this recording's folder."""
        return path[len(self.prefix) + 1:] if path.startswith(f"{self.prefix}/") else path


def _missing(manifest: BrowserRecordingManifest) -> Optional[str]:
    """What's missing of a recording that ended, None for nothing."""
    video = manifest.video or RecordedVideo()
    if not manifest.log_uploaded or manifest.log_missing:
        return "its log, or entries of it"
    if not video.segments or video.missing_segments or video.ended_by in ("limit", "error"):
        return "its video, or segments of it"
    if manifest.frames_on and manifest.missing_frames:
        return f"{manifest.missing_frames} frames"
    return None


def _grant(upload: object) -> WriteNamespaceGrant:
    try:
        return WriteNamespaceGrant.model_validate(upload)
    except ValidationError as e:
        raise InvalidArgument(f"upload must be a write namespace grant: {e.errors()[0]['msg']}") from None


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
