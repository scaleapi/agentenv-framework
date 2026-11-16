"""start_browser_recording and end_browser_recording against a browser computer standing in for its browser: the grant
it's handed through the browser's own card, a lease's recording asked for with its context and frame rate, the artifact
its video, network log, frames, log, manifest and children become, a recording missing any of them failing as an
infrastructure failure, an older browser recording no context refused, what's registered when the browser forgot the
recording, and what a run left open ended at teardown."""

from __future__ import annotations

import json
from typing import Optional

import pytest
from agentenv_protocol.computer import COMPUTER_RECORDING
from agentenv_protocol.website_browser import BROWSER, BROWSER_RECORDING
from agentenv_protocol.website_browser.recording import END_S, START_S

from agent_env.artifact import Artifact
from agent_env.config import set_object_store
from agent_env.env.env import EnvCapabilityUnsupported
from agent_env.providers.env_providers.env_computer_provider import end_recordings
from agent_env.task import teardown
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.browser.recording import (
    RECORDINGS_KEY,
    EndBrowserRecordingTaskStep,
    StartBrowserRecordingTaskStep,
    end_open_browser_recordings,
)
from tst.util.granting_object_store import GrantingObjectStore

_BROWSER = {"name": "chromium", "capabilities": {"extensions": [
    {"uri": BROWSER, "params": {"chromium": "156.0.8078.4", "methods": {}}},
    {"uri": BROWSER_RECORDING, "params": {"max_duration_s": 3600, "max_frames": 100, "frames": True,
                                          "methods": {"start": {"method": "POST"}, "end": {"method": "POST"}}}}]}}
#: A browser that records each context: its video, its network log and the contexts agents make.
_START_FIELDS = ("request_id", "upload", "frames", "context", "fps", "network", "max_edge")
_CONTEXTS_BROWSER = {"name": "chromium", "capabilities": {"extensions": [
    {"uri": BROWSER, "params": {"chromium": "156.0.8078.4", "methods": {}}},
    {"uri": BROWSER_RECORDING, "params": {
        "max_duration_s": 3600, "max_frames": 100, "frames": True, "video": True, "context": True, "network": True,
        "max_objects": 9999, "methods": {"start": {"method": "POST", "request": {"properties": {
            field: {} for field in _START_FIELDS}}}, "end": {"method": "POST"}}}}]}}


class _BrowserComputer:
    """A browser computer whose chromium records: its end uploads a frame, the log and the manifest under its grant,
    with a frame it couldn't take when ``misses``; it ``forgets`` the recording or lists no browser recording."""

    def __init__(self, store, misses: bool = False, forgets: bool = False, recording: bool = True,
                 gaps: bool = False, contexts: bool = False, video: Optional[dict] = None,
                 network: Optional[dict] = None, children: Optional[list] = None) -> None:
        browser = _CONTEXTS_BROWSER if contexts else _BROWSER
        children_envs = [browser if recording else {**browser, "capabilities": {"extensions": [
            browser["capabilities"]["extensions"][0]]}}]
        self.env_id, self.sandbox_type, self.metadata = "browser", "local", None
        self.environment_card = {"name": "browser", "children_environments": children_envs}
        self.store, self.misses, self.forgets, self.gaps = store, misses, forgets, gaps
        self.video, self.network, self.children = video, network, children
        self.calls: list[tuple[str, dict, str]] = []
        self.timeouts: list[float] = []

    async def invoke(self, uri, method, params=None, environment_name=None, timeout=None, **_):
        assert uri == BROWSER_RECORDING
        self.calls.append((method, params, environment_name))
        self.timeouts.append(timeout)
        if method == "start":
            self.root = params["upload"]["root_path"]
            self.store.put(f"{self.root}/rec-1/frames/0000.jpg", b"jpeg-0", content_type="image/jpeg")
            return {"recording_id": "rec-1", "started_at": "2026-11-12T10:00:00.000Z",
                    **({"network": params.get("network", True)} if self.network is not None else {})}
        if self.forgets:
            raise RuntimeError("400 invalid_argument: no recording rec-1 is open here")
        root = params["upload"]["root_path"]
        frames = [{"index": 0, "kind": "first", "at": "t0", "path": "frames/0000.jpg", "content_type": "image/jpeg",
                   "uploaded": True},
                  {"index": 1, "kind": "last", "at": "t1", "uploaded": not self.misses,
                   **({"error": "DeviceUnavailable: the tab painted nothing within 3 s"} if self.misses
                      else {"path": "frames/0001.jpg", "content_type": "image/jpeg"})}]
        if not self.misses:
            self.store.put(f"{root}/rec-1/frames/0001.jpg", b"jpeg-1", content_type="image/jpeg")
        self.store.put(f"{root}/rec-1/log.jsonl", b'{"seq": 1}\n', content_type="application/x-ndjson")
        manifest = {"recording_id": "rec-1", "started_at": "2026-11-12T10:00:00.000Z", "ended_by": params["reason"],
                    "frames": frames, "log": "log.jsonl", "log_after": 0, "log_last": 2, "log_uploaded": True,
                    "log_missing": [2] if self.gaps else []}
        if self.video is not None:
            manifest["video"] = self.video
            for segment in self.video["segments"]:
                if segment["uploaded"]:
                    self.store.put(f"{root}/rec-1/{segment['path']}", b"mp4", content_type="video/mp4")
        if self.network is not None:
            manifest["network"] = self.network
            if self.network["uploaded"]:
                self.store.put(f"{root}/rec-1/network.har", b"{}", content_type="application/json")
        if self.children is not None:
            manifest["children"] = self.children
            for child in self.children:
                for name in ("manifest.json", "log.jsonl", "video/0000.mp4"):
                    self.store.put(f"{root}/rec-1/{child['folder']}{name}", b"x", content_type="application/json")
        self.store.put(f"{root}/rec-1/manifest.json", json.dumps(manifest).encode(), content_type="application/json")
        return manifest


@pytest.fixture
def store(local_stores, tmp_path):
    granting = GrantingObjectStore(str(tmp_path / "objects"))
    set_object_store(granting)
    return granting


def _context(*envs, instance_id: str = "run-1") -> TaskStepContext:
    return TaskStepContext(instance_id=instance_id, deployed_envs=list(envs), metadata={})


def _start(name: str = "run", **kwargs) -> StartBrowserRecordingTaskStep:
    return StartBrowserRecordingTaskStep(id=f"start-{name}", version=None, env_id="browser", name=name, **kwargs)


def _end(name: str = "run", **kwargs) -> EndBrowserRecordingTaskStep:
    return EndBrowserRecordingTaskStep(id=f"end-{name}", version=None, name=name, **kwargs)


@pytest.mark.asyncio
async def test_a_browser_recording_is_started_through_its_browsers_card_and_registered_with_its_frames_and_log(store):
    computer = _BrowserComputer(store)
    context = await _start().execute(_context(computer))
    started = dict(context.metadata[RECORDINGS_KEY]["run"])
    context = await _end().execute(context)
    entry = context.metadata[RECORDINGS_KEY]["run"]

    (_, start, browser), (_, end, _) = computer.calls
    assert (start["request_id"], start["frames"], browser) == ("run-1:start-run", False, "chromium")
    assert start["upload"]["max_objects"] == 102 and "/agentenv-browser-recordings/" in started["namespace"]
    assert computer.timeouts == [START_S, END_S]
    assert end["upload"]["root_path"] == start["upload"]["root_path"] and end["reason"] == "end"
    assert (entry["state"], entry["frames_taken"], entry["missing_frames"], entry["log_uploaded"]) == \
        ("ended", 2, 0, True)
    universe = Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"])
    assert sorted(universe.file_artifact_ids) == ["frames-0000.jpg", "frames-0001.jpg", "log.jsonl", "manifest.json"]
    frame = Artifact.get(universe.file_artifact_ids["frames-0001.jpg"])
    assert frame.object_url.endswith("/rec-1/frames/0001.jpg") and store.get(frame.object_url) == b"jpeg-1"


@pytest.mark.asyncio
async def test_a_missing_frame_fails_the_end_as_an_infrastructure_failure_unless_told_not_to(store):
    context = await _start(frames=True).execute(_context(_BrowserComputer(store, misses=True)))
    with pytest.raises(RuntimeError, match="infrastructure failure: 1 of its 2 frames are missing"):
        await _end().execute(context)

    tolerated = await _start(frames=True).execute(_context(_BrowserComputer(store, misses=True),
                                                            instance_id="run-2"))
    tolerated = await _end(fail_task_on_error=False).execute(tolerated)
    assert tolerated.metadata[RECORDINGS_KEY]["run"]["missing_frames"] == 1


@pytest.mark.asyncio
async def test_a_log_missing_entries_fails_the_end(store):
    context = await _start().execute(_context(_BrowserComputer(store, gaps=True)))
    with pytest.raises(RuntimeError, match="its log is missing 1 of the entries"):
        await _end().execute(context)


@pytest.mark.asyncio
async def test_a_computer_ends_its_browsers_recordings_before_it_goes():
    calls = []

    class Record:
        env_id = "browser"
        environment_card = {"name": "browser", "children_environments": [{"name": "page"}, _BROWSER]}

        async def invoke(self, uri, method, params=None, environment_name=None, timeout=None):
            calls.append((uri, method, params, environment_name))
            if uri == COMPUTER_RECORDING:
                raise EnvCapabilityUnsupported(uri, method, "browser", None, None)
            return {"recordings": []}

    await end_recordings(Record(), "reset")

    assert calls[1:] == [(BROWSER_RECORDING, "end_all", {"reason": "reset"}, "chromium")]


@pytest.mark.asyncio
async def test_a_recording_the_browser_forgot_is_registered_from_its_namespace_and_fails_for_its_missing_log(store):
    computer = _BrowserComputer(store, forgets=True)
    context = await _start().execute(_context(computer))
    store.put(f"{computer.root}/rec-1/frames/0003.png", b"png-3", content_type="image/png")
    with pytest.raises(RuntimeError, match="its log didn't upload"):
        await _end().execute(context)
    entry = context.metadata[RECORDINGS_KEY]["run"]
    universe = Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"])
    manifest = json.loads(store.get(Artifact.get(universe.file_artifact_ids["manifest.json"]).object_url))

    assert (entry["state"], entry["frames_taken"], entry["log_uploaded"]) == ("ended", 2, False)
    assert "no recording rec-1" in entry["error"]
    assert [(f["index"], f["kind"], f["at"], f["content_type"]) for f in manifest["frames"]] == [
        (0, None, None, "image/jpeg"), (3, None, None, "image/png")]


@pytest.mark.asyncio
async def test_a_browser_without_recording_fails_the_start_saying_so(store):
    computer = _BrowserComputer(store, recording=False)
    with pytest.raises(Exception, match="browser-recording/v1"):
        await _start().execute(_context(computer))

    context = await _start(fail_task_on_error=False).execute(_context(computer))
    assert context.metadata[RECORDINGS_KEY]["run"]["state"] == "failed" and computer.calls == []


@pytest.mark.asyncio
async def test_what_a_run_left_open_is_ended_with_a_fresh_grant_and_teardown_ends_it_before_envs_close(store,
                                                                                                    monkeypatch):
    computer = _BrowserComputer(store)
    context = await _start().execute(_context(computer))
    await end_open_browser_recordings(context, ["browser"])
    ended = []

    async def ending(context, env_ids):
        ended.append(env_ids)

    monkeypatch.setattr(teardown, "end_open_browser_recordings", ending)
    await teardown.tear_down(context, [], [], timeout=30)

    (_, start, _), (_, end, _) = computer.calls
    entry = context.metadata[RECORDINGS_KEY]["run"]
    assert end["reason"] == "teardown" and end["upload"] != start["upload"] and entry["state"] == "ended"
    assert ended == [[]]


@pytest.mark.asyncio
async def test_a_start_reusing_an_open_recordings_name_is_refused():
    context = TaskStepContext(instance_id="run-1", metadata={RECORDINGS_KEY: {"run": {"state": "open"}}})
    with pytest.raises(RuntimeError, match="already open"):
        await _start().execute(context)


def test_the_steps_round_trip_writing_a_lease_frame_rate_or_no_network_only_when_asked_for():
    start = _start(frames=True, browser="chromium")
    leased = _start(context="buyer", fps=5, network=False)
    end = _end()

    for step in (start, leased):
        assert StartBrowserRecordingTaskStep.from_dict(step.to_dict()).to_dict() == step.to_dict()
    assert EndBrowserRecordingTaskStep.from_dict(end.to_dict()).to_dict() == end.to_dict()
    assert start.to_dict()["frames"] is True and _start().to_dict()["frames"] is False
    assert not {"context", "fps", "network"} & set(start.to_dict())
    assert {key: leased.to_dict()[key] for key in ("context", "fps", "network")} == {
        "context": "buyer", "fps": 5, "network": False}
    assert start.fail_task_on_error and end.fail_task_on_error
    with pytest.raises(ValueError):
        StartBrowserRecordingTaskStep(id="s", version=None, env_id="browser", name="")
    with pytest.raises(ValueError, match="fps is an integer from 1 to 15"):
        _start(fps=30)


def _video(*uploaded: bool) -> dict:
    return {"format": "video/mp4; codecs=avc1", "fps": 10, "followed": [], "segments": [
        {"index": index, "path": f"video/{index:04d}.mp4", "uploaded": done} for index, done in enumerate(uploaded)]}


def _network(uploaded: bool = True, entries: int = 3) -> dict:
    return {"file": "network.har", "scope": "browser", "uploaded": uploaded, "entries": entries, "during": [4, 9],
            "contexts_open": ["buyer"]}


def _child(missing: Optional[str] = None) -> dict:
    return {"context": "agent-bob-1", "made_by": "bob", "folder": "contexts/agent-bob-1/",
            "started_at": "2026-11-12T10:00:01.000Z", "ended_at": "2026-11-12T10:00:09.000Z",
            "ended_by": "context_ended", "missing": missing}


@pytest.mark.asyncio
async def test_a_leases_recording_is_asked_for_with_its_context_and_registers_its_video_network_log_and_children(
        store):
    computer = _BrowserComputer(store, contexts=True, video=_video(True, True), network=_network(),
                                children=[_child()])
    context = await _start(context="buyer", fps=5).execute(_context(computer))
    context = await _end().execute(context)
    entry = context.metadata[RECORDINGS_KEY]["run"]

    (_, start, _), _ = computer.calls
    assert (start["context"], start["fps"], "network" in start) == ("buyer", 5, False)
    assert start["upload"]["max_objects"] == 9999
    assert (entry["context"], entry["segments"], entry["har_entries"], entry["children"][0]["context"]) == (
        "buyer", 2, 3, "agent-bob-1")
    universe = Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"])
    assert {"video-0000.mp4", "video-0001.mp4", "network.har", "contexts-agent-bob-1-manifest.json",
            "contexts-agent-bob-1-video-0000.mp4"} <= set(universe.file_artifact_ids)
    assert Artifact.get(universe.file_artifact_ids["video-0001.mp4"]).content_type == "video/mp4"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, why", [
    ("no video", "it has no video"), ("a segment", "1 of its 2 video segments didn't upload"),
    ("no network log", "its network log didn't upload"),
    ("a child", "a context an agent made beside it is incomplete: agent-bob-1: its video"),
    ("a video at its limit", "its video stopped at its limit"), ("a failed video", "its video stopped failing"),
    ("unreadable network lines", "its network log left out 2 lines")])
async def test_a_recording_missing_its_video_network_log_or_a_childs_parts_fails_the_end(store, kind, why):
    video = {"no video": _video(), "a segment": _video(True, False),
             "a video at its limit": {**_video(True), "ended_by": "limit"},
             "a failed video": {**_video(True), "ended_by": "error", "error": "OSError: disk"}}.get(kind, _video(True))
    network = {"no network log": _network(uploaded=False),
               "unreadable network lines": {**_network(), "unreadable": 2}}.get(kind, _network())
    children = [_child("its video, or segments of it")] if kind == "a child" else []
    computer = _BrowserComputer(store, contexts=True, video=video, network=network, children=children)
    context = await _start().execute(_context(computer))

    with pytest.raises(RuntimeError, match=f"infrastructure failure: {why}"):
        await _end().execute(context)


@pytest.mark.asyncio
async def test_a_network_log_with_no_entries_is_no_failure_since_an_agent_may_have_loaded_nothing(store):
    computer = _BrowserComputer(store, contexts=True, video=_video(True), network=_network(entries=0))
    context = await _end().execute(await _start().execute(_context(computer)))

    assert context.metadata[RECORDINGS_KEY]["run"]["har_entries"] == 0


@pytest.mark.asyncio
async def test_a_browser_recording_no_context_is_refused_a_leases_recording_before_it_starts_one(store):
    computer = _BrowserComputer(store)

    with pytest.raises(RuntimeError, match="records no context, fps: rebuild it"):
        await _start(context="buyer", fps=5).execute(_context(computer))
    assert computer.calls == []
