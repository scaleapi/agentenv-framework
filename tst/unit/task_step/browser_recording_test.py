"""start_browser_recording and end_browser_recording against a browser computer standing in for its browser: the grant
it's handed through the browser's own card, the artifact its frames, log and manifest become, a recording whose frames
or log are missing failing as an infrastructure failure, what's registered when the browser forgot the recording, and
what a run left open ended at teardown."""

from __future__ import annotations

import json

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


class _BrowserComputer:
    """A browser computer whose chromium records: its end uploads a frame, the log and the manifest under its grant,
    with a frame it couldn't take when ``misses``; it ``forgets`` the recording or lists no browser recording."""

    def __init__(self, store, misses: bool = False, forgets: bool = False, recording: bool = True,
                 gaps: bool = False) -> None:
        children = [_BROWSER if recording else {**_BROWSER, "capabilities": {"extensions": [
            _BROWSER["capabilities"]["extensions"][0]]}}]
        self.env_id, self.sandbox_type, self.metadata = "browser", "local", None
        self.environment_card = {"name": "browser", "children_environments": children}
        self.store, self.misses, self.forgets, self.gaps = store, misses, forgets, gaps
        self.calls: list[tuple[str, dict, str]] = []
        self.timeouts: list[float] = []

    async def invoke(self, uri, method, params=None, environment_name=None, timeout=None, **_):
        assert uri == BROWSER_RECORDING
        self.calls.append((method, params, environment_name))
        self.timeouts.append(timeout)
        if method == "start":
            self.root = params["upload"]["root_path"]
            self.store.put(f"{self.root}/rec-1/frames/0000.jpg", b"jpeg-0", content_type="image/jpeg")
            return {"recording_id": "rec-1", "started_at": "2026-11-12T10:00:00.000Z"}
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
    assert (start["request_id"], start["frames"], browser) == ("run-1:start-run", True, "chromium")
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
    context = await _start().execute(_context(_BrowserComputer(store, misses=True)))
    with pytest.raises(RuntimeError, match="infrastructure failure: 1 of its 2 frames are missing"):
        await _end().execute(context)

    tolerated = await _start().execute(_context(_BrowserComputer(store, misses=True), instance_id="run-2"))
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


def test_the_steps_round_trip():
    start = _start(frames=False, browser="chromium")
    end = _end()

    assert StartBrowserRecordingTaskStep.from_dict(start.to_dict()).to_dict() == start.to_dict()
    assert EndBrowserRecordingTaskStep.from_dict(end.to_dict()).to_dict() == end.to_dict()
    assert start.to_dict()["frames"] is False and start.fail_task_on_error and end.fail_task_on_error
    with pytest.raises(ValueError):
        StartBrowserRecordingTaskStep(id="s", version=None, env_id="browser", name="")
