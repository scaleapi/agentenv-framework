"""start_computer_recording and end_computer_recording against a deployed env standing in for a driver: the grant
it's handed, the artifact its uploaded segments and manifest become, what's registered when the driver forgot the
recording, teardown ending what a run left open, and a reset leaving it going."""

from __future__ import annotations

import asyncio

import pytest
from agentenv_protocol.computer.card import COMPUTER_RECORDING

from agent_env.artifact import Artifact
from agent_env.config import set_object_store
from agent_env.store.base import GrantUnavailableError
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import computer_recording as recording_steps
from agent_env.task_step.task_steps.computer_recording import (
    RECORDINGS_KEY,
    EndComputerRecordingTaskStep,
    StartComputerRecordingTaskStep,
)
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from agent_env.task_step.task_steps.teardown_sandboxes import TeardownSandboxesTaskStep
from tst.util.granting_object_store import GrantingObjectStore

_CARD = {"name": "desktop", "capabilities": {"extensions": [{"uri": COMPUTER_RECORDING, "params": {
    "max_duration_s": 3600, "max_segment_s": 300, "max_concurrent": 2,
    "methods": {"start": {"method": "POST"}, "end": {"method": "POST"}}}}]}}


class _Computer:
    """A deployed env whose driver records: each end uploads one segment per started recording under its grant,
    or leaves the last one out when ``drop_last``; it ``forgets`` the recording, ``garbles`` its manifest, or
    ``hangs`` at end."""

    def __init__(self, store, env_id: str = "desktop", forgets: bool = False, garbles: bool = False,
                 hangs: bool = False, drop_last: bool = False) -> None:
        self.env_id, self.environment_card, self.sandbox_type = env_id, _CARD, "local"
        self.store, self.forgets, self.garbles, self.hangs, self.drop_last = store, forgets, garbles, hangs, drop_last
        self.calls: list[tuple[str, dict]] = []
        self.started: set[str] = set()

    async def invoke(self, uri, method, params=None, **_):
        assert uri == COMPUTER_RECORDING
        self.calls.append((method, params))
        if method == "start":
            if params["request_id"] not in self.started:  # a repeated request_id joins the first recording
                self.started.add(params["request_id"])
                self.store.put(f"{params['upload']['root_path']}/rec-1/0000.mp4", b"mp4-0", content_type="video/mp4")
            return {"recording_id": "rec-1", "started_at": "2026-10-27T10:00:00.000Z", "format": "video/mp4"}
        if self.forgets:
            raise RuntimeError("400 invalid_argument: there is no recording 'rec-1' on this computer")
        if self.garbles:
            return {"recording_id": "rec-1", "segments": "all of them"}
        if self.hangs:
            await asyncio.sleep(3600)
        segments = [{"index": 0, "path": "rec-1/0000.mp4", "uploaded": True},
                    {"index": 1, "path": "rec-1/0001.mp4", "uploaded": not self.drop_last}]
        if not self.drop_last:
            self.store.put(f"{params['upload']['root_path']}/rec-1/0001.mp4", b"mp4-1", content_type="video/mp4")
        return {"recording_id": "rec-1", "ended_by": params["reason"], "segments": segments}


@pytest.fixture
def store(local_stores, tmp_path):
    granting = GrantingObjectStore(str(tmp_path / "objects"))
    set_object_store(granting)
    return granting


def _context(*envs) -> TaskStepContext:
    return TaskStepContext(instance_id="run-1", deployed_envs=list(envs), metadata={})


def _start(name: str = "desk", env_id: str = "desktop", **kwargs) -> StartComputerRecordingTaskStep:
    return StartComputerRecordingTaskStep(id=f"start-{name}", version=None, env_id=env_id, name=name, **kwargs)


def _end(name: str = "desk", **kwargs) -> EndComputerRecordingTaskStep:
    return EndComputerRecordingTaskStep(id=f"end-{name}", version=None, name=name, **kwargs)


@pytest.mark.asyncio
async def test_a_recording_is_started_with_a_grant_and_registered_with_its_manifest_at_end(store):
    computer = _Computer(store)
    context = await _start(fps=5).execute(_context(computer))
    started = dict(context.metadata[RECORDINGS_KEY]["desk"])
    context = await _end().execute(context)
    entry = context.metadata[RECORDINGS_KEY]["desk"]

    (_, start), (_, end) = computer.calls
    assert start["request_id"] == "run-1:start-desk" and start["fps"] == 5 and start["cursor"] is True
    assert "/agentenv-recordings/" in started["namespace"] and started["state"] == "open"
    assert end["upload"]["root_path"] == start["upload"]["root_path"] and end["reason"] == "end"
    assert (entry["state"], entry["ended_by"], entry["segments"], entry["missing_segments"]) == ("ended", "end", 2, 0)
    universe = Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"])
    assert sorted(universe.file_artifact_ids) == ["manifest.json", "rec-1-0000.mp4", "rec-1-0001.mp4"]
    segment = Artifact.get(universe.file_artifact_ids["rec-1-0001.mp4"])
    assert segment.object_url.endswith("/rec-1/0001.mp4") and store.get(segment.object_url) == b"mp4-1"
    assert b'"ended_by": "end"' in store.get(Artifact.get(universe.file_artifact_ids["manifest.json"]).object_url)


@pytest.mark.asyncio
async def test_a_segment_the_driver_couldnt_upload_is_left_out_and_counted(store):
    context = await _start().execute(_context(_Computer(store, drop_last=True)))
    entry = (await _end().execute(context)).metadata[RECORDINGS_KEY]["desk"]
    universe = Artifact.get(entry["artifact"]["id"])
    assert (entry["segments"], entry["missing_segments"]) == (1, 1)
    assert sorted(universe.file_artifact_ids) == ["manifest.json", "rec-1-0000.mp4"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault, error", [("forgets", "no recording 'rec-1'"), ("garbles", "ValidationError")])
async def test_a_recording_the_driver_forgot_or_garbled_is_registered_from_its_namespace(store, fault, error):
    context = await _start().execute(_context(_Computer(store, **{fault: True})))
    entry = (await _end().execute(context)).metadata[RECORDINGS_KEY]["desk"]
    assert entry["state"] == "ended" and entry["ended_by"] == "error" and error in entry["error"]
    assert sorted(Artifact.get(entry["artifact"]["id"]).file_artifact_ids) == ["manifest.json", "rec-1-0000.mp4"]


@pytest.mark.asyncio
async def test_a_store_whose_grants_dont_reach_the_sandbox_fails_the_start_best_effort_or_fatally(store):
    computer = _Computer(store)
    computer.sandbox_type = "modal"
    context = await _start().execute(_context(computer))
    entry = context.metadata[RECORDINGS_KEY]["desk"]
    assert entry["state"] == "failed" and "reaches env desktop's sandbox (modal)" in entry["error"]
    assert computer.calls == []
    await _end().execute(context)
    assert context.metadata[RECORDINGS_KEY]["desk"]["state"] == "failed"
    with pytest.raises(RuntimeError, match="issues no upload grant"):
        await _start(fail_task_on_error=True).execute(_context(computer))


@pytest.mark.asyncio
async def test_ending_a_recording_never_started_is_logged_or_fatal(store):
    context = await _end().execute(_context(_Computer(store)))
    assert RECORDINGS_KEY not in context.metadata
    with pytest.raises(RuntimeError, match="no recording named 'desk'"):
        await _end(fail_task_on_error=True).execute(context)


@pytest.mark.asyncio
async def test_a_fatal_end_raises_when_the_recording_couldnt_be_registered(store, monkeypatch):
    context = await _start().execute(_context(_Computer(store)))

    def broken(*args):
        raise RuntimeError("the document store is down")

    monkeypatch.setattr(recording_steps, "_register", broken)
    with pytest.raises(RuntimeError, match="didn't end: RuntimeError: the document store is down"):
        await _end(fail_task_on_error=True).execute(context)


@pytest.mark.asyncio
async def test_a_reset_leaves_a_recording_going_and_teardown_ends_what_a_run_left_open(store):
    desktop, other = _Computer(store), _Computer(store, env_id="other")
    context = _context(desktop, other)
    context = await _start("a").execute(context)
    context = await _start("b", env_id="other").execute(context)
    await ResetEnvTaskStep(id="reset", version=None, env_id="other").execute(context)
    recordings = context.metadata[RECORDINGS_KEY]
    assert recordings["b"]["state"] == "open" and [method for method, _ in other.calls] == ["start"]
    await teardown_run(context)
    assert (recordings["a"]["ended_by"], recordings["b"]["ended_by"]) == ("teardown", "teardown")
    assert [method for method, _ in desktop.calls] == ["start", "end"]


@pytest.mark.asyncio
async def test_teardown_goes_on_when_a_recording_wont_end_in_time(store):
    context = await _start().execute(_context(_Computer(store, hangs=True)))
    report = await teardown_run(context, timeout=0.2)
    assert context.metadata[RECORDINGS_KEY]["desk"]["state"] == "open" and not report.left


@pytest.mark.asyncio
async def test_a_teardown_sandboxes_step_ends_the_recordings_of_its_envs_only(store):
    desktop, other = _Computer(store), _Computer(store, env_id="other")
    context = _context(desktop, other)
    context = await _start("a").execute(context)
    context = await _start("b", env_id="other").execute(context)
    await TeardownSandboxesTaskStep(id="down", version=None, env_ids=["desktop"]).execute(context)
    recordings = context.metadata[RECORDINGS_KEY]
    assert (recordings["a"]["state"], recordings["b"]["state"]) == ("ended", "open")


def test_the_steps_round_trip_and_refuse_what_they_cant_run():
    start = _start(fps=5, max_edge=1280, cursor=False, fail_task_on_error=True)
    assert StartComputerRecordingTaskStep.from_dict(start.to_dict()).to_dict() == start.to_dict()
    assert EndComputerRecordingTaskStep.from_dict(_end().to_dict()).to_dict() == _end().to_dict()
    assert StartComputerRecordingTaskStep.from_dict({"id": "s", "env_id": "e", "name": "n"}).fail_task_on_error is False
    with pytest.raises(ValueError, match="needs an env_id and a name"):
        _start(name="")
    with pytest.raises(ValueError, match="needs the name"):
        _end(name="")


class _ShortGrants(GrantingObjectStore):
    """A store whose credentials sign grants only as long as its grant lifetime."""

    def issue_upload_policy(self, prefix_url, *, max_object_bytes, expires_in):
        if expires_in > self.grant_lifetime_seconds:
            raise GrantUnavailableError(f"temporary credentials can't sign {expires_in} s")
        self.asked = expires_in
        return super().issue_upload_policy(prefix_url, max_object_bytes=max_object_bytes, expires_in=expires_in)


@pytest.mark.asyncio
async def test_a_store_that_cant_sign_for_the_whole_recording_grants_what_it_can(local_stores, tmp_path):
    short = _ShortGrants(str(tmp_path / "objects"))
    short.grant_lifetime_seconds = 1800
    set_object_store(short)
    context = await _start().execute(_context(_Computer(short)))
    assert context.metadata[RECORDINGS_KEY]["desk"]["state"] == "open" and short.asked == short.grant_lifetime_seconds


@pytest.mark.asyncio
async def test_recording_an_env_this_run_didnt_deploy_fails_the_start(store):
    context = await _start(env_id="elsewhere").execute(_context(_Computer(store)))
    assert "env 'elsewhere' isn't deployed in this run" in context.metadata[RECORDINGS_KEY]["desk"]["error"]


@pytest.mark.asyncio
async def test_a_retried_start_joins_the_first_attempts_namespace_and_the_grant_budgets_early_rollovers(store):
    computer = _Computer(store)
    context = await _start().execute(_context(computer))
    first = context.metadata[RECORDINGS_KEY].pop("desk")
    context = await _start().execute(context)
    (_, one), (_, two) = computer.calls
    assert one["request_id"] == two["request_id"] and one["upload"]["root_path"] == two["upload"]["root_path"]
    assert context.metadata[RECORDINGS_KEY]["desk"]["namespace"] == first["namespace"]
    assert one["upload"]["max_objects"] == recording_steps.MAX_SEGMENTS


@pytest.mark.asyncio
async def test_a_start_reusing_an_open_recordings_name_is_refused_and_leaves_it_alone(store):
    computer = _Computer(store)
    context = await _start().execute(_context(computer))
    opened = dict(context.metadata[RECORDINGS_KEY]["desk"])
    context = await _start(env_id="desktop").execute(context)
    assert context.metadata[RECORDINGS_KEY]["desk"] == opened and len(computer.calls) == 1
    with pytest.raises(RuntimeError, match="a recording named 'desk' is already open"):
        await _start(fail_task_on_error=True).execute(context)


@pytest.mark.asyncio
async def test_teardown_goes_on_when_a_forgotten_recordings_namespace_cant_be_listed(store, monkeypatch):
    context = await _start().execute(_context(_Computer(store, forgets=True)))

    def unlisted(prefix):
        raise ConnectionError("the object store is down")

    monkeypatch.setattr(store, "list_at", unlisted)
    await teardown_run(context)
    entry = context.metadata[RECORDINGS_KEY]["desk"]
    assert entry["state"] == "failed" and "then ConnectionError: the object store is down" in entry["error"]


@pytest.mark.asyncio
async def test_a_recording_that_cant_be_ended_at_all_is_marked_and_the_rest_still_end(store, monkeypatch):
    context = _context(_Computer(store), _Computer(store, env_id="other"))
    context = await _start("a").execute(context)
    context = await _start("b", env_id="other").execute(context)
    real = recording_steps.end_recording

    async def broken(context, name, reason):
        if name == "a":
            raise RuntimeError("unexpected")
        await real(context, name, reason)

    monkeypatch.setattr(recording_steps, "end_recording", broken)
    await recording_steps.end_open_recordings(context)
    recordings = context.metadata[RECORDINGS_KEY]
    assert (recordings["a"]["state"], recordings["b"]["state"]) == ("failed", "ended")
