"""collect_traffic_capture against a real gateway serving traffic/v1 from a capture folder: each capture file stored
as a pcap FileArtifact, named for its container and its place among that container's files, in one universe the run
records; what's recorded when the env doesn't capture, isn't deployed or refuses the token; and teardown collecting a
capturing env the run didn't, before the env closes."""

from __future__ import annotations

import asyncio
import os
import threading
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agent_env.artifact import Artifact
from agent_env.artifact.store import get_artifact_store
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.constants import PCAP_MEDIA_TYPE, TRAFFIC_EXTENSION
from agent_env.env.gateway.gateway import Gateway
from agent_env.task.store import find_task_instance, get_task_instance_store
from agent_env.task.task import _rolled_back_sandboxes
from agent_env.task.teardown import recorded_sandboxes, tear_down
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps import traffic_capture as traffic_capture_module
from agent_env.task_step.task_steps.traffic_capture import (
    TRAFFIC_KEY,
    CollectTrafficCaptureTaskStep,
    collect_uncollected_captures,
)

TOKEN = "control-secret"
_RealAsyncClient = httpx.AsyncClient
_PCAP = bytes.fromhex("d4c3b2a1020004000000000000000000ffff000001000000")


@pytest.fixture
def gateway(tmp_path, monkeypatch, local_stores):
    """A capturing gateway every httpx client in the test reaches, its folder holding two rotated files of the
    server's capture, the newer named first, and one of its frontend's."""
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    folder = tmp_path / "traffic"
    folder.mkdir()
    for name, body, mtime in (("notes-website-backend.pcap0", b"newer", 2000), ("notes-website-backend.pcap1", b"older", 1000),
                              ("notes-website-frontend.pcap0", b"front", 1500)):
        (folder / name).write_bytes(_PCAP + body)
        os.utime(folder / name, (mtime, mtime))
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[], traffic_dir=str(folder),
                 control_token=TOKEN)
    app = gw._asgi_app()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _RealAsyncClient(
        transport=httpx.ASGITransport(app=app), **{k: v for k, v in kw.items() if k != "verify"}))
    yield gw
    gw._close_trajectory_file()


def _deployed(env_id: str = "notes-env", *, token: str | None = TOKEN, capturing: bool = True,
              sandbox_id: str = "vm-1") -> DeployedGatewayEnv:
    card = {"name": "env1", "capabilities": {"extensions": [TRAFFIC_EXTENSION] if capturing else []}}
    return DeployedGatewayEnv(env_id=env_id, env_version=1, gateway_url="http://gateway", sandbox_id=sandbox_id,
                              environment_card_url="http://gateway/.well-known/agent-env.json", environment_card=card,
                              control_token=token)


def _context(*envs) -> TaskStepContext:
    return TaskStepContext(instance_id="run-1", deployed_envs=list(envs), metadata={})


def _step(**kw) -> CollectTrafficCaptureTaskStep:
    return CollectTrafficCaptureTaskStep(id="collect", version=None, env_id="notes-env", **kw)


def test_the_step_is_registered_and_roundtrips():
    step = _step(fail_task_on_error=True)
    assert get_task_step_registry()["collect_traffic_capture"] is CollectTrafficCaptureTaskStep
    again = CollectTrafficCaptureTaskStep.from_dict(step.to_dict())
    assert (again.env_id, again.fail_task_on_error) == ("notes-env", True)
    assert CollectTrafficCaptureTaskStep.from_dict({"id": "c", "env_id": "e"}).fail_task_on_error is False
    with pytest.raises(ValueError, match="needs the env_id"):
        CollectTrafficCaptureTaskStep(id="c", version=None, env_id="")


@pytest.mark.asyncio
async def test_each_capture_is_stored_as_a_pcap_in_one_universe_oldest_first(gateway):
    context = await _step().execute(_context(_deployed()))

    entry = context.metadata[TRAFFIC_KEY]["notes-env"]
    assert entry["state"] == "collected"
    assert entry["files"] == ["notes-website-backend-00.pcap", "notes-website-backend-01.pcap",
                              "notes-website-frontend-00.pcap"]
    assert entry["containers"] == ["notes-website-backend", "notes-website-frontend"]
    assert entry["bytes"] == 3 * len(_PCAP) + 15
    universe = Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"])
    files = universe.get_file_artifacts()
    assert sorted(files) == entry["files"]
    first = files["notes-website-backend-00.pcap"]
    assert first.content_type == PCAP_MEDIA_TYPE
    assert get_artifact_store().get_object(first.object_url) == _PCAP + b"older"
    assert get_artifact_store().get_object(files["notes-website-backend-01.pcap"].object_url) == _PCAP + b"newer"


@pytest.mark.asyncio
async def test_two_collections_of_one_env_in_the_same_second_are_each_stored_under_a_prefix_of_their_own(gateway):
    context = _context(_deployed())
    first = dict(await traffic_capture_module.collect_captures(context, _deployed()))
    second = await traffic_capture_module.collect_captures(context, _deployed())

    assert (first["state"], second["state"]) == ("collected", "collected")
    urls = [{name: artifact.object_url for name, artifact in
             Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"]).get_file_artifacts().items()}
            for entry in (first, second)]
    assert sorted(urls[0]) == sorted(urls[1]) and not set(urls[0].values()) & set(urls[1].values())


@pytest.mark.asyncio
async def test_a_collection_cancelled_mid_upload_leaves_the_upload_its_file_to_finish(gateway, monkeypatch):
    started, release, uploaded = threading.Event(), threading.Event(), []
    real = traffic_capture_module._upload

    def held(universe_id, object_url, path):
        started.set()
        release.wait(10)
        uploaded.append((path, os.path.exists(path)))
        return real(universe_id, object_url, path)

    monkeypatch.setattr(traffic_capture_module, "_upload", held)
    collecting = asyncio.ensure_future(traffic_capture_module.collect_captures(_context(_deployed()), _deployed()))
    assert await asyncio.to_thread(started.wait, 10)
    collecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await collecting
    release.set()
    for _ in range(200):
        if uploaded and not os.path.exists(uploaded[0][0]):
            break
        await asyncio.sleep(0.01)

    [(path, existed)] = uploaded
    assert existed and not os.path.exists(path)


@pytest.mark.asyncio
async def test_an_env_that_doesnt_capture_is_recorded_as_failed_and_fails_the_task_only_when_asked(gateway):
    context = await _step().execute(_context(_deployed(capturing=False)))
    assert context.metadata[TRAFFIC_KEY]["notes-env"]["state"] == "failed"
    assert "traffic/v1" in context.metadata[TRAFFIC_KEY]["notes-env"]["error"]
    with pytest.raises(RuntimeError, match="weren't collected"):
        await _step(fail_task_on_error=True).execute(_context(_deployed(capturing=False)))


@pytest.mark.asyncio
async def test_a_record_without_the_gateways_token_is_refused(gateway):
    context = await _step().execute(_context(_deployed(token="stale")))
    entry = context.metadata[TRAFFIC_KEY]["notes-env"]
    assert entry["state"] == "failed" and "401" in entry["error"]


@pytest.mark.asyncio
async def test_an_env_the_run_didnt_deploy_is_recorded(gateway):
    context = await _step().execute(_context())
    assert context.metadata[TRAFFIC_KEY]["notes-env"] == {
        "state": "failed", "error": "RuntimeError: env 'notes-env' isn't deployed in this run"}
    with pytest.raises(RuntimeError, match="isn't deployed"):
        await _step(fail_task_on_error=True).execute(_context())


@pytest.mark.asyncio
async def test_teardown_collects_each_capturing_env_the_run_didnt_collect_a_failed_collection_again(gateway):
    capturing, plain, done = _deployed(), _deployed("plain-env", capturing=False), _deployed("done-env")
    context = _context(capturing, plain, done)
    context.metadata[TRAFFIC_KEY] = {"done-env": {"state": "collected", "artifact": None},
                                     "notes-env": {"state": "failed", "error": "ConnectError: refused"}}

    await collect_uncollected_captures(context, context.deployed_envs)

    assert context.metadata[TRAFFIC_KEY]["notes-env"]["state"] == "collected"
    assert "plain-env" not in context.metadata[TRAFFIC_KEY]
    assert context.metadata[TRAFFIC_KEY]["done-env"] == {"state": "collected", "artifact": None}


@pytest.mark.asyncio
async def test_tear_down_collects_a_capture_before_its_env_closes(gateway):
    context = _context(_deployed())
    collected_at_close: list[bool] = []

    async def close(env, timeout):
        collected_at_close.append(context.metadata[TRAFFIC_KEY]["notes-env"]["state"] == "collected")

    with patch("agent_env.task.teardown._close", close), \
         patch("agent_env.task.teardown.terminate", AsyncMock(return_value=False)):
        report = await tear_down(context, recorded_sandboxes(context), context.deployed_envs, timeout=30)

    assert collected_at_close == [True] and [s.sandbox_id for s in report.terminated] == ["vm-1"]


@pytest.mark.asyncio
async def test_tear_down_leaves_an_env_already_torn_down_alone(gateway):
    context = _context(_deployed())
    context.metadata["torn_down_sandbox_ids"] = ["vm-1"]
    with patch("agent_env.task.teardown._close", AsyncMock()), \
         patch("agent_env.task.teardown.terminate", AsyncMock(return_value=False)):
        await tear_down(context, recorded_sandboxes(context), context.deployed_envs, timeout=30)
    assert TRAFFIC_KEY not in context.metadata


@pytest.mark.asyncio
async def test_what_teardown_collects_is_saved_to_the_runs_stored_instance(gateway):
    instance = get_task_instance_store().create_instance("task-1", 1, total_steps=1)
    context = _context(_deployed())
    context.instance_id = instance.instance_id

    await collect_uncollected_captures(context, context.deployed_envs)

    stored = find_task_instance(instance.instance_id).context["metadata"][TRAFFIC_KEY]
    assert stored == context.metadata[TRAFFIC_KEY] and stored["notes-env"]["state"] == "collected"


@pytest.mark.asyncio
async def test_what_teardown_collected_before_its_time_ran_out_is_still_saved_to_the_run(gateway, monkeypatch):
    instance = get_task_instance_store().create_instance("task-1", 1, total_steps=1)
    context = _context(_deployed(), _deployed("slow-env"))
    context.instance_id = instance.instance_id
    real = traffic_capture_module.collect_captures

    async def collect(context, deployed, *, key=None):
        if deployed.env_id == "slow-env":
            await asyncio.sleep(30)
        return await real(context, deployed, key=key)

    monkeypatch.setattr(traffic_capture_module, "collect_captures", collect)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(1):
            await collect_uncollected_captures(context, context.deployed_envs)

    stored = find_task_instance(instance.instance_id).context["metadata"][TRAFFIC_KEY]
    assert stored["notes-env"]["state"] == "collected" and "slow-env" not in stored


@pytest.mark.asyncio
async def test_tear_down_collects_a_retrys_rolled_back_deployment_under_its_own_key_before_its_vm_goes(gateway):
    rolled_back, live = _deployed(sandbox_id="vm-1"), _deployed(sandbox_id="vm-2")
    gone = _rolled_back_sandboxes(TaskStepContext(deployed_envs=[rolled_back]).to_dict(), TaskStepContext().to_dict(), {})
    context = _context(live)
    context.metadata["retry_resets"] = [{"attempt": 2, "rolled_back_sandboxes": gone}]
    collected_before_terminate: list[set] = []

    async def terminate(sandbox):
        collected_before_terminate.append({key for key, entry in context.metadata[TRAFFIC_KEY].items()
                                           if entry["state"] == "collected"})
        return False

    with patch("agent_env.task.teardown._close", AsyncMock()), patch("agent_env.task.teardown.terminate", terminate):
        report = await tear_down(context, recorded_sandboxes(context), context.deployed_envs, timeout=30)

    assert sorted(s.sandbox_id for s in report.terminated) == ["vm-1", "vm-2"]
    assert collected_before_terminate == [{"notes-env", "notes-env-rolled-back-1"}] * 2
    assert context.metadata[TRAFFIC_KEY]["notes-env-rolled-back-1"]["artifact"] != \
        context.metadata[TRAFFIC_KEY]["notes-env"]["artifact"]
