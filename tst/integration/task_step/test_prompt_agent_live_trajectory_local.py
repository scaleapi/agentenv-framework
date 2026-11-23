"""prompt_agent follows a deployed agent's trajectory while each turn runs and stores it in chunks, on the local
defaults. The agents are the echo agent (tst/data/a2a_agent), which appends a step a second before its echo, so a
turn runs for a few seconds; over a two-turn conversation with a user-sim, each turn is stored under its own prefix,
in more than one chunk, and its chunks join to that turn's final trajectory; each turn names the one after it, and
the last none. The echo agent declares its native format and an ATIF adapter, so its card offers both; its native reads
name their format, its atif/1 reads are valid ATIF, and a read naming another format is refused with a code agent-env
reads. A snapshot of it carries its atif/1 copy's turn count to the fresh agent it is restored into, and its atif/1
reads leave out the model key a command it runs prints, which its native read keeps. An image sent to it is spooled
as its copy's media file, and pushed under an exact-object grant from the local store's grant server. Needs Docker and a
throwaway local registry."""

import base64
import hashlib
import json
import re
import shutil
import socket
import subprocess
import time
import uuid

import httpx
import pytest

from agentenv_protocol.atif import FORMAT as ATIF, PROFILE_VERSION, SCHEMA_VERSION, errors_in, media_paths, validate

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent.object_transfer import write_object
from agent_env.a2a_agent.protocol import poll_a2a_task, raise_for_extension_status, send_a2a_message
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, get_config, reset_config, set_image_store
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.snapshot_agent_state import SnapshotAgentStateTaskStep
from tst.integration.utils.registry import REGISTRY_IMAGE
from tst.util.a2a_test_agent import put_test_agent
from tst.util.images import png

pytestmark = pytest.mark.integration

REGISTRY_READY_SECONDS = 45
# A turn runs for about this many seconds, a read a second apart, so the follower reads it more than once.
LIVE_STEPS = 4


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def local_registry(monkeypatch, tmp_path):
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    port = _free_port()
    name = f"live-trajectory-registry-{uuid.uuid4().hex[:8]}"
    started = _docker("run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:5000", REGISTRY_IMAGE)
    assert started.returncode == 0, started.stderr
    host = f"localhost:{port}"
    deadline = time.time() + REGISTRY_READY_SECONDS
    while time.time() < deadline:
        try:
            if httpx.get(f"http://{host}/v2/", timeout=2).status_code in (200, 401):
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    configure()
    set_image_store(LocalRegistryImageStore(host))
    reset_artifact_store()
    try:
        yield tmp_path
    finally:
        # Only this test's containers: every local sandbox it deployed has a work dir in its own sandbox root, and its
        # container is named after that sandbox. Images are shared, so matching by image would reach other runs.
        for work_dir in sandboxes.glob("agent-env-*"):
            if m := re.match(r"agent-env-(local-[0-9a-f]+)-", work_dir.name):
                _docker("rm", "-f", f"agent-{m.group(1)}")
        _docker("rm", "-f", name)
        shutil.rmtree(sandboxes, ignore_errors=True)
        # The agents uploaded through this process's grant server, whose certificate is from this test's state root;
        # a later test's agents trust another root's CA, so it must start afresh.
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def _prescribing_atif() -> TaskStepContext:
    """A run that prescribes atif/1, so its prompt steps have the agents they prompt build it."""
    return TaskStepContext(metadata={"user_overrides": {"trajectory_formats": [ATIF]}})


def _deploy(agent, name: str, **snapshot) -> DeployAgentTaskStep:
    return DeployAgentTaskStep(
        id=f"deploy-{name}", version=None, env_ids=[], a2a_agent_id=agent.id, a2a_agent_version=agent.version,
        agent_name=name, sandbox_type="local", **snapshot,
    )


def _prompt(step_id: str, agent_name: str, context_id: str) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(
        id=step_id, version=None, prompt_id=step_id, agent_name=agent_name, prompt=step_id, context_id=context_id,
    )


async def _atif_users(deployed, context_id: str) -> tuple[dict, list[dict]]:
    """The atif/1 document of the context's latest task, and its user steps' profile records."""
    url = deployed.a2a_url + A2AAgent.find_extension(deployed.a2a_card, A2AAgent.EXT_TRAJECTORY)["params"]["endpoint"]
    async with httpx.AsyncClient(timeout=30, headers=deployed.control_headers) as client:
        page = (await client.post(url, json={"context_id": context_id, "after": 0})).json()
        read = (await client.post(url, json={"task_id": page["task_id"], "format": ATIF})).json()
    document = read["trajectory"]
    assert errors_in(validate(document)) == []
    return document, [step["extra"]["agentenv"] for step in document["steps"] if step["source"] == "user"]


def _json_lines(store, key: str) -> list:
    return [json.loads(line) for line in store.get(store.object_url(key)).decode().splitlines()]


@pytest.mark.asyncio
async def test_each_turn_is_stored_live_under_its_own_prefix_and_joins_to_its_final_trajectory(local_registry):
    suffix = uuid.uuid4().hex[:8]
    store = get_config().get_object_store()
    prefix_key = f"live-trajectory-{suffix}/"
    agent = put_test_agent(f"live-trajectory-agent-{suffix}")
    task = Task.put(id=f"live-trajectory-{suffix}", steps=[
        _deploy(agent, "solver"),
        _deploy(agent, "user"),
        PromptAgentTaskStep(
            id="ask", version=None, prompt_id="ask", agent_name="solver", user_agent_name="user",
            max_conversation_turns=2, poll_interval_seconds=1, prompt=f"live-steps {LIVE_STEPS}",
            trajectory_output_prefix=store.object_url(prefix_key), live_trajectory=True,
        ),
    ])

    context = await task.run()

    assert not context.metadata.get("failed_steps"), context.metadata.get("failed_steps")
    finals = context.prompt_responses[-1].target_agent_per_turn_trajectory_object_urls
    assert len(finals) == 2 and all(finals)
    assert context.prompt_responses[-1].target_agent_per_turn_trajectories == [
        {"format": "agentenv-echo-agent/v1", "atif": None}] * 2
    turns = [store.get_object_key(url).removeprefix(f"{prefix_key}trajectory-").removesuffix(".json") for url in finals]
    assert len(set(turns)) == 2
    for turn, final_url in zip(turns, finals):
        final = json.loads(store.get(final_url))
        assert [event["type"] for event in final] == ["step"] * LIVE_STEPS + ["echo"]
        live = f"{prefix_key}{turn}/live/"
        chunks = sorted(key for key in store.list(live) if key.endswith(".jsonl"))
        assert len(chunks) > 1, chunks
        assert [event for key in chunks for event in _json_lines(store, key)] == final
        assert json.loads(store.get(store.object_url(f"{live}meta.json"))) == {"format": "agentenv-echo-agent/v1"}
        assert json.loads(store.get(store.object_url(f"{live}end.json"))) == {"state": "completed", "next": len(final)}
    links = [json.loads(store.get(store.object_url(f"{prefix_key}{turn}/live/next.json"))) for turn in turns]
    assert links == [{"turn": turns[1]}, {"turn": None}]


@pytest.mark.asyncio
async def test_a_deployed_agent_serves_its_native_format_and_atif_and_refuses_another(local_registry):
    """And the run, which prescribes atif/1, has the turn's copy as its trajectory, stored in its folder with the image's
    file beside it, and reads no native trajectory."""
    suffix = uuid.uuid4().hex[:8]
    native = "agentenv-echo-agent/v1"
    image = png(6, 4)
    agent = put_test_agent(f"trajectory-formats-agent-{suffix}")
    task = Task.put(id=f"trajectory-formats-{suffix}", steps=[
        _deploy(agent, "solver"),
        PromptAgentTaskStep(
            id="ask", version=None, prompt_id="ask", agent_name="solver", context_id=f"formats-{suffix}",
            parts=[{"kind": "text", "text": "hello"},
                   {"kind": "file", "file": {"bytes": base64.b64encode(image).decode(), "mimeType": "image/png",
                                             "name": "shot.png"}}],
        ),
    ])

    context = await task.run(context=_prescribing_atif())

    assert not context.metadata.get("failed_steps"), context.metadata.get("failed_steps")
    [deployed] = context.deployed_agents
    params = A2AAgent.find_extension(deployed.a2a_card, A2AAgent.EXT_TRAJECTORY)["params"]
    assert (params["formats"], params["native"], params["default"]) == ([native, ATIF], native, native)
    url = deployed.a2a_url + params["endpoint"]
    async with httpx.AsyncClient(timeout=30, headers=deployed.control_headers) as client:
        page = (await client.post(url, json={"context_id": f"formats-{suffix}", "after": 0})).json()
        assert page["format"] == native
        for named in ({}, {"format": native}):
            final = (await client.post(url, json={"task_id": page["task_id"], **named})).json()
            assert (final["format"], [event["type"] for event in final["trajectory"]]) == (native, ["echo"])
        atif = (await client.post(url, json={"task_id": page["task_id"], "format": ATIF})).json()
        cursor = (await client.post(url, json={"context_id": f"formats-{suffix}", "after": 0, "format": ATIF})).json()
        refused = await client.post(url, json={"task_id": page["task_id"], "format": "other-events/1"})
    document = atif["trajectory"]
    assert (atif["format"], atif["schema_version"], atif["profile_version"]) == (ATIF, SCHEMA_VERSION, PROFILE_VERSION)
    assert errors_in(validate(document)) == []
    [shot] = media_paths(document)
    assert document["steps"][0]["message"] == [{"type": "text", "text": "hello"},
                                               {"type": "image", "source": {"media_type": "image/png", "path": shot}}]
    assert document["steps"][1]["message"].startswith("Echo: hello")
    assert (document["session_id"], atif["turn_context"]["task_id"]) == (f"formats-{suffix}", page["task_id"])
    assert cursor["events"] == document["steps"] and cursor["header"]["trajectory_id"] == document["trajectory_id"]
    with pytest.raises(httpx.HTTPStatusError, match="HTTP 400: format_not_offered"):
        raise_for_extension_status(refused, operation="trajectory read")
    [turn] = context.prompt_responses[-1].target_agent_per_turn_trajectories
    record, store = turn["atif"], get_config().get_object_store()
    folder = context.metadata["atif_copies"]["agents"]["solver"]["folder"]
    assert (turn["format"], record["status"], record["rules"]) == (ATIF, "valid", [])
    assert context.prompt_responses[-1].target_agent_per_turn_trajectory_object_urls == [record["url"]]
    assert record["url"] == f"{folder}{document['trajectory_id']}.json"
    assert json.loads(store.get(record["url"])) == document
    assert json.loads(store.get(record["turn_context_url"])) == atif["turn_context"]
    assert store.get(f"{folder}{shot}") == image
    assert errors_in(validate(document, media={shot: store.get(f"{folder}{shot}")})) == []


@pytest.mark.asyncio
async def test_a_snapshot_carries_the_turn_count_to_the_fresh_agent_it_is_restored_into(local_registry):
    suffix = uuid.uuid4().hex[:8]
    source, restored, snapshot_id = f"turns-{suffix}", f"restored-{suffix}", f"turns-snapshot-{suffix}"
    agent = put_test_agent(f"turn-count-agent-{suffix}")
    task = Task.put(id=f"turn-count-{suffix}", steps=[
        _deploy(agent, "solver"),
        _prompt("one", "solver", source),
        _prompt("two", "solver", source),
        SnapshotAgentStateTaskStep(id="snapshot", version=None, artifact_id=snapshot_id, prompt_id="two",
                                   agent_name="solver"),
        _deploy(agent, "restored", agent_snapshot_files_artifact_id=snapshot_id,
                agent_snapshot_target_context_id=restored),
        _prompt("three", "restored", restored),
    ])

    context = await task.run(context=_prescribing_atif())

    assert not context.metadata.get("failed_steps"), context.metadata.get("failed_steps")
    store = get_config().get_object_store()
    universe = FileArtifactUniverse.get(snapshot_id)
    state = store.get(store.object_url(store.get_object_key(universe.bundle_object_url).rstrip("/") + "/state"))
    assert json.loads(state) == {"turns": 2}
    agents = {deployed.agent_name: deployed for deployed in context.deployed_agents}
    before, before_users = await _atif_users(agents["solver"], source)
    after, after_users = await _atif_users(agents["restored"], restored)
    assert [(user["turn"], user["context_id"]) for user in before_users] == [(0, source), (1, source)]
    assert [(user["turn"], user["context_id"]) for user in after_users] == [(2, restored)]
    assert (after["session_id"], [step["message"] for step in after["steps"]]) == (restored, ["three", "Echo: three"])
    assert after["trajectory_id"] != before["trajectory_id"]


@pytest.mark.asyncio
async def test_a_model_key_a_command_prints_is_redacted_in_atif_and_kept_in_the_native_log(local_registry):
    suffix = uuid.uuid4().hex[:8]
    context_id = f"scrub-{suffix}"
    agent = put_test_agent(f"scrub-agent-{suffix}", env={"LITELLM_API_KEY": f"sk-echo-{uuid.uuid4().hex}"})
    task = Task.put(id=f"scrub-{suffix}", steps=[
        _deploy(agent, "solver"),
        PromptAgentTaskStep(id="ask", version=None, prompt_id="ask", agent_name="solver",
                            prompt="run printenv LITELLM_API_KEY", context_id=context_id),
    ])

    context = await task.run(context=_prescribing_atif())

    assert not context.metadata.get("failed_steps"), context.metadata.get("failed_steps")
    [deployed] = context.deployed_agents
    url = deployed.a2a_url + A2AAgent.find_extension(deployed.a2a_card, A2AAgent.EXT_TRAJECTORY)["params"]["endpoint"]
    async with httpx.AsyncClient(timeout=30, headers=deployed.control_headers) as client:
        page = (await client.post(url, json={"context_id": context_id, "after": 0})).json()
        native = (await client.post(url, json={"task_id": page["task_id"]})).json()["trajectory"]
        atif = (await client.post(url, json={"task_id": page["task_id"], "format": ATIF})).json()
    # The key the agent's sandbox holds, as the command printed it: the deploy may resolve one of its own.
    [echo] = [event for event in native if event["type"] == "echo"]
    key = re.search(r"ran printenv LITELLM_API_KEY: (\S+)", echo["output"]).group(1)
    assert len(key) >= 8
    served = json.dumps(atif)
    assert key not in served and f"[redacted:{len(key)}]" in served
    assert errors_in(validate(atif["trajectory"])) == []


@pytest.mark.asyncio
async def test_a_deployed_agent_pushes_its_copys_media_under_exact_object_grants(local_registry):
    suffix = uuid.uuid4().hex[:8]
    context_id = f"media-{suffix}"
    agent = put_test_agent(f"media-agent-{suffix}")
    context = await Task.put(id=f"media-{suffix}", steps=[_deploy(agent, "solver")]).run()
    assert not context.metadata.get("failed_steps"), context.metadata.get("failed_steps")
    [deployed] = context.deployed_agents
    image = png(6, 4)
    parts = [{"kind": "text", "text": "look"},
             {"kind": "file", "file": {"bytes": base64.b64encode(image).decode(), "mimeType": "image/png",
                                       "name": "shot.png"}}]
    trajectory = A2AAgent.find_extension(deployed.a2a_card, A2AAgent.EXT_TRAJECTORY)
    _, media_endpoint = A2AAgent.operation(trajectory, "media")
    _, prescribe_endpoint = A2AAgent.operation(trajectory, "prescribe")
    read_url = deployed.a2a_url + trajectory["params"]["endpoint"]
    unprescribed, _ = await send_a2a_message(deployed.a2a_url, parts, f"m0-{suffix}", f"unprescribed-{suffix}", 60)
    await poll_a2a_task(deployed.a2a_url, unprescribed, 120, 1)
    async with httpx.AsyncClient(timeout=60, headers=deployed.control_headers) as client:
        refused = await client.post(read_url, json={"task_id": unprescribed, "format": ATIF})
        prescribed = (await client.post(deployed.a2a_url + prescribe_endpoint, json={"formats": [ATIF]})).json()
    with pytest.raises(httpx.HTTPStatusError, match="HTTP 409: format_not_prescribed"):
        raise_for_extension_status(refused, operation="trajectory read")
    assert prescribed == {"formats": ["agentenv-echo-agent/v1", ATIF]}
    task_id, _ = await send_a2a_message(deployed.a2a_url, parts, f"m-{suffix}", context_id, 60)
    await poll_a2a_task(deployed.a2a_url, task_id, 120, 1)
    store = get_config().get_object_store()
    async with httpx.AsyncClient(timeout=60, headers=deployed.control_headers) as client:
        atif = (await client.post(read_url, json={"task_id": task_id, "format": ATIF})).json()
        [path] = media_paths(atif["trajectory"])
        url = store.object_url(f"atif-media-{suffix}/{path}")
        grant = write_object(store, url, media_type="image/png", max_bytes=len(image))
        pushed = await client.post(deployed.a2a_url + media_endpoint, json={
            "context_id": context_id, "objects": {path: grant.model_dump(mode="json", exclude_none=True)}})
    raise_for_extension_status(pushed, operation="media push")
    digest = hashlib.sha256(image).hexdigest()
    assert path == f"media/{digest}.png"
    assert pushed.json() == {"objects": {path: {"size_bytes": len(image), "sha256": digest}}, "missing": [], "failed": {}}
    assert store.get(url) == image
    [user] = [step for step in atif["trajectory"]["steps"] if step["source"] == "user"]
    assert user["extra"]["agentenv"]["media"][path] == {"sha256": digest, "bytes": len(image), "width": 6, "height": 4,
                                                       "name": "shot.png"}
