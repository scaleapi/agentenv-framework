"""prompt_agent follows a deployed agent's trajectory while each turn runs and stores it in chunks, on the local
defaults. The agents are the echo agent (tst/data/a2a_agent), which appends a step a second before its echo, so a
turn runs for a few seconds; over a two-turn conversation with a user-sim, each turn is stored under its own prefix,
in more than one chunk, and its chunks join to that turn's final trajectory; each turn names the one after it, and
the last none. Needs Docker and a throwaway local registry."""

import json
import re
import shutil
import socket
import subprocess
import time
import uuid

import httpx
import pytest

from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, get_config, reset_config, set_image_store
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.task import Task
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.util.a2a_test_agent import put_test_agent

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
    started = _docker("run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:5000", "registry:2")
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


def _deploy(agent, name: str) -> DeployAgentTaskStep:
    return DeployAgentTaskStep(
        id=f"deploy-{name}", version=None, env_ids=[], a2a_agent_id=agent.id, a2a_agent_version=agent.version,
        agent_name=name, sandbox_type="local",
    )


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
