"""A server env on the local backend: its one MCP server in a LocalSandbox container, with no gateway and no Modal.

The env deploys, a later process restores its record and loads data into the server, and the teardown step, which
rebuilds each sandbox from disk, removes the container. A load is timed by the size of the payload it staged, through
the deploy's own handle and a restored one; an environment of several files is staged file by file at their paths, in a
folder of the load's own that an earlier load's file can't be in the way of, and handed to the server's data/add
together, timed by their total size; and the server's state reads as JSON whether ``data/get`` answers with
data or with a file bundle. A loop of one's own lists, declares and calls the server's tools through ``ToolSession``,
and plays a task's agent as an episode, around the echo agent. Requires a docker daemon; spins up a throwaway
``registry:2`` and skips if it can't start.
"""

import asyncio
import json
import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx
import pytest
from agentenv_protocol import ToolSession, ToolSessionError
from agentenv_protocol import client as protocol_v1

from agent_env.artifact import DockerImageArtifact, EnvironmentArtifact, FileArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, get_config, reset_config, set_image_store
from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedSandboxEnv, Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.gateway import constants
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.task import EpisodeSpec, Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task.teardown import TORN_DOWN_KEY
from agent_env.task_step.task_steps.teardown_sandboxes import TeardownSandboxesTaskStep
from tst.util.a2a_test_agent import put_test_agent

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

_REPO = Path(__file__).resolve().parents[4]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _wait_ready(url: str, codes: tuple[int, ...], timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code in codes:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture
def local_stack(monkeypatch, tmp_path):
    """Local stores and a throwaway registry; yields the container names the test registers, removed afterwards."""
    if _docker("info").returncode != 0:
        pytest.skip("docker daemon not available")
    reg_port = _free_port()
    reg_name = f"server-e2e-registry-{uuid.uuid4().hex[:8]}"
    if _docker("run", "-d", "--rm", "--name", reg_name, "-p", f"127.0.0.1:{reg_port}:5000", "registry:2").returncode != 0:
        pytest.skip("could not start registry:2")
    host = f"localhost:{reg_port}"
    if not _wait_ready(f"http://{host}/v2/", (200, 401)):
        _docker("rm", "-f", reg_name)
        pytest.skip("local registry did not become ready")

    owned: list[str] = []
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    for var in ("DOCUMENT", "OBJECT", "IMAGE"):
        monkeypatch.setenv(f"AGENT_ENV_{var}_STORE", "local")
    configure()
    set_image_store(LocalRegistryImageStore(host))
    reset_artifact_store()
    try:
        yield owned
    finally:
        # A run's sandboxes aren't known up front: each has a work dir in this test's sandbox root, named after it.
        for work_dir in sandboxes.glob("agent-env-*"):
            if m := re.match(r"agent-env-(local-[0-9a-f]+)-", work_dir.name):
                owned.append(f"agent-{m.group(1)}")
        for container_name in owned:
            _docker("rm", "-f", container_name)
        # Agents upload through this process's grant server, whose certificate is from this test's state root.
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()
        _docker("rm", "-f", reg_name)


@pytest.mark.asyncio
async def test_a_server_env_deploys_restores_loads_and_tears_down_on_local_containers(local_stack):
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-{uid}", docker_image_artifact=_put_items_image(f"server-items-{uid}"),
                           environment_name="items", env_provider_type="server")

    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    container = f"agent-{deployed.sandbox_id}"
    local_stack.append(container)
    assert type(deployed) is DeployedSandboxEnv and deployed.sandbox_type == "local"
    assert deployed.environment_card["name"] == "items" and deployed.mcp_url.startswith("http://127.0.0.1:")

    restored = await Env.from_instance_id(deployed.instance_id)
    await restored.load_environment_artifact(_items_artifact(uid))
    base_url = await legacy_protocol.v1_base_url(deployed, None, "items")
    assert (await protocol_v1.get_data(base_url)).model_dump()["parts"][0]["data"] == {"items": ["snap-x", "snap-y"]}

    step = TeardownSandboxesTaskStep(id="teardown", version=None, env_ids=[env.id])
    context = await step.execute(TaskStepContext(deployed_envs=[deployed]))
    assert context.metadata[TORN_DOWN_KEY] == [deployed.sandbox_id]
    assert _docker("ps", "-aq", "--filter", f"name=^/{container}$").stdout.strip() == ""


@pytest.mark.asyncio
async def test_a_load_is_timed_by_its_payload_and_the_servers_state_reads_as_json_from_a_file_export(local_stack, monkeypatch):
    staged_sizes, timeout_for = [], constants.data_plane_load_timeout_s
    monkeypatch.setattr(constants, "data_plane_load_timeout_s", lambda size: staged_sizes.append(size) or timeout_for(size))
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-{uid}", docker_image_artifact=_put_items_image(f"server-items-{uid}"),
                           environment_name="items", env_provider_type="server")
    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    local_stack.append(f"agent-{deployed.sandbox_id}")
    artifact = _items_artifact(uid)

    await env.load_environment_artifact(artifact)  # the deploy's container-mode handle
    await (await Env.from_instance_id(deployed.instance_id)).load_environment_artifact(artifact)  # a restored one

    assert staged_sizes == [len(artifact.get_file_artifact().load())] * 2
    items = {"items": ["snap-x", "snap-y"]}
    assert await legacy_protocol.service_state(deployed, None, "items") == items
    base_url = await legacy_protocol.v1_base_url(deployed, None, "items")
    card = await protocol_v1.get_card(base_url)
    await protocol_v1.invoke_extension(base_url, card, "urn:agentenv:export-as-file/v1", {"enabled": True})
    assert [part.kind for part in (await protocol_v1.get_data(base_url)).parts] == ["file"]
    assert await legacy_protocol.service_state(deployed, None, "items") == items


@pytest.mark.asyncio
async def test_an_environment_of_several_files_loads_each_from_its_path_in_one_add(local_stack, monkeypatch):
    staged_sizes, timeout_for = [], constants.data_plane_load_timeout_s
    monkeypatch.setattr(constants, "data_plane_load_timeout_s", lambda size: staged_sizes.append(size) or timeout_for(size))
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-{uid}", docker_image_artifact=_put_items_image(f"server-items-{uid}"),
                           environment_name="items", env_provider_type="server")
    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    local_stack.append(f"agent-{deployed.sandbox_id}")
    contents = {path: json.dumps({"items": items}).encode()
                for path, items in (("inbox/a.json", ["in-a", "in-b"]), ("top.json", ["top"]))}
    artifact = EnvironmentArtifact.put(id=f"server-items-files-{uid}", environment_name="items", file_artifacts={
        path: FileArtifact.put_bytes(id=f"server-items-file-{index}-{uid}", description="server env e2e",
                                     filename=path.rsplit("/", 1)[-1], content=content, content_type="application/json")
        for index, (path, content) in enumerate(contents.items())})
    # A file an earlier load left at /data/inbox, where this load's inbox/ folder would go.
    earlier = EnvironmentArtifact.put(id=f"server-items-inbox-{uid}", environment_name="items", file_artifact=FileArtifact.put_bytes(
        id=f"server-items-inbox-file-{uid}", description="server env e2e", filename="inbox",
        content=json.dumps({"items": ["earlier"]}).encode(), content_type="application/json"))
    restored = await Env.from_instance_id(deployed.instance_id)

    await restored.load_environment_artifact(earlier)
    await restored.load_environment_artifact(artifact)

    assert await legacy_protocol.service_state(deployed, None, "items") == {"items": ["in-a", "in-b", "top"]}
    assert staged_sizes[-1] == sum(map(len, contents.values()))


@pytest.mark.asyncio
async def test_a_deployed_envs_tools_are_listed_and_called_from_a_loop_of_ones_own(local_stack):
    """A trainer's view: the env's tools listed with their schemas and called, the env's data plane agreeing on the state
    they left, a failing tool as an error result, concurrent sessions each answered as their own caller, and a restart
    as a lost session."""
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-{uid}", docker_image_artifact=_put_items_image(f"server-items-{uid}"),
                           environment_name="items", env_provider_type="server")
    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    container = f"agent-{deployed.sandbox_id}"
    local_stack.append(container)
    base_url = await legacy_protocol.v1_base_url(deployed, None, "items")

    async with ToolSession(mcp_url=deployed.mcp_url) as session:
        tools = {tool.name: tool for tool in await session.list_tools()}
        added = await session.call_tool("items_add_item", {"item": "crate", "times": 2})
        listed = await session.call_tool("list_items")
        await protocol_v1.invoke_extension(base_url, await protocol_v1.get_card(base_url), "urn:agentenv:set-errors/v1",
                                           {"tool_name": "list_items", "error_rate": 1.0})
        failed = await session.call_tool("list_items")

    assert {"items_add_item", "list_items"} <= set(tools)
    assert tools["items_add_item"].inputSchema["properties"]["item"]["description"] == "The item to add."
    assert json.loads(added.text) == {"count": 2} and json.loads(listed.text) == {"items": ["crate", "crate"]}
    assert failed.isError and "injected error for list_items" in failed.text
    assert (await protocol_v1.get_data(base_url)).parts[0].data == {"items": ["crate", "crate"]}

    async def add(i: int) -> tuple[int, dict, str]:
        async with ToolSession(mcp_url=deployed.mcp_url, role=f"worker{i}@example.com") as own:
            count = json.loads((await own.call_tool("items_add_item", {"item": f"w{i}"})).text)["count"]
            return count, json.loads((await own.call_tool("items_whoami")).text), own.session_id
    added_by = await asyncio.gather(*(add(i) for i in range(8)))
    assert sorted(count for count, _, _ in added_by) == list(range(3, 11))
    assert [caller for _, caller, _ in added_by] == [{"role": f"worker{i}@example.com", "session": session_id}
                                                     for i, (_, _, session_id) in enumerate(added_by)]
    assert sorted((await protocol_v1.get_data(base_url)).parts[0].data["items"]) == ["crate"] * 2 + [f"w{i}" for i in range(8)]

    async with ToolSession(mcp_url=deployed.mcp_url) as before_restart:
        assert _docker("restart", container).returncode == 0
        assert _wait_ready(f"{base_url}/.well-known/agent-env.json", (200,))
        with pytest.raises(ToolSessionError, match="no longer knows session"):
            await before_restart.call_tool("list_items")
    async with ToolSession(mcp_url=deployed.mcp_url) as after_restart:
        assert json.loads((await after_restart.call_tool("list_items")).text) == {"items": []}


VERIFY_CRATE = """
from agentenv_protocol import client


async def verify(mcp_url):
    items = (await client.get_data(mcp_url.removesuffix("/mcp"))).parts[0].data["items"]
    return [{"id": "crate", "result": "crate" in items}]
"""


ADD_A_CRATE = {"role": "assistant", "content": "Adding one.", "tool_calls": [
    {"id": "c1", "type": "function", "function": {"name": "items_add_item", "arguments": '{"item": "crate"}'}},
    {"id": "c2", "type": "function", "function": {"name": "items_whoami", "arguments": "{}"}}]}


@pytest.mark.asyncio
async def test_a_tasks_agent_is_played_from_a_loop_of_ones_own_as_an_episode(local_stack, tmp_path):
    """A trainer's view of a seeded task: the env deployed and loaded and the echo agent prompted around the outside
    agent, which plays its turn through the env's tools as its role through a proxy served in this process, which
    records the model's calls and the tools they ran; its reply scored under the verifier's step; and every episode,
    played, not played, abandoned or forgotten past its turn's deadline, leaving nothing behind."""
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-{uid}", docker_image_artifact=_put_items_image(f"server-items-{uid}"),
                           environment_name="items", env_provider_type="server")
    verify = FileArtifact.put_bytes(id=f"verify-crate-{uid}", description="episode e2e", filename="verify.py",
                                    content=VERIFY_CRATE.encode(), content_type="text/x-python")
    task = Task.from_dict({"id": f"crate-{uid}", "steps": [
        {"id": "env", "type": "deploy_env", "env_id": "<env>"},
        {"id": "load", "type": "load_artifact", "env_id": "<env>", "artifact_id": _items_artifact(uid).id},
        {"id": "policy", "type": "deploy_agent", "agent_name": "policy", "env_ids": ["<env>"],
         "role": "<who>@example.com"},
        {"id": "echo", "type": "deploy_agent", "agent_name": "echo", "a2a_agent_id": put_test_agent(f"echo-{uid}").id},
        {"id": "ask-echo", "type": "prompt_agent", "agent_name": "echo", "prompt_id": "ask-echo",
         "prompt": "Count the crates."},
        {"id": "play", "type": "prompt_agent", "agent_name": "policy", "prompt_id": "play", "prompt": "Add a <item>."},
        {"id": "check", "type": "env_outcome_verifier", "env_id": "<env>", "file_artifact_id": verify.id,
         "verifier_id": "crate-<week>"},
    ]})
    seed = {"env": env.id, "who": "alice", "item": "crate", "week": "3"}
    spec = EpisodeSpec.from_task(task, trainer_agents=["policy"], seed=seed)

    async def play(episode, add):
        turn = await episode.next_turn()
        assert turn.prompt == "Add a crate."
        assert [(agent["agent_name"], bool(agent["sandbox_id"]))
                for agent in Task.get_instance(episode.id).context["deployed_agents"]] == [("policy", False),
                                                                                            ("echo", True)]
        if add:
            async with turn.tools() as tools:
                names = {tool.name for tool in tools.list()}
                assert {"items_add_item", "items_whoami"} <= names and "bash" not in names
                await turn.record_model_call(ADD_A_CRATE, model="policy-model")
                await tools.call("items_add_item", {"item": "crate"})
                assert json.loads((await tools.call("items_whoami")).text)["role"] == "alice@example.com"
            await turn.record_model_call("Added a crate.", model="policy-model")
        await turn.end("Added a crate." if add else None)
        return await episode.result()

    sandboxes = tmp_path / "sandboxes"
    played, unplayed = await spec.start(), await spec.start()
    assert await asyncio.gather(play(played, True), play(unplayed, False)) == [{"check": 1.0}, {"check": 0.0}]
    assert list(sandboxes.glob("agent-env-*")) == []
    record = Task.get_instance(played.id)
    assert record.status == "completed" and record.context["metadata"]["episode"]["trainer_agents"] == ["policy"]
    assert [agent["agent_name"] for agent in record.context["deployed_agents"]] == ["policy", "echo"]
    replies = {response["step_id"]: response["response"] for response in record.context["prompt_responses"]}
    assert replies["play"] == "Added a crate." and replies["ask-echo"].startswith("Echo:")
    [play] = [response for response in record.context["prompt_responses"] if response["step_id"] == "play"]
    assert [span["name"] for span in _stored(play["agent_trajectory_object_url"])] == [
        "chat", "items_add_item", "items_whoami", "chat", "turn"]

    abandoned = await spec.start()
    await abandoned.next_turn()
    assert len(list(sandboxes.glob("agent-env-*"))) == 2
    await abandoned.abandon("test")
    assert (Task.get_instance(abandoned.id).status, Task.get_instance(abandoned.id).error) == ("cancelled", "test")
    assert list(sandboxes.glob("agent-env-*")) == []

    forgotten = await EpisodeSpec.from_task(task, trainer_agents=["policy"], seed=seed, turn_timeout=3).start()
    await forgotten.next_turn()
    with pytest.raises(TimeoutError, match="did not complete within 3s"):
        await forgotten.next_turn()
    assert Task.get_instance(forgotten.id).status == "failed"
    assert list(sandboxes.glob("agent-env-*")) == []
    for episode in (played, unplayed, abandoned, forgotten):
        context = Task.get_instance(episode.id).context
        sandbox_ids = [entry["sandbox_id"] for key in ("deployed_envs", "deployed_agents")
                       for entry in context[key] if entry.get("sandbox_id")]
        assert len(sandbox_ids) == 2
        for sandbox_id in sandbox_ids:
            assert _docker("ps", "-aq", "--filter", f"name=^/agent-{sandbox_id}$").stdout.strip() == ""


def _put_items_image(artifact_id: str) -> DockerImageArtifact:
    """The in-memory items server (card 'items'), built with the vendored agentenv_protocol and pushed to the registry."""
    data = _REPO / "tst" / "data" / "agentenv_mcp"
    with tempfile.TemporaryDirectory() as build_dir:
        bd = Path(build_dir)
        shutil.copytree(_REPO / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol", bd / "agentenv_protocol",
                        ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("server.py", "Dockerfile", "seed.json"):
            shutil.copy(data / name, bd / name)
        build = _docker("build", "-t", artifact_id, str(bd))
        assert build.returncode == 0, f"items build failed: {build.stderr[-2000:]}"
    return DockerImageArtifact.put(id=artifact_id, description="server env e2e", image_name=artifact_id)


def _items_artifact(uid: str) -> EnvironmentArtifact:
    return EnvironmentArtifact.put(
        id=f"server-items-data-{uid}",
        environment_name="items",
        file_artifact=FileArtifact.put_bytes(id=f"server-items-file-{uid}", description="server env e2e", filename="items.json",
                                             content=json.dumps({"items": ["snap-x", "snap-y"]}).encode(), content_type="application/json"),
    )


@pytest.mark.asyncio
async def test_an_agent_that_works_in_its_sandbox_is_played_through_a_proxy_deployed_there(local_stack, tmp_path):
    """A task that loads files into its agent, gives it an env, and checks what the agent left in both: the agent is
    never deployed, a proxy is, in its place, and every call the loop makes runs in it, ``bash`` as the agent's user in
    its folder with no model credentials, the env's tools as its role; the checks score what the calls left, the proxy
    records the turn, and nothing is left running."""
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-{uid}", docker_image_artifact=_put_items_image(f"server-items-{uid}"),
                           environment_name="items", env_provider_type="server")
    verify = FileArtifact.put_bytes(id=f"verify-crate-{uid}", description="proxy e2e", filename="verify.py",
                                    content=VERIFY_CRATE.encode(), content_type="text/x-python")
    greeting = FileArtifact.put_bytes(id=f"greeting-{uid}", description="proxy e2e", filename="greeting.txt",
                                      content=b"hello", content_type="text/plain")
    task = Task.from_dict({"id": f"copy-{uid}", "steps": [
        {"id": "env", "type": "deploy_env", "env_id": env.id},
        {"id": "agent", "type": "deploy_agent", "agent_name": "solver", "a2a_agent_id": f"never-deployed-{uid}",
         "env_ids": [env.id], "role": "bob@example.com"},
        {"id": "load", "type": "load_artifact", "agent_name": "solver", "artifact_id": greeting.id,
         "destination_path": "/app/inputs"},
        {"id": "solve", "type": "prompt_agent", "agent_name": "solver", "prompt_id": "solve",
         "prompt": "Copy inputs/greeting.txt to outputs/answer.txt, and add a crate.", "poll_interval_seconds": 0.5},
        {"id": "check", "type": "verify_sandbox", "agent_name": "solver", "criteria": [
            {"type": "probe_file_contains", "criterion": "the greeting is copied", "paths": ["outputs/answer.txt"],
             "expected": "^hello$"}]},
        {"id": "crate", "type": "env_outcome_verifier", "env_id": env.id, "file_artifact_id": verify.id,
         "verifier_id": "crate"},
    ]})
    copy = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps(
            {"command": "mkdir -p outputs && cp inputs/greeting.txt outputs/answer.txt"})}},
        {"id": "c2", "type": "function", "function": {"name": "items_add_item", "arguments": '{"item": "crate"}'}}]}

    episode = await EpisodeSpec.from_task(task).start()
    turn = await episode.next_turn()
    async with turn.tools() as tools:
        names = [tool.name for tool in tools.list()]
        assert names[0] == "bash" and {"items_add_item", "items_whoami"} <= set(names)
        assert (await tools.call("bash", {"command": "pwd; whoami; cat inputs/greeting.txt"})).text == (
            "/app\nagent\nhello")
        assert (await tools.call("bash", {"command": "env | grep -c ^LITELLM_ || true"})).text == "0\n"
        assert json.loads((await tools.call("items_whoami")).text)["role"] == "bob@example.com"
        await turn.record_model_call(copy, model="solver-model", usage={"prompt_tokens": 12, "completion_tokens": 5})
        await tools.call("bash", {"command": "mkdir -p outputs && cp inputs/greeting.txt outputs/answer.txt"})
        await tools.call("items_add_item", {"item": "crate"})
    await turn.record_model_call("Copied, and added a crate.", model="solver-model")
    await turn.end("Copied, and added a crate.")

    assert await episode.result() == {"check": 1.0, "crate": 1.0}
    record = Task.get_instance(episode.id)
    [solver] = record.context["deployed_agents"]
    assert solver["agent_name"] == "solver" and solver["sandbox_id"]
    [response] = record.context["prompt_responses"]
    assert (response["response"], response["tool_call_count"]) == ("Copied, and added a crate.", 5)
    stored = _stored(response["agent_trajectory_object_url"])
    assert [span["name"] for span in stored] == [
        "bash", "bash", "items_whoami", "chat", "bash", "items_add_item", "chat", "turn"]
    assert stored[-1]["attributes"]["gen_ai.request.model"] == "solver-model"
    assert list((tmp_path / "sandboxes").glob("agent-env-*")) == []
    for entry in (*record.context["deployed_agents"], *record.context["deployed_envs"]):
        assert _docker("ps", "-aq", "--filter", f"name=^/agent-{entry['sandbox_id']}$").stdout.strip() == ""


def _stored(url: str) -> list[dict]:
    return json.loads(get_config().get_object_store().get(url))
