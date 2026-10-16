"""``build_app`` on local Docker, served by uvicorn and driven by OpenEnv's own clients over its WebSocket: a bundle
task played and unplayed scores as ``agent-env run`` scores it, every episode-ending failure arrives as an error
observation, each session's episode is torn down however it ends, and OpenEnv's validator accepts the server.

Requires a docker daemon; reuses the items server env and throwaway registry of ``server_env_local_test``."""

import asyncio
import contextlib
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest
import uvicorn
from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient

from agent_env.artifact import EnvironmentArtifact, FileArtifact
from agent_env.bundle import BundleError, run_bundle
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.openenv import build_app
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from tst.integration.env.envs.server_env_local_test import VERIFY_CRATE, _docker, _free_port, _put_items_image
from tst.util.a2a_test_agent import put_test_agent

pytestmark = pytest.mark.integration

RAISING_VERIFY = "async def verify(mcp_url):\n    raise RuntimeError('the verifier broke')\n"


@pytest.fixture
def items(local_stack, tmp_path, monkeypatch):
    """The items env and its verifier in the store, and a writer of bundle folders whose tasks play it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"openenv-items-{uid}", docker_image_artifact=_put_items_image(f"openenv-items-{uid}"),
                           environment_name="items", env_provider_type="server")
    verifiers = {name: FileArtifact.put_bytes(id=f"openenv-{name}-{uid}", description="openenv e2e",
                                              filename="verify.py", content=code.encode(),
                                              content_type="text/x-python").id
                 for name, code in (("crate", VERIFY_CRATE), ("raising", RAISING_VERIFY))}
    broken = EnvironmentArtifact.put(id=f"openenv-broken-{uid}", environment_name="items", file_artifact=(
        FileArtifact.put_bytes(id=f"openenv-broken-file-{uid}", description="openenv e2e", filename="items.json",
                               content=b"not json", content_type="application/json")))

    def task(*, agent="nobody", verifier="crate", load=None, turns=1):
        steps = [{"id": "env", "type": "deploy_env", "env_id": env.id}]
        if load is not None:
            steps.append({"id": "load", "type": "load_artifact", "env_id": env.id, "artifact_id": load})
        steps.append({"id": "policy", "type": "deploy_agent", "agent_name": "policy", "a2a_agent_id": agent,
                      "env_ids": [env.id], "role": "alice@example.com", "system_prompt": "You keep the items."})
        for n in range(turns):
            steps.append({"id": f"play-{n}", "type": "prompt_agent", "agent_name": "policy", "prompt_id": f"play-{n}",
                          "prompt": "Add a crate." if n == 0 else "Add another.", "timeout_seconds": 300,
                          **({"depends_on": [f"play-{n - 1}"]} if n else {})})
        steps.append({"id": "check", "type": "env_outcome_verifier", "env_id": env.id,
                      "file_artifact_id": verifiers[verifier], "verifier_id": "crate"})
        return steps

    def bundle(name, tasks, evals=None):
        root = tmp_path / name
        for task_name, steps in tasks.items():
            (root / "tasks").mkdir(parents=True, exist_ok=True)
            (root / "tasks" / f"{task_name}.json").write_text(json.dumps(steps))
        for eval_name, names in (evals or {}).items():
            (root / "evals").mkdir(exist_ok=True)
            (root / "evals" / f"{eval_name}.toml").write_text(f"tasks = {json.dumps(names)}\n")
        return root

    task.bundle, task.broken, task.sandboxes = bundle, broken.id, tmp_path / "sandboxes"
    return task


@contextlib.contextmanager
def _serving(app):
    """``app`` under uvicorn on a thread of its own, its lifespan included, as a deployment serves it."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=_free_port(), log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "uvicorn didn't start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{server.config.port}"
    finally:
        server.should_exit = True
        thread.join(timeout=300)


def _left(sandboxes: Path) -> list[Path]:
    return list(sandboxes.glob("agent-env-*"))


async def _play(url, *, add, split="all", index=0):
    async with GenericEnvClient(base_url=url, message_timeout_s=300) as env:
        reset = await env.reset(split=split, index=index)
        if add:
            await env.step({"type": "call_tool", "tool_name": "items_add_item", "arguments": {"item": "crate"}})
        ended = await env.step({"type": "call_tool", "tool_name": "end_turn", "arguments": {"response": "Done."}})
        return reset, ended


@pytest.mark.asyncio
async def test_a_bundle_task_played_by_openenvs_clients_scores_as_agent_env_run_scores_it(items):
    root = items.bundle("dock", {"crate": items(), "wiring": items()[:1]}, evals={"smoke": ["crate"]})
    app = build_app(str(root))

    with _serving(app) as url:
        async with httpx.AsyncClient(base_url=url) as http:
            assert [split["name"] for split in (await http.get("/dock/splits")).json()] == ["all", "smoke"]
            task = (await http.post("/dock/task", json={"split": "all", "index": 0})).json()["task"]
        assert (task["name"], task["prompt"], task["system_prompt"], task["agent"]) == (
            "crate", "Add a crate.", "You keep the items.", "policy")

        async with GenericEnvClient(base_url=url, message_timeout_s=300) as env:
            reset = await env.reset(split="smoke", index=0)
            assert (reset.done, reset.observation["metadata"]["prompt"]) == (False, "Add a crate.")
            listed = (await env.step({"type": "list_tools"})).observation["tools"]
            assert {"items_add_item", "items_whoami", "list_items"} <= {tool["name"] for tool in listed}
            assert listed[-1]["name"] == "end_turn"
            added = await env.step({"type": "call_tool", "tool_name": "items_add_item", "arguments": {"item": "crate"}})
            assert json.loads(added.observation["result"]["content"][0]["text"]) == {"count": 1}
            ended = await env.step({"type": "call_tool", "tool_name": "end_turn",
                                    "arguments": {"response": "Added a crate."}})
            assert (ended.done, ended.reward, ended.observation["metadata"]["scores"]) == (True, 1.0, {"check": 1.0})

        (_, unplayed), (_, played) = await asyncio.gather(_play(url, add=False), _play(url, add=True))
        assert (unplayed.reward, played.reward) == (0.0, 1.0)

        async with MCPToolClient(base_url=url, message_timeout_s=300) as mcp:
            await mcp.reset(split="all", index=0)
            assert "items_add_item" in {tool.name for tool in await mcp.list_tools()}
            whoami = await mcp.call_tool("items_whoami")  # /mcp answers a plain Environment's call with its observation
            assert json.loads(whoami["result"]["data"])["role"] == "alice@example.com"
    assert _left(items.sandboxes) == []

    echo = items.bundle("dock-echo", {"crate": items(agent=put_test_agent(f"echo-{uuid.uuid4().hex[:8]}").id)})
    [run] = (await asyncio.to_thread(run_bundle, echo)).runs
    assert run.scores == {"check": unplayed.reward}


@pytest.mark.asyncio
async def test_a_two_turn_task_hands_over_its_next_prompt_then_its_reward(items):
    with _serving(build_app(str(items.bundle("dock", {"twice": items(turns=2)})))) as url:
        async with GenericEnvClient(base_url=url, message_timeout_s=300) as env:
            await env.reset()
            first = await env.step({"type": "call_tool", "tool_name": "end_turn", "arguments": {"response": "One."}})
            await env.step({"type": "call_tool", "tool_name": "items_add_item", "arguments": {"item": "crate"}})
            second = await env.step({"type": "call_tool", "tool_name": "end_turn", "arguments": {"response": "Two."}})

    assert (first.done, first.observation["result"]["prompt"]) == (False, "Add another.")
    assert (second.done, second.reward) == (True, 1.0)


@pytest.mark.asyncio
async def test_every_failure_that_ends_an_episode_is_an_error_observation_and_a_tool_error_isnt(items):
    root = items.bundle("dock", {"crate": items(), "broken-load": items(load=items.broken),
                                 "broken-verify": items(verifier="raising")})

    with _serving(build_app(str(root))) as url:
        async with GenericEnvClient(base_url=url, message_timeout_s=300) as env:
            failed_setup = await env.reset(split="all", index=0)
            assert (failed_setup.done, failed_setup.observation["metadata"]["status"]) == (True, "error")
            assert failed_setup.observation["metadata"]["task"]["name"] == "broken-load"

            await env.reset(split="all", index=1)
            bad_call = await env.step({"type": "call_tool", "tool_name": "items_add_item",
                                       "arguments": {"times": "many"}})
            assert (bad_call.done, bad_call.observation["error"]["error_type"]) == (False, "execution_error")
            unknown = await env.step({"type": "call_tool", "tool_name": "nope", "arguments": {}})
            assert unknown.observation["error"]["error_type"] == "tool_not_found"
            broken_verifier = await env.step({"type": "call_tool", "tool_name": "end_turn", "arguments": {}})
            assert (broken_verifier.done, broken_verifier.reward) == (True, None)
            assert "the verifier broke" in broken_verifier.observation["metadata"]["error"]
            after = await env.step({"type": "list_tools"})
            assert after.observation["metadata"]["error"] == "the episode has ended; call reset"

            await env.reset(split="all", index=2)
            state = await env.state()
            context = Task.get_instance(state["episode_id"]).context
            [deployed] = context["deployed_envs"]
            _docker("kill", f"agent-{deployed['sandbox_id']}")
            killed = await env.step({"type": "call_tool", "tool_name": "items_add_item", "arguments": {"item": "x"}})
            assert (killed.done, killed.observation["error"]["error_type"]) == (False, "transport_error")
            ended = await env.step({"type": "call_tool", "tool_name": "end_turn", "arguments": {}})
            assert (ended.done, ended.reward, ended.observation["metadata"]["status"]) == (True, None, "error")
    assert _left(items.sandboxes) == []


@pytest.mark.asyncio
@pytest.mark.int_test_slow
async def test_each_session_holds_its_own_deployment_and_every_way_it_ends_tears_it_down(items):
    root = items.bundle("dock", {"crate": items()})

    with _serving(build_app(str(root), max_concurrent_envs=2)) as url:
        rewards = await asyncio.gather(_play(url, add=True), _play(url, add=False))
        assert [ended.reward for _, ended in rewards] == [1.0, 0.0]
        assert _left(items.sandboxes) == []

        held = GenericEnvClient(base_url=url, message_timeout_s=300)
        await held.connect()
        await held.reset()
        first = (await held.state())["episode_id"]
        await held.reset()
        assert Task.get_instance(first).status == "cancelled"
        assert len(_left(items.sandboxes)) == 1

        other = GenericEnvClient(base_url=url, message_timeout_s=300)
        await other.connect()
        await other.reset()
        with pytest.raises(Exception, match="(?i)capacity"):
            async with GenericEnvClient(base_url=url, message_timeout_s=60) as third:
                await third.reset()
        await other.close()
        await _until(lambda: len(_left(items.sandboxes)) == 1)
        open_at_shutdown = (await held.state())["episode_id"]

    assert _left(items.sandboxes) == []
    with namespace_routing():
        assert Task.get_instance(open_at_shutdown).status == "cancelled"


def test_openenvs_own_validator_runs_against_the_server(items):
    root = items.bundle("dock", {"crate": items()})

    with _serving(build_app(str(root))) as url:
        report = subprocess.run([str(Path(sys.executable).with_name("openenv")), "validate", "--url", url, "--level",
                                 "runtime", "--json", "--timeout", "300"], capture_output=True, text=True, timeout=900)

    findings = json.loads(report.stdout)
    assert (report.returncode, findings["passed"], findings["summary"]["failed_criteria"]) == (0, True, []), (
        report.stdout)


def test_a_bundle_naming_an_env_the_store_doesnt_hold_is_refused_at_startup(items):
    root = items.bundle("dock", {"crate": [{**items()[0], "env_id": "no-such-env"}, *items()[1:]]})

    with pytest.raises(BundleError, match="no-such-env"):
        build_app(str(root))


async def _until(holds, timeout=120):
    async with asyncio.timeout(timeout):
        while not holds():
            await asyncio.sleep(0.2)
