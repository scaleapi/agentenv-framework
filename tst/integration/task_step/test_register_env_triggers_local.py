"""Named nl-trigger executors on a local gateway: each nl action reaches only the executor it names, that executor acts
in the env as its own role, and personas react to each other through `when.roles`. Requires a docker daemon; spins up a
throwaway ``registry:2``."""

import asyncio
import functools
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER
from agent_env.env.gateway.constants import TRIGGER_IN_FLIGHT_STATUSES
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs, local_stack as local_stack

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]


@pytest.mark.asyncio
async def test_each_nl_action_reaches_the_executor_it_names(local_stack):
    _seed_bootstrap_envs()
    uid = uuid.uuid4().hex[:8]
    items = MCPServerEnv.put(id=f"trig-items-{uid}", environment_name="items",
                             docker_image_artifact=_put_items_image(f"trig-items-{uid}"))
    env = MultiEnv.put(id=f"trig-{uid}", mcp_server_envs=[items], name="suite")
    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    # host.docker.internal: loopback on Docker Desktop, the bridge gateway on Linux.
    host = env._sandbox.host_ips[-1]
    started: list[_Executor] = []
    triggers = [
        {"id": "to-alice", "when": {"type": "action", "tool": "list_items"},
         "actions": [{"type": "nl", "executor": "alice", "instruction": "ALICE-ACTS"}]},
        {"id": "to-bob", "when": {"type": "action", "tool": "list_items"},
         "actions": [{"type": "nl", "executor": "bob", "instruction": "BOB-ACTS"}]},
        {"id": "on-add", "when": {"type": "action", "tool": "items_add_item", "repeat": True}, "actions": []},
    ]
    try:
        alice = _Executor("alice", "alice@example.com", deployed.gateway_url, host, started)
        bob = _Executor("bob", "bob@example.com", deployed.gateway_url, host, started)
        context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[alice.agent, bob.agent], metadata={})
        await RegisterEnvTriggersStep(id="reg", version=None, env_id=env.id, triggers=triggers,
                                      watch_roles=["default"], executor_timeout_seconds=60).execute(context)
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            provoke = await client.post("/step", json={"action": "call_tool", "tool_name": "list_items", "arguments": {}})
            assert provoke.status_code == 200, provoke.text
            state = await _quiet(client, {"to-alice": 1, "to-bob": 1, "on-add": 0})
            stored = await client.post("/step", json={"action": "call_tool", "tool_name": "list_items", "arguments": {}},
                                       headers={AGENT_ENV_ROLE_HEADER: "harness"})
    finally:
        for executor in started:
            executor.stop()
        await env.close()

    statuses = {t["id"]: t["status"] for t in state["triggers"]}
    assert statuses == {"to-alice": "fired", "to-bob": "fired", "on-add": "armed"}, state["events"]
    assert (alice.instructions, bob.instructions) == (["ALICE-ACTS"], ["BOB-ACTS"])
    assert (alice.acted_as, bob.acted_as) == (["alice@example.com"], ["bob@example.com"])
    assert sorted(json.loads(stored.json()["content"][0]["text"])["items"]) == [
        "ALICE-ACTS by alice@example.com", "BOB-ACTS by bob@example.com"]
    named = {e["trigger_id"]: e["detail"].get("executor") for e in state["events"] if e["kind"] == "action_ok"}
    assert named == {"to-alice": "alice", "to-bob": "bob"}
    assert state["config"]["executors"] == ["alice", "bob"]
    assert context.metadata["env_trigger_registrations"][0]["executor_agent_names"] == ["alice", "bob"]


@pytest.mark.asyncio
async def test_personas_react_to_each_other_until_one_stays_quiet(local_stack):
    _seed_bootstrap_envs()
    uid = uuid.uuid4().hex[:8]
    items = MCPServerEnv.put(id=f"chain-items-{uid}", environment_name="items",
                             docker_image_artifact=_put_items_image(f"chain-items-{uid}"))
    env = MultiEnv.put(id=f"chain-{uid}", mcp_server_envs=[items], name="suite")
    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    host = env._sandbox.host_ips[-1]
    started: list[_Executor] = []
    triggers = [
        {"id": "denise-hears-solver", "when": {"type": "action", "tool": "list_items"},
         "actions": [{"type": "nl", "executor": "denise", "instruction": "DENISE-TO-SOLVER"}]},
        {"id": "bob-hears-denise", "when": {"type": "action", "tool": "items_add_item", "repeat": True,
                                            "roles": ["denise@example.com"]},
         "actions": [{"type": "nl", "executor": "bob", "instruction": "BOB-TO-DENISE"}]},
        {"id": "denise-hears-bob", "when": {"type": "action", "tool": "items_add_item", "repeat": True,
                                            "roles": ["bob@example.com"]},
         "actions": [{"type": "nl", "executor": "denise", "instruction": "DENISE-TO-BOB"}]},
    ]
    try:
        denise = _Executor("denise", "denise@example.com", deployed.gateway_url, host, started)
        bob = _Executor("bob", "bob@example.com", deployed.gateway_url, host, started, replies=1)
        context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[denise.agent, bob.agent], metadata={})
        await RegisterEnvTriggersStep(id="reg", version=None, env_id=env.id, triggers=triggers,
                                      watch_roles=["default"], executor_timeout_seconds=60).execute(context)
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            provoke = await client.post("/step", json={"action": "call_tool", "tool_name": "list_items", "arguments": {}})
            assert provoke.status_code == 200, provoke.text
            state = await _quiet(client, {"denise-hears-solver": 1, "bob-hears-denise": 2, "denise-hears-bob": 1})
            stored = await client.post("/step", json={"action": "call_tool", "tool_name": "list_items", "arguments": {}},
                                       headers={AGENT_ENV_ROLE_HEADER: "harness"})
    finally:
        for executor in started:
            executor.stop()
        await env.close()

    assert (denise.instructions, bob.instructions) == (["DENISE-TO-SOLVER", "DENISE-TO-BOB"],
                                                       ["BOB-TO-DENISE", "BOB-TO-DENISE"])
    assert json.loads(stored.json()["content"][0]["text"])["items"] == [
        "DENISE-TO-SOLVER by denise@example.com", "BOB-TO-DENISE by bob@example.com",
        "DENISE-TO-BOB by denise@example.com"]
    provokers = [(e["trigger_id"], e["provoking"]["role"]) for e in state["events"] if e["kind"] == "detected"]
    assert provokers == [("denise-hears-solver", "default"), ("bob-hears-denise", "denise@example.com"),
                         ("denise-hears-bob", "bob@example.com"), ("bob-hears-denise", "denise@example.com")]


class _Executor:
    """An A2A executor on this machine that acts on each instruction through the gateway, under its own role; with
    ``replies``, it stops acting after that many."""

    def __init__(self, name: str, role: str, gateway_url: str, host: str, started: list,
                 replies: int | None = None) -> None:
        self.role, self.gateway_url, self.replies = role, gateway_url, replies
        self.instructions: list[str] = []
        self.acted_as: list[str] = []
        self._server = ThreadingHTTPServer((host, 0), functools.partial(_ExecutorHandler, self))
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        started.append(self)
        url = f"http://{host}:{self._server.server_port}"
        self.agent = DeployedAgent(agent_name=name, api_url=url, a2a_url=url, role=role)

    def act(self, instruction: str) -> None:
        self.instructions.append(instruction)
        if self.replies is not None and len(self.instructions) > self.replies:
            return
        whoami = self._call("items_whoami", {})
        self.acted_as.append(json.loads(whoami["content"][0]["text"])["role"])
        self._call("items_add_item", {"item": f"{instruction} by {self.role}"})

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _call(self, tool: str, arguments: dict) -> dict:
        response = httpx.post(f"{self.gateway_url}/step", json={"action": "call_tool", "tool_name": tool,
                                                                "arguments": arguments},
                              headers={AGENT_ENV_ROLE_HEADER: self.role}, timeout=60)
        response.raise_for_status()
        return response.json()


class _ExecutorHandler(BaseHTTPRequestHandler):
    def __init__(self, executor: _Executor, *args, **kwargs) -> None:
        self.executor = executor
        super().__init__(*args, **kwargs)

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if request["method"] == "message/send":
            self.executor.act(request["params"]["message"]["parts"][0]["text"])
            result = {"id": uuid.uuid4().hex, "status": {"state": "submitted"}}
        else:
            result = {"id": request["params"]["id"],
                      "status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": "done"}]}}}
        payload = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


async def _quiet(client: httpx.AsyncClient, fires: dict[str, int], timeout_s: float = 120) -> dict:
    """The gateway's trigger state once each trigger in ``fires`` has fired that many times, none is in flight, and
    nothing else fires for a few seconds after."""
    deadline = time.monotonic() + timeout_s
    while True:
        state = (await client.get("/triggers/state")).raise_for_status().json()
        triggers = {t["id"]: t for t in state["triggers"]}
        if all(triggers[tid]["fire_count"] >= n and triggers[tid]["status"] not in TRIGGER_IN_FLIGHT_STATUSES
               for tid, n in fires.items()):
            await asyncio.sleep(3)
            later = (await client.get("/triggers/state")).raise_for_status().json()
            assert {t["id"]: t["fire_count"] for t in later["triggers"]} == fires, later["events"][-6:]
            return later
        if time.monotonic() > deadline:
            raise AssertionError(f"chain did not settle: { {tid: t['fire_count'] for tid, t in triggers.items()} }")
        await asyncio.sleep(1)
