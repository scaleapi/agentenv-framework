"""Named nl-trigger executors on a local gateway: each nl action reaches only the executor it names, that executor acts
in the env as its own role, personas react to each other through `when.roles`, replies wait out their delay and working
hours in virtual time, deadlines fire on silence, and the trajectory names who made each call. Requires a docker daemon;
spins up a throwaway ``registry:2``."""

import asyncio
import functools
import json
import threading
import time
import uuid
from datetime import datetime, timezone
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


@pytest.mark.asyncio
async def test_a_reply_waits_its_delay_and_hours_and_its_writes_name_the_fire(local_stack):
    _seed_bootstrap_envs()
    env, deployed, host = await _deploy_items("pace")
    started: list[_Executor] = []
    triggers = [{"id": "bob-hears-solver", "when": {"type": "action", "tool": "list_items"},
                 "actions": [{"type": "nl", "executor": "bob", "instruction": "BOB-ACTS", "delay": "PT2M"}]}]
    try:
        bob = _Executor("bob", "bob@example.com", deployed.gateway_url, host, started)
        context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[bob.agent], metadata={})
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            await RegisterEnvTriggersStep(id="reg", version=None, env_id=env.id, triggers=triggers, watch_roles=["default"],
                                          executor_timeout_seconds=60,
                                          executor_hours={"bob": {"start": "09:00", "end": "17:00"}}).execute(context)
            # At 60x the delay ends about 08:52, and bob's hours hold him to 09:00: some 10 real seconds in all.
            await _arm_clock(client, "2026-10-14T08:50:00Z", 60)
            (await client.post("/step", json={"action": "call_tool", "tool_name": "list_items", "arguments": {}})).raise_for_status()
            state = await _quiet(client, {"bob-hears-solver": 1})
            trajectory = [json.loads(line) for line in (await client.get("/trajectory")).raise_for_status().text.splitlines()]
    finally:
        for executor in started:
            executor.stop()
        await env.close()

    paced = [e for e in state["events"] if e["kind"] in ("delayed", "deferred", "action_ok")]
    assert [e["kind"] for e in paced] == ["delayed", "deferred", "action_ok"]
    delayed, deferred, _ = paced
    assert deferred["until"] == "2026-10-14T09:00:00Z"
    assert (_virtual(deferred) - _virtual(delayed)).total_seconds() >= 120
    calls = [e for e in trajectory if e.get("event_type") == "tool_call"]
    cause = [{"trigger_id": "bob-hears-solver", "action_index": 0, "fire": 1}]
    assert ("list_items", "default", None) in [(e["tool_call"]["function_name"], e["role"], e.get("caused_by"))
                                               for e in calls]
    bobs = [e for e in calls if e["role"] == "bob@example.com"]
    assert [(e["tool_call"]["function_name"], e.get("caused_by")) for e in bobs] == [("items_whoami", cause),
                                                                                     ("items_add_item", cause)]
    assert all(_virtual(e) >= datetime(2026, 10, 14, 9, tzinfo=timezone.utc) for e in bobs)


@pytest.mark.asyncio
async def test_a_deadline_fires_on_silence_and_a_reply_cancels_it(local_stack):
    _seed_bootstrap_envs()
    env, deployed, host = await _deploy_items("deadline")
    started: list[_Executor] = []
    triggers = []
    for n in (1, 2):
        triggers += [{"id": f"dm-{n}", "when": {"type": "action", "tool": "list_items"}, "actions": []},
                     {"id": f"deadline-{n}", "when": {"type": "time", "after": f"dm-{n}", "offset": "PT2M",
                                                      "unless": {"tool": "items_add_item", "roles": ["default"],
                                                                 "where": {"item": {"regex": f"^REPLY-{n}"}}}},
                      "actions": [{"type": "nl", "executor": "bob", "instruction": f"ESCALATE-{n}"}]}]
    try:
        bob = _Executor("bob", "bob@example.com", deployed.gateway_url, host, started)
        context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[bob.agent], metadata={})
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            await _arm_clock(client, "2026-10-14T10:00:00Z", 60)
            await RegisterEnvTriggersStep(id="reg", version=None, env_id=env.id, triggers=triggers, watch_roles=["default"],
                                          executor_timeout_seconds=60).execute(context)
            (await client.post("/step", json={"action": "call_tool", "tool_name": "list_items", "arguments": {}})).raise_for_status()
            (await client.post("/step", json={"action": "call_tool", "tool_name": "items_add_item",
                                              "arguments": {"item": "REPLY-2"}})).raise_for_status()
            state = await _quiet(client, {"dm-1": 1, "dm-2": 1, "deadline-1": 1, "deadline-2": 0})
    finally:
        for executor in started:
            executor.stop()
        await env.close()

    statuses = {t["id"]: t["status"] for t in state["triggers"]}
    assert (statuses["deadline-1"], statuses["deadline-2"]) == ("fired", "cancelled"), state["events"][-8:]
    assert bob.instructions == ["ESCALATE-1"]
    [cancelled] = [e for e in state["events"] if e["kind"] == "cancelled"]
    assert (cancelled["trigger_id"], cancelled["provoking"]["role"]) == ("deadline-2", "default")


@pytest.mark.asyncio
async def test_the_step_can_wait_for_start_of_run_time_triggers(local_stack):
    _seed_bootstrap_envs()
    env, deployed, host = await _deploy_items("await")
    triggers = [{"id": "wf-1", "when": {"type": "time", "at": "2026-10-14T10:02:00Z"},
                 "actions": [{"type": "tool", "tool": "items_add_item", "args": {"item": "WORLD-1"}}]},
                {"id": "wf-2", "when": {"type": "time", "at": "2026-10-14T10:03:00Z"},
                 "actions": [{"type": "tool", "tool": "items_add_item", "args": {"item": "WORLD-2"}}]},
                {"id": "later", "when": {"type": "time", "at": "2026-10-14T18:00:00Z"},
                 "actions": [{"type": "tool", "tool": "items_add_item", "args": {"item": "LATER"}}]}]
    context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[], metadata={})
    try:
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            await _arm_clock(client, "2026-10-14T10:00:00Z", 60)
            started = time.monotonic()
            await RegisterEnvTriggersStep(id="reg", version=None, env_id=env.id, triggers=triggers, watch_roles=["default"],
                                          await_fired=["wf-1", "wf-2"], await_timeout_seconds=60).execute(context)
            waited = time.monotonic() - started
            state = (await client.get("/triggers/state")).raise_for_status().json()
            with pytest.raises(RuntimeError, match=r"did not all fire within 3s; statuses \{'later': 'armed'\}"):
                await RegisterEnvTriggersStep(id="reg-later", version=None, env_id=env.id, triggers=triggers[2:],
                                              watch_roles=["default"], await_fired=["later"],
                                              await_timeout_seconds=3).execute(context)
    finally:
        await env.close()

    awaited_first, awaited_later = (e["awaited"] for e in context.metadata["env_trigger_registrations"])
    assert awaited_first == {"wf-1": "fired", "wf-2": "fired"}
    assert awaited_later == {"later": "armed"}  # the failed wait still records its registration and last statuses
    assert waited >= 2.0, waited  # three virtual minutes at sixty-to-one
    assert {t["id"]: t["fire_count"] for t in state["triggers"]} == {"wf-1": 1, "wf-2": 1, "later": 0}, state["events"][-6:]


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


async def _deploy_items(label: str):
    """A MultiEnv of the in-repo items server on local compose, and the host address its gateway reaches this machine at."""
    uid = uuid.uuid4().hex[:8]
    items = MCPServerEnv.put(id=f"{label}-items-{uid}", environment_name="items",
                             docker_image_artifact=_put_items_image(f"{label}-items-{uid}"))
    env = MultiEnv.put(id=f"{label}-{uid}", mcp_server_envs=[items], name="suite")
    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    return env, deployed, env._sandbox.host_ips[-1]


def _virtual(event: dict) -> datetime:
    return datetime.fromisoformat(event["virtual_time"].replace("Z", "+00:00"))


async def _arm_clock(client: httpx.AsyncClient, virtual_time: str, rate: float) -> None:
    (await client.put("/clock/set-time", json={"virtual_time": virtual_time,
                                                "virtual_seconds_per_real_second": rate})).raise_for_status()
