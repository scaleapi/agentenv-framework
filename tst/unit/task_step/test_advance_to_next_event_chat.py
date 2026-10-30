"""A multi-day chat end to end, with no model: prompt_agent runs the advance-to-next-event hook against an in-process
gateway's real clock and notifications, a scripted agent, and a scripted user sim that answers reply, no_reply or
done."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import agent_env
from agent_env.a2a_agent import conversation_store
from agent_env.artifact import FileArtifact
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.constants import GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import Gateway
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.unit.task_step.test_prompt_agent_user_side import _USER_SIM_CARD

HOOK = Path(agent_env.__file__).parent / "examples" / "advance-to-next-event" / "artifacts" / "advance-to-next-event" / "hook.py"
GATEWAY = "http://gateway.test"
AGENT = "http://agent.test"
NORA = "http://nora.test"
MONDAY = "2026-03-02T08:00:00Z"

pytestmark = pytest.mark.usefixtures("local_stores")


class Chat:
    """The agent and Nora, scripted: each answers the next message it is sent with the next of its lines, and the
    agent's line may first do something on the gateway, as its tool calls would."""

    def __init__(self, monkeypatch, gateway: Gateway, agent: list, nora: list[dict]):
        self.agent, self.nora = list(agent), list(nora)
        self.nora_read: list[str] = []
        self.agent_read: list[str] = []
        self._tasks: dict[str, list[dict]] = {}
        self._gateway = httpx.ASGITransport(app=gateway._mcp.streamable_http_app())
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(self._answer)
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_client(*a, **{**kw, "transport": transport}))

    async def _answer(self, request: httpx.Request) -> httpx.Response:
        origin = f"{request.url.scheme}://{request.url.host}"
        if origin == GATEWAY:
            response = await self._gateway.handle_async_request(request)
            await response.aread()
            return response
        if request.url.path == "/ext/agent-config":
            return httpx.Response(200, json={"ok": True})
        body = json.loads(request.content)
        if body["method"] == "message/send":
            read = "".join(p["text"] for p in body["params"]["message"]["parts"])
            task_id = f"task-{len(self._tasks) + 1}"
            if origin == AGENT:
                self.agent_read.append(read)
                act, line = self.agent.pop(0)
                if act is not None:
                    act()
                self._tasks[task_id] = [{"kind": "text", "text": line}]
            else:
                self.nora_read.append(read)
                self._tasks[task_id] = [{"kind": "text", "text": json.dumps(self.nora.pop(0))}]
            result = {"id": task_id, "contextId": "ctx"}
        else:
            parts = self._tasks[body["params"]["id"]]
            result = {"id": body["params"]["id"], "status": {"state": "completed", "message": {"parts": parts}}}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    gw = Gateway(host="127.0.0.1", port=0, server_name="AgentEnvGateway", internal_mcp_servers=[])
    gw._clock = Clock()
    gw._clock.set_time(MONDAY, rate=0)
    gw._notifications.configure({"roles": ["default"]})
    yield gw
    gw._close_trajectory_file()


def _context() -> TaskStepContext:
    ctx = TaskStepContext(instance_id="ti-1")
    ctx.deployed_envs.append(DeployedGatewayEnv(
        env_id="curbfinch", env_version=1, gateway_url=GATEWAY, mcp_url="", sandbox_id="sb",
        environment_card_url=f"{GATEWAY}{WELL_KNOWN_PATH}", metadata={"deploy_step_id": "deploy_env"},
        environment_card={"capabilities": {"extensions": list(GATEWAY_EXTENSIONS)}}))
    ctx.deployed_agents.append(DeployedAgent(agent_name="solver", api_url=AGENT, a2a_url=AGENT, a2a_card={}))
    ctx.deployed_agents.append(DeployedAgent(agent_name="human_agent", api_url=NORA, a2a_url=NORA,
                                             sandbox_id="sb-nora", sandbox_type="local", a2a_card=_USER_SIM_CARD))
    return ctx


def _week() -> PromptAgentTaskStep:
    FileArtifact.put_bytes("advance-to-next-event", description="hook", filename="hook.py", content=HOOK.read_bytes())
    return PromptAgentTaskStep(
        id="week", version=None, agent_name="solver", poll_interval_seconds=0, max_conversation_turns=12,
        prompt="Keep an eye on curbfinch this week for an 8x10 rug under $300.",
        post_turn_hooks=[{"file_artifact_id": "advance-to-next-event", "params": {
            "end_time": "P5D", "person_messages": [{"at": "2026-03-04T12:00:00Z",
                                                    "text": "actually it needs to be 9x12"}]}}])


@pytest.mark.asyncio
async def test_a_week_in_one_chat_she_replies_when_asked_stays_quiet_otherwise_and_ends_it(monkeypatch, gateway):
    def evenings():
        gateway._notifications.schedule("look for new rugs", every="P1D", at="2026-03-02T18:00:00Z", until="P4D")

    chat = Chat(monkeypatch, gateway, agent=[
        (evenings, "On it. I'll look every evening."),
        (None, ""),  # Monday's check finds nothing worth telling her
        (None, "Found a 1970s wool 8x10 for $280. Want me to message the seller?"),
        (None, "Messaged the seller; I'll tell you when they answer."),
        (None, "Got it, 9x12 now. The wool one is 8x10, so I'll pass on it."),
        (None, "Bought a 9x12 jute rug for $250; it arrives Friday."),
    ], nora=[
        {"outcome": "no_reply", "message": ""},
        {"outcome": "reply", "message": "yes please, ask if it's still there"},
        {"outcome": "no_reply", "message": ""},
        {"outcome": "no_reply", "message": ""},  # told of the line she sent
        {"outcome": "no_reply", "message": ""},
        {"outcome": "done", "message": "all sorted, thanks!"},
    ])
    ctx = _context()

    await _week().execute(ctx)

    assert chat.agent_read == [
        "Keep an eye on curbfinch this week for an 8x10 rug under $300.",
        "[Mon Mar 2, 6:00 PM · your scheduled check] look for new rugs",
        "[Tue Mar 3, 6:00 PM · your scheduled check] look for new rugs",
        "yes please, ask if it's still there",
        "[Wed Mar 4, 12:00 PM] actually it needs to be 9x12",
        "[Wed Mar 4, 6:00 PM · your scheduled check] look for new rugs",
    ]
    assert chat.nora_read == [
        "On it. I'll look every evening.",
        "Found a 1970s wool 8x10 for $280. Want me to message the seller?",
        "Messaged the seller; I'll tell you when they answer.",
        'You just sent your assistant this message: "actually it needs to be 9x12"',
        "Got it, 9x12 now. The wool one is 8x10, so I'll pass on it.",
        "Bought a 9x12 jute rug for $250; it arrives Friday.",
    ]
    assert chat.agent == [] and chat.nora == []
    assert gateway._clock.read()["virtual_time"] == "2026-03-04T18:00:00Z"
    assert [c["status"] for c in gateway._notifications.state("default")["checks"]] == ["active"]
    records = ctx.metadata["post_turn_hooks"]["week"]
    assert [(r["turn"], r["person"]) for r in records if "person" in r] == [
        (1, "no_reply"), (2, "not_asked"), (3, "reply"), (4, "no_reply"), (4, "no_reply"), (5, "no_reply"),
        (6, "done")]
    conversation = conversation_store.get_conversation(ctx.metadata["a2a_conversations"]["week"])
    assert conversation["status"] == "closed"
    assert all("You just sent" not in p.get("text", "") for m in conversation["messages"] for p in m["parts"])
