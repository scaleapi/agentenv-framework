"""prompt_agent's user side as it runs today: each request a conversation makes, in order, and what it records."""

from __future__ import annotations

import json
import re

import httpx
import pytest

from agent_env.a2a_agent import conversation_store
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.gateway.constants import EXT_TRIGGERS_URI, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

AGENT = "http://agent.test"
USER = "http://user.test"
HUMAN = "http://human.test"
ENV = "http://env.test"
HANG = "hang"

_ID_RE = re.compile(r"[0-9a-f]{32}")

pytestmark = pytest.mark.usefixtures("local_stores")

_AGENT_CONFIG = {"uri": "urn:agentenv:agent-config/v1", "params": {
    "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": ["output_format", "model"]}}}}}
_USER_SIM_CARD = {"capabilities": {"extensions": [_AGENT_CONFIG]}}
_TRIGGERS_CARD = {"capabilities": {"extensions": [
    _AGENT_CONFIG,
    {"uri": "urn:agentenv:triggers/v1",
     "params": {"endpoint": "/ext/triggers", "methods": {"decide": {"endpoint": "/ext/triggers/decide"}}}},
]}}
_ENV_TRIGGERS = {"triggers": [{"id": "school-email", "status": "fired"}, {"id": "reminder", "status": "armed"}]}
_FIRING_LOG = {"firing_log": [{"turn": 1, "fired": ["nudge"]}]}


def text(value: str) -> list[dict]:
    return [{"kind": "text", "text": value}]


def sim(message: str, done: bool = False, **extra) -> list[dict]:
    return text(json.dumps({"message": message, "done": done, **extra}))


class Peers:
    """Scripted agent, user side and env; ``requests`` lists each request, ids named <ctx> (the conversation), <idN>."""

    def __init__(self, monkeypatch, replies: dict[str, list], *, decide: list[dict] | None = None, on_poll=None):
        self.replies = {origin: list(answers) for origin, answers in replies.items()}
        self.decide = list(decide or [])
        self.on_poll = on_poll
        self.tasks: dict[str, object] = {}
        self.raw: list[tuple] = []
        self._ids: dict[str, str] = {}
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(self._answer)
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_client(*a, transport=transport, **kw))

    def requests(self, ctx: TaskStepContext) -> list[tuple]:
        self._ids = {ctx.metadata["a2a_conversations"]["talk"]: "<ctx>"}
        return self._norm(self.raw)

    def _norm(self, value):
        if isinstance(value, dict):
            return {k: self._norm(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._norm(v) for v in value)
        if isinstance(value, str) and _ID_RE.fullmatch(value):
            return self._ids.setdefault(value, f"<id{len(self._ids)}>")
        return value

    def _answer(self, request: httpx.Request) -> httpx.Response:
        origin = f"{request.url.scheme}://{request.url.host}"
        body = json.loads(request.content) if request.content else None
        self.raw.append((request.method, str(request.url), body, request.extensions["timeout"]["read"]))
        path = request.url.path
        if path == "/a2a":
            return self._a2a(origin, body)
        if path == "/ext/agent-config":
            return httpx.Response(200, json={"ok": True})
        if path == "/ext/triggers/decide":
            return httpx.Response(200, json=self.decide.pop(0))
        if path == "/ext/triggers":
            return httpx.Response(200, json=_FIRING_LOG)
        if origin == ENV and path == "/triggers/state":
            return httpx.Response(200, json=_ENV_TRIGGERS)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    def _a2a(self, origin: str, body: dict) -> httpx.Response:
        if body["method"] == "message/send":
            task_id = f"task-{len(self.tasks) + 1}"
            self.tasks[task_id] = self.replies[origin].pop(0)
            result = {"id": task_id, "contextId": "ctx"}
        elif body["method"] == "tasks/get":
            reply = self.tasks[body["params"]["id"]]
            if self.on_poll is not None:
                self.on_poll(origin)
            if reply == HANG:
                result = {"id": body["params"]["id"], "status": {"state": "working"}}
            else:
                state, parts = reply if isinstance(reply, tuple) else ("completed", reply)
                result = {"id": body["params"]["id"], "status": {"state": state, "message": {"parts": parts}}}
        else:
            result = {}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


def send(origin: str, parts: list[dict], message_id: str, context_id: str, timeout: int = 600) -> tuple:
    message = {"messageId": message_id, "role": "user", "parts": parts, "contextId": context_id}
    return ("POST", f"{origin}/a2a", {"jsonrpc": "2.0", "id": "1", "method": "message/send",
                                      "params": {"message": message, "configuration": {"blocking": False}}}, timeout)


def poll(origin: str, task_id: str) -> tuple:
    return ("POST", f"{origin}/a2a", {"jsonrpc": "2.0", "id": "poll", "method": "tasks/get",
                                      "params": {"id": task_id}}, 30)


def exchange(origin: str, parts: list[dict], message_id: str, context_id: str, task_id: str, timeout: int = 600):
    return [send(origin, parts, message_id, context_id, timeout), poll(origin, task_id)]


USER_SIM_OUTPUT_FORMAT = {"type": "json_schema", "schema": {"type": "object", "properties": {
    "message": {"type": "string", "description": "Your next reply"},
    "done": {"type": "boolean", "description": "Set true when you're satisfied and the conversation should end."},
}, "required": ["message", "done"]}}
USER_SIM_CONFIG = ("POST", f"{USER}/ext/agent-config", {"output_format": USER_SIM_OUTPUT_FORMAT}, 60)
ENV_STATE = ("GET", f"{ENV}/triggers/state", None, 15)
FIRING_LOG = ("GET", f"{USER}/ext/triggers", None, 15)


def decide(turn: int, solver_message: str, context_id: str) -> tuple:
    return ("POST", f"{USER}/ext/triggers/decide",
            {"turn": turn, "solver_message": solver_message, "context_id": context_id,
             "env_triggers": {"env": {"school-email": "fired", "reminder": "armed"}}}, 600)


def _context(*, user_card: dict | None = None, triggers: bool = False) -> TaskStepContext:
    ctx = TaskStepContext(instance_id="ti-1")
    ctx.deployed_agents.append(DeployedAgent(agent_name="solver", api_url=AGENT, a2a_url=AGENT, a2a_card={}))
    if user_card is not None:
        ctx.deployed_agents.append(DeployedAgent(agent_name="human_agent", api_url=USER, a2a_url=USER,
                                                 sandbox_id="sb-user", sandbox_type="local", a2a_card=user_card))
    if triggers:
        extensions = [e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_TRIGGERS_URI]
        ctx.deployed_envs.append(DeployedGatewayEnv(
            env_id="env", env_version=1, gateway_url=ENV, mcp_url="", sandbox_id="sb-env",
            environment_card_url=f"{ENV}{WELL_KNOWN_PATH}",
            environment_card={"capabilities": {"extensions": extensions}}))
        ctx.metadata["agent_trigger_registrations"] = [
            {"step_id": "reg", "agent_name": "human_agent", "added": ["nudge"]}]
    return ctx


def _step(turns: int = 4, **kw) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="talk", version=None, prompt="plan my week", agent_name="solver",
                               poll_interval_seconds=0, max_conversation_turns=turns, **kw)


def _stored(ctx: TaskStepContext, peers: Peers) -> dict:
    """The stored conversation, its message timestamps dropped and its ids named as ``peers.requests`` names them."""
    peers.requests(ctx)
    doc = conversation_store.get_conversation(ctx.metadata["a2a_conversations"]["talk"])
    for message in doc["messages"]:
        assert message.pop("ts")
    return peers._norm({
        "status": doc["status"],
        "messages": doc["messages"],
        "a2a_tasks": [(t["a2a_task_id"], t["state"], t["input_message_idx"], t.get("response_message_idx"))
                      for t in doc["a2a_tasks"]],
    })


def _user_side_metadata(ctx: TaskStepContext) -> dict:
    keys = ("env_trigger_snapshots", "agent_trigger_firings", "usersim_turn_outputs", "agent_trigger_state")
    return {k: ctx.metadata[k] for k in keys if k in ctx.metadata}


# ---- the LLM user simulator ----------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_user_sims_reply_is_the_agents_next_prompt_and_its_done_ends_the_run_unrecorded(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two")],
                                USER: [sim("make it shorter"), sim("perfect, thanks", done=True)]})
    ctx = _context(user_card=_USER_SIM_CARD)

    result = await _step().execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text("draft one"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("make it shorter"), "<id3>", "<ctx>", "task-3"),
        *exchange(USER, text("draft two"), "<id4>", "<ctx>", "task-4"),
    ]
    assert _stored(ctx, peers) == {
        "status": "closed",
        "messages": [{"role": "user", "parts": text("plan my week")}, {"role": "agent", "parts": text("draft one")},
                     {"role": "user", "parts": text("make it shorter")}, {"role": "agent", "parts": text("draft two")}],
        "a2a_tasks": [("<id1>", "completed", 0, 1), ("<id3>", "completed", 2, 3)],
    }
    assert result.prompt_responses[-1].response == "draft two"
    assert result.prompt_responses[-1].source_agent_per_turn_prompt_parts == [None, text("make it shorter")]
    assert _user_side_metadata(ctx) == {}


@pytest.mark.asyncio
async def test_malformed_user_sim_output_is_sent_on_as_raw_text(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two")], USER: [text("not json at all")]})
    ctx = _context(user_card=_USER_SIM_CARD)

    await _step(turns=2).execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text("draft one"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("not json at all"), "<id3>", "<ctx>", "task-3"),
    ]
    assert _stored(ctx, peers)["messages"][2] == {"role": "user", "parts": text("not json at all")}


@pytest.mark.asyncio
async def test_a_user_sim_reply_missing_done_is_malformed_too(monkeypatch):
    raw = json.dumps({"message": "shorter"})
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two")], USER: [text(raw)]})
    ctx = _context(user_card=_USER_SIM_CARD)

    await _step(turns=2).execute(ctx)

    assert peers.requests(ctx)[-2] == send(AGENT, text(raw), "<id3>", "<ctx>")
    assert _user_side_metadata(ctx) == {}


@pytest.mark.asyncio
async def test_extra_user_sim_fields_are_recorded_per_turn_and_not_sent(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two"), text("draft three")],
                                USER: [sim("shorter", mood="impatient", score=2), sim("good")]})
    ctx = _context(user_card=_USER_SIM_CARD)

    await _step(turns=3).execute(ctx)

    sent = [r for r in peers.requests(ctx) if r[1] == f"{AGENT}/a2a" and r[2]["method"] == "message/send"]
    assert sent == [
        send(AGENT, text("plan my week"), "<id1>", "<ctx>"),
        send(AGENT, text("shorter"), "<id3>", "<ctx>"),
        send(AGENT, text("good"), "<id5>", "<ctx>"),
    ]
    assert ctx.metadata["usersim_turn_outputs"] == {"talk": [{"turn": 1, "fields": {"mood": "impatient", "score": 2}}]}


@pytest.mark.asyncio
async def test_a_user_sim_that_times_out_ends_the_run_and_its_task_is_cancelled(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [HANG]})
    ctx = _context(user_card=_USER_SIM_CARD)

    result = await _step(user_agent_timeout_seconds=0).execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        send(USER, text("draft one"), "<id2>", "<ctx>", timeout=0),
        ("POST", f"{USER}/a2a", {"jsonrpc": "2.0", "id": "cancel", "method": "tasks/cancel",
                                 "params": {"id": "task-2"}}, 10),
    ]
    assert _stored(ctx, peers)["status"] == "closed"
    assert result.prompt_responses[-1].response == "draft one"


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["failed", "canceled", "rejected"])
async def test_a_user_side_task_that_does_not_complete_ends_the_run(monkeypatch, state):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [(state, sim("ignored"))]})
    ctx = _context(user_card=_USER_SIM_CARD)

    result = await _step().execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text("draft one"), "<id2>", "<ctx>", "task-2"),
    ]
    assert _stored(ctx, peers) == {
        "status": "closed",
        "messages": [{"role": "user", "parts": text("plan my week")}, {"role": "agent", "parts": text("draft one")}],
        "a2a_tasks": [("<id1>", "completed", 0, 1)],
    }
    assert result.prompt_responses[-1].response == "draft one"


@pytest.mark.asyncio
async def test_a_user_side_reply_with_no_parts_ends_the_run(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [[]]})
    ctx = _context(user_card=_USER_SIM_CARD)

    await _step().execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text("draft one"), "<id2>", "<ctx>", "task-2"),
    ]
    assert _stored(ctx, peers)["status"] == "closed"


@pytest.mark.asyncio
async def test_the_turn_limit_ends_the_run_after_the_agents_last_turn_without_asking_the_user(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("one"), text("two"), text("three")], USER: [sim("again"), sim("more")]})
    ctx = _context(user_card=_USER_SIM_CARD)

    result = await _step(turns=3).execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text("one"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("again"), "<id3>", "<ctx>", "task-3"),
        *exchange(USER, text("two"), "<id4>", "<ctx>", "task-4"),
        *exchange(AGENT, text("more"), "<id5>", "<ctx>", "task-5"),
    ]
    assert _stored(ctx, peers)["status"] == "closed"
    assert len(_stored(ctx, peers)["messages"]) == 6
    assert result.prompt_responses[-1].response == "three"


@pytest.mark.asyncio
async def test_a_conversation_closed_from_outside_ends_the_run_after_that_turn(monkeypatch):
    ctx = _context(user_card=_USER_SIM_CARD)

    def close_on_second_agent_turn(origin):
        if origin == AGENT and len(peers.tasks) == 3:
            conversation_store.mark_closed(ctx.metadata["a2a_conversations"]["talk"])

    peers = Peers(monkeypatch, {AGENT: [text("one"), text("two")], USER: [sim("again")]},
                  on_poll=close_on_second_agent_turn)

    await _step().execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text("one"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("again"), "<id3>", "<ctx>", "task-3"),
    ]
    assert _stored(ctx, peers)["messages"][-1] == {"role": "agent", "parts": text("two")}


@pytest.mark.asyncio
async def test_an_agent_failure_ends_the_run_without_asking_the_user_and_fails_the_step(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("one"), ("failed", text("boom"))], USER: [sim("again")]})
    ctx = _context(user_card=_USER_SIM_CARD)

    with pytest.raises(RuntimeError, match="A2A task failed"):
        await _step().execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text("one"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("again"), "<id3>", "<ctx>", "task-3"),
    ]
    assert _stored(ctx, peers) == {
        "status": "closed",
        "messages": [{"role": "user", "parts": text("plan my week")}, {"role": "agent", "parts": text("one")},
                     {"role": "user", "parts": text("again")}, {"role": "agent", "parts": text("boom")}],
        "a2a_tasks": [("<id1>", "completed", 0, 1), ("<id3>", "completed", 2, 3)],
    }


# ---- agent triggers -------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_fired_agent_trigger_speaks_for_the_user_that_turn(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("one"), text("two"), text("three")], USER: [sim("over", done=True)]},
                  decide=[{"fired": ["nudge"], "done": False, "parts": text("[school] pick-up moved to 3pm")},
                          {"fired": [], "done": False}])
    ctx = _context(user_card=_TRIGGERS_CARD, triggers=True)

    await _step().execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        ENV_STATE,
        decide(1, "one", "<ctx>"),
        *exchange(AGENT, text("[school] pick-up moved to 3pm"), "<id2>", "<ctx>", "task-2"),
        ENV_STATE,
        decide(2, "two", "<ctx>"),
        *exchange(USER, text("two"), "<id3>", "<ctx>", "task-3"),
        FIRING_LOG,
    ]
    assert _stored(ctx, peers)["messages"] == [
        {"role": "user", "parts": text("plan my week")}, {"role": "agent", "parts": text("one")},
        {"role": "user", "parts": text("[school] pick-up moved to 3pm")}, {"role": "agent", "parts": text("two")},
    ]
    snapshot = {"env": {"school-email": "fired", "reminder": "armed"}}
    assert _user_side_metadata(ctx) == {
        "env_trigger_snapshots": {"talk": [{"turn": 1, "envs": snapshot}, {"turn": 2, "envs": snapshot}]},
        "agent_trigger_firings": {"talk": [{"turn": 1, "fired": ["nudge"]}, {"turn": 2, "fired": []}]},
        "agent_trigger_state": {"talk": _FIRING_LOG["firing_log"]},
    }


@pytest.mark.asyncio
async def test_an_agent_trigger_that_ends_the_conversation_records_its_closing_message(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("one")], USER: []},
                  decide=[{"fired": ["wrap-up"], "done": True, "parts": text("[school] that's all for today")}])
    ctx = _context(user_card=_TRIGGERS_CARD, triggers=True)

    result = await _step().execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        ENV_STATE,
        decide(1, "one", "<ctx>"),
        FIRING_LOG,
    ]
    assert _stored(ctx, peers) == {
        "status": "closed",
        "messages": [{"role": "user", "parts": text("plan my week")}, {"role": "agent", "parts": text("one")},
                     {"role": "user", "parts": text("[school] that's all for today")}],
        "a2a_tasks": [("<id1>", "completed", 0, 1), ("<id2>", "canceled", 2, None)],
    }
    assert ctx.metadata["agent_trigger_firings"] == {"talk": [{"turn": 1, "fired": ["wrap-up"]}]}
    assert result.prompt_responses[-1].source_agent_per_turn_prompt_parts == [None]


@pytest.mark.asyncio
async def test_with_no_agent_trigger_fired_the_llm_user_sim_answers(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("one"), text("two")], USER: [sim("again")]},
                  decide=[{"fired": [], "done": False}])
    ctx = _context(user_card=_TRIGGERS_CARD, triggers=True)

    await _step(turns=2).execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        ENV_STATE,
        decide(1, "one", "<ctx>"),
        *exchange(USER, text("one"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("again"), "<id3>", "<ctx>", "task-3"),
        FIRING_LOG,
    ]
    assert ctx.metadata["agent_trigger_firings"] == {"talk": [{"turn": 1, "fired": []}]}


# ---- a human peer ---------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_human_peer_is_sent_the_agents_reply_as_it_is_and_its_reply_is_sent_on_unparsed(monkeypatch):
    reply = text(json.dumps({"message": "kept as typed", "done": True}))
    peers = Peers(monkeypatch, {AGENT: [text("one"), text("two")], HUMAN: [reply]})
    ctx = _context(user_card=_USER_SIM_CARD)

    await _step(turns=2, user_a2a_url=HUMAN).execute(ctx)

    assert peers.requests(ctx) == [
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(HUMAN, text("one"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, reply, "<id3>", "<ctx>", "task-3"),
    ]
    assert _stored(ctx, peers)["messages"][2] == {"role": "user", "parts": reply}


@pytest.mark.asyncio
async def test_with_no_user_agent_the_configured_human_peer_answers(monkeypatch):
    monkeypatch.setenv("AGENT_ENV_HUMAN_A2A_URL", HUMAN)
    peers = Peers(monkeypatch, {AGENT: [text("one"), text("two")], HUMAN: [text("go on")]})
    ctx = _context()

    await _step(turns=2).execute(ctx)

    assert peers.requests(ctx) == [
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(HUMAN, text("one"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("go on"), "<id3>", "<ctx>", "task-3"),
    ]
