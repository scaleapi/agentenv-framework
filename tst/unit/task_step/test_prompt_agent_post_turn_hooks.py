"""prompt_agent's post_turn_hooks: what a hook is handed, how its answer is used, and the env clock on messages."""

from __future__ import annotations

import copy
import json
import sys
import textwrap

import httpx
import pytest
from agentenv_protocol.a2a_agent._triggers import TriggerEngine

from agent_env.artifact import FileArtifact
from agent_env.bundle import BundleKind, parse_bundle
from agent_env.bundle.installed import CORE, installed_bundles
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.entity.refs import ref_sites
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.gateway.constants import (
    EXT_CLOCK_URI, EXT_STEP_URI, EXT_TRIGGERS_URI, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH,
)
from agent_env.task_step.context import HEARTBEAT_KEY
from agent_env.task_step.task_steps.post_turn_hooks import PostTurnHooks
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.unit.task_step.test_prompt_agent_user_side import (
    AGENT, ENV, USER, USER_SIM_CONFIG, _TRIGGERS_CARD, _USER_SIM_CARD, Peers, _context, _stored, decide, exchange,
    send, sim, text,
)

pytestmark = pytest.mark.usefixtures("local_stores")

TUESDAY = "2026-03-03T09:00:00Z"
# A user agent whose triggers/v1 decide takes record_unfired, so a hook may call user_triggers() more than once a turn.
_QUIET_TRIGGERS_CARD = copy.deepcopy(_TRIGGERS_CARD)
_QUIET_TRIGGERS_CARD["capabilities"]["extensions"][1]["params"]["methods"]["decide"]["request"] = {
    "required": ["turn"], "optional": ["solver_message", "context_id", "env_triggers", "record_unfired"]}


class EnvPeers(Peers):
    """``Peers`` plus the env's clock and /step, so hooks and the env clock stamp have a gateway to read."""

    def __init__(self, monkeypatch, replies, *, clock: dict | None = None, engine: TriggerEngine | None = None, **kw):
        super().__init__(monkeypatch, replies, **kw)
        self.clock = clock or {"armed": False}
        self.engine = engine
        self.steps: list[tuple[str, dict]] = []

    def _answer(self, request: httpx.Request) -> httpx.Response:
        origin = f"{request.url.scheme}://{request.url.host}"
        if self.engine is not None and request.url.path == "/ext/triggers/decide":
            self.raw.append((request.method, str(request.url), json.loads(request.content), 600))
            return httpx.Response(200, json=self.engine.decide(**json.loads(request.content)))
        if origin == ENV and request.url.path == "/clock/state":
            self.raw.append((request.method, str(request.url), None, request.extensions["timeout"]["read"]))
            return httpx.Response(200, json=self.clock)
        if origin == ENV and request.url.path == "/step":
            self.steps.append((request.headers.get("AgentEnv-Role"), json.loads(request.content)))
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "changelog_id": 7})
        return super()._answer(request)


def _hook(hook_id: str, source: str) -> str:
    FileArtifact.put_bytes(hook_id, description=hook_id, filename="hook.py",
                           content=textwrap.dedent(source).encode())
    return hook_id


def _env(*uris: str, step_features: list | None = None) -> DeployedGatewayEnv:
    extensions = [dict(e) for e in GATEWAY_EXTENSIONS if e["uri"] in uris]
    for e in extensions:
        if e["uri"] == EXT_STEP_URI and step_features is not None:
            e["params"] = {**e["params"], "features": step_features}
    return DeployedGatewayEnv(env_id="env", env_version=1, gateway_url=ENV, mcp_url="", sandbox_id="sb-env",
                              environment_card_url=f"{ENV}{WELL_KNOWN_PATH}", instance_id="env-inst-1",
                              metadata={"deploy_step_id": "deploy_env"},
                              environment_card={"capabilities": {"extensions": extensions}})


def _step(*hooks: str, turns: int = 4, params: dict | None = None, **kw) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="talk", version=None, prompt="plan my week", agent_name="solver",
                               poll_interval_seconds=0, max_conversation_turns=turns,
                               post_turn_hooks=[{"file_artifact_id": h, "params": params or {}} for h in hooks], **kw)


def _recorded(ctx) -> list[dict]:
    return ctx.metadata.get("post_turn_hooks", {}).get("talk", [])


SAYS_NEXT = """
    async def post_turn(turn, params):
        turn.record({"number": turn.number, "last": turn.last, "reply": turn.agent_reply})
        return {"message": f"wake {turn.number}"}
"""


# ---- the field ------------------------------------------------------------------------------------------------------

def test_a_step_without_hooks_stores_no_hook_field():
    step = PromptAgentTaskStep(id="talk", version=None, prompt="hi")
    assert "post_turn_hooks" not in step.to_dict()


def test_hooks_round_trip_and_are_entity_refs_a_bundle_or_export_rewrites():
    hooks = [{"file_artifact_id": "next-env-event", "file_artifact_version": 3, "params": {"end_time": "P4D"}}]
    step = PromptAgentTaskStep(id="talk", version=None, prompt="hi", post_turn_hooks=hooks)

    stored = step.to_dict()

    assert PromptAgentTaskStep.from_dict(stored).post_turn_hooks == hooks == stored["post_turn_hooks"]
    [site] = [s for s in ref_sites(PromptAgentTaskStep.entity_refs, stored) if s.ref.field == "post_turn_hooks"]
    assert (site.path, site.value, site.version, site.version_key) == (
        "post_turn_hooks[0].file_artifact_id", "next-env-event", 3, "file_artifact_version")


def test_a_bundle_task_naming_a_hook_carries_the_hook_artifact_into_its_run():
    bundle = next(b for b in installed_bundles() if b.name == "next-env-event" and b.dist == CORE)

    plan = plan_bundle(resolve_bundle(parse_bundle(bundle.root, id_root=bundle.id_root)), installed=installed_bundles)

    assert (BundleKind.ARTIFACT, f"{bundle.id_root}/next-env-event") in [(w.kind, w.id) for w in plan.writes]


@pytest.mark.parametrize("hooks, fragment", [
    ("next-env-event", "must be a list"),
    (["next-env-event"], "post_turn_hooks[0] must be an object"),
    ([{"file_artifact_id": "h", "wake": "decide"}], "unknown key(s) ['wake']"),
    ([{"params": {}}], "file_artifact_id must be a non-empty string"),
    ([{"file_artifact_id": "h", "file_artifact_version": 0}], "file_artifact_version must be a positive integer"),
    ([{"file_artifact_id": "h", "params": []}], "params must be an object"),
])
def test_a_hook_list_the_step_cannot_load_is_refused_at_construction(hooks, fragment):
    with pytest.raises(ValueError) as e:
        PromptAgentTaskStep(id="talk", version=None, prompt="hi", post_turn_hooks=hooks)
    assert fragment in str(e.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["def post_turn(turn, params):\n    return None\n", "VALUE = 1\n"])
async def test_a_hook_that_is_not_an_async_post_turn_fails_the_step_before_the_agent_is_prompted(monkeypatch, source):
    peers = EnvPeers(monkeypatch, {AGENT: []})
    ctx = _context()

    with pytest.raises(ValueError, match="must define `async def post_turn"):
        await _step(_hook("sync", source)).execute(ctx)

    assert peers.raw == []


@pytest.mark.asyncio
async def test_a_hook_with_a_dataclass_and_postponed_annotations_loads_and_close_unregisters_it():
    source = """
        from __future__ import annotations
        from dataclasses import dataclass

        @dataclass
        class Plan:
            at: str

        async def post_turn(turn, params):
            return {"message": Plan("tue").at}
        """
    loaded = await PostTurnHooks.load([{"file_artifact_id": _hook("dataclass", source)}], "week")
    [name] = loaded._modules
    assert name in sys.modules and await loaded.hooks[0].post_turn(None, {}) == {"message": "tue"}
    loaded.close()
    assert name not in sys.modules


# ---- what a hook decides --------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_hook_message_is_the_agents_next_prompt_and_no_user_side_is_asked_or_looked_up(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one"), text("two"), text("three")]})
    ctx = _context()

    await _step(_hook("next", SAYS_NEXT), turns=3).execute(ctx)

    assert peers.requests(ctx) == [
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(AGENT, text("wake 1"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("wake 2"), "<id3>", "<ctx>", "task-3"),
    ]
    assert [(e["turn"], e["last"], e["reply"]) for e in _recorded(ctx)] == [
        (1, False, "one"), (2, False, "two"), (3, True, "three")]
    assert {e["hook"] for e in _recorded(ctx)} == {"next"}
    assert _stored(ctx, peers)["messages"][2] == {"role": "user", "parts": text("wake 1")}


@pytest.mark.asyncio
async def test_an_end_closes_the_conversation_recording_its_closing_message(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one")]})
    ctx = _context()
    hook = _hook("bye", """
        async def post_turn(turn, params):
            return {"end": "no_pending_events", "message": "nothing more this week"}
    """)

    await _step(hook).execute(ctx)

    assert peers.requests(ctx) == [*exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1")]
    stored = _stored(ctx, peers)
    assert stored["status"] == "closed"
    assert stored["messages"][-1] == {"role": "user", "parts": text("nothing more this week")}


@pytest.mark.asyncio
async def test_the_first_hook_to_decide_wins_and_one_returning_none_passes(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one"), text("two")]})
    ctx = _context()
    passes = _hook("passes", """
        async def post_turn(turn, params):
            turn.record({"saw": turn.number})
            return None
    """)
    later = _hook("later", """
        async def post_turn(turn, params):
            raise AssertionError("an earlier hook decided")
    """)

    await _step(passes, _hook("next", SAYS_NEXT), later, turns=2).execute(ctx)

    assert [e["hook"] for e in _recorded(ctx)] == ["passes", "next", "passes", "next"]
    assert peers.requests(ctx)[2] == send(AGENT, text("wake 1"), "<id2>", "<ctx>")


@pytest.mark.asyncio
async def test_with_no_hook_deciding_the_user_side_is_asked_as_without_hooks(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one"), text("two")], USER: [sim("again")]})
    ctx = _context(user_card=_USER_SIM_CARD)
    hook = _hook("quiet", "async def post_turn(turn, params):\n    return None\n")

    await _step(hook, turns=2).execute(ctx)

    assert peers.requests(ctx) == [
        USER_SIM_CONFIG,
        *exchange(AGENT, text("plan my week"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text("one"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("again"), "<id3>", "<ctx>", "task-3"),
    ]


@pytest.mark.asyncio
async def test_a_hook_that_asks_the_user_reaches_the_configured_human_peer(monkeypatch):
    monkeypatch.setenv("AGENT_ENV_HUMAN_A2A_URL", "http://human.test")
    peers = EnvPeers(monkeypatch, {AGENT: [text("one"), text("two")], "http://human.test": [text("go on")]})
    ctx = _context()
    hook = _hook("asks", """
        async def post_turn(turn, params):
            return None if turn.last else await turn.ask_user()
    """)

    await _step(hook, turns=2).execute(ctx)

    assert peers.requests(ctx)[2:5] == [
        *exchange("http://human.test", text("one"), "<id2>", "<ctx>", "task-2"),
        send(AGENT, text("go on"), "<id3>", "<ctx>"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("returned, fragment", [
    ("'wake'", "returned 'wake'"),
    ("{'message': 'x', 'mood': 'calm'}", "returned {'message': 'x', 'mood': 'calm'}"),
    ("{'end': ''}", "the reason is a string"),
    ("{'message': []}", "a message is text or a non-empty list of A2A parts"),
    ("{'message': [{'kind': 'nope'}]}", "message part 0 is not an A2A Part"),
])
async def test_a_hook_returning_something_else_fails_the_step(monkeypatch, returned, fragment):
    EnvPeers(monkeypatch, {AGENT: [text("one")]})
    hook = _hook("odd", f"async def post_turn(turn, params):\n    return {returned}\n")

    with pytest.raises(RuntimeError) as e:
        await _step(hook).execute(_context())
    assert fragment in str(e.value)


@pytest.mark.asyncio
async def test_a_message_returned_after_the_last_turn_is_ignored(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one")]})
    ctx = _context()

    result = await _step(_hook("next", SAYS_NEXT), turns=1).execute(ctx)

    assert [(e["turn"], e["last"]) for e in _recorded(ctx)] == [(1, True)]
    assert len(peers.requests(ctx)) == 2 and result.prompt_responses[-1].response == "one"


# ---- asking the user side once --------------------------------------------------------------------------------------

CHECKS_TRIGGERS_THEN_PASSES = """
    async def post_turn(turn, params):
        if turn.last:
            return None
        answer = await turn.user_triggers()
        turn.record({"triggers": answer})
        return answer
"""


@pytest.mark.asyncio
async def test_agent_triggers_that_fire_for_a_hook_answer_once_and_ask_user_returns_the_same(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one"), text("two")]},
                     decide=[{"fired": ["nudge"], "done": False, "parts": text("[school] pick-up moved")}])
    ctx = _context(user_card=_QUIET_TRIGGERS_CARD, triggers=True)
    hook = _hook("both", """
        async def post_turn(turn, params):
            if turn.last:
                return None
            fired = await turn.user_triggers()
            assert await turn.ask_user() == fired and await turn.user_triggers() == fired
            return fired
    """)

    await _step(hook, turns=2).execute(ctx)

    assert [r for r in peers.requests(ctx) if "decide" in r[1]] == [quiet(decide(1, "one", "<ctx>"))]
    assert peers.requests(ctx)[-3] == send(AGENT, text("[school] pick-up moved"), "<id2>", "<ctx>")
    assert ctx.metadata["agent_trigger_firings"] == {"talk": [{"turn": 1, "fired": ["nudge"]}]}


@pytest.mark.asyncio
async def test_agent_triggers_that_do_not_fire_leave_no_trace_and_the_user_side_is_then_asked_afresh(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one"), text("two")], USER: [sim("again")]},
                     decide=[{"fired": [], "done": False}, {"fired": [], "done": False}])
    ctx = _context(user_card=_QUIET_TRIGGERS_CARD, triggers=True)

    await _step(_hook("checks", CHECKS_TRIGGERS_THEN_PASSES), turns=2).execute(ctx)

    assert [r for r in peers.requests(ctx) if "decide" in r[1]] == [quiet(decide(1, "one", "<ctx>")),
                                                                    decide(1, "one", "<ctx>")]
    assert ctx.metadata["agent_trigger_firings"] == {"talk": [{"turn": 1, "fired": []}]}
    assert len(ctx.metadata["env_trigger_snapshots"]["talk"]) == 1
    assert _recorded(ctx) == [{"hook": "checks", "turn": 1, "triggers": None}]


def quiet(request: tuple) -> tuple:
    method, url, body, timeout = request
    return method, url, {**body, "record_unfired": False}, timeout


@pytest.mark.asyncio
async def test_a_once_trigger_fires_once_however_often_a_hook_checks_the_agent_triggers(monkeypatch):
    engine = TriggerEngine()
    engine.register({"triggers": [{"id": "julia", "when": {"type": "env_trigger", "env_id": "env",
                                                            "trigger_id": "school-email", "status": "fired"},
                                   "actions": [{"type": "say", "text": "[Julia] did you see the school email?"}]}]})
    peers = EnvPeers(monkeypatch, {AGENT: [text("one"), text("two"), text("three")], USER: [sim("ok")]}, engine=engine)
    ctx = _context(user_card=_QUIET_TRIGGERS_CARD, triggers=True)
    hook = _hook("twice", """
        async def post_turn(turn, params):
            if turn.last:
                return None
            return await turn.user_triggers() or await turn.user_triggers()
    """)

    await _step(hook, turns=3).execute(ctx)

    sent = [r[2]["params"]["message"]["parts"] for r in peers.requests(ctx) if r[1] == f"{AGENT}/a2a"
            and r[2]["method"] == "message/send"]
    assert sent == [text("plan my week"), text("[Julia] did you see the school email?"), text("ok")]
    assert [e["kind"] for e in engine.state()["firing_log"]] == ["registered", "fired"]


@pytest.mark.asyncio
async def test_a_user_agent_whose_decide_records_every_call_is_refused_before_its_once_triggers_reset(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one")]})
    ctx = _context(user_card=_TRIGGERS_CARD, triggers=True)

    with pytest.raises(RuntimeError, match="takes record_unfired"):
        await _step(_hook("checks", CHECKS_TRIGGERS_THEN_PASSES), turns=2).execute(ctx)
    assert not [r for r in peers.requests(ctx) if "decide" in r[1]]


@pytest.mark.asyncio
async def test_a_hook_passing_on_an_agent_trigger_s_end_records_its_closing_message_even_when_empty(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one")]}, decide=[{"fired": ["wrap-up"], "done": True}])
    ctx = _context(user_card=_QUIET_TRIGGERS_CARD, triggers=True)

    await _step(_hook("checks", CHECKS_TRIGGERS_THEN_PASSES)).execute(ctx)

    stored = _stored(ctx, peers)
    assert stored["status"] == "closed" and stored["messages"][-1] == {"role": "user", "parts": []}
    assert _recorded(ctx) == [{"hook": "checks", "turn": 1, "triggers": {"end": "agent_trigger_done", "message": []}}]


@pytest.mark.asyncio
async def test_a_hook_asking_the_user_twice_at_once_asks_once(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one"), text("two")], USER: [sim("again")]})
    ctx = _context(user_card=_USER_SIM_CARD)
    hook = _hook("eager", """
        import asyncio

        async def post_turn(turn, params):
            if turn.last:
                return None
            first, second = await asyncio.gather(turn.ask_user(), turn.ask_user())
            assert first == second
            return first
    """)

    await _step(hook, turns=2).execute(ctx)

    assert [r[1] for r in peers.requests(ctx)].count(f"{USER}/a2a") == 2


@pytest.mark.asyncio
async def test_a_hook_that_asked_and_returns_another_answer_fails_the_step(monkeypatch):
    EnvPeers(monkeypatch, {AGENT: [text("one")], USER: [sim("again")]})
    hook = _hook("liar", """
        async def post_turn(turn, params):
            await turn.ask_user()
            return {"message": "something else"}
    """)

    with pytest.raises(RuntimeError, match="asked the user side at turn 1 and must return its answer"):
        await _step(hook).execute(_context(user_card=_USER_SIM_CARD))


# ---- what a hook is handed ------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_hook_is_handed_the_turn_the_seed_a_safe_copy_of_the_metadata_and_the_envs(monkeypatch):
    EnvPeers(monkeypatch, {AGENT: [[{"kind": "text", "text": "one"}, {"kind": "data", "data": {"n": 1}}]]})
    ctx = _context()
    ctx.deployed_envs.append(_env(EXT_CLOCK_URI, EXT_TRIGGERS_URI))
    ctx.metadata.update(seed={"city": "Lyon"}, user_overrides={"litellm_api_key": "sk-test", "agent_effort": "low"})
    beats = []
    ctx.metadata[HEARTBEAT_KEY] = lambda: beats.append(1)
    hook = _hook("looks", """
        async def post_turn(turn, params):
            try:
                turn.metadata["x"] = 1
                writable = True
            except TypeError:
                writable = False
            [env] = turn.envs
            turn.heartbeat()
            turn.record({"params": params, "parts": turn.agent_parts, "seed": turn.seed, "writable": writable,
                         "overrides": turn.metadata["user_overrides"], "env": [env.env_id, env.env_step_id,
                         env.instance_id, env.supports("urn:agentenv:clock/v1", "advance")]})
            return None
    """)

    await _step(hook, turns=1, params={"end_time": "P4D"}).execute(ctx)

    [entry] = _recorded(ctx)
    assert entry["params"] == {"end_time": "P4D"} and entry["seed"] == {"city": "Lyon"}
    assert entry["parts"] == [{"kind": "text", "text": "one"}, {"kind": "data", "data": {"n": 1}}]
    assert entry["writable"] is False and entry["overrides"] == {"agent_effort": "low"}
    assert entry["env"] == ["env", "deploy_env", "env-inst-1", True] and beats == [1]


@pytest.mark.asyncio
async def test_a_hook_calls_a_tool_through_step_as_the_role_it_names_tagged_as_the_hooks(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one")]})
    ctx = _context()
    ctx.deployed_envs.append(_env(EXT_STEP_URI))
    hook = _hook("mailer", """
        async def post_turn(turn, params):
            [env] = turn.envs
            turn.record({"result": await env.call_tool("send_email", {"to": "a@example.com"}, role="school")})
            return None
    """)

    await _step(hook, turns=1).execute(ctx)

    assert peers.steps == [("school", {"action": "call_tool", "tool_name": "send_email",
                                       "arguments": {"to": "a@example.com"}, "source": "post_turn_hook",
                                       "hook_artifact_id": "mailer"})]
    assert _recorded(ctx)[0]["result"]["changelog_id"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("call, fragment", [
    ("env.call_tool('send_email', {}, role='')", "needs the AgentEnv-Role"),
    ("env.invoke('urn:agentenv:step/v1', 'step', {})", "call tools with call_tool"),
])
async def test_a_hook_env_refuses_a_call_with_no_role_or_outside_clock_and_triggers(monkeypatch, call, fragment):
    EnvPeers(monkeypatch, {AGENT: [text("one")]})
    ctx = _context()
    ctx.deployed_envs.append(_env(EXT_STEP_URI))
    hook = _hook("bad", f"async def post_turn(turn, params):\n    [env] = turn.envs\n    await {call}\n")

    with pytest.raises(ValueError, match=fragment):
        await _step(hook, turns=1).execute(ctx)


@pytest.mark.asyncio
async def test_a_tool_call_on_a_gateway_that_cannot_tag_it_as_the_hooks_is_refused(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one")]})
    ctx = _context()
    ctx.deployed_envs.append(_env(EXT_STEP_URI, step_features=[]))
    hook = _hook("mailer", "async def post_turn(turn, params):\n    await turn.envs[0].call_tool('x', {}, 'r')\n")

    with pytest.raises(RuntimeError, match="rebuild its gateway image"):
        await _step(hook, turns=1).execute(ctx)
    assert peers.steps == []


# ---- the env clock on stored messages -------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_every_stored_message_carries_each_armed_env_clock(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one"), text("two")]},
                     clock={"armed": True, "virtual_time": TUESDAY, "t0": TUESDAY})
    ctx = _context()
    ctx.deployed_envs.append(_env(EXT_CLOCK_URI))
    hook = _hook("bye", """
        async def post_turn(turn, params):
            return {"message": "wake"} if turn.number == 1 else {"end": "done", "message": "bye"}
    """)

    await _step(hook, turns=3).execute(ctx)

    stamp = [{"env_id": "env", "env_step_id": "deploy_env", "virtual_time": TUESDAY}]
    messages = _stored(ctx, peers)["messages"]
    assert [m["parts"] for m in messages] == [text("plan my week"), text("one"), text("wake"), text("two"), text("bye")]
    assert all(m["env_clock"] == stamp for m in messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("armed", [False, True])
async def test_without_hooks_an_env_offering_a_clock_adds_clock_reads_and_nothing_else(monkeypatch, armed):
    real_client = httpx.AsyncClient

    def run(full_card: bool):
        monkeypatch.setattr(httpx, "AsyncClient", real_client)
        clock = {"armed": True, "virtual_time": TUESDAY, "t0": TUESDAY} if armed else None
        peers = EnvPeers(monkeypatch, {AGENT: [text("one"), text("two")], USER: [sim("over", done=True)]},
                         decide=[{"fired": ["nudge"], "done": False, "parts": text("[school] pick-up moved to 3pm")},
                                 {"fired": [], "done": False}], clock=clock)
        ctx = _context(user_card=_TRIGGERS_CARD, triggers=True)
        if full_card:
            ctx.deployed_envs[0].environment_card = {"capabilities": {"extensions": GATEWAY_EXTENSIONS}}
        return peers, ctx

    base_peers, base_ctx = run(full_card=False)
    await PromptAgentTaskStep(id="talk", version=None, prompt="plan my week", agent_name="solver",
                              poll_interval_seconds=0, max_conversation_turns=4).execute(base_ctx)
    peers, ctx = run(full_card=True)
    await PromptAgentTaskStep(id="talk", version=None, prompt="plan my week", agent_name="solver",
                              poll_interval_seconds=0, max_conversation_turns=4).execute(ctx)

    requests = peers.requests(ctx)
    assert [r for r in requests if r[1] != f"{ENV}/clock/state"] == base_peers.requests(base_ctx)
    assert [r for r in requests if r[1] == f"{ENV}/clock/state"] == [("GET", f"{ENV}/clock/state", None, 5)] * 4
    stored, base = _stored(ctx, peers)["messages"], _stored(base_ctx, base_peers)["messages"]
    stamp = [{"env_id": "env", "env_step_id": None, "virtual_time": TUESDAY}]
    assert [{k: v for k, v in m.items() if k != "env_clock"} for m in stored] == base
    assert [m.get("env_clock") for m in stored] == [stamp if armed else None] * len(base)


@pytest.mark.asyncio
async def test_with_no_armed_env_clock_stored_messages_carry_none(monkeypatch):
    peers = EnvPeers(monkeypatch, {AGENT: [text("one")]})
    ctx = _context()
    ctx.deployed_envs.append(_env(EXT_CLOCK_URI))

    await PromptAgentTaskStep(id="talk", version=None, prompt="plan my week", agent_name="solver",
                              poll_interval_seconds=0).execute(ctx)

    assert all("env_clock" not in m for m in _stored(ctx, peers)["messages"])
