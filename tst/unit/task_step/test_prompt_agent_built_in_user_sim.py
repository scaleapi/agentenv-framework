"""prompt_agent with a user deployed by deploy_user_sim_agent: each agent reply is one read as always_on renders it, and
the user's outcome decides the next prompt or the end; a user it can't run as deployed is refused."""

from __future__ import annotations

import json

import pytest

from agent_env.a2a_agent import A2AAgent, conversation_store
from agent_env.store import Filter, get_config
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent import user_sim_versions
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import USER_SIMS_KEY
from agent_env.task_step.task_steps import prompt_agent as pa
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep, StoredTrajectory
from agent_env.task_step.task_steps.verifiers.agent_prompt_response_verifier import AgentPromptResponseVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.user_sim_judge import _chat_reads
from tst.unit.task_step.test_deploy_user_sim_agent import HANNAH, _published, _step as _deploy_step
from tst.unit.task_step.test_prompt_agent_user_side import (
    _SWAP, AGENT, HANG, HUMAN, USER, Peers, _stored, exchange, pattern_judge, text)  # noqa: F401

pytestmark = pytest.mark.usefixtures("local_stores", "pattern_judge")

_CONFIG_CARD = {"capabilities": {"extensions": [{"uri": "urn:agentenv:agent-config/v1", "params": {
    "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": ["output_format", "model"]}}}}}]}}


PROMPT = "plan my parents' visit"


def said(outcome: str, message: str = "", used: str = "none") -> list[dict]:
    return text(json.dumps({"outcome": outcome, "message": message, "used": used}))


# A user sim task that failed as an agent reports it: its error in a data part.
FAILED = ("failed", [{"kind": "data", "data": {"error_type": "api_error", "error_message": "the model was refused"}}])
FAILED_ERROR = "state failed: error_type=api_error: the model was refused"


async def _context(monkeypatch, **deployed) -> TaskStepContext:
    """A run where the real deploy_user_sim_agent step recorded hannah, who answers at USER."""
    record = await _published(monkeypatch, _deploy_step(**{"multi_day": False, "opening_prompt": PROMPT, **deployed}))
    ctx = TaskStepContext(instance_id="ti-1")
    ctx.deployed_agents += [
        DeployedAgent(agent_name="solver", api_url=AGENT, a2a_url=AGENT, a2a_card={}),
        DeployedAgent(agent_name="hannah", api_url=USER, a2a_url=USER, sandbox_id="sb-user", sandbox_type="local",
                      a2a_card=_CONFIG_CARD)]
    ctx.metadata[USER_SIMS_KEY] = {"hannah": record}
    return ctx


def _step(turns: int = 5, **kw) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="talk", version=None, prompt=PROMPT, agent_name="solver",
                               user_agent_name="hannah", poll_interval_seconds=0, max_conversation_turns=turns, **kw)


def _reads(ctx: TaskStepContext) -> list[dict]:
    return ctx.metadata["user_sim_reads"]["talk"]


def _full(read: dict) -> dict:
    """The whole read, as stored at the summary's object_url."""
    return json.loads(get_config().get_object_store().get(read["object_url"]))


@pytest.mark.asyncio
async def test_a_reply_is_the_next_prompt_and_a_done_with_text_reaches_the_agent_whose_answer_ends_the_run(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two"), text("glad to help")],
                                USER: [said("reply", "make it shorter", "D1"), said("done", "perfect, thanks!")]})
    ctx = await _context(monkeypatch)

    result = await _step().execute(ctx)

    reads = _reads(ctx)
    # Neither agent-config nor anything else goes to the user sim: deploy_user_sim_agent set its model and format.
    assert peers.requests(ctx) == [
        *exchange(AGENT, text("plan my parents' visit"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text(_full(reads[0])["input"]), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("make it shorter"), "<id3>", "<ctx>", "task-3"),
        *exchange(USER, text(_full(reads[1])["input"]), "<id4>", "<ctx>", "task-4"),
        *exchange(AGENT, text("perfect, thanks!"), "<id5>", "<ctx>", "task-5"),
    ]
    assert _stored(ctx, peers)["status"] == "closed"
    assert result.prompt_responses[-1].response == "glad to help"
    assert [(r["turn"], r["outcome"], r["message"], r["used"], r["retried"], r["user_sim_turns"]) for r in reads] == [
        (1, "reply", "make it shorter", "D1", False, 1), (2, "done", "perfect, thanks!", "none", False, 1)]
    assert ctx.metadata["models"]["user simulator"]["talk"] == "sim-model"
    assert result.prompt_responses[-1].error_type is None


@pytest.mark.asyncio
async def test_each_read_is_rendered_as_always_on_renders_it_at_the_task_clock_with_the_opening_in_the_first_only(
        monkeypatch):
    Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two"), text("draft three")],
                        USER: [said("reply", "shorter"), said("no_reply")]})
    ctx = await _context(monkeypatch)
    record = ctx.metadata[USER_SIMS_KEY]["hannah"]

    async def armed_clock(self, envs):  # one armed clock, standing still at 9 AM
        return {"env_clock": [{"env_id": "env", "env_step_id": "clock", "virtual_time": "2026-11-16T09:00:00+00:00"}]}

    monkeypatch.setattr(PromptAgentTaskStep, "_env_clock", armed_clock)

    await _step().execute(ctx)

    first, second = map(_full, _reads(ctx))
    assert first["read"]["time"] == second["read"]["time"] == "2026-11-16T09:00:00Z"
    assert first["read"]["you_sent"] == [{"at": "2026-11-16T09:00:00Z", "text": "plan my parents' visit"}]
    assert second["read"]["you_sent"] == []
    for r in (first, second):
        assert r["read"]["phone"] == [] and r["read"]["done_ends_run"] is True
    assert first["read"]["assistant_messages"] == [{"at": first["read"]["time"], "text": "draft one"}]
    person = {**ao.person_of(record), "time_zone": record["time_zone"] or "UTC", "done_ends_run": True}
    assert first["input"] == user_sim_versions.render(person, [], first["read"])
    assert second["input"] == user_sim_versions.render(
        person, [{"read": first["read"], "answer": {"outcome": "reply", "message": "shorter"}}], second["read"])


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [said("done"), said("no_reply"), said("reply", "  ")])
async def test_a_done_without_text_or_no_reply_ends_the_run_at_once(monkeypatch, answer):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [answer]})
    ctx = await _context(monkeypatch)

    result = await _step().execute(ctx)

    assert [r[1] for r in peers.requests(ctx)] == [f"{AGENT}/a2a"] * 2 + [f"{USER}/a2a"] * 2
    assert _stored(ctx, peers)["messages"] == [{"role": "user", "parts": text("plan my parents' visit")},
                                               {"role": "agent", "parts": text("draft one")}]
    assert result.prompt_responses[-1].response == "draft one"


@pytest.mark.asyncio
async def test_a_malformed_answer_is_retried_once_in_the_same_session(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two")],
                                USER: [text("shorter, obviously"), said("reply", "shorter")]})
    ctx = await _context(monkeypatch)

    await _step(turns=2).execute(ctx)

    summary = _reads(ctx)[0]
    read = _full(summary)
    assert peers.requests(ctx) == [
        *exchange(AGENT, text("plan my parents' visit"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text(read["input"]), "<id2>", "<ctx>", "task-2"),
        *exchange(USER, text(user_sim_versions.RETRY), "<id3>", "<ctx>", "task-3"),
        *exchange(AGENT, text("shorter"), "<id4>", "<ctx>", "task-4"),
    ]
    assert (summary["outcome"], summary["retried"], summary["user_sim_turns"]) == ("reply", True, 2)
    assert summary["first_answer"] == {"schema_errors": ["the answer is not a JSON object"]}
    assert read["raw"] == said("reply", "shorter")[0]["text"]
    assert read["first_answer"] == {"raw": "shorter, obviously", "schema_errors": ["the answer is not a JSON object"]}


@pytest.mark.asyncio
async def test_a_user_sim_turn_that_does_not_complete_ends_the_run_and_is_recorded(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [FAILED]})
    ctx = await _context(monkeypatch)

    await _step().execute(ctx)

    assert len(peers.requests(ctx)) == 4
    assert _reads(ctx)[0]["user_sim_error"] == FAILED_ERROR
    assert _stored(ctx, peers)["status"] == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize("user,timeout,error", [(FAILED, 600, FAILED_ERROR),
                                                (HANG, 0, "timeout: ")], ids=["failed", "timeout"])
async def test_a_user_sim_failure_marks_the_run_as_an_error_so_verifiers_skip_it(monkeypatch, user, timeout, error):
    Peers(monkeypatch, {AGENT: [text("draft one")], USER: [user]})
    ctx = await _context(monkeypatch)
    step = _step(user_agent_timeout_seconds=timeout)

    await step.execute(ctx)

    response = ctx.prompt_responses[-1]
    assert (response.response, response.error_type) == ("draft one", "user_sim_error")
    assert response.error_message.startswith(f"The user simulator failed: {error}")
    verifier = AgentPromptResponseVerifierTaskStep(id="check", version=None, prompt_id=step.prompt_id, verifier_id="v",
                                                   criteria=[{"type": "response_contains", "needles": ["draft one"]}])
    await verifier.execute(ctx)
    assert ctx.metadata["verifications"]["v"]["results"][0]["id"] == "prompt_error"


@pytest.mark.asyncio
async def test_running_out_of_turns_before_done_ends_after_the_agents_last_answer_with_no_error(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two")], USER: [said("reply", "shorter")]})
    ctx = await _context(monkeypatch)

    result = await _step(turns=2).execute(ctx)

    assert [r[1] for r in peers.requests(ctx)] == [f"{AGENT}/a2a"] * 2 + [f"{USER}/a2a"] * 2 + [f"{AGENT}/a2a"] * 2
    assert [r["outcome"] for r in _reads(ctx)] == ["reply"]
    assert _stored(ctx, peers)["status"] == "closed"
    assert (result.prompt_responses[-1].response, result.prompt_responses[-1].error_type) == ("draft two", None)


@pytest.mark.asyncio
async def test_a_user_a2a_url_beside_a_built_in_user_takes_the_old_path_unchecked(monkeypatch):
    """The human at user_a2a_url answers; the deployed user's record isn't read, so even its refused settings pass."""
    peers = Peers(monkeypatch, {AGENT: [text("one"), text("two")], HUMAN: [text("go on")]})
    ctx = await _context(monkeypatch)
    ctx.metadata[USER_SIMS_KEY]["hannah"]["multi_day"] = True

    await _step(turns=2, user_a2a_url=HUMAN).execute(ctx)

    assert peers.requests(ctx) == [
        *exchange(AGENT, text(PROMPT), "<id1>", "<ctx>", "task-1"),
        *exchange(HUMAN, text("one"), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("go on"), "<id3>", "<ctx>", "task-3"),
    ]
    assert "user_sim_reads" not in ctx.metadata
    assert "user simulator" not in ctx.metadata.get("models", {})


@pytest.mark.asyncio
async def test_the_run_keeps_a_small_summary_of_each_read_and_stores_the_whole_read(monkeypatch):
    long_reply = "x" * 100_000
    Peers(monkeypatch, {AGENT: [text(long_reply), text(long_reply), text("ok")],
                        USER: [said("reply", "shorter", "D1"), said("done", "thanks")]})
    ctx = await _context(monkeypatch)

    await _step().execute(ctx)

    reads = _reads(ctx)
    assert [sorted(r) for r in reads] == [sorted([
        "turn", "outcome", "message", "used", "score", "schema_errors", "retried", "user_sim_error", "user_sim_turns",
        "user_sim_trajectory_urls", "object_url"])] * 2
    # Neither the agent's reply nor the rendered input is kept, so the summary doesn't grow with them.
    assert all(len(json.dumps(r)) < 1_000 for r in reads)
    full = _full(reads[1])
    assert long_reply in full["input"] and full["read"]["assistant_messages"][0]["text"] == long_reply


@pytest.mark.asyncio
async def test_each_user_sim_turn_records_its_trajectory_as_always_on_does(monkeypatch):
    Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("done")]})
    ctx = await _context(monkeypatch)
    user = next(a for a in ctx.deployed_agents if a.agent_name == "hannah")
    user.a2a_card = {"capabilities": {"extensions": [*_CONFIG_CARD["capabilities"]["extensions"],
                                                     {"uri": A2AAgent.EXT_TRAJECTORY, "params": {}}]}}
    fetched = []

    async def fetch(url, ext, task_id, prefix, name, *, sandbox_type, headers, card=None):
        fetched.append((url, task_id, prefix, sandbox_type))
        return StoredTrajectory(f"{prefix}{name}.json")

    monkeypatch.setattr(pa, "fetch_native_trajectory", fetch)
    step = _step()

    await step.execute(ctx)

    # always_on's prefix for the user sim's trajectories, joined the same way
    assert fetched == [(USER, "task-2", step._trajectory_prefix() + "user_sim/", "local")]
    [url] = _reads(ctx)[0]["user_sim_trajectory_urls"]
    assert url.startswith(step._trajectory_prefix() + "user_sim/") and url.endswith(".json")


_SCHEDULE = {"wake": "07:00", "sleep": "23:00"}


@pytest.mark.asyncio
@pytest.mark.parametrize("deployed,metadata,step,problem", [
    ({"multi_day": True}, {}, {}, "multi_day True, so its prompt says the week goes on"),
    ({}, {}, {"user_output_format": {"type": "json_schema", "schema": {}}}, "drop user_output_format"),
    ({}, {}, {"user_model": "claude-sonnet-4-6"}, "drop user_model from this step and set model there"),
    ({}, {"user_overrides": {"step_params": {"talk": {"user_model": "claude-sonnet-4-6"}}}}, {},
     "drop user_model from this step's step_params"),
    ({}, {}, {"prompt": "something else"}, "isn't the opening_prompt its user 'hannah' was deployed with"),
    ({"persona": {**HANNAH, "schedule": _SCHEDULE}}, {}, {}, "with persona schedule, which one sitting ignores"),
    ({"persona": {**HANNAH, "schedule": _SCHEDULE, "response_times": {"free": 5}}}, {}, {},
     "with persona schedule and response_times, which one sitting ignores"),
    ({"persona": {**HANNAH, "time_zone": "America/Chicago"}},
     {"clock_configurations": [{"step_id": "clock", "timezone": "America/New_York"}]}, {},
     r"^prompt_agent 'talk': user sim 'hannah' is in America/Chicago but sync_env_clock set \['America/New_York'\]$"),
], ids=["multi_day", "user_output_format", "user_model", "step_params_user_model", "prompt", "schedule",
        "response_times", "time_zone"])
async def test_a_refused_setting_fails_before_the_conversation_starts(monkeypatch, deployed, metadata, step, problem):
    ctx = await _context(monkeypatch, **{k: v for k, v in deployed.items() if k != "multi_day"})
    if "multi_day" in deployed:
        ctx.metadata[USER_SIMS_KEY]["hannah"]["multi_day"] = deployed["multi_day"]
    ctx.metadata.update(metadata)
    peers = Peers(monkeypatch, {})
    fields = {"max_conversation_turns": 3, **step}
    with pytest.raises(ValueError, match=problem):
        await PromptAgentTaskStep(id="talk", version=None, prompt=fields.pop("prompt", PROMPT), agent_name="solver",
                                  user_agent_name="hannah", poll_interval_seconds=0, **fields).execute(ctx)
    assert peers.raw == []  # nothing sent to the agent or the user sim, no agent-config either
    assert "a2a_conversations" not in ctx.metadata
    assert conversation_store._doc_store().query(conversation_store._COLLECTION_NAME,
                                                 Filter.of(task_instance_id="ti-1")) == []


@pytest.mark.asyncio
async def test_the_refusal_comes_before_the_hooks_load(monkeypatch):
    ctx = await _context(monkeypatch)
    ctx.metadata[USER_SIMS_KEY]["hannah"]["multi_day"] = True
    with pytest.raises(ValueError, match="multi_day True"):  # not the missing hook artifact
        await _step(post_turn_hooks=[{"file_artifact_id": "no-such-hook", "params": {}}]).execute(ctx)


@pytest.mark.asyncio
async def test_the_opening_is_compared_with_the_seed_filled_in(monkeypatch):
    Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("done")]})
    ctx = await _context(monkeypatch, opening_prompt="plan my parents' visit to <city>")
    ctx.metadata["seed"] = {"city": "Denver"}
    step = PromptAgentTaskStep(id="talk", version=None, prompt="plan my parents' visit to <city>", agent_name="solver",
                               user_agent_name="hannah", poll_interval_seconds=0, max_conversation_turns=3)
    await step.execute(ctx)
    assert _reads(ctx)[0]["outcome"] == "done"


@pytest.mark.asyncio
async def test_after_max_replies_the_last_reply_reaches_the_agent_then_the_run_ends(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two")],
                                USER: [said("reply", "make it shorter")]})
    ctx = await _context(monkeypatch, persona={**HANNAH, "max_replies": 1})

    result = await _step().execute(ctx)

    assert [r[1] for r in peers.requests(ctx)] == [f"{AGENT}/a2a"] * 2 + [f"{USER}/a2a"] * 2 + [f"{AGENT}/a2a"] * 2
    assert len(_reads(ctx)) == 1
    assert _stored(ctx, peers)["status"] == "closed"
    assert result.prompt_responses[-1].response == "draft two"


@pytest.mark.asyncio
async def test_an_opening_split_across_text_parts_is_compared_joined(monkeypatch):
    Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("done")]})
    ctx = await _context(monkeypatch, opening_prompt="plan my parents' visit\nthey land friday")
    step = PromptAgentTaskStep(id="talk", version=None, agent_name="solver", user_agent_name="hannah",
                               parts=[{"kind": "text", "text": "plan my parents' visit"},
                                      {"kind": "text", "text": "they land friday"}],
                               poll_interval_seconds=0, max_conversation_turns=3)
    await step.execute(ctx)
    assert _full(_reads(ctx)[0])["read"]["you_sent"][0]["text"] == "plan my parents' visit\nthey land friday"


@pytest.mark.asyncio
@pytest.mark.parametrize("persona_extra,user", [({"max_replies": 1}, [said("reply", "make it shorter")]),
                                                ({}, [said("done", "make it shorter")])])
async def test_the_agents_answer_to_the_users_last_message_still_runs_the_hooks_as_the_last_turn(
        monkeypatch, persona_extra, user):
    from tst.unit.task_step.test_prompt_agent_post_turn_hooks import _hook, _recorded
    hook = _hook("records", """
        async def post_turn(turn, params):
            turn.record({"number": turn.number, "last": turn.last, "reply": turn.agent_reply})
            return None
        """)
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two")], USER: user})
    ctx = await _context(monkeypatch, persona={**HANNAH, **persona_extra})

    await _step(post_turn_hooks=[{"file_artifact_id": hook, "params": {}}]).execute(ctx)

    assert [(r["number"], r["last"], r["reply"]) for r in _recorded(ctx)] == [
        (1, False, "draft one"), (2, True, "draft two")]
    assert [r[1] for r in peers.requests(ctx)].count(f"{USER}/a2a") == 2  # asked once


@pytest.mark.asyncio
@pytest.mark.parametrize("persona_extra,user", [({"max_replies": 1}, [said("reply", "make it shorter")]),
                                                ({}, [said("done", "make it shorter")])])
async def test_a_hook_that_asks_the_user_on_the_last_turn_gets_the_end_without_a_user_sim_request(
        monkeypatch, persona_extra, user):
    from tst.unit.task_step.test_prompt_agent_post_turn_hooks import _hook, _recorded
    hook = _hook("asks", """
        async def post_turn(turn, params):
            answer = await turn.ask_user()
            turn.record({"number": turn.number, "last": turn.last, "answer": answer})
            return answer
        """)
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two")], USER: user})
    ctx = await _context(monkeypatch, persona={**HANNAH, **persona_extra})

    await _step(post_turn_hooks=[{"file_artifact_id": hook, "params": {}}]).execute(ctx)

    end = "user_gave_up" if persona_extra else "user_done"
    assert [(r["number"], r["last"], r["answer"]) for r in _recorded(ctx)] == [
        (1, False, {"message": text("make it shorter"), "end_after": end}), (2, True, {"end": end})]
    assert [r[1] for r in peers.requests(ctx)].count(f"{USER}/a2a") == 2  # asked once
    assert len(_reads(ctx)) == 1
    assert _stored(ctx, peers)["status"] == "closed"


@pytest.mark.asyncio
async def test_the_user_reads_every_text_part_of_the_agents_reply(monkeypatch):
    Peers(monkeypatch, {AGENT: [text("Blue house, $280 a night.") + text("Book it?")], USER: [said("done")]})
    ctx = await _context(monkeypatch)

    await _step().execute(ctx)

    [entry] = _reads(ctx)
    stored = _full(entry)
    assert "Blue house, $280 a night." in stored["input"] and "Book it?" in stored["input"]


@pytest.mark.asyncio
async def test_agent_triggers_registered_for_the_built_in_user_are_refused_before_the_chat_starts(monkeypatch):
    from agent_env.env.env import DeployedGatewayEnv
    from agent_env.env.gateway.constants import EXT_TRIGGERS_URI, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
    from tst.unit.task_step.test_prompt_agent_user_side import ENV, _TRIGGERS_CARD

    Peers(monkeypatch, {AGENT: [text("Blue house or green?"), text("Booked the blue one."), text("Enjoy!")],
                        USER: [said("done", "great, thanks")]},
          decide=[{"fired": ["pick"], "done": False, "parts": text("blue please")}, {"fired": [], "done": False}])
    ctx = await _context(monkeypatch)
    hannah = next(a for a in ctx.deployed_agents if a.agent_name == "hannah")
    hannah.a2a_card = _TRIGGERS_CARD
    ctx.deployed_envs.append(DeployedGatewayEnv(
        env_id="env", env_version=1, gateway_url=ENV, mcp_url="", sandbox_id="sb-env",
        environment_card_url=f"{ENV}{WELL_KNOWN_PATH}",
        environment_card={"capabilities": {"extensions": [e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_TRIGGERS_URI]}}))
    ctx.metadata["agent_trigger_registrations"] = [{"step_id": "reg", "agent_name": "hannah", "added": ["pick"]}]

    with pytest.raises(ValueError, match="agent triggers are registered for it; scripted user lines with the built-in "
                                         "user sim aren't supported yet"):
        await _step().execute(ctx)
    assert "user_sim_reads" not in ctx.metadata


# ---- the user-sim gate ----------------------------------------------------------------------------------------------
def _gated(turns: int = 3, **gate) -> PromptAgentTaskStep:
    return _step(turns=turns, user_sim_gate={"patterns": True, "judge_model": "judge", **gate})


def _to(peers: Peers, ctx: TaskStepContext, url: str) -> list[str]:
    sent = [r for r in peers.requests(ctx) if r[2] and r[2].get("method") == "message/send"]
    return [r[2]["params"]["message"]["parts"][0]["text"] for r in sent if r[1].startswith(url)]


@pytest.mark.asyncio
async def test_the_gate_holds_back_a_built_in_users_answer_that_reads_as_the_assistants_and_she_answers_again(
        monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two"), text("you're welcome")],
                                USER: [said("reply", _SWAP), said("reply", "do those numbers hold up?", "D1"),
                                       said("done", "great, thanks")]})
    ctx = await _context(monkeypatch)

    await _gated().execute(ctx)

    to_user, to_agent = _to(peers, ctx, USER), _to(peers, ctx, AGENT)
    assert to_user[1].startswith("[Conversation system note, not from the assistant.] Your last message was not sent")
    assert to_agent == ["plan my parents' visit", "do those numbers hold up?", "great, thanks"]  # no swap reached it
    verdicts = ctx.metadata["user_sim_gate"]["talk"]
    assert [(v["turn"], v["attempt"], v["passed"]) for v in verdicts] == [(1, 0, False), (1, 1, True), (2, 0, True)]
    first = _reads(ctx)[0]
    # As in always_on: the read holds what reached the agent, and keeps what she first answered for whoever grades her.
    assert (first["message"], first["first_message"]) == ("do those numbers hold up?", _SWAP)
    assert [(r["outcome"], r["message"], r["used"]) for r in first["gate_rewrites"]] == [
        ("reply", "do those numbers hold up?", "D1")]
    assert "user_sim_invalid" not in ctx.metadata
    # The stored read says the same, and the user-sim judge, which reads it over the summary, grades what was sent.
    stored = _full(first)
    assert (stored["message"], stored["used"], stored["first_message"]) == ("do those numbers hold up?", "D1", _SWAP)
    assert stored["gate_rewrites"] == first["gate_rewrites"] and stored["raw"] is not None
    judged, _ = _chat_reads(_reads(ctx))
    assert judged[0]["message"] == "do those numbers hold up?"


@pytest.mark.asyncio
async def test_a_built_in_users_answer_that_still_fails_fails_the_step_as_a_failed_user_sim(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("reply", _SWAP)] * 3})
    ctx = await _context(monkeypatch)

    with pytest.raises(RuntimeError, match=r"user_sim_gate: user-sim message at turn 1 still reads as the assistant's"):
        await _gated().execute(ctx)

    assert [v["passed"] for v in ctx.metadata["user_sim_gate"]["talk"]] == [False, False, False]
    assert ctx.metadata["user_sim_invalid"]["talk"]["turn"] == 1
    assert len(_reads(ctx)[0]["gate_rewrites"]) == 2
    # Nothing reached the agent, so the read is a no_reply with no message, as in always_on.
    read = _reads(ctx)[0]
    assert (read["outcome"], read["message"], read["first_message"]) == ("no_reply", "", _SWAP)
    # A failed user sim: verifiers skip the run and it is redone, rather than graded as the agent's answer.
    assert ctx.prompt_responses[-1].error_type == "user_sim_error"
    assert ctx.prompt_responses[-1].error_message.startswith(
        "The user simulator failed: user_sim_gate: user-sim message at turn 1")
    assert _to(peers, ctx, AGENT) == ["plan my parents' visit"]   # nothing she wrote reached the agent


@pytest.mark.asyncio
async def test_a_rewrite_that_says_nothing_ends_the_conversation_as_her_no_reply(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("reply", _SWAP), said("no_reply")]})
    ctx = await _context(monkeypatch)

    await _gated().execute(ctx)

    assert _to(peers, ctx, AGENT) == ["plan my parents' visit"]
    assert "user_sim_invalid" not in ctx.metadata
    assert [(v["attempt"], v["passed"]) for v in ctx.metadata["user_sim_gate"]["talk"]] == [(0, False)]


@pytest.mark.asyncio
async def test_a_done_rewritten_into_a_reply_counts_toward_max_replies(monkeypatch):
    # Her first answer is a done, which isn't counted; its rewrite is a reply, which is, and it is her last.
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two")],
                                USER: [said("done", _SWAP), said("reply", "do those numbers hold up?")]})
    ctx = await _context(monkeypatch, persona={**HANNAH, "max_replies": 1})

    result = await _gated().execute(ctx)

    assert _to(peers, ctx, AGENT) == ["plan my parents' visit", "do those numbers hold up?"]
    assert len(_to(peers, ctx, USER)) == 2         # her read and the rewrite: she isn't asked again past the limit
    assert result.prompt_responses[-1].response == "draft two"
    assert result.prompt_responses[-1].error_type is None


@pytest.mark.asyncio
async def test_a_reply_rewritten_into_a_done_no_longer_counts(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two"), text("draft three")],
                                USER: [said("reply", _SWAP), said("reply", "shorter, please"),
                                       said("reply", "and cheaper")]})
    ctx = await _context(monkeypatch, persona={**HANNAH, "max_replies": 2})

    await _gated().execute(ctx)

    # The held reply and its rewrite count once, so her second reply is the one that reaches the limit.
    assert _to(peers, ctx, AGENT) == ["plan my parents' visit", "shorter, please", "and cheaper"]


@pytest.mark.asyncio
async def test_a_built_in_user_sim_is_sent_its_saved_tool_settings_again_and_nothing_else(monkeypatch):
    # deploy_user_sim_agent set her model and format; her own tools and peer tools, stamped on her card by deploy, go
    # again, so a user sim that restarted since doesn't get its tools back.
    Peers(monkeypatch, {AGENT: [text("draft one"), text("glad to help")], USER: [said("done", "perfect")]})
    ctx = await _context(monkeypatch)
    hannah = next(a for a in ctx.deployed_agents if a.agent_name == "hannah")
    card = json.loads(json.dumps(_CONFIG_CARD))
    card["capabilities"]["extensions"][0]["params"]["methods"]["set"]["request"]["supported"] += [
        "builtin_tools", "peer_tools"]
    hannah.a2a_card = {**card, "builtin_tools": {"disable": "*"}, "peer_tools": False}
    posted = []

    async def post_agent_config(url, payload, timeout_seconds=60, *, headers=None):
        posted.append((url, payload))

    # _context's deploy step left its own recorder in place of post_agent_config; record this run's posts instead.
    monkeypatch.setattr("agent_env.a2a_agent.protocol.post_agent_config", post_agent_config)

    await _step().execute(ctx)

    assert [p for url, p in posted if url == f"{USER}/ext/agent-config"] == [
        {"builtin_tools": {"disable": "*"}, "peer_tools": False}]
    assert all(not url.startswith(USER) or set(p) == {"builtin_tools", "peer_tools"} for url, p in posted)


@pytest.mark.asyncio
async def test_a_hook_note_is_in_the_users_read_after_the_reply_and_never_reaches_the_agent(monkeypatch):
    from tst.unit.task_step.test_prompt_agent_post_turn_hooks import _hook
    hook = _hook("notes", """
        async def post_turn(turn, params):
            return None if turn.last else await turn.ask_user(note="The sheet has no total.")
        """)
    peers = Peers(monkeypatch, {AGENT: [text("Sheet's in the folder."), text("Added it.")],
                                USER: [said("reply", "can it have a total?")]})
    ctx = await _context(monkeypatch)

    await _step(turns=2, post_turn_hooks=[{"file_artifact_id": hook, "params": {}}]).execute(ctx)

    stored = _full(_reads(ctx)[0])
    assert stored["read"]["user_note"] == "The sheet has no total."
    # recorded where the peer path records it, so there's one place to look for what the user sim was told
    assert [n["note"] for n in ctx.metadata["user_side_notes"]["talk"]] == ["The sheet has no total."]
    rendered = stored["input"]
    assert rendered.index("Sheet's in the folder.") < rendered.index(
        "WHAT YOU FOUND WHEN YOU CHECKED ITS WORK YOURSELF (the assistant doesn't see this):\n  The sheet has no total.")
    sent_to_agent = [r[2]["params"]["message"]["parts"] for r in peers.requests(ctx) if r[1] == f"{AGENT}/a2a"
                     and r[2].get("method") == "message/send"]
    assert all("no total" not in json.dumps(parts) for parts in sent_to_agent)


@pytest.mark.asyncio
async def test_the_judge_reads_the_built_in_users_own_instructions(monkeypatch):
    from agent_env.task_step.task_steps.user_sim_gate import UserSimGate

    seen = []

    async def judge(self, candidate, history, call_config, instructions=None, **_):
        seen.append(instructions)
        return {"in_role": True, "reason": "asks"}

    monkeypatch.setattr(UserSimGate, "_judge", judge)
    Peers(monkeypatch, {AGENT: [text("draft one"), text("you're welcome")],
                        USER: [said("done", "great, thanks")]})
    ctx = await _context(monkeypatch)

    await _gated().execute(ctx)

    assert len(seen) == 1 and seen[0] and "Hannah" in seen[0]


# ---- a milestone user keeps replying while a hook's note is in play -------------------------------------------------
NOTE_HOOK = """
    async def post_turn(turn, params):
        return None if turn.last else await turn.ask_user(note="check who the reply went to.")
"""


def _noted(turns: int = 3) -> PromptAgentTaskStep:
    from tst.unit.task_step.test_prompt_agent_post_turn_hooks import _hook
    return _step(turns=turns, post_turn_hooks=[{"file_artifact_id": _hook("noted", NOTE_HOOK), "params": {}}])


def _to_agent(peers, ctx) -> list[str]:
    return [r[2]["params"]["message"]["parts"][0]["text"] for r in peers.requests(ctx)
            if r[1] == f"{AGENT}/a2a" and r[2].get("method") == "message/send"]


def _asked_user(peers, ctx) -> list[str]:
    return [r[2]["params"]["message"]["parts"][0]["text"] for r in peers.requests(ctx)
            if r[1] == f"{USER}/a2a" and r[2].get("method") == "message/send"]


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["milestone", "chat"])
async def test_a_user_who_always_answers_and_says_no_reply_is_asked_again_in_the_answers_own_format(
        monkeypatch, preset):
    peers = Peers(monkeypatch, {AGENT: [text("Sent it to Leonard."), text("Added Arthur."), text("ok")],
                                USER: [said("no_reply"), said("reply", "who else should be on it?"),
                                       said("reply", "thanks")]})
    ctx = await _context(monkeypatch, preset=preset)

    result = await _noted().execute(ctx)

    assert _to_agent(peers, ctx)[:2] == ["plan my parents' visit", "who else should be on it?"]
    person = {"preset": preset}
    assert _asked_user(peers, ctx)[1] == user_sim_versions.retry_line(person)  # not one of reply | done: format retry
    assert '"no_reply"' not in user_sim_versions.retry_line(person)
    first = _reads(ctx)[0]
    assert (first["outcome"], first["retried"]) == ("reply", True)
    assert "user_sim_note_retries" not in ctx.metadata and result.prompt_responses[-1].error_type is None


@pytest.mark.asyncio
async def test_a_milestone_user_who_keeps_answering_no_reply_to_a_note_ends_the_run_as_a_user_sim_error(monkeypatch):
    # The format retry first, then the guard's tries, each with its own format retry: none of them a reply.
    Peers(monkeypatch, {AGENT: [text("Sent it to Leonard.")], USER: [said("no_reply")] * (2 + 2 * pa.NOTE_REPLY_TRIES)})
    ctx = await _context(monkeypatch, preset="milestone")

    result = await _noted().execute(ctx)

    response = result.prompt_responses[-1]
    assert response.error_type == "user_sim_error" and "no_reply" in response.error_message
    assert len(ctx.metadata["user_sim_note_retries"]["talk"]) == pa.NOTE_REPLY_TRIES


@pytest.mark.asyncio
async def test_without_a_preset_a_no_reply_to_a_note_still_ends_the_run(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("Sent it to Leonard.")], USER: [said("no_reply")]})
    ctx = await _context(monkeypatch)

    result = await _noted().execute(ctx)

    assert _to_agent(peers, ctx) == ["plan my parents' visit"]
    assert "user_sim_note_retries" not in ctx.metadata and result.prompt_responses[-1].error_type is None


@pytest.mark.asyncio
async def test_a_milestone_users_gate_rewrite_that_comes_back_as_no_reply_is_asked_again(monkeypatch):
    from tst.unit.task_step.test_prompt_agent_post_turn_hooks import _hook
    peers = Peers(monkeypatch, {AGENT: [text("Sent it to Leonard."), text("Added Arthur."), text("ok")],
                                USER: [said("reply", _SWAP), said("no_reply"), said("reply", "who else should be on it?"),
                                       said("reply", "thanks")]})
    ctx = await _context(monkeypatch, preset="milestone")
    hook = _hook("noted-gated", NOTE_HOOK)

    step = _step(turns=3, user_sim_gate={"patterns": True, "judge_model": "judge"},
                 post_turn_hooks=[{"file_artifact_id": hook, "params": {}}])
    result = await step.execute(ctx)

    # The gate held the swap back; its rewrite came back as no_reply, which a milestone user may not give, so it was
    # asked again in the answer's own format, and the turn went on.
    assert _to_agent(peers, ctx)[:2] == ["plan my parents' visit", "who else should be on it?"]
    assert _asked_user(peers, ctx)[2] == user_sim_versions.retry_line({"preset": "milestone"})
    assert "user_sim_invalid" not in ctx.metadata and result.prompt_responses[-1].error_type is None
