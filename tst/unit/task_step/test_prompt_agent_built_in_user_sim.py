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
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.verifiers.agent_prompt_response_verifier import AgentPromptResponseVerifierTaskStep
from tst.unit.task_step.test_deploy_user_sim_agent import HANNAH, _published, _step as _deploy_step
from tst.unit.task_step.test_prompt_agent_user_side import AGENT, HANG, HUMAN, USER, Peers, _stored, exchange, text

pytestmark = pytest.mark.usefixtures("local_stores")

_CONFIG_CARD = {"capabilities": {"extensions": [{"uri": "urn:agentenv:agent-config/v1", "params": {
    "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": ["output_format", "model"]}}}}}]}}


PROMPT = "plan my parents' visit"


def said(outcome: str, message: str = "", used: str = "none") -> list[dict]:
    return text(json.dumps({"outcome": outcome, "message": message, "used": used}))


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
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [("failed", said("reply", "ignored"))]})
    ctx = await _context(monkeypatch)

    await _step().execute(ctx)

    assert len(peers.requests(ctx)) == 4
    assert _reads(ctx)[0]["user_sim_error"] == "state failed"
    assert _stored(ctx, peers)["status"] == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize("user,timeout,error", [(("failed", said("reply", "ignored")), 600, "state failed"),
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

    async def fetch(self, url, ext, task_id, prefix, name, *, sandbox_type, card=None):
        fetched.append((url, task_id, prefix, sandbox_type))
        return f"{prefix}{name}.json"

    monkeypatch.setattr(PromptAgentTaskStep, "_fetch_trajectory", fetch)
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
