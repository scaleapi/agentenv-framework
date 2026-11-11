"""prompt_agent with a user deployed by deploy_user_sim_agent: each agent reply is one read as always_on renders it, and
the user's outcome decides the next prompt or the end; a user it can't run as deployed is refused."""

from __future__ import annotations

import json

import pytest

from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps import user_sim_versions
from agent_env.task_step.task_steps.deploy_user_sim_agent import USER_SIMS_KEY
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.unit.task_step.test_deploy_user_sim_agent import _published, _step as _deploy_step
from tst.unit.task_step.test_prompt_agent_user_side import AGENT, USER, Peers, _stored, exchange, text

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
        *exchange(USER, text(reads[0]["input"]), "<id2>", "<ctx>", "task-2"),
        *exchange(AGENT, text("make it shorter"), "<id3>", "<ctx>", "task-3"),
        *exchange(USER, text(reads[1]["input"]), "<id4>", "<ctx>", "task-4"),
        *exchange(AGENT, text("perfect, thanks!"), "<id5>", "<ctx>", "task-5"),
    ]
    assert _stored(ctx, peers)["status"] == "closed"
    assert result.prompt_responses[-1].response == "glad to help"
    assert [(r["turn"], r["outcome"], r["message"], r["used"], r["retried"], r["turns"]) for r in reads] == [
        (1, "reply", "make it shorter", "D1", False, 1), (2, "done", "perfect, thanks!", "none", False, 1)]
    assert ctx.metadata["models"]["user simulator"]["talk"] == "sim-model"


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

    first, second = _reads(ctx)
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

    read = _reads(ctx)[0]
    assert peers.requests(ctx) == [
        *exchange(AGENT, text("plan my parents' visit"), "<id1>", "<ctx>", "task-1"),
        *exchange(USER, text(read["input"]), "<id2>", "<ctx>", "task-2"),
        *exchange(USER, text(user_sim_versions.RETRY), "<id3>", "<ctx>", "task-3"),
        *exchange(AGENT, text("shorter"), "<id4>", "<ctx>", "task-4"),
    ]
    assert (read["outcome"], read["retried"], read["turns"]) == ("reply", True, 2)
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
async def test_a_multi_day_user_sim_is_refused(monkeypatch):
    ctx = await _context(monkeypatch)
    ctx.metadata[USER_SIMS_KEY]["hannah"]["multi_day"] = True
    with pytest.raises(ValueError, match="multi_day True, so its prompt says the week goes on"):
        await _step().execute(ctx)


@pytest.mark.asyncio
async def test_a_user_output_format_is_refused(monkeypatch):
    ctx = await _context(monkeypatch)
    with pytest.raises(ValueError, match="drop user_output_format"):
        await _step(user_output_format={"type": "json_schema", "schema": {}}).execute(ctx)


@pytest.mark.asyncio
async def test_a_user_model_on_the_step_is_refused(monkeypatch):
    ctx = await _context(monkeypatch)
    with pytest.raises(ValueError, match="drop user_model from this step"):
        await _step(user_model="claude-sonnet-4-6").execute(ctx)


@pytest.mark.asyncio
async def test_a_prompt_other_than_the_deployed_opening_is_refused(monkeypatch):
    ctx = await _context(monkeypatch)
    step = PromptAgentTaskStep(id="talk", version=None, prompt="something else", agent_name="solver",
                               user_agent_name="hannah", poll_interval_seconds=0, max_conversation_turns=3)
    with pytest.raises(ValueError, match="isn't the opening_prompt its user 'hannah' was deployed with"):
        await step.execute(ctx)


@pytest.mark.asyncio
async def test_the_opening_is_compared_with_the_seed_filled_in(monkeypatch):
    Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("done")]})
    ctx = await _context(monkeypatch, opening_prompt="plan my parents' visit to <city>")
    ctx.metadata["seed"] = {"city": "Denver"}
    step = PromptAgentTaskStep(id="talk", version=None, prompt="plan my parents' visit to <city>", agent_name="solver",
                               user_agent_name="hannah", poll_interval_seconds=0, max_conversation_turns=3)
    await step.execute(ctx)
    assert _reads(ctx)[0]["outcome"] == "done"
