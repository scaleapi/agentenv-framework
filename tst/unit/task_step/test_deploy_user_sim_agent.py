"""The deploy_user_sim_agent step: the prompt it sets on the user's agent is the tested one byte for byte, and so are
the reads always_on renders from what it records; its checks when the task loads; it fails closed on a card that
won't take the prompt or the model; the run-wide agent pick never reaches it; its model's precedence; and always_on
with a user it deployed (prompt_agent's are in test_prompt_agent_built_in_user_sim.py)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent import protocol
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent import user_sim_versions
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import USER_SIMS_KEY, DeployUserSimAgentTaskStep
from tst.unit.task_step.always_on_harness import USER_SIM, Rig

DATA = Path(__file__).parents[2] / "data" / "user_sim"
PERSON = json.loads((DATA / "hannah_person.json").read_text())
PROMPT = (DATA / "hannah_system_prompt.txt").read_text()
RUNS = json.loads((DATA / "hannah_reads.json").read_text())["runs"]
HANNAH = {"display_name": PERSON["name"], "first_name": PERSON["first"], "description": PERSON["description"],
          "permissions": PERSON["permissions"], "keep_private": PERSON["keep_private"]}
DESCRIPTION = {"who_you_are": ["Linda Murphy, planning Noor's 30th in Mexico City."]}
AGENT_URL = "https://user-sim.example"


def _config_card(*supported: str) -> dict:
    return {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
        "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": list(supported)}}}}}]}}


TAKES_ALL = _config_card("name", "model", "system_prompt", "output_format")


def _step(**fields) -> DeployUserSimAgentTaskStep:
    return DeployUserSimAgentTaskStep.from_dict({
        "id": "user", "type": "deploy_user_sim_agent", "a2a_agent_id": "sim-image", "agent_name": "hannah",
        "model": "sim-model", "multi_day": True, "opening_prompt": PERSON["opening"]["text"], "persona": HANNAH, **fields})


class _Deploy:
    """Runs a deploy step against an agent double whose card is ``card``; ``posts`` has each agent-config post."""

    def __init__(self, monkeypatch, card: dict = TAKES_ALL):
        self.posts: list[tuple[str, dict]] = []
        self.agent = MagicMock(metadata={})
        self.agent.close = AsyncMock()
        self.agent.deploy = AsyncMock(return_value=MagicMock(a2a_url=AGENT_URL, agent_card=card, sandbox_type=None))

        async def post(url, fields):
            self.posts.append((url, fields))

        monkeypatch.setattr(protocol, "post_agent_config", post)

    async def run(self, step: DeployAgentTaskStep, context: TaskStepContext | None = None) -> TaskStepContext:
        context = context or TaskStepContext()
        with (patch.object(A2AAgent, "get", return_value=self.agent),
              patch("agent_env.task_step.task_steps.deploy_agent.get_config") as get_config):
            get_config.return_value.get_model_for_role.return_value = None
            return await step.execute(context)

    def fields(self, name: str) -> list:
        return [fields[name] for _, fields in self.posts if name in fields]


async def _published(monkeypatch, step: DeployUserSimAgentTaskStep, context: TaskStepContext | None = None) -> dict:
    """What ``step`` records for always_on."""
    context = await _Deploy(monkeypatch).run(step, context)
    return context.metadata[USER_SIMS_KEY][step.agent_name]


# ---- the prompt and the reads, byte for byte ------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_the_prompt_set_on_the_user_s_agent_is_the_tested_one_byte_for_byte(monkeypatch):
    deploy = _Deploy(monkeypatch)
    context = await deploy.run(_step())

    assert deploy.fields("system_prompt") == [PROMPT]
    assert deploy.fields("model") == ["sim-model"]
    assert deploy.fields("output_format") == [user_sim_versions.OUTPUT_FORMAT]
    record = context.metadata[USER_SIMS_KEY]["hannah"]
    assert record["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
    assert (record["prompt_set"], record["multi_day"], record["user_version"]) == (True, True, "ours")
    assert record["setup"]["system_prompt_chars"] == len(PROMPT)
    assert record["setup"]["accepted_fields"] == ["model", "output_format", "system_prompt"]
    assert [a.agent_name for a in context.deployed_agents] == ["hannah"]
    assert context.deployed_agents[0].env_ids == []  # no env tools unless the task gives some
    assert context.metadata["models"]["user simulator"]["user"] == "sim-model"


@pytest.mark.asyncio
async def test_a_one_sitting_user_gets_the_done_is_a_sign_off_line(monkeypatch):
    deploy = _Deploy(monkeypatch)
    await deploy.run(_step(multi_day=False))
    [prompt] = deploy.fields("system_prompt")
    assert user_sim_versions.DONE_ENDS_RUN in prompt and user_sim_versions.DONE_MULTI_DAY not in prompt
    assert prompt == PROMPT.replace(user_sim_versions.DONE_MULTI_DAY, user_sim_versions.DONE_ENDS_RUN)


@pytest.mark.asyncio
async def test_the_seed_fills_the_opening_and_nothing_else(monkeypatch):
    deploy = _Deploy(monkeypatch)
    persona = {**HANNAH, "description": {**HANNAH["description"], "how_you_write": ["You write <city> in lowercase."]}}
    context = TaskStepContext()
    context.metadata["seed"] = {"city": "sf"}
    await deploy.run(_step(opening_prompt="sort <city> for us", persona=persona), context)
    [prompt] = deploy.fields("system_prompt")
    assert '"sort sf for us"' in prompt and "You write <city> in lowercase." in prompt


@pytest.mark.asyncio
async def test_the_reads_always_on_renders_from_the_record_are_the_tested_ones(monkeypatch):
    record = await _published(monkeypatch, _step())
    us = {**record}
    person = {**ao.person_of(us),
              "time_zone": ao._task_zone(us, [{"timezone": PERSON["time_zone"]}])}
    for data in RUNS.values():
        for k, read in enumerate(data["reads"]):
            history = [{"read": r, "answer": user_sim_versions.parse(a)}
                       for r, a in zip(data["reads"][:k], data["answers"][:k])]
            assert user_sim_versions.render(person, history, read) == data["expected"][k]


# ---- the checks when the task loads ---------------------------------------------------------------------------------
def test_the_step_round_trips_and_holds_no_system_prompt():
    step = _step()
    assert DeployUserSimAgentTaskStep.from_dict(step.to_dict()).to_dict() == step.to_dict()
    assert "system_prompt" not in step.to_dict() and step.env_ids == []


@pytest.mark.parametrize("fields, problem", [
    ({"system_prompt": "You are Hannah."}, "system_prompt is written from the persona"),
    ({"agent_name": None}, "agent_name is required"),
    ({"model": None}, "model is required"),
    ({"multi_day": None}, "multi_day must be true or false"),
        ({"persona": {**HANNAH, "time_zone": "Pacific"}}, "is not an IANA time zone"),
({"opening_prompt": ""}, "opening_prompt is required"),
    ({"opening_prompt": None}, "opening_prompt is required"),
    ({"persona": "Hannah."}, "persona must be an object"),
    ({"persona": {**HANNAH, "display_name": ""}}, "persona.display_name is required"),
    ({"persona": {**HANNAH, "first_name": " "}}, "first_name must be the name the user goes by"),
    ({"persona": {**HANNAH, "description": None}}, "description must be an object of sections"),
    ({"persona": {**HANNAH, "description": {"who_you_are": ["Linda."], "events": ["Tue"]}}},
     "unknown sections ['events']"),
    ({"persona": {**HANNAH, "permissions": {"maybe": ["book it"]}}}, "permissions has unknown kinds ['maybe']"),
    ({"persona": {**HANNAH, "keep_private": "her age"}}, "keep_private must be a list of non-empty lines"),
    ({"persona": {**HANNAH, "schedule": ["07:30", "23:30"]}}, "schedule must be an object"),
    ({"persona": {**HANNAH, "schedule": {"wake": "7am"}}}, "schedule.wake must be a time like 07:30"),
    ({"persona": {**HANNAH, "schedule": {"lunch": "12:00"}}}, "schedule has unknown keys ['lunch']"),
    ({"persona": {**HANNAH, "schedule": {"work": ["09:00"]}}}, "schedule.work must be [start, end]"),
    ({"persona": {**HANNAH, "schedule": {"work_days": ["monday"]}}}, "schedule.work_days must be a list of"),
    ({"persona": {**HANNAH, "schedule": {"busy": [{"start": "Thu 9am", "end": "Thu 6pm"}]}}},
     "schedule.busy must be a list of {start, end}"),
    ({"persona": {**HANNAH, "schedule": {"wake": "07:00"}, "response_times": {"work": -5}}},
     "response_times.work must be a number of minutes"),
    ({"persona": {**HANNAH, "schedule": {"wake": "07:00"}, "response_times": {"evening": 30}}},
     "response_times has unknown keys ['evening']"),
    ({"persona": {**HANNAH, "max_replies": 0}}, "max_replies must be a whole number"),
    ({"persona": {**HANNAH, "max_replies": True}}, "max_replies must be a whole number"),
    ({"persona": {**HANNAH, "response_times": {"work": 30}}}, "response_times needs a schedule"),
    ({"persona": {**HANNAH, "response_times": {"work": 30}, "schedule": {}}}, "response_times needs a schedule"),
])
def test_a_bad_setting_is_refused_when_the_task_loads(fields, problem):
    with pytest.raises(ValueError, match="deploy_user_sim_agent '") as e:
        _step(**fields)
    assert problem in str(e.value)


def test_the_tested_schedule_loads():
    _step(persona={**HANNAH, "schedule": {"wake": "07:30", "sleep": "23:30", "work": ["09:00", "18:00"],
                                          "work_days": ["mon", "tue", "wed", "thu", "fri"],
                                          "busy": [{"start": "2026-11-12T09:00", "end": "2026-11-12T18:00"}]},
                   "response_times": {"work": 150, "free": 30, "morning": 15}, "max_replies": 40})


# ---- it plays the user, never the agent under test -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_its_agent_is_deployed_and_recorded_exactly_as_deploy_agent_deploys_one(monkeypatch):
    """Everything deploy_agent gives an agent, the user sim's gets: the same deployed_agents entry and the same posts,
    with only its model and answer format set after."""
    shared = {"a2a_agent_id": "sim-image", "agent_name": "hannah", "role": "user", "agent_description": "the user",
              "env_vars": {"A": "1"}}
    card = _config_card("name", "model", "system_prompt", "output_format", "role", "description")
    user_sim_step = _step(**shared)
    runs = []
    for step in (user_sim_step, DeployAgentTaskStep("deploy", None, system_prompt=PROMPT, **shared)):
        deploy = _Deploy(monkeypatch, card=json.loads(json.dumps(card)))
        deployed = deploy.agent.deploy.return_value
        deployed.sandbox_id, deployed.instance_id, deployed.network_policy = "sb-1", "i-1", None
        runs.append((await deploy.run(step), deploy))
    (sim_ctx, sim_deploy), (plain_ctx, plain_deploy) = runs

    assert sim_ctx.deployed_agents == plain_ctx.deployed_agents
    assert sim_deploy.agent.deploy.call_args == plain_deploy.agent.deploy.call_args
    assert sim_deploy.posts[:-1] == plain_deploy.posts
    assert set(sim_deploy.posts[-1][1]) == {"model", "output_format"}


def test_a_run_wide_agent_pick_swaps_the_tested_agent_but_not_the_user_sim():
    context = TaskStepContext()
    context.metadata["user_overrides"] = {"a2a_agent_id": "candidate", "a2a_agent_version": 7}
    solver = DeployAgentTaskStep("deploy", None, a2a_agent_id="stored", agent_name="solver")
    assert solver._resolve_agent_ref(context) == ("candidate", 7)
    assert _step(a2a_agent_version=2)._resolve_agent_ref(context) == ("sim-image", 2)


def test_the_user_sim_s_own_step_params_still_pick_its_agent():
    context = TaskStepContext()
    context.metadata["user_overrides"] = {"a2a_agent_id": "candidate",
                                          "step_params": {"user": {"a2a_agent_id": "other-sim"}}}
    assert _step(a2a_agent_version=2)._resolve_agent_ref(context) == ("other-sim", None)
    context.metadata["user_overrides"]["step_params"]["user"] = {"a2a_agent_version": 5}
    assert _step(a2a_agent_version=2)._resolve_agent_ref(context) == ("sim-image", 5)


@pytest.mark.parametrize("overrides, model, source", [
    ({}, "sim-model", "user_sim_step"),
    ({"user_model": "run-model"}, "run-model", "run_user_model"),
    ({"user_model": "run-model", "step_params": {"user": {"model": "step-model"}}}, "step-model",
     "user_sim_step_params"),
], ids=["the step's", "the run's", "the step's step_params"])
@pytest.mark.asyncio
async def test_the_model_s_precedence_and_its_recorded_source(monkeypatch, overrides, model, source):
    deploy = _Deploy(monkeypatch)
    context = TaskStepContext()
    context.metadata["user_overrides"] = overrides
    context = await deploy.run(_step(), context)
    assert deploy.fields("model") == [model]
    record = context.metadata[USER_SIMS_KEY]["hannah"]
    assert (record["model"], record["model_source"]) == (model, source)
    assert (record["setup"]["model"], record["setup"]["model_from"], record["setup"]["user_sim_step_model"]) == (
        model, source, "sim-model")


# ---- it fails closed --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("card", [{}, _config_card("name", "model")], ids=["no-agent-config", "no-system-prompt"])
@pytest.mark.asyncio
async def test_a_card_that_won_t_take_the_prompt_fails_the_step(monkeypatch, card):
    deploy = _Deploy(monkeypatch, card)
    context = TaskStepContext()
    with pytest.raises(RuntimeError, match="can't set its system prompt"):
        await deploy.run(_step(), context)
    assert deploy.fields("system_prompt") == [] and deploy.agent.close.await_count == 1
    assert context.deployed_agents == [] and USER_SIMS_KEY not in context.metadata


@pytest.mark.parametrize("card", [{}, _config_card("name", "model")], ids=["no-agent-config", "no-system-prompt"])
@pytest.mark.asyncio
async def test_plain_deploy_agent_still_only_warns(monkeypatch, caplog, card):
    deploy = _Deploy(monkeypatch, card)
    with caplog.at_level(logging.WARNING, logger="agent_env.task_step.task_steps.deploy_agent"):
        context = await deploy.run(DeployAgentTaskStep("deploy", None, a2a_agent_id="a", agent_name="solver",
                                                       system_prompt="You are helpful."))
    assert [a.agent_name for a in context.deployed_agents] == ["solver"]
    assert any("skipping" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_a_card_that_won_t_take_the_model_fails_the_step(monkeypatch):
    deploy = _Deploy(monkeypatch, _config_card("name", "system_prompt", "output_format"))
    context = TaskStepContext()
    with pytest.raises(RuntimeError, match="takes no model"):
        await deploy.run(_step(), context)
    assert USER_SIMS_KEY not in context.metadata
    deploy.agent.close.assert_awaited_once()  # the deployed agent doesn't keep running
    assert context.deployed_agents == []  # and a retry doesn't find it


@pytest.mark.asyncio
async def test_a_failed_model_post_closes_the_agent_and_leaves_no_record(monkeypatch):
    deploy = _Deploy(monkeypatch)

    async def refuse(url, fields):
        if "model" in fields:
            raise RuntimeError("agent-config refused")

    monkeypatch.setattr(protocol, "post_agent_config", refuse)
    context = TaskStepContext()
    with pytest.raises(RuntimeError, match="agent-config refused"):
        await deploy.run(_step(), context)
    deploy.agent.close.assert_awaited_once()
    assert context.deployed_agents == [] and USER_SIMS_KEY not in context.metadata


# ---- always_on with the user it deployed -------------------------------------------------------------------------
def _always_on(**settings) -> ao.AlwaysOnTaskStep:
    return ao.AlwaysOnTaskStep("week", None, user_sim="user-sim", prompt_id="ask", agent_name="solver", **settings)


async def _rig_context(monkeypatch, rig: Rig, **fields) -> TaskStepContext:
    """The Rig's world, with the user sim as deploy_user_sim_agent records it (agent ``user-sim``)."""
    step = _step(**{"agent_name": "user-sim", "opening_prompt": "sort the cdmx weekend",
                    "persona": {"display_name": "Linda Murphy", "description": DESCRIPTION}, **fields})
    record = await _published(monkeypatch, step)
    ctx = rig.context()
    ctx.metadata[USER_SIMS_KEY] = {"user-sim": record}
    ctx.metadata["clock_configurations"] = [{"step_id": "clock", "timezone": "America/New_York"}]
    return ctx


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_always_on_runs_the_deployed_user_from_its_opening_and_sets_up_nothing(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            rig.message("Blue house or green?")
            return "sent"
        rig.quiet()
        return ""

    ctx_posts: list = []
    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=user_sim_versions,
              user_texts=['{"outcome": "reply", "message": "blue!", "used": "W1"}'])
    await rig.arm()
    ctx = await _rig_context(monkeypatch, rig)
    async def post(url, fields):
        ctx_posts.append((url, fields))

    monkeypatch.setattr(ao.protocol, "post_agent_config", post)
    ctx = await rig.run(_always_on(end_time="P3D"), ctx)

    assert not [u for u, _ in ctx_posts if u.startswith(USER_SIM)]  # its prompt and model were set at deploy
    assert any("sort the cdmx weekend" in t for t in rig.sent[0])
    [call] = rig.user_calls
    assert call["parts"][0]["text"] == (
        "It's Mon, Nov 9, 9:00 AM.\n"
        "You last messaged your AI assistant at 9:00 AM.\n"
        "\nYOU SENT YOUR AI ASSISTANT:\n  [9:00 AM] sort the cdmx weekend\n"
        "\nNEW FROM YOUR AI ASSISTANT (this is what you're answering):\n  [9:00 AM]\n    Blue house or green?\n"
        "\nYour AI assistant sent you this. As Linda, decide: reply, don't reply, or say you're done.\n"
        "Answer in JSON: " + user_sim_versions.ANSWER_LINE)
    [read] = rig.events(ctx, "user_sim_read")
    assert (read["outcome"], read["message"], read["used"], read["schema_errors"]) == ("reply", "blue!", "W1", [])
    record = rig.record(ctx)["user_sim"]
    assert (record["model"], record["model_from"], record["model_set"]) == ("sim-model", "user_sim_step", True)
    assert rig.events(ctx, "start")[0]["user_model"] == "sim-model"
    assert rig.events(ctx, "start")[0]["time_zone"] == "America/New_York"


@pytest.mark.usefixtures("local_stores")
@pytest.mark.parametrize("multi_day, end_time", [(True, None), (False, "P3D")], ids=["one-sitting", "multi-day"])
@pytest.mark.asyncio
async def test_always_on_refuses_a_user_deployed_for_the_other_kind_of_run(monkeypatch, tmp_path, multi_day,
                                                                           end_time):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    ctx = await _rig_context(monkeypatch, rig, multi_day=multi_day)
    with pytest.raises(ValueError, match="was deployed with multi_day"):
        await rig.run(_always_on(**({"end_time": end_time} if end_time else {})), ctx)
    assert rig.sent == []


@pytest.mark.parametrize("given", [{"prompt": "sort the cdmx weekend"},
                                   {"parts": [{"kind": "text", "text": "sort the cdmx weekend"}]}])
def test_always_on_takes_no_prompt_of_its_own(given):
    with pytest.raises(ValueError, match="always_on takes no (prompt|parts): the user's first message is the "
                                         "deploy_user_sim_agent step's opening_prompt"):
        _always_on(**given)


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_always_on_with_no_prompt_needs_the_user_s_opening(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    ctx = await _rig_context(monkeypatch, rig, multi_day=False)
    ctx.metadata[USER_SIMS_KEY]["user-sim"]["opening"] = None  # a record written by hand: the step itself requires one
    with pytest.raises(ValueError, match="has no opening_prompt, the user's first message"):
        await rig.run(_always_on(), ctx)


def test_always_on_without_a_prompt_round_trips_without_one():
    step = _always_on()
    assert step.prompt is None and ao.AlwaysOnTaskStep.from_dict(step.to_dict()).prompt is None


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_after_the_final_check_in_the_user_reads_the_wrap_up_like_any_other_read(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if any("send me the rundown" in t for t in texts):
            rig.message("House booked; La Ola is open Friday.")
            return "sent"
        if n == 1:
            rig.message("Blue house or green?")
            return "sent"
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=user_sim_versions,
              user_texts=['{"outcome": "reply", "message": "blue!", "used": "W1"}',
                          '{"outcome": "done", "message": "perfect, thank you!!", "used": "none"}'])
    await rig.arm()
    ctx = await rig.run(_always_on(end_time="P2D", final="send me the rundown"), await _rig_context(monkeypatch, rig))

    first, last = rig.user_calls
    text = last["parts"][0]["text"]
    assert text.startswith("It's Wed, Nov 11, 9:00 AM.\nYou last messaged your AI assistant at 9:00 AM.\n")
    assert "\nYOU SENT YOUR AI ASSISTANT:\n  [9:00 AM] send me the rundown\n" in text
    assert "(this is what you're answering):\n  [9:00 AM]\n    House booked; La Ola is open Friday.\n" in text
    assert text.endswith("Answer in JSON: " + user_sim_versions.ANSWER_LINE)
    reads = rig.events(ctx, "user_sim_read")
    assert [r["last_read"] for r in reads] == [False, True]
    assert (reads[-1]["outcome"], reads[-1]["message"], reads[-1]["schema_errors"]) == (
        "done", "perfect, thank you!!", [])
    assert rig.record(ctx)["end_reason"] == "end_time"
    assert not any("perfect, thank you" in t for texts in rig.sent for t in texts)


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_malformed_answer_is_retried_once_in_the_same_user_sim_session(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            rig.message("Blue house or green?")
            return "sent"
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=user_sim_versions,
              user_texts=["blue, obviously", '{"outcome": "reply", "message": "blue!", "used": "W1"}'])
    await rig.arm()
    ctx = await rig.run(_always_on(end_time="P3D"), await _rig_context(monkeypatch, rig))

    first, retry = rig.user_calls
    assert retry["parts"] == [{"kind": "text", "text": user_sim_versions.RETRY}]
    assert first["context_id"] == retry["context_id"]
    [read] = rig.events(ctx, "user_sim_read")
    assert (read["outcome"], read["message"], read["schema_errors"]) == ("reply", "blue!", [])
    assert read["user_sim_turns"] == 2
    assert read["retried"] is True
    assert read["first_answer"]["raw"] == "blue, obviously" and read["first_answer"]["schema_errors"]
    assert any("] Linda: blue!" in t for texts in rig.sent for t in texts)


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_after_max_replies_the_user_stops_reading(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        rig.message(f"question {n}?")
        return "sent"

    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=user_sim_versions,
              user_texts=['{"outcome": "reply", "message": "yes", "used": "W1"}'] * 5)
    await rig.arm()
    ctx = await _rig_context(monkeypatch, rig, persona={"display_name": "Linda Murphy", "description": DESCRIPTION,
                                                        "max_replies": 2})
    assert ctx.metadata[USER_SIMS_KEY]["user-sim"]["max_replies"] == 2
    ctx = await rig.run(_always_on(end_time="P3D", max_wakes=6), ctx)
    assert len(rig.events(ctx, "user_gave_up")) == 1
    assert [r["outcome"] for r in rig.events(ctx, "user_sim_read")][:2] == ["reply", "reply"]


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_the_user_s_own_time_zone_is_used_when_the_clock_sets_none(monkeypatch, tmp_path):
    """The clock runs in UTC (the apps take no zone), and the user lives in their own: reads show times in it."""
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    ctx = await _rig_context(monkeypatch, rig, multi_day=False,
                             persona={"display_name": "Linda Murphy", "description": DESCRIPTION,
                                      "time_zone": "America/Chicago"})
    ctx.metadata["clock_configurations"] = [{"step_id": "clock"}]
    assert ao._task_zone(ctx.metadata[USER_SIMS_KEY]["user-sim"], ctx.metadata["clock_configurations"]) \
        == "America/Chicago"
    ctx.metadata["clock_configurations"] = [{"step_id": "clock", "timezone": "America/New_York"}]
    with pytest.raises(ValueError, match="is in America/Chicago but sync_env_clock set"):
        ao._task_zone(ctx.metadata[USER_SIMS_KEY]["user-sim"], ctx.metadata["clock_configurations"])
