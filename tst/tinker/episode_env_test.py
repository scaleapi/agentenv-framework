"""The Tinker helper's episodes on the Episode API's fake steps, as tinker-cookbook's rollouts step them: a rollout's
tool calls run on its turn and its reply ends it with the task's reward, filed in its run group; the turn bound ends a
turn; a step's logs are clipped; a rollout the cookbook cuts is scored as it is cut, freeing its slot for the rest of
its group; a bundle's splits as prepared for a trainer to play; and the tasks a dataset plays."""

import asyncio
import json
import logging

import pytest

from agent_env.openenv.reward import _score_at
from agent_env.runner import store as run_store
from agent_env.task import Task
from tst.unit.bundle._support import layout
from tst.unit.task.episode.run_test import ALICE, ITEMS, REPLIES, TURN, _spec
from tst.util.capabilities import TINKER, missing_capability_reason

tinker = pytest.importorskip("agent_env.training.tinker", reason=missing_capability_reason(TINKER),
                             exc_type=ImportError)
base = pytest.importorskip("tinker_cookbook.renderers.base", reason=missing_capability_reason(TINKER))
sdk = pytest.importorskip("tinker", reason=missing_capability_reason(TINKER))

GROUP = "dock-run/train/step-0003"


class _Renderer:
    """Stands in for a cookbook renderer: the system prompt and the names of the tools it is given, as one message,
    and a conversation as a token per message."""

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        return [{"role": "system", "content": json.dumps({"system": system_prompt,
                                                          "tools": [tool["name"] for tool in tools]})}]

    def get_stop_sequences(self):
        return []

    def build_generation_prompt(self, messages, **kwargs):
        return sdk.ModelInput.from_ints(list(range(len(messages))))


def _env(*steps, max_turns=8):
    return tinker.EpisodeEnv("dock", _spec(*steps), renderer=_Renderer(), reward=_score_at("replies"),
                             model_name="policy", sandbox=None, max_turns=max_turns, tool_timeout=5,
                             max_concurrent_episodes=4, run_group_id=GROUP)


def _call(name, **arguments):
    return base.ToolCall(function=base.ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)),
                         id=f"call-{name}")


@pytest.mark.asyncio
async def test_a_rollouts_calls_run_on_its_turn_and_its_reply_ends_it_with_the_tasks_reward_in_its_group(
        teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, REPLIES)

    system, user = await env.initial_observation()
    assert json.loads(system["content"]) == {"system": "You keep the items.", "tools": ["add_item"]}
    assert user == {"role": "user", "content": "Add a crate."}
    called = await env.step({"role": "assistant", "content": "", "tool_calls": [_call("add_item", item="crate")]})
    assert not called.episode_done
    assert called.next_messages[-1] == {"role": "tool", "content": "add_item {'item': 'crate'}",
                                        "tool_call_id": "call-add_item", "name": "add_item"}
    ended = await env.step({"role": "assistant", "content": "Added a crate."})

    assert (ended.episode_done, ended.reward) == (True, 1.0)
    assert {key: ended.metrics[key] for key in ("score/replies", "messages", "tool_calls", "tool_errors")} == {
        "score/replies": 1.0, "messages": 2, "tool_calls": 1, "tool_errors": 0}
    run = run_store.get_run(f"episode-{ended.logs['episode']}")
    assert (run.status, run.overrides["metadata"]["run_group_id"]) == ("COMPLETED", GROUP)
    assert await env.score_cut() == (0.0, {"cut": 0.0})


@pytest.mark.asyncio
async def test_a_turn_ends_at_the_bound_on_the_models_messages_with_no_reply(teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, REPLIES, max_turns=1)
    await env.initial_observation()

    ended = await env.step({"role": "assistant", "content": "", "tool_calls": [_call("add_item", item="crate")]})

    assert (ended.episode_done, ended.reward, ended.metrics["tool_calls"]) == (True, 0.0, 1)


@pytest.mark.asyncio
async def test_a_steps_logs_clip_the_models_message_and_each_call(teardowns, tools, monkeypatch):
    monkeypatch.setattr(tinker.EpisodeEnv, "LOGGED_CHARS", 8)
    env = _env(ITEMS, ALICE, TURN, REPLIES)
    await env.initial_observation()

    called = await env.step({"role": "assistant", "content": "Adding a crate now.",
                             "tool_calls": [_call("add_item", item="crate")]})
    await env.close()

    assert called.logs["assistant"] == "Adding a… (11 more characters)"
    assert called.logs["tool_call_0"] == ('add_item({"item":… (9 more characters)) -> '
                                          'add_item… (18 more characters)')


@pytest.mark.asyncio
async def test_a_rollout_the_cookbook_cut_has_its_open_turn_ended_and_is_scored_once(teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, REPLIES)
    await env.initial_observation()

    reward, metrics = await env.score_cut()

    assert (reward, metrics["score/replies"], metrics["cut"]) == (0.0, 0.0, 1.0)
    assert await env.score_cut() == (0.0, {"cut": 0.0})


def _group(group_size, max_concurrent_episodes, max_trajectory_tokens=None):
    settings = tinker._Settings(model_name="policy", renderer_name="stub", sandbox=None, max_turns=8, tool_timeout=5,
                                max_concurrent_episodes=max_concurrent_episodes,
                                max_trajectory_tokens=max_trajectory_tokens, failed_reward=-0.5, tag="dock")
    settings.renderer = _Renderer
    steps = tinker.TrainingSteps()
    return tinker.EpisodeGroupBuilder("dock", _spec(ITEMS, ALICE, TURN, REPLIES), _score_at("replies"), settings,
                                      group_size, tinker.RunGroup("dock-run", steps, steps.trained(3)))


@pytest.mark.asyncio
async def test_rollouts_the_cookbook_cuts_free_their_slots_so_a_group_larger_than_the_slots_finishes(
        teardowns, tools):
    group = _group(group_size=3, max_concurrent_episodes=1)
    envs = await group.make_envs()

    async def cut_at_the_token_limit(env):
        await env.initial_observation()
        return await env.step([], extra={"stop_reason": "length"})

    try:
        # Each rollout holds the one slot until it is scored: unscored when cut, the next rollout would never start.
        cut = await asyncio.wait_for(asyncio.gather(*(cut_at_the_token_limit(env) for env in envs)), timeout=60)
    finally:
        await group.cleanup()

    assert [(step.episode_done, step.reward, step.metrics["cut"], step.metrics["score/replies"]) for step in cut] == [
        (True, -0.5, 1.0, 0.0)] * 3
    assert {run_store.get_run(f"episode-{env.message_env._episode.id}").status for env in envs} == {"COMPLETED"}


@pytest.mark.asyncio
async def test_a_rollout_whose_prompt_is_over_the_token_budget_is_scored_and_frees_its_slot(teardowns, tools):
    group = _group(group_size=2, max_concurrent_episodes=1, max_trajectory_tokens=1)
    envs = await group.make_envs()

    try:
        overflows = await asyncio.wait_for(asyncio.gather(*(env.initial_observation() for env in envs)), timeout=60)
    finally:
        await group.cleanup()

    assert [(type(o).__name__, o.reward, o.metrics["cut"]) for o in overflows] == [
        ("InitialObservationOverflow", -0.5, 1.0)] * 2


def test_a_split_is_an_evals_tasks_prepared_without_the_agent_the_trainer_plays(teardowns, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    played = json.dumps([ITEMS, {**ALICE, "a2a_agent_id": "nobody", "env_ids": []}, TURN, REPLIES])
    root = layout(tmp_path / "dock", {"tasks/crate.json": played, "tasks/again.json": played,
                                      "evals/train.toml": 'tasks = ["crate"]\n'})

    splits = tinker._splits(root, ["all", "train"], None)

    assert {split: sorted(name for name, _ in tasks) for split, tasks in splits.items()} == {
        "all": ["again", "crate"], "train": ["crate"]}


ENV = {"id": "env", "type": "deploy_env", "env_id": "retail"}
AGENT = {"id": "agent", "type": "deploy_agent", "agent_name": "policy", "env_ids": ["retail"],
         "role": "agent@retail.example", "system_prompt": "You run the store."}
PLAY = {"id": "play", "type": "prompt_agent", "agent_name": "policy", "prompt_id": "play", "prompt": "Help."}
AGAIN = {**PLAY, "id": "again", "prompt_id": "again", "prompt": "Anything else?"}
BOB = {"id": "bob", "type": "deploy_agent", "agent_name": "bob", "a2a_agent_id": "echo"}
ASK_BOB = {"id": "ask-bob", "type": "prompt_agent", "agent_name": "bob", "prompt": "Check."}


def _verifier(name):
    return {"id": name, "type": "env_outcome_verifier", "env_id": "retail", "file_artifact_id": "check",
            "verifier_id": name}


def _task(id, *steps):
    return id, Task.from_dict({"id": id, "version": 1, "steps": [dict(step) for step in steps]})


def test_tasks_that_cant_run_as_an_episode_are_left_out_and_the_rest_rewarded(caplog):
    played = _task("played", ENV, AGENT, PLAY, _verifier("records"), _verifier("db_match"))
    two_turns = _task("two-turns", ENV, AGENT, PLAY, {**AGAIN, "system_prompt": "You run the store."},
                      _verifier("records"))
    two_agents = _task("two-agents", ENV, AGENT, PLAY, BOB, ASK_BOB, _verifier("records"))
    new_instructions = _task("new-instructions", ENV, AGENT, PLAY, {**AGAIN, "system_prompt": "You audit the store."},
                             _verifier("records"))
    with caplog.at_level(logging.WARNING):
        kept = tinker.playable([played, two_turns, two_agents, new_instructions], "records", None)
    assert [(name, spec.trainer_agents) for name, spec, _ in kept] == [("played", ("policy",)),
                                                                       ("two-turns", ("policy",))]
    assert kept[0][2]({"records": 0.25, "db_match": 0.0}) == 0.25
    assert "two-agents" in caplog.text
    assert "'new-instructions' changes the played agent's system prompt between turns" in caplog.text


@pytest.mark.parametrize(("tasks", "reward", "refusal"), [
    ([_task("unscored", ENV, AGENT, PLAY)], None, "records no score"),
    ([_task("two", ENV, AGENT, PLAY, _verifier("records"), _verifier("db_match"))], None, "name the one"),
    ([_task("two-agents", ENV, AGENT, PLAY, BOB, ASK_BOB, _verifier("records"))], "records", "no task can be played"),
])
def test_a_dataset_whose_tasks_make_no_reward_or_no_episode_is_refused(tasks, reward, refusal):
    with pytest.raises(ValueError, match=refusal):
        tinker.playable(tasks, reward, None)
