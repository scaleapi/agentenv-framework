"""The Tinker helper's episodes on the Episode API's fake steps, as tinker-cookbook's rollouts step them: a rollout's
tool calls run on its turn and its reply ends it with the task's reward, filed in its run group; the turn bound ends a
turn; a rollout the cookbook cut is scored once its group is done; a bundle's splits as prepared for a trainer to play;
and the tasks a dataset plays."""

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

GROUP = "dock-run/train/step-0003"


class _Renderer:
    """Stands in for a cookbook renderer: the system prompt and the names of the tools it is given, as one message."""

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        return [{"role": "system", "content": json.dumps({"system": system_prompt,
                                                          "tools": [tool["name"] for tool in tools]})}]


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
async def test_a_rollout_the_cookbook_cut_is_scored_once_its_group_is_done(teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, REPLIES)
    await env.initial_observation()

    reward, metrics = await env.score_cut()

    assert (reward, metrics["score/replies"], metrics["cut"]) == (0.0, 0.0, 1.0)
    assert await env.score_cut() == (0.0, {"cut": 0.0})


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
BOB = {"id": "bob", "type": "deploy_agent", "agent_name": "bob", "a2a_agent_id": "echo"}
ASK_BOB = {"id": "ask-bob", "type": "prompt_agent", "agent_name": "bob", "prompt": "Check."}


def _verifier(name):
    return {"id": name, "type": "env_outcome_verifier", "env_id": "retail", "file_artifact_id": "check",
            "verifier_id": name}


def _task(id, *steps):
    return id, Task.from_dict({"id": id, "version": 1, "steps": [dict(step) for step in steps]})


def test_tasks_that_cant_run_as_an_episode_are_left_out_and_the_rest_rewarded(caplog):
    played = _task("played", ENV, AGENT, PLAY, _verifier("records"), _verifier("db_match"))
    two_agents = _task("two-agents", ENV, AGENT, PLAY, BOB, ASK_BOB, _verifier("records"))
    with caplog.at_level(logging.WARNING):
        [(name, spec, reward)] = tinker.playable([played, two_agents], "records", None)
    assert (name, spec.trainer_agents) == ("played", ("policy",))
    assert reward({"records": 0.25, "db_match": 0.0}) == 0.25
    assert "two-agents" in caplog.text


@pytest.mark.parametrize(("tasks", "reward", "refusal"), [
    ([_task("unscored", ENV, AGENT, PLAY)], None, "records no score"),
    ([_task("two", ENV, AGENT, PLAY, _verifier("records"), _verifier("db_match"))], None, "name the one"),
    ([_task("two-agents", ENV, AGENT, PLAY, BOB, ASK_BOB, _verifier("records"))], "records", "no task can be played"),
])
def test_a_dataset_whose_tasks_make_no_reward_or_no_episode_is_refused(tasks, reward, refusal):
    with pytest.raises(ValueError, match=refusal):
        tinker.playable(tasks, reward, None)
