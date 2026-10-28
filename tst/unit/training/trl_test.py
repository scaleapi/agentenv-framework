"""``AgentEnv`` on the OpenEnv server ``build_app`` makes, served on loopback over the Episode API's fake steps and
used as TRL's trainers use it: the dataset and the environment classes, rollouts reset, stepped and scored on a reused
environment, the turns a model leaves open, episodes left unscored, a server at capacity or missing, tasks with other
tools, and a bundle served by ``AgentEnv.local`` whose episodes end at close and at exit."""

import asyncio
import inspect
import json
import logging

import pytest
from agentenv_protocol import EnvironmentTool, ToolResult
from agentenv_protocol.a2a_agent import proxy as proxy_module

from agent_env.config.runtime import Config
from agent_env.openenv.app import _app, _score_at
from agent_env.openenv.environment import Served, ServedTask
from agent_env.task import EpisodeSpec, Task
from agent_env.task_step.task_step import TaskStep
from agent_env.training import trl
from agent_env.training.trl import ENDED, FAILED, AgentEnv
from tst.unit.bundle._support import layout
from tst.unit.openenv.environment_test import AGAIN
from tst.unit.task.episode.run_test import ALICE, ITEMS, PANTRY, PANTRY_MCP, REPLIES, TURN, _tool_sessions

PLAYED = [ITEMS, ALICE, TURN, REPLIES]
SYSTEM = {"role": "system", "content": "You keep the items."}


@pytest.fixture
def served(teardowns, tools):
    """Serves ``tasks`` by name, every one in the split ``all`` and those ``splits`` name in theirs, as ``build_app``
    serves a bundle's; closed after the test."""
    opened = []

    def serve(tasks, splits=None, max_concurrent_envs=8, **kwargs):
        by_name = {}
        for name, steps in tasks.items():
            task = Task.from_dict({"id": name, "version": 1, "steps": steps})
            by_name[name] = ServedTask(task, name, EpisodeSpec.from_task(task), _score_at("replies"))
        named = {split: tuple(by_name[name] for name in names) for split, names in (splits or {}).items()}
        server = trl._Server(_app(Served(splits={"all": tuple(by_name.values()), **named}, sandbox=None), "dock",
                                  max_concurrent_envs))
        env = AgentEnv(server.url, **kwargs)
        env._server = server
        opened.append(env)
        return env

    yield serve
    for env in opened:
        env.close()


def _tools(environment):
    return [name for name, _ in inspect.getmembers(environment, inspect.isfunction) if not name.startswith("_")]


def test_the_dataset_holds_each_tasks_prompt_and_the_factory_offers_its_tools(served):
    env = served({"a": PLAYED, "b": [ITEMS, ALICE, {**TURN, "prompt": "Add a barrel."}, REPLIES]}, {"smoke": ["b"]})

    assert env.dataset.to_list() == [
        {"prompt": [SYSTEM, {"role": "user", "content": "Add a crate."}], "split": "all", "index": 0, "task_id": "a"},
        {"prompt": [SYSTEM, {"role": "user", "content": "Add a barrel."}], "split": "all", "index": 1, "task_id": "b"}]
    assert env.dataset_for("smoke").to_list() == [
        {"prompt": [SYSTEM, {"role": "user", "content": "Add a barrel."}], "split": "smoke", "index": 0, "task_id": "b"}]
    assert (env.factory.__name__, _tools(env.factory)) == ("dock", ["add_item", "end_turn", "get_reward", "reset"])
    assert inspect.signature(env.factory.end_turn).parameters["response"].annotation == str | None
    with pytest.raises(ValueError, match="has no split 'train'; it serves all, smoke"):
        env.dataset_for("train")


def test_a_rollout_plays_its_rows_task_and_an_environment_plays_one_after_another(served, teardowns):
    env = served({"a": PLAYED})
    [row] = env.dataset.to_list()
    environment = env.factory()

    assert environment.reset(**row) is None
    assert environment.add_item(item="crate") == "add_item {'item': 'crate'}"
    assert environment.end_turn(response="Added a crate.") == ENDED
    assert environment.add_item(item="crate") == ENDED
    assert environment.get_reward() == 1.0
    environment.reset(**row)
    assert environment.get_reward() == 0.0
    assert environment.get_reward() is None
    assert len(teardowns.finished) == 3


def test_end_turn_answers_with_the_next_turns_prompt_and_a_turn_left_open_ends_empty(served):
    env = served({"a": [ITEMS, ALICE, TURN, AGAIN, REPLIES]})
    environment = env.factory()

    environment.reset(**env.dataset[0])
    assert environment.end_turn(response="Added a crate.") == "Add another."
    assert environment.get_reward() == 0.0
    environment.reset(**env.dataset[0])
    environment.end_turn(response="Added a crate.")
    assert environment.end_turn(response="Added another.") == ENDED
    assert environment.get_reward() == 1.0


def test_an_episode_that_fails_or_isnt_its_rows_is_left_unscored_and_one_with_its_prompt_changed_plays(served, caplog):
    env = served({"a": PLAYED, "broken": [{**ITEMS, "fail": True}, ALICE, TURN, REPLIES]})
    played, broken = env.dataset.to_list()
    environment = env.factory()

    with caplog.at_level(logging.WARNING, logger=trl.__name__):
        environment.reset(**broken)
        assert environment.add_item(item="crate") == FAILED
        assert environment.get_reward() is None
        environment.reset(**{**played, "task_id": "broken"})
        assert environment.get_reward() is None
    environment.reset(**{**played, "prompt": [{"role": "system", "content": "Think first."}, *played["prompt"]]})
    environment.end_turn(response="Added a crate.")

    assert environment.get_reward() == 1.0
    assert [record.getMessage() for record in caplog.records if record.name == trl.__name__] == [
        "rollout left unscored: RuntimeError: the env didn't come up",
        "rollout left unscored: the episode plays task 'a', not the row's 'broken'"]


class _SecondTurnTools:
    """Stands in for the proxy's ``ToolSession``: a turn opens one session, and every second one opened also offers
    ``take_item``, so an episode's second turn offers a tool its first doesn't."""

    def __init__(self):
        self.opened = 0

    def __call__(self, *, mcp_url, headers=None, timeout):
        self.opened += 1
        offered = ["add_item", *(["take_item"] if self.opened % 2 == 0 else [])]

        class _Session:
            async def open(self):
                return self

            async def list_tools(self):
                return [EnvironmentTool(name=name, inputSchema={"type": "object"}) for name in offered]

            async def call_tool(self, name, arguments=None):
                return ToolResult(content=[{"type": "text", "text": f"{name} done"}])

            async def close(self):
                pass

        return _Session()


def test_a_tool_a_later_turn_offers_is_a_method_and_its_new_instructions_reach_the_model(served, monkeypatch):
    monkeypatch.setattr(proxy_module, "ToolSession", _SecondTurnTools())
    again = {**AGAIN, "system_prompt": "Now keep the barrels."}
    env = served({"a": [ITEMS, ALICE, TURN, again, REPLIES]})
    environment = env.factory()

    assert _tools(env.factory) == ["add_item", "end_turn", "get_reward", "reset", "take_item"]
    environment.reset(**env.dataset[0])
    assert "offers no tool 'take_item'" in environment.take_item()
    assert environment.end_turn(response="Added a crate.") == "Now keep the barrels.\n\nAdd another."
    assert environment.take_item() == "take_item done"
    environment.end_turn(response="Took one.")
    assert environment.get_reward() == 1.0


class _Hanging:
    """Stands in for the proxy's ``ToolSession``: every env offers ``wait``, whose call takes three seconds."""

    def __call__(self, *, mcp_url, headers=None, timeout):
        class _Session:
            async def open(self):
                return self

            async def list_tools(self):
                return [EnvironmentTool(name="wait", inputSchema={"type": "object"})]

            async def call_tool(self, name, arguments=None):
                await asyncio.sleep(3)

            async def close(self):
                pass

        return _Session()


def test_a_tool_that_misses_its_deadline_ends_the_episode_unscored_and_its_session_goes_on(
        served, monkeypatch, caplog):
    monkeypatch.setattr(proxy_module, "ToolSession", _Hanging())
    env = served({"a": PLAYED}, tool_timeout=0.5, max_concurrent_envs=1)
    environment = env.factory()

    environment.reset(**env.dataset[0])
    with caplog.at_level(logging.WARNING, logger=trl.__name__):
        assert environment.wait() == FAILED
        assert environment.get_reward() is None
    environment.reset(**env.dataset[0])
    environment.end_turn(response="Added a crate.")

    assert environment.get_reward() == 1.0
    assert "rollout left unscored: wait didn't answer within 0.5s, and the episode ended" in caplog.text


def test_a_session_refused_after_one_was_dropped_leaves_only_that_rollout_unscored(served, caplog):
    env = served({"a": PLAYED}, max_concurrent_envs=1)
    first, second = env.factory(), env.factory()
    first.reset(**env.dataset[0])
    first.end_turn(response="Added a crate.")
    assert first.get_reward() == 1.0
    trl._on_loop(first._close(), 30)

    second.reset(**env.dataset[0])
    with caplog.at_level(logging.WARNING, logger=trl.__name__):
        first.reset(**env.dataset[0])
        assert first.get_reward() is None
    trl._on_loop(second._close(), 30)
    first.reset(**env.dataset[0])
    first.end_turn(response="Added a crate.")

    assert first.get_reward() == 1.0
    assert "its session was dropped, and the server stayed at its limit of 1 sessions for 4s after" in caplog.text


def test_a_rollout_whose_tools_arent_its_classs_stops_training(served, monkeypatch):
    monkeypatch.setattr(proxy_module, "ToolSession", _SecondTurnTools())
    env = served({"a": PLAYED})
    environment = env.factory()

    environment.reset(**env.dataset[0])
    with pytest.raises(RuntimeError, match=r"task 'a' offers the tools \['add_item', 'end_turn', 'take_item'\], not "
                                           r"the \['add_item', 'end_turn'\] its toolset's first task does"):
        environment.get_reward()


class _Slow(TaskStep):
    """Takes a second, as a verifier that grades the episode after its last turn does."""

    type = "trl_test_slow"
    entity_refs = ()

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        await asyncio.sleep(1)
        return context


def test_end_turn_waits_for_the_episodes_score_past_a_tools_deadline(served, monkeypatch):
    registry = Config().task_step_registry()
    monkeypatch.setattr(Config, "task_step_registry", lambda self: {**registry, _Slow.type: _Slow})
    env = served({"a": [ITEMS, ALICE, TURN, {"id": "slow", "type": _Slow.type}, REPLIES]}, tool_timeout=0.2)
    environment = env.factory()

    environment.reset(**env.dataset[0])

    assert environment.end_turn(response="Added a crate.") == ENDED
    assert environment.get_reward() == 1.0


def test_a_server_at_capacity_refuses_the_reset_and_says_how_to_serve_more(served):
    env = served({"a": PLAYED}, max_concurrent_envs=1)
    first, second = env.factory(), env.factory()

    first.reset(**env.dataset[0])
    with pytest.raises(RuntimeError, match="at its limit of 1 sessions at once.* MAX_CONCURRENT_ENVS"):
        second.reset(**env.dataset[0])
    first.end_turn(response="Added a crate.")

    assert first.get_reward() == 1.0


def test_no_server_at_the_url_says_how_to_serve_one():
    with pytest.raises(ConnectionError, match=r"no OpenEnv server answers at http://127.0.0.1:9: serve the bundle"):
        AgentEnv("http://127.0.0.1:9")


def test_tasks_whose_agent_gets_other_tools_take_trls_multi_environment_mode(served, monkeypatch):
    _tool_sessions(monkeypatch, {"http://127.0.0.1:20031/items/mcp": ["add_item"], PANTRY_MCP: ["take_item"]})
    env = served({"a": PLAYED, "p": [PANTRY, {**ALICE, "env_ids": ["pantry"]}, TURN, REPLIES]}, max_concurrent_envs=1)

    assert {name: _tools(each) for name, each in env.factory.items()} == {
        "items": ["add_item", "end_turn", "get_reward", "reset"],
        "pantry": ["end_turn", "get_reward", "reset", "take_item"]}
    assert [(row["environment"], row["index"]) for row in env.dataset] == [("items", 0), ("pantry", 1)]
    pantry = env.factory["pantry"]()
    pantry.reset(**{key: value for key, value in env.dataset[1].items() if key != "environment"})
    assert pantry.take_item() == f"take_item on {PANTRY_MCP}"
    pantry.end_turn(response="Took one.")
    assert pantry.get_reward() == 1.0


def test_a_tool_trl_cant_call_is_refused_with_how_to_fix_it(served, tools):
    tools("add-item")
    env = served({"a": PLAYED})

    with pytest.raises(ValueError, match="tool 'add-item' can't be a TRL tool.*Rename the tool in the env"):
        env.factory  # noqa: B018


class _SlowTeardown:
    """A teardown hold that lets each teardown finish after half a second."""

    async def wait(self):
        await asyncio.sleep(0.5)

    def set(self):
        pass


@pytest.mark.parametrize("end", ["close", "exit"])
def test_a_bundle_served_by_local_plays_and_its_episodes_left_running_end_at_close_and_at_exit_waiting_out_a_slow_teardown(
        tmp_path, monkeypatch, teardowns, end):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(trl, "EXIT_SECONDS", 0.1)
    teardowns.hold = _SlowTeardown()
    unbound = {key: value for key, value in ALICE.items() if key not in ("env_ids", "role")}
    env = AgentEnv.local(layout(tmp_path / "dock", {"tasks/a.json": json.dumps([ITEMS, unbound, TURN, REPLIES])}))
    environment = env.factory()
    environment.reset(**env.dataset[0])
    environment.end_turn(response="Added a crate.")
    assert environment.get_reward() == 1.0
    environment.reset(**env.dataset[0])
    assert "offers no tool 'look'" in environment._call("look", {})

    env.close() if end == "close" else trl._close_all()

    assert (len(teardowns.finished), environment._client, trl._servers) == (3, None, set())
