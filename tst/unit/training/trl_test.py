"""``AgentEnv`` on the OpenEnv server ``build_app`` makes, served on loopback over the Episode API's fake steps and
used as TRL's trainers use it: the dataset and the environment classes, rollouts reset, stepped and scored on a reused
environment, the turns a model leaves open, episodes left unscored, a server at capacity or missing, tasks with other
tools, and a bundle served by ``AgentEnv.local`` whose episodes end at close and at exit. Then the tool methods TRL
reads a tool's schema from."""

import asyncio
import inspect
import json
import logging
from typing import Any, Literal

import pytest
from agentenv_protocol import EnvironmentTool
from agentenv_protocol.a2a_agent import proxy as proxy_module

from agent_env.openenv.app import _app, _score_at
from agent_env.openenv.environment import Served, ServedTask
from agent_env.task import EpisodeSpec, Task
from agent_env.training import trl
from agent_env.training.trl import ENDED, FAILED, AgentEnv, tool_method
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
        {"prompt": [SYSTEM, {"role": "user", "content": "Add a crate."}], "split": "all", "index": 0},
        {"prompt": [SYSTEM, {"role": "user", "content": "Add a barrel."}], "split": "all", "index": 1}]
    assert env.dataset_for("smoke").to_list() == [
        {"prompt": [SYSTEM, {"role": "user", "content": "Add a barrel."}], "split": "smoke", "index": 0}]
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


def test_an_episode_that_fails_or_isnt_its_rows_is_left_unscored_and_the_environment_plays_on(served, caplog):
    env = served({"a": PLAYED, "broken": [{**ITEMS, "fail": True}, ALICE, TURN, REPLIES]})
    played, broken = env.dataset.to_list()
    environment = env.factory()

    with caplog.at_level(logging.WARNING, logger=trl.__name__):
        environment.reset(**broken)
        assert environment.add_item(item="crate") == FAILED
        assert environment.get_reward() is None
        environment.reset(**{**played, "prompt": [{"role": "user", "content": "Add a barrel."}]})
        assert environment.get_reward() is None
    environment.reset(**played)
    environment.end_turn(response="Added a crate.")

    assert environment.get_reward() == 1.0
    assert [record.getMessage() for record in caplog.records if record.name == trl.__name__] == [
        "rollout left unscored: RuntimeError: the env didn't come up",
        "rollout left unscored: the turn's prompt differs from the dataset row's"]


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


def test_a_tool_that_misses_its_deadline_leaves_the_rollout_unscored_and_the_environment_plays_on(
        served, monkeypatch, caplog):
    monkeypatch.setattr(proxy_module, "ToolSession", _Hanging())
    env = served({"a": PLAYED}, tool_timeout=0.5)
    environment = env.factory()

    environment.reset(**env.dataset[0])
    with caplog.at_level(logging.WARNING, logger=trl.__name__):
        assert environment.wait() == FAILED
        assert environment.get_reward() is None
    environment.reset(**env.dataset[0])
    environment.end_turn(response="Added a crate.")

    assert environment.get_reward() == 1.0
    assert "rollout left unscored: wait didn't answer within 0.5s" in caplog.text


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
    env = served({"a": PLAYED, "p": [PANTRY, {**ALICE, "env_ids": ["pantry"]}, TURN, REPLIES]})

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


@pytest.mark.parametrize("end", ["close", "exit"])
def test_a_bundle_served_by_local_plays_and_its_episodes_left_running_end_at_close_and_at_exit(
        tmp_path, monkeypatch, teardowns, end):
    monkeypatch.setenv("HOME", str(tmp_path))
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


def test_a_tools_arguments_are_typed_and_documented_from_its_schema():
    method = tool_method({"name": "restock", "description": "Restock a shelf.", "input_schema": {
        "type": "object", "required": ["shelf", "count"],
        "properties": {
            "shelf": {"type": "string", "enum": ["top", "bottom"], "description": "which shelf"},
            "count": {"type": "integer"},
            "labels": {"type": "array", "items": {"type": "string"}, "title": "Labels"},
            "note": {"type": ["string", "null"], "default": None, "description": "for the night shift"},
            "options": {"type": "object", "properties": {"rush": {"type": "boolean"}}, "required": ["rush"]},
        },
    }})

    parameters = inspect.signature(method).parameters
    assert {name: (parameter.annotation, parameter.default) for name, parameter in parameters.items()} == {
        "self": (inspect.Parameter.empty, inspect.Parameter.empty),
        "shelf": (Literal["top", "bottom"], inspect.Parameter.empty),
        "count": (int, inspect.Parameter.empty),
        "labels": (list[str] | None, None),
        "note": (str | None, None),
        "options": (dict | None, None),
    }
    assert method.__doc__ == (
        "Restock a shelf.\n\nArgs:\n    shelf: which shelf\n    count: count\n    labels: Labels\n"
        "    note: for the night shift\n    options: options Fields: rush (boolean, required).")
    assert tool_method({"name": "ping"}).__doc__ == "Call the ping tool."


@pytest.mark.parametrize(("tool", "message"), [
    ({"name": "get-situation"}, "can't be a TRL tool"),
    ({"name": "reset"}, "can't be a TRL tool"),
    ({"name": "_inspect"}, "can't be a TRL tool"),
    ({"name": "plan", "input_schema": {"properties": {"ship-id": {"type": "integer"}}}}, "argument 'ship-id'"),
    ({"name": "upload", "input_schema": {"properties": {"file": {"type": "file"}}}}, r"upload\.file: JSON Schema"),
], ids=lambda value: value["name"] if isinstance(value, dict) else "")
def test_a_tool_trl_cant_show_the_model_is_refused(tool: dict[str, Any], message):
    with pytest.raises(ValueError, match=message):
        tool_method(tool)
