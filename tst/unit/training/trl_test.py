"""``AgentEnv`` on the OpenEnv server ``build_app`` makes, served on loopback over the Episode API's fake steps and
used as TRL's trainers use it: the dataset and the environment classes, rollouts reset, stepped and scored on a reused
environment, the turns a model leaves open, episodes left unscored, a server at capacity, missing or restarting, tasks
with other tools, and a bundle served by ``AgentEnv.local`` whose episodes end at close and at exit."""

import asyncio
import inspect
import json
import logging
import multiprocessing
import pickle

import pytest
from agentenv_protocol import EnvironmentTool, ToolResult
from agentenv_protocol.a2a_agent import proxy as proxy_module

from agent_env.config.runtime import Config
from agent_env.openenv.environment import AgentEnvEnvironment
from agent_env.task_step.task_step import TaskStep
from agent_env.training import _openenv, trl
from agent_env.training._openenv import ENDED, FAILED, on_session_loop
from agent_env.training.trl import AgentEnv
from tst.unit.bundle._support import layout
from tst.unit.openenv.environment_test import AGAIN
from tst.unit.task.episode.run_test import ALICE, ITEMS, PANTRY, PANTRY_MCP, REPLIES, TURN, _tool_sessions
from tst.unit.training.tools_test import called_by_qwen3, shown_to_qwen3

PLAYED = [ITEMS, ALICE, TURN, REPLIES]
SYSTEM = {"role": "system", "content": "You keep the items."}


@pytest.fixture
def served(serve):
    """``serve``'s tasks as an ``AgentEnv``, whose close stops their server; closed after the test."""
    opened = []

    def env(tasks, splits=None, max_concurrent_envs=8, **kwargs):
        server = serve(tasks, splits, max_concurrent_envs)
        opened.append(AgentEnv(server.url, **kwargs))
        opened[-1]._server = server
        return opened[-1]

    yield env
    for each in opened:
        each.close()


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
    env.close()
    assert len(teardowns.finished) == 3


def test_the_factory_pickles_and_plays_in_a_spawned_process_as_asyncgrpos_rollout_process_does(served):
    env = served({"a": PLAYED})

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(_play, (env.factory, env.dataset[0])) == 1.0
    rebuilt = pickle.loads(pickle.dumps(env.factory))()
    rebuilt.reset(**env.dataset[0])
    assert (_tools(type(rebuilt)), rebuilt.add_item(item="crate")) == (_tools(env.factory), "add_item {'item': 'crate'}")
    env.close()
    assert rebuilt._client is None
    with pytest.raises(pickle.PicklingError, match="recorded"):
        pickle.dumps(type("recorded", (env.factory,), {"get_reward": lambda self: 0.5}))


def _play(factory, row):
    environment = factory()
    environment.reset(**row)
    environment.end_turn(response="Added a crate.")
    return environment.get_reward()


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
    on_session_loop(first._close(), 30)

    second.reset(**env.dataset[0])
    with caplog.at_level(logging.WARNING, logger=trl.__name__):
        first.reset(**env.dataset[0])
        assert first.get_reward() is None
    on_session_loop(second._close(), 30)
    first.reset(**env.dataset[0])
    first.end_turn(response="Added a crate.")

    assert first.get_reward() == 1.0
    assert "its session was dropped, and the server stayed at its limit of 1 sessions for 4s after" in caplog.text


def test_a_session_dropped_by_a_restarting_server_opens_again_once_it_is_back(served, monkeypatch):
    env = served({"a": PLAYED})
    environment = env.factory()
    environment.reset(**env.dataset[0])
    environment.end_turn(response="Added a crate.")
    assert environment.get_reward() == 1.0
    on_session_loop(environment._close(), 30)
    opened, refusals = _openenv.opened, []

    async def restarting(url, timeout, *, token=None):
        if len(refusals) < 3:
            refusals.append(url)
            refused = ConnectionError(f"Failed to connect to {url}/ws")
            refused.__cause__ = ConnectionRefusedError(111, "Connection refused")
            raise ConnectionError(f"couldn't open a session on the server at {url}") from refused
        return await opened(url, timeout, token=token)

    on_loop, budgets = trl.on_session_loop, []

    def budgeted(coroutine, timeout):
        budgets.append(timeout)
        return on_loop(coroutine, timeout)

    monkeypatch.setattr(_openenv, "opened", restarting)
    monkeypatch.setattr(_openenv, "REOPEN_SECONDS", 0.01)
    monkeypatch.setattr(trl, "on_session_loop", budgeted)
    environment.reset(**env.dataset[0])
    environment.end_turn(response="Added a crate.")

    shared = environment._shared
    assert (environment.get_reward(), len(refusals)) == (1.0, 3)
    assert budgets[1] > shared.reset_timeout + shared.reset_timeout + max(shared.reset_timeout, shared.tool_timeout)


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


def test_a_reset_and_an_end_turn_carry_the_resets_deadline_so_end_turn_waits_for_the_score_past_a_tools(
        served, monkeypatch):
    registry = Config().task_step_registry()
    monkeypatch.setattr(Config, "task_step_registry", lambda self: {**registry, _Slow.type: _Slow})
    deadlines = []
    reset_async, end_turn = AgentEnvEnvironment.reset_async, AgentEnvEnvironment._end_turn

    async def reset(self, *args, timeout_s=None, **kwargs):
        deadlines.append(("reset", timeout_s))
        return await reset_async(self, *args, timeout_s=timeout_s, **kwargs)

    async def end(self, name, response, timeout_s):
        deadlines.append((name, timeout_s))
        return await end_turn(self, name, response, timeout_s)

    monkeypatch.setattr(AgentEnvEnvironment, "reset_async", reset)
    monkeypatch.setattr(AgentEnvEnvironment, "_end_turn", end)
    env = served({"a": [ITEMS, ALICE, TURN, {"id": "slow", "type": _Slow.type}, REPLIES]}, tool_timeout=0.2,
                 reset_timeout=30)
    environment = env.factory()

    environment.reset(**env.dataset[0])

    assert environment.end_turn(response="Added a crate.") == ENDED
    assert environment.get_reward() == 1.0
    assert deadlines == [("reset", 30), ("reset", 30), ("end_turn", 30)]


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
    pantry = pickle.loads(pickle.dumps(env.factory))["pantry"]()
    pantry.reset(**{key: value for key, value in env.dataset[1].items() if key != "environment"})
    assert pantry.take_item() == f"take_item on {PANTRY_MCP}"
    pantry.end_turn(response="Took one.")
    assert pantry.get_reward() == 1.0


STACK = EnvironmentTool(name="stack_crates", description="Stack crates on a pallet.", inputSchema={
    "type": "object", "required": ["pallet", "crates"], "properties": {
        "pallet": {"type": "string", "enum": ["north", "south"]},
        "crates": {"type": "array", "description": "bottom crate first", "items": {
            "type": "object", "required": ["kind"], "properties": {
                "kind": {"type": "string", "description": "what the crate holds"},
                "size": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}}}}}})
#: What TRL shows the model of the tools ``add_item`` and ``STACK``: each as its env publishes it, and ``end_turn`` as
#: TRL has shown it since the factory's first release, its method's schema.
SHOWN = [
    {"type": "function", "function": {"name": "add_item", "description": "add_item.", "parameters": {"type": "object"}}},
    {"type": "function", "function": {
        "name": "end_turn",
        "description": "End your turn with your reply. The episode goes on to your next turn, or ends and is scored.",
        "parameters": {"type": "object", "properties": {
            "response": {"type": "string", "nullable": True, "description": "Your reply."}}},
        "return": {"type": "string"}}},
    {"type": "function", "function": {"name": "stack_crates", "description": "Stack crates on a pallet.",
                                      "parameters": STACK.inputSchema}}]


def test_trl_shows_the_model_each_tool_as_its_env_publishes_it_and_end_turn_as_before_and_calls_them(served, tools):
    tools("add_item", STACK)
    env = served({"a": PLAYED})
    environment = env.factory()
    fetched = [getattr(environment, name) for name in _tools(env.factory) if name not in ("reset", "get_reward")]

    assert shown_to_qwen3(fetched) == SHOWN
    from_the_class = env.factory.stack_crates.__get__(environment)
    assert shown_to_qwen3([from_the_class])[0]["function"]["parameters"]["properties"]["crates"]["items"] == {
        "type": "object"}
    environment.reset(**env.dataset[0])
    crates = [{"kind": "apples", "size": [1, 1, 0.5]}]
    calls = called_by_qwen3(fetched, {"name": "stack_crates", "arguments": {"pallet": "north", "crates": crates}},
                            {"name": "end_turn", "arguments": {"response": "Added a crate."}})
    by_name = {tool.__name__: tool for tool in fetched}

    assert [by_name[call["function"]["name"]](**call["function"]["arguments"]) for call in calls] == [
        f"stack_crates {{'pallet': 'north', 'crates': {crates}}}", ENDED]
    assert environment.get_reward() == 1.0


def test_a_transformers_that_renders_a_tool_other_than_as_its_schema_stops_the_factory(served, tools, monkeypatch):
    tools(STACK)
    env = served({"a": PLAYED})
    monkeypatch.setattr(trl, "render_jinja_template", lambda conversations, tools, chat_template: (
        [json.dumps([{**tool, "function": {**tool["function"], "parameters": {"type": "object"}}} for tool in tools])],
        []))

    with pytest.raises(RuntimeError, match=r"renders the tools \['stack_crates'\] other than as their env's"):
        env.factory  # noqa: B018


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
