"""A bundle task as an OpenEnv env, on the Episode API's fake steps: the Task API, an episode reset, stepped and scored,
its tool errors, every way it ends in an error observation, calls that overlap, and how a session's episode is torn
down."""

import asyncio
import threading

import httpx
import pytest
from agentenv_protocol import EnvironmentTool, ToolResult, ToolSessionError
from agentenv_protocol.a2a_agent import proxy as proxy_module
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction, ToolErrorType

import agent_env.task.episode.run as episode_run
from agent_env.openenv.app import _score_at
from agent_env.openenv.environment import ENDED, AgentEnvEnvironment, Served, ServedTask
from agent_env.task import EpisodeSpec, Task
from tst.unit.task.episode.run_test import ALICE, ITEMS, REPLIES, TURN, _gate, _until

AGAIN = {**TURN, "id": "again", "prompt_id": "again", "prompt": "Add another.", "depends_on": ["turn"]}


class _Tools:
    """Stands in for the proxy's ``ToolSession``: every env offers ``offered``, and a call answers with what ``answers`` maps
    its tool to, raising it when it's an exception. A session opens once ``release``, a ``threading.Event`` since the
    in-process proxy runs on a loop of its own, is set, when there is one."""

    def __init__(self, offered, answers):
        self.offered, self.answers, self.closed, self.release = offered, answers, 0, None

    def __call__(self, *, mcp_url, headers=None, timeout):
        tools = self

        class _Session:
            async def open(self):
                if tools.release is not None:
                    await asyncio.to_thread(tools.release.wait)
                return self

            async def list_tools(self):
                return [EnvironmentTool(name=name, description=f"{name}.", inputSchema={"type": "object"})
                        for name in tools.offered]

            async def call_tool(self, name, arguments=None):
                answer = tools.answers.get(name, ToolResult(content=[{"type": "text", "text": f"{name} {arguments}"}]))
                if isinstance(answer, BaseException):
                    raise answer
                return answer

            async def close(self):
                tools.closed += 1

        return _Session()


@pytest.fixture
def tools(monkeypatch):
    def offer(*offered, **answers):
        sessions = _Tools(offered or ("add_item",), answers)
        monkeypatch.setattr(proxy_module, "ToolSession", sessions)
        return sessions

    offer()
    return offer


def _task(*steps, id="dock"):
    return Task.from_dict({"id": id, "version": 1, "steps": [dict(step) for step in steps]})


def _env(*steps, reward=_score_at("replies"), splits=None):
    task = _task(*steps)
    served = (ServedTask(task, "dock", EpisodeSpec.from_task(task), reward),)
    return AgentEnvEnvironment(Served(splits=splits or {"all": served}, sandbox=None))


async def _call(env, tool, **arguments):
    return await env.step_async(CallToolAction(tool_name=tool, arguments=arguments))


@pytest.mark.asyncio
async def test_the_task_api_lists_each_split_with_each_tasks_prompt_and_agent(teardowns, tools):
    tasks = [_task(ITEMS, ALICE, {**TURN, "prompt": f"Add crate {n}."}, REPLIES, id=f"dock-{n}") for n in range(3)]
    served = tuple(ServedTask(task, f"dock-{n}", EpisodeSpec.from_task(task), _score_at("replies"))
                   for n, task in enumerate(tasks))
    env = AgentEnvEnvironment(Served(splits={"smoke": served[:1], "all": served}, sandbox=None))

    assert env.list_splits() == ["smoke", "all"]
    assert (env.num_tasks("smoke"), env.num_tasks("all")) == (1, 3)
    assert env.get_task("all", 2) == {"id": "dock-2", "name": "dock-2", "index": 2, "prompt": "Add crate 2.",
                                      "system_prompt": "You keep the items.", "agent": "alice"}
    assert [task["index"] for task in env.get_task_range("all", 1)] == [1, 2]
    assert [task["id"] for task in env.list_tasks("smoke")] == ["dock-0"]
    with pytest.raises(IndexError, match="split 'all' has 3 tasks, so it has no task 3"):
        env.get_task("all", 3)
    with pytest.raises(IndexError, match="there is no split 'train'; the splits are smoke, all"):
        env.num_tasks("train")
    assert (await env.reset_async()).metadata["task"] == {"id": "dock-0", "name": "dock-0", "split": "smoke", "index": 0}
    await asyncio.to_thread(env.close)


@pytest.mark.asyncio
@pytest.mark.parametrize(("reply", "reward"), [("Added a crate.", 1.0), ("", 0.0)])
async def test_an_episode_resets_to_its_turn_steps_through_the_envs_tools_and_is_scored_on_end_turn(
        teardowns, tools, reply, reward):
    env = _env(ITEMS, ALICE, TURN, REPLIES)

    reset = await env.reset_async(split="all", index=0)
    assert (reset.done, reset.reward) == (False, None)
    assert reset.metadata == {"prompt": "Add a crate.", "system_prompt": "You keep the items.",
                              "task": {"id": "dock", "name": "dock", "split": "all", "index": 0}}
    listed = await env.step_async(ListToolsAction())
    assert [(tool.name, tool.description) for tool in listed.tools][0] == ("add_item", "add_item.")
    assert listed.tools[-1].name == "end_turn"
    called = await _call(env, "add_item", item="crate")
    assert called.error is None
    assert called.result == {"content": [{"type": "text", "text": "add_item {'item': 'crate'}"}],
                             "structured_content": None, "meta": None, "data": "add_item {'item': 'crate'}",
                             "is_error": False}

    ended = await _call(env, "end_turn", response=reply)

    assert (ended.done, ended.reward) == (True, reward)
    assert ended.metadata == {"scores": {"replies": reward}, "task": reset.metadata["task"]}
    assert Task.get_instance(env.state.episode_id).status == "completed"
    assert (env.state.task_id, env.state.step_count, env.state.turn, env.state.done) == ("dock", 2, "turn", True)
    after = await _call(env, "add_item")
    assert (after.done, after.reward, after.metadata["status"], after.metadata["error"]) == (
        True, None, "error", ENDED)
    assert teardowns.finished == [env.state.episode_id]


@pytest.mark.asyncio
async def test_a_step_from_a_worker_thread_runs_on_the_episodes_loop_as_openenvs_playground_steps(teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, REPLIES)
    assert (await asyncio.to_thread(env.step, ListToolsAction())).metadata["error"] == ENDED
    await env.reset_async()

    listed = await asyncio.to_thread(env.step, ListToolsAction())
    ended = await asyncio.to_thread(env.step, CallToolAction(tool_name="end_turn", arguments={"response": "Added."}))

    assert [tool.name for tool in listed.tools] == ["add_item", "end_turn"]
    assert (ended.done, ended.reward) == (True, 1.0)


@pytest.mark.asyncio
async def test_a_task_with_two_turns_hands_over_the_next_prompt_then_the_reward(teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, AGAIN, REPLIES)
    await env.reset_async()

    first = await _call(env, "end_turn", response="Added a crate.")
    assert (first.done, first.reward, first.result) == (
        False, None, {"prompt": "Add another.", "system_prompt": "You keep the items."})
    assert env.state.turn == "again"
    second = await _call(env, "end_turn", response="Added another.")

    assert (second.done, second.reward) == (True, 1.0)


@pytest.mark.asyncio
async def test_a_failed_tool_call_is_a_tool_error_and_the_episode_goes_on(teardowns, tools):
    tools("add_item", "slow", "flaky", "broken", slow=TimeoutError("tools/call did not finish within 600s"),
          flaky=httpx.ConnectError("refused"), broken=ToolSessionError("tools/call failed: boom (code -32603)"),
          add_item=ToolResult(content=[{"type": "text", "text": "no room"}], isError=True))
    env = _env(ITEMS, ALICE, TURN, REPLIES)
    await env.reset_async()

    errors = [(observation.error.error_type, observation.error.message, observation.done) for observation in [
        await _call(env, name) for name in ("add_item", "slow", "flaky", "broken", "missing")]]

    assert errors == [
        (ToolErrorType.EXECUTION_ERROR, "no room", False),
        (ToolErrorType.EXECUTION_ERROR, "error: the tool call timed out after 600s", False),
        (ToolErrorType.EXECUTION_ERROR, "error: ConnectError: refused", False),
        (ToolErrorType.EXECUTION_ERROR, "error: ToolSessionError: tools/call failed: boom (code -32603)", False),
        (ToolErrorType.TOOL_NOT_FOUND, "EpisodeError: turn 'turn' offers no tool 'missing'; it offers "
                                       "['add_item', 'broken', 'flaky', 'slow']", False)]
    assert (await _call(env, "end_turn", response="Done.")).reward == 1.0


@pytest.mark.asyncio
async def test_a_deploy_that_fails_ends_the_episode_at_reset_in_an_error_observation(teardowns, tools):
    env = _env({**ITEMS, "fail": True}, ALICE, TURN, REPLIES)

    reset = await env.reset_async()

    assert (reset.done, reset.reward, reset.metadata["status"]) == (True, None, "error")
    assert reset.metadata["error"] == "RuntimeError: the env didn't come up"
    assert Task.get_instance(env.state.episode_id).status == "failed"
    assert teardowns.finished == [env.state.episode_id]
    assert (await env.step_async(ListToolsAction())).metadata["error"] == ENDED


@pytest.mark.asyncio
async def test_a_run_that_fails_mid_turn_ends_the_episode_at_the_next_step(teardowns, tools):
    env = _env(ITEMS, ALICE, {**TURN, "timeout_seconds": 1}, REPLIES)
    await env.reset_async()
    async with asyncio.timeout(5):
        while not env._episode.finished:
            await asyncio.sleep(0.01)

    step = await _call(env, "add_item")

    assert (step.done, step.reward, step.metadata["status"]) == (True, None, "error")
    assert step.metadata["error"].startswith("TimeoutError: A2A task ")
    assert step.metadata["error"].endswith(" did not complete within 1s")
    assert (await _call(env, "add_item")).metadata["error"] == ENDED


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["state", "end_turn"])
async def test_an_env_offering_a_name_openenv_reserves_is_refused_at_reset_and_torn_down(teardowns, tools, tool):
    tools("add_item", tool)
    env = _env(ITEMS, ALICE, TURN, REPLIES)

    reset = await env.reset_async()

    assert (reset.done, reset.metadata["error"]) == (True, f"EpisodeError: the env offers tools OpenEnv reserves: {tool}")
    assert Task.get_instance(env.state.episode_id).status == "cancelled"
    assert teardowns.finished == [env.state.episode_id]


@pytest.mark.asyncio
async def test_a_split_or_index_with_no_task_or_one_without_the_other_is_an_error_observation(teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, REPLIES)

    assert (await env.reset_async(split="train", index=0)).metadata["error"] == (
        "there is no split 'train'; the splits are all")
    assert (await env.reset_async(split="all", index=1)).metadata["error"] == (
        "split 'all' has 1 tasks, so it has no task 1")
    assert (await env.reset_async(index=0)).metadata["error"] == (
        "split and index go together: give both, or neither for the first task served")
    assert teardowns.started == []

    env._served.splits = {"refused": (), **env._served.splits}
    assert (await env.reset_async()).metadata["task"]["split"] == "all"
    await asyncio.to_thread(env.close)


@pytest.mark.asyncio
async def test_a_reset_on_a_live_session_cancels_and_tears_down_its_episode_first(teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, REPLIES)
    await env.reset_async()
    first = env.state.episode_id

    await env.reset_async()

    assert (Task.get_instance(first).status, Task.get_instance(first).error) == ("cancelled", "reset")
    assert teardowns.finished == [first]
    assert (await _call(env, "end_turn", response="Added.")).reward == 1.0


@pytest.mark.asyncio
async def test_a_reset_cancelled_while_the_last_episode_tears_down_leaves_that_teardown_tracked_and_waited_for(
        teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, REPLIES)
    await env.reset_async()
    first = env.state.episode_id
    teardowns.hold = asyncio.Event()
    resetting = asyncio.create_task(env.reset_async())
    await _until(lambda: first in teardowns.started)
    resetting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await resetting

    again = asyncio.create_task(env.reset_async())
    await asyncio.sleep(0.2)
    assert (len(env._served.ending), list(episode_run._live), again.done()) == (1, [first], False)
    teardowns.hold.set()
    await again

    assert teardowns.finished == [first] and env.state.episode_id != first
    assert (await _call(env, "end_turn", response="Added.")).reward == 1.0


@pytest.mark.asyncio
async def test_calls_that_overlap_on_one_session_take_turns_as_openenvs_http_sessions_can_overlap_them(
        teardowns, tools):
    up, judge = {"id": "up", "type": "episode_test_gate"}, {"id": "judge", "type": "episode_test_gate"}
    env = _env(ITEMS, up, {**ALICE, "depends_on": ["items", "up"]}, TURN, {**judge, "depends_on": ["turn"]},
               {**REPLIES, "depends_on": ["judge"]})
    resetting = asyncio.create_task(env.reset_async())
    listing = asyncio.create_task(env.step_async(ListToolsAction()))
    await asyncio.sleep(0.05)
    _gate("up").set()
    await resetting
    assert [tool.name for tool in (await listing).tools] == ["add_item", "end_turn"]

    ending = asyncio.create_task(_call(env, "end_turn", response="Added."))
    calling = asyncio.create_task(_call(env, "add_item"))
    await asyncio.sleep(0.05)
    _gate("judge").set()

    assert (await ending).reward == 1.0
    assert (await calling).metadata["error"] == ENDED


@pytest.mark.asyncio
async def test_an_end_turn_cancelled_before_the_next_turn_opens_ends_the_episode_at_the_next_step(teardowns, tools):
    between = {"id": "between", "type": "episode_test_gate", "depends_on": ["turn"]}
    env = _env(ITEMS, ALICE, TURN, between, {**AGAIN, "depends_on": ["between"]}, REPLIES)
    await env.reset_async()
    ending = asyncio.create_task(_call(env, "end_turn", response="Added a crate."))
    await asyncio.sleep(0.05)
    ending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ending

    listed = await env.step_async(ListToolsAction())

    assert (listed.done, listed.metadata["status"], listed.metadata["error"]) == (
        True, "error", "the call opening the episode's turn was cancelled")
    assert (Task.get_instance(env.state.episode_id).status, teardowns.finished) == ("cancelled", [env.state.episode_id])


@pytest.mark.asyncio
async def test_a_session_closed_while_its_turn_opens_is_torn_down_with_the_turns_tools_once_it_has_opened(
        teardowns, tools):
    sessions = tools()
    sessions.release = threading.Event()
    env = _env(ITEMS, ALICE, TURN, REPLIES)
    resetting = asyncio.create_task(env.reset_async())
    async with asyncio.timeout(5):
        while env._stack is None:
            await asyncio.sleep(0.01)

    closing = asyncio.create_task(asyncio.to_thread(env.close))
    await asyncio.sleep(0.05)
    sessions.release.set()
    await resetting
    await closing
    async with asyncio.timeout(5):  # the proxy closes the session the turn opened as it stops, after the teardown
        while not sessions.closed:
            await asyncio.sleep(0.01)

    assert (Task.get_instance(env.state.episode_id).status, teardowns.finished) == ("cancelled", [env.state.episode_id])
    assert (await env.step_async(ListToolsAction())).metadata["error"] == ENDED


@pytest.mark.asyncio
async def test_closing_a_session_from_a_worker_thread_waits_for_its_teardown(teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, REPLIES)
    await env.reset_async()

    await asyncio.to_thread(env.close)

    assert (Task.get_instance(env.state.episode_id).status, teardowns.finished) == ("cancelled", [env.state.episode_id])
    assert env._served.live == set()


@pytest.mark.asyncio
async def test_closing_on_the_servers_loop_tears_down_later_and_the_servers_close_waits(teardowns, tools):
    env = _env(ITEMS, ALICE, TURN, REPLIES)
    await env.reset_async()
    other = AgentEnvEnvironment(env._served)
    await other.reset_async()

    env.close()
    assert teardowns.finished == []
    await env._served.close("the server shut down")

    assert sorted(teardowns.finished) == sorted([env.state.episode_id, other.state.episode_id])
    assert Task.get_instance(other.state.episode_id).error == "the server shut down"


@pytest.mark.asyncio
async def test_a_reward_names_its_score_and_a_missing_or_raising_one_leaves_the_episode_unscored(teardowns, tools):
    async def ended(**kwargs):
        env = _env(ITEMS, ALICE, TURN, REPLIES, **kwargs)
        await env.reset_async()
        return await _call(env, "end_turn", response="Added.")

    assert (await ended(reward=lambda scores: scores["replies"] * 2)).reward == 2.0
    missing = await ended(reward=_score_at("other"))
    assert (missing.reward, missing.metadata["error"]) == (
        None, "no reward: LookupError: the run recorded no score at step 'other'")
    raising = await ended(reward=lambda scores: scores["other"])
    assert (raising.reward, raising.metadata["error"]) == (None, "no reward: KeyError: 'other'")
    assert (await ended(reward=lambda scores: None)).metadata == {"scores": {"replies": 1.0}, "task": {
        "id": "dock", "name": "dock", "split": "all", "index": 0}}
