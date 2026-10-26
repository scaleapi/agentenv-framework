"""The run of an episode, on fake steps: its turns, its scores, abandoning it, teardown and its record."""

import asyncio
import json
import threading
import time

import httpx
import pytest
import pytest_asyncio
import uvicorn
from agentenv_protocol import EnvironmentTool, ToolResult, ToolSessionError
from agentenv_protocol.a2a_agent import (
    AgentEnvAgent,
    AgentIdentity,
    ProxyBashAction,
    ProxyBashResult,
    ProxyTextPart,
    ProxyTool,
    ProxyToolResult,
    TaskRequest,
    TaskResult,
    a2a_agent,
)
from agentenv_protocol.a2a_agent import proxy as proxy_module

import agent_env.task.episode.run as episode_run
from agent_env.config import get_config
from agent_env.config.runtime import Config
from agent_env.env.env import DeployedEnv
from agent_env.task import EpisodeError, EpisodeSpec, Task
from agent_env.task.teardown import TeardownReport
from agent_env.task_step.task_step import TaskStep

class _Env(TaskStep):
    """Deploys ``env_id`` with no sandbox, or fails as an env that doesn't come up."""

    type = "episode_test_env"
    entity_refs = ()

    def __init__(self, id, version, env_id="items", fail=False, depends_on=None, fail_task_on_error=True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.fail = fail

    def to_dict(self):
        return {**super().to_dict(), "env_id": self.env_id, "fail": self.fail}

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), env_id=data.get("env_id", "items"), fail=data.get("fail", False))

    async def execute(self, context):
        if self.fail:
            raise RuntimeError("the env didn't come up")
        context.deployed_envs.append(DeployedEnv(
            env_id=self.env_id, env_version=1, mcp_url=f"http://127.0.0.1:20031/{self.env_id}/mcp",
            metadata={"deploy_step_id": self.id},
        ))
        return context


GATES: dict[str, asyncio.Event] = {}


def _gate(step_id):
    return GATES.setdefault(step_id, asyncio.Event())


class _Gate(TaskStep):
    """Waits until the test opens the gate of its id."""

    type = "episode_test_gate"
    entity_refs = ()

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        await _gate(self.id).wait()
        return context


class _Replies(TaskStep):
    """Scores 1 when the run recorded replies and every one has text, else 0."""

    type = "episode_test_replies"
    entity_refs = ()

    def __init__(self, id, version, verifier_id="replies", depends_on=None, fail_task_on_error=True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.verifier_id = verifier_id

    def to_dict(self):
        return {**super().to_dict(), "verifier_id": self.verifier_id}

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), verifier_id=data["verifier_id"])

    async def execute(self, context):
        replies = [response.response for response in context.prompt_responses]
        context.metadata.setdefault("verifications", {})[self.verifier_id] = {
            "score": float(bool(replies) and all(replies))}
        return context


class _Teardowns:
    """Stands in for ``teardown_run``: records each run it tears down, and waits on ``hold`` when it's set."""

    def __init__(self):
        self.started = []
        self.finished = []
        self.hold = None

    async def __call__(self, context, **_):
        self.started.append(context.instance_id)
        if self.hold is not None:
            await self.hold.wait()
        self.finished.append(context.instance_id)
        return TeardownReport()


class _ToolSessions:
    """Stands in for the proxy's ``ToolSession``: the session on each URL serves the tools ``offered`` names for it,
    and every session opened, call made and session closed is recorded."""

    def __init__(self, offered, raises=None):
        self.offered = offered
        self.raises = raises or {}
        self.opened = []
        self.calls = []
        self.closed = []

    def __call__(self, *, mcp_url, headers=None, timeout):
        sessions = self

        class _Session:
            async def open(self):
                sessions.opened.append((mcp_url, headers))
                return self

            async def list_tools(self):
                return [EnvironmentTool(name=name) for name in sessions.offered[mcp_url]]

            async def call_tool(self, name, arguments=None):
                sessions.calls.append((mcp_url, name, arguments))
                if name in sessions.raises:
                    raise sessions.raises[name]
                return ToolResult(content=[{"type": "text", "text": f"{name} on {mcp_url}"}])

            async def close(self):
                sessions.closed.append(mcp_url)

        return _Session()


@pytest_asyncio.fixture
async def teardowns(monkeypatch, local_stores):
    """Fake steps registered and ``teardown_run`` stood in for; an episode the test leaves running is abandoned
    before they are undone."""
    steps = {**Config().task_step_registry(), **{step.type: step for step in (_Env, _Gate, _Replies)}}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    GATES.clear()
    teardowns = _Teardowns()
    monkeypatch.setattr(episode_run, "teardown_run", teardowns)
    yield teardowns
    for gate in GATES.values():
        gate.set()
    if teardowns.hold is not None:
        teardowns.hold.set()
    for episode in list(episode_run._live.values()):
        await episode.abandon("the test ended")


ITEMS = {"id": "items", "type": "episode_test_env"}
ALICE = {"id": "alice", "type": "deploy_agent", "agent_name": "alice", "env_ids": ["items"],
         "role": "alice@example.com", "system_prompt": "You keep the items."}
TURN = {"id": "turn", "type": "prompt_agent", "agent_name": "alice", "prompt_id": "turn", "prompt": "Add a crate.",
        "timeout_seconds": 5, "poll_interval_seconds": 0.05}
REPLIES = {"id": "replies", "type": "episode_test_replies", "verifier_id": "replies"}


def _spec(*steps, **kwargs):
    task = Task.from_dict({"id": "dock", "version": 4, "steps": [dict(step) for step in steps]})
    return EpisodeSpec.from_task(task, **kwargs)


async def _ended(episode):
    await _until(lambda: Task.get_instance(episode.id).status != "running")


async def _until(holds):
    async with asyncio.timeout(5):
        while not holds():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_the_run_waits_at_an_outside_turn_and_carries_on_once_it_ends(teardowns):
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    assert Task.get_instance(episode.id).status == "running"

    turn = await episode.next_turn()
    assert (turn.agent, turn.step_id, turn.prompt, turn.system_prompt) == (
        "alice", "turn", "Add a crate.", "You keep the items.")

    await turn.end("Added a crate.")
    assert await episode.result() == {"replies": 1.0}
    assert await episode.next_turn() is None
    assert teardowns.finished == [episode.id]
    record = Task.get_instance(episode.id)
    assert record.status == "completed"
    assert record.context["metadata"]["episode"] == {"id": episode.id, "trainer_agents": ["alice"]}
    assert [(r["step_id"], r["prompt_text"], r["response"]) for r in record.context["prompt_responses"]] == [
        ("turn", "Add a crate.", "Added a crate.")]
    assert [(agent["agent_name"], agent["sandbox_id"]) for agent in record.context["deployed_agents"]] == [
        ("alice", None)]


ITEMS_MCP, PANTRY_MCP = "http://127.0.0.1:20031/items/mcp", "http://127.0.0.1:20031/pantry/mcp"
PANTRY = {"id": "pantry", "type": "episode_test_env", "env_id": "pantry"}
STOCKED = {**ALICE, "env_ids": ["items", "pantry"]}
ROLE = "AgentEnv-Role"


def _tool_sessions(monkeypatch, offered, raises=None):
    sessions = _ToolSessions(offered, raises)
    monkeypatch.setattr(proxy_module, "ToolSession", sessions)
    monkeypatch.setattr(episode_run, "sandbox_request_headers_for_url", lambda url: {"X-Sandbox": url})
    return sessions


@pytest.mark.asyncio
async def test_a_turns_env_tools_run_in_its_proxy_with_the_headers_each_url_needs_and_the_role(teardowns, monkeypatch):
    sessions = _tool_sessions(monkeypatch, {ITEMS_MCP: ["items_add_item"], PANTRY_MCP: ["pantry_count"]})
    episode = await _spec(ITEMS, PANTRY, STOCKED, TURN, REPLIES).start()
    turn = await episode.next_turn()

    async with turn.tools(timeout=30) as tools:
        assert [tool.name for tool in tools.list()] == ["items_add_item", "pantry_count"]
        assert (await tools.call("pantry_count")).text == f"pantry_count on {PANTRY_MCP}"
    await turn.end("Counted.")
    await episode.result()

    assert sessions.opened == [(url, {"X-Sandbox": url, ROLE: "alice@example.com"}) for url in (ITEMS_MCP, PANTRY_MCP)]
    assert sessions.calls == [(PANTRY_MCP, "pantry_count", {})]
    assert sorted(sessions.closed) == [ITEMS_MCP, PANTRY_MCP]


@pytest.mark.asyncio
async def test_a_tool_two_envs_offer_is_refused_as_the_tools_open(teardowns, monkeypatch):
    _tool_sessions(monkeypatch, {ITEMS_MCP: ["count"], PANTRY_MCP: ["count"]})
    episode = await _spec(ITEMS, PANTRY, STOCKED, TURN, REPLIES).start()
    turn = await episode.next_turn()

    with pytest.raises(EpisodeError, match="tool 'count' is offered by both 'items' and 'pantry'"):
        async with turn.tools():
            pass


@pytest.mark.asyncio
async def test_a_tool_the_turn_doesnt_offer_is_refused(teardowns, monkeypatch):
    _tool_sessions(monkeypatch, {ITEMS_MCP: ["items_add_item"]})
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    turn = await episode.next_turn()

    async with turn.tools() as tools:
        with pytest.raises(EpisodeError) as raised:
            await tools.call("nope")
    assert str(raised.value) == "turn 'turn' offers no tool 'nope'; it offers ['items_add_item']"


@pytest.mark.asyncio
async def test_an_env_tool_that_fails_answers_an_error_and_is_recorded(teardowns, monkeypatch):
    _tool_sessions(monkeypatch, {ITEMS_MCP: ["items_add_item"]},
                   raises={"items_add_item": ToolSessionError("the env didn't answer")})
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    turn = await episode.next_turn()

    async with turn.tools() as tools:
        result = await tools.call("items_add_item", {"name": "crate"})
    await turn.end("Gave up.")
    await episode.result()

    assert (result.isError, result.text) == (True, "error: ToolSessionError: the env didn't answer")
    [response] = Task.get_instance(episode.id).context["prompt_responses"]
    stored = json.loads(get_config().get_object_store().get(response["agent_trajectory_object_url"]))
    assert json.loads(stored[0]["attributes"]["gen_ai.completion"]) == {
        "error": "error: ToolSessionError: the env didn't answer"}


@pytest.mark.asyncio
async def test_two_outside_agents_turns_are_each_handed_out_once(teardowns):
    bob = {**ALICE, "id": "bob", "agent_name": "bob", "role": None, "depends_on": ["items"]}
    bob_turn = {**TURN, "id": "bob-turn", "agent_name": "bob", "prompt_id": "bob-turn", "depends_on": ["bob"]}
    episode = await _spec(
        ITEMS, {**ALICE, "depends_on": ["items"]}, bob, {**TURN, "depends_on": ["alice"]}, bob_turn,
        {**REPLIES, "depends_on": ["turn", "bob-turn"]}, trainer_agents=["alice", "bob"],
    ).start()

    first = await episode.next_turn()
    second = await episode.next_turn()
    assert {first.agent, second.agent} == {"alice", "bob"}
    await first.end("one")
    await second.end("two")

    assert await episode.result() == {"replies": 1.0}


@pytest.mark.asyncio
async def test_episodes_of_one_spec_run_side_by_side_each_with_its_own_record(teardowns):
    spec = _spec(ITEMS, ALICE, TURN, REPLIES)

    async def play(n):
        episode = await spec.start()
        await (await episode.next_turn()).end(f"crate {n}")
        return episode.id, await episode.result()

    played = dict(await asyncio.gather(*(play(n) for n in range(8))))

    assert list(played.values()) == [{"replies": 1.0}] * 8
    assert sorted(teardowns.finished) == sorted(played)
    assert sorted(Task.get_instance(id).context["prompt_responses"][0]["response"] for id in played) == [
        f"crate {n}" for n in range(8)]


@pytest.mark.asyncio
async def test_a_seed_reaches_the_turns_prompt_env_binding_and_role(teardowns, monkeypatch):
    sessions = _tool_sessions(monkeypatch, {PANTRY_MCP: ["pantry_count"]})
    episode = await _spec(
        {**ITEMS, "env_id": "<env>"},
        {**ALICE, "env_ids": ["<env>"], "role": "<who>", "system_prompt": "You keep the <env>."},
        {**TURN, "prompt": "Add a crate to the <env>."}, REPLIES,
        seed={"env": "pantry", "who": "bob@example.com"},
    ).start()

    turn = await episode.next_turn()
    async with turn.tools():
        pass

    assert (turn.prompt, turn.system_prompt) == ("Add a crate to the pantry.", "You keep the pantry.")
    assert sessions.opened == [(PANTRY_MCP, {"X-Sandbox": PANTRY_MCP, ROLE: "bob@example.com"})]


@pytest.mark.asyncio
async def test_a_setup_failure_reaches_the_caller_waiting_for_a_turn(teardowns):
    episode = await _spec({**ITEMS, "fail": True}, ALICE, TURN, REPLIES).start()

    with pytest.raises(RuntimeError, match="the env didn't come up"):
        await episode.next_turn()
    with pytest.raises(RuntimeError, match="the env didn't come up"):
        await episode.result()
    assert Task.get_instance(episode.id).status == "failed"
    assert teardowns.finished == [episode.id]


@pytest.mark.asyncio
async def test_a_turn_not_taken_by_its_deadline_fails_the_run_and_is_never_handed_out(teardowns):
    episode = await _spec(ITEMS, ALICE, {**TURN, "timeout_seconds": 1}, REPLIES).start()
    await _ended(episode)

    with pytest.raises(TimeoutError, match="did not complete within 1s"):
        await episode.next_turn()
    assert Task.get_instance(episode.id).status == "failed"


@pytest.mark.asyncio
async def test_a_turn_ended_after_its_deadline_changes_nothing(teardowns):
    episode = await _spec(ITEMS, ALICE, {**TURN, "timeout_seconds": 1}, REPLIES).start()
    turn = await episode.next_turn()
    await _ended(episode)

    await turn.end("too late")

    with pytest.raises(TimeoutError, match="did not complete within 1s"):
        await episode.result()
    record = Task.get_instance(episode.id)
    assert record.status == "failed" and "did not complete within 1s" in record.error
    assert record.context["prompt_responses"] == []


@pytest.mark.asyncio
async def test_a_tolerant_turn_past_its_deadline_is_scored_without_a_reply(teardowns):
    tolerant = {**TURN, "timeout_seconds": 1, "fail_task_on_error": False}
    episode = await _spec(ITEMS, ALICE, tolerant, REPLIES).start()
    await _ended(episode)

    assert await episode.result() == {"replies": 0.0}
    assert await episode.next_turn() is None
    assert [f["step_id"] for f in Task.get_instance(episode.id).context["metadata"]["failed_steps"]] == ["turn"]


@pytest.mark.asyncio
async def test_the_result_is_refused_while_a_turn_waits_or_is_open(teardowns):
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()

    with pytest.raises(EpisodeError, match="turn 'turn' of agent 'alice' isn't ended, so the run can't finish"):
        await episode.result()
    turn = await episode.next_turn()
    with pytest.raises(EpisodeError, match="isn't ended"):
        await episode.result()
    await turn.end("done")
    assert await episode.result() == {"replies": 1.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(("reply", "score"), [("Added a crate.", 1.0), (None, 0.0)])
async def test_the_reply_is_graded_as_a_prompt_agents_would_be(teardowns, reply, score):
    graded = {"id": "graded", "type": "agent_prompt_response_verifier", "prompt_id": "turn", "verifier_id": "graded",
              "criteria": [{"type": "response_contains", "needles": ["crate"]}]}
    episode = await _spec(ITEMS, ALICE, TURN, graded).start()

    await (await episode.next_turn()).end(reply)

    assert await episode.result() == {"graded": score}


@pytest.mark.asyncio
async def test_an_episode_abandoned_as_it_starts_is_recorded_cancelled_with_the_reason(teardowns):
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()

    await episode.abandon("the trainer moved on")

    record = Task.get_instance(episode.id)
    assert (record.status, record.error) == ("cancelled", "the trainer moved on")
    assert teardowns.finished == [episode.id]
    for ask in (episode.next_turn, episode.result):
        with pytest.raises(EpisodeError, match="episode abandoned: the trainer moved on"):
            await ask()


@pytest.mark.asyncio
async def test_an_episode_abandoned_mid_turn_drops_the_turn(teardowns):
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    turn = await episode.next_turn()

    await episode.abandon("a tool call failed")
    await turn.end("too late")

    record = Task.get_instance(episode.id)
    assert (record.status, record.error, record.context["prompt_responses"]) == (
        "cancelled", "a tool call failed", [])
    assert teardowns.finished == [episode.id]


@pytest.mark.asyncio
async def test_a_turn_waiting_when_the_episode_is_abandoned_is_never_handed_out(teardowns):
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    await _until(lambda: episode._waiting)

    abandoning = asyncio.create_task(episode.abandon("stop"))
    await asyncio.sleep(0)
    with pytest.raises(EpisodeError, match="episode abandoned: stop"):
        await episode.next_turn()
    await abandoning


@pytest.mark.asyncio
async def test_abandoning_during_teardown_waits_for_it_and_leaves_the_runs_outcome(teardowns):
    teardowns.hold = asyncio.Event()
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    await (await episode.next_turn()).end("done")
    await _until(lambda: teardowns.started)

    abandoning = asyncio.create_task(episode.abandon("too late"))
    await asyncio.sleep(0.05)
    assert not abandoning.done()
    teardowns.hold.set()
    await abandoning

    assert teardowns.finished == [episode.id]
    assert await episode.result() == {"replies": 1.0}
    assert Task.get_instance(episode.id).status == "completed"


@pytest.mark.asyncio
async def test_cancelling_a_caller_cancels_only_its_wait(teardowns):
    gate = {"id": "before", "type": "episode_test_gate"}
    episode = await _spec(ITEMS, ALICE, gate, TURN, {**gate, "id": "after"}, REPLIES).start()

    waiting = asyncio.create_task(episode.next_turn())
    await asyncio.sleep(0.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    _gate("before").set()
    await (await episode.next_turn()).end("done")

    scoring = asyncio.create_task(episode.result())
    await asyncio.sleep(0.01)
    scoring.cancel()
    with pytest.raises(asyncio.CancelledError):
        await scoring
    _gate("after").set()
    assert await episode.result() == {"replies": 1.0}


@pytest.mark.asyncio
async def test_an_agent_named_by_the_seed_is_played_from_outside_in_the_run(teardowns):
    episode = await _spec(
        ITEMS, {**ALICE, "agent_name": "<who>"}, {**TURN, "agent_name": "<who>"}, REPLIES, seed={"who": "bob"},
    ).start()

    turn = await episode.next_turn()

    assert turn.agent == "bob"
    await turn.end("done")
    assert await episode.result() == {"replies": 1.0}


@pytest.mark.asyncio
async def test_a_turn_ended_twice_keeps_its_first_reply(teardowns):
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    turn = await episode.next_turn()

    await turn.end("first")
    await turn.end("second")
    await episode.result()

    assert [r["response"] for r in Task.get_instance(episode.id).context["prompt_responses"]] == ["first"]


@pytest.mark.asyncio
async def test_the_scores_are_every_verifiers_by_step_with_an_aggregate_beside_its_parts(teardowns):
    def graded(step_id, needle):
        return {"id": step_id, "type": "agent_prompt_response_verifier", "prompt_id": "turn",
                "verifier_id": f"has-{needle}", "criteria": [{"type": "response_contains", "needles": [needle]}]}
    overall = {"id": "overall", "type": "aggregate_verifiers", "verifier_ids": ["has-crate", "has-pallet"],
               "verifier_id": "overall"}
    episode = await _spec(ITEMS, ALICE, TURN, graded("crates", "crate"), graded("pallets", "pallet"), overall).start()

    await (await episode.next_turn()).end("Added a crate.")

    assert await episode.result() == {"crates": 1.0, "pallets": 0.0, "overall": 0.5}


@pytest.mark.asyncio
async def test_a_score_is_keyed_by_its_step_when_the_seed_names_its_verifier(teardowns):
    episode = await _spec(ITEMS, ALICE, TURN, {**REPLIES, "verifier_id": "replies-<week>"}, seed={"week": "3"}).start()

    await (await episode.next_turn()).end("done")

    assert await episode.result() == {"replies": 1.0}


@pytest.mark.asyncio
async def test_a_tolerated_verifier_that_recorded_no_score_is_missing_from_the_scores(teardowns):
    unscored = {"id": "graded", "type": "agent_prompt_response_verifier", "prompt_id": "missing",
                "verifier_id": "graded", "criteria": [{"type": "response_contains", "needles": ["crate"]}],
                "fail_task_on_error": False}
    episode = await _spec(ITEMS, ALICE, TURN, unscored, REPLIES).start()
    await (await episode.next_turn()).end("done")

    assert await episode.result() == {"replies": 1.0}


BREAKS_WHILE_THE_TURN_IS_OPEN = (
    ITEMS, {**ALICE, "depends_on": ["items"]}, {**TURN, "depends_on": ["alice"]},
    {"id": "later", "type": "episode_test_gate", "depends_on": ["items"]},
    {"id": "broken", "type": "episode_test_env", "env_id": "crane", "fail": True, "depends_on": ["later"]},
    {**REPLIES, "depends_on": ["turn", "broken"]},
)


@pytest.mark.asyncio
async def test_a_failure_elsewhere_in_the_task_surfaces_once_the_open_turn_ends(teardowns):
    episode = await _spec(*BREAKS_WHILE_THE_TURN_IS_OPEN).start()
    turn = await episode.next_turn()
    _gate("later").set()
    await _until(lambda: episode._context.metadata.get("failed_steps"))

    await turn.end("done")

    with pytest.raises(RuntimeError, match="the env didn't come up"):
        await episode.result()
    assert Task.get_instance(episode.id).status == "failed"


@pytest.mark.asyncio
async def test_abandoning_a_run_a_step_has_failed_keeps_its_error(teardowns):
    episode = await _spec(*BREAKS_WHILE_THE_TURN_IS_OPEN).start()
    await episode.next_turn()
    _gate("later").set()
    await _until(lambda: episode._context.metadata.get("failed_steps"))

    await episode.abandon("the trainer moved on")

    with pytest.raises(RuntimeError, match="the env didn't come up"):
        await episode.next_turn()
    record = Task.get_instance(episode.id)
    assert (record.status, record.error) == ("failed", "the env didn't come up")
    assert teardowns.finished == [episode.id]


@pytest.mark.asyncio
async def test_cancelling_the_caller_of_abandon_still_abandons_the_run(teardowns):
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    await episode.next_turn()

    abandoning = asyncio.create_task(episode.abandon("stop"))
    await asyncio.sleep(0)
    abandoning.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoning
    with pytest.raises(EpisodeError, match="episode abandoned: stop"):
        await episode.result()

    assert (Task.get_instance(episode.id).status, Task.get_instance(episode.id).error) == ("cancelled", "stop")
    assert teardowns.finished == [episode.id]


@pytest.mark.asyncio
async def test_a_cancelled_start_takes_its_run_down_with_it(teardowns):
    spec = _spec(ITEMS, ALICE, TURN, REPLIES)

    starting = asyncio.create_task(spec.start())
    await asyncio.sleep(0)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting

    assert not episode_run._live
    assert teardowns.finished == teardowns.started and len(teardowns.finished) == 1


class _SandboxedProxy:
    """Stands in for the client of a proxy deployed in a sandbox: offers ``items_add_item`` and ``bash``, answers each
    action from ``results`` or ran ``<command>``, and records each action and reply."""

    def __init__(self, results=None):
        self.results = results or {}
        self.actions = []
        self.replies = []

    async def tools(self, task_id):
        return [ProxyTool(name="items_add_item", server="items"), ProxyTool(name="bash", description="Run a command.")]

    async def act(self, task_id, action):
        self.actions.append(action)
        if isinstance(action, ProxyBashAction):
            return self.results.get(action.command) or ProxyBashResult(
                exit_code=0, stdout=f"ran {action.command}\n", stderr="", timed_out=False)
        return ProxyToolResult(content=[{"type": "text", "text": "added"}], is_error=False)

    async def reply(self, task_id, *, parts=None, error=None):
        self.replies.append((task_id, parts, error))
        return True


def _sandboxed_turn(proxy):
    return episode_run.Turn(agent="alice", step_id="turn", prompt="Add a crate.", system_prompt=None, skills=[],
                            proxy=proxy, task_id="t-1")


@pytest.mark.asyncio
async def test_a_turn_in_the_agents_sandbox_offers_bash_and_shows_its_output_and_a_failed_exit():
    proxy = _SandboxedProxy({"slow": ProxyBashResult(stdout="", stderr="", timed_out=True),
                             "both": ProxyBashResult(exit_code=2, stdout="out\n", stderr="err\n", timed_out=False),
                             "quiet": ProxyBashResult(exit_code=1, stdout="", stderr="", timed_out=False)})

    async with _sandboxed_turn(proxy).tools() as tools:
        assert [tool.name for tool in tools.list()] == ["items_add_item", "bash"]
        ran = await tools.call("bash", {"command": "ls inputs"})
        both = await tools.call("bash", {"command": "both"})
        quiet = await tools.call("bash", {"command": "quiet"})
        timed_out = await tools.call("bash", {"command": "slow"})
        with pytest.raises(EpisodeError, match="tool 'bash' takes a command string, not 3"):
            await tools.call("bash", {"command": 3})
        assert (await tools.call("items_add_item", {"name": "crate"})).text == "added"

    assert [(result.isError, result.text) for result in (ran, both, quiet, timed_out)] == [
        (False, "ran ls inputs\n"), (True, "out\nerr\nexit code 2"), (True, "exit code 1"),
        (True, "error: the command timed out")]
    assert [action.model_dump(exclude_none=True) for action in proxy.actions] == [
        {"kind": "bash", "command": "ls inputs"}, {"kind": "bash", "command": "both"},
        {"kind": "bash", "command": "quiet"}, {"kind": "bash", "command": "slow"},
        {"kind": "tool", "server": "items", "name": "items_add_item", "arguments": {"name": "crate"},
         "timeout_seconds": 600.0}]


@pytest.mark.asyncio
async def test_a_turn_ends_with_its_reply_as_parts():
    proxy = _SandboxedProxy()
    turn = _sandboxed_turn(proxy)

    await turn.end("Added a crate.")
    await turn.end("Again.")

    assert proxy.replies == [("t-1", [ProxyTextPart(text="Added a crate.")], None)]


@pytest.mark.asyncio
async def test_a_reply_that_cant_reach_the_proxy_leaves_the_runs_own_failure_to_report(caplog):
    class _Unreachable(_SandboxedProxy):
        async def reply(self, task_id, **reply):
            raise httpx.ConnectError("the sandbox is gone")

    turn = _sandboxed_turn(_Unreachable())

    await turn.end("done")

    assert "its reply didn't reach the proxy: the sandbox is gone" in caplog.text


ADD_A_CRATE = {"role": "assistant", "content": "Adding one.", "tool_calls": [
    {"id": "c1", "type": "function", "function": {"name": "items_add_item", "arguments": '{"name": "crate"}'}}]}


@pytest.mark.asyncio
async def test_recorded_model_calls_are_held_to_the_calls_that_run(teardowns, monkeypatch):
    _tool_sessions(monkeypatch, {ITEMS_MCP: ["items_add_item"]})
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    turn = await episode.next_turn()

    async with turn.tools() as tools:
        await turn.record_model_call(ADD_A_CRATE, model="qwen", usage={"prompt_tokens": 9, "completion_tokens": 4})
        with pytest.raises(EpisodeError, match="didn't make this call to 'items_add_item'"):
            await tools.call("items_add_item", {"name": "box"})
        with pytest.raises(EpisodeError, match="haven't run"):
            await turn.end("Added.")
        await tools.call("items_add_item", {"name": "crate"})
    await turn.record_model_call("Added a crate.", model="qwen")
    await turn.end("Added a crate.")
    assert await episode.result() == {"replies": 1.0}

    [response] = Task.get_instance(episode.id).context["prompt_responses"]
    stored = json.loads(get_config().get_object_store().get(response["agent_trajectory_object_url"]))
    assert [(span["name"], span["attributes"]["gen_ai.operation.name"]) for span in stored] == [
        ("chat", "chat"), ("items_add_item", "execute_tool"), ("chat", "chat"), ("turn", "chain")]
    assert json.loads(stored[-1]["attributes"]["gen_ai.prompt"]) == {
        "prompt": "Add a crate.", "system_prompt": "You keep the items."}
    assert stored[-1]["attributes"]["gen_ai.request.model"] == "qwen"


@pytest.mark.asyncio
async def test_a_model_call_in_a_shape_the_proxy_cant_read_is_refused(teardowns):
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    turn = await episode.next_turn()

    with pytest.raises(EpisodeError, match="OpenAI chat message, an Anthropic message or a string"):
        await turn.record_model_call({"role": "assistant", "content": {"not": "readable"}})


@a2a_agent(identity=AgentIdentity(name="bob", description="Asks for one thing more.", version="1.0.0"))
class _Bob(AgentEnvAgent):
    async def run(self, request: TaskRequest) -> TaskResult:
        return TaskResult.text("And a box.")


@pytest.fixture
def bob():
    """A counterpart on loopback that the outside agent's prompt converses with."""
    server = uvicorn.Server(uvicorn.Config(_Bob().create_app(), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    server.should_exit = True
    thread.join(5)


@pytest.mark.asyncio
async def test_a_conversation_with_the_outside_agent_is_played_a_turn_per_message(teardowns, bob):
    episode = await _spec(ITEMS, ALICE, {**TURN, "max_conversation_turns": 2, "user_a2a_url": bob}, REPLIES).start()

    first = await episode.next_turn()
    await first.end("Added a crate.")
    second = await episode.next_turn()
    await second.end("Added a box.")

    assert [(turn.step_id, turn.prompt) for turn in (first, second)] == [
        ("turn", "Add a crate."), ("turn", "And a box.")]
    assert await episode.result() == {"replies": 1.0}
    [response] = Task.get_instance(episode.id).context["prompt_responses"]
    assert response["response"] == "Added a box."


@pytest.mark.asyncio
async def test_a_prompt_continuing_a_conversation_gets_its_own_turn(teardowns):
    first = {**TURN, "context_id": "chat"}
    again = {**TURN, "id": "again", "prompt_id": "again", "prompt": "And another.", "depends_on": ["turn"],
             "context_id": "chat"}
    episode = await _spec(ITEMS, ALICE, first, again, REPLIES).start()

    turn = await episode.next_turn()
    assert (turn.step_id, turn.prompt) == ("turn", "Add a crate.")
    await turn.end("Added.")
    turn = await episode.next_turn()
    assert (turn.step_id, turn.prompt) == ("again", "And another.")
    await turn.end("Added another.")
    assert await episode.result() == {"replies": 1.0}


@pytest.mark.asyncio
async def test_the_turns_trajectory_is_stored_as_any_agents_is(teardowns, monkeypatch):
    _tool_sessions(monkeypatch, {ITEMS_MCP: ["items_add_item"]})
    episode = await _spec(ITEMS, ALICE, TURN, REPLIES).start()
    turn = await episode.next_turn()

    async with turn.tools() as tools:
        await tools.call("items_add_item", {"name": "crate"})
    await turn.end("Added a crate.")
    await episode.result()

    [response] = Task.get_instance(episode.id).context["prompt_responses"]
    stored = json.loads(get_config().get_object_store().get(response["agent_trajectory_object_url"]))
    assert [span["name"] for span in stored] == ["items_add_item", "turn"]
    assert response["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_a_prompts_own_system_prompt_wins_over_the_agents(teardowns):
    episode = await _spec(ITEMS, ALICE, {**TURN, "system_prompt": "Count first."}, REPLIES).start()

    assert (await episode.next_turn()).system_prompt == "Count first."
