"""prompt_agent following a turn's trajectory live and storing it in chunks, against a live SDK
agent served in-process."""

import asyncio
import json
import logging

import httpx
import pytest
from agentenv_protocol.a2a_agent import (
    TRAJECTORY_V1,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    a2a_agent,
    enable,
)

from agent_env.config import configure
from agent_env.store.object_store import LocalFilesystemObjectStore
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.snapshot_utils import live_trajectory
from agent_env.task_step.task_steps import prompt_agent as pa
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

_PREFIX = "prompt_agent_trajectories/prompt_id=solve/"
_EVENTS = [{"type": "system", "subtype": "init"}, {"type": "assistant", "step": 0}]
_MORE = [{"type": "assistant", "step": 1}, {"type": "result", "result": "done"}]


def _agent(store: LocalFilesystemObjectStore, *, live: bool, wait_for_chunk: bool) -> AgentEnvAgent:
    """Appends two events, optionally waits until a chunk of them is stored, then two more."""

    @a2a_agent(
        identity=AgentIdentity(name="solver", description="test", version="1"),
        extensions=(enable(TRAJECTORY_V1, live=live),),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            request.trajectory.set_format("test-events/1")
            for event in _EVENTS:
                request.trajectory.append(event)
            if wait_for_chunk:
                async with asyncio.timeout(5):
                    while not _chunk_keys(store):
                        await asyncio.sleep(0.01)
            for event in _MORE:
                request.trajectory.append(event)
            return TaskResult.text("done")

    return Agent()


def _chunk_keys(store: LocalFilesystemObjectStore) -> list[str]:
    return sorted(key for key in store.list(_PREFIX) if key.endswith(".jsonl"))


class _Run:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path, *, live: bool, wait_for_chunk: bool) -> None:
        self.store = LocalFilesystemObjectStore(str(tmp_path))
        configure(object_store=self.store)
        for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task", "mark_closed", "get_conversation"):
            monkeypatch.setattr(pa.conversation_store, fn, lambda *a, **kw: None)
        app = _agent(self.store, live=live, wait_for_chunk=wait_for_chunk).create_app()
        self.card = app.state.agentenv_a2a.card.model_dump(mode="json", exclude_none=True)
        self.cursor_reads: list[dict] = []
        transport = httpx.ASGITransport(app=app)
        real_client = httpx.AsyncClient

        async def record(request: httpx.Request) -> None:
            if request.url.path == "/ext/trajectory":
                body = json.loads(await request.aread())
                if "after" in body:
                    self.cursor_reads.append(body)

        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda *a, **kw: real_client(*a, transport=transport, event_hooks={"request": [record]}, **kw),
        )

    async def execute(self, step: PromptAgentTaskStep, context: TaskStepContext | None = None) -> TaskStepContext:
        context = context or TaskStepContext(instance_id="ti-1")
        context.deployed_agents.append(
            DeployedAgent(agent_name="solver", api_url="http://agent.test", a2a_url="http://agent.test", a2a_card=self.card)
        )
        return await step.execute(context)

    def step(self, **kwargs) -> PromptAgentTaskStep:
        return PromptAgentTaskStep(
            id="solve", version=None, prompt="go", agent_name="solver", poll_interval_seconds=0,
            trajectory_output_prefix=self.store.object_url(_PREFIX), **kwargs,
        )


@pytest.mark.asyncio
async def test_a_live_turn_is_stored_in_chunks_while_it_runs(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path, live=True, wait_for_chunk=True)

    result = await run.execute(run.step(live_trajectory=True))

    final_url = result.prompt_responses[-1].agent_trajectory_object_url
    turn_id = run.store.get_object_key(final_url).removeprefix(f"{_PREFIX}trajectory-").removesuffix(".json")
    live = f"{_PREFIX}{turn_id}/live/"
    chunks = _chunk_keys(run.store)
    assert chunks and all(key.startswith(live) for key in chunks)
    assert chunks[0] == f"{live}00000000-{len(_EVENTS):08d}.jsonl"
    joined = [
        json.loads(line)
        for key in chunks
        for line in run.store.get(run.store.object_url(key)).decode().splitlines()
    ]
    assert joined == _EVENTS + _MORE == json.loads(run.store.get(final_url))
    assert json.loads(run.store.get(run.store.object_url(f"{live}meta.json"))) == {"format": "test-events/1"}
    assert json.loads(run.store.get(run.store.object_url(f"{live}end.json"))) == {
        "state": "completed",
        "next": len(_EVENTS + _MORE),
    }
    assert json.loads(run.store.get(run.store.object_url(f"{live}next.json"))) == {"turn": None}


@pytest.mark.asyncio
async def test_a_turn_whose_follower_stopped_early_is_marked_ended_in_the_turns_state(monkeypatch, tmp_path):
    async def stops_at_once(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(live_trajectory, "follow_trajectory", stops_at_once)
    run = _Run(monkeypatch, tmp_path, live=True, wait_for_chunk=False)

    result = await run.execute(run.step(live_trajectory=True))

    final_key = run.store.get_object_key(result.prompt_responses[-1].agent_trajectory_object_url)
    turn_id = final_key.removeprefix(f"{_PREFIX}trajectory-").removesuffix(".json")
    end_key = f"{_PREFIX}{turn_id}/live/end.json"
    next_key = f"{_PREFIX}{turn_id}/live/next.json"
    assert sorted(run.store.list(_PREFIX)) == sorted([final_key, end_key, next_key])
    assert json.loads(run.store.get(run.store.object_url(end_key))) == {
        "state": "completed",
        "next": 0,
    }


@pytest.mark.asyncio
async def test_a_step_that_fails_after_a_turn_marks_that_turn_last(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path, live=True, wait_for_chunk=False)

    def unavailable(*_args, **_kwargs) -> None:
        raise RuntimeError("conversation store unavailable")

    monkeypatch.setattr(pa.conversation_store, "complete_a2a_task", unavailable)

    with pytest.raises(RuntimeError, match="conversation store unavailable"):
        await run.execute(run.step(live_trajectory=True))

    (next_key,) = [key for key in run.store.list(_PREFIX) if key.endswith("/live/next.json")]
    assert json.loads(run.store.get(run.store.object_url(next_key))) == {"turn": None}


@pytest.mark.asyncio
async def test_without_live_trajectory_nothing_is_followed(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path, live=True, wait_for_chunk=False)

    result = await run.execute(run.step())

    assert run.cursor_reads == []
    assert _chunk_keys(run.store) == []
    assert json.loads(run.store.get(result.prompt_responses[-1].agent_trajectory_object_url)) == _EVENTS + _MORE


@pytest.mark.asyncio
async def test_a_step_params_override_turns_it_on(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path, live=True, wait_for_chunk=True)
    context = TaskStepContext(instance_id="ti-1")
    context.metadata["user_overrides"] = {"step_params": {"solve": {"live_trajectory": True}}}

    await run.execute(run.step(), context)

    assert _chunk_keys(run.store)


@pytest.mark.asyncio
async def test_an_agent_without_the_live_read_is_not_followed(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path, live=False, wait_for_chunk=False)

    result = await run.execute(run.step(live_trajectory=True))

    assert run.cursor_reads == []
    assert _chunk_keys(run.store) == []
    assert json.loads(run.store.get(result.prompt_responses[-1].agent_trajectory_object_url)) == _EVENTS + _MORE


def test_a_prefix_outside_the_store_is_not_followed(monkeypatch, tmp_path, caplog):
    run = _Run(monkeypatch, tmp_path, live=True, wait_for_chunk=False)
    step = PromptAgentTaskStep(id="solve", version=None, prompt="go", live_trajectory=True)

    with caplog.at_level(logging.WARNING):
        target = step._live_trajectory_target(
            TaskStepContext(), run.card, "http://agent.test", "s3://another-bucket/prefix/"
        )

    assert target is None
    assert "not following the live trajectory" in caplog.text


def test_live_trajectory_is_serialized_only_when_on():
    off = PromptAgentTaskStep(id="solve", version=None, prompt="go")
    on = PromptAgentTaskStep(id="solve", version=None, prompt="go", live_trajectory=True)

    assert "live_trajectory" not in off.to_dict()
    assert on.to_dict()["live_trajectory"] is True
    assert PromptAgentTaskStep.from_dict(on.to_dict()).live_trajectory is True
    assert PromptAgentTaskStep.from_dict(off.to_dict()).live_trajectory is False
