"""A prompt step leaves no conversation active: one it exits by an error or a cancellation is closed, and a re-run
closes what an earlier run of the step left behind, unless the step shares its conversation through `context_id`."""

import asyncio
import logging

import pytest
from a2a.types import TaskState

from agent_env.a2a_agent import conversation_store as cs
from agent_env.store import Filter, get_config
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

INSTANCE = "ti-1"


@pytest.fixture(autouse=True)
def conversation_indexes(local_stores):
    """As deployed: the unique conversation id is what makes a shared conversation's re-run reuse it."""
    cs.ensure_indexes()


def _wire(monkeypatch, replies: list) -> None:
    """The solver's n-th poll returns a completed reply, or raises when ``replies[n]`` is an exception."""
    turns = iter(replies)

    async def send(url, parts, message_id, context_id, timeout):
        return f"task-{message_id}", None

    async def poll(url, task_id, timeout, interval, **_):
        reply = next(turns)
        if isinstance(reply, BaseException):
            raise reply
        return {"status": {"state": TaskState.completed, "message": {"parts": [{"kind": "text", "text": reply}]}}}

    monkeypatch.setattr(pa.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa.protocol, "poll_a2a_task", poll)


def _context() -> TaskStepContext:
    ctx = TaskStepContext(instance_id=INSTANCE)
    ctx.deployed_agents.append(DeployedAgent(agent_name="solver", api_url="http://solver", a2a_url="http://solver",
                                             a2a_card={}))
    return ctx


def _step(**kwargs) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver", **kwargs)


def _conversations() -> list[dict]:
    docs = get_config().get_document_store().query(cs._COLLECTION_NAME, Filter.of(task_instance_id=INSTANCE))
    return sorted(docs, key=lambda d: d["created_at_utc"])


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("agent unreachable"), asyncio.CancelledError()])
async def test_a_step_that_exits_by_an_error_or_a_cancellation_closes_its_conversation(local_stores, monkeypatch,
                                                                                        error):
    _wire(monkeypatch, [error])

    with pytest.raises(type(error)) as raised:
        await _step().execute(_context())

    assert raised.value is error
    [conversation] = _conversations()
    assert conversation["status"] == "closed"
    assert conversation["step_id"] == "solve"
    assert conversation["a2a_tasks"][0]["state"] == TaskState.canceled.value


@pytest.mark.asyncio
async def test_a_close_that_fails_leaves_the_steps_own_error(local_stores, monkeypatch, caplog):
    error = RuntimeError("agent unreachable")
    _wire(monkeypatch, [error])

    def store_down(conversation_id):
        raise ConnectionError("store down")

    monkeypatch.setattr(pa.conversation_store, "mark_closed", store_down)

    with caplog.at_level(logging.WARNING, logger=pa.logger.name), pytest.raises(RuntimeError) as raised:
        await _step().execute(_context())

    assert raised.value is error
    assert "could not close conversation" in caplog.text and "store down" in caplog.text


@pytest.mark.asyncio
async def test_a_rerun_closes_the_conversation_a_dead_run_left_active(local_stores, monkeypatch):
    # The first run's worker dies mid-step, so nothing it would run on the way out runs.
    _wire(monkeypatch, [RuntimeError("worker died"), "done"])
    with monkeypatch.context() as dying:
        dying.setattr(PromptAgentTaskStep, "_close_left_open", lambda self, conversation_id: None)
        with pytest.raises(RuntimeError):
            await _step().execute(_context())
    [left] = _conversations()
    assert left["status"] == "active"

    ctx = await _step().execute(_context())

    earlier, rerun = _conversations()
    assert earlier["conversation_id"] == left["conversation_id"]
    assert earlier["status"] == "closed"
    assert rerun["conversation_id"] == ctx.metadata["a2a_conversations"]["solve"] != earlier["conversation_id"]
    assert rerun["status"] == "closed"
    assert ctx.prompt_responses[-1].response == "done"


@pytest.mark.asyncio
async def test_a_rerun_that_fails_before_its_own_conversation_still_closes_the_one_left_active(local_stores,
                                                                                              monkeypatch):
    _wire(monkeypatch, [RuntimeError("worker died")])
    with monkeypatch.context() as dying:
        dying.setattr(PromptAgentTaskStep, "_close_left_open", lambda self, conversation_id: None)
        with pytest.raises(RuntimeError):
            await _step().execute(_context())

    async def agent_unreachable(*args, **kwargs):
        raise ConnectionError("agent unreachable")

    monkeypatch.setattr(pa, "prescribe_formats", agent_unreachable)
    with pytest.raises(ConnectionError):
        await _step().execute(_context())

    [left] = _conversations()
    assert left["status"] == "closed"


@pytest.mark.asyncio
async def test_a_shared_conversation_is_left_as_before(local_stores, monkeypatch):
    _wire(monkeypatch, [RuntimeError("agent unreachable"), "done"])

    with pytest.raises(RuntimeError):
        await _step(context_id="shared").execute(_context())
    [shared] = _conversations()
    assert shared["status"] == "active"
    assert "step_id" not in shared

    ctx = await _step(context_id="shared").execute(_context())

    [reused] = _conversations()
    assert reused["conversation_id"] == shared["conversation_id"] == ctx.metadata["a2a_conversations"]["solve"]
    assert reused["status"] == "closed"
