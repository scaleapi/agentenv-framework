import asyncio
import json
import time

import httpx
import pytest
from agentenv_protocol.a2a_agent import TrajectoryState

from agent_env.a2a_agent.trajectory_follower import (
    TrajectoryBatch,
    follow_trajectory,
    live_trajectory_endpoint,
)
from agent_env.store.object_store import LocalFilesystemObjectStore
from agent_env.task_step.snapshot_utils import live_trajectory
from agent_env.task_step.snapshot_utils.live_trajectory import (
    LiveTrajectoryChunks,
    LiveTurns,
    following_live_trajectory,
)

_ENDPOINT = "http://agent.test/ext/trajectory"


def _page(after: int, events: list, state: str, *, has_more: bool = False) -> dict:
    return {
        "context_id": "ctx",
        "task_id": "task-1",
        "state": state,
        "format": "test-events/1",
        "events": events,
        "next": after + len(events),
        "has_more": has_more,
    }


class _Agent:
    """Answers each cursor read from a script of responses, recording the ``after`` it was sent."""

    def __init__(self, *answers: httpx.Response) -> None:
        self.answers = list(answers)
        self.afters: list[int] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._answer))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.afters.append(json.loads(request.content)["after"])
        return self.answers.pop(0)


async def _follow(agent: _Agent) -> list[TrajectoryBatch]:
    batches: list[TrajectoryBatch] = []

    async def sink(batch: TrajectoryBatch) -> None:
        batches.append(batch)

    async with agent.client() as client:
        await follow_trajectory(
            _ENDPOINT, "task-1", sink, poll_interval_seconds=0, stop=asyncio.Event(), client=client
        )
    return batches


def test_the_live_endpoint_comes_from_the_card():
    def card(request: dict) -> dict:
        return {
            "capabilities": {
                "extensions": [
                    {
                        "uri": "urn:agentenv:trajectory/v1",
                        "params": {"endpoint": "/ext/trajectory", "methods": {"get": {"method": "POST", "request": request}}},
                    }
                ]
            }
        }

    live = card({"oneOf": [{"required": ["task_id"]}, {"required": ["task_id", "after"], "optional": ["limit"]}]})
    assert live_trajectory_endpoint(live, "http://agent.test") == _ENDPOINT
    assert live_trajectory_endpoint(card({"required": ["task_id"]}), "http://agent.test") is None
    assert live_trajectory_endpoint({}, "http://agent.test") is None


@pytest.mark.asyncio
async def test_pages_are_followed_until_the_task_ends_and_every_event_is_read():
    agent = _Agent(
        httpx.Response(200, json=_page(0, [], "pending")),
        httpx.Response(200, json=_page(0, [{"i": 0}], "running")),
        httpx.Response(200, json=_page(1, [{"i": 1}], "completed", has_more=True)),
        httpx.Response(200, json=_page(2, [{"i": 2}], "completed")),
    )

    batches = await _follow(agent)

    assert agent.afters == [0, 0, 1, 2]
    assert [(b.after, b.next, b.events, b.last) for b in batches] == [
        (0, 1, [{"i": 0}], False),
        (1, 2, [{"i": 1}], False),
        (2, 3, [{"i": 2}], True),
    ]
    assert batches[-1].state is TrajectoryState.COMPLETED


@pytest.mark.asyncio
async def test_an_ended_task_with_nothing_new_still_gets_a_last_batch():
    agent = _Agent(httpx.Response(200, json=_page(0, [], "failed")))

    batches = await _follow(agent)

    assert [(b.events, b.state, b.last) for b in batches] == [([], TrajectoryState.FAILED, True)]


@pytest.mark.asyncio
async def test_a_server_error_is_retried(monkeypatch):
    monkeypatch.setattr("agent_env.a2a_agent.trajectory_follower._MAX_BACKOFF_SECONDS", 0)
    agent = _Agent(
        httpx.Response(503),
        httpx.Response(200, json=_page(0, [{"i": 0}], "completed")),
    )

    batches = await _follow(agent)

    assert agent.afters == [0, 0]
    assert batches[-1].events == [{"i": 0}]


@pytest.mark.asyncio
async def test_a_task_the_agent_no_longer_knows_ends_following():
    agent = _Agent(httpx.Response(404, json={"detail": "Unknown task"}))

    assert await _follow(agent) == []


@pytest.mark.asyncio
async def test_a_client_error_is_raised():
    agent = _Agent(httpx.Response(400, json={"detail": "bad"}))

    with pytest.raises(httpx.HTTPStatusError):
        await _follow(agent)


@pytest.mark.asyncio
async def test_chunks_meta_and_end_are_stored_once(tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path))
    chunks = LiveTrajectoryChunks(store, "prefix/", "turn")
    batch = TrajectoryBatch(
        after=0, next=2, events=[{"i": 0}, {"i": 1}], format="test-events/1",
        state=TrajectoryState.COMPLETED, last=True,
    )

    await chunks(batch)
    await LiveTrajectoryChunks(store, "prefix/", "turn")(batch)

    assert sorted(store.list("prefix/")) == [
        "prefix/turn/live/00000000-00000002.jsonl",
        "prefix/turn/live/end.json",
        "prefix/turn/live/meta.json",
    ]
    assert store.get(store.object_url("prefix/turn/live/00000000-00000002.jsonl")) == b'{"i":0}\n{"i":1}\n'
    assert json.loads(store.get(store.object_url("prefix/turn/live/end.json"))) == {"state": "completed", "next": 2}


@pytest.mark.asyncio
async def test_a_failing_write_is_retried_then_raised(monkeypatch, tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path))
    attempts: list[str] = []

    def failing_put(key, body, *, content_type):
        attempts.append(key)
        raise OSError("disk full")

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(store, "put", failing_put)
    monkeypatch.setattr(live_trajectory.asyncio, "sleep", no_sleep)
    batch = TrajectoryBatch(after=0, next=1, events=[{"i": 0}], format=None, state=TrajectoryState.RUNNING, last=False)

    with pytest.raises(OSError):
        await LiveTrajectoryChunks(store, "prefix/", "turn")(batch)
    assert len(attempts) == live_trajectory._WRITE_ATTEMPTS


def _agent_answering(monkeypatch, *answers: dict | httpx.Response) -> None:
    """Every httpx client reaches an agent that answers each cursor read with the next answer, a
    page or a response, then keeps answering with the last one."""
    remaining = list(answers)

    def answer(_request: httpx.Request) -> httpx.Response:
        given = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return given if isinstance(given, httpx.Response) else httpx.Response(200, json=given)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, real=httpx.AsyncClient, **kw: real(*a, transport=httpx.MockTransport(answer), **kw),
    )


def _end(store: LocalFilesystemObjectStore) -> dict:
    return json.loads(store.get(store.object_url("prefix/turn/live/end.json")))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "state"),
    [(RuntimeError("turn failed"), "failed"), (asyncio.CancelledError(), "canceled")],
)
async def test_a_failed_turn_stops_its_follower_without_draining_and_marks_it_ended(
    monkeypatch, tmp_path, error, state
):
    """An agent that never ends: only a drain would keep the follower reading."""
    _agent_answering(monkeypatch, _page(0, [], "running"))
    store = LocalFilesystemObjectStore(str(tmp_path))
    started = time.monotonic()

    with pytest.raises(type(error)):
        async with following_live_trajectory(
            _ENDPOINT, store, "prefix/", "turn", poll_interval_seconds=0
        ) as turn:
            turn.follow("task-1")
            await asyncio.sleep(0.05)
            raise error

    assert time.monotonic() - started < 2
    assert store.list("prefix/") == ["prefix/turn/live/end.json"]
    assert _end(store) == {"state": state, "next": 0}


@pytest.mark.asyncio
async def test_a_cancelled_turn_is_marked_ended_after_the_events_it_stored(monkeypatch, tmp_path):
    _agent_answering(monkeypatch, _page(0, [{"i": 0}, {"i": 1}], "running"), _page(2, [], "running"))
    store = LocalFilesystemObjectStore(str(tmp_path))

    with pytest.raises(asyncio.CancelledError):
        async with following_live_trajectory(
            _ENDPOINT, store, "prefix/", "turn", poll_interval_seconds=0
        ) as turn:
            turn.follow("task-1")
            async with asyncio.timeout(5):
                while "prefix/turn/live/00000000-00000002.jsonl" not in store.list("prefix/"):
                    await asyncio.sleep(0.01)
            raise asyncio.CancelledError()

    assert _end(store) == {"state": "canceled", "next": 2}


@pytest.mark.asyncio
async def test_a_turn_the_follower_saw_end_keeps_its_state(monkeypatch, tmp_path):
    _agent_answering(monkeypatch, _page(0, [{"i": 0}], "completed"))
    store = LocalFilesystemObjectStore(str(tmp_path))

    with pytest.raises(RuntimeError):
        async with following_live_trajectory(
            _ENDPOINT, store, "prefix/", "turn", poll_interval_seconds=0
        ) as turn:
            turn.follow("task-1")
            async with asyncio.timeout(5):
                while "prefix/turn/live/end.json" not in store.list("prefix/"):
                    await asyncio.sleep(0.01)
            raise RuntimeError("the step failed after the turn ended")

    assert _end(store) == {"state": "completed", "next": 1}


async def _until_stored(store: LocalFilesystemObjectStore, key: str) -> None:
    async with asyncio.timeout(5):
        while key not in store.list("prefix/"):
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stops_with",
    [
        pytest.param(httpx.Response(400, json={"detail": "bad"}), id="client-error"),
        pytest.param(httpx.Response(404, json={"detail": "Unknown task"}), id="task-evicted"),
        pytest.param(_page(2, [], "running"), id="drain-timeout"),
    ],
)
async def test_a_turn_the_follower_stopped_before_its_end_is_marked_ended_in_the_reported_state(
    monkeypatch, tmp_path, stops_with
):
    monkeypatch.setattr(live_trajectory, "_DRAIN_SECONDS", 0.1)
    _agent_answering(monkeypatch, _page(0, [{"i": 0}, {"i": 1}], "running"), stops_with)
    store = LocalFilesystemObjectStore(str(tmp_path))

    async with following_live_trajectory(_ENDPOINT, store, "prefix/", "turn", poll_interval_seconds=0) as turn:
        turn.follow("task-1")
        await _until_stored(store, "prefix/turn/live/00000000-00000002.jsonl")
        turn.ended("completed")

    assert _end(store) == {"state": "completed", "next": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize(("task_state", "state"), [("canceled", "canceled"), ("failed", "failed"), ("rejected", "failed")])
async def test_a_reported_task_state_ends_the_turn_as_its_trajectory_state(monkeypatch, tmp_path, task_state, state):
    _agent_answering(monkeypatch, httpx.Response(404, json={"detail": "Unknown task"}))
    store = LocalFilesystemObjectStore(str(tmp_path))

    async with following_live_trajectory(_ENDPOINT, store, "prefix/", "turn", poll_interval_seconds=0) as turn:
        turn.follow("task-1")
        turn.ended(task_state)

    assert _end(store) == {"state": state, "next": 0}


def _next(store: LocalFilesystemObjectStore, turn: str) -> dict:
    return json.loads(store.get(store.object_url(f"prefix/{turn}/live/next.json")))


@pytest.mark.asyncio
async def test_each_turn_names_the_next_and_the_last_is_marked_last(monkeypatch, tmp_path):
    _agent_answering(monkeypatch, _page(0, [{"i": 0}], "completed"))
    store = LocalFilesystemObjectStore(str(tmp_path))
    turns = LiveTurns(_ENDPOINT, store, "prefix/", poll_interval_seconds=0)

    for turn_id in ("turn-1", "turn-2"):
        async with turns.following(turn_id) as turn:
            turn.follow(f"task-{turn_id}")
            await _until_stored(store, f"prefix/{turn_id}/live/end.json")
            turn.ended("completed")
    assert "prefix/turn-2/live/next.json" not in store.list("prefix/")
    await turns.end()

    assert _next(store, "turn-1") == {"turn": "turn-2"}
    assert _next(store, "turn-2") == {"turn": None}


@pytest.mark.asyncio
async def test_a_turn_never_sent_is_not_linked_and_the_last_sent_is_marked_last(monkeypatch, tmp_path):
    _agent_answering(monkeypatch, _page(0, [{"i": 0}], "completed"))
    store = LocalFilesystemObjectStore(str(tmp_path))
    turns = LiveTurns(_ENDPOINT, store, "prefix/", poll_interval_seconds=0)
    async with turns.following("turn-1") as turn:
        turn.follow("task-1")
        turn.ended("completed")

    with pytest.raises(RuntimeError):
        async with turns.following("turn-2"):
            raise RuntimeError("the message could not be sent")
    await turns.end()

    assert _next(store, "turn-1") == {"turn": None}
    assert not any(key.startswith("prefix/turn-2/") for key in store.list("prefix/"))


@pytest.mark.asyncio
async def test_a_step_that_sent_no_turn_marks_none(tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path))

    await LiveTurns(_ENDPOINT, store, "prefix/", poll_interval_seconds=0).end()

    assert store.list("prefix/") == []
