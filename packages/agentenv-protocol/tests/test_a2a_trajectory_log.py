from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

import pytest
from starlette.testclient import TestClient

from agentenv_protocol.a2a_agent import (
    TRAJECTORY_V1,
    AgentEnvAgent,
    AgentIdentity,
    ContextTrajectoryRequest,
    DefaultExtensionHandlers,
    TaskEventsTrajectoryRequest,
    TaskObjectTrajectoryRequest,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TaskTrajectoryRequest,
    TrajectoryLog,
    TrajectoryState,
    Uploaded,
    a2a_agent,
    build_registry,
    card_request_accepts,
    enable,
    extension,
)
from agentenv_protocol.a2a_agent import framework
from agentenv_protocol.a2a_agent._trajectory_log import TaskTrajectories
from agentenv_protocol.a2a_agent.framework import _SdkServices
from agentenv_protocol.a2a_agent.tasks import v1 as tasks_v1

_IDENTITY = AgentIdentity(name="live", description="test", version="1")


class _Gates:
    """Lets a test hold the agent before it appends (``go``) and before it returns
    (``release``); both are set by default."""

    def __init__(self, *, go: bool = True, release: bool = True) -> None:
        self.go = threading.Event()
        self.release = threading.Event()
        if go:
            self.go.set()
        if release:
            self.release.set()


async def _wait(event: threading.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0.01)


def _live_agent(gates: _Gates, *, events: int = 3, result: str = "done") -> AgentEnvAgent:
    @a2a_agent(identity=_IDENTITY, extensions=(enable(TRAJECTORY_V1, live=True),))
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            request.trajectory.set_format("test-events/1")
            await _wait(gates.go)
            for index in range(events):
                request.trajectory.append({"text": request.parts[0].text, "index": index})
            await _wait(gates.release)
            if result == "failed":
                return TaskResult.failure("agent.failed", "it failed")
            if result == "raise":
                raise RuntimeError("boom")
            return TaskResult.text(result)

    return Agent()


def _send(client: TestClient, text: str, message_id: str, context_id: str = "context-1") -> str:
    response = client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": message_id,
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "messageId": message_id,
                    "role": "user",
                    "contextId": context_id,
                    "parts": [{"kind": "text", "text": text}],
                },
                "configuration": {"blocking": False},
            },
        },
    )
    return response.json()["result"]["id"]


def _task_state(client: TestClient, task_id: str) -> str:
    response = client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": "get", "method": "tasks/get", "params": {"id": task_id}},
    )
    return response.json()["result"]["status"]["state"]


def _cancel(client: TestClient, task_id: str) -> dict[str, Any]:
    return client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "cancel",
            "method": "tasks/cancel",
            "params": {"id": task_id},
        },
    ).json()


def _trajectory(client: TestClient, payload: dict[str, Any]) -> Any:
    return client.post("/ext/trajectory", json=payload)


def _within(seconds: float, condition) -> bool:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _open(trajectories: TaskTrajectories, task_id: str, context_id: str) -> TrajectoryLog:
    log = trajectories.register(task_id, context_id)
    log.set_format("test-events/1")
    return log


def _events(text: str, count: int) -> list[dict[str, Any]]:
    return [{"text": text, "index": index} for index in range(count)]


def test_live_card_advertises_the_cursor_reads() -> None:
    app = _live_agent(_Gates()).create_app()
    card = app.state.agentenv_a2a.card.model_dump(mode="json", exclude_none=True)
    (extension_card,) = [
        item for item in card["capabilities"]["extensions"] if item["uri"] == TRAJECTORY_V1.uri
    ]
    request = extension_card["params"]["methods"]["get"]["request"]

    assert request == {
        "oneOf": [
            {"required": ["task_id"]},
            {"required": ["task_id", "objects"]},
            {"required": ["context_id"]},
            {"required": ["context_id", "objects"]},
            {"required": ["task_id", "after"], "optional": ["limit"]},
            {"required": ["context_id", "after"], "optional": ["limit"]},
        ]
    }
    assert card_request_accepts(request, {"task_id"})
    assert card_request_accepts(request, {"task_id", "after"})
    assert card_request_accepts(request, {"task_id", "after", "limit"})
    assert card_request_accepts(request, {"context_id", "after"})
    with pytest.raises(TypeError, match="live must be a boolean"):
        enable(TRAJECTORY_V1, live="yes")


def test_a_full_get_override_must_take_the_cursor_reads_when_live() -> None:
    class Agent:
        @extension(TRAJECTORY_V1.get)
        async def get(self, request: TaskTrajectoryRequest | TaskObjectTrajectoryRequest):
            return {"trajectory": []}

    build_registry(Agent(), [enable(TRAJECTORY_V1)])
    with pytest.raises(ValueError, match="must annotate"):
        build_registry(Agent(), [enable(TRAJECTORY_V1, live=True)])


def test_task_read_follows_a_running_task() -> None:
    gates = _Gates(go=False, release=False)
    with TestClient(_live_agent(gates).create_app()) as client:
        task_id = _send(client, "hello", "message-1")
        pending = _trajectory(client, {"task_id": task_id, "after": 0})
        assert pending.status_code == 200
        assert {key: pending.json()[key] for key in ("task_id", "state", "events", "next")} == {
            "task_id": task_id,
            "state": "pending",
            "events": [],
            "next": 0,
        }

        gates.go.set()
        assert _within(
            5,
            lambda: _trajectory(client, {"task_id": task_id, "after": 0}).json()["next"] == 3,
        )
        first = _trajectory(client, {"task_id": task_id, "after": 0, "limit": 2}).json()
        assert first["state"] == "running"
        assert first["events"] == _events("hello", 3)[:2]
        assert (first["next"], first["has_more"]) == (2, True)
        rest = _trajectory(client, {"task_id": task_id, "after": 2}).json()
        assert rest["events"] == _events("hello", 3)[2:]
        assert (rest["next"], rest["has_more"]) == (3, False)
        assert _trajectory(client, {"task_id": task_id, "after": 3}).json()["events"] == []
        assert _trajectory(client, {"task_id": task_id, "after": 4}).status_code == 400
        assert _trajectory(client, {"task_id": task_id}).status_code == 404

        gates.release.set()
        assert _within(5, lambda: _task_state(client, task_id) == "completed")
        final = _trajectory(client, {"task_id": task_id})
        assert final.json() == {"trajectory": _events("hello", 3)}
        done = _trajectory(client, {"task_id": task_id, "after": 0}).json()
        assert done["state"] == "completed"
        assert done["events"] == _events("hello", 3)


@pytest.mark.parametrize(
    "payload",
    [
        {"task_id": "unknown", "after": 0},
        {"context_id": "unknown", "after": 0},
        {"context_id": "unknown"},
    ],
)
def test_unknown_ids_are_not_found(payload: dict[str, Any]) -> None:
    with TestClient(_live_agent(_Gates()).create_app()) as client:
        assert _trajectory(client, payload).status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"task_id": "task", "after": True},
        {"task_id": "task", "after": "3"},
        {"task_id": "task", "after": -1},
        {"task_id": "task", "after": 0, "limit": 0},
        {"task_id": "task", "after": 0, "limit": 1_001},
        {"context_id": "context", "after": 0, "limit": 1_001},
    ],
)
def test_cursor_fields_are_strict(payload: dict[str, Any]) -> None:
    with TestClient(_live_agent(_Gates()).create_app()) as client:
        assert _trajectory(client, payload).status_code == 400


@pytest.mark.parametrize(("result", "a2a_state"), [("failed", "failed"), ("raise", "failed")])
def test_a_failed_task_keeps_its_events(result: str, a2a_state: str) -> None:
    with TestClient(_live_agent(_Gates(), result=result).create_app()) as client:
        task_id = _send(client, "hello", "message-1")
        assert _within(5, lambda: _task_state(client, task_id) == a2a_state)
        read = _trajectory(client, {"task_id": task_id, "after": 0}).json()
        assert read["state"] == "failed"
        assert read["events"] == _events("hello", 3)
        assert _trajectory(client, {"task_id": task_id}).json() == {
            "trajectory": _events("hello", 3)
        }


def test_cancel_seals_the_log_canceled_with_its_cleanup_events() -> None:
    @a2a_agent(identity=_IDENTITY, extensions=(enable(TRAJECTORY_V1, live=True),))
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            request.trajectory.set_format("test-events/1")
            request.trajectory.append({"step": "working"})
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                request.trajectory.append({"step": "cleanup"})
                raise
            return TaskResult.text("unreachable")

    def read(client: TestClient, task_id: str) -> dict[str, Any]:
        return _trajectory(client, {"task_id": task_id, "after": 0}).json()

    with TestClient(Agent().create_app()) as client:
        task_id = _send(client, "hang", "message-1")
        assert _within(5, lambda: read(client, task_id)["state"] == "running")

        assert _cancel(client, task_id)["result"]["status"]["state"] == "canceled"

        assert _within(5, lambda: read(client, task_id)["state"] == "canceled")
        assert read(client, task_id)["events"] == [{"step": "working"}, {"step": "cleanup"}]


def test_a_cancel_the_agent_swallows_still_seals_canceled() -> None:
    services = _SdkServices((enable(TRAJECTORY_V1, live=True),), None)
    trajectories = services.task_trajectories
    log = _open(trajectories, "task-1", "context-1")
    trajectories.start("task-1")
    log.append({"step": 1})

    trajectories.request_cancel("task-1")
    trajectories.seal("task-1", TrajectoryState.COMPLETED)

    page = trajectories.task_page("task-1", 0, 10, 1_000)
    assert page is not None and page.state is TrajectoryState.CANCELED


def test_a_native_trajectory_is_the_final_record(caplog: pytest.LogCaptureFixture) -> None:
    @a2a_agent(identity=_IDENTITY, extensions=(enable(TRAJECTORY_V1, live=True),))
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            request.trajectory.set_format("test-events/1")
            request.trajectory.append({"live": True})
            return (
                TaskResult.builder()
                .succeeded()
                .add_text("done")
                .native_trajectory(format="native/1", payload=[{"native": True}])
                .build()
            )

    with caplog.at_level(logging.WARNING), TestClient(Agent().create_app()) as client:
        task_id = _send(client, "hello", "message-1")
        assert _within(5, lambda: _task_state(client, task_id) == "completed")

        assert _trajectory(client, {"task_id": task_id}).json() == {
            "trajectory": [{"native": True}]
        }
        assert _trajectory(client, {"task_id": task_id, "after": 0}).json()["events"] == [
            {"live": True}
        ]
    assert "the native trajectory is its final record" in caplog.text


def test_context_reads_span_turns_in_order() -> None:
    with TestClient(_live_agent(_Gates(), events=2).create_app()) as client:
        first = _send(client, "one", "message-1")
        assert _within(5, lambda: _task_state(client, first) == "completed")
        second = _send(client, "two", "message-2")
        assert _within(5, lambda: _task_state(client, second) == "completed")

        everything = _events("one", 2) + _events("two", 2)
        read = _trajectory(client, {"context_id": "context-1", "after": 0}).json()
        assert read == {
            "context_id": "context-1",
            "task_id": second,
            "state": "completed",
            "format": "test-events/1",
            "events": everything,
            "next": 4,
            "has_more": False,
        }
        tail = _trajectory(client, {"context_id": "context-1", "after": 1, "limit": 2}).json()
        assert tail["events"] == everything[1:3]
        assert (tail["next"], tail["has_more"]) == (3, True)
        assert _trajectory(client, {"context_id": "context-1"}).json() == {
            "trajectory": everything
        }


def test_an_agent_context_handler_wins_over_the_framework() -> None:
    @a2a_agent(identity=_IDENTITY, extensions=(enable(TRAJECTORY_V1, live=True),))
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            request.trajectory.set_format("test-events/1")
            request.trajectory.append({"live": True})
            return TaskResult.text("done")

        @extension(TRAJECTORY_V1.get.context)
        async def get_context(self, request: ContextTrajectoryRequest) -> dict[str, Any]:
            return {"trajectory": "the agent's own"}

    with TestClient(Agent().create_app()) as client:
        task_id = _send(client, "hello", "message-1")
        assert _within(5, lambda: _task_state(client, task_id) == "completed")

        assert _trajectory(client, {"context_id": "context-1"}).json() == {
            "trajectory": "the agent's own"
        }
        read = _trajectory(client, {"context_id": "context-1", "after": 0}).json()
        assert read["events"] == [{"live": True}]


def test_a_streaming_run_appends_the_same_way() -> None:
    @a2a_agent(identity=_IDENTITY, extensions=(enable(TRAJECTORY_V1, live=True),))
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> AsyncIterator[Any]:
            request.trajectory.set_format("test-events/1")
            request.trajectory.append({"step": 1})
            yield TaskProgress.text("working")
            request.trajectory.append({"step": 2})
            yield TaskResult.text("done")

    with TestClient(Agent().create_app()) as client:
        task_id = _send(client, "hello", "message-1")
        assert _within(5, lambda: _task_state(client, task_id) == "completed")
        read = _trajectory(client, {"task_id": task_id, "after": 0}).json()
        assert read["state"] == "completed"
        assert read["events"] == [{"step": 1}, {"step": 2}]


@pytest.mark.asyncio
async def test_default_handlers_delegate_the_cursor_read() -> None:
    services = _SdkServices((enable(TRAJECTORY_V1, live=True),), None)
    log = _open(services.task_trajectories, "task-1", "context-1")
    services.task_trajectories.start("task-1")
    log.append({"step": 1})

    result = await DefaultExtensionHandlers(services).call(
        TRAJECTORY_V1.get, TaskEventsTrajectoryRequest(task_id="task-1", after=0)
    )

    assert result.events == [{"step": 1}]
    assert result.state is TrajectoryState.RUNNING


def test_an_appended_event_never_changes() -> None:
    log = TrajectoryLog()
    log.set_format("test-events/1")
    event = {"nested": {"value": 1}}
    log.append(event)
    event["nested"]["value"] = 2

    assert log._events == [b'{"nested":{"value":1}}']
    for bad in ({"value": float("nan")}, {"value": {1, 2}}):
        with pytest.raises((TypeError, ValueError)):
            log.append(bad)
    with pytest.raises(TypeError, match="JSON object"):
        log.append(["not", "an", "object"])  # type: ignore[arg-type]
    assert len(log) == 1
    with pytest.raises(ValueError, match="non-empty"):
        log.set_format(" ")


def test_the_format_cannot_change_once_an_event_is_appended() -> None:
    log = TrajectoryLog()
    log.set_format("first/1")
    log.set_format("second/1")
    log.append({"step": 1})

    log.set_format("second/1")
    with pytest.raises(RuntimeError, match="cannot change"):
        log.set_format("third/1")
    assert log.format == "second/1"


def test_a_sealed_log_refuses_more_events() -> None:
    trajectories = TaskTrajectories()
    log = _open(trajectories, "task-1", "context-1")
    log.append({"step": 1})
    trajectories.seal("task-1", TrajectoryState.COMPLETED)

    with pytest.raises(RuntimeError, match="can no longer change"):
        log.append({"step": 2})
    with pytest.raises(RuntimeError, match="can no longer change"):
        log.set_format("late/1")
    assert trajectories.final("task-1") == b'[{"step":1}]'


def test_reads_never_return_a_partial_event() -> None:
    trajectories = TaskTrajectories()
    log = _open(trajectories, "task-1", "context-1")
    trajectories.start("task-1")
    seen: list[bytes] = []
    for index in range(50):
        log.append({"index": index, "padding": "x" * index})
        page = trajectories.task_page("task-1", len(seen), 7, 1_000)
        assert page is not None
        seen.extend(page.events)
    while len(seen) < 50:
        page = trajectories.task_page("task-1", len(seen), 7, 1_000)
        assert page is not None
        seen.extend(page.events)

    assert [json.loads(item) for item in seen] == [
        {"index": index, "padding": "x" * index} for index in range(50)
    ]


def test_the_byte_budget_still_returns_one_event() -> None:
    trajectories = TaskTrajectories()
    log = _open(trajectories, "task-1", "context-1")
    trajectories.start("task-1")
    log.append({"big": "x" * 100})
    log.append({"big": "y" * 100})

    page = trajectories.task_page("task-1", 0, 10, 10)

    assert page is not None and len(page.events) == 1


def test_a_task_that_starts_later_moves_without_shifting_positions() -> None:
    trajectories = TaskTrajectories()
    queued = _open(trajectories, "queued", "context-1")
    running = _open(trajectories, "running", "context-1")
    trajectories.start("running")
    running.append({"from": "running"})
    trajectories.start("queued")
    queued.append({"from": "queued"})

    page = trajectories.context_page("context-1", 0, 10, 1_000)

    assert page is not None
    assert [json.loads(item) for item in page.events] == [
        {"from": "running"},
        {"from": "queued"},
    ]
    assert page.task_id == "running"


def test_eviction_drops_completed_contexts_but_never_running_ones() -> None:
    trajectories = TaskTrajectories(max_contexts=2)
    running = _open(trajectories, "running", "context-running")
    trajectories.start("running")
    running.append({"step": 1})
    for number in (1, 2, 3):
        trajectories.register(f"done-{number}", f"context-{number}")
        trajectories.seal(f"done-{number}", TrajectoryState.COMPLETED, native=b"[1]")

    assert trajectories.task_page("running", 0, 10, 1_000) is not None
    assert trajectories.final("done-1") is None
    assert trajectories.final("done-2") is None
    assert trajectories.final("done-3") == b"[1]"


def test_the_context_whose_task_just_ended_is_kept_for_its_final_read() -> None:
    trajectories = TaskTrajectories(max_contexts=1)
    log = _open(trajectories, "ending", "context-ending")
    trajectories.start("ending")
    log.append({"step": 1})
    for number in (1, 2):
        trajectories.register(f"queued-{number}", f"context-queued-{number}")

    trajectories.seal("ending", TrajectoryState.COMPLETED)

    assert trajectories.final("ending") == b'[{"step":1}]'


def test_a_large_running_trajectory_warns_once(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks_v1, "_LARGE_LIVE_TRAJECTORY_BYTES", 10)
    log = _open(TaskTrajectories(), "task-1", "context-1")

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            log.append({"padding": "x" * 20})

    assert caplog.text.count("trajectory of running task task-1 exceeds") == 1


def test_appending_before_set_format_is_refused() -> None:
    log = TrajectoryLog()

    with pytest.raises(RuntimeError, match="set_format"):
        log.append({"step": 1})
    assert len(log) == 0


def test_a_context_read_keeps_the_format_while_a_new_turn_is_pending() -> None:
    trajectories = TaskTrajectories()
    first = _open(trajectories, "first", "context-1")
    trajectories.start("first")
    first.append({"step": 1})
    trajectories.seal("first", TrajectoryState.COMPLETED)
    trajectories.register("second", "context-1")
    trajectories.start("second")

    page = trajectories.context_page("context-1", 0, 10, 1_000)

    assert page is not None
    assert (page.task_id, page.state, page.format) == (
        "second",
        TrajectoryState.PENDING,
        "test-events/1",
    )


# Events shaped like a CLI's: nested objects, non-ASCII and line-separator text, floats, nulls.
_NATIVE_EVENTS = [
    {"type": "system", "subtype": "init", "tools": ["Read", "mcp__db__query"], "cwd": "/app"},
    {"type": "assistant", "message": {"content": [{"type": "text", "text": "Prüfe die Daten — ✓ a\u2028b"}]}},
    {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok", "is_error": False}]}},
    {"type": "result", "subtype": "success", "result": "done", "total_cost_usd": 0.0123, "usage": None},
]


def test_the_log_is_byte_identical_to_the_native_trajectory_it_replaces(monkeypatch):
    """An agent that appends its events and one that returns them as a native trajectory answer the
    task read with the same trajectory and upload the same bytes, so porting an agent to the log does
    not change its final trajectory."""

    @a2a_agent(identity=_IDENTITY, extensions=(enable(TRAJECTORY_V1),))
    class Native(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return (
                TaskResult.builder()
                .succeeded()
                .add_text("done")
                .native_trajectory(format="test-events/1", payload=_NATIVE_EVENTS)
                .build()
            )

    @a2a_agent(identity=_IDENTITY, extensions=(enable(TRAJECTORY_V1, live=True),))
    class Live(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            request.trajectory.set_format("test-events/1")
            for event in _NATIVE_EVENTS:
                request.trajectory.append(event)
            return TaskResult.text("done")

    uploaded: list[bytes] = []

    async def capture_upload(_target, body: bytes):
        uploaded.append(body)
        return Uploaded(size_bytes=len(body))

    monkeypatch.setattr(framework, "upload", capture_upload)
    grant = {
        "media_type": "application/json",
        "max_bytes": 1_000_000,
        "write": {
            "kind": "http-put",
            "url": "https://objects.example.test/write",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        },
    }
    inline: list[Any] = []
    for agent in (Native(), Live()):
        with TestClient(agent.create_app()) as client:
            task_id = _send(client, "go", "message-1")
            assert _within(5, lambda: _task_state(client, task_id) == "completed")
            inline.append(_trajectory(client, {"task_id": task_id}).json())
            response = _trajectory(client, {"task_id": task_id, "objects": {"trajectory": grant}})
            assert response.status_code == 200, response.text

    assert inline[0] == inline[1] == {"trajectory": _NATIVE_EVENTS}
    assert uploaded[0] == uploaded[1]
    assert json.loads(uploaded[1]) == _NATIVE_EVENTS
