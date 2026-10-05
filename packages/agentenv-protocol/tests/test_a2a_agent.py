from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import FrozenInstanceError
from functools import wraps
from typing import Any, Union
from unittest.mock import MagicMock

import httpx
import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCapabilities as UpstreamAgentCapabilities,
)
from a2a.types import AgentExtension
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.types import (
    TextPart as A2ATextPart,
)
from agentenv_protocol.a2a_agent import (
    AGENT_CONFIG_V1,
    ATTRIBUTION_PROBE_V1,
    MCP_CONFIG_V1,
    PEER_AGENTS_V1,
    SKILL_CONFIG_V1,
    SNAPSHOT_V1,
    STAGING_V1_URI,
    TRAJECTORY_V1,
    TRIGGERS_V1,
    AgentCapabilities,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    AttributionProbeResponse,
    BundleSkillRequest,
    ContextObjectTrajectoryRequest,
    ContextTrajectoryRequest,
    DataPart,
    DefaultExtensionHandlers,
    ExtensionConfiguration,
    ExtensionDefinition,
    FieldSchema,
    ImplementationOwner,
    InlineSkillRequest,
    McpAddRequest,
    NamespaceChangelogEnableRequest,
    NativeTrajectory,
    ObjectChangelogApplyRequest,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    OperationDefinition,
    PeerAgentsSetRequest,
    RequestDefinition,
    RequestVariant,
    StagingStore,
    TaskObjectTrajectoryRequest,
    TaskOutcome,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TaskTrajectoryRequest,
    TextPart,
    TriggerDecideRequest,
    TriggerRegisterRequest,
    Usage,
    WriteOnly,
    a2a_agent,
    build_registry,
    create_app,
    custom_extension,
    enable,
    extension,
    serve,
)
from agentenv_protocol.a2a_agent._triggers import (
    MAX_SOLVER_MESSAGE_LENGTH,
    TriggerEngine,
    TriggerError,
    TriggerTimeoutError,
)
from agentenv_protocol.a2a_agent.framework import (
    _BoundedContextLocks,
    _BoundedContextSessions,
    _BoundedTaskTrajectories,
    _SdkServices,
    _StandardExecutor,
)
from pydantic import BaseModel, Field, ValidationError, field_serializer
from starlette.testclient import TestClient


def _skill_bundle_payload(name: str = "review") -> dict[str, Any]:
    return {
        "name": name,
        "description": "Review code",
        "skill_bundle": {
            "max_total_bytes": 1024,
            "files": [
                {
                    "path": "SKILL.md",
                    "object": {
                        "media_type": "text/markdown",
                        "max_bytes": 1024,
                        "size_bytes": 8,
                        "read": {
                            "kind": "http-get",
                            "url": "https://objects.example.test/read?signature=secret",
                            "expires_at": "2099-01-01T00:00:00Z",
                        },
                    },
                }
            ],
        },
    }


def test_agent_capabilities_is_the_upstream_a2a_type() -> None:
    assert AgentCapabilities is UpstreamAgentCapabilities


def test_sync_extension_handler_is_rejected_at_startup() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="sync-extension", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @custom_extension(
            uri="urn:example:sync/v1",
            operation="read",
            method="GET",
            path="/ext/sync/v1",
        )
        def read(self):
            return {"status": "ok"}

    with pytest.raises(ValueError, match="must be an async function"):
        Agent().create_app()


def test_sync_run_is_rejected() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="sync-agent", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text(f"handled {request.context_id}")

    with pytest.raises(TypeError, match="async function or async generator"):
        Agent().create_app()


def test_invalid_run_signature_is_rejected_when_app_is_created() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="invalid-run", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self) -> TaskResult:
            return TaskResult.text("ok")

    with pytest.raises(TypeError, match="exactly one request argument"):
        Agent().create_app()


def test_run_request_must_be_annotated_as_task_request() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="invalid-run", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: str) -> TaskResult:
            return TaskResult.text(request)

    with pytest.raises(TypeError, match="annotated as TaskRequest"):
        Agent().create_app()


def test_run_request_must_accept_the_configured_config_type() -> None:
    class RuntimeConfig(AgentConfig):
        model: str | None = None

    class OtherConfig(AgentConfig):
        harness: str | None = None

    @a2a_agent(
        identity=AgentIdentity(name="invalid-run", description="test", version="1"),
        config=RuntimeConfig,
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[OtherConfig]) -> TaskResult:
            return TaskResult.text("ok")

    Agent.run.__annotations__["request"] = TaskRequest[OtherConfig]
    with pytest.raises(TypeError, match="configured RuntimeConfig"):
        Agent().create_app()


def test_run_request_resolves_function_local_parent_config_annotation() -> None:
    class ParentConfig(AgentConfig):
        model: str | None = None

    class ChildConfig(ParentConfig):
        temperature: float = 0

    @a2a_agent(
        identity=AgentIdentity(name="parent-config", description="test", version="1"),
        config=ChildConfig,
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[ParentConfig]) -> TaskResult:
            return TaskResult.text(request.config.model or "ok")

    with TestClient(Agent().create_app()) as client:
        response = client.post("/a2a", json=_message_request())

    assert response.json()["result"]["status"]["state"] == "completed"


def test_context_session_cache_evicts_least_recently_used_context() -> None:
    sessions = _BoundedContextSessions(max_entries=2)
    sessions.set("first", "session-1")
    sessions.set("second", "session-2")
    assert sessions.get("first") == "session-1"

    sessions.set("third", "session-3")

    assert sessions.get("first") == "session-1"
    assert sessions.get("second") is None
    assert sessions.get("third") == "session-3"


def test_trajectory_cache_evicts_least_recently_used_task() -> None:
    trajectories = _BoundedTaskTrajectories(max_entries=2)
    trajectories["first"] = {"event": 1}
    trajectories["second"] = {"event": 2}
    assert trajectories.get("first") == {"event": 1}

    trajectories["third"] = {"event": 3}

    assert trajectories.get("first") == {"event": 1}
    assert trajectories.get("second") is None
    assert trajectories.get("third") == {"event": 3}
    assert len(trajectories) == 2


@pytest.mark.asyncio
async def test_context_lock_cache_evicts_idle_contexts() -> None:
    locks = _BoundedContextLocks(max_entries=2)

    async with locks.acquire("first"):
        pass
    async with locks.acquire("second"):
        pass
    async with locks.acquire("third"):
        pass

    assert list(locks._entries) == ["second", "third"]


def test_trigger_engine_bounds_persistent_trigger_context_and_log_state() -> None:
    engine = TriggerEngine(max_triggers=2, max_contexts=2, max_log_entries=3)

    def trigger(trigger_id: str) -> dict[str, Any]:
        return {
            "id": trigger_id,
            "when": {"type": "step", "turn": 1, "cmp": "gte"},
            "actions": [{"type": "say", "text": trigger_id}],
        }

    engine.register({"triggers": [trigger("first"), trigger("second")]})
    with pytest.raises(TriggerError, match="trigger limit"):
        engine.register({"triggers": [trigger("third")]})

    assert engine.decide(turn=1, context_id="one")["fired"] == ["first", "second"]
    assert engine.decide(turn=1, context_id="two")["fired"] == ["first", "second"]

    # A full cache does not evict a context and silently replay once-only actions.
    with pytest.raises(TriggerError, match="context limit"):
        engine.decide(turn=1, context_id="three")
    assert engine.decide(turn=2, context_id="one")["fired"] == []
    firing_log = engine.state()["firing_log"]
    assert len(firing_log) == 3
    assert [entry["seq"] for entry in firing_log] == [3, 4, 5]

    with pytest.raises(TriggerError, match="context_id"):
        engine.decide(turn=1, context_id="x" * 257)


def test_trigger_engine_bounds_retained_trigger_content() -> None:
    engine = TriggerEngine()

    def register(when: dict[str, Any], actions: list[dict[str, Any]]) -> None:
        engine.register(
            {"triggers": [{"id": "bounded", "when": when, "actions": actions}]}
        )

    with pytest.raises(TriggerError, match="say.text"):
        register(
            {"type": "step", "turn": 1},
            [{"type": "say", "text": "x" * 8_193}],
        )
    with pytest.raises(TriggerError, match="env_trigger.env_id"):
        register(
            {
                "type": "env_trigger",
                "env_id": "x" * 257,
                "trigger_id": "source",
            },
            [{"type": "end"}],
        )
    with pytest.raises(TriggerError, match="env_trigger.trigger_id"):
        register(
            {
                "type": "env_trigger",
                "env_id": "environment",
                "trigger_id": "x" * 129,
            },
            [{"type": "end"}],
        )
    with pytest.raises(TriggerError, match="env_trigger.status"):
        register(
            {
                "type": "env_trigger",
                "env_id": "environment",
                "trigger_id": "source",
                "status": "x" * 129,
            },
            [{"type": "end"}],
        )
    with pytest.raises(TriggerError, match="specification"):
        register(
            {
                "type": "conversational",
                "where": {"message": {"equals": "x" * 20_000}},
            },
            [{"type": "end"}],
        )


def test_trigger_engine_bounds_solver_messages() -> None:
    engine = TriggerEngine()

    with pytest.raises(TriggerError, match="solver_message.*at most"):
        engine.decide(
            turn=1,
            solver_message="x" * (MAX_SOLVER_MESSAGE_LENGTH + 1),
        )


def test_trigger_engine_bounds_regex_time_without_partial_state() -> None:
    engine = TriggerEngine(regex_budget_seconds=0.01)
    engine.register(
        {
            "triggers": [
                {
                    "id": "first",
                    "when": {
                        "type": "conversational",
                        "where": {"message": {"regex": "^a"}},
                    },
                    "actions": [{"type": "say", "text": "first"}],
                },
                {
                    "id": "pathological",
                    "when": {
                        "type": "conversational",
                        "where": {"message": {"regex": "^(a|aa)+$"}},
                    },
                    "actions": [{"type": "say", "text": "second"}],
                },
            ]
        }
    )

    with pytest.raises(TriggerTimeoutError, match="decision budget"):
        engine.decide(
            turn=1,
            solver_message="a" * 40 + "b",
            context_id="context",
        )

    assert "context" not in engine._contexts
    assert [entry["kind"] for entry in engine.state()["firing_log"]] == ["registered"]
    assert engine.decide(
        turn=1,
        solver_message="aardvark",
        context_id="context",
    )["fired"] == ["first"]


@pytest.mark.parametrize(
    "budget", [0, -1, float("inf"), float("-inf"), float("nan"), True]
)
def test_trigger_engine_rejects_invalid_regex_budgets(budget: Any) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        TriggerEngine(regex_budget_seconds=budget)


def _extension(card: dict[str, Any], uri: str) -> dict[str, Any]:
    return next(
        item for item in card["capabilities"]["extensions"] if item["uri"] == uri
    )


def _operation(
    client: TestClient,
    card: dict[str, Any],
    uri: str,
    name: str,
    payload: dict[str, Any] | None = None,
):
    extension = _extension(card, uri)
    specification = extension["params"]["methods"][name]
    path = specification.get("endpoint", extension["params"].get("endpoint"))
    return client.request(specification["method"], path, json=payload)


@pytest.mark.parametrize(
    ("operation", "payload", "message"),
    [
        ("register", {"triggers": []}, "'triggers' must be a non-empty list"),
        (
            "register",
            {
                "triggers": [
                    {
                        "id": "bad-type",
                        "when": {"type": "unknown"},
                        "actions": [{"type": "end"}],
                    }
                ]
            },
            "unknown when.type 'unknown'",
        ),
        (
            "register",
            {
                "triggers": [
                    {
                        "id": "bad-regex",
                        "when": {
                            "type": "conversational",
                            "where": {"message": {"regex": "("}},
                        },
                        "actions": [{"type": "end"}],
                    }
                ]
            },
            "invalid regex",
        ),
        (
            "register",
            {
                "triggers": [
                    {
                        "id": "oversized-action",
                        "when": {"type": "step", "turn": 1},
                        "actions": [{"type": "say", "text": "x" * 8_193}],
                    }
                ]
            },
            "say.text must be at most 8192 characters",
        ),
        ("decide", {"turn": 0}, "'turn' must be a positive integer"),
        (
            "decide",
            {
                "turn": 1,
                "solver_message": "x" * (MAX_SOLVER_MESSAGE_LENGTH + 1),
            },
            "at most 200000 characters",
        ),
    ],
)
def test_trigger_validation_errors_are_http_bad_requests(
    operation: str,
    payload: dict[str, Any],
    message: str,
) -> None:
    @a2a_agent(
        identity=AgentIdentity(name="trigger-agent", description="test", version="1"),
        extensions=(TRIGGERS_V1,),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        response = _operation(client, card, TRIGGERS_V1.uri, operation, payload)

    assert response.status_code == 400
    assert message in response.text


def test_trigger_regex_timeout_is_an_http_gateway_timeout() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="trigger-agent", description="test", version="1"),
        extensions=(TRIGGERS_V1,),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

    agent = Agent()
    app = agent.create_app()
    agent._agentenv_a2a_application.services.trigger_engine = TriggerEngine(
        regex_budget_seconds=0.01
    )
    with TestClient(app) as client:
        card = client.get("/.well-known/agent.json").json()
        registration = _operation(
            client,
            card,
            TRIGGERS_V1.uri,
            "register",
            {
                "triggers": [
                    {
                        "id": "pathological",
                        "when": {
                            "type": "conversational",
                            "where": {"message": {"regex": "^(a|aa)+$"}},
                        },
                        "actions": [{"type": "end"}],
                    }
                ]
            },
        )
        response = _operation(
            client,
            card,
            TRIGGERS_V1.uri,
            "decide",
            {"turn": 1, "solver_message": "a" * 40 + "b"},
        )

    assert registration.status_code == 200
    assert response.status_code == 504
    assert "decision budget" in response.text


def _message_request(method: str = "message/send") -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "request-1",
        "method": method,
        "params": {
            "message": {
                "kind": "message",
                "messageId": "message-1",
                "role": "user",
                "contextId": "context-1",
                "parts": [{"kind": "text", "text": "hello"}],
            },
            "configuration": {"blocking": True},
        },
    }


def _sse_results(response) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))["result"]
        for line in response.iter_lines()
        if line.startswith("data: ")
    ]


def _request_context(task_id: str, context_id: str) -> RequestContext:
    return RequestContext(
        request=MessageSendParams(
            message=Message(
                message_id=f"message-{task_id}",
                role=Role.user,
                task_id=task_id,
                context_id=context_id,
                parts=[Part(root=A2ATextPart(text="hello"))],
            )
        )
    )


def test_agent_base_and_module_application_helpers(monkeypatch) -> None:
    @a2a_agent(
        identity=AgentIdentity(name="helper-agent", description="test", version="1"),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[AgentConfig]) -> TaskResult:
            return TaskResult.text("ok")

    agent = Agent()
    assert "create_app" not in Agent.__dict__
    app = agent.create_app()
    assert app.state.agentenv_a2a.agent is agent
    assert create_app(agent) is app

    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_run(app, **kwargs):
        calls.append((app, kwargs))

    monkeypatch.setattr("uvicorn.run", fake_run)
    serve(agent, host="127.0.0.1", port=9000, log_level="warning")
    assert calls[0][0] is app
    assert calls[0][1] == {
        "host": "127.0.0.1",
        "port": 9000,
        "log_level": "warning",
    }


def test_agent_card_is_served_at_current_and_legacy_well_known_paths() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="card-paths", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        current = client.get("/.well-known/agent-card.json")
        legacy = client.get("/.well-known/agent.json")

    assert current.status_code == 200
    assert legacy.status_code == 200
    assert current.json() == legacy.json()
    assert current.json()["skills"] == []


def test_absolute_agent_url_is_preserved_while_its_path_is_mounted() -> None:
    @a2a_agent(
        identity=AgentIdentity(
            name="absolute-url",
            description="test",
            version="1",
            url="https://agent.example.test/custom/a2a",
        )
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent-card.json").json()
        response = client.post("/custom/a2a", json=_message_request())

    assert card["url"] == "https://agent.example.test/custom/a2a"
    assert response.json()["result"]["status"]["state"] == "completed"


def test_invalid_agent_url_has_a_framework_error() -> None:
    @a2a_agent(
        identity=AgentIdentity(
            name="invalid-url", description="test", version="1", url="a2a"
        )
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

    with pytest.raises(ValueError, match="path must start"):
        Agent().create_app()


def test_a2a_agent_decorator_requires_typed_base() -> None:
    with pytest.raises(TypeError, match="requires an AgentEnvAgent subclass"):

        @a2a_agent(
            identity=AgentIdentity(name="plain-agent", description="test", version="1"),
        )
        class PlainAgent:
            pass


def test_non_streaming_capabilities_are_derived_from_run() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="streaming-agent", description="test", version="1"),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

    card = Agent().create_app().state.agentenv_a2a.card
    assert card.capabilities == AgentCapabilities(
        streaming=False,
        push_notifications=False,
        state_transition_history=False,
        extensions=[AgentExtension.model_validate(StagingStore().card_extension())],
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'capabilities'"):
        a2a_agent(
            identity=AgentIdentity(name="legacy", description="test", version="1"),
            capabilities=AgentCapabilities(streaming=False),  # type: ignore[call-arg]
        )


def test_streaming_progress_and_status_events_are_forwarded_in_order() -> None:
    cleaned_up = False

    @a2a_agent(
        identity=AgentIdentity(name="streaming-agent", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest):
            nonlocal cleaned_up
            try:
                yield TaskProgress.text("Searching...", metadata={"step": 1})
                yield TaskStatusUpdateEvent(
                    task_id=request.task_id,
                    context_id=request.context_id,
                    final=False,
                    status=TaskStatus(state=TaskState.working),
                )
                yield TaskResult.text("done", session_ref="stream-session")
                raise AssertionError(
                    "items after the terminal result must not be consumed"
                )
            finally:
                cleaned_up = True

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        assert card["capabilities"] == {
            "extensions": [StagingStore().card_extension()],
            "pushNotifications": False,
            "stateTransitionHistory": False,
            "streaming": True,
        }

        with client.stream(
            "POST", "/a2a", json=_message_request("message/stream")
        ) as response:
            assert response.status_code == 200
            events = _sse_results(response)

        assert [event["kind"] for event in events] == [
            "task",
            "status-update",
            "status-update",
            "status-update",
            "status-update",
        ]
        progress = events[2]
        assert progress["final"] is False
        assert progress["metadata"] == {"step": 1}
        assert progress["status"]["message"]["parts"] == [
            {"kind": "text", "text": "Searching..."}
        ]
        assert events[3]["status"]["state"] == "working"
        assert events[-1]["final"] is True
        assert events[-1]["status"]["state"] == "completed"
        assert events[-1]["status"]["message"]["parts"][0]["text"] == "done"

    assert cleaned_up is True


def test_coroutine_decorator_preserves_streaming_run_contract() -> None:
    wrapper_calls = 0

    def trace(handler):
        @wraps(handler)
        async def wrapped(*args, **kwargs):
            nonlocal wrapper_calls
            wrapper_calls += 1
            return handler(*args, **kwargs)

        return wrapped

    @a2a_agent(
        identity=AgentIdentity(name="streaming-agent", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        @trace
        async def run(self, request: TaskRequest):
            yield TaskProgress.text("working")
            yield TaskResult.text("done")

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        assert card["capabilities"]["streaming"] is True
        with client.stream(
            "POST", "/a2a", json=_message_request("message/stream")
        ) as response:
            events = _sse_results(response)

    assert wrapper_calls == 1
    assert events[-1]["status"]["state"] == "completed"
    assert events[-1]["status"]["message"]["parts"][0]["text"] == "done"


def test_streaming_run_supports_send_polling_and_session_reuse() -> None:
    sessions: list[str | None] = []

    @a2a_agent(
        identity=AgentIdentity(name="streaming-agent", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest):
            sessions.append(request.session_ref)
            yield TaskProgress.text("working")
            yield TaskResult.text("done", session_ref="native-session")

    with TestClient(Agent().create_app()) as client:
        first = client.post("/a2a", json=_message_request()).json()["result"]
        second_request = _message_request()
        second_request["id"] = "request-2"
        second_request["params"]["message"]["messageId"] = "message-2"
        second = client.post("/a2a", json=second_request).json()["result"]

        non_blocking_request = _message_request()
        non_blocking_request["id"] = "request-3"
        non_blocking_request["params"]["message"]["messageId"] = "message-3"
        non_blocking_request["params"]["configuration"]["blocking"] = False
        submitted = client.post("/a2a", json=non_blocking_request).json()["result"]
        deadline = time.monotonic() + 1
        while True:
            polled = client.post(
                "/a2a",
                json={
                    "jsonrpc": "2.0",
                    "id": "poll-3",
                    "method": "tasks/get",
                    "params": {"id": submitted["id"]},
                },
            ).json()["result"]
            if polled["status"]["state"] in {"completed", "failed"}:
                break
            if time.monotonic() >= deadline:
                pytest.fail("non-blocking task did not reach a terminal state")
            time.sleep(0.01)

    assert first["status"]["state"] == "completed"
    assert second["status"]["state"] == "completed"
    assert submitted["status"]["state"] == "submitted"
    assert polled["status"]["state"] == "completed"
    assert sessions == [None, "native-session", "native-session"]


def test_failure_after_task_creation_emits_terminal_failed_status() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="broken-config", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            pytest.fail("run must not be called when task config construction fails")

    app = Agent().create_app()
    app.state.agentenv_a2a.services.task_config = MagicMock(
        side_effect=ValueError("broken task config")
    )
    with TestClient(app) as client:
        task = client.post("/a2a", json=_message_request()).json()["result"]

    assert task["status"]["state"] == "failed"
    error = task["status"]["message"]["parts"][-1]["data"]
    assert error["error_type"] == "infra_error"
    assert error["error_code"] == "framework.unhandled_exception"
    assert "broken task config" not in error["error_message"]


def test_invalid_message_before_task_creation_returns_invalid_params() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="invalid-input", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

    request = _message_request()
    request["params"]["message"]["parts"] = [{"kind": "text", "text": ""}]
    with TestClient(Agent().create_app()) as client:
        response = client.post("/a2a", json=request)

    assert response.json()["error"]["code"] == -32602
    assert "TextPart content cannot be empty" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_streaming_runs_serialize_by_context_before_reading_session() -> None:
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    sessions: list[str | None] = []

    async def run(request: TaskRequest):
        index = len(sessions)
        sessions.append(request.session_ref)
        if index == 0:
            first_entered.set()
            await release_first.wait()
        yield TaskResult.text("done", session_ref=f"session-{index + 1}")

    executor = _StandardExecutor(
        run,
        _SdkServices((), None),
        workspace=None,
        streaming=True,
    )
    first = asyncio.create_task(
        executor.execute(_request_context("task-1", "same"), EventQueue())
    )
    await first_entered.wait()
    second = asyncio.create_task(
        executor.execute(_request_context("task-2", "same"), EventQueue())
    )
    await asyncio.sleep(0)
    assert sessions == [None]

    release_first.set()
    await asyncio.gather(first, second)
    assert sessions == [None, "session-1"]


@pytest.mark.asyncio
async def test_streaming_runs_allow_different_contexts_to_execute_concurrently() -> (
    None
):
    both_entered = asyncio.Event()
    release = asyncio.Event()
    contexts: list[str] = []

    async def run(request: TaskRequest):
        contexts.append(request.context_id)
        if len(contexts) == 2:
            both_entered.set()
        await release.wait()
        yield TaskResult.text("done")

    executor = _StandardExecutor(
        run,
        _SdkServices((), None),
        workspace=None,
        streaming=True,
    )
    first = asyncio.create_task(
        executor.execute(_request_context("task-1", "first"), EventQueue())
    )
    second = asyncio.create_task(
        executor.execute(_request_context("task-2", "second"), EventQueue())
    )
    try:
        await asyncio.wait_for(both_entered.wait(), timeout=1)
    finally:
        release.set()
        await asyncio.gather(first, second)

    assert set(contexts) == {"first", "second"}


@pytest.mark.parametrize("failure_mode", ["invalid_item", "missing_result"])
def test_streaming_contract_errors_become_safe_failed_tasks(failure_mode: str) -> None:
    @a2a_agent(
        identity=AgentIdentity(name="bad-stream", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest):
            yield TaskProgress.text("partial")
            if failure_mode == "invalid_item":
                yield "private invalid stream value"

    with TestClient(Agent().create_app()) as client:
        response = client.post("/a2a", json=_message_request())

    task = response.json()["result"]
    assert task["status"]["state"] == "failed"
    error = task["status"]["message"]["parts"][-1]["data"]
    assert error["error_type"] == "infra_error"
    assert error["error_code"] == "framework.unhandled_exception"
    assert "private invalid stream value" not in response.text


@pytest.mark.parametrize("invalid", ["identity", "terminal"])
def test_streaming_rejects_direct_events_that_take_over_the_lifecycle(
    invalid: str,
) -> None:
    @a2a_agent(
        identity=AgentIdentity(name="bad-events", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest):
            yield TaskStatusUpdateEvent(
                task_id="another-task" if invalid == "identity" else request.task_id,
                context_id=request.context_id,
                final=invalid == "terminal",
                status=TaskStatus(
                    state=(
                        TaskState.completed
                        if invalid == "terminal"
                        else TaskState.working
                    )
                ),
            )
            yield TaskResult.text("must not complete")

    with TestClient(Agent().create_app()) as client:
        task = client.post("/a2a", json=_message_request()).json()["result"]

    assert task["status"]["state"] == "failed"
    error = task["status"]["message"]["parts"][-1]["data"]
    assert error["error_type"] == "infra_error"
    assert error["error_code"] == "framework.unhandled_exception"


def test_task_request_is_detached_and_json_serializable() -> None:
    class Config(AgentConfig):
        allowed_tools: list[str] = Field(default_factory=lambda: ["read"])
        model_params: dict[str, Any] = Field(
            default_factory=lambda: {"temperature": 0.2}
        )

    config = Config()
    mcp_servers = {
        "mcp_one": {"url": "https://mcp.test", "headers": {"X-Test": "value"}}
    }
    skills = ({"name": "skill", "tags": ["test"]},)
    metadata = {"trace": {"ids": ["one"]}}
    request = TaskRequest(
        task_id="task",
        context_id="context",
        parts=(TextPart(text="hello"),),
        config=config,
        mcp_servers=mcp_servers,
        skills=skills,
        metadata=metadata,
    )

    with pytest.raises(FrozenInstanceError):
        request.task_id = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        request.config.role = "other"  # type: ignore[misc]

    assert isinstance(request.config.allowed_tools, list)
    assert isinstance(request.config.model_params, dict)
    assert request.config.model_dump(mode="json") == {
        "name": None,
        "description": None,
        "role": None,
        "timeout_seconds": 600,
        "allowed_tools": ["read"],
        "model_params": {"temperature": 0.2},
    }
    assert request.config.model_copy(deep=True) == request.config
    assert json.loads(json.dumps(request.mcp_servers)) == mcp_servers
    assert json.loads(json.dumps(request.skills)) == list(skills)
    assert json.loads(json.dumps(request.metadata)) == metadata

    request.mcp_servers["mcp_one"]["headers"]["X-Test"] = "changed"
    request.skills[0]["tags"].append("local")
    request.metadata["trace"]["ids"].append("two")
    request.config.allowed_tools.append("write")
    request.config.model_params["temperature"] = 0.9
    assert mcp_servers["mcp_one"]["headers"]["X-Test"] == "value"
    assert skills[0]["tags"] == ["test"]
    assert metadata["trace"]["ids"] == ["one"]
    assert config.allowed_tools == ["read"]
    assert config.model_params == {"temperature": 0.2}


def test_data_part_preserves_json_native_container_types() -> None:
    source = {"items": [{"value": 1}]}
    part = DataPart(data=source)

    assert json.loads(json.dumps(part.data)) == source
    part.data["items"][0]["value"] = 2
    assert source == {"items": [{"value": 1}]}


def test_task_result_builder_validates_and_factories_share_it() -> None:
    with pytest.raises(ValueError, match="outcome is required"):
        TaskResult.builder().add_text("missing outcome").build()

    result = TaskResult.text(
        "done",
        session_ref="session",
        usage=Usage(
            tool_call_count=2,
            input_tokens=10,
            provider_details={"models": {"small": {"output_tokens": 3}}},
        ),
        native_trajectory=NativeTrajectory(format="events/v1", payload=[{"ok": True}]),
    )
    assert result.outcome is TaskOutcome.SUCCEEDED
    assert result.parts == (TextPart(text="done"),)
    assert result.session_ref == "session"
    assert result.usage.tool_call_count == 2
    assert result.usage.input_tokens == 10
    assert result.usage.to_dict() == {
        "tool_call_count": 2,
        "input_tokens": 10,
        "provider_details": {"models": {"small": {"output_tokens": 3}}},
    }
    assert result.native_trajectory is not None

    structured = (
        TaskResult.builder()
        .succeeded()
        .add_text("done")
        .add_structured_output({"answer": 42})
        .build()
    )
    assert structured.parts == (
        TextPart(text="done"),
        DataPart(data={"structured_output": {"answer": 42}}),
    )

    failure = TaskResult.failure("cli_error", "failed")
    assert failure.outcome is TaskOutcome.FAILED
    assert failure.error is not None
    assert failure.error.code == "cli_error"
    assert failure.error.message == "failed"
    assert failure.error.error_type == "agent_error"
    infra_failure = TaskResult.failure(
        "provider_timeout", "try again", error_type="infra_error"
    )
    assert infra_failure.error is not None
    assert infra_failure.error.error_type == "infra_error"
    with pytest.raises(ValueError, match="error_type"):
        TaskResult.failure("cli_error", "failed", error_type="custom")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unexpected keyword argument 'details'"):
        TaskResult.failure("cli_error", "failed", details={})  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword argument 'details'"):
        TaskResult.builder().failed(  # type: ignore[call-arg]
            "cli_error", "failed", details={}
        )
    with pytest.raises(ValueError, match="requires at least one part"):
        TaskResult.success()


def test_expected_run_failure_is_a_structured_failed_task() -> None:
    @a2a_agent(
        identity=AgentIdentity(
            name="expected-failure", description="test", version="1"
        ),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.failure(
                code="provider_rate_limited",
                message="The model provider is temporarily unavailable",
                parts=(TextPart(text="Retry later"),),
            )

    payload = {
        "jsonrpc": "2.0",
        "id": "expected-failure-request",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "messageId": "expected-failure-message",
                "role": "user",
                "parts": [{"kind": "text", "text": "hello"}],
            },
            "configuration": {"blocking": True},
        },
    }

    with TestClient(Agent().create_app()) as client:
        response = client.post("/a2a", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "error" not in body
    task = body["result"]
    assert task["status"]["state"] == "failed"
    assert task["status"]["message"]["parts"] == [
        {"kind": "text", "text": "Retry later"},
        {
            "kind": "data",
            "data": {
                "error_type": "agent_error",
                "error_code": "provider_rate_limited",
                "error_message": "The model provider is temporarily unavailable",
            },
        },
    ]


@pytest.mark.parametrize("failure_mode", ["exception", "invalid_result", "mapping"])
def test_unhandled_run_failures_are_opaque_and_correlated(
    failure_mode: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    internal_marker = f"private-{failure_mode}-detail"

    @a2a_agent(
        identity=AgentIdentity(
            name="unhandled-failure", description="test", version="1"
        ),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            if failure_mode == "exception":
                raise RuntimeError(internal_marker)
            if failure_mode == "invalid_result":
                return internal_marker  # type: ignore[return-value]
            return TaskResult.success(parts=(object(),))  # type: ignore[arg-type]

    payload = {
        "jsonrpc": "2.0",
        "id": f"{failure_mode}-request",
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "messageId": f"{failure_mode}-message",
                "role": "user",
                "parts": [{"kind": "text", "text": "hello"}],
            },
            "configuration": {"blocking": True},
        },
    }

    with TestClient(Agent().create_app()) as client:
        response = client.post("/a2a", json=payload)

    assert response.status_code == 200
    task = response.json()["result"]
    assert task["status"]["state"] == "failed"
    error_data = task["status"]["message"]["parts"][-1]["data"]
    assert error_data["error_type"] == "infra_error"
    assert error_data["error_code"] == "framework.unhandled_exception"
    prefix = "An unexpected framework error occurred. Correlation ID: "
    assert error_data["error_message"].startswith(prefix)
    correlation_id = error_data["error_message"].removeprefix(prefix)
    assert len(correlation_id) == 32
    int(correlation_id, 16)
    assert correlation_id in caplog.text
    assert internal_marker not in response.text


def test_sdk_skill_contract_is_portable_only() -> None:
    skill_add = SKILL_CONFIG_V1.operation("add")
    skill_request = skill_add.request
    assert skill_request is not None
    assert skill_request.to_card() == {
        "required": ["name", "description"],
        "oneOf": [
            {"required": ["skill_md"]},
            {"required": ["skill_bundle"]},
        ],
    }
    assert skill_add.response is not None
    assert skill_add.response.to_card() == {"required": ["name"]}
    assert skill_request.select_variant(
        {"name": "x", "description": "x", "skill_md": "# x"}
    ) == "inline"
    assert (
        skill_request.select_variant(_skill_bundle_payload(name="x")) == "bundle"
    )
    with pytest.raises(ValueError, match="request must match exactly one"):
        skill_request.select_variant(
            {
                "name": "x",
                "description": "x",
                "skill_s3_url": "s3://bucket/skill",
            }
        )


@pytest.mark.parametrize("name", ["../app", "a/b", "a\\b", ".hidden", "", "x" * 129])
def test_skill_names_are_one_safe_path_segment(name: str) -> None:
    with pytest.raises(ValidationError, match="name"):
        InlineSkillRequest(name=name, description="x", skill_md="# x")
    with pytest.raises(ValidationError, match="name"):
        BundleSkillRequest.model_validate(_skill_bundle_payload(name=name))


def test_runtime_can_implement_the_portable_skill_operation() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="variant-agent", description="test", version="1"),
    )
    class VariantAgent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @extension(SKILL_CONFIG_V1.add.inline)
        async def add_inline(self, request: InlineSkillRequest):
            return {"name": request.name}

        @extension(SKILL_CONFIG_V1.add.bundle)
        async def add_bundle(self, request: BundleSkillRequest):
            return {"name": request.name}

    with TestClient(VariantAgent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        added = _operation(
            client,
            card,
            SKILL_CONFIG_V1.uri,
            "add",
            _skill_bundle_payload(name="bundle-skill"),
        )
        assert added.json() == {"name": "bundle-skill"}
        assert set(
            _operation(client, card, SKILL_CONFIG_V1.uri, "list").json()["skills"]
        ) == {"bundle-skill"}


def test_handler_receives_the_canonical_request_model() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="peer-agent", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        def __init__(self) -> None:
            self.peers = []

        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @extension(PEER_AGENTS_V1.set)
        async def set_peers(self, update: PeerAgentsSetRequest):
            self.peers = [peer.model_dump(exclude_unset=True) for peer in update.peers]
            return {"status": "updated"}

        @extension(PEER_AGENTS_V1.list)
        async def list_peers(self):
            return {"peers": self.peers}

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        peers = [{"name": "reviewer", "url": "https://peer.test/a2a"}]
        assert (
            _operation(
                client,
                card,
                PEER_AGENTS_V1.uri,
                "set",
                {"peers": peers, "undeclared": True},
            ).status_code
            == 400
        )
        assert (
            _operation(client, card, PEER_AGENTS_V1.uri, "set", {"peers": peers})
            .json()
            .get("status")
            == "updated"
        )
        assert _operation(client, card, PEER_AGENTS_V1.uri, "list").json() == {
            "peers": peers
        }


def test_attribution_probe_reports_what_the_agent_last_sent() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="attributing-agent", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        def __init__(self) -> None:
            self.sent: dict[str, str] = {}

        async def run(self, request: TaskRequest) -> TaskResult:
            self.sent = {"task_id": "task-1", "team": "evals"}
            return TaskResult.text("ok")

        @extension(ATTRIBUTION_PROBE_V1.probe)
        async def probe(self):
            return AttributionProbeResponse(
                last_seen_attribution=self.sent, last_seen_at_utc="2026-01-01T00:00:00Z"
            )

    agent = Agent()
    with TestClient(agent.create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        assert _extension(card, ATTRIBUTION_PROBE_V1.uri)["params"]["endpoint"] == (
            "/ext/attribution-probe"
        )
        assert _operation(client, card, ATTRIBUTION_PROBE_V1.uri, "probe", {}).json() == {
            "last_seen_attribution": {},
            "last_seen_at_utc": "2026-01-01T00:00:00Z",
        }
        response = client.post("/a2a", json=_message_request())
        assert response.json()["result"]["status"]["state"] == "completed"
        assert _operation(client, card, ATTRIBUTION_PROBE_V1.uri, "probe", {}).json() == {
            "last_seen_attribution": {"task_id": "task-1", "team": "evals"},
            "last_seen_at_utc": "2026-01-01T00:00:00Z",
        }
        assert (
            _operation(
                client, card, ATTRIBUTION_PROBE_V1.uri, "probe", {"unexpected": "x"}
            ).status_code
            == 400
        )


def test_attribution_probe_rejects_an_undeclared_response_field() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="attributing-agent", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @extension(ATTRIBUTION_PROBE_V1.probe)
        async def probe(self):
            return {"last_seen_attribution": {}, "headers": {"x-secret": "value"}}

    with TestClient(Agent().create_app(), raise_server_exceptions=False) as client:
        card = client.get("/.well-known/agent.json").json()
        response = _operation(client, card, ATTRIBUTION_PROBE_V1.uri, "probe", {})
        assert response.status_code == 500
        assert "x-secret" not in response.text


def test_attribution_probe_is_only_declared_by_a_handler() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="plain-agent", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        uris = {item["uri"] for item in card["capabilities"]["extensions"]}
        assert ATTRIBUTION_PROBE_V1.uri not in uris
        assert client.post("/ext/attribution-probe", json={}).status_code in (404, 405)


def test_handler_advertises_the_canonical_request_model() -> None:
    class Agent:
        @extension(SNAPSHOT_V1.save)
        async def save(self, request: ObjectSnapshotSaveRequest):
            return request

        @extension(SNAPSHOT_V1.load)
        async def load(self, request: ObjectSnapshotLoadRequest):
            return request

        @extension(SNAPSHOT_V1.changelog.enable)
        async def enable_changelog(self, request: NamespaceChangelogEnableRequest):
            return {"roots": request.roots or []}

        @extension(SNAPSHOT_V1.changelog.apply)
        async def apply_changelog(self, request: ObjectChangelogApplyRequest):
            return {"count": len(request.increments)}

    registry = build_registry(Agent())
    params = registry.card_extensions()[0]["params"]
    assert set(params) == {"endpoint", "methods"}
    methods = params["methods"]
    assert methods["save"]["request"] == {"required": ["context_id", "objects"]}
    assert methods["save"]["response"] == {"required": ["context_id", "objects"]}
    assert methods["load"]["request"] == {
        "required": ["objects"],
        "optional": ["target_context_id"],
    }
    assert methods["load"]["response"] == {"required": ["context_id"]}
    assert methods["enable-changelog"]["request"] == {
        "required": ["write_namespace"],
        "optional": ["roots"],
    }
    assert methods["enable-changelog"]["response"] == {"required": ["roots"]}
    assert methods["apply-changelog"]["request"] == {
        "required": ["increments"],
        "optional": [
            "resume_conversation",
            "target_context_id",
        ],
    }
    assert methods["apply-changelog"]["response"] == {
        "required": ["count"],
        "optional": ["context_id"],
    }


def test_request_variants_are_selected_by_model_validation() -> None:
    class InlineRequest(BaseModel):
        model_config = {"extra": "forbid"}

        name: str
        content: str | None = None

    class StoredRequest(BaseModel):
        model_config = {"extra": "forbid"}

        name: str
        uri: str | None = None

    request = RequestDefinition(
        variants=(
            RequestVariant("inline", InlineRequest),
            RequestVariant("stored", StoredRequest),
        )
    )

    assert request.select_variant({"name": "review", "content": "body"}) == "inline"
    assert request.select_variant({"name": "review", "uri": "s3://bucket/key"}) == (
        "stored"
    )
    with pytest.raises(ValueError, match="exactly one"):
        request.select_variant({"name": "review"})


def test_operation_response_models_must_be_closed() -> None:
    class OpenResponse(BaseModel):
        value: str

    with pytest.raises(TypeError, match="response model.*extra='forbid'"):
        OperationDefinition(
            name="send",
            method="POST",
            path="/ext/send",
            implementation=ImplementationOwner.RUNTIME,
            response=OpenResponse,
        )


def test_handler_must_use_the_canonical_request_model() -> None:
    class MissingPeers(BaseModel):
        note: str | None = None

    class MissingRequiredField:
        @extension(PEER_AGENTS_V1.set)
        async def set_peers(self, request: MissingPeers):
            return {"status": "updated"}

        @extension(PEER_AGENTS_V1.list)
        async def list_peers(self):
            return {"peers": []}

    MissingRequiredField.set_peers.__annotations__["request"] = MissingPeers
    with pytest.raises(ValueError, match="must annotate.*PeerAgentsSetRequest"):
        build_registry(MissingRequiredField())

    class UntypedHandler:
        @extension(PEER_AGENTS_V1.set)
        async def set_peers(self, payload):
            return {"status": "updated"}

        @extension(PEER_AGENTS_V1.list)
        async def list_peers(self):
            return {"peers": []}

    with pytest.raises(ValueError, match="must annotate.*PeerAgentsSetRequest"):
        build_registry(UntypedHandler())


def test_enable_uses_versioned_definition_and_validates_configuration() -> None:
    activation = enable(
        AGENT_CONFIG_V1,
        description="Configure this test runtime.",
        fields=("model", "timeout_seconds", "harness"),
        defaults={"model": "default-model"},
        readback=True,
    )
    assert activation.definition is AGENT_CONFIG_V1
    assert activation.description == "Configure this test runtime."
    assert activation.options["defaults"] == {"model": "default-model"}
    assert activation.features == frozenset({"readback"})
    assert activation.wire_params["methods"]["set"]["request"]["supported"] == [
        "harness",
        "model",
        "timeout_seconds",
    ]
    with pytest.raises(TypeError, match="required keyword-only argument.*fields"):
        enable(AGENT_CONFIG_V1)
    with pytest.raises(TypeError, match="unsupported configuration"):
        enable(MCP_CONFIG_V1, unknown=True)
    with pytest.raises(ValueError, match="description must not be empty"):
        enable(MCP_CONFIG_V1, description="  ")
    with pytest.raises(ValueError, match="defaults contain unsupported fields"):
        enable(
            AGENT_CONFIG_V1,
            fields=("model",),
            defaults={"timeout_seconds": 600},
        )


def test_extension_definition_owns_its_configuration_schema() -> None:
    def validate_diagnostics(
        *, level: str, sample_limit: int = 10
    ) -> ExtensionConfiguration:
        if level not in {"summary", "full"}:
            raise ValueError("level must be 'summary' or 'full'")
        if sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        return ExtensionConfiguration(
            wire_params={"level": level},
            options={"sample_limit": sample_limit},
        )

    diagnostics = ExtensionDefinition(
        uri="urn:example:diagnostics/v1",
        description="Diagnostics.",
        endpoint=None,
        configuration_validator=validate_diagnostics,
    )

    assert tuple(inspect.signature(diagnostics.configuration_validator).parameters) == (
        "level",
        "sample_limit",
    )
    activation = enable(diagnostics, level="full", sample_limit=25)
    assert activation.wire_params == {"level": "full"}
    assert activation.options == {"sample_limit": 25}

    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
        enable(diagnostics, level="summary", unknown=True)
    with pytest.raises(ValueError, match="sample_limit must be positive"):
        enable(diagnostics, level="summary", sample_limit=0)


def test_agent_config_extension_is_derived_from_typed_model() -> None:
    class RuntimeConfig(AgentConfig):
        model: str | None = None
        model_params: WriteOnly[dict[str, Any] | None] = None
        runtime_specific_option: bool = False

    @a2a_agent(
        identity=AgentIdentity(
            name="configured-agent", description="test", version="1"
        ),
        config=RuntimeConfig,
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[RuntimeConfig]) -> TaskResult:
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        supported = _extension(card, AGENT_CONFIG_V1.uri)["params"]["methods"]["set"][
            "request"
        ]["supported"]
        assert supported == sorted(RuntimeConfig.model_fields)
        assert "timeout_seconds" in supported

        assert (
            _operation(
                client,
                card,
                AGENT_CONFIG_V1.uri,
                "set",
                {"timeout_seconds": "not-an-integer"},
            ).status_code
            == 400
        )

        assert (
            _operation(
                client,
                card,
                AGENT_CONFIG_V1.uri,
                "set",
                {
                    "role": "auditor",
                    "model_params": {"api_key": "sk-live-secret"},
                },
            ).status_code
            == 200
        )
        config_response = _operation(
            client,
            card,
            AGENT_CONFIG_V1.uri,
            "get",
        )
        assert config_response.json() == {
            "config": {"role": "auditor", "model_params": "***"}
        }

        assert client.post("/ext/agent-config", json={"name": None}).status_code == 400
        assert (
            client.post("/ext/agent-config", json={"description": ""}).status_code
            == 400
        )
        unchanged_card = client.get("/.well-known/agent.json").json()
        assert unchanged_card["name"] == "configured-agent"
        assert unchanged_card["description"] == "test"

    with pytest.raises(TypeError, match="automatically enables"):
        a2a_agent(
            identity=AgentIdentity(name="duplicate", description="test", version="1"),
            config=RuntimeConfig,
            extensions=(enable(AGENT_CONFIG_V1, fields=("model",), readback=True),),
        )

    with pytest.raises(TypeError, match="derived from config"):
        a2a_agent(
            identity=AgentIdentity(name="legacy", description="test", version="1"),
            extensions=(enable(AGENT_CONFIG_V1, fields=("model",), readback=True),),
        )

    class RequiredConfig(AgentConfig):
        required_value: str

    with pytest.raises(TypeError, match="fields must have defaults"):
        a2a_agent(
            identity=AgentIdentity(name="required", description="test", version="1"),
            config=RequiredConfig,
        )

    with pytest.raises(TypeError, match="config_description requires config"):
        a2a_agent(
            identity=AgentIdentity(name="missing", description="test", version="1"),
            config_description="Cannot be applied",
        )

    with pytest.raises(TypeError, match="config_readback=False requires config"):
        a2a_agent(
            identity=AgentIdentity(name="missing", description="test", version="1"),
            config_readback=False,
        )

    with pytest.raises(TypeError, match="config_readback must be a boolean"):
        a2a_agent(
            identity=AgentIdentity(name="invalid", description="test", version="1"),
            config=RuntimeConfig,
            config_readback="yes",  # type: ignore[arg-type]
        )


def test_agent_card_config_schema_omits_literal_defaults() -> None:
    secret = "runtime-derived-secret"

    class NestedConfig(BaseModel):
        token: str = secret

    class RuntimeConfig(AgentConfig):
        default: str = secret
        api_key: str = secret
        nested: NestedConfig = NestedConfig()

    @a2a_agent(
        identity=AgentIdentity(name="private-defaults", description="test", version="1"),
        config=RuntimeConfig,
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[RuntimeConfig]) -> TaskResult:
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent-card.json").json()

    contract = _extension(card, AGENT_CONFIG_V1.uri)["params"]["methods"]["set"]
    schema = contract["request"]["schema"]
    assert secret not in json.dumps(contract)
    assert "default" in schema["properties"]
    assert "default" not in schema["properties"]["default"]
    assert "default" not in schema["properties"]["api_key"]
    assert "default" not in schema["$defs"]["NestedConfig"]["properties"]["token"]


def test_validation_errors_do_not_echo_rejected_values() -> None:
    secret = "sk-live-SUPERSECRET"

    class RuntimeConfig(AgentConfig):
        litellm_api_key: int | None = None

    class SecretRequest(BaseModel):
        model_config = {"extra": "forbid"}

        token: int

    @a2a_agent(
        identity=AgentIdentity(name="safe-errors", description="test", version="1"),
        config=RuntimeConfig,
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[RuntimeConfig]) -> TaskResult:
            return TaskResult.text("ok")

        @custom_extension(
            uri="urn:example:safe-errors/v1",
            operation="validate",
            method="POST",
            path="/ext/safe-errors/v1",
            request=SecretRequest,
        )
        async def validate(self, request: SecretRequest) -> dict[str, bool]:
            return {"ok": True}

    Agent.validate.__annotations__["request"] = SecretRequest
    with TestClient(Agent().create_app()) as client:
        config_response = client.post(
            "/ext/agent-config", json={"litellm_api_key": secret}
        )
        extension_response = client.post(
            "/ext/safe-errors/v1", json={"token": secret}
        )

    assert config_response.status_code == 400
    assert extension_response.status_code == 400
    assert secret not in config_response.text
    assert secret not in extension_response.text
    assert "input" not in json.loads(config_response.text)[0]
    assert "input" not in json.loads(extension_response.text)[0]


def test_typed_config_with_nested_alias_round_trips_between_requests() -> None:
    seen = []

    class ModelParams(BaseModel):
        api_key: str = Field(validation_alias="apiKey")

    class RuntimeConfig(AgentConfig):
        model_params: ModelParams | None = None

    @a2a_agent(
        identity=AgentIdentity(name="aliased-config", description="test", version="1"),
        config=RuntimeConfig,
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[RuntimeConfig]) -> TaskResult:
            seen.append(request.config)
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        assert (
            client.post(
                "/ext/agent-config",
                json={"model_params": {"apiKey": "secret"}},
            ).status_code
            == 200
        )
        task = client.post("/a2a", json=_message_request()).json()["result"]

    assert task["status"]["state"] == "completed"
    assert seen[0].model_params.api_key == "secret"


def test_typed_config_redacts_only_declared_write_only_fields_from_readback() -> None:
    seen: list[str | None] = []

    class RuntimeConfig(AgentConfig):
        runtime_options: dict[str, Any] | None = None
        provider_token: WriteOnly[str | None] = Field(
            default=None, alias="providerToken"
        )

    @a2a_agent(
        identity=AgentIdentity(name="aliased-config", description="test", version="1"),
        config=RuntimeConfig,
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[RuntimeConfig]) -> TaskResult:
            seen.append(request.config.provider_token)
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        schema = _extension(card, AGENT_CONFIG_V1.uri)["params"]["methods"]["set"][
            "request"
        ]["schema"]
        assert schema["properties"]["providerToken"]["writeOnly"] is True
        assert (
            client.post(
                "/ext/agent-config",
                json={
                    "runtime_options": {"temperature": 0.2},
                    "providerToken": "provider-secret",
                },
            ).status_code
            == 200
        )
        assert client.get("/ext/agent-config").json() == {
            "config": {
                "runtime_options": {"temperature": 0.2},
                "providerToken": "***",
            }
        }
        task = client.post("/a2a", json=_message_request()).json()["result"]

    assert task["status"]["state"] == "completed"
    assert seen == ["provider-secret"]


def test_typed_config_uses_top_level_alias_as_its_wire_name() -> None:
    seen = []

    class RuntimeConfig(AgentConfig):
        api_key: str = Field(default="default", alias="apiKey")

    @a2a_agent(
        identity=AgentIdentity(name="aliased-config", description="test", version="1"),
        config=RuntimeConfig,
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[RuntimeConfig]) -> TaskResult:
            seen.append(request.config)
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        config = _extension(card, AGENT_CONFIG_V1.uri)
        request_contract = config["params"]["methods"]["set"]["request"]
        assert "apiKey" in request_contract["supported"]
        assert "api_key" not in request_contract["supported"]
        assert "apiKey" in request_contract["schema"]["properties"]

        assert (
            client.post("/ext/agent-config", json={"apiKey": "secret"}).status_code
            == 200
        )
        assert (
            client.post("/ext/agent-config", json={"api_key": "secret"}).status_code
            == 400
        )
        assert client.get("/ext/agent-config").json() == {
            "config": {"apiKey": "secret"}
        }
        task = client.post("/a2a", json=_message_request()).json()["result"]

    assert task["status"]["state"] == "completed"
    assert seen[0].api_key == "secret"


def test_non_round_trippable_config_defaults_fail_at_decoration() -> None:
    class InvalidConfig(AgentConfig):
        value: int = 1

        @field_serializer("value")
        def serialize_value(self, value: int) -> dict[str, int]:
            return {"value": value}

    with pytest.raises(
        TypeError,
        match="serializers must return values accepted by their declared field types",
    ):
        a2a_agent(
            identity=AgentIdentity(name="invalid", description="test", version="1"),
            config=InvalidConfig,
        )


def test_typed_agent_config_can_disable_readback() -> None:
    class RuntimeConfig(AgentConfig):
        model: str | None = None

    @a2a_agent(
        identity=AgentIdentity(
            name="configured-agent", description="test", version="1"
        ),
        config=RuntimeConfig,
        config_readback=False,
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[RuntimeConfig]) -> TaskResult:
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        methods = _extension(card, AGENT_CONFIG_V1.uri)["params"]["methods"]
        assert set(methods) == {"set"}
        assert client.get("/ext/agent-config").status_code == 405


def test_reserved_extension_uri_requires_canonical_sdk_definition() -> None:
    counterfeit = ExtensionDefinition(
        uri=MCP_CONFIG_V1.uri,
        description="counterfeit",
        endpoint="/evil",
        core_operations={
            "add": OperationDefinition(
                name="add",
                method="POST",
                path="/evil",
                implementation=ImplementationOwner.RUNTIME,
            )
        },
    )

    with pytest.raises(ValueError, match="must use its canonical SDK definition"):
        enable(counterfeit)
    with pytest.raises(ValueError, match="must use its canonical SDK definition"):
        extension(counterfeit.add)


def test_activation_description_overrides_sdk_default_and_merges_with_handlers() -> (
    None
):
    class Agent:
        @extension(SNAPSHOT_V1.save)
        async def save(self, request: ObjectSnapshotSaveRequest):
            return request

        @extension(SNAPSHOT_V1.load)
        async def load(self, request: ObjectSnapshotLoadRequest):
            return request

    registry = build_registry(
        Agent(),
        [
            enable(
                SNAPSHOT_V1,
                description="Save and restore this runtime's native state.",
            )
        ],
    )

    assert registry.card_extensions()[0]["description"] == (
        "Save and restore this runtime's native state."
    )


def test_trajectory_context_variant_is_advertised_only_with_handler() -> None:
    async def sdk_handler(request: TaskTrajectoryRequest | TaskObjectTrajectoryRequest):
        return {"trajectory": []}

    task_only = build_registry(
        object(),
        [enable(TRAJECTORY_V1)],
        sdk_handlers={("urn:agentenv:trajectory/v1", "get"): sdk_handler},
    )
    request = task_only.card_extensions()[0]["params"]["methods"]["get"]["request"]
    assert request == {
        "required": ["task_id"],
        "oneOf": [{}, {"required": ["objects"]}],
    }

    class ContextAgent:
        @extension(TRAJECTORY_V1.get.context)
        async def get_by_context(self, request: ContextTrajectoryRequest):
            return {"trajectory": []}

        @extension(TRAJECTORY_V1.get.context_objects)
        async def upload_by_context(self, request: ContextObjectTrajectoryRequest):
            return {"objects": {"trajectory": {"size_bytes": 1}}}

    with_context = build_registry(
        ContextAgent(),
        [enable(TRAJECTORY_V1)],
        sdk_handlers={("urn:agentenv:trajectory/v1", "get"): sdk_handler},
    )
    request = with_context.card_extensions()[0]["params"]["methods"]["get"]["request"]
    assert request == {
        "oneOf": [
            {"required": ["task_id"]},
            {"required": ["task_id", "objects"]},
            {"required": ["context_id"]},
            {"required": ["context_id", "objects"]},
        ],
    }

    with pytest.raises(TypeError, match="unsupported configuration"):
        enable(TRAJECTORY_V1, source="task_result")


@pytest.mark.asyncio
async def test_default_handler_delegation_accepts_sdk_request_variant() -> None:
    services = _SdkServices([enable(TRAJECTORY_V1)], None)
    services.task_trajectories["task-1"] = NativeTrajectory(
        format="events/v1", payload=[{"type": "result"}]
    )

    result = await DefaultExtensionHandlers(services).call(
        TRAJECTORY_V1.get,
        TaskTrajectoryRequest(task_id="task-1"),
    )

    assert result == {"trajectory": [{"type": "result"}]}


def test_trajectory_override_annotates_every_sdk_variant() -> None:
    class Complete:
        @extension(TRAJECTORY_V1.get)
        async def get(self, request: TaskTrajectoryRequest | TaskObjectTrajectoryRequest):
            return {"trajectory": []}

    class TypingUnion:
        @extension(TRAJECTORY_V1.get)
        async def get(self, request: Union[TaskTrajectoryRequest, TaskObjectTrajectoryRequest]):
            return {"trajectory": []}

    for agent in (Complete(), TypingUnion()):
        assert build_registry(agent).conformance() == {
            "standard_operation_overrides": [
                {"uri": TRAJECTORY_V1.uri, "operation": "get"}
            ]
        }

    class TaskOnly:
        @extension(TRAJECTORY_V1.get)
        async def get(self, request: TaskTrajectoryRequest):
            return {"trajectory": []}

    with pytest.raises(ValueError, match="must annotate.*TaskObjectTrajectoryRequest"):
        build_registry(TaskOnly())


def test_snapshot_requires_atomic_core_and_changelog_groups() -> None:
    class IncompleteSnapshot:
        @extension(SNAPSHOT_V1.save)
        async def save(self, request: ObjectSnapshotSaveRequest):
            return request

    with pytest.raises(ValueError, match="missing runtime handler.*load"):
        build_registry(IncompleteSnapshot())

    class IncompleteChangelog:
        @extension(SNAPSHOT_V1.save)
        async def save(self, request: ObjectSnapshotSaveRequest):
            return request

        @extension(SNAPSHOT_V1.load)
        async def load(self, request: ObjectSnapshotLoadRequest):
            return request

        @extension(SNAPSHOT_V1.changelog.enable)
        async def enable(self, request: NamespaceChangelogEnableRequest):
            return request

    with pytest.raises(ValueError, match="feature .*changelog is incomplete"):
        build_registry(IncompleteChangelog())

    class CompleteSnapshot:
        @extension(SNAPSHOT_V1.save)
        async def save(self, request: ObjectSnapshotSaveRequest):
            return request

        @extension(SNAPSHOT_V1.load)
        async def load(self, request: ObjectSnapshotLoadRequest):
            return request

        @extension(SNAPSHOT_V1.changelog.enable)
        async def enable(self, request: NamespaceChangelogEnableRequest):
            return request

        @extension(SNAPSHOT_V1.changelog.apply)
        async def apply(self, request: ObjectChangelogApplyRequest):
            return request

    registered_snapshot = build_registry(CompleteSnapshot()).extension(SNAPSHOT_V1.uri)
    assert registered_snapshot is not None
    assert {
        operation.definition.name
        for operation in registered_snapshot.operations.values()
    } == {
        "save",
        "load",
        "enable-changelog",
        "apply-changelog",
    }


def test_standard_extension_handler_implicitly_activates_definition() -> None:
    class Agent:
        @extension(SNAPSHOT_V1.save)
        async def save(self, request: ObjectSnapshotSaveRequest):
            return request

    with pytest.raises(ValueError, match="missing runtime handler.*load"):
        build_registry(Agent())

    class CompleteAgent:
        @extension(SNAPSHOT_V1.save)
        async def save(self, request: ObjectSnapshotSaveRequest):
            return request

        @extension(SNAPSHOT_V1.load)
        async def load(self, request: ObjectSnapshotLoadRequest):
            return request

    registry = build_registry(CompleteAgent())
    assert registry.extension(SNAPSHOT_V1.uri) is not None


def test_sdk_operations_reject_partial_override() -> None:
    class Agent:
        @extension(TRIGGERS_V1.decide)
        async def decide(self, request: TriggerDecideRequest):
            return {"parts": [], "done": False, "fired": []}

    async def register(request: TriggerRegisterRequest):
        return {}

    async def decide(request: TriggerDecideRequest):
        return {}

    async def state():
        return {"firing_log": []}

    handlers = {
        ("urn:agentenv:triggers/v1", "register"): register,
        ("urn:agentenv:triggers/v1", "decide"): decide,
        ("urn:agentenv:triggers/v1", "state"): state,
    }
    with pytest.raises(ValueError, match="must be overridden together"):
        build_registry(Agent(), sdk_handlers=handlers)


def test_complete_mcp_override_can_call_defaults_without_orphaning_sdk_state() -> None:
    seen: list[TaskRequest] = []

    @a2a_agent(
        identity=AgentIdentity(name="mcp-override", description="test", version="1"),
        extensions=(MCP_CONFIG_V1,),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            seen.append(request)
            return TaskResult.text("ok")

        @extension(MCP_CONFIG_V1.add)
        async def add_mcp(self, request: McpAddRequest):
            result = await self.default_handlers.call(MCP_CONFIG_V1.add, request)
            return {**result, "augmented": True}

        @extension(MCP_CONFIG_V1.list)
        async def list_mcp(self):
            return await self.default_handlers.call(MCP_CONFIG_V1.list)

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        added = _operation(
            client,
            card,
            MCP_CONFIG_V1.uri,
            "add",
            {"url": "https://example.test/mcp"},
        ).json()
        assert added["augmented"] is True

        registered = _operation(client, card, MCP_CONFIG_V1.uri, "list").json()
        assert registered["mcp_servers"][added["name"]]["url"] == (
            "https://example.test/mcp"
        )

        response = client.post("/a2a", json=_message_request())
        assert response.json()["result"]["status"]["state"] == "completed"

    assert seen[0].mcp_servers[added["name"]]["url"] == "https://example.test/mcp"


@pytest.mark.asyncio
async def test_concurrent_duplicate_skill_install_is_rejected() -> None:
    install_started = asyncio.Event()
    finish_install = asyncio.Event()
    installs = 0

    @a2a_agent(
        identity=AgentIdentity(name="skill-agent", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @extension(SKILL_CONFIG_V1.add.inline)
        async def add_inline(self, request: InlineSkillRequest):
            return {"name": request.name}

        @extension(SKILL_CONFIG_V1.add.bundle)
        async def add_bundle(self, request: BundleSkillRequest):
            nonlocal installs
            installs += 1
            install_started.set()
            await finish_install.wait()
            return {"name": request.name}

    payload = _skill_bundle_payload()
    transport = httpx.ASGITransport(app=Agent().create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.post("/ext/skill-config", json=payload))
        await install_started.wait()
        second = asyncio.create_task(client.post("/ext/skill-config", json=payload))
        await asyncio.sleep(0)
        finish_install.set()
        first_response, second_response = await asyncio.gather(first, second)

        assert first_response.status_code == 200
        assert second_response.status_code == 409
        assert installs == 1
        skill_list = await client.get("/ext/skill-config")
        assert skill_list.json()["skills"] == {
            "review": {"description": "Review code"}
        }


def test_bundle_skill_registration_keeps_no_read_grants() -> None:
    seen: list[TaskRequest] = []

    @a2a_agent(
        identity=AgentIdentity(name="skill-agent", description="test", version="1")
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            seen.append(request)
            return TaskResult.text("ok")

        @extension(SKILL_CONFIG_V1.add.inline)
        async def add_inline(self, request: InlineSkillRequest):
            return {"name": request.name}

        @extension(SKILL_CONFIG_V1.add.bundle)
        async def add_bundle(self, request: BundleSkillRequest):
            return {"name": request.name}

    with TestClient(Agent().create_app()) as client:
        assert (
            client.post("/ext/skill-config", json=_skill_bundle_payload()).status_code
            == 200
        )
        response = client.post("/a2a", json=_message_request())
        assert response.json()["result"]["status"]["state"] == "completed"

    assert seen[0].skills == ({"name": "review", "description": "Review code"},)


def test_custom_extension_cannot_claim_agentenv_namespace() -> None:
    with pytest.raises(ValueError, match="cannot redefine urn:agentenv"):
        custom_extension(
            uri="urn:agentenv:private/v1",
            operation="run",
            method="POST",
            path="/ext/private/v1",
        )


def test_shared_custom_definition_supports_multiple_operations_per_uri() -> None:
    class CollectRequest(BaseModel):
        model_config = {"extra": "forbid"}

        scope: str

    diagnostics = ExtensionDefinition(
        uri="urn:example:diagnostics/v1",
        description="Collect and inspect diagnostics.",
        endpoint="/ext/diagnostics/v1",
        core_operations={
            "collect": OperationDefinition(
                name="collect",
                method="POST",
                path="/ext/diagnostics/v1",
                implementation=ImplementationOwner.RUNTIME,
                request=CollectRequest,
                response=FieldSchema(required=("diagnostics",)),
            ),
            "status": OperationDefinition(
                name="status",
                method="GET",
                path="/ext/diagnostics/v1",
                implementation=ImplementationOwner.RUNTIME,
                response=FieldSchema(required=("status",)),
            ),
        },
    )

    @a2a_agent(
        identity=AgentIdentity(
            name="diagnostic-agent", description="test", version="1"
        ),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @extension(diagnostics.collect)
        async def collect(self, request: CollectRequest):
            return {"diagnostics": {"scope": request.scope}}

        @extension(diagnostics.status)
        async def status(self):
            return {"status": "ready"}

    Agent.collect.__annotations__["request"] = CollectRequest

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        declaration = _extension(card, diagnostics.uri)
        assert set(declaration["params"]["methods"]) == {"collect", "status"}
        assert _operation(
            client,
            card,
            diagnostics.uri,
            "collect",
            {"scope": "runtime"},
        ).json() == {"diagnostics": {"scope": "runtime"}}
        assert _operation(client, card, diagnostics.uri, "status").json() == {
            "status": "ready"
        }


def test_custom_extension_is_single_operation_sugar() -> None:
    class Agent:
        @custom_extension(
            uri="urn:example:diagnostics/v1",
            operation="collect",
            method="POST",
            path="/ext/diagnostics/v1",
        )
        async def collect(self):
            return {}

        @custom_extension(
            uri="urn:example:diagnostics/v1",
            operation="status",
            method="GET",
            path="/ext/diagnostics/v1",
        )
        async def status(self):
            return {}

    with pytest.raises(ValueError, match="define one shared ExtensionDefinition"):
        build_registry(Agent())


def test_route_collision_is_rejected_but_versioned_route_can_coexist() -> None:
    class Agent:
        @custom_extension(
            uri="urn:example:echo/v1",
            operation="run",
            method="POST",
            path="/ext/echo",
        )
        async def v1(self):
            return {}

        @custom_extension(
            uri="urn:example:echo/v2",
            operation="run",
            method="POST",
            path="/ext/echo/v2",
        )
        async def v2(self):
            return {}

    registry = build_registry(Agent())
    assert len(registry.extensions) == 2

    class CollidingAgent:
        @custom_extension(
            uri="urn:example:echo/v1",
            operation="run",
            method="POST",
            path="/ext/echo",
        )
        async def v1(self):
            return {}

        @custom_extension(
            uri="urn:example:echo/v3",
            operation="run",
            method="POST",
            path="/ext/echo",
        )
        async def v3(self):
            return {}

    with pytest.raises(ValueError, match="route POST /ext/echo is shared"):
        build_registry(CollidingAgent())


@pytest.mark.parametrize("error_type", [ValueError, TypeError])
def test_handler_implementation_errors_are_not_reported_as_bad_requests(
    error_type: type[Exception],
    caplog: pytest.LogCaptureFixture,
) -> None:
    internal_marker = "implementation bug"

    class FailingRequest(BaseModel):
        model_config = {"extra": "forbid"}

        value: str

    @a2a_agent(
        identity=AgentIdentity(name="failing-agent", description="test", version="1"),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @custom_extension(
            uri="urn:example:failing/v1",
            operation="run",
            method="POST",
            path="/ext/failing/v1",
            request=FailingRequest,
        )
        async def fail(self, request: FailingRequest):
            raise error_type(internal_marker)

    Agent.fail.__annotations__["request"] = FailingRequest

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        malformed = _operation(
            client,
            card,
            "urn:example:failing/v1",
            "run",
            {},
        )
        assert malformed.status_code == 400

        implementation_failure = _operation(
            client,
            card,
            "urn:example:failing/v1",
            "run",
            {"value": "valid"},
        )
        assert implementation_failure.status_code == 500
        detail = implementation_failure.text
        prefix = "An unexpected framework error occurred. Correlation ID: "
        assert detail.startswith(prefix)
        correlation_id = detail.removeprefix(prefix)
        assert len(correlation_id) == 32
        int(correlation_id, 16)
        assert correlation_id in caplog.text
        assert internal_marker not in implementation_failure.text


def test_extension_resolution_failures_are_opaque_and_correlated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    internal_marker = "private resolution detail"

    @a2a_agent(
        identity=AgentIdentity(name="failing-agent", description="test", version="1"),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @custom_extension(
            uri="urn:example:failing-resolution/v1",
            operation="run",
            method="POST",
            path="/ext/failing-resolution/v1",
        )
        async def fail(self):
            return {}

    def fail_selection(self, payload):
        raise RuntimeError(internal_marker)

    monkeypatch.setattr(
        "agentenv_protocol.a2a_agent.registry.RegisteredOperation.select_handler",
        fail_selection,
    )

    with TestClient(Agent().create_app()) as client:
        response = client.post("/ext/failing-resolution/v1", json={})

    assert response.status_code == 500
    detail = response.text
    prefix = "An unexpected framework error occurred. Correlation ID: "
    assert detail.startswith(prefix)
    correlation_id = detail.removeprefix(prefix)
    assert len(correlation_id) == 32
    int(correlation_id, 16)
    assert correlation_id in caplog.text
    assert internal_marker not in response.text


def test_generated_app_serves_sdk_extensions_and_a2a_lifecycle() -> None:
    seen: list[TaskRequest] = []
    skill_installs: list[str] = []

    class RuntimeConfig(AgentConfig):
        model: str | None = None
        harness: str | None = None

    @a2a_agent(
        identity=AgentIdentity(
            name="test-agent",
            description="test",
            version="1.0.0",
            input_modes=("text", "image/png"),
            skills=(
                {
                    "id": "built-in",
                    "name": "built-in",
                    "description": "Built-in capability",
                    "tags": ["built-in"],
                },
            ),
        ),
        config=RuntimeConfig,
        extensions=(
            MCP_CONFIG_V1,
            TRAJECTORY_V1,
            TRIGGERS_V1,
        ),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest[RuntimeConfig]) -> TaskResult:
            seen.append(request)
            return (
                TaskResult.builder()
                .succeeded()
                .add_text("ok")
                .add_structured_output({"answer": 42})
                .session_ref("native-session")
                .usage(Usage(tool_call_count=1))
                .native_trajectory(
                    format="test-events/v1", payload=[{"type": "result"}]
                )
                .build()
            )

        @extension(SKILL_CONFIG_V1.add.inline)
        async def add_inline_skill(self, request: InlineSkillRequest):
            skill_installs.append(request.name)
            return {"name": request.name}

        @extension(SKILL_CONFIG_V1.add.bundle)
        async def add_bundle_skill(self, request: BundleSkillRequest):
            skill_installs.append(request.name)
            return {"name": request.name}

    with TestClient(Agent().create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        assert card["capabilities"]["streaming"] is False
        assert card["capabilities"]["pushNotifications"] is False
        assert card["capabilities"]["stateTransitionHistory"] is False
        uris = {item["uri"] for item in card["capabilities"]["extensions"]}
        assert uris == {
            "urn:agentenv:agent-config/v1",
            "urn:agentenv:mcp-config/v1",
            "urn:agentenv:skill-config/v1",
            "urn:agentenv:trajectory/v1",
            "urn:agentenv:triggers/v1",
            STAGING_V1_URI,
        }

        assert (
            _operation(
                client,
                card,
                "urn:agentenv:agent-config/v1",
                "set",
                {
                    "model": "test-model",
                    "harness": "gateway-compatible-runtime",
                    "role": "auditor",
                },
            ).status_code
            == 200
        )
        config_response = _operation(
            client,
            card,
            "urn:agentenv:agent-config/v1",
            "get",
        )
        assert config_response.json()["config"]["role"] == "auditor"
        assert (
            _operation(
                client,
                card,
                "urn:agentenv:agent-config/v1",
                "set",
                {"metadata": {"trace_id": "not-yet-a-config-field"}},
            ).status_code
            == 400
        )
        assert (
            _operation(
                client,
                card,
                "urn:agentenv:agent-config/v1",
                "set",
                {"task_id": "external-task"},
            ).status_code
            == 400
        )
        assert (
            _operation(
                client,
                card,
                "urn:agentenv:agent-config/v1",
                "set",
                {"model_params": {"temperature": 0.2}},
            ).status_code
            == 400
        )
        assert (
            _operation(
                client,
                card,
                "urn:agentenv:agent-config/v1",
                "set",
                {"unknown": True},
            ).status_code
            == 400
        )

        assert _extension(card, MCP_CONFIG_V1.uri)["params"]["methods"]["add"][
            "request"
        ] == {
            "required": ["url"],
            "optional": ["headers", "name"],
        }
        mcp_add_response = _operation(
            client,
            card,
            MCP_CONFIG_V1.uri,
            "add",
            {"url": "https://example.test/mcp", "headers": {"X-Test": "secret"}},
        )
        assert mcp_add_response.status_code == 200
        mcp_name = mcp_add_response.json()["name"]
        assert mcp_name.startswith("mcp_")
        named_response = _operation(
            client,
            card,
            MCP_CONFIG_V1.uri,
            "add",
            {"url": "https://example.test/env/mcp", "name": "env"},
        )
        assert named_response.json()["name"] == "env"
        assert (
            _operation(client, card, MCP_CONFIG_V1.uri, "add", {"url": "https://example.test/other/mcp", "name": "env"}).status_code
            == 409
        )
        assert _operation(client, card, MCP_CONFIG_V1.uri, "list").json() == {
            "mcp_servers": {
                mcp_name: {
                    "url": "https://example.test/mcp",
                    "has_headers": True,
                },
                "env": {
                    "url": "https://example.test/env/mcp",
                    "has_headers": False,
                },
            }
        }
        skill_payload = {
            "name": "review",
            "description": "Review code",
            "skill_md": "# Review",
        }
        assert (
            _operation(
                client,
                card,
                SKILL_CONFIG_V1.uri,
                "add",
                skill_payload,
            ).status_code
            == 200
        )
        assert skill_installs == ["review"]
        assert _operation(client, card, SKILL_CONFIG_V1.uri, "list").json() == {
            "skills": {
                "built-in": {"description": "Built-in capability"},
                "review": {"description": "Review code"},
            }
        }
        updated_card = client.get("/.well-known/agent.json").json()
        assert {skill["name"]: skill for skill in updated_card["skills"]}["review"] == {
            "id": "skill-review",
            "name": "review",
            "description": "Review code",
            "tags": ["skill"],
        }
        duplicate = _operation(
            client,
            card,
            SKILL_CONFIG_V1.uri,
            "add",
            skill_payload,
        )
        assert duplicate.status_code == 409
        assert skill_installs == ["review"]

        registration = _operation(
            client,
            card,
            "urn:agentenv:triggers/v1",
            "register",
            {
                "triggers": [
                    {
                        "id": "first",
                        "when": {"type": "step", "turn": 1},
                        "actions": [{"type": "say", "text": "hello"}],
                    }
                ]
            },
        )
        assert registration.status_code == 200
        decision = _operation(
            client,
            card,
            "urn:agentenv:triggers/v1",
            "decide",
            {"turn": 1, "context_id": "context"},
        )
        assert decision.json() == {
            "parts": [{"kind": "text", "text": "hello"}],
            "done": False,
            "fired": ["first"],
        }

        payload = {
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "messageId": "message-1",
                    "role": "user",
                    "contextId": "context",
                    "parts": [{"kind": "text", "text": "hello"}],
                    "metadata": {
                        "trace_id": "trace-1",
                    },
                },
                "configuration": {"blocking": True},
            },
        }
        response = client.post("/a2a", json=payload)
        assert response.status_code == 200
        task = response.json()["result"]
        assert task["status"]["state"] == "completed"
        assert task["status"]["message"]["parts"][-2]["data"] == {
            "structured_output": {"answer": 42}
        }
        assert task["status"]["message"]["parts"][-1]["data"] == {
            "usage": {"tool_call_count": 1}
        }
        assert isinstance(seen[0].config, RuntimeConfig)
        assert seen[0].config.model == "test-model"
        assert seen[0].config.harness == "gateway-compatible-runtime"
        assert seen[0].config.timeout_seconds == 600
        assert seen[0].metadata == {
            "trace_id": "trace-1",
            "role": "auditor",
        }
        assert seen[0].mcp_servers
        assert seen[0].skills == (
            {
                "name": "review",
                "description": "Review code",
                "skill_md": "# Review",
            },
        )

        trajectory_response = _operation(
            client,
            card,
            "urn:agentenv:trajectory/v1",
            "get",
            {"task_id": task["id"]},
        )
        assert trajectory_response.json() == {"trajectory": [{"type": "result"}]}


def test_run_is_required() -> None:
    identity = AgentIdentity(name="test", description="test", version="1")

    @a2a_agent(identity=identity)
    class MissingRun(AgentEnvAgent):
        pass

    with pytest.raises(TypeError, match=r"must define run\(request\)"):
        MissingRun().create_app()


def test_a2a_agent_does_not_accept_a_custom_executor() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'executor'"):
        a2a_agent(
            identity=AgentIdentity(name="test", description="test", version="1"),
            executor=object(),  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("rpc_url", "extension_path", "method", "owner"),
    [
        ("/a2a", "/a2a", "POST", "A2A JSON-RPC"),
        ("/agentenv", "/agentenv", "POST", "A2A JSON-RPC"),
        ("/a2a", "/health", "GET", "health"),
        ("/a2a", "/health", "HEAD", "health"),
        ("/a2a", "/.well-known/agent.json", "GET", "Agent Card"),
        ("/a2a", "/.well-known/agent-card.json", "GET", "Agent Card"),
    ],
)
def test_extension_cannot_shadow_framework_route(
    rpc_url: str, extension_path: str, method: str, owner: str
) -> None:
    @a2a_agent(
        identity=AgentIdentity(
            name="test",
            description="test",
            version="1",
            url=rpc_url,
        ),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @custom_extension(
            uri="urn:example:shadow/v1",
            operation="shadow",
            method=method,
            path=extension_path,
        )
        async def shadow(self):
            return {}

    with pytest.raises(ValueError, match=rf"conflicts with {owner} route"):
        Agent().create_app()
