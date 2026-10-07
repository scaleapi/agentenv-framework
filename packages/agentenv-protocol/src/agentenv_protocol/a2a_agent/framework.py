"""A2A agent application assembly built from declarations and handlers."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, TypeVar, get_args, get_origin, get_type_hints
from urllib.parse import urlparse

from a2a.types import AgentCapabilities
from pydantic import BaseModel, ValidationError
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ._triggers import TriggerEngine, TriggerError
from .extensions import (
    AGENT_CONFIG_V1,
    MCP_CONFIG_V1,
    SKILL_CONFIG_V1,
    TRAJECTORY_V1,
    TRIGGERS_V1,
    ExtensionActivation,
    ExtensionDefinition,
    ImplementationOwner,
    McpAddRequest,
    OperationReference,
    TaskObjectTrajectoryRequest,
    TaskTrajectoryRequest,
    TriggerDecideRequest,
    TriggerRegisterRequest,
    enable,
)
from .registry import RegisteredOperation, build_registry
from .staging import STAGING_ENDPOINT, StagingStore, staging_routes
from .tasks.v1 import (
    AgentConfig,
    AgentRunResult,
    TaskOutcome,
    TaskProgress,
    TaskRequest,
    TaskResult,
)
from .tasks.v1 import (
    DataPart as TaskDataPart,
)
from .tasks.v1 import (
    FilePart as TaskFilePart,
)
from .tasks.v1 import (
    TextPart as TaskTextPart,
)
from ..transfers import TransferError, upload

logger = logging.getLogger(__name__)

_AGENT_DEFINITION = "_agentenv_a2a_definition"
_AgentT = TypeVar("_AgentT", bound="AgentEnvAgent")
_MAX_CACHED_CONTEXTS = 1_024
_MAX_CACHED_TRAJECTORIES = 1_024


def _config_wire_names(config: type[AgentConfig]) -> dict[str, str]:
    """Map model field names to their single top-level validation name."""
    result: dict[str, str] = {}
    for field_name, field_info in config.model_fields.items():
        validation_alias = field_info.validation_alias
        if validation_alias is None:
            wire_name = field_name
        elif isinstance(validation_alias, str):
            wire_name = validation_alias
        else:
            raise TypeError(
                f"AgentConfig field {field_name!r} must use a simple string alias"
            )
        if wire_name in result.values():
            raise TypeError(f"AgentConfig fields have duplicate alias {wire_name!r}")
        result[field_name] = wire_name
    return result


def _config_write_only_wire_names(config: type[AgentConfig]) -> set[str]:
    """Return wire names explicitly marked as write-only in the config schema."""
    wire_names = _config_wire_names(config)
    return {
        wire_names[field_name]
        for field_name, field_info in config.model_fields.items()
        if isinstance(field_info.json_schema_extra, Mapping)
        and field_info.json_schema_extra.get("writeOnly") is True
    }


class _ConfigSchemaGenerator(GenerateJsonSchema):
    """Generate config schemas without exposing their runtime defaults."""

    def default_schema(
        self, schema: core_schema.WithDefaultSchema
    ) -> JsonSchemaValue:
        return self.generate_inner(schema["schema"])


class _BoundedContextSessions:
    """Least-recently-used in-process cache of native runtime sessions."""

    def __init__(self, max_entries: int = _MAX_CACHED_CONTEXTS) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, str] = OrderedDict()

    def get(self, context_id: str) -> str | None:
        try:
            value = self._entries.pop(context_id)
        except KeyError:
            return None
        self._entries[context_id] = value
        return value

    def set(self, context_id: str, session_ref: str) -> None:
        self._entries.pop(context_id, None)
        self._entries[context_id] = session_ref
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


class _BoundedTaskTrajectories:
    """Least-recently-used cache of completed native task trajectories."""

    def __init__(self, max_entries: int = _MAX_CACHED_TRAJECTORIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, Any] = OrderedDict()

    def __setitem__(self, task_id: str, trajectory: Any) -> None:
        self._entries.pop(task_id, None)
        self._entries[task_id] = trajectory
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def get(self, task_id: str) -> Any | None:
        try:
            trajectory = self._entries.pop(task_id)
        except KeyError:
            return None
        self._entries[task_id] = trajectory
        return trajectory

    def pop(self, task_id: str, default: Any = None) -> Any:
        return self._entries.pop(task_id, default)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(slots=True)
class _ContextLockState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class _BoundedContextLocks:
    """Bound idle per-context locks without disrupting active or waiting tasks."""

    def __init__(self, max_entries: int = _MAX_CACHED_CONTEXTS) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, _ContextLockState] = OrderedDict()

    @asynccontextmanager
    async def acquire(self, context_id: str) -> AsyncIterator[None]:
        state = self._entries.get(context_id)
        if state is None:
            state = _ContextLockState()
            self._entries[context_id] = state
        else:
            self._entries.move_to_end(context_id)
        state.users += 1
        try:
            async with state.lock:
                yield
        finally:
            state.users -= 1
            self._entries.move_to_end(context_id)
            self._trim_idle()

    def _trim_idle(self) -> None:
        while len(self._entries) > self._max_entries:
            for context_id, state in tuple(self._entries.items()):
                if state.users == 0:
                    del self._entries[context_id]
                    break
            else:
                # The temporary excess is proportional only to live work; it
                # is removed as soon as one of those contexts becomes idle.
                return


class AgentEnvAgent:
    """Typed base class for AgentEnv-compatible A2A agents."""

    def run(self, request: TaskRequest[Any]) -> AgentRunResult:
        """Execute one normalized A2A task."""
        raise NotImplementedError

    def create_app(self) -> Starlette:
        """Build this agent's unserved Starlette application."""
        return create_app(self)

    @property
    def default_handlers(self) -> DefaultExtensionHandlers:
        """Access default SDK handlers for composing extension overrides."""
        application = getattr(self, "_agentenv_a2a_application", None)
        if application is None:
            raise RuntimeError("default handlers are available after create_app()")
        return application.default_handlers

    def session_ref_for_context(self, context_id: str) -> str | None:
        """Return the framework-managed native session reference for a context."""
        application = getattr(self, "_agentenv_a2a_application", None)
        if application is None:
            raise RuntimeError("session references are available after create_app()")
        return application.services.session_ref_for_context(context_id)

    def set_session_ref_for_context(self, context_id: str, session_ref: str) -> None:
        """Associate a native session reference with an A2A context."""
        if not context_id:
            raise ValueError("context_id must not be empty")
        if not session_ref:
            raise ValueError("session_ref must not be empty")
        application = getattr(self, "_agentenv_a2a_application", None)
        if application is None:
            raise RuntimeError("session references are available after create_app()")
        application.services.set_session_ref_for_context(context_id, session_ref)

    def serve(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        **uvicorn_kwargs: Any,
    ) -> None:
        """Build and serve this agent with uvicorn."""
        serve(self, host=host, port=port, **uvicorn_kwargs)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentIdentity:
    name: str
    description: str
    version: str
    input_modes: tuple[str, ...] = ("text",)
    output_modes: tuple[str, ...] = ("text",)
    skills: tuple[Any, ...] = ()
    url: str = "/a2a"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_modes", tuple(self.input_modes))
        object.__setattr__(self, "output_modes", tuple(self.output_modes))
        object.__setattr__(self, "skills", tuple(self.skills))


def _rpc_path(card_url: str) -> str:
    """Resolve the local RPC route while preserving the card's public URL."""
    parsed = urlparse(card_url)
    if parsed.query or parsed.fragment:
        raise ValueError("AgentIdentity.url must not contain a query or fragment")
    if parsed.scheme or parsed.netloc:
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("AgentIdentity.url must be an HTTP(S) URL or absolute path")
        return parsed.path or "/"
    if not parsed.path.startswith("/"):
        raise ValueError("AgentIdentity.url path must start with '/'")
    return parsed.path


@dataclass(frozen=True, slots=True)
class _AgentDefinition:
    identity: AgentIdentity
    extensions: tuple[ExtensionActivation, ...]
    config: type[AgentConfig] | None = None
    lifespan: Callable[[Starlette], AbstractAsyncContextManager] | None = None
    workspace: Path | None = None


def a2a_agent(
    *,
    identity: AgentIdentity,
    extensions: Iterable[ExtensionDefinition | ExtensionActivation] = (),
    config: type[AgentConfig] | None = None,
    config_description: str | None = None,
    config_readback: bool = True,
    lifespan: Callable[[Starlette], AbstractAsyncContextManager] | None = None,
    workspace: str | Path | None = None,
) -> Callable[[type[_AgentT]], type[_AgentT]]:
    """Attach an A2A definition to an ``AgentEnvAgent`` subclass."""

    declared_extensions = tuple(
        ExtensionActivation(item) if isinstance(item, ExtensionDefinition) else item
        for item in extensions
    )
    if not all(isinstance(item, ExtensionActivation) for item in declared_extensions):
        raise TypeError(
            "extensions must contain ExtensionDefinition or ExtensionActivation"
        )
    if not isinstance(config_readback, bool):
        raise TypeError("config_readback must be a boolean")
    has_explicit_agent_config = any(
        activation.definition.uri == AGENT_CONFIG_V1.uri
        for activation in declared_extensions
    )
    if config is None and config_description is not None:
        raise TypeError("config_description requires config=")
    if config is None and not config_readback:
        raise TypeError("config_readback=False requires config=")
    if config is None and has_explicit_agent_config:
        raise TypeError(
            "agent-config/v1 is derived from config=; replace the explicit "
            "AGENT_CONFIG_V1 declaration with an AgentConfig subclass"
        )
    if config is not None:
        if not isinstance(config, type) or not issubclass(config, AgentConfig):
            raise TypeError("config must be an AgentConfig subclass")
        if config.model_config.get("extra") != "forbid" or not config.model_config.get(
            "frozen"
        ):
            raise TypeError(
                "AgentConfig subclasses must retain extra='forbid' and frozen=True"
            )
        if has_explicit_agent_config:
            raise TypeError(
                "config= automatically enables agent-config/v1; remove the explicit "
                "AGENT_CONFIG_V1 declaration"
            )
        try:
            default_config = config()
        except ValidationError as exc:
            raise TypeError(
                "AgentConfig fields must have defaults so partial deployment-time "
                "configuration updates can be validated"
            ) from exc
        wire_names = _config_wire_names(config)
        internal_defaults = default_config.model_dump(mode="python")
        default_values = {
            wire_names[name]: value for name, value in internal_defaults.items()
        }
        try:
            config.model_validate(default_values, by_name=True)
        except ValidationError as exc:
            raise TypeError(
                "AgentConfig serializers must return values accepted by their "
                "declared field types"
            ) from exc
        config_activation = enable(
            AGENT_CONFIG_V1,
            description=config_description,
            fields=tuple(wire_names.values()),
            defaults=default_values,
            schema=config.model_json_schema(
                mode="validation", schema_generator=_ConfigSchemaGenerator
            ),
            readback=config_readback,
        )
        declared_extensions = (config_activation, *declared_extensions)

    definition = _AgentDefinition(
        identity=identity,
        extensions=declared_extensions,
        config=config,
        lifespan=lifespan,
        workspace=Path(workspace) if workspace is not None else None,
    )

    def decorate(cls: type[_AgentT]) -> type[_AgentT]:
        if not issubclass(cls, AgentEnvAgent):
            raise TypeError("@a2a_agent requires an AgentEnvAgent subclass")
        setattr(cls, _AGENT_DEFINITION, definition)
        return cls

    return decorate


def create_app(agent: AgentEnvAgent) -> Starlette:
    """Build an unserved Starlette application for a declared agent."""
    application = getattr(agent, "_agentenv_a2a_application", None)
    if application is not None:
        return application.app
    return A2AAgentApplication(agent).app


def serve(
    agent: AgentEnvAgent,
    *,
    host: str | None = None,
    port: int | None = None,
    **uvicorn_kwargs: Any,
) -> None:
    """Build and serve a declared agent with uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - installation error
        raise ImportError(
            "Serving A2A agents requires: pip install 'agentenv-framework-protocol[agent]'"
        ) from exc

    resolved_host = host or os.environ.get("A2A_HOST", "0.0.0.0")
    resolved_port = (
        port if port is not None else int(os.environ.get("A2A_PORT", "8000"))
    )
    uvicorn.run(
        create_app(agent),
        host=resolved_host,
        port=resolved_port,
        **uvicorn_kwargs,
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_thaw(item) for item in value]
    return value


def _validation_error_detail(exc: ValidationError) -> str:
    """Return client-safe Pydantic errors without rejected values or docs URLs."""
    return exc.json(include_input=False, include_url=False)


async def _invoke(handler: Callable, request: Any | None) -> Any:
    arguments = () if request is None else (request,)
    return await handler(*arguments)


def _validate_run_handler_signature(
    handler: Callable, config_model: type[AgentConfig]
) -> None:
    parameters = list(inspect.signature(handler).parameters.values())
    if len(parameters) != 1 or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("agent.run must accept exactly one request argument")
    if not (
        inspect.iscoroutinefunction(handler) or inspect.isasyncgenfunction(handler)
    ):
        raise TypeError("agent.run must be an async function or async generator")
    parameter = parameters[0]
    # Include parents so postponed local annotations like TaskRequest[Parent] resolve.
    config_types = {
        candidate.__name__: candidate
        for candidate in config_model.__mro__
        if isinstance(candidate, type) and issubclass(candidate, AgentConfig)
    }
    try:
        annotation = get_type_hints(handler, localns=config_types).get(
            parameter.name, parameter.annotation
        )
    except (NameError, TypeError):
        annotation = parameter.annotation
    if annotation is TaskRequest:
        return
    if get_origin(annotation) is not TaskRequest:
        raise TypeError(
            "agent.run request argument must be annotated as TaskRequest or "
            f"TaskRequest[{config_model.__name__}]"
        )
    config_arguments = get_args(annotation)
    accepted_config = config_arguments[0] if len(config_arguments) == 1 else None
    if accepted_config is Any:
        return
    if not (
        isinstance(accepted_config, type)
        and issubclass(config_model, accepted_config)
        and issubclass(accepted_config, AgentConfig)
    ):
        raise TypeError(
            "agent.run request argument must accept the configured "
            f"{config_model.__name__} type"
        )


def _unwrap_bound_handler(handler: Callable) -> Callable:
    """Unwrap a decorated method while preserving its original binding."""
    resolved = inspect.unwrap(handler)
    if inspect.ismethod(handler) and not inspect.ismethod(resolved):
        owner = handler.__self__
        owner_type = owner if isinstance(owner, type) else type(owner)
        resolved = resolved.__get__(owner, owner_type)
    return resolved


def _opaque_extension_error(stage: str) -> HTTPException:
    """Log an extension failure while keeping implementation details off the wire."""
    correlation_id = uuid.uuid4().hex
    logger.exception("extension %s failed (correlation_id=%s)", stage, correlation_id)
    return HTTPException(
        status_code=500,
        detail=(
            f"An unexpected framework error occurred. Correlation ID: {correlation_id}"
        ),
    )


class _SdkServices:
    def __init__(
        self,
        activations: Iterable[ExtensionActivation],
        config_model: type[AgentConfig] | None,
    ) -> None:
        self._activations = {
            activation.definition.uri: activation for activation in activations
        }
        agent_config = self._activations.get(AGENT_CONFIG_V1.uri)
        self.config_defaults: dict[str, Any] = (
            dict(agent_config.options.get("defaults", {}))
            if agent_config is not None
            else {}
        )
        self.config: dict[str, Any] = {}
        self.config_model = config_model or AgentConfig
        self._write_only_config_fields = _config_write_only_wire_names(
            self.config_model
        )
        self.mcp_servers: dict[str, dict[str, Any]] = {}
        self.task_trajectories = _BoundedTaskTrajectories()
        self._context_sessions = _BoundedContextSessions()
        self.skills: list[Mapping[str, Any]] = []
        self.identity_skills: dict[str, str] = {}
        self._skill_registration_lock = asyncio.Lock()
        self.card: Any = None
        self.trigger_engine = TriggerEngine()

    def task_config(self) -> AgentConfig:
        return self.config_model.model_validate(
            {**self.config_defaults, **self.config}, by_name=True
        )

    def session_ref_for_context(self, context_id: str) -> str | None:
        return self._context_sessions.get(context_id)

    def set_session_ref_for_context(self, context_id: str, session_ref: str) -> None:
        self._context_sessions.set(context_id, session_ref)

    def attach_card(self, card: Any) -> None:
        self.card = card
        self.identity_skills = {skill.name: skill.description for skill in card.skills}

    def record_skill(self, registration: Mapping[str, Any]) -> None:
        """Record a successfully installed skill for subsequent task executions.

        A bundle's read grants are secret and short-lived, so they are not kept."""
        name = str(registration["name"])
        self.skills.append(
            {
                key: registration[key]
                for key in ("name", "description", "skill_md")
                if key in registration
            }
        )
        if self.card is not None:
            from a2a.types import AgentSkill

            self.card.skills.append(
                AgentSkill(
                    id=f"skill-{name}",
                    name=name,
                    description=str(registration["description"]),
                    tags=["skill"],
                )
            )

    def ensure_skill_is_new(self, registration: Mapping[str, Any]) -> None:
        name = str(registration["name"])
        if name in self.identity_skills or any(
            skill["name"] == name for skill in self.skills
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Skill '{name}' is already registered",
            )

    @asynccontextmanager
    async def skill_registration(
        self, registration: Mapping[str, Any]
    ) -> AsyncIterator[None]:
        """Serialize duplicate checking, installation, and registration."""
        async with self._skill_registration_lock:
            self.ensure_skill_is_new(registration)
            yield
            self.record_skill(registration)

    def handlers(self) -> dict[tuple[str, str], Callable]:
        return {
            (AGENT_CONFIG_V1.uri, "set"): self.agent_config_set,
            (AGENT_CONFIG_V1.uri, "get"): self.agent_config_get,
            (MCP_CONFIG_V1.uri, "add"): self.mcp_add,
            (MCP_CONFIG_V1.uri, "list"): self.mcp_list,
            (SKILL_CONFIG_V1.uri, "list"): self.skill_list,
            (TRAJECTORY_V1.uri, "get"): self.trajectory_get,
            (TRIGGERS_V1.uri, "register"): self.triggers_register,
            (TRIGGERS_V1.uri, "decide"): self.triggers_decide,
            (TRIGGERS_V1.uri, "state"): self.triggers_state,
        }

    async def agent_config_set(self, request: AgentConfig) -> dict[str, Any]:
        normalized_request = request.model_dump(mode="python")
        wire_names = _config_wire_names(type(request))
        payload = {
            wire_names[field_name]: normalized_request[field_name]
            for field_name in request.model_fields_set
        }
        try:
            validated = self.config_model.model_validate(
                {**self.config_defaults, **self.config, **payload}, by_name=True
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=400, detail=_validation_error_detail(exc)
            ) from exc
        normalized = validated.model_dump(mode="python")
        field_for_wire = {
            wire_name: field_name
            for field_name, wire_name in _config_wire_names(self.config_model).items()
        }
        pending = {
            **self.config,
            **{
                wire_name: normalized[field_for_wire[wire_name]]
                for wire_name in payload
            },
        }
        try:
            self.config_model.model_validate(
                {**self.config_defaults, **pending}, by_name=True
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "AgentConfig serializers must return values accepted by their "
                    "declared field types: "
                    f"{_validation_error_detail(exc)}"
                ),
            ) from exc
        provided_fields = {field_for_wire[name] for name in payload}
        for identity_field in ("name", "description"):
            if identity_field in provided_fields:
                identity_value = getattr(validated, identity_field)
                if not isinstance(identity_value, str) or not identity_value.strip():
                    raise HTTPException(
                        status_code=400,
                        detail=f"{identity_field} must be a non-empty string",
                    )
        self.config.update(pending)
        if self.card is not None:
            if "name" in provided_fields:
                self.card.name = validated.name
            if "description" in provided_fields:
                self.card.description = validated.description
        return {"status": "updated"}

    async def agent_config_get(self) -> dict[str, Any]:
        config = _thaw(self.config)
        for wire_name in self._write_only_config_fields:
            if wire_name in config:
                config[wire_name] = "***"
        return {"config": config}

    async def mcp_add(self, request: McpAddRequest) -> dict[str, Any]:
        url = request.url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="url must not be empty")
        for existing_name, existing in self.mcp_servers.items():
            if existing["url"] == url:
                raise HTTPException(
                    status_code=409,
                    detail=f"MCP URL '{url}' already registered as '{existing_name}'",
                )
        name = request.name or f"mcp_{uuid.uuid4().hex[:8]}"
        if name in self.mcp_servers:
            raise HTTPException(status_code=409, detail=f"MCP name '{name}' is already registered")
        self.mcp_servers[name] = {
            "url": url,
            "headers": dict(request.headers) if request.headers else None,
        }
        return {"status": "added", "name": name, "url": url}

    async def mcp_list(self) -> dict[str, Any]:
        return {
            "mcp_servers": {
                name: {
                    "url": registration["url"],
                    "has_headers": bool(registration.get("headers")),
                }
                for name, registration in self.mcp_servers.items()
            }
        }

    async def skill_list(self) -> dict[str, Any]:
        skills = {
            name: {"description": description}
            for name, description in self.identity_skills.items()
        }
        skills.update(
            {
                str(registration["name"]): {
                    "description": str(registration["description"])
                }
                for registration in self.skills
            }
        )
        return {"skills": skills}

    async def trajectory_get(
        self, request: TaskTrajectoryRequest | TaskObjectTrajectoryRequest
    ) -> dict[str, Any]:
        task_id = request.task_id
        trajectory = self.task_trajectories.get(task_id)
        if trajectory is None:
            raise HTTPException(
                status_code=404,
                detail=f"No completed trajectory available for task '{task_id}'",
            )
        if isinstance(request, TaskTrajectoryRequest):
            return {"trajectory": _thaw(trajectory.payload)}
        body = json.dumps(
            _thaw(trajectory.payload), separators=(",", ":"), default=str
        ).encode()
        uploaded = await upload(request.objects.trajectory, body)
        return {
            "objects": {
                "trajectory": uploaded.model_dump(mode="json"),
            }
        }

    async def triggers_register(
        self, request: TriggerRegisterRequest
    ) -> dict[str, Any]:
        return self.trigger_engine.register(request.model_dump(mode="python"))

    async def triggers_decide(self, request: TriggerDecideRequest) -> dict[str, Any]:
        return self.trigger_engine.decide(
            turn=request.turn,
            solver_message=request.solver_message,
            context_id=request.context_id,
            env_triggers=request.env_triggers,
        )

    async def triggers_state(self) -> dict[str, Any]:
        return self.trigger_engine.state()


class DefaultExtensionHandlers:
    """Stable access to default implementations of SDK-owned operations."""

    def __init__(self, services: _SdkServices) -> None:
        self._services = services

    async def call(
        self,
        operation: OperationReference,
        request: BaseModel | None = None,
    ) -> Any:
        if not isinstance(operation, OperationReference):
            raise TypeError("operation must be an OperationReference")
        if operation.request_variant is not None:
            raise ValueError("SDK delegation does not accept request variants")
        definition = operation.extension_definition.operation(operation.operation)
        if definition.implementation is not ImplementationOwner.SDK:
            raise ValueError(
                f"{operation.extension_definition.uri}.{operation.operation} "
                "is not SDK-owned"
            )
        handler = self._services.handlers().get(
            (operation.extension_definition.uri, operation.operation)
        )
        if handler is None:
            raise ValueError(
                f"no SDK implementation for "
                f"{operation.extension_definition.uri}.{operation.operation}"
            )
        expected = None
        if definition.request is not None:
            if definition.request.model is not None:
                expected = definition.request.model
            else:
                sdk_variant_models = [
                    variant.model
                    for variant in definition.request.variants
                    if (variant.implementation or definition.implementation)
                    is ImplementationOwner.SDK
                ]
                matching_models = [
                    model for model in sdk_variant_models if isinstance(request, model)
                ]
                if len(matching_models) == 1:
                    expected = matching_models[0]
        if definition.request is None:
            if request is not None:
                raise TypeError(f"{definition.name} does not accept a request")
        elif expected is None or not isinstance(request, expected):
            expected_names = (
                [variant.model.__name__ for variant in definition.request.variants]
                if definition.request.variants
                else [definition.request.model.__name__]
            )
            raise TypeError(
                f"{definition.name} requires one of {expected_names}, got "
                f"{type(request).__name__}"
            )
        return await _invoke(handler, request)


class A2AAgentApplication:
    """Resolved Agent Card, extension routes, and A2A task executor."""

    def __init__(self, agent: Any, definition: _AgentDefinition | None = None) -> None:
        definition = definition or getattr(agent, _AGENT_DEFINITION, None)
        if definition is None:
            raise TypeError("agent class must be decorated with @a2a_agent")
        if getattr(agent, "_agentenv_a2a_application", None) is not None:
            raise RuntimeError(
                "agent instance is already bound to an application; create a new "
                "agent instance for each application"
            )
        self.agent = agent
        self.definition = definition
        self.rpc_path = _rpc_path(definition.identity.url)
        handler = getattr(agent, "run", None)
        if (
            not callable(handler)
            or getattr(type(agent), "run", None) is AgentEnvAgent.run
        ):
            raise TypeError("an agent must define run(request)")
        resolved_handler = _unwrap_bound_handler(handler)
        _validate_run_handler_signature(
            resolved_handler, definition.config or AgentConfig
        )
        self.run_handler = handler
        self.streaming = inspect.isasyncgenfunction(resolved_handler)
        self.services = _SdkServices(definition.extensions, definition.config)
        self.default_handlers = DefaultExtensionHandlers(self.services)
        self.registry = build_registry(
            agent,
            definition.extensions,
            sdk_handlers=self.services.handlers(),
            request_model_overrides=(
                {(AGENT_CONFIG_V1.uri, "set"): definition.config}
                if definition.config is not None
                else None
            ),
        )
        self.registry.reject_framework_route_collisions(
            {
                ("/health", "GET"): "health",
                (self.rpc_path, "POST"): "A2A JSON-RPC",
                ("/.well-known/agent.json", "GET"): "Agent Card",
                ("/.well-known/agent-card.json", "GET"): "Agent Card",
            }
        )
        # Objects agent-env moves through the agent when its object store's grants cannot reach it.
        self.staging = StagingStore()
        for extension in self.registry.extensions if self.staging.max_bytes else ():
            for operation in extension.operations.values():
                if operation.definition.path.startswith(STAGING_ENDPOINT + "/"):
                    raise ValueError(
                        f"extension operation {extension.definition.uri}.{operation.definition.name} "
                        f"conflicts with the staging routes below {STAGING_ENDPOINT}"
                    )
        conformance = self.registry.conformance()
        for override in conformance["standard_operation_overrides"]:
            logger.warning(
                "Agent overrides SDK operation %s.%s",
                override["uri"],
                override["operation"],
            )
        self.card = self._build_card()
        self.services.attach_card(self.card)
        try:
            setattr(agent, "_agentenv_a2a_application", self)
        except AttributeError:
            # Slotted agents can still use the framework; only advanced
            # runtime integrations that need application state lose this hook.
            pass

        self.executor = _StandardExecutor(
            self.run_handler,
            self.services,
            workspace=definition.workspace,
            streaming=self.streaming,
        )

        routes = [Route("/health", self._health, methods=["GET"])]
        for extension in self.registry.extensions:
            for operation in extension.operations.values():
                routes.append(
                    Route(
                        operation.definition.path,
                        self._route_handler(extension.definition.uri, operation),
                        methods=[operation.definition.method],
                    )
                )
        if self.staging.max_bytes:
            routes.extend(staging_routes(self.staging))
        kwargs = {"routes": routes}
        if definition.lifespan is not None:
            kwargs["lifespan"] = definition.lifespan
        self.app = Starlette(**kwargs)

        try:
            from a2a.server.apps import A2AStarletteApplication
            from a2a.server.request_handlers import DefaultRequestHandler
            from a2a.server.tasks import InMemoryTaskStore
        except ImportError as exc:  # pragma: no cover - depends on installation extra
            raise ImportError(
                "A2A agent applications require: pip install 'agentenv-framework-protocol[agent]'"
            ) from exc

        request_handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=InMemoryTaskStore(),
        )
        A2AStarletteApplication(
            agent_card=self.card,
            http_handler=request_handler,
        ).add_routes_to_app(
            self.app,
            rpc_url=self.rpc_path,
        )
        self.app.state.agentenv_a2a = self

    async def _health(self, _request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    def _build_card(self) -> Any:
        try:
            from a2a.types import AgentCard, AgentExtension, AgentSkill
        except ImportError as exc:  # pragma: no cover - depends on installation extra
            raise ImportError(
                "A2A agent applications require: pip install 'agentenv-framework-protocol[agent]'"
            ) from exc

        identity = self.definition.identity
        skills = [
            skill if isinstance(skill, AgentSkill) else AgentSkill.model_validate(skill)
            for skill in identity.skills
        ]
        extensions = [
            AgentExtension.model_validate(item)
            for item in self.registry.card_extensions()
        ]
        if self.staging.max_bytes:
            extensions.append(AgentExtension.model_validate(self.staging.card_extension()))
        resolved_capabilities = AgentCapabilities(
            streaming=self.streaming,
            push_notifications=False,
            state_transition_history=False,
            extensions=extensions,
        )
        return AgentCard(
            name=identity.name,
            description=identity.description,
            version=identity.version,
            url=identity.url,
            default_input_modes=list(identity.input_modes),
            default_output_modes=list(identity.output_modes),
            skills=skills,
            capabilities=resolved_capabilities,
        )

    def _route_handler(
        self, extension_uri: str, operation: RegisteredOperation
    ) -> Callable:
        async def invoke(request: Request) -> JSONResponse:
            try:
                if operation.definition.method == "GET":
                    payload = dict(request.query_params)
                else:
                    raw = await request.body()
                    payload = json.loads(raw) if raw else {}
                    if not isinstance(payload, dict):
                        raise ValueError("request body must be a JSON object")
                input_payload = dict(payload)
                handler, variant = operation.select_handler(payload)
                if handler is None:
                    raise RuntimeError("operation has no implementation")
                request_model = operation.request_model(variant)
                if request_model is None:
                    if payload:
                        raise ValueError(
                            f"{operation.definition.name} does not accept request fields"
                        )
                    extension_request = None
                else:
                    extension_request = request_model.model_validate(payload)
            except HTTPException:
                raise
            except ValidationError as exc:
                raise HTTPException(
                    status_code=400, detail=_validation_error_detail(exc)
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise _opaque_extension_error("request resolution") from exc

            try:
                async def invoke_and_validate() -> Any:
                    result = await _invoke(handler, extension_request)
                    result = (
                        _thaw(result.model_dump(mode="json", by_alias=True))
                        if isinstance(result, BaseModel)
                        else _thaw(result)
                    )
                    if result is None:
                        result = {}
                    if operation.definition.response is not None:
                        if not isinstance(result, Mapping):
                            raise RuntimeError(
                                "operation response must be a JSON object"
                            )
                        response_model = operation.definition.response_model
                        if response_model is not None:
                            return response_model.model_validate(result).model_dump(
                                mode="json", by_alias=True, exclude_none=True
                            )
                        missing = set(operation.definition.response.required) - set(
                            result
                        )
                        if missing:
                            raise RuntimeError(
                                "operation response is missing fields: "
                                f"{sorted(missing)}"
                            )
                    return result

                if (
                    extension_uri == SKILL_CONFIG_V1.uri
                    and operation.definition.name == "add"
                ):
                    async with self.services.skill_registration(input_payload):
                        result = await invoke_and_validate()
                else:
                    result = await invoke_and_validate()
                return JSONResponse(result)
            except HTTPException:
                raise
            except TriggerError as exc:
                raise HTTPException(
                    status_code=exc.status_code, detail=str(exc)
                ) from exc
            except TransferError as exc:
                return JSONResponse(exc.body(), status_code=exc.status_code)
            except Exception as exc:
                raise _opaque_extension_error("invocation") from exc

        return invoke


class _StandardExecutor:
    """A2A SDK executor that delegates one normalized request to ``agent.run``."""

    def __init__(
        self,
        handler: Callable,
        services: _SdkServices,
        *,
        workspace: Path | None,
        streaming: bool,
    ) -> None:
        self._handler = handler
        self._services = services
        self._workspace = workspace
        self._streaming = streaming
        self._context_locks = _BoundedContextLocks()

    async def execute(self, context: Any, event_queue: Any) -> None:
        from a2a.server.tasks import TaskUpdater
        from a2a.types import InvalidParamsError
        from a2a.utils import new_task
        from a2a.utils.errors import ServerError

        try:
            task = context.current_task or new_task(context.message)
        except Exception as exc:
            raise ServerError(error=InvalidParamsError(message=str(exc))) from exc

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            if context.current_task is None:
                await event_queue.enqueue_event(task)
            await updater.start_work()

            task_config = self._services.task_config()
            metadata = {
                **_thaw(getattr(context.message, "metadata", None) or {}),
                **({"role": task_config.role} if task_config.role is not None else {}),
            }
            async with self._context_locks.acquire(task.context_id):
                request = TaskRequest(
                    task_id=task.id,
                    context_id=task.context_id,
                    parts=tuple(
                        _from_a2a_part(part) for part in (context.message.parts or [])
                    ),
                    config=task_config,
                    mcp_servers=self._services.mcp_servers,
                    skills=tuple(self._services.skills),
                    metadata=metadata,
                    session_ref=self._services.session_ref_for_context(task.context_id),
                    workspace=self._workspace,
                )
                if self._streaming:
                    result = await self._run_streaming(request, updater, event_queue)
                else:
                    result = await self._handler(request)
                if not isinstance(result, TaskResult):
                    raise TypeError("agent.run must return TaskResult")
                if result.session_ref is not None:
                    self._services.set_session_ref_for_context(
                        task.context_id, result.session_ref
                    )
                if result.native_trajectory is not None:
                    self._services.task_trajectories[task.id] = result.native_trajectory

            parts = _to_a2a_parts(result)
            message = updater.new_agent_message(parts=parts)
            if result.outcome is TaskOutcome.FAILED:
                await updater.failed(message=message)
            else:
                await updater.complete(message=message)
        except Exception:
            correlation_id = uuid.uuid4().hex
            logger.exception(
                "unhandled task execution error (correlation_id=%s)",
                correlation_id,
            )
            failure = TaskResult.failure(
                "framework.unhandled_exception",
                "An unexpected framework error occurred. "
                f"Correlation ID: {correlation_id}",
                error_type="infra_error",
            )
            await updater.failed(
                message=updater.new_agent_message(parts=_to_a2a_parts(failure))
            )

    async def _run_streaming(
        self,
        request: TaskRequest[Any],
        updater: Any,
        event_queue: Any,
    ) -> TaskResult:
        from a2a.types import (
            TaskState,
            TaskStatusUpdateEvent,
        )

        stream = self._handler(request)
        if inspect.isawaitable(stream):
            stream = await stream
        try:
            async for item in stream:
                if isinstance(item, TaskResult):
                    return item
                if isinstance(item, TaskProgress):
                    await updater.update_status(
                        TaskState.working,
                        message=updater.new_agent_message(
                            parts=_to_a2a_parts_collection(item.parts)
                        ),
                        metadata=_thaw(item.metadata) or None,
                    )
                    continue
                if isinstance(item, TaskStatusUpdateEvent):
                    self._validate_status_event(item, request)
                    await event_queue.enqueue_event(item)
                    continue
                raise TypeError(
                    "agent.run (streaming) must yield TaskProgress, "
                    "a non-terminal A2A status update event, or a terminal "
                    "TaskResult"
                )
        finally:
            await stream.aclose()
        raise TypeError(
            "agent.run (streaming) must yield a TaskResult before completing"
        )

    @staticmethod
    def _validate_event_identity(item: Any, request: TaskRequest[Any]) -> None:
        if item.task_id != request.task_id or item.context_id != request.context_id:
            raise ValueError(
                "streaming A2A event task_id and context_id must match the request"
            )

    @classmethod
    def _validate_status_event(cls, item: Any, request: TaskRequest[Any]) -> None:
        from a2a.types import TaskState

        cls._validate_event_identity(item, request)
        terminal_states = {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.rejected,
        }
        if item.final or item.status.state in terminal_states:
            raise ValueError(
                "streaming TaskStatusUpdateEvent must be non-terminal and final=False"
            )

    async def cancel(self, context: Any, event_queue: Any) -> None:
        """Mark the task canceled. The SDK then cancels the coroutine running it, so ``agent.run`` gets
        ``CancelledError`` at its next ``await`` and should stop whatever it started, such as a CLI process."""
        from a2a.server.tasks import TaskUpdater

        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def _from_a2a_part(part: Any) -> Any:
    root = part.root
    kind = getattr(root, "kind", None)
    metadata = getattr(root, "metadata", None) or {}
    if kind == "text":
        return TaskTextPart(text=root.text, metadata=metadata)
    if kind == "data":
        return TaskDataPart(data=root.data, metadata=metadata)
    if kind == "file":
        file = root.file
        return TaskFilePart(
            name=getattr(file, "name", None),
            mime_type=getattr(file, "mime_type", None),
            bytes=getattr(file, "bytes", None),
            uri=getattr(file, "uri", None),
            metadata=metadata,
        )
    raise ValueError(f"unsupported A2A part kind: {kind!r}")


def _to_a2a_parts(result: TaskResult) -> list[Any]:
    converted = _to_a2a_parts_collection(result.parts)

    from a2a.types import DataPart, Part

    metadata: dict[str, Any] = {}
    if not result.usage.is_empty:
        metadata["usage"] = result.usage.to_dict()
    if result.error is not None:
        metadata["error_type"] = result.error.error_type
        metadata["error_code"] = result.error.code
        metadata["error_message"] = result.error.message
    if metadata:
        converted.append(Part(root=DataPart(data=metadata)))
    return converted


def _to_a2a_parts_collection(parts: Iterable[Any]) -> list[Any]:
    from a2a.types import DataPart, FilePart, FileWithBytes, FileWithUri, Part, TextPart

    converted = []
    for part in parts:
        metadata = _thaw(part.metadata) or None
        if isinstance(part, TaskTextPart):
            root = TextPart(text=part.text, metadata=metadata)
        elif isinstance(part, TaskDataPart):
            root = DataPart(data=_thaw(part.data), metadata=metadata)
        elif isinstance(part, TaskFilePart):
            file = (
                FileWithBytes(
                    bytes=part.bytes, name=part.name, mime_type=part.mime_type
                )
                if part.bytes is not None
                else FileWithUri(uri=part.uri, name=part.name, mime_type=part.mime_type)
            )
            root = FilePart(file=file, metadata=metadata)
        else:  # pragma: no cover - TaskPart is closed
            raise TypeError(f"unsupported task result part: {type(part).__name__}")
        converted.append(Part(root=root))

    return converted
