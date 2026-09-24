"""Canonical AgentEnv A2A extension definitions and handler decorators.

Extension contracts live here once.  Agent classes bind runtime behavior to a
known operation or request variant; they never repeat URIs, routes, or schemas.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ._triggers import MAX_SOLVER_MESSAGE_LENGTH
from .tasks.v1 import AgentConfig


class ImplementationOwner(str, Enum):
    SDK = "sdk"
    RUNTIME = "runtime"


class ExtensionRequest(BaseModel):
    """Base class for validated extension request bodies."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class McpAddRequest(ExtensionRequest):
    url: str
    headers: dict[str, str] | None = None
    name: str | None = None


class InlineSkillRequest(ExtensionRequest):
    name: str
    description: str
    skill_md: str


class S3SkillRequest(ExtensionRequest):
    name: str
    description: str
    skill_s3_url: str


class TaskTrajectoryRequest(ExtensionRequest):
    task_id: str
    trajectory_s3_prefix: str | None = None


class ContextTrajectoryRequest(ExtensionRequest):
    context_id: str
    trajectory_s3_prefix: str | None = None


class SnapshotSaveRequest(ExtensionRequest):
    context_id: str
    s3_prefix: str
    presigned_post: dict[str, Any] | None = None


class SnapshotLoadRequest(ExtensionRequest):
    s3_prefix: str
    target_context_id: str | None = None


class ChangelogEnableRequest(ExtensionRequest):
    s3_prefix: str
    roots: list[str] | None = None


class ChangelogApplyRequest(ExtensionRequest):
    s3_prefix: str
    up_to_tool_call_exclusive: int | None = None
    resume_conversation: bool = False
    target_context_id: str | None = None


class PeerAgent(ExtensionRequest):
    name: str
    url: str
    card: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class PeerAgentsSetRequest(ExtensionRequest):
    peers: list[PeerAgent]


class TriggerRegisterRequest(ExtensionRequest):
    triggers: list[dict[str, Any]]


class TriggerDecideRequest(ExtensionRequest):
    turn: int
    solver_message: str = Field(default="", max_length=MAX_SOLVER_MESSAGE_LENGTH)
    context_id: str = "default"
    env_triggers: dict[str, dict[str, Any]] | None = None


def _tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True, slots=True)
class FieldSchema:
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "required", _tuple(self.required))
        object.__setattr__(self, "optional", _tuple(self.optional))
        overlap = set(self.required) & set(self.optional)
        if overlap:
            raise ValueError(
                f"fields cannot be both required and optional: {sorted(overlap)}"
            )
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def to_card(self) -> dict[str, Any]:
        result = dict(self.extra)
        if self.required:
            result["required"] = list(self.required)
        if self.optional:
            result["optional"] = list(self.optional)
        return result


@dataclass(frozen=True, slots=True)
class RequestVariant:
    name: str
    model: type[BaseModel]
    support_required: bool = True
    implementation: ImplementationOwner | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, type) or not issubclass(self.model, BaseModel):
            raise TypeError(
                "request variant model must be a Pydantic BaseModel subclass"
            )
        if self.model.model_config.get("extra") != "forbid":
            raise TypeError(
                f"request model {self.model.__name__} must set extra='forbid'"
            )

    @property
    def fields(self) -> FieldSchema:
        return _model_field_schema(self.model)


@dataclass(frozen=True, slots=True)
class RequestDefinition:
    """Typed request shape with optional mutually-exclusive alternatives.

    Variants marked ``support_required`` are part of the extension's core
    contract; the others appear only when an agent binds or explicitly enables
    them.
    """

    model: type[BaseModel] | None = None
    variants: tuple[RequestVariant, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "variants", tuple(self.variants))
        if (self.model is None) == (not self.variants):
            raise ValueError(
                "request definition requires exactly one model or one or more variants"
            )
        if self.model is not None and (
            not isinstance(self.model, type) or not issubclass(self.model, BaseModel)
        ):
            raise TypeError("request model must be a Pydantic BaseModel subclass")
        if self.model is not None and self.model.model_config.get("extra") != "forbid":
            raise TypeError(
                f"request model {self.model.__name__} must set extra='forbid'"
            )
        names = [variant.name for variant in self.variants]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate request variant: {names}")

    @property
    def common(self) -> FieldSchema:
        if self.model is not None:
            return _model_field_schema(self.model)
        schemas = [variant.fields for variant in self.variants]
        first = schemas[0]
        required = tuple(
            name
            for name in first.required
            if all(name in schema.required for schema in schemas[1:])
        )
        optional = tuple(
            name
            for name in first.optional
            if all(name in schema.optional for schema in schemas[1:])
        )
        return FieldSchema(required=required, optional=optional)

    @property
    def core_variants(self) -> frozenset[str]:
        return frozenset(
            variant.name for variant in self.variants if variant.support_required
        )

    @property
    def optional_variants(self) -> frozenset[str]:
        return frozenset(
            variant.name for variant in self.variants if not variant.support_required
        )

    def variant(self, name: str) -> RequestVariant:
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise KeyError(name)

    def enabled_variants(
        self, optional: Iterable[str] = ()
    ) -> tuple[RequestVariant, ...]:
        requested = frozenset(optional)
        unknown = requested - self.optional_variants
        if unknown:
            raise ValueError(f"unknown optional request variants: {sorted(unknown)}")
        return tuple(
            variant
            for variant in self.variants
            if variant.support_required or variant.name in requested
        )

    def to_card(self, optional: Iterable[str] = ()) -> dict[str, Any]:
        result = self.common.to_card()
        variants = self.enabled_variants(optional)
        if not variants:
            return result

        common_names = set(self.common.required) | set(self.common.optional)

        if len(variants) == 1:
            only = variants[0].fields
            required = only.required
            optional_fields = only.optional
            if required:
                result["required"] = list(required)
            if optional_fields:
                result["optional"] = list(optional_fields)
            return result

        result["oneOf"] = [
            FieldSchema(
                required=tuple(
                    name for name in variant.fields.required if name not in common_names
                ),
                optional=tuple(
                    name for name in variant.fields.optional if name not in common_names
                ),
            ).to_card()
            for variant in variants
        ]
        return result

    def model_for_variant(self, variant: str | None) -> type[BaseModel]:
        if variant is None:
            if self.model is None:
                raise ValueError("request variant was not selected")
            return self.model
        return self.variant(variant).model

    def select_variant(
        self, payload: Mapping[str, Any], optional: Iterable[str] = ()
    ) -> str | None:
        variants = self.enabled_variants(optional)
        if not variants:
            missing = set(self.common.required) - set(payload)
            if missing:
                raise ValueError(f"missing required fields: {sorted(missing)}")
            return None

        common_names = set(self.common.required) | set(self.common.optional)
        disabled = {
            field_name
            for variant in self.variants
            if variant not in variants
            for field_name in (*variant.fields.required, *variant.fields.optional)
            if field_name not in common_names
        }
        supplied_disabled = disabled & set(payload)
        if supplied_disabled:
            raise ValueError(
                f"unsupported request variant fields: {sorted(supplied_disabled)}"
            )

        missing_common = set(self.common.required) - set(payload)
        if missing_common:
            raise ValueError(f"missing required fields: {sorted(missing_common)}")

        matches = []
        failures: dict[str, str] = {}
        for variant in variants:
            try:
                variant.model.model_validate(payload)
            except ValidationError as exc:
                failures[variant.name] = "; ".join(
                    f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
                    for error in exc.errors(include_input=False, include_url=False)
                )
                continue
            matches.append(variant.name)
        if len(matches) != 1:
            names = [variant.name for variant in variants]
            detail = "; ".join(
                f"{name} ({failures[name]})" for name in names if name in failures
            )
            suffix = f"; validation errors: {detail}" if detail else ""
            raise ValueError(
                f"request must match exactly one of variants: {names}{suffix}"
            )
        return matches[0]


def _model_field_schema(model: type[BaseModel]) -> FieldSchema:
    required: list[str] = []
    optional: list[str] = []
    for name, field_info in model.model_fields.items():
        alias = field_info.validation_alias
        if alias is not None and not isinstance(alias, str):
            raise TypeError(
                f"request model {model.__name__}.{name} must use a string validation alias"
            )
        wire_name = alias or name
        (required if field_info.is_required() else optional).append(wire_name)
    return FieldSchema(required=tuple(required), optional=tuple(optional))


@dataclass(frozen=True, slots=True)
class OperationDefinition:
    name: str
    method: str
    path: str
    implementation: ImplementationOwner
    request: RequestDefinition | type[BaseModel] | None = None
    response: FieldSchema | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", self.method.upper())
        if isinstance(self.request, type) and issubclass(self.request, BaseModel):
            object.__setattr__(self, "request", RequestDefinition(model=self.request))
        elif self.request is not None and not isinstance(
            self.request, RequestDefinition
        ):
            raise TypeError(
                "operation request must be a Pydantic BaseModel subclass or "
                "RequestDefinition"
            )
        if not self.path.startswith("/"):
            raise ValueError(
                f"extension operation path must be absolute: {self.path!r}"
            )


@dataclass(frozen=True, slots=True)
class OperationGroup:
    operations: Mapping[str, OperationDefinition]
    required_together: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", MappingProxyType(dict(self.operations)))


@dataclass(frozen=True, slots=True)
class ExtensionConfiguration:
    """Validated configuration contributed by an extension definition."""

    wire_params: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "wire_params", MappingProxyType(dict(self.wire_params))
        )
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        object.__setattr__(self, "features", frozenset(self.features))


@dataclass(frozen=True, slots=True)
class ExtensionDefinition:
    uri: str
    description: str
    endpoint: str | None
    core_operations: Mapping[str, OperationDefinition] = field(default_factory=dict)
    optional_features: Mapping[str, OperationGroup] = field(default_factory=dict)
    configuration_validator: Callable[..., ExtensionConfiguration] | None = None

    def __post_init__(self) -> None:
        if self.endpoint is not None and not self.endpoint.startswith("/"):
            raise ValueError(f"extension endpoint must be absolute: {self.endpoint!r}")
        core = dict(self.core_operations)
        optional = {name: group for name, group in self.optional_features.items()}
        all_names = [operation.name for operation in core.values()]
        for group in optional.values():
            all_names.extend(operation.name for operation in group.operations.values())
        if len(all_names) != len(set(all_names)):
            raise ValueError(f"duplicate operation in {self.uri}")
        if self.configuration_validator is not None and not callable(
            self.configuration_validator
        ):
            raise TypeError("configuration_validator must be callable")
        object.__setattr__(self, "core_operations", MappingProxyType(core))
        object.__setattr__(self, "optional_features", MappingProxyType(optional))

    @property
    def operations(self) -> Mapping[str, OperationDefinition]:
        result = dict(self.core_operations)
        for group in self.optional_features.values():
            result.update(group.operations)
        return MappingProxyType(result)

    def operation(self, name: str) -> OperationDefinition:
        try:
            return self.operations[name]
        except KeyError as exc:
            raise ValueError(f"{self.uri} has no operation {name!r}") from exc

    def __getattr__(self, name: str) -> "OperationReference | _FeatureOperations":
        if name in self.optional_features:
            return _FeatureOperations(self, name)
        try:
            self.operation(name)
        except ValueError as exc:
            raise AttributeError(name) from exc
        return OperationReference(self, name)


@dataclass(frozen=True, slots=True)
class ExtensionActivation:
    definition: ExtensionDefinition
    description: str | None = None
    features: frozenset[str] = frozenset()
    variants: Mapping[str, frozenset[str]] = field(default_factory=dict)
    wire_params: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    required: bool | None = None

    def __post_init__(self) -> None:
        if self.description is not None:
            if not isinstance(self.description, str):
                raise TypeError("description must be a string")
            if not self.description.strip():
                raise ValueError("extension description must not be empty")
        object.__setattr__(self, "features", frozenset(self.features))
        unknown_features = set(self.features) - set(self.definition.optional_features)
        if unknown_features:
            raise ValueError(
                f"unknown features for {self.definition.uri}: "
                f"{sorted(unknown_features)}"
            )
        variants = {name: frozenset(values) for name, values in self.variants.items()}
        unknown_operations = set(variants) - set(self.definition.operations)
        if unknown_operations:
            raise ValueError(
                f"unknown variant operations for {self.definition.uri}: "
                f"{sorted(unknown_operations)}"
            )
        for operation_name, enabled in variants.items():
            request = self.definition.operation(operation_name).request
            if request is None:
                raise ValueError(
                    f"{self.definition.uri}.{operation_name} has no request variants"
                )
            unknown_variants = set(enabled) - request.optional_variants
            if unknown_variants:
                raise ValueError(
                    f"unknown optional request variants for "
                    f"{self.definition.uri}.{operation_name}: {sorted(unknown_variants)}"
                )
        object.__setattr__(self, "variants", MappingProxyType(variants))
        object.__setattr__(
            self, "wire_params", MappingProxyType(dict(self.wire_params))
        )
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class HandlerBinding:
    extension: ExtensionDefinition
    operation: str
    variant: str | None = None


_HANDLER_BINDING = "_agentenv_a2a_handler_binding"


def _bind(fn: Callable, binding: HandlerBinding) -> Callable:
    if getattr(fn, _HANDLER_BINDING, None) is not None:
        raise TypeError(f"{fn.__name__} already has an A2A extension binding")
    setattr(fn, _HANDLER_BINDING, binding)
    return fn


@dataclass(frozen=True, slots=True)
class OperationReference:
    """Typed reference passed to the generic :func:`extension` decorator."""

    extension_definition: ExtensionDefinition
    operation: str
    request_variant: str | None = None

    def variant(self, name: str) -> "OperationReference":
        operation = self.extension_definition.operation(self.operation)
        if operation.request is None:
            raise ValueError(f"{operation.name} has no request variants")
        try:
            operation.request.variant(name)
        except KeyError as exc:
            raise ValueError(
                f"{operation.name} has no request variant {name!r}"
            ) from exc
        return OperationReference(self.extension_definition, self.operation, name)

    def __getattr__(self, name: str) -> "OperationReference":
        try:
            return self.variant(name)
        except ValueError as exc:
            raise AttributeError(name) from exc


class _FeatureOperations:
    def __init__(self, extension: ExtensionDefinition, feature: str) -> None:
        self._extension = extension
        self._feature = feature

    def __getattr__(self, operation: str) -> OperationReference:
        group = self._extension.optional_features[self._feature]
        if operation not in group.operations:
            raise AttributeError(operation)
        return OperationReference(self._extension, operation)


def extension(operation: OperationReference) -> Callable[[Callable], Callable]:
    """Bind a method to an operation or request variant.

    Binding any operation activates its containing extension. A shared
    non-AgentEnv ``ExtensionDefinition`` is the canonical way to bind multiple
    custom operations under one URI.
    """
    if not isinstance(operation, OperationReference):
        raise TypeError("@extension expects an OperationReference")
    _validate_reserved_definition(operation.extension_definition)

    def decorator(fn: Callable) -> Callable:
        return _bind(
            fn,
            HandlerBinding(
                operation.extension_definition,
                operation.operation,
                variant=operation.request_variant,
            ),
        )

    return decorator


def custom_extension(
    *,
    uri: str,
    operation: str,
    method: str,
    path: str,
    request: type[BaseModel] | None = None,
    response: FieldSchema | None = None,
    description: str = "",
) -> Callable[[Callable], Callable]:
    """Single-operation sugar for a non-AgentEnv extension."""
    if uri.startswith("urn:agentenv:"):
        raise ValueError(
            "custom_extension cannot redefine urn:agentenv:*; use a canonical hook "
            "or scoped override"
        )
    definition = ExtensionDefinition(
        uri=uri,
        description=description,
        endpoint=path,
        core_operations={
            operation: OperationDefinition(
                name=operation,
                method=method,
                path=path,
                implementation=ImplementationOwner.RUNTIME,
                request=request,
                response=response,
            )
        },
    )

    def decorator(fn: Callable) -> Callable:
        return extension(OperationReference(definition, operation))(fn)

    return decorator


def _op(
    name: str,
    method: str,
    path: str,
    owner: ImplementationOwner,
    *,
    request_model: type[BaseModel] | None = None,
    response_required: Iterable[str] = (),
    response_optional: Iterable[str] = (),
    variants: Iterable[RequestVariant] = (),
) -> OperationDefinition:
    response_required = _tuple(response_required)
    response_optional = _tuple(response_optional)
    variants = tuple(variants)
    if request_model is not None and variants:
        raise TypeError("an operation cannot declare both a request model and variants")
    request: RequestDefinition | type[BaseModel] | None = request_model
    if variants:
        request = RequestDefinition(variants=variants)
    return OperationDefinition(
        name=name,
        method=method,
        path=path,
        implementation=owner,
        request=request,
        response=(
            FieldSchema(required=response_required, optional=response_optional)
            if response_required or response_optional
            else None
        ),
    )


def _configure_agent_config(
    *,
    fields: Iterable[str],
    defaults: Mapping[str, Any] | None = None,
    schema: Mapping[str, Any] | None = None,
    readback: bool = False,
) -> ExtensionConfiguration:
    if isinstance(fields, str):
        raise TypeError("fields must be an iterable of field names, not a string")
    fields = tuple(fields)
    if not all(isinstance(name, str) and name for name in fields):
        raise TypeError("fields must contain non-empty strings")
    supported_fields = frozenset(fields)
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, Mapping):
        raise TypeError("defaults must be a mapping")
    unsupported_defaults = set(defaults) - supported_fields
    if unsupported_defaults:
        raise ValueError(
            "agent config defaults contain unsupported fields: "
            f"{sorted(unsupported_defaults)}"
        )
    if not isinstance(readback, bool):
        raise TypeError("readback must be a boolean")
    if schema is not None and not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping")
    request: dict[str, Any] = {"supported": sorted(supported_fields)}
    if schema is not None:
        request["schema"] = dict(schema)
    return ExtensionConfiguration(
        wire_params={"methods": {"set": {"request": request}}},
        options={"defaults": dict(defaults)},
        features=frozenset({"readback"}) if readback else frozenset(),
    )


def _configure_install(**declaration: Any) -> ExtensionConfiguration:
    return ExtensionConfiguration(wire_params=declaration)


SDK = ImplementationOwner.SDK
RUNTIME = ImplementationOwner.RUNTIME

AGENT_CONFIG_V1 = ExtensionDefinition(
    uri="urn:agentenv:agent-config/v1",
    description="Set or read agent configuration.",
    endpoint="/ext/agent-config",
    core_operations={
        "set": _op(
            "set",
            "POST",
            "/ext/agent-config",
            SDK,
            request_model=AgentConfig,
        )
    },
    optional_features={
        "readback": OperationGroup(
            {
                "get": _op(
                    "get",
                    "GET",
                    "/ext/agent-config",
                    SDK,
                    response_required=("config",),
                )
            }
        )
    },
    configuration_validator=_configure_agent_config,
)

MCP_CONFIG_V1 = ExtensionDefinition(
    uri="urn:agentenv:mcp-config/v1",
    description="Register MCP servers the agent should use.",
    endpoint="/ext/mcp-config",
    core_operations={
        "add": _op(
            "add",
            "POST",
            "/ext/mcp-config",
            SDK,
            request_model=McpAddRequest,
        ),
        "list": _op(
            "list", "GET", "/ext/mcp-config", SDK, response_required=("mcp_servers",)
        ),
    },
)

SKILL_CONFIG_V1 = ExtensionDefinition(
    uri="urn:agentenv:skill-config/v1",
    description="Register skills made available to the agent.",
    endpoint="/ext/skill-config",
    core_operations={
        "add": OperationDefinition(
            name="add",
            method="POST",
            path="/ext/skill-config",
            implementation=RUNTIME,
            request=RequestDefinition(
                variants=(
                    RequestVariant("inline", InlineSkillRequest),
                    RequestVariant("s3", S3SkillRequest),
                ),
            ),
        ),
        "list": _op(
            "list", "GET", "/ext/skill-config", SDK, response_required=("skills",)
        ),
    },
)

TRAJECTORY_V1 = ExtensionDefinition(
    uri="urn:agentenv:trajectory/v1",
    description="Retrieve an agent trajectory.",
    endpoint="/ext/trajectory",
    core_operations={
        "get": OperationDefinition(
            name="get",
            method="POST",
            path="/ext/trajectory",
            implementation=SDK,
            request=RequestDefinition(
                variants=(
                    RequestVariant("task", TaskTrajectoryRequest),
                    RequestVariant(
                        "context",
                        ContextTrajectoryRequest,
                        support_required=False,
                        implementation=RUNTIME,
                    ),
                ),
            ),
        )
    },
)

SNAPSHOT_V1 = ExtensionDefinition(
    uri="urn:agentenv:snapshot/v1",
    description="Save and restore agent-native state.",
    endpoint="/ext/snapshot",
    core_operations={
        "save": _op(
            "save",
            "POST",
            "/ext/snapshot",
            RUNTIME,
            request_model=SnapshotSaveRequest,
            response_required=("s3_prefix", "context_id", "timestamp"),
            response_optional=(
                "object_key",
                "file_count",
                "byte_count",
                "bytes_written",
            ),
        ),
        "load": _op(
            "load",
            "PUT",
            "/ext/snapshot",
            RUNTIME,
            request_model=SnapshotLoadRequest,
            response_required=("ok", "context_id"),
        ),
    },
    optional_features={
        "changelog": OperationGroup(
            operations={
                "enable": _op(
                    "enable-changelog",
                    "POST",
                    "/ext/snapshot/changelog",
                    RUNTIME,
                    request_model=ChangelogEnableRequest,
                    response_required=("ok", "s3_prefix", "roots"),
                ),
                "apply": _op(
                    "apply-changelog",
                    "PUT",
                    "/ext/snapshot/changelog",
                    RUNTIME,
                    request_model=ChangelogApplyRequest,
                    response_required=("ok", "count"),
                    response_optional=("context_id",),
                ),
            }
        )
    },
)

PEER_AGENTS_V1 = ExtensionDefinition(
    uri="urn:agentenv:peer-agents/v1",
    description="Configure peer A2A agents available for delegation.",
    endpoint="/ext/peer-agents",
    core_operations={
        "set": _op(
            "set",
            "POST",
            "/ext/peer-agents",
            RUNTIME,
            request_model=PeerAgentsSetRequest,
        ),
        "list": _op(
            "list", "GET", "/ext/peer-agents", RUNTIME, response_required=("peers",)
        ),
    },
)

TRIGGERS_V1 = ExtensionDefinition(
    uri="urn:agentenv:triggers/v1",
    description="Register triggers and decide deterministic responses.",
    endpoint="/ext/triggers",
    core_operations={
        "register": _op(
            "register",
            "POST",
            "/ext/triggers",
            SDK,
            request_model=TriggerRegisterRequest,
        ),
        "decide": _op(
            "decide",
            "POST",
            "/ext/triggers/decide",
            SDK,
            request_model=TriggerDecideRequest,
            response_required=("parts", "done", "fired"),
        ),
        "state": _op(
            "state",
            "GET",
            "/ext/triggers",
            SDK,
            response_required=("firing_log",),
        ),
    },
)

INSTALL_V1 = ExtensionDefinition(
    uri="urn:agentenv:install/v1",
    description="Declare commands for installing the agent into an existing target.",
    endpoint=None,
    configuration_validator=_configure_install,
)

STANDARD_EXTENSIONS: Mapping[str, ExtensionDefinition] = MappingProxyType(
    {
        extension.uri: extension
        for extension in (
            AGENT_CONFIG_V1,
            MCP_CONFIG_V1,
            SKILL_CONFIG_V1,
            TRAJECTORY_V1,
            SNAPSHOT_V1,
            PEER_AGENTS_V1,
            TRIGGERS_V1,
            INSTALL_V1,
        )
    }
)


def _validate_reserved_definition(definition: ExtensionDefinition) -> None:
    """Require reserved AgentEnv URIs to use their canonical SDK object."""
    if not definition.uri.startswith("urn:agentenv:"):
        return
    canonical = STANDARD_EXTENSIONS.get(definition.uri)
    if canonical is None or definition is not canonical:
        raise ValueError(
            f"reserved extension URI {definition.uri!r} must use its canonical "
            "SDK definition"
        )


def enable(
    definition: ExtensionDefinition,
    *,
    description: str | None = None,
    features: Iterable[str] = (),
    variants: Mapping[str, Iterable[str]] | None = None,
    required: bool | None = None,
    **configuration: Any,
) -> ExtensionActivation:
    """Enable a versioned extension definition with semantic configuration."""
    if not isinstance(definition, ExtensionDefinition):
        raise TypeError("enable expects an ExtensionDefinition")
    _validate_reserved_definition(definition)
    if description is not None and not isinstance(description, str):
        raise TypeError("description must be a string")

    validator = definition.configuration_validator
    if validator is None:
        validated = ExtensionConfiguration()
    else:
        validated = validator(**configuration)
        if not isinstance(validated, ExtensionConfiguration):
            raise TypeError(
                f"configuration validator for {definition.uri} must return "
                "ExtensionConfiguration"
            )

    if validator is None and configuration:
        raise TypeError(
            f"unsupported configuration for {definition.uri}: {sorted(configuration)}"
        )

    selected_features = set(features)
    selected_features.update(validated.features)

    return ExtensionActivation(
        definition=definition,
        description=description,
        features=frozenset(selected_features),
        variants={
            operation: frozenset(names) for operation, names in (variants or {}).items()
        },
        wire_params=validated.wire_params,
        options=validated.options,
        required=required,
    )
