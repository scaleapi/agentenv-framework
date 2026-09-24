"""Discovery and validation for A2A extension implementations."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel

from .extensions import (
    _HANDLER_BINDING,
    ExtensionActivation,
    ExtensionDefinition,
    HandlerBinding,
    ImplementationOwner,
    OperationDefinition,
    _validate_reserved_definition,
)


def _validate_handler_signature(
    handler: Callable,
    request_model: type[BaseModel] | None,
    *,
    label: str,
    allow_request_supertype: bool = False,
) -> None:
    if not inspect.iscoroutinefunction(handler):
        raise ValueError(f"handler for {label} must be an async function")
    parameters = list(inspect.signature(handler).parameters.values())
    if request_model is None:
        if parameters:
            raise ValueError(f"handler for {label} must not accept a request argument")
        return
    if len(parameters) != 1 or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError(
            f"handler for {label} must accept exactly one typed request argument"
        )
    parameter = parameters[0]
    try:
        annotation = get_type_hints(handler).get(parameter.name, parameter.annotation)
    except (NameError, TypeError):
        annotation = parameter.annotation
        if isinstance(annotation, str):
            annotation = handler.__globals__.get(annotation, annotation)
    valid_annotation = annotation is request_model
    if (
        allow_request_supertype
        and isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
    ):
        valid_annotation = issubclass(request_model, annotation)
    if not valid_annotation:
        raise ValueError(
            f"handler for {label} must annotate its request argument as "
            f"{request_model.__name__}"
        )


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _effective_methods(method: str) -> tuple[str, ...]:
    method = method.upper()
    return (method, "HEAD") if method == "GET" else (method,)


@dataclass(frozen=True, slots=True)
class RegisteredOperation:
    definition: OperationDefinition
    handler: Callable | None
    variant_handlers: Mapping[str, Callable]
    enabled_optional_variants: frozenset[str]
    request_model_override: type[BaseModel] | None = None
    overridden: bool = False

    def select_handler(
        self, payload: Mapping[str, Any]
    ) -> tuple[Callable | None, str | None]:
        variant = None
        if self.definition.request is not None:
            variant = self.definition.request.select_variant(
                payload, self.enabled_optional_variants
            )
        return self.variant_handlers.get(variant, self.handler), variant

    def request_model(self, variant: str | None) -> type[BaseModel] | None:
        if self.definition.request is None:
            return None
        if variant is None and self.request_model_override is not None:
            return self.request_model_override
        return self.definition.request.model_for_variant(variant)


@dataclass(frozen=True, slots=True)
class RegisteredExtension:
    activation: ExtensionActivation
    operations: Mapping[str, RegisteredOperation]

    @property
    def definition(self) -> ExtensionDefinition:
        return self.activation.definition

    def to_card(self) -> dict[str, Any]:
        definition = self.definition
        params: dict[str, Any] = {}
        if definition.endpoint is not None:
            params["endpoint"] = definition.endpoint
        methods: dict[str, Any] = {}
        configured_methods = self.activation.wire_params.get("methods", {})
        for operation in self.operations.values():
            spec = operation.definition
            method: dict[str, Any] = {"method": spec.method}
            if definition.endpoint != spec.path:
                method["endpoint"] = spec.path
            configured_method = configured_methods.get(spec.name, {})
            if spec.request is not None and "request" not in configured_method:
                method["request"] = spec.request.to_card(
                    operation.enabled_optional_variants
                )
            if spec.response is not None:
                method["response"] = spec.response.to_card()
            methods[spec.name] = method
        if methods:
            params["methods"] = methods
        params = _deep_merge(params, self.activation.wire_params)
        card = {
            "uri": definition.uri,
            "description": (
                self.activation.description
                if self.activation.description is not None
                else definition.description
            ),
            "params": params,
        }
        if self.activation.required is not None:
            card["required"] = self.activation.required
        return card


class ExtensionRegistry:
    """Resolved extensions, handlers, routes, and generated card entries."""

    def __init__(self, extensions: Iterable[RegisteredExtension]) -> None:
        self.extensions = tuple(extensions)
        self._by_uri = {
            extension.definition.uri: extension for extension in self.extensions
        }
        if len(self._by_uri) != len(self.extensions):
            raise ValueError("duplicate extension URI")

        seen_routes: dict[tuple[str, str], tuple[str, str]] = {}
        for extension in self.extensions:
            for operation in extension.operations.values():
                identity = (extension.definition.uri, operation.definition.name)
                for method in _effective_methods(operation.definition.method):
                    route = (operation.definition.path, method)
                    previous = seen_routes.get(route)
                    if previous is not None:
                        raise ValueError(
                            f"route {route[1]} {route[0]} is shared by "
                            f"{previous} and {identity}"
                        )
                    seen_routes[route] = identity
        self._routes = seen_routes

    def reject_framework_route_collisions(
        self, routes: Mapping[tuple[str, str], str]
    ) -> None:
        """Reject extension operations that would intercept framework routes."""
        for (path, method), owner in routes.items():
            for effective_method in _effective_methods(method):
                operation = self._routes.get((path, effective_method))
                if operation is not None:
                    raise ValueError(
                        f"extension operation {operation} conflicts with {owner} "
                        f"route {effective_method} {path}"
                    )

    def extension(self, uri: str) -> RegisteredExtension | None:
        return self._by_uri.get(uri)

    def card_extensions(self) -> list[dict[str, Any]]:
        return [extension.to_card() for extension in self.extensions]

    def conformance(self) -> dict[str, Any]:
        overrides = []
        for extension in self.extensions:
            for operation in extension.operations.values():
                if operation.overridden:
                    overrides.append(
                        {
                            "uri": extension.definition.uri,
                            "operation": operation.definition.name,
                        }
                    )
        return {"standard_operation_overrides": overrides}


def _discover_bindings(agent: Any) -> list[tuple[HandlerBinding, Callable]]:
    discovered = []
    for name in dir(agent):
        if name.startswith("__"):
            continue
        static_member = inspect.getattr_static(agent, name)
        binding = getattr(static_member, _HANDLER_BINDING, None)
        if binding is not None:
            member = getattr(agent, name)
            discovered.append((binding, member))
    return discovered


def _feature_for_operation(
    definition: ExtensionDefinition, operation_name: str
) -> str | None:
    for feature_name, group in definition.optional_features.items():
        if operation_name in group.operations:
            return feature_name
    return None


def build_registry(
    agent: Any,
    declared: Iterable[ExtensionActivation | ExtensionDefinition] = (),
    *,
    sdk_handlers: Mapping[tuple[str, str], Callable] | None = None,
    request_model_overrides: Mapping[tuple[str, str], type[BaseModel]] | None = None,
) -> ExtensionRegistry:
    """Build one validated registry from declarations and decorated methods."""
    sdk_handlers = sdk_handlers or {}
    request_model_overrides = request_model_overrides or {}
    activations: dict[str, ExtensionActivation] = {}
    for declaration in declared:
        if isinstance(declaration, ExtensionDefinition):
            activation = ExtensionActivation(declaration)
        elif isinstance(declaration, ExtensionActivation):
            activation = declaration
        else:
            raise TypeError(
                "extensions must contain ExtensionDefinition or ExtensionActivation"
            )
        _validate_reserved_definition(activation.definition)
        uri = activation.definition.uri
        if uri in activations:
            raise ValueError(f"extension {uri} is enabled more than once")
        activations[uri] = activation

    operation_handlers: dict[tuple[str, str], tuple[Callable, bool]] = {}
    variant_handlers: dict[tuple[str, str, str], Callable] = {}
    definitions: dict[str, ExtensionDefinition] = {
        uri: activation.definition for uri, activation in activations.items()
    }

    for binding, handler in _discover_bindings(agent):
        _validate_reserved_definition(binding.extension)
        uri = binding.extension.uri
        existing_definition = definitions.get(uri)
        if existing_definition is not None and existing_definition != binding.extension:
            raise ValueError(
                f"conflicting definitions for extension {uri}; define one shared "
                "ExtensionDefinition and bind each operation with @extension(...)"
            )
        definitions[uri] = binding.extension
        # A decorated handler is itself an implementation declaration, so it
        # activates both standard and custom extension definitions. An explicit
        # activation, when present, retains its configuration and metadata.
        activations.setdefault(uri, ExtensionActivation(binding.extension))
        operation = binding.extension.operation(binding.operation)
        if binding.variant is not None:
            if operation.request is None:
                raise ValueError(f"{uri}.{operation.name} has no request variants")
            try:
                variant = operation.request.variant(binding.variant)
            except KeyError as exc:
                raise ValueError(
                    f"unknown request variant {binding.variant!r} for {uri}.{operation.name}"
                ) from exc
            variant_owner = variant.implementation or operation.implementation
            if variant_owner is not ImplementationOwner.RUNTIME:
                raise ValueError(
                    f"{uri}.{operation.name}.{variant.name} is not runtime-backed"
                )
            key = (uri, binding.operation, binding.variant)
            if key in variant_handlers:
                raise ValueError(
                    f"multiple handlers for {uri}.{operation.name}.{binding.variant}"
                )
            variant_handlers[key] = handler
            continue

        key = (uri, binding.operation)
        if key in operation_handlers:
            raise ValueError(f"multiple handlers for {uri}.{operation.name}")
        operation_handlers[key] = (
            handler,
            operation.implementation is ImplementationOwner.SDK,
        )

    registered: list[RegisteredExtension] = []
    for uri, activation in activations.items():
        definition = definitions[uri]
        enabled_features = set(activation.features)

        for feature_name, group in definition.optional_features.items():
            implemented = {
                operation_name
                for operation_name in group.operations
                if (uri, operation_name) in operation_handlers
                or any(
                    variant_uri == uri and variant_operation == operation_name
                    for variant_uri, variant_operation, _ in variant_handlers
                )
            }
            if implemented:
                if group.required_together and implemented != set(group.operations):
                    missing = set(group.operations) - implemented
                    raise ValueError(
                        f"feature {uri}.{feature_name} is incomplete; missing {sorted(missing)}"
                    )
                enabled_features.add(feature_name)

        active_operations: dict[str, OperationDefinition] = dict(
            definition.core_operations
        )
        for feature_name in enabled_features:
            active_operations.update(
                definition.optional_features[feature_name].operations
            )

        resolved_operations: dict[str, RegisteredOperation] = {}
        for operation_key, operation in active_operations.items():
            bound = operation_handlers.get((uri, operation_key))
            sdk_handler = sdk_handlers.get((uri, operation_key))
            if bound is not None:
                handler, overridden = bound
            else:
                handler = sdk_handler
                overridden = False

            enabled_variants = set(activation.variants.get(operation_key, frozenset()))
            per_variant: dict[str, Callable] = {}
            if operation.request is not None and operation.request.variants:
                for variant in operation.request.variants:
                    variant_handler = variant_handlers.get(
                        (uri, operation_key, variant.name)
                    )
                    if variant_handler is not None:
                        per_variant[variant.name] = variant_handler
                        if not variant.support_required:
                            enabled_variants.add(variant.name)

                for variant in operation.request.enabled_variants(enabled_variants):
                    selected = per_variant.get(variant.name, handler)
                    if selected is not None:
                        continue
                    variant_owner = variant.implementation or operation.implementation
                    owner = (
                        "runtime"
                        if variant_owner is ImplementationOwner.RUNTIME
                        else "SDK"
                    )
                    raise ValueError(
                        f"missing {owner} handler for "
                        f"{uri}.{operation.name}.{variant.name}"
                    )

                for variant in operation.request.enabled_variants(enabled_variants):
                    selected = per_variant.get(variant.name, handler)
                    assert selected is not None
                    _validate_handler_signature(
                        selected,
                        variant.model,
                        label=f"{uri}.{operation.name}.{variant.name}",
                        allow_request_supertype=(
                            selected is sdk_handler and not overridden
                        ),
                    )
            elif handler is None:
                owner = (
                    "runtime"
                    if operation.implementation is ImplementationOwner.RUNTIME
                    else "SDK"
                )
                raise ValueError(f"missing {owner} handler for {uri}.{operation.name}")
            else:
                request_model = request_model_overrides.get(
                    (uri, operation_key),
                    operation.request.model if operation.request is not None else None,
                )
                _validate_handler_signature(
                    handler,
                    request_model,
                    label=f"{uri}.{operation.name}",
                    allow_request_supertype=handler is sdk_handler and not overridden,
                )

            resolved_operations[operation_key] = RegisteredOperation(
                definition=operation,
                handler=handler,
                variant_handlers=per_variant,
                enabled_optional_variants=frozenset(enabled_variants),
                request_model_override=request_model_overrides.get(
                    (uri, operation_key)
                ),
                overridden=overridden,
            )

        active_sdk_operations = {
            name
            for name, operation in resolved_operations.items()
            if operation.definition.implementation is ImplementationOwner.SDK
        }
        overridden_sdk_operations = {
            name
            for name in active_sdk_operations
            if resolved_operations[name].overridden
        }
        if (
            overridden_sdk_operations
            and overridden_sdk_operations != active_sdk_operations
        ):
            sdk_backed = active_sdk_operations - overridden_sdk_operations
            raise ValueError(
                f"SDK operations for extension {uri} must be overridden together; "
                f"overridden={sorted(overridden_sdk_operations)}, "
                f"SDK-backed={sorted(sdk_backed)}"
            )

        registered.append(
            RegisteredExtension(activation=activation, operations=resolved_operations)
        )

    return ExtensionRegistry(registered)
