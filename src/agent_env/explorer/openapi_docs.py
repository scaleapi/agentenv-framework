"""Docs support: describe *this* install's primitives alongside its API schema.

The Docs surface renders two things — the explorer's OpenAPI operations, and a catalogue of
the agent-env primitives available here (artifact types, env types, task-step types).

The primitive catalogue is built by walking the same registries the runtime dispatches
on, so anything registered through ``[artifacts]`` / ``[envs]`` / ``[task_steps]`` in a
``config.toml`` shows up in the docs automatically.

Each entry carries where the class actually lives (module + line), so a reader can jump
from "this task step exists" to its source without a search.
"""

from __future__ import annotations

import copy
import inspect
import logging
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

PRIMITIVES_VERSION = 1


_PRIMITIVE_TYPE_MAP: dict[Any, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    dict: {"type": "object"},
    list: {"type": "array"},
}


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    """A shallow JSON-Schema sketch of a type annotation for the Docs page; anything it
    cannot map cleanly is reported by name under ``x-python-type``."""
    import typing

    if annotation is inspect.Parameter.empty:
        return {}
    if annotation in _PRIMITIVE_TYPE_MAP:
        return dict(_PRIMITIVE_TYPE_MAP[annotation])

    origin = typing.get_origin(annotation)
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if origin in (list, set, tuple):
        return {"type": "array", "items": _annotation_schema(args[0]) if args else {}}
    if origin is dict:
        return {"type": "object"}
    if origin is not None and args:              # Optional[X] / Union[X, None]
        schema = _annotation_schema(args[0])
        if len(args) == 1:
            schema["nullable"] = True
            return schema

    name = getattr(annotation, "__name__", None) or str(annotation)
    return {"x-python-type": name}


def _resolved_hints(func: Callable) -> dict[str, Any]:
    """Real type objects for ``func``'s annotations (via ``get_type_hints``, which resolves
    the string annotations ``from __future__ import annotations`` produces), or ``{}`` if
    resolution fails."""
    import typing

    try:
        return typing.get_type_hints(func)
    except Exception:
        return {}


def _constructor_schema(cls: type, primitive_type: str) -> dict[str, Any]:
    """Describe ``cls.__init__`` as an object schema: one property per keyword."""
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {"type": "object", "title": cls.__name__}
    hints = _resolved_hints(cls.__init__)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if name in ("self", "args", "kwargs") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        ):
            continue
        prop = _annotation_schema(hints.get(name, param.annotation))
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            try:                                  # only inline JSON-representable defaults
                import json
                json.dumps(param.default)
                prop["default"] = param.default
            except (TypeError, ValueError):
                pass
        properties[name] = prop

    schema: dict[str, Any] = {
        "type": "object",
        "title": cls.__name__,
        "description": _description(cls) or f"Constructor arguments for {primitive_type!r}.",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _component_name(group: str, cls: type) -> str:
    return f"AgentEnv{group[:1].upper()}{group[1:]}_{cls.__name__}"


def _description(cls: type) -> str:
    """First paragraph of the class's own ``__doc__`` (not ``inspect.getdoc``, which would
    inherit a base class's docstring), collapsed to one line."""
    doc = cls.__dict__.get("__doc__") or ""
    first = doc.strip().split("\n\n", 1)[0]
    return " ".join(first.split())


def _source_location(cls: type) -> dict[str, Any]:
    """Where the class is defined. Best-effort: a C or dynamically-built class has none."""
    try:
        module = inspect.getmodule(cls)
        file = inspect.getsourcefile(cls) if module else None
        _, line = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        return {"module": getattr(cls, "__module__", "")}
    return {"module": getattr(cls, "__module__", ""), "file": file, "line": line}


def _aliases_for_class(registry: Mapping[str, type], primitive_type: str, cls: type) -> list[str]:
    """Other keys the same class is registered under (e.g. service/environment spellings)."""
    return sorted(k for k, v in registry.items() if v is cls and k != primitive_type)


def _collect(
    group: str, load: Callable[[], Mapping[str, type]], components: dict[str, Any]
) -> list[dict[str, Any]]:
    """One catalogue group, sorted by type. A registry that cannot be built is skipped."""
    try:
        registry = load()
    except Exception:
        logger.warning("Could not load the %s registry for docs", group, exc_info=True)
        return []

    entries: list[dict[str, Any]] = []
    for primitive_type, cls in sorted(registry.items()):
        if not inspect.isclass(cls):
            continue
        entry: dict[str, Any] = {
            "type": primitive_type,
            "className": cls.__name__,
            "module": cls.__module__,
            "description": _description(cls),
            "source": _source_location(cls),
        }
        component_name = _component_name(group, cls)
        components.setdefault(component_name, _constructor_schema(cls, primitive_type))
        entry["component"] = f"#/components/schemas/{component_name}"
        aliases = _aliases_for_class(registry, primitive_type, cls)
        if aliases:
            entry["aliases"] = aliases
        entries.append(entry)
    return entries


def collect_primitives(components: dict[str, Any] | None = None) -> dict[str, Any]:
    """The artifact / env / task-step catalogue for this install.

    Constructor schemas are written into ``components`` (the OpenAPI
    ``components.schemas`` map) and referenced by each entry, so the Docs page can render
    a primitive's arguments the same way it renders a request body.
    """
    from agent_env.artifact.registry import get_artifact_registry
    from agent_env.env.registry import get_env_registry
    from agent_env.task_step.registry import get_task_step_registry

    components = {} if components is None else components
    return {
        "version": PRIMITIVES_VERSION,
        "artifacts": _collect("artifacts", get_artifact_registry, components),
        "envs": _collect("envs", get_env_registry, components),
        "taskSteps": _collect("taskSteps", get_task_step_registry, components),
    }


def enrich_openapi_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Attach the primitive catalogue + a docs block to a generated OpenAPI schema.

    Deep-copies the input first: the caller passes ``app.openapi()``, whose nested
    ``components.schemas`` dict FastAPI caches — mutating it in place would make
    ``/openapi.json`` (and the Swagger page) grow every time Docs is opened."""
    schema = copy.deepcopy(schema)
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    primitives = collect_primitives(components)
    schema["x-agent-env-primitives"] = primitives
    schema["x-agent-env-docs"] = {
        "source": "live",
        "description": (
            "Served from the running explorer, so this reflects the primitives registered in "
            "this install — including any added through .agentenv/config.toml."
        ),
        "counts": {
            "artifacts": len(primitives["artifacts"]),
            "envs": len(primitives["envs"]),
            "taskSteps": len(primitives["taskSteps"]),
            "operations": sum(
                1
                for ops in (schema.get("paths") or {}).values()
                for method in ops
                if method in {"get", "post", "put", "patch", "delete"}
            ),
        },
    }
    return schema


def _package_version(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return ""


def docs_metadata(schema: dict[str, Any]) -> dict[str, Any]:
    """Provenance for the live served spec: source, timestamp, OpenAPI version, and
    package versions."""
    return {
        "source": "live",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "openapi_version": schema.get("openapi", ""),
        "versions": {
            "agentenv-framework": _package_version("agentenv-framework"),
            "agentenv-protocol": _package_version("agentenv-framework-protocol") or _package_version("agentenv-protocol"),
        },
    }
