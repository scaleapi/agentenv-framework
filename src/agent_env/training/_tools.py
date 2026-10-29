"""Python methods for an env's tools, typed and documented from each tool's JSON Schema the way transformers'
``get_json_schema`` reads a function, which is how TRL finds an environment's tools, and the ``SchemaTool`` each one
reaches the model as. Built without generating source; a method calls its instance's ``_call(name, arguments)``."""

from __future__ import annotations

import inspect
import keyword
import operator
from collections.abc import Callable, Mapping
from functools import reduce
from typing import Any, Literal

_TYPES: dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool, "object": dict,
                          "null": type(None)}
#: The names TRL reads as an environment's own methods rather than its tools.
RESERVED = frozenset({"reset", "get_reward"})


def tool_method(tool: Mapping[str, Any]) -> Callable[..., str]:
    """A method calling ``tool``, typed and documented from its JSON Schema the way transformers' ``get_json_schema``
    reads a function, so TRL shows the model the tool as the env describes it. Built without generating source."""
    name = tool["name"]
    if not _callable_name(name) or name in RESERVED:
        raise ValueError(f"the env's tool {name!r} can't be a TRL tool: TRL calls an environment's tools as its "
                         "methods, so a tool's name must be a public Python identifier other than reset and "
                         "get_reward. Rename the tool in the env")
    schema = tool.get("input_schema") or {}
    required = set(schema.get("required") or [])
    parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    annotations: dict[str, Any] = {}
    described = []
    for argument, spec in (schema.get("properties") or {}).items():
        if not _callable_name(argument):
            raise ValueError(f"the env's tool {name!r} takes an argument {argument!r}, which TRL can't pass: an "
                             "argument's name must be a public Python identifier. Rename it in the env")
        hint = _hint(spec, f"{name}.{argument}")
        default = inspect.Parameter.empty
        if argument not in required:
            hint, default = hint | None, spec.get("default")
        parameters.append(inspect.Parameter(argument, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=hint))
        annotations[argument] = hint
        described.append(f"    {argument}: {_describe(argument, spec)}")
    annotations["return"] = str

    def method(self: Any, **arguments: Any) -> str:
        return self._call(name, arguments)

    method.__name__ = method.__qualname__ = name
    method.__doc__ = _summary(tool) + (
        "\n\nArgs:\n" + "\n".join(described) if described else "")
    method.__annotations__ = annotations
    method.__signature__ = inspect.Signature(parameters, return_annotation=str)
    return method


class SchemaTool(dict):
    """A tool as both its env's JSON Schema, which transformers renders as given, and the callable TRL calls: a method's
    type hints have no nested types. TRL lists the methods on the class and takes the tools from the instance, which
    sets one over each; a tool has no docstring, so ``get_json_schema`` refuses it rather than make up a schema."""

    def __init__(self, tool: Mapping[str, Any], call: Callable[..., str]):
        super().__init__(type="function", function={
            "name": tool["name"], "description": _summary(tool), "parameters": tool["input_schema"]})
        self.__name__, self.__doc__ = tool["name"], None
        self._function = call

    def __call__(self, **arguments: Any) -> str:
        return self._function(**arguments)


def _summary(tool: Mapping[str, Any]) -> str:
    return tool.get("description") or f"Call the {tool['name']} tool."


def _callable_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name) and not name.startswith("_")


def _hint(schema: Mapping[str, Any], where: str) -> Any:
    if options := schema.get("anyOf") or schema.get("oneOf"):
        return _union([_hint(option, where) for option in options])
    if "enum" in schema:
        return Literal[tuple(schema["enum"])]
    kind = schema.get("type")
    if isinstance(kind, list):
        return _union([_hint({**schema, "type": each}, where) for each in kind])
    if kind == "array":
        items = schema.get("items")
        return list[_hint(items, where)] if items else list
    if kind is None:
        return Any
    if kind not in _TYPES:
        raise ValueError(f"{where}: JSON Schema type {kind!r} has no Python type TRL can show the model")
    return _TYPES[kind]


def _union(hints: list[Any]) -> Any:
    return reduce(operator.or_, list(dict.fromkeys(hints)))


def _describe(name: str, spec: Mapping[str, Any]) -> str:
    """One line, since transformers ends an argument's description at the next line that looks like another."""
    text = spec.get("description") or spec.get("title") or name
    return " ".join(f"{text} {_fields(spec)}".split())


def _fields(spec: Mapping[str, Any]) -> str:
    """The fields of an object argument, or of an array's entries, which a Python type hint can't carry."""
    for option in [spec, *(spec.get("anyOf") or spec.get("oneOf") or [])]:
        if option.get("type") == "array" and (items := option.get("items") or {}).get("properties"):
            return "Each entry: " + _field_list(items)
        if option.get("type") == "object" and option.get("properties"):
            return "Fields: " + _field_list(option)
    return ""


def _field_list(schema: Mapping[str, Any]) -> str:
    required = set(schema.get("required") or [])
    parts = []
    for name, spec in schema["properties"].items():
        kind = _kind(spec) + (", required" if name in required else "")
        parts.append(f"{name} ({kind})" + (f": {spec['description']}" if spec.get("description") else ""))
    return "; ".join(parts) + "."


def _kind(spec: Mapping[str, Any]) -> str:
    if options := spec.get("anyOf") or spec.get("oneOf"):
        return " or ".join(_kind(option) for option in options)
    kind = spec.get("type", "any")
    return " or ".join(kind) if isinstance(kind, list) else kind
