"""The tool calls in a model's reply, in the format its chat template teaches it: ``hermes`` (Qwen2.5, Qwen3), a JSON
object of name and arguments in each ``<tool_call>`` block, or ``qwen3_coder`` (Qwen3.5, Qwen3-Coder), a
``<function=name>`` with a ``<parameter=name>`` block per argument, each typed by the tool's JSON Schema."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

FORMATS = ("hermes", "qwen3_coder")

_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
_FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Unparsed:
    """A ``<tool_call>`` block that isn't a call of one of the tools, and what the model is told of it."""

    why: str


def parse(reply: str, tool_format: str, tools: Mapping[str, Mapping[str, Any]]) -> tuple[list[Call | Unparsed], str]:
    """Each ``<tool_call>`` block of ``reply``'s answer in order, as a call of one of ``tools`` (each function by its
    name) or why it isn't one, and the answer's text outside them, which ends a turn. The answer is what follows the
    reply's reasoning, so a call the model only thinks about isn't run."""
    if tool_format not in FORMATS:
        raise ValueError(f"tool_format must be one of {', '.join(FORMATS)}, not {tool_format!r}")
    reading = _hermes if tool_format == "hermes" else _qwen3_coder
    answer = _answer(reply)
    calls = [_checked(reading(block.strip(), tools), tools) for block in _BLOCK.findall(answer)]
    return calls, _BLOCK.sub("", answer).strip()


def _answer(reply: str) -> str:
    """What follows the reply's first ``</think>`` outside a tool call, which closes reasoning the reply or its chat
    template's prompt opened, up to a ``<think>`` outside a tool call that opens more: a tool's argument may hold
    either tag as text."""
    calls = [match.span() for match in _BLOCK.finditer(reply)]

    def outside(tag: str, start: int) -> int:
        found = reply.find(tag, start)
        while found != -1 and any(begin <= found < end for begin, end in calls):
            found = reply.find(tag, found + 1)
        return found

    closed = outside("</think>", 0)
    start = 0 if closed == -1 else closed + len("</think>")
    opened = outside("<think>", start)
    return reply[start:] if opened == -1 else reply[start:opened]


def _hermes(block: str, tools: Mapping[str, Mapping[str, Any]]) -> Call | Unparsed:
    try:
        call = json.loads(block)
    except json.JSONDecodeError as e:
        return Unparsed(f"the tool call isn't JSON ({e}): write it as "
                        '{"name": <tool name>, "arguments": <arguments as a JSON object>}')
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return Unparsed('the tool call names no tool: write it as {"name": <tool name>, "arguments": <arguments>}')
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    if not isinstance(arguments, dict):
        return Unparsed(f"the arguments of {call['name']} aren't a JSON object")
    return Call(call["name"], arguments)


def _qwen3_coder(block: str, tools: Mapping[str, Mapping[str, Any]]) -> Call | Unparsed:
    if not (function := _FUNCTION.search(block)):
        return Unparsed("the tool call names no tool: write it as <function=tool name> with a "
                        "<parameter=argument name> block per argument")
    name, body = function[1], function[2]
    properties = ((tools.get(name) or {}).get("parameters") or {}).get("properties") or {}
    return Call(name, {argument: _typed(_trimmed(value), properties.get(argument) or {})
                       for argument, value in _PARAMETER.findall(body)})


def _checked(call: Call | Unparsed, tools: Mapping[str, Mapping[str, Any]]) -> Call | Unparsed:
    if isinstance(call, Call) and call.name not in tools:
        return Unparsed(f"there's no tool {call.name!r}; the tools are {', '.join(sorted(tools))}")
    return call


def _trimmed(value: str) -> str:
    """A parameter's value without the one newline the format puts on each side of it."""
    return value.removeprefix("\n").removesuffix("\n")


def _typed(value: str, schema: Mapping[str, Any]) -> Any:
    """``value`` as the type its schema names, the first of a union that it parses as; as written when it parses as
    none, or its schema names no type and it isn't JSON."""
    kinds = schema.get("type")
    for kind in kinds if isinstance(kinds, list) else [kinds]:
        if kind == "string":
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        if kind is None or _is(parsed, kind):
            return parsed
    return value


def _is(value: Any, kind: str) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, {"boolean": bool, "array": list, "object": dict, "null": type(None)}.get(kind, object))
