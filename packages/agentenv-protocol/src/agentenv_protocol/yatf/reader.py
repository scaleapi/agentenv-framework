"""Reading a yatf/1 stream: parse a JSON array or JSONL, order it by (turn, seq), fold its events into ATIF-style steps
and pair each tool call with its results."""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Iterable, Mapping

from .models import Event, StepEvent, ToolCall, ToolResult, YatfError, dump_event, parse_event

#: The design's worked examples, checked in as the spec's canonical fixtures.
EXAMPLES = ("coding", "multi_agent", "computer_use")


def _strict_json(constant: str) -> Any:
    raise ValueError(f"{constant} isn't JSON")


def parse_events(data: str | bytes | Iterable[Mapping[str, Any]]) -> list[Event]:
    """A stream's events, from a JSON array (as on the wire), JSONL (as stored) or already-parsed objects. JSON is
    read strictly: UTF-8, with no NaN or Infinity."""
    if isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise YatfError(f"not UTF-8: {exc}") from exc
    if isinstance(data, str):
        text = data.lstrip("﻿")
        if text.lstrip().startswith("["):
            try:
                items = json.loads(text, parse_constant=_strict_json)
            except ValueError as exc:
                raise YatfError(f"not a JSON array: {exc}") from exc
        else:
            items = []
            for number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    items.append(json.loads(line, parse_constant=_strict_json))
                except ValueError as exc:
                    raise YatfError(f"line {number} isn't JSON: {exc}") from exc
    else:
        items = list(data)
    if not isinstance(items, list):
        raise YatfError("a stream is a JSON array or JSONL of events")
    events = []
    for position, item in enumerate(items):
        try:
            events.append(parse_event(item))
        except YatfError as exc:
            raise YatfError(f"event {position}: {exc}") from exc
    return events


def dump_events(events: Iterable[Event], *, jsonl: bool = True) -> str:
    """Events as JSONL, one per line, or as one JSON array, ASCII-escaped as the SDK writes native logs, so any text
    an event holds encodes."""
    dumped = [dump_event(event) for event in events]
    if jsonl:
        return "".join(json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n" for item in dumped)
    return json.dumps(dumped, separators=(",", ":"), allow_nan=False)


def ordered(events: Iterable[Event]) -> list[Event]:
    """Events in stream order, (turn, seq), never by timestamp."""
    return sorted(events, key=lambda event: (event.turn, event.seq))


def load_example(name: str) -> list[Event]:
    """One of the spec's worked examples, by name (``EXAMPLES``)."""
    if name not in EXAMPLES:
        raise YatfError(f"no example {name!r}: there are {', '.join(EXAMPLES)}")
    return parse_events(files(__package__).joinpath("examples", f"{name}.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ToolPair:
    """A tool call and the results that answer it, in stream order; a call with no result has none."""

    call: ToolCall
    results: tuple[ToolResult, ...]


def pair_tool_calls(events: Iterable[Event]) -> list[ToolPair]:
    """Every tool call in the stream with its results. A result whose call isn't in the stream is left out."""
    calls: dict[str, ToolCall] = {}
    results: dict[str, list[ToolResult]] = {}
    for event in ordered(events):
        if isinstance(event, ToolCall):
            calls.setdefault(event.tool_call_id, event)
            results.setdefault(event.tool_call_id, [])
        elif isinstance(event, ToolResult) and event.source_call_id in calls:
            results[event.source_call_id].append(event)
    return [ToolPair(call, tuple(results[call_id])) for call_id, call in calls.items()]


@dataclass(frozen=True)
class FoldedStep:
    """One ATIF-style step: the content events that share a turn and a ``step`` key, its ``step`` event (None for an
    inbound message), and the results that answer its calls, regrouped from wherever they arrived."""

    turn: int
    key: str
    actor: str
    step: StepEvent | None
    events: tuple[Event, ...]
    results: tuple[ToolResult, ...]


def fold_steps(events: Iterable[Event]) -> list[FoldedStep]:
    """The stream's steps, in the order each first appears."""
    events = ordered(events)
    groups: dict[tuple[int, str], dict[str, Any]] = {}
    call_step: dict[str, tuple[int, str]] = {}
    for event in events:
        if event.step is None or isinstance(event, ToolResult):
            continue
        group = groups.setdefault((event.turn, event.step), {"actor": event.actor, "step": None, "events": []})
        if isinstance(event, StepEvent):
            group["step"] = event
            group["actor"] = event.actor
        else:
            group["events"].append(event)
            if isinstance(event, ToolCall):
                call_step.setdefault(event.tool_call_id, (event.turn, event.step))
    results: dict[tuple[int, str], list[ToolResult]] = {}
    for event in events:
        if isinstance(event, ToolResult) and event.source_call_id in call_step:
            results.setdefault(call_step[event.source_call_id], []).append(event)
    return [FoldedStep(turn=turn, key=key, actor=group["actor"], step=group["step"], events=tuple(group["events"]),
                       results=tuple(results.get((turn, key), ())))
            for (turn, key), group in groups.items()]
