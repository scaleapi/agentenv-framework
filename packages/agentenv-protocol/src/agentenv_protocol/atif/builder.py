"""The typed step builder: what an adapter's handlers write ATIF steps through, and the document it keeps for one A2A
context. It holds each trajectory's open step and closes it when the next one starts or the task ends, puts every
result in its call's step, writes usage in ATIF's terms, embeds sub-agents, hands media bytes to a sink and checks
every value is JSON. A live reader is served closed steps only, so nothing it has seen ever changes."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence, get_args

from pydantic import ValidationError

from .adapter import AdapterSpec
from .models.content import AudioMediaType, ContentPart, ImageSource
from .profile import (
    COST_SOURCES,
    ERROR_KINDS,
    PROFILE_KEY,
    PROFILE_VERSION,
    RAW_ARGUMENTS,
    RESULT_STATUSES,
    SCHEMA_VERSION,
    TASK_STATES,
    VISIBILITIES,
    AtifError,
)

#: Media types ATIF has a part type for; any other medium is written as a text note naming its file.
IMAGE_TYPES = frozenset(get_args(ImageSource.model_fields["media_type"].annotation))
AUDIO_TYPES = frozenset(get_args(AudioMediaType))
_EXTENSIONS = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp", "image/heic": ".heic",
    "image/svg+xml": ".svg", "audio/wav": ".wav", "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/aac": ".aac",
    "audio/ogg": ".ogg", "audio/flac": ".flac", "audio/webm": ".webm", "audio/aiff": ".aiff", "video/mp4": ".mp4",
    "video/webm": ".webm", "video/quicktime": ".mov", "application/pdf": ".pdf", "text/plain": ".txt",
    "text/csv": ".csv", "application/json": ".json", "application/zip": ".zip",
}
_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?([Zz]|[+-]\d{2}:?\d{2})$")
_JOIN = "\n\n"
_UNSET: Any = object()


def timestamp(value: str | datetime | None) -> str | None:
    """An RFC 3339 time as the builder writes it: UTC, ``Z``, and 3 or 6 fractional digits (or none), since ATIF's
    Step validator on Python 3.10 reads no other precision."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise AtifError(f"{value!r} has no time zone")
        moment = value.astimezone(timezone.utc)
        fraction = f".{moment.microsecond:06d}" if moment.microsecond else ""
        return moment.strftime("%Y-%m-%dT%H:%M:%S") + fraction + "Z"
    match = _TIME.match(value.strip()) if isinstance(value, str) else None
    if not match:
        raise AtifError(f"{value!r} isn't an RFC 3339 time with a zone")
    date, hours, minutes, seconds, digits, zone = match.groups()
    zone = "+00:00" if zone in ("Z", "z") else (zone if ":" in zone else f"{zone[:3]}:{zone[3:]}")
    micro = int((digits + "000000")[:6]) if digits else 0
    try:
        moment = datetime.fromisoformat(f"{date}T{hours}:{minutes}:{seconds}.{micro:06d}{zone}").astimezone(timezone.utc)
    except ValueError as exc:
        raise AtifError(f"{value!r}: {exc}") from exc
    text = moment.strftime("%Y-%m-%dT%H:%M:%S")
    if digits:
        text += f".{moment.microsecond // 1000:03d}" if len(digits) <= 3 else f".{moment.microsecond:06d}"
    return text + "Z"


def _json(value: Any, what: str) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AtifError(f"{what} isn't JSON: {exc}") from exc
    return json.loads(json.dumps(value))


def _size(count: int) -> str:
    if count < 1000:
        return f"{count} B"
    return f"{count / 1000:.1f} KB" if count < 1_000_000 else f"{count / 1_000_000:.1f} MB"


@dataclass(frozen=True)
class Media:
    """A media file stored at ``path`` (``media/<sha256>.<ext>``, relative to the trajectory). As content it is an
    image or audio part when ATIF has the type, and otherwise a plain text note naming the file."""

    path: str
    media_type: str
    info: Mapping[str, Any]

    @property
    def typed(self) -> bool:
        return self.media_type in IMAGE_TYPES or self.media_type in AUDIO_TYPES

    def part(self) -> dict[str, Any]:
        if self.media_type in IMAGE_TYPES:
            return {"type": "image", "source": {"media_type": self.media_type, "path": self.path}}
        if self.media_type in AUDIO_TYPES:
            source: dict[str, Any] = {"media_type": self.media_type, "path": self.path}
            if self.info.get("duration_ms") is not None:
                source["duration_sec"] = self.info["duration_ms"] / 1000
            return {"type": "audio", "source": source}
        major = self.media_type.split("/", 1)[0]
        kind = major if major in ("image", "audio", "video") else (
            "document" if self.media_type == "application/pdf" else "file")
        details = [self.path, _size(self.info["bytes"])]
        if self.info.get("duration_ms") is not None:
            details.append(f"{round(self.info['duration_ms'] / 1000)} s")
        if self.info.get("pages") is not None:
            details.append(f"{self.info['pages']} pages")
        return {"type": "text", "text": f"[{kind}: {', '.join(details)}]"}

    def profile(self) -> dict[str, Any]:
        info = dict(self.info)
        if not self.typed:
            info["media_type"] = self.media_type
        return info


@dataclass(frozen=True, kw_only=True)
class TurnContext:
    """What the SDK knows of an A2A task when it starts: its ids and turn index, the inbound message's content, and
    the agent as its card names it. The builder writes the task's user step from it."""

    index: int
    context_id: str
    task_id: str
    message_id: str | None = None
    inbound: Sequence[Any] = ()
    agent_name: str | None = None
    agent_version: str | None = None
    agent_model: str | None = None
    started_at: str | datetime | None = None


@dataclass(eq=False)
class _Step:
    source: str
    key: str | None
    task: int
    index: int
    attempt: int
    parts: list[dict[str, Any]] = field(default_factory=list)
    media: dict[str, Any] = field(default_factory=dict)
    reasoning: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metrics_profile: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    pending: set[str] = field(default_factory=set)
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    llm_call_count: int | None = None
    is_copied_context: bool | None = None
    time: str | None = None
    step_time: str | None = None
    closed: bool = False
    ended: bool = False


@dataclass(eq=False)
class _Trajectory:
    actor: str | None
    agent: dict[str, Any]
    steps: list[_Step] = field(default_factory=list)
    open: _Step | None = None


class Document:
    """One A2A context's ATIF document, built a task at a time. Not thread-safe: records are fed one at a time."""

    def __init__(self, spec: AdapterSpec, sink: Callable[[str, bytes], None] | None = None) -> None:
        self.spec = spec
        self.media: dict[str, bytes] = {}
        self._sink = sink
        self._root: _Trajectory | None = None
        self._subs: dict[str, _Trajectory] = {}
        self._call_home: dict[str, tuple[_Trajectory, _Step]] = {}
        self._spawned: dict[str, str] = {}
        self._unmatched: Counter[str] = Counter()
        self._task_costs: list[tuple[float, str] | None] = []
        self._context: TurnContext | None = None
        self._task = -1
        self._attempt = 1

    # Tasks

    @property
    def task_open(self) -> bool:
        return self._context is not None

    def start_task(self, context: TurnContext) -> AtifTurn:
        if self._context is not None:
            raise AtifError(f"task {self._context.task_id} is still open")
        if self._root is None:
            self._root = _Trajectory(None, _agent(context.agent_name, context.agent_version, context.agent_model))
            self._context_id = context.context_id
        elif context.context_id != self._context_id:
            raise AtifError(f"context {context.context_id} isn't this document's {self._context_id}")
        self._context, self._task, self._attempt = context, self._task + 1, 1
        self._task_costs.append(None)
        step = self._start(self._root, "user", None)
        step.time = timestamp(context.started_at)
        self._add_parts(step.parts, step.media, context.inbound, None)
        step.profile.update({key: value for key, value in (
            ("context_id", context.context_id), ("task_id", context.task_id), ("message_id", context.message_id),
            ("turn", context.index)) if value is not None})
        return AtifTurn(self, context)

    def end_task(self, *, state: str = "completed", final: Sequence[Any] | None = None,
                 ended_at: str | datetime | None = None) -> None:
        if self._context is None:
            raise AtifError("no task is open")
        if state not in TASK_STATES:
            raise AtifError(f"state {state!r} isn't one of {', '.join(TASK_STATES)}")
        root = self._root
        assert root is not None
        if final:
            parts: list[dict[str, Any]] = []
            media: dict[str, Any] = {}
            self._add_parts(parts, media, final, None)
            last = next((s for s in reversed(root.steps) if s.task == self._task and s.source == "agent"), None)
            if last is None or _message(last.parts) != _message(parts):
                step = self._start(root, "agent", None)
                step.parts, step.media, step.llm_call_count = parts, media, 0
                step.time = timestamp(ended_at)
        root.steps[-1].profile["state"] = state
        for trajectory in self._trajs():
            for step in trajectory.steps:
                if step.task == self._task:
                    step.closed = step.ended = True
            trajectory.open = None
        self._context = None

    # What handlers call, through AtifTurn

    def step_for(self, key: str, agent: str | None) -> _Step:
        trajectory = self._traj(agent)
        step = trajectory.open
        if step is None or step.key != key or step.source != "agent":
            step = self._start(trajectory, "agent", key)
        return step

    def open_step(self, agent: str | None) -> _Step:
        step = self._traj(agent).open
        if step is None:
            raise AtifError("no step is open: name the step this belongs to")
        return step

    def call(self, step: _Step, call_id: str, name: str, arguments: Any) -> None:
        if not isinstance(call_id, str) or not call_id:
            raise AtifError("tool_call_id is a non-empty string")
        if call_id in self._call_home:
            raise AtifError(f"tool_call_id {call_id!r} is already in this document")
        if isinstance(arguments, Mapping):
            typed, raw = _json(dict(arguments), f"{name}'s arguments"), _UNSET
        else:
            raw = _json(arguments, f"{name}'s arguments")
            typed = {} if arguments is None else ({"input": arguments} if isinstance(arguments, str)
                                                  else {"value": raw})
        step.calls.append({"tool_call_id": call_id, "function_name": name, "arguments": typed, "raw": raw})
        step.pending.add(call_id)
        self._call_home[call_id] = (self._owner(step), step)

    def result(self, call_id: str, content: Sequence[Any], status: str, exit_code: int | None,
               at: str | None) -> None:
        if call_id not in self._call_home:
            raise AtifError(f"a result for {call_id!r}, which no call in this document made")
        if status not in RESULT_STATUSES:
            raise AtifError(f"status {status!r} isn't one of {', '.join(RESULT_STATUSES)}")
        trajectory, home = self._call_home[call_id]
        parts: list[dict[str, Any]] = []
        media: dict[str, Any] = {}
        self._add_parts(parts, media, content, None)
        profile: dict[str, Any] = {}
        if status != "ok":
            profile["status"] = status
        if exit_code is not None:
            profile["exit_code"] = _json(exit_code, "exit_code")
        if home.closed:
            opened = trajectory.open
            profile["arrived"] = {key: value for key, value in (
                ("at", at), ("step_id", opened.index + 1 if opened is not None else None)) if value is not None}
        if media:
            profile["media"] = media
        result: dict[str, Any] = {"source_call_id": call_id, "parts": parts, "profile": profile}
        if call_id in self._spawned:
            result["subagent"] = self._spawned[call_id]
        home.results.append(result)
        home.pending.discard(call_id)

    def subagent(self, agent_id: str, spawned_by: str, name: str | None, version: str | None,
                 model_name: str | None) -> None:
        if agent_id in self._subs:
            raise AtifError(f"sub-agent {agent_id!r} is already declared")
        self._subs[agent_id] = _Trajectory(agent_id, _agent(name or agent_id, version, model_name))
        self._spawned[spawned_by] = agent_id

    def store(self, data: bytes, media_type: str, info: Mapping[str, Any]) -> Media:
        if not isinstance(data, (bytes, bytearray)):
            raise AtifError("media bytes are bytes")
        digest = hashlib.sha256(data).hexdigest()
        extension = _EXTENSIONS.get(media_type) or mimetypes.guess_extension(media_type, strict=False) or ".bin"
        path = f"media/{digest}{extension}"
        if path not in self.media:
            self.media[path] = bytes(data)
            if self._sink is not None:
                self._sink(path, bytes(data))
        details = {"sha256": digest, "bytes": len(data), **{k: v for k, v in info.items() if v is not None}}
        return Media(path, media_type, _json(details, "media details"))

    def count_unmatched(self, record: Any) -> None:
        kind = record.get("type") if isinstance(record, Mapping) else None
        self._unmatched[kind if isinstance(kind, str) else type(record).__name__] += 1

    def task_cost(self, cost_usd: float, source: str) -> None:
        if source not in COST_SOURCES:
            raise AtifError(f"cost_source {source!r} isn't one of {', '.join(COST_SOURCES)}")
        self._task_costs[self._task] = (_json(cost_usd, "cost_usd"), source)

    def set_attempt(self, number: int) -> None:
        if not isinstance(number, int) or number < 1:
            raise AtifError("an attempt number is 1 or more")
        self._attempt = number

    def add_parts(self, step: _Step, content: Sequence[Any] | None, text: str | None) -> None:
        self._add_parts(step.parts, step.media, content, text)

    # Reading the document

    def document(self) -> dict[str, Any] | None:
        """The whole document so far, or None before its first step. A fresh copy each time."""
        return self._render(closed_only=False)

    def page(self) -> dict[str, Any] | None:
        """What a live reader is served: each trajectory's closed steps, up to the first one still waiting on a
        result, or None while there are none."""
        return self._render(closed_only=True)

    # Internals

    def _trajs(self) -> list[_Trajectory]:
        return ([self._root] if self._root is not None else []) + list(self._subs.values())

    def _traj(self, agent: str | None) -> _Trajectory:
        if self._root is None or self._context is None:
            raise AtifError("no task is open")
        if agent is None:
            return self._root
        if agent not in self._subs:
            raise AtifError(f"sub-agent {agent!r} isn't declared: call turn.subagent first")
        return self._subs[agent]

    def _owner(self, step: _Step) -> _Trajectory:
        return next(t for t in self._trajs() if any(s is step for s in t.steps))

    def _start(self, trajectory: _Trajectory, source: str, key: str | None) -> _Step:
        if trajectory.open is not None:
            trajectory.open.closed = True
        step = _Step(source=source, key=key, task=self._task, index=len(trajectory.steps), attempt=self._attempt)
        if self._attempt > 1:
            step.profile["attempt"] = self._attempt
        trajectory.steps.append(step)
        trajectory.open = step
        return step

    def _add_parts(self, parts: list[dict[str, Any]], media: dict[str, Any], content: Iterable[Any] | None,
                   text: str | None) -> None:
        if text is not None:
            if content is not None:
                raise AtifError("give content or text, not both")
            content = [text]
        for item in content or ():
            if isinstance(item, Media):
                parts.append(item.part())
                media[item.path] = item.profile()
                continue
            if isinstance(item, str):
                part: Any = {"type": "text", "text": item}
            elif isinstance(item, Mapping):
                part = _json(dict(item), "a content part")
            else:
                raise AtifError(f"content is text, a Media or an ATIF part, not {type(item).__name__}")
            try:
                ContentPart.model_validate(part)
            except ValidationError as exc:
                raise AtifError(f"content part: {exc}") from exc
            parts.append(part)

    def _servable(self, trajectory: _Trajectory) -> int:
        count = 0
        for step in trajectory.steps:
            if not step.closed or (step.pending and not step.ended):
                break
            count += 1
        return count

    def _render(self, *, closed_only: bool) -> dict[str, Any] | None:
        root = self._root
        if root is None:
            return None
        root_count = self._servable(root) if closed_only else len(root.steps)
        if root_count == 0:
            return None
        doc: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "session_id": self._context_id,
                               "agent": dict(root.agent),
                               "steps": [_render_step(s) for s in root.steps[:root_count]]}
        subs = []
        for trajectory in self._subs.values():
            count = self._servable(trajectory) if closed_only else len(trajectory.steps)
            if count:
                subs.append({"schema_version": SCHEMA_VERSION, "trajectory_id": trajectory.actor,
                             "agent": dict(trajectory.agent),
                             "steps": [_render_step(s) for s in trajectory.steps[:count]]})
        if subs:
            doc["subagent_trajectories"] = subs
        if self._context is None:
            doc["final_metrics"] = self._final_metrics(root)
        profile: dict[str, Any] = {"version": PROFILE_VERSION, "native_format": self.spec.native_format,
                                   "adapter": {"name": self.spec.name, "version": self.spec.version}}
        if self._unmatched:
            profile["unmatched"] = dict(sorted(self._unmatched.items()))
        doc["extra"] = {PROFILE_KEY: profile}
        return doc

    def _final_metrics(self, root: _Trajectory) -> dict[str, Any]:
        steps = [s for t in self._trajs() for s in t.steps
                 if s.source == "agent" and s.llm_call_count != 0 and (s.metrics or s.metrics_profile)]
        totals: dict[str, Any] = {}
        for key, total in (("prompt_tokens", "total_prompt_tokens"), ("completion_tokens", "total_completion_tokens"),
                           ("cached_tokens", "total_cached_tokens")):
            if steps and all(key in s.metrics for s in steps):
                totals[total] = sum(s.metrics[key] for s in steps)
        profile: dict[str, Any] = {}
        costs = self._task_costs
        if costs and all(cost is not None for cost in costs):
            totals["total_cost_usd"] = round(sum(cost[0] for cost in costs), 10)
            sources = {cost[1] for cost in costs}
            if len(sources) == 1:
                profile["cost_source"] = sources.pop()
        elif steps and all("cost_usd" in s.metrics for s in steps):
            totals["total_cost_usd"] = round(sum(s.metrics["cost_usd"] for s in steps), 10)
            sources = {s.metrics_profile.get("cost_source") for s in steps}
            if len(sources) == 1 and None not in sources:
                profile["cost_source"] = sources.pop()
        totals["total_steps"] = len(root.steps)
        if steps and all("cache_write_tokens" in s.metrics_profile for s in steps):
            profile["total_cache_write_tokens"] = sum(s.metrics_profile["cache_write_tokens"] for s in steps)
        if profile:
            totals["extra"] = {PROFILE_KEY: profile}
        return totals


def _agent(name: str | None, version: str | None, model: str | None) -> dict[str, Any]:
    agent = {"name": name or "unknown", "version": version or "unknown"}
    if model:
        agent["model_name"] = model
    return agent


def _message(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """A step's message: one string when every part is text, else the parts."""
    if all(part["type"] == "text" for part in parts):
        return _JOIN.join(part["text"] for part in parts)
    return [dict(part) for part in parts]


def _render_step(step: _Step) -> dict[str, Any]:
    agentic = step.source == "agent"
    out: dict[str, Any] = {"step_id": step.index + 1}
    when = step.step_time or step.time
    if when:
        out["timestamp"] = when
    out["source"] = step.source
    if agentic and step.model_name:
        out["model_name"] = step.model_name
    if agentic and step.reasoning_effort is not None:
        out["reasoning_effort"] = step.reasoning_effort
    out["message"] = _message(step.parts)
    texts = [block["text"] for block in step.reasoning if block.get("text")]
    if agentic and texts and step.llm_call_count != 0:
        out["reasoning_content"] = _JOIN.join(texts)
    if agentic and step.calls:
        calls = []
        for call in step.calls:
            rendered = {k: call[k] for k in ("tool_call_id", "function_name", "arguments")}
            if call["raw"] is not _UNSET:
                rendered["extra"] = {RAW_ARGUMENTS: call["raw"]}
            calls.append(rendered)
        out["tool_calls"] = calls
    if step.results:
        results = []
        for result in step.results:
            rendered = {"source_call_id": result["source_call_id"], "content": _message(result["parts"])}
            if "subagent" in result:
                rendered["subagent_trajectory_ref"] = [{"trajectory_id": result["subagent"]}]
            if result["profile"]:
                rendered["extra"] = {PROFILE_KEY: dict(result["profile"])}
            results.append(rendered)
        out["observation"] = {"results": results}
    if agentic and step.llm_call_count != 0 and (step.metrics or step.metrics_profile):
        metrics = dict(step.metrics)
        if step.metrics_profile:
            metrics["extra"] = {PROFILE_KEY: dict(step.metrics_profile)}
        out["metrics"] = metrics
    if agentic and step.llm_call_count is not None:
        out["llm_call_count"] = step.llm_call_count
    if step.is_copied_context is not None:
        out["is_copied_context"] = step.is_copied_context
    profile = dict(step.profile)
    if step.media:
        profile["media"] = dict(step.media)
    blocks = step.reasoning
    if blocks and (len(blocks) > 1 or any(b.get("signature") or b.get("data") or b["visibility"] != "raw"
                                          for b in blocks)):
        profile["reasoning"] = [{key: value for key, value in (
            ("chars", len(b.get("text") or "")), ("visibility", b["visibility"]), ("signature", b.get("signature")),
            ("data", b.get("data"))) if value is not None} for b in blocks]
    if profile:
        out["extra"] = {PROFILE_KEY: profile}
    return json.loads(json.dumps(out))


class AtifTurn:
    """What a handler writes one task's steps through. ``step`` keys group the content of one model response (its
    message id): content under a new key starts a step and closes the open one. ``agent`` names a declared sub-agent,
    whose steps go in its own embedded trajectory."""

    def __init__(self, document: Document, context: TurnContext) -> None:
        self._doc = document
        self.context = context

    @property
    def spec(self) -> AdapterSpec:
        return self._doc.spec

    def message(self, *, step: str, text: str | None = None, content: Sequence[Any] | None = None,
                time: str | datetime | None = None, agent: str | None = None) -> None:
        """Assistant content in step ``step``: text, Media from ``media``, or ATIF parts."""
        target = self._doc.step_for(step, agent)
        self._doc.add_parts(target, content, text)
        self._touch(target, time)

    def reasoning(self, *, step: str, text: str | None = None, visibility: str = "raw", signature: str | None = None,
                  data: str | None = None, time: str | datetime | None = None, agent: str | None = None) -> None:
        """One reasoning block: its text (None when withheld), its visibility, and the provider's signature or a
        redacted block's ciphertext, kept byte for byte."""
        if visibility not in VISIBILITIES:
            raise AtifError(f"visibility {visibility!r} isn't one of {', '.join(VISIBILITIES)}")
        for name, value in (("text", text), ("signature", signature), ("data", data)):
            if value is not None and not isinstance(value, str):
                raise AtifError(f"reasoning {name} is a string")
        target = self._doc.step_for(step, agent)
        target.reasoning.append({"text": text, "visibility": visibility, "signature": signature, "data": data})
        self._touch(target, time)

    def tool_call(self, *, step: str, tool_call_id: str, function_name: str, arguments: Any,
                  time: str | datetime | None = None, agent: str | None = None) -> None:
        """A call in step ``step``. Arguments that aren't a JSON object are wrapped as Harbor wraps them (a string as
        ``{"input": ...}``, None as ``{}``, any other value as ``{"value": ...}``) and kept whole in ``raw_arguments``."""
        target = self._doc.step_for(step, agent)
        self._doc.call(target, tool_call_id, function_name, arguments)
        self._touch(target, time)

    def tool_result(self, *, source_call_id: str, text: str | None = None, content: Sequence[Any] | None = None,
                    status: str = "ok", exit_code: int | None = None, time: str | datetime | None = None) -> None:
        """A result, written in its call's step. One that comes after that step closed keeps when it arrived."""
        if (text is None) == (content is None):
            raise AtifError("give a result's content or its text")
        self._doc.result(source_call_id, [text] if text is not None else list(content or ()), status, exit_code,
                         timestamp(time))

    def step(self, *, key: str, model_name: str | None = None, llm_call_count: int | None = 1,
             reasoning_effort: str | float | None = None, input_tokens: int | None = None,
             output_tokens: int | None = None, cache_read_tokens: int | None = None,
             cache_write_tokens: int | None = None, cost_usd: float | None = None, cost_source: str | None = None,
             time: str | datetime | None = None, is_copied_context: bool | None = None,
             agent: str | None = None) -> None:
        """What one model response cost and produced. Token counts are the native's; ATIF's ``prompt_tokens``
        includes cache reads and writes, so the adapter's ``input_includes_cache`` decides the sum. A cost is
        the harness's or the provider's (``cost_source``), never one agentenv computed."""
        target = self._doc.step_for(key, agent)
        if model_name is not None:
            target.model_name = model_name
        if reasoning_effort is not None:
            target.reasoning_effort = reasoning_effort
        target.llm_call_count = llm_call_count
        if is_copied_context is not None:
            target.is_copied_context = is_copied_context
        counts = (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
        if any(value is not None for value in counts) or cost_usd is not None:
            if llm_call_count == 0:
                raise AtifError("a step with no model call (llm_call_count 0) has no usage or cost")
            for value in counts:
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                    raise AtifError(f"token counts are non-negative integers, not {value!r}")
        if input_tokens is not None or cache_read_tokens is not None or cache_write_tokens is not None:
            includes = self.spec.input_includes_cache
            if includes is None:
                raise AtifError("declare input_includes_cache on @atif_adapter to write prompt tokens")
            prompt = (input_tokens or 0) + (0 if includes else (cache_read_tokens or 0) + (cache_write_tokens or 0))
            target.metrics["prompt_tokens"] = prompt
            if cache_read_tokens is not None:
                target.metrics["cached_tokens"] = cache_read_tokens
            if cache_write_tokens is not None:
                target.metrics_profile["cache_write_tokens"] = cache_write_tokens
        if output_tokens is not None:
            target.metrics["completion_tokens"] = output_tokens
        if cost_usd is not None:
            if cost_source not in COST_SOURCES:
                raise AtifError(f"a cost names its cost_source: one of {', '.join(COST_SOURCES)}")
            target.metrics["cost_usd"] = _json(cost_usd, "cost_usd")
            target.metrics_profile["cost_source"] = cost_source
        if time is not None:
            target.step_time = timestamp(time)

    def task_cost(self, cost_usd: float, *, source: str) -> None:
        """The task's total cost, as the harness or provider reported it, when its steps carry none."""
        self._doc.task_cost(cost_usd, source)

    def error(self, *, message: str, kind: str = "agent_error", code: str | None = None, step: str | None = None,
              agent: str | None = None) -> None:
        """A step-level error: a refused or failed model call, or a harness error. On step ``step``, else the open
        step; a later error on the same step replaces it."""
        if kind not in ERROR_KINDS:
            raise AtifError(f"kind {kind!r} isn't one of {', '.join(ERROR_KINDS)}")
        target = self._doc.step_for(step, agent) if step is not None else self._doc.open_step(agent)
        target.profile["error"] = {k: v for k, v in (("kind", kind), ("message", message), ("code", code))
                                   if v is not None}

    def attempt(self, number: int) -> None:
        """A harness retry: steps from here on carry attempt ``number``."""
        self._doc.set_attempt(number)

    def subagent(self, agent_id: str, *, spawned_by: str, name: str | None = None, version: str | None = None,
                 model_name: str | None = None) -> None:
        """Declare a sub-agent the call ``spawned_by`` started. Its steps (written with ``agent=agent_id``) form an
        embedded trajectory, and the spawning call's result points at it."""
        self._doc.subagent(agent_id, spawned_by, name, version, model_name)

    def media(self, data: bytes, media_type: str, *, width: int | None = None, height: int | None = None,
              duration_ms: int | None = None, pages: int | None = None, name: str | None = None) -> Media:
        """Store media bytes (through the sink) and return the Media to put in content."""
        return self._doc.store(data, media_type, {"width": width, "height": height, "duration_ms": duration_ms,
                                                  "pages": pages, "name": name})

    def unmatched(self, record: Any) -> None:
        """Count a native record nothing handled, by its type, in the root profile."""
        self._doc.count_unmatched(record)

    def _touch(self, step: _Step, time: str | datetime | None) -> None:
        if step.time is None and time is not None:
            step.time = timestamp(time)
