"""Typed Action/State trigger engine (urn:agentenv:triggers/v1) hosted on the gateway:
post-call detection on watched-role traffic, async fail-open firing, additive fail-loud registration."""
from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import math
import random
import re
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from mcp.types import CallToolResult, TextContent

from .clock import ClockError, _advance, _iso, _parse_duration, _parse_rfc3339
from .constants import AGENT_ENV_ROLE_META_KEY, DEFAULT_ROLE, TRIGGER_IN_FLIGHT_STATUSES, TRIGGER_STATUSES

logger = logging.getLogger(__name__)

_PREDICATE_KEYS = ("regex", "equals", "exists")
_ACTION_TYPES = ("nl", "tool", "permission")
_TERMINAL_A2A_STATES = ("completed", "failed", "canceled", "rejected")
# Bounds the per-evaluation catch-up burst; the backlog drains on later evaluations.
_MAX_CATCHUP = 1000
# Head + tail of the firing log kept in memory; the driver fires for the gateway's whole life.
_EVENTS_HEAD = 1000
_EVENTS_TAIL = 9000
# Schedule keys state() echoes; excludes the unbounded `check`/`where`.
_WHEN_SUMMARY_KEYS = ("type", "tool", "repeat", "at", "after", "offset", "every", "count", "until", "seed")
# Placeholder roots reading the provoking call; a bare name is a check-pipeline binding.
_CTX_PREFIXES = ("args", "result")
# Where a barrier waits; a further wait point is a new value here, not a new top-level field.
_BARRIER_AT = ("provoking_call",)
_BARRIER_KEYS = ("at", "timeout_seconds")
_DROP = object()  # an unresolved `${x?}` used as a whole value: drop the key / element
# An allowlist, not a size bound: identifier/metadata keys only, since bodies and addresses are PII.
_ECHO_KEYS = frozenset({
    "action", "createdTime", "fileName", "kind", "mimeType", "modifiedTime", "name", "role",
    "title", "type",
})
_ECHO_ID_RE = re.compile(r"(?:^id|Id|_id)$")  # id, fileId, documentId, email_id, …
_ECHO_VALUE_MAX = 200  # chars; an allowlisted scalar longer than this is a payload wearing an id's name
_ECHO_DIGEST_CHARS = 16

# Firing-log event vocabulary (the GET /triggers/state contract); _emit rejects anything not here.
_EVENT_KINDS = frozenset({
    "added", "removed",               # registration lifecycle
    "detected",                       # a trigger's condition matched
    "anchored",                       # an event-anchored time-trigger's mark was stamped
    "reanchored",                     # a clock re-arm dropped a time-trigger's old-axis mark/backlog
    "action_ok", "action_failed",     # one fire-action within a trigger
    "verify_ok", "verify_failed",     # typed acceptance of an action
    "fired", "failed", "eval_error",  # per trigger; `failed` re-arms an action/state trigger, retires a time one
})
# Trigger-level failures; the per-action kinds are details of one of these, so this counts once per failure.
_FAILURE_KINDS = ("failed", "eval_error")
_VAR_RE = re.compile(r"\$\{([\w.\[\]*]+\??)\}")
# The strict pattern is both finder and resolver, so a shape it cannot parse would be sent verbatim.
_VAR_LOOSE_RE = re.compile(r"\$\{[^{}]*\}?")


class TriggerError(ValueError):
    """Registration validation failure (surfaced as HTTP 400)."""


class TemplateError(LookupError):
    """A `${args.*}` / `${result.*}` placeholder had no value in the provoking call (fires as action_failed)."""


def _time_mark_spec(value: Any) -> tuple[str, Any]:
    """('abs', datetime) for an RFC3339 stamp, or ('rel', timedelta) for a t0-relative duration."""
    if isinstance(value, str) and value.startswith("P"):
        return ("rel", _parse_duration(value))
    return ("abs", _parse_rfc3339(value))


def _dur_or_400(value: Any, where: str) -> timedelta:
    try:
        return _parse_duration(value)
    except ClockError as e:
        raise TriggerError(f"{where}: {e}")


def _mark_or_400(value: Any, where: str) -> tuple[str, Any]:
    try:
        return _time_mark_spec(value)
    except ClockError as e:
        raise TriggerError(f"{where}: {e}")


def _make_rng(spec: dict) -> random.Random | None:
    """A per-trigger seeded RNG for a stochastic time-trigger, else None."""
    when = spec["when"]
    if when["type"] == "time" and isinstance(when.get("every"), dict):
        return random.Random(when["seed"])
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _eval_predicate(pred: dict, value: Any) -> bool:
    if "regex" in pred:
        return re.search(pred["regex"], _as_text(value)) is not None
    if "equals" in pred:
        return value == pred["equals"]
    non_empty = value is not None and value != "" and value != []
    return non_empty == pred["exists"]


def _apply_path(data: Any, path: str) -> Any:
    """Dot path with [n] indices and a trailing [*] meaning 'the list itself'."""
    current = data
    for segment in path.split("."):
        m = re.fullmatch(r"(\w+)((?:\[(?:\d+|\*)\])*)", segment)
        if m is None:
            raise TriggerError(f"invalid extract path segment {segment!r}")
        key, indices = m.group(1), m.group(2)
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
        for idx in re.findall(r"\[(\d+|\*)\]", indices):
            if idx == "*":
                continue
            if not isinstance(current, list) or int(idx) >= len(current):
                return None
            current = current[int(idx)]
    return current


def _apply_extract(extract: Any, data: Any, raw_text: str) -> Any:
    if extract is None:
        return raw_text
    if isinstance(extract, str):
        return _apply_path(data, extract)
    items = _apply_path(data, extract["path"])
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        match = extract.get("match", {})
        if all(_eval_predicate(pred, item.get(field)) for field, pred in match.items()):
            return item.get(extract["take"]) if extract.get("take") else item
    return None


def _result_data(result: Any) -> Any:
    """A CallToolResult's text parsed as JSON (dict / list / scalar), the raw text when it is not JSON, or None."""
    content = getattr(result, "content", None)
    if not content:
        return None
    text = "\n".join(c.text for c in content if getattr(c, "type", None) == "text")
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def _ctx_path(name: str) -> tuple[str | None, str]:
    """The `args`/`result` root a placeholder or `where` key addresses and the path beneath it;
    `(None, "")` for a check-pipeline binding name. Pass a name with any trailing `?` removed —
    the optional marker belongs to placeholders, not to `where` keys."""
    head, _, path = name.partition(".")
    return (head, path) if head in _CTX_PREFIXES else (None, "")


def _ctx_lookup(name: str, bindings: dict, ctx: dict | None) -> tuple[bool, Any]:
    """(found, value) for a placeholder name: check-pipeline bindings win; `args` / `result` roots (with an
    optional dotted path) read the provoking call's arguments / parsed result."""
    name = name.rstrip("?")
    if name in bindings:
        return True, bindings[name]
    root_name, path = _ctx_path(name)
    if root_name is None or ctx is None:
        return False, None
    root = ctx.get(root_name)
    if not path:
        return root is not None, root
    value = _apply_path(root, path) if isinstance(root, dict) else None
    return value is not None, value


def _resolve(name: str, literal: str, bindings: dict, ctx: dict | None) -> tuple[bool, Any]:
    """(found, value) for one placeholder. An unresolved required `args.*` / `result.*` is fail-loud —
    the target tool must never receive the literal `${...}` text."""
    found, value = _ctx_lookup(name, bindings, ctx)
    if not found and not name.endswith("?") and _ctx_path(name.rstrip("?"))[0] is not None:
        raise TemplateError(f"unresolved placeholder {literal}")
    return found, value


def _is_ctx_attempt(token: str) -> bool:
    """Whether placeholder-shaped text is reaching for the provoking call, whitespace aside.

    For nl instructions, which are never templated: `${result.x}` there can only be an author
    expecting substitution, while `${HOME}` is ordinary prose. All whitespace is removed before the
    root is read, so `${result .x}` is recognised as the attempt it is rather than mistaken for a
    shell variable -- the strict pattern would not resolve it either, so it must not register."""
    compact = re.sub(r"\s+", "", token)
    return any(compact.startswith(f"${{{root}.") for root in _CTX_PREFIXES)


def _template(args: Any, bindings: dict, ctx: dict | None = None) -> Any:
    """Substitute `${...}` placeholders recursively through dicts and lists. A string that is exactly one
    placeholder resolves to the raw value; an embedded one stringifies. Unresolved: a required
    `args.*` / `result.*` raises, a bare name stays literal, an optional whole value is dropped."""
    def sub(value: Any) -> Any:
        # Dropping the key rather than sending null is what lets the target tool apply its own default.
        if isinstance(value, dict):
            return {k: r for k, v in value.items() if (r := sub(v)) is not _DROP}
        if isinstance(value, list):
            return [r for v in value if (r := sub(v)) is not _DROP]
        if not isinstance(value, str):
            return value
        whole = _VAR_RE.fullmatch(value)
        if whole is not None:
            name = whole.group(1)
            found, resolved = _resolve(name, value, bindings, ctx)
            if not found:
                return _DROP if name.endswith("?") else value
            return resolved if _ctx_path(name.rstrip("?"))[0] else str(resolved)

        def repl(m: re.Match) -> str:
            found, resolved = _resolve(m.group(1), m.group(0), bindings, ctx)
            if not found:
                return "" if m.group(1).endswith("?") else m.group(0)
            return resolved if isinstance(resolved, str) else json.dumps(resolved, default=str)
        return _VAR_RE.sub(repl, value)

    out = sub(args)
    return args if out is _DROP else out  # a whole-value drop has nowhere to go at the top level


def _where_value(ctx: dict, field: str) -> Any:
    """The value a `where` key names: `result.<path>` / `args.<path>` via _apply_path, a bare key is a
    top-level argument (the v1 behaviour)."""
    root_name, path = _ctx_path(field)
    if root_name is not None and path:
        root = ctx.get(root_name)
        return _apply_path(root, path) if isinstance(root, dict) else None
    args = ctx.get("args")
    return args.get(field) if isinstance(args, dict) else None


def _echoable(key: str) -> bool:
    return key in _ECHO_KEYS or _ECHO_ID_RE.search(key) is not None


def _shape(value: Any) -> dict:
    """What a redacted value was, without any of what it said."""
    shape: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, (str, bytes, list, dict)):
        shape["len"] = len(value)
    return {"redacted": shape}


def _echo_args(value: Any, key: str | None = None) -> Any:
    """Trajectory-safe projection of a mirror's resolved args.

    Allowlisted identifier/metadata scalars pass through so a grader can see *which* object the
    mirror wrote; every other value becomes a `{"redacted": {...}}` shape descriptor. Structure is
    preserved either way, so the arg *names* a mirror sent stay reviewable — it is only the values
    that are withheld. Keys are matched at every depth, so a nested `{"file": {"content": ...}}`
    is redacted just like a top-level one.
    """
    if isinstance(value, dict):
        return {k: _echo_args(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_echo_args(v, key) for v in value]
    if key is not None and _echoable(key) and isinstance(value, (str, int, float, bool, type(None))):
        # A bool is an int subclass; both are unconditionally safe. Only a str can hide a payload.
        if isinstance(value, str) and len(value) > _ECHO_VALUE_MAX:
            return _shape(value)
        return value
    return _shape(value)


def _strings(value: Any) -> list[str]:
    """Every string anywhere in an args structure, so the two placeholder scans share one walk."""
    if isinstance(value, dict):
        return [t for v in value.values() for t in _strings(v)]
    if isinstance(value, list):
        return [t for v in value for t in _strings(v)]
    return [value] if isinstance(value, str) else []


def _placeholders(value: Any) -> list[str]:
    return [n for text in _strings(value) for n in _VAR_RE.findall(text)]


def _inert_placeholders(value: Any) -> list[str]:
    """Placeholder-shaped text the templater will not act on, so the tool would receive it verbatim."""
    return [m for text in _strings(value) for m in _VAR_LOOSE_RE.findall(text)
            if _VAR_RE.fullmatch(m) is None]


def _validate_path(path: str, where: str) -> None:
    for segment in path.split("."):
        if re.fullmatch(r"\w+((?:\[(?:\d+|\*)\])*)", segment) is None:
            raise TriggerError(f"{where}: invalid path segment {segment!r} in {path!r}")


def _validate_placeholders(value: Any, where: str, allow_ctx: bool) -> None:
    """Fail-loud at registration: `${args.*}` / `${result.*}` only on action triggers (they alone carry a
    provoking call), any other placeholder must be a bare check-binding name. A shape the templater
    cannot parse is rejected first, or it would reach the tool as literal text."""
    inert = _inert_placeholders(value)
    if inert:
        raise TriggerError(f"{where}: {inert[0]!r} is not a placeholder the engine can resolve; "
                           f"expected ${{args.<path>}}, ${{result.<path>}} or ${{<check binding>}}")
    for name in _placeholders(value):
        name = name.rstrip("?")
        root_name, path = _ctx_path(name)
        if root_name is None:
            if re.fullmatch(r"\w+", name) is None:
                raise TriggerError(f"{where}: placeholder ${{{name}}} must be a check binding name or an args./result. path")
        elif not allow_ctx:
            raise TriggerError(f"{where}: ${{{name}}} needs a provoking tool call; only action triggers carry args/result")
        elif path:
            _validate_path(path, where)
        elif "." in name:
            raise TriggerError(f"{where}: placeholder ${{{name}}} has an empty path")


def _check_tools(check: dict) -> list[str]:
    return [step["tool"] for step in check["steps"]]


def _validate_predicate(pred: Any, where: str) -> None:
    if not isinstance(pred, dict) or sum(k in pred for k in _PREDICATE_KEYS) != 1:
        raise TriggerError(f"{where}: predicate must have exactly one of {_PREDICATE_KEYS}")
    if "regex" in pred:
        try:
            re.compile(pred["regex"])
        except re.error as e:
            raise TriggerError(f"{where}: invalid regex: {e}")
    if "exists" in pred and not isinstance(pred["exists"], bool):
        raise TriggerError(f"{where}: exists must be a boolean")


def _validate_check(check: Any, where: str) -> dict:
    if not isinstance(check, dict):
        raise TriggerError(f"{where}: check must be an object")
    steps = check.get("steps", [check] if "tool" in check else None)
    if not isinstance(steps, list) or not steps:
        raise TriggerError(f"{where}: check needs 'steps' or a single-step tool/predicate form")
    for i, step in enumerate(steps):
        w = f"{where}.steps[{i}]"
        if not isinstance(step.get("tool"), str) or not step["tool"]:
            raise TriggerError(f"{w}: tool must be a non-empty string")
        if not isinstance(step.get("args", {}), dict):
            raise TriggerError(f"{w}: args must be an object")
        extract = step.get("extract")
        if extract is not None and not isinstance(extract, str):
            if not isinstance(extract, dict) or "path" not in extract:
                raise TriggerError(f"{w}: extract must be a dot-path string or {{path, match, take}}")
            for field, pred in extract.get("match", {}).items():
                _validate_predicate(pred, f"{w}.extract.match.{field}")
        if "predicate" in step:
            _validate_predicate(step["predicate"], f"{w}.predicate")
    if "predicate" not in steps[-1]:
        raise TriggerError(f"{where}: the last check step must carry a predicate")
    return {"steps": steps}


class TriggerEngine:
    def __init__(self, gateway):
        self._gw = gateway
        self._executor: dict | None = None
        self._executor_role: str | None = None
        self._watch_roles: set[str] = set()
        self._config_set = False
        self._triggers: dict[str, dict] = {}
        self._events: list[dict] = []                                  # head of the firing log
        self._events_tail: deque[dict] = deque(maxlen=_EVENTS_TAIL)    # most-recent window
        self._seq = 0
        self._tasks: set[asyncio.Task] = set()
        self._readonly_tools: set[str] | None = None
        self._has_time_triggers = False
        self._driver_task: asyncio.Task | None = None
        self._driver_interval = 1.0  # real seconds between autonomous time-trigger polls
        self._time_trigger_gate = asyncio.Event()  # parks the driver while no time-trigger is armed

    def register(self, body: Any) -> dict:
        """Additively add each trigger by id (idempotent upsert; conflicting id or bad shape -> 400); watch_roles/executor are set once."""
        if not isinstance(body, dict):
            raise TriggerError("body must be a JSON object")
        watch_roles = self._resolve_watch_roles(body)
        executor = self._resolve_executor(body, watch_roles)
        triggers = body.get("triggers")
        if not isinstance(triggers, list):
            raise TriggerError("triggers must be a list")
        known = self._known_tools()
        batch_order: list[str] = []
        to_add: list[tuple[str, dict]] = []
        for raw in triggers:
            if not isinstance(raw, dict):
                raise TriggerError("every trigger needs a non-empty string id")
            spec = copy.deepcopy(raw)  # validation normalizes in place; never mutate the caller's dict
            tid = spec.get("id")
            if tid is None:
                tid = f"trg_{uuid.uuid4().hex[:12]}"
                spec["id"] = tid
            if not isinstance(tid, str) or not tid:
                raise TriggerError("every trigger needs a non-empty string id")
            if tid in batch_order:
                raise TriggerError(f"trigger '{tid}': duplicate id")
            batch_order.append(tid)
            spec = self._normalize_trigger(spec, executor)
            if known is not None:
                self._check_tools_known(spec, known)
            existing = self._triggers.get(tid)
            if existing is not None:
                if existing["spec"] != spec:
                    raise TriggerError(f"trigger '{tid}': already registered with a different spec — remove it first or use a new id")
                continue  # identical re-add: no-op (a fired trigger stays fired)
            to_add.append((tid, spec))
        self._readonly_tools = None
        self._watch_roles = watch_roles
        self._executor = executor
        self._executor_role = executor["role"] if executor else None
        self._config_set = True
        for tid, spec in to_add:
            self._triggers[tid] = {"spec": spec, "status": "armed", "detected_at": None,
                                   "fired_at": None, "evaluating": False,
                                   "next_mark": None, "resolved": False, "fire_count": 0,
                                   "failure_count": 0, "last_failure_at": None,
                                   "rng": _make_rng(spec), "mark_gen": None,
                                   "pending": deque(), "task": None}
            self._emit("added", tid)
        self._recompute_time_flag()
        return {"ok": True, "added": batch_order, "all": list(self._triggers)}

    def _known_tools(self) -> set[str] | None:
        """Every tool name the gateway can route to, or None before discovery (then unknown tools surface at fire time)."""
        server_tools = getattr(self._gw, "_server_tools", None)
        if not server_tools or not getattr(self._gw, "_tools_discovered", True):
            return None
        known = {tool.name for tools in server_tools.values() for tool in tools}
        mcp = getattr(self._gw, "_mcp", None)
        if mcp is not None:
            known.update(mcp._tool_manager._tools)
        return known

    @staticmethod
    def _check_tools_known(spec: dict, known: set[str]) -> None:
        tid = spec["id"]
        when = spec["when"]
        named: list[tuple[str, str]] = []
        if when["type"] == "action":
            named.append((when["tool"], "when.tool"))
        elif when["type"] == "state":
            named.extend((t, "when.check") for t in _check_tools(when["check"]))
        for i, action in enumerate(spec["actions"]):
            if action["type"] == "tool":
                named.append((action["tool"], f"actions[{i}].tool"))
            if "verify" in action:
                named.extend((t, f"actions[{i}].verify") for t in _check_tools(action["verify"]))
        for tool, where in named:
            if tool not in known:
                raise TriggerError(f"trigger '{tid}': {where} names unknown tool {tool!r}")

    def remove(self, body: Any) -> dict:
        """Disarm + delete triggers by id (idempotent: unknown ids are ignored). Removed ids may be re-added."""
        if not isinstance(body, dict):
            raise TriggerError("body must be a JSON object")
        ids = body.get("ids")
        if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
            raise TriggerError("ids must be a list of non-empty strings")
        removed = []
        for tid in ids:
            if self._triggers.pop(tid, None) is not None:
                removed.append(tid)
                self._emit("removed", tid)
        self._recompute_time_flag()
        return {"ok": True, "removed": removed}

    def clear(self) -> dict:
        """Full reset — triggers, config, and the event log (setup/testing only; never mid-task)."""
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        self._triggers = {}
        self._events = []
        self._events_tail.clear()
        self._seq = 0
        self._watch_roles = set()
        self._executor = None
        self._executor_role = None
        self._config_set = False
        self._readonly_tools = None
        self._recompute_time_flag()
        return {"ok": True}

    def _recompute_time_flag(self) -> None:
        # "firing" counts: a recurrence mid-fire returns to "armed" and still needs the driver.
        self._has_time_triggers = any(
            t["spec"]["when"]["type"] == "time" and t["status"] in ("armed", "firing")
            for t in self._triggers.values())
        if self._has_time_triggers:
            self._time_trigger_gate.set()
        else:
            self._time_trigger_gate.clear()

    def start_driver(self) -> None:
        """Start the background poller (idempotent; requires a running loop). The gateway's process lifespan starts
        it once at startup."""
        if self._driver_task is None or self._driver_task.done():
            self._driver_task = asyncio.get_running_loop().create_task(self._drive_time())

    async def _drive_time(self) -> None:
        """Poll armed time-triggers every _driver_interval real seconds; parks while none are armed."""
        while True:
            try:
                await self._time_trigger_gate.wait()  # park until a time-trigger is armed
                await asyncio.sleep(self._driver_interval)
                if self._has_time_triggers and self._gw._clock.armed:
                    self._eval_time_triggers()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("clock time-driver tick failed")

    async def stop_driver(self) -> None:
        """Cancel the background poller at process shutdown (idempotent)."""
        task, self._driver_task = self._driver_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _resolve_watch_roles(self, body: dict) -> set[str]:
        raw = body.get("watch_roles")
        if raw is None:
            return self._watch_roles if self._config_set else {"default"}
        if not isinstance(raw, list) or not all(isinstance(r, str) and r for r in raw):
            raise TriggerError("watch_roles must be a list of non-empty strings")
        new = set(raw)
        if self._config_set and new != self._watch_roles:
            raise TriggerError(f"watch_roles {sorted(new)} conflicts with the already-registered {sorted(self._watch_roles)} — clear to change it")
        return new

    def _resolve_executor(self, body: dict, watch_roles: set[str]) -> dict | None:
        raw = body.get("executor")
        if raw is None:
            return self._executor
        if not isinstance(raw, dict) or not isinstance(raw.get("a2a_url"), str) or not raw["a2a_url"]:
            raise TriggerError("executor.a2a_url must be a non-empty string")
        role = raw.get("role")
        if not isinstance(role, str) or not role:
            raise TriggerError("executor.role must be a non-empty string (the deployed executor's AgentEnv-Role, kept out of watch_roles so its own tool calls never fire triggers)")
        if role in watch_roles:
            raise TriggerError(f"executor.role {role!r} must not be in watch_roles {sorted(watch_roles)} — the executor's own tool calls would fire triggers (cascade)")
        normalized = {"a2a_url": raw["a2a_url"].rstrip("/"),
                      "timeout_seconds": float(raw.get("timeout_seconds", 120)),
                      "role": role}
        if self._executor is not None and self._executor != normalized:
            raise TriggerError("executor conflicts with the already-registered executor — clear to change it")
        return normalized

    @staticmethod
    def _normalize_barrier(spec: dict, tid: str, has_ctx: bool) -> None:
        """Validate `barrier` in place. Absent stays absent and the timeout is resolved at wait time,
        so an idempotent re-add still matches even after the gateway default moves."""
        barrier = spec.get("barrier")
        if barrier is None:
            spec.pop("barrier", None)
            return
        w = f"trigger '{tid}': barrier"
        if not isinstance(barrier, dict):
            raise TriggerError(f"{w} must be an object, e.g. {{\"at\": \"provoking_call\"}}")
        unknown = sorted(set(barrier) - set(_BARRIER_KEYS))
        if unknown:
            raise TriggerError(f"{w} has unknown key(s) {unknown}; expected {list(_BARRIER_KEYS)}")
        at = barrier.get("at")
        if at not in _BARRIER_AT:
            raise TriggerError(f"{w}.at must be one of {list(_BARRIER_AT)} (got {at!r})")
        if at == "provoking_call" and not has_ctx:
            raise TriggerError(f"{w}.at 'provoking_call' is only valid on action triggers — they alone have a "
                               "provoking call to hold")
        timeout = barrier.get("timeout_seconds")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or not math.isfinite(timeout) or timeout <= 0):
            raise TriggerError(f"{w}.timeout_seconds must be a positive, finite number (got {timeout!r})")

    def _normalize_trigger(self, spec: dict, executor: dict | None) -> dict:
        """Validate + normalize a trigger spec in place (fail-loud); returns it, so pass a copy."""
        tid = spec["id"]
        when = spec.get("when")
        if not isinstance(when, dict) or when.get("type") not in ("action", "state", "time"):
            raise TriggerError(f"trigger '{tid}': when.type must be 'action', 'state', or 'time'")
        if when["type"] == "action":
            self._normalize_action_when(when, tid)
        elif when["type"] == "state":
            if "repeat" in when:
                raise TriggerError(f"trigger '{tid}': when.repeat is only valid on action triggers")
            when["check"] = _validate_check(when.get("check"), f"trigger '{tid}': when.check")
            for j, step in enumerate(when["check"]["steps"]):
                _validate_placeholders(step.get("args", {}), f"trigger '{tid}': when.check.steps[{j}].args", allow_ctx=False)
        else:
            if "repeat" in when:
                raise TriggerError(f"trigger '{tid}': when.repeat is only valid on action triggers (time triggers recur via 'every')")
            self._normalize_time_when(when, tid)
        has_ctx = when["type"] == "action"
        self._normalize_barrier(spec, tid, has_ctx)
        actions = spec.get("actions")
        if not isinstance(actions, list):
            raise TriggerError(f"trigger '{tid}': actions must be a list")
        for i, action in enumerate(actions):
            self._normalize_action(action, f"trigger '{tid}': actions[{i}]", has_ctx, executor)
        return spec

    @staticmethod
    def _normalize_action_when(when: dict, tid: str) -> None:
        """Validate a when.type=='action' spec: the provoking tool, its `where` predicates, `repeat`."""
        if not isinstance(when.get("tool"), str) or not when["tool"]:
            raise TriggerError(f"trigger '{tid}': when.tool must be a non-empty string")
        where = when.get("where", {})
        if not isinstance(where, dict):
            raise TriggerError(f"trigger '{tid}': when.where must be an object")
        for field, pred in where.items():
            _validate_predicate(pred, f"trigger '{tid}': where.{field}")
            _, path = _ctx_path(field)
            if path:
                _validate_path(path, f"trigger '{tid}': where.{field}")
        if "repeat" in when and not isinstance(when["repeat"], bool):
            raise TriggerError(f"trigger '{tid}': when.repeat must be a boolean")

    @staticmethod
    def _normalize_action(action: Any, w: str, has_ctx: bool, executor: dict | None) -> None:
        """Validate one fire-action in place, plus its optional `verify` check and `as` role."""
        atype = action.get("type") if isinstance(action, dict) else None
        if atype not in _ACTION_TYPES:
            raise TriggerError(f"{w}: type must be one of {_ACTION_TYPES}")
        if "as" in action and atype != "tool":
            raise TriggerError(f"{w}: 'as' is only valid on tool actions")
        if atype == "nl":
            if not isinstance(action.get("instruction"), str) or not action["instruction"]:
                raise TriggerError(f"{w}: instruction must be a non-empty string")
            # nl is never templated; only context-rooted shapes reject, since prose carries `${HOME}`.
            for token in _VAR_LOOSE_RE.findall(action["instruction"]):
                if _is_ctx_attempt(token):
                    raise TriggerError(f"{w}: instruction contains {token!r}, but nl instructions are "
                                       f"not templated; read the provoking call from a tool action's args")
            if executor is None:
                raise TriggerError(f"{w}: nl actions require an executor")
        elif atype == "tool":
            if not isinstance(action.get("tool"), str) or not action["tool"]:
                raise TriggerError(f"{w}: tool must be a non-empty string")
            if not isinstance(action.get("args", {}), dict):
                raise TriggerError(f"{w}: args must be an object")
            _validate_placeholders(action.get("args", {}), f"{w}.args", allow_ctx=has_ctx)
            acting = action.get("as")
            if acting is not None:
                # the default role means "not forwarded" to the child, so naming it can only be a mistake
                if not isinstance(acting, str) or not acting.strip() or acting == DEFAULT_ROLE:
                    raise TriggerError(f"{w}: 'as' must be a non-empty AgentEnv-Role other than {DEFAULT_ROLE!r}")
                if _VAR_LOOSE_RE.search(acting):  # never run through _template, so a placeholder would be sent verbatim
                    raise TriggerError(f"{w}: 'as' is not templated; name the role literally")
        else:
            if action.get("action") not in ("enable", "disable"):
                raise TriggerError(f"{w}: action must be 'enable' or 'disable'")
            if not isinstance(action.get("role"), str) or not action["role"]:
                raise TriggerError(f"{w}: role must be a non-empty string")
            tools = action.get("tools")
            if tools != "*" and (not isinstance(tools, list) or not all(isinstance(t, str) for t in tools)):
                raise TriggerError(f"{w}: tools must be a list of strings or '*'")
        if "verify" in action:
            action["verify"] = _validate_check(action["verify"], f"{w}.verify")
            for j, step in enumerate(action["verify"]["steps"]):
                _validate_placeholders(step.get("args", {}), f"{w}.verify.steps[{j}].args", allow_ctx=has_ctx)

    def _normalize_time_when(self, when: dict, tid: str) -> None:
        """Validate a when.type=='time' spec fail-loud, leaving it JSON-clean."""
        w = f"trigger '{tid}'"
        has_at, has_after, has_every = "at" in when, "after" in when, "every" in when
        if not (has_at or has_after or has_every):
            raise TriggerError(f"{w}: when.type=='time' needs at least one of 'at', 'after', or 'every'")
        if has_at and has_after:
            raise TriggerError(f"{w}: when may have at most one of 'at' / 'after'")
        if has_after:
            if not isinstance(when["after"], str) or not when["after"]:
                raise TriggerError(f"{w}: when.after must be a non-empty trigger id")
            if not isinstance(when.get("offset"), str) or not when["offset"]:
                raise TriggerError(f"{w}: when.after requires a non-empty 'offset' duration")
            _dur_or_400(when["offset"], f"{w}: offset")
        elif "offset" in when:
            raise TriggerError(f"{w}: 'offset' is only valid with 'after'")
        if has_at:
            _mark_or_400(when["at"], f"{w}: at")
        is_stochastic = has_every and isinstance(when["every"], dict)
        if has_every:
            every = when["every"]
            if isinstance(every, str):
                if _dur_or_400(every, f"{w}: every") <= timedelta(0):
                    raise TriggerError(f"{w}: every must be a positive duration")
            elif isinstance(every, dict):
                if every.get("dist") != "exp":
                    raise TriggerError(f"{w}: every.dist must be 'exp'")
                if not isinstance(every.get("mean"), str) or _dur_or_400(every["mean"], f"{w}: every.mean") <= timedelta(0):
                    raise TriggerError(f"{w}: every.mean must be a positive ISO-8601 duration")
                if not isinstance(when.get("seed"), int) or isinstance(when.get("seed"), bool):
                    raise TriggerError(f"{w}: a stochastic 'every' requires an integer 'seed'")
            else:
                raise TriggerError(f"{w}: every must be a duration string or {{dist, mean}}")
        if "seed" in when and not is_stochastic:
            raise TriggerError(f"{w}: 'seed' is only valid with a stochastic 'every'")
        for term in ("until", "count"):
            if term in when and not has_every:
                raise TriggerError(f"{w}: '{term}' requires a recurring 'every'")
        if "count" in when and (not isinstance(when["count"], int) or isinstance(when["count"], bool) or when["count"] <= 0):
            raise TriggerError(f"{w}: count must be a positive integer")
        if "until" in when:
            _mark_or_400(when["until"], f"{w}: until")

    def state(self) -> dict:
        events = self._events + list(self._events_tail)
        for t in self._triggers.values():
            assert t["status"] in TRIGGER_STATUSES, f"unknown trigger status {t['status']!r}"
        return {
            "config": {"watch_roles": sorted(self._watch_roles),
                       "executor_configured": self._executor is not None},
            "triggers": [
                {"id": tid, "type": t["spec"]["when"]["type"],
                 "when": {k: t["spec"]["when"][k] for k in _WHEN_SUMMARY_KEYS if k in t["spec"]["when"]},
                 "status": t["status"], "detected_at": t["detected_at"],
                 "fired_at": t["fired_at"], "fire_count": t["fire_count"],
                 "failure_count": t["failure_count"], "last_failure_at": t["last_failure_at"],
                 "barrier": t["spec"].get("barrier"),
                 "pending": len(t["pending"]),
                 "next_mark": _iso(t["next_mark"]) if t["next_mark"] else None}
                for tid, t in self._triggers.items()
            ],
            "events": events,
            "events_dropped": max(0, self._seq - len(events)),
        }

    def on_tool_call(self, role: str, name: str, arguments: dict,
                     result: Any) -> tuple[list[asyncio.Task], float | None]:
        """Synchronous post-call detection hook (invoked from the gateway's role-filter wrapper).

        Returns the fire tasks of triggers whose barrier waits at `provoking_call`, and how long to
        hold the call for: the widest of their timeouts, counting an omitted one as the gateway
        default so a neighbour's tighter ask cannot cut it short.
        """
        pending: list[asyncio.Task] = []
        timeout_s: float | None = None
        if role == self._executor_role:  # executor's own traffic is never watched (prevents cascades)
            return pending, timeout_s
        if role not in self._watch_roles or not self._triggers:
            return pending, timeout_s
        # Before the isError guard: an errored call is still a clock tick.
        if self._has_time_triggers:
            self._eval_time_triggers()
        # Handled refusals arrive flagged too: BaseService.flag_handled_errors sets isError at the wire.
        if getattr(result, "isError", False):
            return pending, timeout_s
        data = _result_data(result)
        ctx = {"args": arguments, "result": data}
        provoking = {"tool": name, "role": role}
        for tid, trig in self._triggers.items():
            when = trig["spec"]["when"]
            if when["type"] != "action" or when["tool"] != name:
                continue
            if trig["status"] != "armed" and not (when.get("repeat") and trig["status"] in TRIGGER_IN_FLIGHT_STATUSES):
                continue
            if not all(_eval_predicate(pred, _where_value(ctx, field))
                       for field, pred in when.get("where", {}).items()):
                continue
            task = self._mark_detected(tid, trig, provoking, ctx)
            barrier = trig["spec"].get("barrier") or {}
            if task is not None and barrier.get("at") == "provoking_call":
                pending.append(task)
                asked = barrier.get("timeout_seconds") or self._gw.TRIGGER_BARRIER_TIMEOUT_S
                timeout_s = asked if timeout_s is None else max(timeout_s, asked)
        if not self._is_readonly(name):
            for tid, trig in self._triggers.items():
                if trig["status"] == "armed" and trig["spec"]["when"]["type"] == "state" and not trig["evaluating"]:
                    trig["evaluating"] = True
                    self._schedule(self._eval_state(tid, trig, provoking))
        return pending, timeout_s

    def _mark_detected(self, tid: str, trig: dict, provoking: dict, ctx: dict | None = None) -> asyncio.Task | None:
        ctx = ctx if ctx is not None else {}
        if trig["status"] in TRIGGER_IN_FLIGHT_STATUSES:
            # Mid-fire: queue it; the running fire drains the queue before it settles.
            trig["pending"].append(ctx)
            trig["status"] = "queued"
            self._emit("detected", tid, provoking=provoking, queued=len(trig["pending"]))
            return trig["task"]
        trig["status"] = "firing"
        trig["detected_at"] = _now()
        self._emit("detected", tid, provoking=provoking)
        trig["task"] = self._schedule(self._fire(tid, trig, ctx))
        trig["task"].set_name(tid)
        return trig["task"]

    def _eval_time_triggers(self) -> None:
        """Poll armed time-triggers against the virtual clock and schedule the due ones."""
        clock = self._gw._clock
        now = clock.now()
        if now is None:  # clock unarmed -> nothing can be due yet
            return
        gen = clock.generation
        for tid, trig in list(self._triggers.items()):
            when = trig["spec"]["when"]
            if when["type"] != "time" or trig["status"] != "armed":
                continue
            # Re-anchor to the new t0 across a clock re-arm, per-trigger so a mid-fire one isn't skipped.
            # An event-anchored mark is an absolute instant its fire-once anchor can't re-derive -> kept.
            if "after" not in when and trig["resolved"] and trig["mark_gen"] != gen:
                # The old mark (and any backlog behind it) is on a timeline the new clock never reached.
                self._emit("reanchored", tid, generation=gen,
                           dropped_mark=_iso(trig["next_mark"]) if trig["next_mark"] else None)
                trig["resolved"] = False
                trig["next_mark"] = None
                trig["rng"] = _make_rng(trig["spec"])
            if not trig["resolved"]:
                self._resolve_first_mark(trig, now)
                if trig["resolved"]:
                    trig["mark_gen"] = gen
            mark = trig["next_mark"]
            if mark is None or now < mark:
                continue
            marks, capped, terminated = self._advance_marks(trig, now)
            if marks:
                # The cause is time passing, not whichever path observed it.
                self._mark_time_detected(tid, trig, marks, capped, terminated, {"source": "clock"})
            elif terminated:  # retired without firing (e.g. `until` already passed)
                trig["status"] = "fired"
                trig["next_mark"] = None
                self._recompute_time_flag()

    def _resolve_first_mark(self, trig: dict, now: datetime) -> None:
        """Stamp the first mark for a non-anchored time-trigger; event-anchored ones wait for _resolve_dependents."""
        when = trig["spec"]["when"]
        if "after" in when:
            return
        t0 = self._gw._clock.t0()
        if "at" in when:
            kind, val = _time_mark_spec(when["at"])
            start = val if kind == "abs" else _advance(t0, val.total_seconds())
        else:  # recurring with no explicit start: first fire is one interval after t0
            start = _advance(t0, self._next_interval_seconds(trig))
        trig["next_mark"] = start
        trig["resolved"] = True

    def _resolve_dependents(self, anchor_tid: str) -> None:
        """Stamp the concrete mark (virtual_now + offset) of every time-trigger anchored on `anchor_tid`."""
        # Safe despite _fire_time recomputing the flag first: an unresolved `after` dependent is itself
        # an armed time trigger, so the flag stays set for exactly as long as one can be stranded.
        if not self._has_time_triggers:
            return
        now = self._gw._clock.now()
        if now is None:
            return
        for tid, trig in self._triggers.items():
            when = trig["spec"]["when"]
            if when["type"] != "time" or when.get("after") != anchor_tid or trig["resolved"]:
                continue
            trig["next_mark"] = _advance(now, _parse_duration(when["offset"]).total_seconds())
            trig["resolved"] = True
            self._emit("anchored", tid, anchor=anchor_tid, mark=_iso(trig["next_mark"]))

    def _until_dt(self, trig: dict) -> datetime | None:
        when = trig["spec"]["when"]
        if "until" not in when:
            return None
        kind, val = _time_mark_spec(when["until"])
        return val if kind == "abs" else _advance(self._gw._clock.t0(), val.total_seconds())

    def _next_interval_seconds(self, trig: dict) -> float:
        """Virtual seconds to the next mark; drawn on the sync path so the sequence depends only on the seed."""
        every = trig["spec"]["when"]["every"]
        if isinstance(every, dict):
            mean_s = _parse_duration(every["mean"]).total_seconds()
            return trig["rng"].expovariate(1.0 / mean_s)
        return _parse_duration(every).total_seconds()

    def _advance_marks(self, trig: dict, now: datetime) -> tuple[list[datetime], bool, bool]:
        """SYNC: the marks due at `now` (<= _MAX_CATCHUP), advancing next_mark past them.

        Returns (due_marks, capped, terminated). A capped backlog drains on later evaluations of the
        same clock generation; a re-arm drops it. Marks are returned so each arrival can record the
        virtual instant it was due at."""
        when = trig["spec"]["when"]
        recurring = "every" in when
        count = when.get("count")
        until = self._until_dt(trig)
        marks: list[datetime] = []
        capped = False
        terminated = False
        while trig["next_mark"] is not None and now >= trig["next_mark"]:
            if until is not None and trig["next_mark"] > until:
                trig["next_mark"] = None
                terminated = True
                break
            marks.append(trig["next_mark"])
            if not recurring:
                trig["next_mark"] = None
                terminated = True
                break
            if count is not None and trig["fire_count"] + len(marks) >= count:
                trig["next_mark"] = None
                terminated = True
                break
            prev_mark = trig["next_mark"]
            trig["next_mark"] = _advance(prev_mark, self._next_interval_seconds(trig))
            if trig["next_mark"] <= prev_mark:  # saturated at the datetime ceiling: can't advance -> retire
                trig["next_mark"] = None       # (else now==next_mark==max would re-burst _MAX_CATCHUP every call)
                terminated = True
                break
            if len(marks) >= _MAX_CATCHUP:
                capped = True
                break
        return marks, capped, terminated

    def _mark_time_detected(self, tid: str, trig: dict, marks: list[datetime], capped: bool,
                            terminated: bool, provoking: dict) -> None:
        trig["status"] = "firing"
        trig["detected_at"] = _now()
        # first/last only: a burst can carry _MAX_CATCHUP marks, each with its own `fired` event.
        self._emit("detected", tid, provoking=provoking, due=len(marks), capped=capped,
                   first_mark=_iso(marks[0]), last_mark=_iso(marks[-1]))
        self._schedule(self._fire_time(tid, trig, marks, capped, terminated))

    async def _fire_time(self, tid: str, trig: dict, marks: list[datetime], capped: bool, terminated: bool) -> None:
        """Run the actions once per due mark, then re-arm (recurring) or retire (one-shot / terminated)."""
        try:
            await self._gw._log_event({"event_type": "trigger_fired", "source": "trigger_engine", "trigger_id": tid})
            for mark in marks:
                for index, action in enumerate(trig["spec"]["actions"]):
                    if not await self._run_action(tid, index, action, {}):
                        trig["status"] = "failed"
                        trig["next_mark"] = None
                        self._emit("failed", tid, action_index=index)
                        return
                trig["fire_count"] += 1
                trig["fired_at"] = _now()
                self._emit("fired", tid, fire_count=trig["fire_count"], mark=_iso(mark), capped=capped)
            trig["status"] = "fired" if terminated else "armed"
        except Exception as e:
            logger.exception(f"time trigger '{tid}' firing failed")
            trig["status"] = "failed"
            trig["next_mark"] = None
            self._emit("failed", tid, detail=str(e)[:300])
            return
        finally:
            self._recompute_time_flag()  # a terminated trigger lets the driver park again
        self._resolve_dependents(tid)  # success-only, outside the try (see _fire)

    def invalidate_readonly_cache(self) -> None:
        """Drop the memoized readOnlyHint set; a stale memo classes a read-only tool as writable."""
        self._readonly_tools = None

    def _is_readonly(self, name: str) -> bool:
        if self._readonly_tools is None:
            readonly: set[str] = set()
            for tools in self._gw._server_tools.values():
                for tool in tools:
                    annotations = getattr(tool, "annotations", None)
                    if annotations is not None and getattr(annotations, "readOnlyHint", None) is True:
                        readonly.add(tool.name)
            if not self._gw._server_tools:
                return False
            self._readonly_tools = readonly
        return name in self._readonly_tools

    def _schedule(self, coro) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _emit(self, kind: str, trigger_id: str | None, **payload) -> dict:
        assert kind in _EVENT_KINDS, f"unknown trigger event kind {kind!r}"
        self._seq += 1
        # `ts` is real; a catch-up burst collapses into a few ms of it, hence `virtual_time` too
        # (omitted when unarmed, so a clock-less deployment sees unchanged events).
        virtual = self._gw._clock.now()
        event = {"seq": self._seq, "ts": _now(),
                 **({"virtual_time": _iso(virtual)} if virtual is not None else {}),
                 "kind": kind, **({"trigger_id": trigger_id} if trigger_id else {}),
                 **payload}
        (self._events if len(self._events) < _EVENTS_HEAD else self._events_tail).append(event)
        if kind in _FAILURE_KINDS and trigger_id:
            # A failure no longer parks the status and the event log is capped, so this is the lasting record.
            trig = self._triggers.get(trigger_id)
            if trig is not None:
                trig["failure_count"] += 1
                trig["last_failure_at"] = event["ts"]
        logger.info(f"trigger event: {event}")
        return event

    async def _eval_state(self, tid: str, trig: dict, provoking: dict) -> None:
        try:
            matched = await self._run_check(trig["spec"]["when"]["check"])
        except Exception as e:
            self._emit("eval_error", tid, detail=str(e)[:300])
            matched = False
        finally:
            trig["evaluating"] = False
        if matched and trig["status"] == "armed":
            self._mark_detected(tid, trig, provoking)

    async def _run_check(self, check: dict, ctx: dict | None = None, role: str | None = None) -> bool:
        bindings: dict[str, Any] = {}
        for step in check["steps"]:
            result = await self._internal_call(step["tool"], _template(step.get("args", {}), bindings, ctx), role=role)
            if getattr(result, "isError", False):
                return False
            raw_text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
            try:
                data = json.loads(raw_text)
            except (ValueError, TypeError):
                data = None
            value = _apply_extract(step.get("extract"), data, raw_text)
            bind = step.get("bind")
            if bind is not None:
                if value in (None, "", []):
                    return False
                bindings[bind] = value
            if "predicate" in step and not _eval_predicate(step["predicate"], value):
                return False
        return True

    async def _internal_call(self, tool_name: str, arguments: dict, role: str | None = None) -> CallToolResult:
        """A child call stamped as `role` when one is given (a tool action's `as`); the pin otherwise. Gateway-native
        tools have no child and no identity, so the role is ignored there."""
        gw = self._gw
        await gw._ensure_tools_discovered()
        server_url = gw._tool_server_urls.get(tool_name)
        if server_url is None:
            tool = gw._mcp._tool_manager._tools.get(tool_name)
            if tool is None:
                raise RuntimeError(f"unknown tool {tool_name!r}")
            result = await tool.run(arguments)
            if isinstance(result, CallToolResult):
                return result
            text = result if isinstance(result, str) else json.dumps(result, default=str)
            return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)
        meta = {AGENT_ENV_ROLE_META_KEY: role} if role is not None else None
        return await gw._call_child_tool(server_url, tool_name, arguments, meta=meta)

    async def _fire(self, tid: str, trig: dict, ctx: dict) -> None:
        """Run the actions for one provoking call, then drain any calls queued while firing (repeat
        triggers). A failure always re-arms: a mirror states a standing invariant, so one transient
        error must not silence it for the rest of the run. Ends armed or fired (one-shot success)."""
        spec = trig["spec"]
        repeat = bool(spec["when"].get("repeat"))
        try:
            while True:
                ok = True
                try:
                    await self._gw._log_event({"event_type": "trigger_fired", "source": "trigger_engine", "trigger_id": tid,
                                               "provoking_tool": (ctx.get("args") is not None and spec["when"].get("tool")) or None})
                    for index, action in enumerate(spec["actions"]):
                        if not await self._run_action(tid, index, action, ctx):
                            ok = False
                            self._emit("failed", tid, action_index=index)
                            break
                    if ok:
                        trig["fire_count"] += 1
                        trig["fired_at"] = _now()
                        self._emit("fired", tid, fire_count=trig["fire_count"])
                except Exception as e:
                    logger.exception(f"trigger '{tid}' firing failed")
                    ok = False
                    self._emit("failed", tid, detail=str(e)[:300])
                if ok:
                    self._resolve_dependents(tid)  # success-only, outside the try so it can't flip a fired trigger to failed
                if trig["pending"]:
                    ctx = trig["pending"].popleft()
                    trig["status"] = "firing"
                    trig["detected_at"] = _now()
                    continue
                trig["status"] = "armed" if (repeat or not ok) else "fired"
                return
        finally:
            # An abnormal exit must not wedge the trigger in firing/queued, nor drop its queue silently.
            if trig["status"] in TRIGGER_IN_FLIGHT_STATUSES:
                self._emit("failed", tid, detail="firing did not settle", dropped=len(trig["pending"]))
                trig["status"] = "armed"
                trig["pending"].clear()
            trig["task"] = None

    async def _run_action(self, tid: str, index: int, action: dict, ctx: dict) -> bool:
        changelog_before = self._gw._query_changelog_id()
        try:
            if action["type"] == "permission":
                tools = action["tools"] if isinstance(action["tools"], list) else [action["tools"]]
                async with self._gw._role_rules_lock:
                    for tool in tools:
                        self._gw._apply_rule(action["role"], tool, action["action"] == "disable")
                detail = {"role": action["role"], "action": action["action"], "tools": tools}
            elif action["type"] == "tool":
                role = action.get("as")
                acting = {"as": role} if role else {}
                try:
                    args = _template(action.get("args", {}), {}, ctx)
                except TemplateError as e:
                    self._emit("action_failed", tid, action_index=index, detail=str(e)[:300], tool=action["tool"])
                    return False
                result = await self._internal_call(action["tool"], args, role=role)
                text = _result_text(result)
                failed = bool(getattr(result, "isError", False))
                # The write itself, not just the trigger_fired marker: a Harbor grader has no timeline.
                await self._gw._log_event({
                    "event_type": "internal_tool_call", "source": "trigger_engine", "trigger_id": tid,
                    "action_index": index, "tool": action["tool"], "arguments": _echo_args(args), "as": role,
                    "ok": not failed, "result_len": len(text),
                    "result_sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:_ECHO_DIGEST_CHARS],
                    "changelog_id_before": changelog_before, "changelog_id_after": self._gw._query_changelog_id()})
                if failed:
                    # The refusal text is the only account of why a mirror broke; counters cannot replace it.
                    self._emit("action_failed", tid, action_index=index, detail=text[:300],
                               tool=action["tool"], args=_echo_args(args), **acting)
                    return False
                detail = {"tool": action["tool"], "args": _echo_args(args), **acting}
            else:
                if not await self._run_nl(tid, index, action, ctx):
                    return False
                detail = {"nl": action["instruction"][:120]}
            if "verify" in action and action["type"] != "nl":
                # only an action's own verify inherits its `as`; when.check state probes read as the pin
                if not await self._run_verify(tid, index, action["verify"], ctx, role=action.get("as")):
                    return False
            self._emit("action_ok", tid, action_index=index, detail=detail,
                       changelog_id_before=changelog_before, changelog_id_after=self._gw._query_changelog_id())
            return True
        except Exception as e:
            self._emit("action_failed", tid, action_index=index, detail=str(e)[:300])
            return False

    async def _run_verify(self, tid: str, index: int, verify: dict, ctx: dict | None = None,
                          role: str | None = None) -> bool:
        if await self._run_check(verify, ctx, role=role):
            self._emit("verify_ok", tid, action_index=index)
            return True
        self._emit("verify_failed", tid, action_index=index)
        return False

    async def _run_nl(self, tid: str, index: int, action: dict, ctx: dict | None = None) -> bool:
        instruction = action["instruction"]
        for attempt in (1, 2):
            reply = await self._call_executor(instruction, context_id=f"trigger-{tid}")
            if reply is None:
                self._emit("action_failed", tid, action_index=index, detail=f"executor attempt {attempt} did not complete")
                return False
            if "verify" not in action:
                return True
            if await self._run_verify(tid, index, action["verify"], ctx):
                return True
            if attempt == 1:
                instruction = (f"{action['instruction']}\n\nYour previous attempt did not pass verification — "
                               f"the expected change is not observable in the environment. Do it again exactly as described.")
        self._emit("action_failed", tid, action_index=index, detail="verification failed after retry")
        return False

    async def _call_executor(self, instruction: str, context_id: str) -> str | None:
        executor = self._executor
        if executor is None:
            return None
        url = f"{executor['a2a_url']}/a2a"
        deadline = asyncio.get_running_loop().time() + executor["timeout_seconds"]
        async with httpx.AsyncClient(timeout=30) as client:
            send = await client.post(url, json={
                "jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "message/send",
                "params": {"message": {"role": "user", "messageId": uuid.uuid4().hex,
                                       "contextId": context_id,
                                       "parts": [{"kind": "text", "text": instruction}]}},
            })
            send.raise_for_status()
            task_id = (send.json().get("result") or {}).get("id")
            if not task_id:
                return None
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(2)
                poll = await client.post(url, json={"jsonrpc": "2.0", "id": str(uuid.uuid4()),
                                                    "method": "tasks/get", "params": {"id": task_id}})
                poll.raise_for_status()
                status = ((poll.json().get("result") or {}).get("status") or {})
                if status.get("state") in _TERMINAL_A2A_STATES:
                    if status.get("state") != "completed":
                        return None
                    parts = (status.get("message") or {}).get("parts") or []
                    return "\n".join(p.get("text", "") for p in parts if p.get("kind") == "text")
        return None


def _result_text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
