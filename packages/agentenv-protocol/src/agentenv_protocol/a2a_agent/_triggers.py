"""Deterministic implementation of ``urn:agentenv:triggers/v1``."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import regex

_MAX_TRIGGERS = 256
_MAX_CONTEXTS = 1_024
_MAX_LOG_ENTRIES = 1_024
_MAX_TRIGGER_ID_LENGTH = 128
_MAX_CONTEXT_ID_LENGTH = 256
_MAX_ENV_ID_LENGTH = 256
_MAX_STATUS_LENGTH = 128
_MAX_ACTION_TEXT_LENGTH = 8_192
_MAX_REGEX_LENGTH = 4_096
_MAX_TRIGGER_SPEC_BYTES = 16 * 1_024
MAX_SOLVER_MESSAGE_LENGTH = 200_000
_DEFAULT_REGEX_BUDGET_SECONDS = 1.0
_MAX_PATTERN_ERROR_LENGTH = 80


class TriggerError(ValueError):
    """Raised when a trigger registration or decision request is invalid."""

    status_code = 400


class TriggerTimeoutError(TriggerError):
    """Raised when regex evaluation exhausts a decision's shared time budget."""

    status_code = 504


class _RegexBudget:
    """One monotonic deadline shared by every regex in a decision."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.deadline = time.monotonic() + seconds

    def remaining(self) -> float:
        return self.deadline - time.monotonic()


def _validate_string_length(value: str, where: str, maximum: int) -> None:
    if len(value) > maximum:
        raise TriggerError(f"{where} must be at most {maximum} characters")


def _validate_specification_size(
    specification: dict[str, Any], trigger_id: str
) -> None:
    """Reject an accepted trigger whose retained JSON representation is too large."""
    try:
        size = len(
            json.dumps(specification, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    except (TypeError, ValueError) as exc:
        raise TriggerError(
            f"trigger '{trigger_id}' specification must be JSON-compatible"
        ) from exc
    if size > _MAX_TRIGGER_SPEC_BYTES:
        raise TriggerError(
            f"trigger '{trigger_id}' specification must be at most "
            f"{_MAX_TRIGGER_SPEC_BYTES} bytes"
        )


def _validate_predicate(predicate: Any, where: str) -> None:
    keys = ("regex", "equals", "exists")
    if not isinstance(predicate, dict) or sum(key in predicate for key in keys) != 1:
        raise TriggerError(f"{where}: predicate must have exactly one of {keys}")
    if "regex" in predicate:
        if not isinstance(predicate["regex"], str):
            raise TriggerError(f"{where}: regex must be a string")
        _validate_string_length(
            predicate["regex"], f"{where}: regex", _MAX_REGEX_LENGTH
        )
        try:
            regex.compile(predicate["regex"])
        except regex.error as exc:
            raise TriggerError(f"{where}: invalid regex: {exc}") from exc
    if "exists" in predicate and not isinstance(predicate["exists"], bool):
        raise TriggerError(f"{where}: exists must be a boolean")


def _matches_predicate(
    predicate: dict[str, Any], value: Any, budget: _RegexBudget
) -> bool:
    if "regex" in predicate:
        text = value if isinstance(value, str) else "" if value is None else str(value)
        pattern = predicate["regex"]
        remaining = budget.remaining()
        display_pattern = (
            pattern
            if len(pattern) <= _MAX_PATTERN_ERROR_LENGTH
            else f"{pattern[:_MAX_PATTERN_ERROR_LENGTH]}..."
        )
        if remaining <= 0:
            raise TriggerTimeoutError(
                "decide: regex budget of "
                f"{budget.seconds}s exhausted before evaluating {display_pattern!r}"
            )
        try:
            return regex.search(pattern, text, timeout=remaining) is not None
        except TimeoutError as exc:
            raise TriggerTimeoutError(
                f"decide: regex {display_pattern!r} exceeded the "
                f"{budget.seconds}s decision budget"
            ) from exc
    if "equals" in predicate:
        return value == predicate["equals"]
    present = value is not None and value != "" and value != []
    return present == predicate["exists"]


@dataclass
class _ContextState:
    fired: set[str] = field(default_factory=set)
    last_turn: int | None = None


class TriggerEngine:
    """Small, bounded stateful engine shared by every framework-backed agent."""

    def __init__(
        self,
        *,
        max_triggers: int = _MAX_TRIGGERS,
        max_contexts: int = _MAX_CONTEXTS,
        max_log_entries: int = _MAX_LOG_ENTRIES,
        regex_budget_seconds: float = _DEFAULT_REGEX_BUDGET_SECONDS,
    ) -> None:
        if min(max_triggers, max_contexts, max_log_entries) < 1:
            raise ValueError("trigger engine limits must be positive")
        if (
            isinstance(regex_budget_seconds, bool)
            or not isinstance(regex_budget_seconds, (int, float))
            or not math.isfinite(regex_budget_seconds)
            or regex_budget_seconds <= 0
        ):
            raise ValueError("regex_budget_seconds must be a finite positive number")
        self._max_triggers = max_triggers
        self._max_contexts = max_contexts
        self._regex_budget_seconds = float(regex_budget_seconds)
        self._triggers: dict[str, dict[str, Any]] = {}
        self._contexts: OrderedDict[str, _ContextState] = OrderedDict()
        self._log: deque[dict[str, Any]] = deque(maxlen=max_log_entries)
        self._sequence = 0

    def register(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TriggerError("registration body must be an object")
        triggers = payload.get("triggers")
        if not isinstance(triggers, list) or not triggers:
            raise TriggerError("'triggers' must be a non-empty list")
        if len(triggers) > self._max_triggers:
            raise TriggerError(
                f"registration may contain at most {self._max_triggers} triggers"
            )

        normalized: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for index, trigger in enumerate(triggers):
            trigger_id, specification = self._normalize_trigger(trigger, index)
            if trigger_id in seen:
                raise TriggerError(f"duplicate trigger id '{trigger_id}' in this batch")
            seen.add(trigger_id)
            normalized.append((trigger_id, specification))

        for trigger_id, specification in normalized:
            existing = self._triggers.get(trigger_id)
            if existing is not None and existing != specification:
                raise TriggerError(
                    f"trigger '{trigger_id}' already registered with a different spec "
                    "(use a new id)"
                )
        new_trigger_count = sum(
            trigger_id not in self._triggers for trigger_id, _ in normalized
        )
        if len(self._triggers) + new_trigger_count > self._max_triggers:
            raise TriggerError(
                f"registration would exceed the {self._max_triggers} trigger limit"
            )
        for trigger_id, specification in normalized:
            self._triggers[trigger_id] = specification
        added = [trigger_id for trigger_id, _ in normalized]
        self._emit("registered", detail={"added": added})
        return {"ok": True, "added": added, "all": list(self._triggers)}

    def decide(
        self,
        *,
        turn: Any,
        solver_message: str = "",
        context_id: str = "default",
        env_triggers: Any = None,
    ) -> dict[str, Any]:
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
            raise TriggerError("decide: 'turn' must be a positive integer")
        if not isinstance(context_id, str) or not context_id:
            raise TriggerError("decide: 'context_id' must be a non-empty string")
        if len(context_id) > _MAX_CONTEXT_ID_LENGTH:
            raise TriggerError(
                f"decide: 'context_id' must be at most {_MAX_CONTEXT_ID_LENGTH} characters"
            )
        if not isinstance(solver_message, str):
            raise TriggerError("decide: 'solver_message' must be a string")
        if len(solver_message) > MAX_SOLVER_MESSAGE_LENGTH:
            raise TriggerError(
                "decide: 'solver_message' must be at most "
                f"{MAX_SOLVER_MESSAGE_LENGTH} characters"
            )
        environment_state = env_triggers if isinstance(env_triggers, dict) else {}
        context = self._contexts.get(context_id)
        if context is None and len(self._contexts) >= self._max_contexts:
            raise TriggerError(
                f"decide: context limit of {self._max_contexts} has been reached"
            )
        reset = (
            context is not None
            and context.last_turn is not None
            and turn <= context.last_turn
        )
        already_fired = set() if reset or context is None else context.fired
        budget = _RegexBudget(self._regex_budget_seconds)

        texts: list[str] = []
        fired_ids: list[str] = []
        done = False
        for trigger_id, specification in self._triggers.items():
            if specification["once"] and trigger_id in already_fired:
                continue
            if self._matches(
                specification["when"],
                turn,
                solver_message,
                environment_state,
                budget,
            ):
                fired_ids.append(trigger_id)
                for action in specification["actions"]:
                    if action["type"] == "say":
                        texts.append(action["text"])
                    elif action["type"] == "end":
                        done = True

        # Commit only after every predicate succeeds. A timed-out decision can then
        # be retried without consuming the turn or partially firing once-only rules.
        context = self._context_state(context_id)
        if reset:
            context.fired.clear()
            self._emit("reset", context_id=context_id, turn=turn)
        context.last_turn = turn
        for trigger_id in fired_ids:
            context.fired.add(trigger_id)
            self._emit(
                "fired",
                trigger_id=trigger_id,
                turn=turn,
                context_id=context_id,
            )
        parts = [{"kind": "text", "text": "\n".join(texts)}] if texts else []
        return {"parts": parts, "done": done, "fired": fired_ids}

    def state(self) -> dict[str, Any]:
        return {
            "triggers": [
                {
                    "id": trigger_id,
                    "when_type": specification["when"]["type"],
                    "once": specification["once"],
                }
                for trigger_id, specification in self._triggers.items()
            ],
            "firing_log": list(self._log),
        }

    def _normalize_trigger(
        self, trigger: Any, index: int
    ) -> tuple[str, dict[str, Any]]:
        if not isinstance(trigger, dict):
            raise TriggerError(f"trigger[{index}] must be an object")
        trigger_id = trigger.get("id")
        if trigger_id is None:
            trigger_id = "trg_" + uuid.uuid4().hex[:12]
        elif not isinstance(trigger_id, str) or not trigger_id:
            raise TriggerError(f"trigger[{index}].id must be a non-empty string")
        elif len(trigger_id) > _MAX_TRIGGER_ID_LENGTH:
            raise TriggerError(
                f"trigger[{index}].id must be at most {_MAX_TRIGGER_ID_LENGTH} characters"
            )
        once = trigger.get("once", True)
        if not isinstance(once, bool):
            raise TriggerError(f"trigger '{trigger_id}'.once must be a boolean")
        when = self._normalize_when(trigger.get("when"), trigger_id)
        actions = trigger.get("actions")
        if not isinstance(actions, list) or not actions:
            raise TriggerError(
                f"trigger '{trigger_id}'.actions must be a non-empty list"
            )
        specification = {
            "id": trigger_id,
            "once": once,
            "when": when,
            "actions": [
                self._normalize_action(action, trigger_id) for action in actions
            ],
        }
        _validate_specification_size(specification, trigger_id)
        return trigger_id, specification

    def _normalize_when(self, when: Any, trigger_id: str) -> dict[str, Any]:
        if not isinstance(when, dict) or "type" not in when:
            raise TriggerError(
                f"trigger '{trigger_id}': 'when' must be an object with a 'type'"
            )
        kind = when["type"]
        if kind == "step":
            turn = when.get("turn", 1)
            if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
                raise TriggerError(
                    f"trigger '{trigger_id}': step.turn must be a positive integer"
                )
            comparison = when.get("cmp", "eq")
            if comparison not in ("eq", "gte"):
                raise TriggerError(
                    f"trigger '{trigger_id}': step.cmp must be 'eq' or 'gte'"
                )
            return {"type": kind, "turn": turn, "cmp": comparison}
        if kind == "env_trigger":
            env_id = when.get("env_id")
            source_id = when.get("trigger_id")
            if not isinstance(env_id, str) or not env_id:
                raise TriggerError(
                    f"trigger '{trigger_id}': env_trigger.env_id must be a non-empty string"
                )
            _validate_string_length(
                env_id,
                f"trigger '{trigger_id}': env_trigger.env_id",
                _MAX_ENV_ID_LENGTH,
            )
            if not isinstance(source_id, str) or not source_id:
                raise TriggerError(
                    f"trigger '{trigger_id}': env_trigger.trigger_id must be a non-empty string"
                )
            _validate_string_length(
                source_id,
                f"trigger '{trigger_id}': env_trigger.trigger_id",
                _MAX_TRIGGER_ID_LENGTH,
            )
            status = when.get("status", "fired")
            if not isinstance(status, str) or not status:
                raise TriggerError(
                    f"trigger '{trigger_id}': env_trigger.status must be a non-empty string"
                )
            _validate_string_length(
                status,
                f"trigger '{trigger_id}': env_trigger.status",
                _MAX_STATUS_LENGTH,
            )
            return {
                "type": kind,
                "env_id": env_id,
                "trigger_id": source_id,
                "status": status,
            }
        if kind == "conversational":
            where = when.get("where")
            if not isinstance(where, dict) or not where:
                raise TriggerError(
                    f"trigger '{trigger_id}': conversational.where must be a non-empty object"
                )
            for field_name, predicate in where.items():
                if field_name != "message":
                    raise TriggerError(
                        f"trigger '{trigger_id}': conversational.where field "
                        f"{field_name!r} not in ('message',)"
                    )
                _validate_predicate(
                    predicate, f"trigger '{trigger_id}': where.{field_name}"
                )
            return {"type": kind, "where": where}
        if kind in ("all", "any"):
            children = when.get("of")
            if not isinstance(children, list) or not children:
                raise TriggerError(
                    f"trigger '{trigger_id}': {kind}.of must be a non-empty list"
                )
            return {
                "type": kind,
                "of": [self._normalize_when(child, trigger_id) for child in children],
            }
        raise TriggerError(f"trigger '{trigger_id}': unknown when.type {kind!r}")

    def _normalize_action(self, action: Any, trigger_id: str) -> dict[str, Any]:
        if not isinstance(action, dict) or "type" not in action:
            raise TriggerError(
                f"trigger '{trigger_id}': each action must be an object with a 'type'"
            )
        kind = action["type"]
        if kind == "say":
            text = action.get("text")
            if not isinstance(text, str) or not text:
                raise TriggerError(
                    f"trigger '{trigger_id}': say.text must be a non-empty string"
                )
            _validate_string_length(
                text,
                f"trigger '{trigger_id}': say.text",
                _MAX_ACTION_TEXT_LENGTH,
            )
            return {"type": kind, "text": text}
        if kind == "end":
            return {"type": kind}
        raise TriggerError(f"trigger '{trigger_id}': unknown action.type {kind!r}")

    def _matches(
        self,
        when: dict[str, Any],
        turn: int,
        solver_message: str,
        env_triggers: dict[str, Any],
        budget: _RegexBudget,
    ) -> bool:
        kind = when["type"]
        if kind == "step":
            return turn == when["turn"] if when["cmp"] == "eq" else turn >= when["turn"]
        if kind == "env_trigger":
            return (
                env_triggers.get(when["env_id"], {}).get(when["trigger_id"])
                == when["status"]
            )
        if kind == "conversational":
            return all(
                _matches_predicate(predicate, solver_message, budget)
                for predicate in when["where"].values()
            )
        if kind == "all":
            return all(
                self._matches(child, turn, solver_message, env_triggers, budget)
                for child in when["of"]
            )
        if kind == "any":
            return any(
                self._matches(child, turn, solver_message, env_triggers, budget)
                for child in when["of"]
            )
        return False

    def _context_state(self, context_id: str) -> _ContextState:
        """Return context state without evicting one-shot firing history.

        A ``once`` trigger is once per context.  Silently evicting a context
        would discard that history and replay actions if the caller returned to
        the same context, so a full bounded cache rejects new contexts instead.
        """
        context = self._contexts.get(context_id)
        if context is not None:
            self._contexts.move_to_end(context_id)
            return context
        if len(self._contexts) >= self._max_contexts:
            raise TriggerError(
                f"decide: context limit of {self._max_contexts} has been reached"
            )
        context = _ContextState()
        self._contexts[context_id] = context
        return context

    def _emit(self, kind: str, **fields: Any) -> None:
        self._sequence += 1
        entry = {
            "seq": self._sequence,
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
        }
        entry.update(fields)
        self._log.append(entry)
