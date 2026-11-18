"""What a step failure means for its run: the role the failed step played and why it failed.

A tolerated failure (``fail_task_on_error=False``) lets the run complete, so these facts are the only record of what
the run lost. Each ``context.metadata.failed_steps`` entry carries them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import httpx

from agent_env.task_step.context import VerifierNotPassingError
from agent_env.task_step.task_step import StepRole

if TYPE_CHECKING:
    from agent_env.task_step.task_step import TaskStep

__all__ = ["FailedStep", "FailureCause", "StepRole", "classify_failure"]


class FailureCause(StrEnum):
    KEY_BUDGET = "key_budget"
    KEY_AUTH = "key_auth"
    TRANSIENT = "transient"
    # The grader ran and the run failed its check: the grade exists, so nothing was lost.
    VERDICT = "verdict"
    UNKNOWN = "unknown"


# Each pattern matches only wording that the model proxy, the HTTP client or the model SDK writes. An error's text also
# carries env responses, tool output and model text, where a bare "401" or "status 500" says nothing about our key or
# our infra. Checked in order: a key that is out of budget or rejected fails every retry, so it wins over a timeout.
_CAUSE_PATTERNS: tuple[tuple[FailureCause, re.Pattern[str]], ...] = (
    (FailureCause.KEY_BUDGET, re.compile(
        r"Budget has been exceeded!|ExceededBudget:|ExceededTokenBudget|BudgetExceededError"
        r"|Budget exceeded for \S+: spent \$"
    )),
    (FailureCause.KEY_AUTH, re.compile(
        r"Authentication Error(?:,| -|:)|Invalid proxy server token|token_not_found_in_db|litellm\.AuthenticationError"
    )),
    (FailureCause.TRANSIENT, re.compile(
        r"\bAPI Error: (?:Request rejected \()?(?:5\d\d|429)\b|\bError code: (?:5\d\d|429)\b"
        r'|\bAPI Error: \{"error":\{[^\n]*"code":"(?:5\d\d|429)"\}\}|All connection attempts failed'
        r"|unexpected status (?:408|429|5\d\d)\b|Server error '5\d\d |Client error '429 |\brawError=429\b"
        r"|Server disconnected without sending a response|Request timed out"
        r"|\b(?:litellm|openai)\.(?:RateLimitError|APIConnectionError|APITimeoutError|Timeout|InternalServerError"
        r"|ServiceUnavailableError)\b"
        r"|OOMKilled|exit(?:ed)? (?:with )?code 137"
    )),
)

# A failure of one of these types is one the A2A client or the model SDK retries, whatever its message says.
_TRANSIENT_TYPES = (TimeoutError, ConnectionError, httpx.TransportError)


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and every exception it was explicitly raised from, each once.

    Only ``__cause__`` (``raise ... from``) and an exception group's members count. ``__context__`` is set for any
    exception raised inside an ``except``, including one the code had already recovered from, so it can't say why.
    """
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.append(current.__cause__)
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)


def classify_failure(exc: BaseException) -> FailureCause:
    """Why ``exc`` happened, read from the types and messages of ``exc`` and the exceptions it was raised from.

    Matching text is the only option for most causes: a judge or agent error crosses A2A and the model proxy as a
    message string, so its original type is gone by the time a step raises.
    """
    if isinstance(exc, VerifierNotPassingError):
        return FailureCause.VERDICT
    chain = list(_chain(exc))
    text = "\n".join(f"{type(e).__name__}: {e}" for e in chain)
    for cause, pattern in _CAUSE_PATTERNS:
        if pattern.search(text):
            return cause
    if any(isinstance(e, _TRANSIENT_TYPES) for e in chain):
        return FailureCause.TRANSIENT
    return FailureCause.UNKNOWN


@dataclass(frozen=True)
class FailedStep:
    """One ``context.metadata.failed_steps`` entry."""

    step_id: str
    step_type: str
    error: str
    error_type: str
    started_at_utc: str
    duration_seconds: float
    is_fatal: bool
    retried: bool = False
    step_role: StepRole = StepRole.OTHER
    cause: FailureCause = FailureCause.UNKNOWN
    # Set once a later action (e.g. a regrade) repaired what this failure lost.
    resolved_by: str | None = None

    @classmethod
    def from_exception(
        cls, step: TaskStep, exc: BaseException, *, started_at_utc: str, duration_seconds: float, is_fatal: bool,
    ) -> FailedStep:
        return cls(
            step_id=step.id,
            step_type=step.type,
            error=str(exc),
            error_type=type(exc).__name__,
            started_at_utc=started_at_utc,
            duration_seconds=duration_seconds,
            is_fatal=is_fatal,
            step_role=getattr(step, "step_role", StepRole.OTHER),
            cause=classify_failure(exc),
        )

    def to_dict(self) -> dict[str, Any]:
        """The stored form: the keys every reader already knows, plus ``step_role`` and ``cause``. ``retried`` and
        ``resolved_by`` appear only once set, as ``retried`` always has."""
        out: dict[str, Any] = {
            "step_id": self.step_id, "step_type": self.step_type,
            "error": self.error, "error_type": self.error_type,
            "started_at_utc": self.started_at_utc, "duration_seconds": self.duration_seconds,
            "is_fatal": self.is_fatal,
            "step_role": self.step_role.value, "cause": self.cause.value,
        }
        if self.retried:
            out["retried"] = True
        if self.resolved_by is not None:
            out["resolved_by"] = self.resolved_by
        return out

