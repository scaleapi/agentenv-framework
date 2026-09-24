"""Human-in-the-loop review: pause the task until an operator continues or aborts.

Registers an ``awaiting`` review and blocks (heartbeating to keep the activity
alive) until the hub records a decision or the timeout elapses — continue returns
normally, abort raises so downstream steps don't run. The wait holds no sandbox,
so place it after the agent's output is persisted and before ``collect_artifacts``
(an abort then excludes the run's artifacts).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10800  # 3h
DEFAULT_POLL_INTERVAL_SECONDS = 5
_UPSTREAM_RESPONSE_MAX_CHARS = 4000
_UPSTREAM_STRUCTURED_MAX_CHARS = 16000


class ReviewAbortedError(RuntimeError):
    """Raised when an operator chooses not to continue past a review."""


def _bound_structured(value: Any) -> Any:
    """Return `value` as-is when small, else a truncated preview so a huge output can't bloat the polled review doc."""
    try:
        serialized = json.dumps(value, default=str)
    except (TypeError, ValueError):
        serialized = str(value)
    if len(serialized) <= _UPSTREAM_STRUCTURED_MAX_CHARS:
        return value
    return {
        "_truncated": True,
        "note": "output truncated for review; full output in the trajectory",
        "preview": serialized[:_UPSTREAM_STRUCTURED_MAX_CHARS],
    }


class ReviewTaskStep(TaskStep):
    type: ClassVar[str] = "review"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: Optional[int],
        label: Optional[str] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.label = label
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["label"] = self.label
        base["timeout_seconds"] = self.timeout_seconds
        base["poll_interval_seconds"] = self.poll_interval_seconds
        return base

    @classmethod
    def from_dict(cls, data: dict) -> ReviewTaskStep:
        return cls(
            **cls._base_from_dict(data),
            label=data.get("label"),
            timeout_seconds=data.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            poll_interval_seconds=data.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS),
        )

    def _upstream_summary(self, context: TaskStepContext) -> dict[str, Any]:
        """Most recent prior step's output for the operator to review — structured verdict if present, else trimmed text. Bounded (polled by the UI)."""
        for response in reversed(context.prompt_responses):
            if response.structured_output is not None:
                return {
                    "step_id": response.step_id,
                    "structured_output": _bound_structured(response.structured_output),
                }
        if context.prompt_responses:
            last = context.prompt_responses[-1]
            return {"step_id": last.step_id, "response": (last.response or "")[:_UPSTREAM_RESPONSE_MAX_CHARS]}
        return {}

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        if not context.instance_id:
            raise RuntimeError("review: context has no instance_id — cannot register a decision")
        store = self._register_awaiting(context)
        record = await self._await_decision(store, context)
        return self._apply_decision(context, record)

    def _register_awaiting(self, context: TaskStepContext):
        from agent_env.task_step.review_store import get_review_store

        store = get_review_store()
        store.put_awaiting(
            context.instance_id, self.id, label=self.label, upstream=self._upstream_summary(context)
        )
        logger.info(f"review '{self.id}' awaiting operator decision (timeout={self.timeout_seconds}s)")
        return store

    async def _await_decision(self, store, context: TaskStepContext) -> dict:
        """Poll until decided (return the record) or the wall-clock deadline (raise TimeoutError)."""
        from agent_env.task_step.review_store import ABORT, CONTINUE

        heartbeat = context.metadata.get("_heartbeat_fn")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds  # wall-clock, so a non-positive interval can't spin forever
        interval = self.poll_interval_seconds if self.poll_interval_seconds > 0 else DEFAULT_POLL_INTERVAL_SECONDS
        while loop.time() < deadline:
            self._beat(heartbeat)
            record = store.get(context.instance_id, self.id) or {}
            if record.get("state") in (CONTINUE, ABORT):
                return record
            await asyncio.sleep(min(interval, max(0.0, deadline - loop.time())))
        raise TimeoutError(
            f"review '{self.id}' timed out after {self.timeout_seconds}s awaiting a decision"
        )

    def _apply_decision(self, context: TaskStepContext, record: dict) -> TaskStepContext:
        from agent_env.task_step.review_store import CONTINUE

        decided_by = record.get("decided_by")
        if record.get("state") == CONTINUE:
            logger.info(f"review '{self.id}': operator chose continue")
            context.metadata.setdefault("review_decisions", {})[self.id] = {
                "decision": CONTINUE,
                "decided_by": decided_by,
            }
            return context
        logger.info(f"review '{self.id}': operator chose abort")
        raise ReviewAbortedError(
            f"review '{self.id}' aborted by operator" + (f" ({decided_by})" if decided_by else "")
        )

    @staticmethod
    def _beat(heartbeat) -> None:
        if callable(heartbeat):
            try:
                heartbeat()
            except Exception:  # best-effort; never fail the wait on it
                pass
