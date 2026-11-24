"""A judge step's record of its run, beside its usage: each model call the judge made and the settings those calls
carried, for its grade (``judge_calls``, ``verifier_settings``). The record of the run in progress is held per task,
since a step is shared by every run of its task: a judge step's ``execute`` scopes it (``scope``) and its judging starts
it (``start``)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from typing import Any, Optional

from agentenv_protocol.a2a_agent import Usage

from agent_env.config.model import ModelRole
from agent_env.config.model_pricing import ReportedBy, Tokens, price
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.usage import StepUsage


class JudgeUsage(StepUsage):
    """A judge step's usage, written as every step's is, and its own record of the run: each model call it made
    (``calls``) and the settings those calls carried (``carried``): those sent, those it couldn't send, and the endpoint
    host they reached."""

    def __init__(self, context: TaskStepContext, step_id: str) -> None:
        super().__init__(context, step_id)
        self.calls: list[dict[str, Any]] = []
        self.sent: Optional[dict[str, Any]] = None
        self.not_sent: dict[str, Any] = {}
        self.endpoint_host: Optional[str] = None

    def call(self, role: ModelRole, usage: Optional[Usage], *, model: Optional[str], endpoint_host: Optional[str],
             reported_by: ReportedBy = ReportedBy.HARNESS, cost: Optional[Mapping[str, Any]] = None) -> None:
        """The base's, and the answered call in the record: ``cost``, the cost keys the call itself carried (a direct
        call's, from ``model_call``), else its price from LiteLLM's map."""
        super().call(role, usage, model=model, endpoint_host=endpoint_host, reported_by=reported_by)
        self._record("answered", usage, model, endpoint_host, reported_by, cost=cost)

    def cut_off_call(self, role: ModelRole, *, model: Optional[str], endpoint_host: Optional[str],
                     reported_by: ReportedBy = ReportedBy.HARNESS) -> None:
        super().cut_off_call(role, model=model, endpoint_host=endpoint_host, reported_by=reported_by)
        self._record("cut_off", None, model, endpoint_host, reported_by)

    def refused_call(self, *, model: Optional[str], endpoint_host: Optional[str], error: str) -> None:
        """A call that got no answer and cost nothing, as one LiteLLM refused before sending it or one the endpoint
        refused: in the record, not in the step's usage."""
        self._record("refused", None, model, endpoint_host, ReportedBy.PROVIDER, error=error)

    def carried(self, sent: dict[str, Any], not_sent: dict[str, Any], endpoint_host: Optional[str]) -> None:
        """The settings the run's calls carried, kept from its first call, since every call of a run carries the same."""
        if self.sent is None:
            self.sent, self.not_sent, self.endpoint_host = dict(sent), dict(not_sent), endpoint_host

    def _record(self, outcome: str, usage: Optional[Usage], model: Optional[str], endpoint_host: Optional[str],
                reported_by: ReportedBy, *, error: Optional[str] = None,
                cost: Optional[Mapping[str, Any]] = None) -> None:
        if cost is None:
            cost = price(Tokens.of(usage.to_dict()) if usage is not None else None,
                         usage.cost_usd if usage is not None else None, model=model, prices=self._prices,
                         reported_by=reported_by).fields()
        self.calls.append({"outcome": outcome, "model": model, "endpoint_host": endpoint_host,
                           "usage": usage.to_dict() if usage is not None else None, **cost,
                           "batch": _BATCH.get(), "attempt": _ATTEMPT.get(),
                           **({"error": error} if error is not None else {})})


_RUN: ContextVar[Optional[JudgeUsage]] = ContextVar("judge_run", default=None)
_BATCH: ContextVar[Optional[int]] = ContextVar("judge_batch", default=None)
_ATTEMPT: ContextVar[Optional[int]] = ContextVar("judge_attempt", default=None)


@contextlib.contextmanager
def scope() -> Iterator[None]:
    """One run of a judge step: a record started within it is the one ``current`` gives until the run ends."""
    token = _RUN.set(None)
    try:
        yield
    finally:
        _RUN.reset(token)


def start(context: TaskStepContext, step_id: str) -> JudgeUsage:
    """The run's record, started: its usage written to ``context`` under ``step_id``."""
    judge = JudgeUsage(context, step_id)
    _RUN.set(judge)
    return judge


def current() -> Optional[JudgeUsage]:
    """The record of the judge run this task is in, None outside one or before it started."""
    return _RUN.get()


def calls() -> list[dict[str, Any]]:
    """The record of each model call the judge run this task is in has made so far."""
    judge = _RUN.get()
    return list(judge.calls) if judge is not None else []


def batch(number: Optional[int]) -> None:
    """The judge batch the calls made from here in this task belong to (None for a run in one batch)."""
    _BATCH.set(number)


def attempt(number: int) -> None:
    """The output attempt the calls made from here in this task belong to."""
    _ATTEMPT.set(number)
