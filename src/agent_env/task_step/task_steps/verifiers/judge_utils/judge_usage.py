"""A judge step's record of its run, beside its usage: each model call the judge made, with a reference to exactly what
it was sent, and the settings and trajectory sources those calls carried, for its grade (``judge_calls``,
``verifier_settings``). The record of the run in progress is held per task, since a step is shared by every run of its
task: a judge step's ``execute`` scopes it (``scope``) and its judging starts it (``start``).

What each call was sent is uploaded in the background (``judge_inputs``) and its reference filled into the call's record
when the run settles (``JudgeUsage.settle``), before the grade is written. An agent call's input names the files the
judge step staged in the judge's sandbox and, for a judge the task deployed, what the task's own steps loaded there as
the run recorded them (``load_artifact.loads_into``): best effort, since a file a step wrote by running a command there
is recorded nowhere."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional, Union

from agentenv_protocol.a2a_agent import Usage

from agent_env.config.model import ModelRole
from agent_env.config.model_pricing import ReportedBy, Tokens, price
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils import judge_inputs
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_inputs import StagingRecorder, Uploads
from agent_env.task_step.usage import StepUsage


@dataclass(frozen=True)
class AgentInput:
    """What an agent call was sent, stored when the run settles, after the files staged for it."""

    message: str
    files: list[dict[str, Any]]
    loaded: list[dict[str, Any]]


#: What a call was sent: a direct call's input as uploaded (``Uploads.put``'s reference), or an agent call's, uploaded
#: when the run settles.
Sent = Union[Mapping[str, Any], AgentInput, None]


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
        self.staged: list[dict[str, Any]] = []
        self.loaded: list[dict[str, Any]] = []
        self.sources: Optional[list[dict[str, Any]]] = None
        self.uploads = Uploads()
        self._staged_in: dict[str, list[dict[str, Any]]] = {}
        self._deferred: list[tuple[dict[str, Any], AgentInput]] = []

    def call(self, role: ModelRole, usage: Optional[Usage], *, model: Optional[str], endpoint_host: Optional[str],
             reported_by: ReportedBy = ReportedBy.HARNESS, cost: Optional[Mapping[str, Any]] = None,
             sent: Sent = None) -> None:
        """The base's, and the answered call in the record: ``cost``, the cost keys the call itself carried (a direct
        call's, from ``model_call``), else its price from LiteLLM's map; ``sent``, what it was sent (``input``, or
        ``agent_input``, stored when the run settles)."""
        super().call(role, usage, model=model, endpoint_host=endpoint_host, reported_by=reported_by)
        self._record("answered", usage, model, endpoint_host, reported_by, cost=cost, sent=sent)

    def cut_off_call(self, role: ModelRole, *, model: Optional[str], endpoint_host: Optional[str],
                     reported_by: ReportedBy = ReportedBy.HARNESS, sent: Sent = None) -> None:
        super().cut_off_call(role, model=model, endpoint_host=endpoint_host, reported_by=reported_by)
        self._record("cut_off", None, model, endpoint_host, reported_by, sent=sent)

    def refused_call(self, *, model: Optional[str], endpoint_host: Optional[str], error: str,
                     sent: Sent = None) -> None:
        """A call that got no answer and cost nothing, as one LiteLLM refused before sending it or one the endpoint
        refused: in the record, not in the step's usage."""
        self._record("refused", None, model, endpoint_host, ReportedBy.PROVIDER, error=error, sent=sent)

    def read(self, sources: list[dict[str, Any]]) -> None:
        """The trajectory sources the run's judge read: one per turn, with its format and, for an ATIF copy, the
        adapter version that made it."""
        self.sources = list(sources)

    async def input(self, body: Any) -> dict[str, Any]:
        """``body`` as the JSON a direct call was sent, uploaded: its reference."""
        return await self.uploads.put_json(body)

    def agent_input(self, message: str, sandbox_id: Optional[str] = None) -> AgentInput:
        """What an agent call is sent now: its message, the files staged so far in its judge's sandbox
        (``sandbox_id``; every one staged, without) and the task's loads into the judge."""
        files = self._staged_in[sandbox_id] if sandbox_id in self._staged_in else self.staged
        return AgentInput(message, list(files), list(self.loaded))

    def recording(self, sandbox: Any, sandbox_id: Optional[str] = None) -> StagingRecorder:
        """``sandbox``, every file written into it kept in ``staged`` and, when it's named, in its own sandbox's list,
        which the calls to that sandbox's judge name, as when each sample has a judge of its own."""
        manifests = (self.staged,) if sandbox_id is None else (self.staged, self._staged_in.setdefault(sandbox_id, []))
        return StagingRecorder(sandbox, manifests, self.uploads)

    async def settle(self) -> None:
        """Waits, up to ``SETTLE_SECONDS`` all told, for the run's uploads; stores each agent call's input once its
        staged files are; and marks each staged file and call whose input wasn't stored by then ``input_error``."""
        deadline = self.uploads.deadline()
        await self.uploads.wait(deadline)
        for entry in self.staged:
            if error := self.uploads.failed(entry.get("sha256")):
                entry.pop("object_url", None)
                entry["input_error"] = error
        deferred, self._deferred = self._deferred, []
        for record, sent in deferred:
            body = {"message": sent.message, "files": sent.files, "loaded": sent.loaded}
            try:
                record["input"] = await asyncio.wait_for(self.uploads.put_json(body), self.uploads.left(deadline))
            except TimeoutError:
                record["input_error"] = f"TimeoutError: not encoded within {judge_inputs.SETTLE_SECONDS:.0f}s"
        await self.uploads.wait(deadline)
        for record in self.calls:
            if error := self.uploads.failed((record.get("input") or {}).get("sha256")):
                del record["input"]
                record["input_error"] = error

    def carried(self, sent: dict[str, Any], not_sent: dict[str, Any], endpoint_host: Optional[str]) -> None:
        """The settings the run's calls carried, kept from its first call, since every call of a run carries the same."""
        if self.sent is None:
            self.sent, self.not_sent, self.endpoint_host = dict(sent), dict(not_sent), endpoint_host

    def _record(self, outcome: str, usage: Optional[Usage], model: Optional[str], endpoint_host: Optional[str],
                reported_by: ReportedBy, *, error: Optional[str] = None, cost: Optional[Mapping[str, Any]] = None,
                sent: Sent = None) -> None:
        if cost is None:
            cost = price(Tokens.of(usage.to_dict()) if usage is not None else None,
                         usage.cost_usd if usage is not None else None, model=model, prices=self._prices,
                         reported_by=reported_by).fields()
        record = {"outcome": outcome, "model": model, "endpoint_host": endpoint_host,
                  "usage": usage.to_dict() if usage is not None else None, **cost,
                  "batch": _BATCH.get(), "sample": _SAMPLE.get(), "attempt": _ATTEMPT.get(),
                  **({"input": dict(sent)} if isinstance(sent, Mapping) else {}),
                  **({"error": error} if error is not None else {})}
        self.calls.append(record)
        if isinstance(sent, AgentInput):
            self._deferred.append((record, sent))


_RUN: ContextVar[Optional[JudgeUsage]] = ContextVar("judge_run", default=None)
_BATCH: ContextVar[Optional[int]] = ContextVar("judge_batch", default=None)
_SAMPLE: ContextVar[Optional[int]] = ContextVar("judge_sample", default=None)
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
    """The record of each model call the judge run this task is in has made so far. A record is final once the run
    settles (``JudgeUsage.settle``), which fills in its ``input`` or marks it ``input_error``."""
    judge = _RUN.get()
    return list(judge.calls) if judge is not None else []


def batch(number: Optional[int]) -> None:
    """The judge batch the calls made from here in this task belong to (None for a run in one batch)."""
    _BATCH.set(number)


def sample(number: Optional[int]) -> None:
    """The judge sample the calls made from here in this task belong to (None for a run that takes one)."""
    _SAMPLE.set(number)


def attempt(number: int) -> None:
    """The output attempt the calls made from here in this task belong to."""
    _ATTEMPT.set(number)
