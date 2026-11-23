"""A judge step's record of its run, beside its usage: each model call the judge made, with a reference to exactly what
it was sent, and the settings and trajectory sources those calls carried, for its grade (``judge_calls``,
``verifier_settings``). The record of the run in progress is held per task, since a step is shared by every run of its
task: a judge step's ``execute`` scopes it (``scope``) and its judging starts it (``start``).

What a call was sent is stored by its sha256 under ``judge-inputs/`` in the object store, so a retried request's input or
a file another run staged is stored once, and a reader checks it against the key it's stored under. Uploads run in the
background, off the judging's path (``JudgeUsage.upload``); the step waits for them once, before it writes its grade
(``JudgeUsage.settle``), when an upload that failed or didn't finish is marked ``input_error``, which never fails the grade.
An agent call's input names the files the judge step staged in the judge's sandbox and, for a judge the task deployed,
what the task's own steps loaded there as the run recorded them (``loads_into``): best effort, since a file a step wrote
by running a command there is recorded nowhere."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional, Union

from agentenv_protocol.a2a_agent import Usage

from agent_env.config import get_config
from agent_env.config.model import ModelRole
from agent_env.config.model_pricing import ReportedBy, Tokens, price
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.thread_work import finish_on_thread
from agent_env.task_step.usage import StepUsage

logger = logging.getLogger(__name__)

INPUTS_PREFIX = "judge-inputs"
#: The run's records of what its steps loaded into an agent's sandbox (``load_artifact``), each entry naming the agent.
LOAD_RECORDS = ("loaded_file_artifact_universes", "loaded_environment_artifacts", "loaded_urls")
#: How many of a judge run's uploads run at once, and how long its grade waits for those still running.
UPLOAD_CONCURRENCY = 16
SETTLE_SECONDS = 120.0


@dataclass(frozen=True)
class AgentInput:
    """What an agent call was sent, stored when the run settles, after the files staged for it."""

    message: str
    files: list[dict[str, Any]]
    loaded: list[dict[str, Any]]


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
        self._uploads: dict[str, asyncio.Task] = {}
        self._gate = asyncio.Semaphore(UPLOAD_CONCURRENCY)
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

    def upload(self, data: bytes, content_type: str) -> dict[str, Any]:
        """``data`` stored under ``judge-inputs/<sha256>`` in the background, once however often the run sends it: its
        reference (``object_url``, ``bytes``, ``sha256``) at once; ``settle`` marks it if it wasn't stored."""
        digest = hashlib.sha256(data).hexdigest()
        config = get_config()
        store, key = config.get_object_store(), f"{config.get_artifact_key_prefix()}{INPUTS_PREFIX}/{digest}"
        if digest not in self._uploads:
            task = asyncio.ensure_future(self._put(store, key, data, content_type))
            task.add_done_callback(_logged)
            self._uploads[digest] = task
        return {"object_url": store.object_url(key), "bytes": len(data), "sha256": digest}

    async def input(self, body: Any) -> dict[str, Any]:
        """``body`` as the JSON a call was sent, uploaded (``upload``): ``{"input": reference}``."""
        return {"input": self.upload(await asyncio.to_thread(_encoded, body), "application/json")}

    def agent_input(self, message: str) -> AgentInput:
        """What an agent call is sent now: its message, the files staged so far and the task's loads into the judge."""
        return AgentInput(message, list(self.staged), list(self.loaded))

    async def settle(self) -> None:
        """Waits, up to ``SETTLE_SECONDS`` all told, for the run's uploads; stores each agent call's input once its
        staged files are; and marks each staged file and call whose upload failed or hadn't finished ``input_error``."""
        deadline = asyncio.get_running_loop().time() + SETTLE_SECONDS
        await self._wait(deadline)
        for entry in self.staged:
            if error := self._failed(entry.get("sha256")):
                entry.pop("object_url", None)
                entry["input_error"] = error
        deferred, self._deferred = self._deferred, []
        for record, sent in deferred:
            record.update(await self.input({"message": sent.message, "files": sent.files, "loaded": sent.loaded}))
        await self._wait(deadline)
        for record in self.calls:
            if error := self._failed((record.get("input") or {}).get("sha256")):
                del record["input"]
                record["input_error"] = error

    async def _put(self, store: Any, key: str, data: bytes, content_type: str) -> None:
        async with self._gate:
            await finish_on_thread(functools.partial(_put_once, store, key, data, content_type),
                                   "Storing a judge's input")

    async def _wait(self, deadline: float) -> None:
        pending = [task for task in self._uploads.values() if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=max(0.0, deadline - asyncio.get_running_loop().time()))

    def _failed(self, digest: Optional[str]) -> Optional[str]:
        """Why the run's upload of ``digest`` didn't store it; None when it did, or the run uploaded no such thing."""
        task = self._uploads.get(digest) if digest else None
        if task is None:
            return None
        if not task.done():
            task.cancel()
            return f"TimeoutError: still uploading after {SETTLE_SECONDS:.0f}s"
        if task.cancelled():
            return "CancelledError: the upload was cancelled"
        exc = task.exception()
        return f"{type(exc).__name__}: {exc}"[:300] if exc is not None else None

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
                  "batch": _BATCH.get(), "attempt": _ATTEMPT.get(),
                  **(sent if isinstance(sent, Mapping) else {}), **({"error": error} if error is not None else {})}
        self.calls.append(record)
        if isinstance(sent, AgentInput):
            self._deferred.append((record, sent))


def loads_into(context: TaskStepContext, agent_name: str) -> list[dict[str, Any]]:
    """What the run's own steps loaded into ``agent_name``'s sandbox, each entry as its load recorded it, under the key
    it's recorded under (``record``)."""
    return [{"record": key, **entry} for key in LOAD_RECORDS for entry in context.metadata.get(key) or []
            if isinstance(entry, Mapping) and entry.get("agent_name") == agent_name]


def _encoded(body: Any) -> bytes:
    return json.dumps(body, ensure_ascii=False, sort_keys=True).encode()


def _put_once(store: Any, key: str, data: bytes, content_type: str) -> None:
    if not store.exists(key):
        store.put(key, data, content_type=content_type, allow_overwrite=True)


def _logged(task: asyncio.Task) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("a judge's input wasn't stored: %s: %s", type(exc).__name__, exc)


class StagingRecorder:
    """The judge's sandbox, every file written into it kept in its run's manifest (``JudgeUsage.staged``): a file
    written as text stored by its sha256 too, in the background (``JudgeUsage.upload``), one copied from a stored object
    named by that object's URL."""

    def __init__(self, sandbox: Any, judge: JudgeUsage) -> None:
        self._sandbox, self._judge = sandbox, judge

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sandbox, name)

    async def write_file_from_text(self, text: str, path: str, **kwargs: Any) -> Any:
        written = await self._sandbox.write_file_from_text(text, path, **kwargs)
        self._judge.staged.append({"path": path, **self._judge.upload(text.encode(), "text/plain; charset=utf-8")})
        return written

    async def write_file_from_object(self, object_url: str, path: str, **kwargs: Any) -> Any:
        written = await self._sandbox.write_file_from_object(object_url, path, **kwargs)
        self._judge.staged.append({"path": path, "object_url": object_url})
        return written


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
