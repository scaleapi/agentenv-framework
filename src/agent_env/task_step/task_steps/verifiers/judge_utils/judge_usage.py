"""A judge step's record of its run, beside its usage: each model call the judge made, with a reference to exactly what
it was sent, and the settings and trajectory sources those calls carried, for its grade (``judge_calls``,
``verifier_settings``). The record of the run in progress is held per task, since a step is shared by every run of its
task: a judge step's ``execute`` scopes it (``scope``) and its judging starts it (``start``).

What a call was sent is stored by its sha256 under ``judge-inputs/`` in the object store (``store_input``), so a retried
request's input or a file another run staged is stored once, and a reader checks it against the key it's stored under."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from typing import Any, Optional

from agentenv_protocol.a2a_agent import Usage

from agent_env.config import get_config
from agent_env.config.model import ModelRole
from agent_env.config.model_pricing import ReportedBy, Tokens, price
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.thread_work import finish_on_thread
from agent_env.task_step.usage import StepUsage

logger = logging.getLogger(__name__)

INPUTS_PREFIX = "judge-inputs"


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
        self.sources: Optional[list[dict[str, Any]]] = None

    def call(self, role: ModelRole, usage: Optional[Usage], *, model: Optional[str], endpoint_host: Optional[str],
             reported_by: ReportedBy = ReportedBy.HARNESS, cost: Optional[Mapping[str, Any]] = None,
             sent: Optional[Mapping[str, Any]] = None) -> None:
        """The base's, and the answered call in the record: ``cost``, the cost keys the call itself carried (a direct
        call's, from ``model_call``), else its price from LiteLLM's map; ``sent``, the reference to what it was sent
        (``store_input``)."""
        super().call(role, usage, model=model, endpoint_host=endpoint_host, reported_by=reported_by)
        self._record("answered", usage, model, endpoint_host, reported_by, cost=cost, sent=sent)

    def cut_off_call(self, role: ModelRole, *, model: Optional[str], endpoint_host: Optional[str],
                     reported_by: ReportedBy = ReportedBy.HARNESS, sent: Optional[Mapping[str, Any]] = None) -> None:
        super().cut_off_call(role, model=model, endpoint_host=endpoint_host, reported_by=reported_by)
        self._record("cut_off", None, model, endpoint_host, reported_by, sent=sent)

    def refused_call(self, *, model: Optional[str], endpoint_host: Optional[str], error: str,
                     sent: Optional[Mapping[str, Any]] = None) -> None:
        """A call that got no answer and cost nothing, as one LiteLLM refused before sending it or one the endpoint
        refused: in the record, not in the step's usage."""
        self._record("refused", None, model, endpoint_host, ReportedBy.PROVIDER, error=error, sent=sent)

    def read(self, sources: list[dict[str, Any]]) -> None:
        """The trajectory sources the run's judge read: one per turn, with its format and, for an ATIF copy, the
        adapter version that made it."""
        self.sources = list(sources)

    def carried(self, sent: dict[str, Any], not_sent: dict[str, Any], endpoint_host: Optional[str]) -> None:
        """The settings the run's calls carried, kept from its first call, since every call of a run carries the same."""
        if self.sent is None:
            self.sent, self.not_sent, self.endpoint_host = dict(sent), dict(not_sent), endpoint_host

    def _record(self, outcome: str, usage: Optional[Usage], model: Optional[str], endpoint_host: Optional[str],
                reported_by: ReportedBy, *, error: Optional[str] = None, cost: Optional[Mapping[str, Any]] = None,
                sent: Optional[Mapping[str, Any]] = None) -> None:
        if cost is None:
            cost = price(Tokens.of(usage.to_dict()) if usage is not None else None,
                         usage.cost_usd if usage is not None else None, model=model, prices=self._prices,
                         reported_by=reported_by).fields()
        self.calls.append({"outcome": outcome, "model": model, "endpoint_host": endpoint_host,
                           "usage": usage.to_dict() if usage is not None else None, **cost,
                           "batch": _BATCH.get(), "attempt": _ATTEMPT.get(), **(sent or {}),
                           **({"error": error} if error is not None else {})})


async def store_input(body: Any) -> dict[str, Any]:
    """``body`` as the JSON a judge call was sent, stored by its sha256 (``store_bytes``): ``{"input": reference}``, or
    ``{"input_error": why}`` when the store refused it, which never fails the grade."""
    data = json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
    try:
        return {"input": await store_bytes(data, "application/json")}
    except Exception as exc:
        logger.warning("a judge call's input wasn't stored: %s", exc)
        return {"input_error": f"{type(exc).__name__}: {exc}"[:300]}


async def store_bytes(data: bytes, content_type: str) -> dict[str, Any]:
    """``data`` in the object store under ``judge-inputs/<sha256>``, once whatever writes it: its ``object_url``,
    ``bytes`` and ``sha256``."""
    digest = hashlib.sha256(data).hexdigest()
    url = await finish_on_thread(functools.partial(_put_once, digest, data, content_type),
                                 "Storing a judge call's input")
    return {"object_url": url, "bytes": len(data), "sha256": digest}


def _put_once(digest: str, data: bytes, content_type: str) -> str:
    config = get_config()
    store, key = config.get_object_store(), f"{config.get_artifact_key_prefix()}{INPUTS_PREFIX}/{digest}"
    return store.object_url(key) if store.exists(key) else store.put(key, data, content_type=content_type,
                                                                      allow_overwrite=True)


class StagingRecorder:
    """The judge's sandbox, every file written into it kept in its run's manifest (``JudgeUsage.staged``): a file
    written as text stored by its sha256 too, one copied from a stored object named by that object's URL."""

    def __init__(self, sandbox: Any, judge: JudgeUsage) -> None:
        self._sandbox, self._judge = sandbox, judge

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sandbox, name)

    async def write_file_from_text(self, text: str, path: str, **kwargs: Any) -> Any:
        written = await self._sandbox.write_file_from_text(text, path, **kwargs)
        data = text.encode()
        try:
            stored = await store_bytes(data, "text/plain; charset=utf-8")
        except Exception as exc:
            logger.warning("a judge's staged file %s wasn't stored: %s", path, exc)
            stored = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "input_error": f"{type(exc).__name__}: {exc}"[:300]}
        self._judge.staged.append({"path": path, **stored})
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
