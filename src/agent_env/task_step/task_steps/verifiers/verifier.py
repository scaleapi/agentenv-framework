"""The base every verifier step subclasses. A verifier implements ``verify``; the base scores what it returns and writes
the step's named grade to ``context.metadata["grades"][<step id>]``, beside the ``verifications`` entries today's
readers take."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import packages_distributions
from typing import Any, ClassVar, Optional, TypedDict

from packaging.utils import canonicalize_name

from agent_env.config import get_config
from agent_env.plugins import _requirements
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import RetryConfig, TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from agent_env.task_step.thread_work import finish_on_thread
from agent_env.utils.canonical import canonical_json

logger = logging.getLogger(__name__)

GRADES_KEY = "grades"
VERIFICATIONS_KEY = "verifications"
# The version of the rules the base applies to every verifier's rows and pass rule; a change that can move a grade
# bumps it, and every grade records it in its verifier_settings.
SCORING_RULES = 1
DEFAULT_MIN_PASSING_SCORE = 1.0
OUTPUT_INLINE_BYTES = 16 * 1024
SETTING_INLINE_BYTES = 1024
# The step's fields that name or schedule it rather than say how it grades, left out of its verifier_settings.
_NOT_SETTINGS = ("id", "type", "version", "depends_on", "fail_task_on_error", "retry_config", "verifier_id")
_FRAMEWORK = "agentenv-framework"


class VerifierStatus(str, Enum):
    GRADED = "graded"
    SKIPPED = "skipped"
    FAILED = "failed"


class _GradeFields(TypedDict):
    name: str
    score: float
    min_passing_score: float
    passed: bool
    verifier_status: str
    verifier_error_reasoning: Optional[str]
    verifier_skip_reasoning: Optional[str]
    verifier_output: Any
    verifier_criteria: list[dict[str, Any]]
    verifier_type: str
    verifier_version: Optional[str]
    verifier_settings: dict[str, Any]


class Grade(_GradeFields, total=False):
    """A verifier step's named grade, as ``context.metadata["grades"][<step id>]`` holds it. ``verifier_status`` is a
    ``VerifierStatus`` value; each ``verifier_criteria`` row has ``id``, ``score``, ``result``, ``weight`` and
    ``skipped``, besides any keys the verifier gave it. The two fields here are written only for a large output."""

    verifier_output_ref: dict[str, Any]
    verifier_output_offload_error: str


@dataclass
class Verification:
    """What one run of a verifier found.

    ``criteria``: a row per check, read for ``id``, ``score``, ``result``, ``weight`` and ``skipped``, any other keys
    kept. ``score``: the verifier's own score when it isn't the step's aggregate of the rows, which it then replaces.
    ``output``: anything else worth keeping, such as a judge's justification or a test's log. ``legacy_entries``: the
    ``verifications`` entries to write, by key, when they aren't ``{verifier_id: {"results", "score"}}``. ``skipped``:
    why the verifier judged nothing. ``raise_after``: an exception the step raises once both are written, for a verdict
    that also fails its step."""

    criteria: list[dict[str, Any]] = field(default_factory=list)
    score: Optional[float] = None
    output: Any = None
    legacy_entries: Optional[dict[str, dict[str, Any]]] = None
    skipped: Optional[str] = None
    raise_after: Optional[BaseException] = None


class VerifierTaskStep(TaskStep):
    """A step that grades a run. ``execute`` runs ``verify`` and writes the step's grade, and the legacy entries after
    it; an exception from ``verify`` is a failed grade, raised again. Subclasses implement ``verify``, not ``execute``,
    and their ``__init__`` takes and passes on ``verifier_id``, ``score_aggregator`` and ``min_passing_score``, which
    ``_base_from_dict`` reads for them."""

    DEFAULT_SCORE_AGGREGATOR: ClassVar[ScoreAggregator] = ScoreAggregator.ALL_PASS

    def __init__(
        self,
        id: str,
        version: Optional[int],
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        retry_config: Optional[RetryConfig | dict] = None,
        *,
        verifier_id: Optional[str] = None,
        score_aggregator: Optional[ScoreAggregator | str] = None,
        min_passing_score: Optional[float] = None,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         retry_config=retry_config)
        self.verifier_id = verifier_id or id
        self.score_aggregator = ScoreAggregator(score_aggregator) if score_aggregator else self.DEFAULT_SCORE_AGGREGATOR
        self.min_passing_score = DEFAULT_MIN_PASSING_SCORE if min_passing_score is None else float(min_passing_score)

    @abstractmethod
    async def verify(self, context: TaskStepContext) -> Verification:
        """Grade the run in ``context``."""

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        try:
            verification = await self.verify(context)
        except Exception as exc:
            self._write_grade(context, self._grade(context, VerifierStatus.FAILED, 0.0, [],
                                                   error=f"{type(exc).__name__}: {exc}"))
            raise
        if verification.skipped is not None:
            status, score = VerifierStatus.SKIPPED, None
        elif verification.score is not None:
            status, score = VerifierStatus.GRADED, verification.score
        else:
            status, score = VerifierStatus.GRADED, aggregate_score(verification.criteria, self.score_aggregator)
        grade = self._grade(context, status, 0.0 if score is None else float(score), verification.criteria,
                            skip=verification.skipped)
        grade.update(await self._output(context, verification.output))
        self._write_grade(context, grade)
        entries = verification.legacy_entries
        if entries is None:
            entry = {"results": verification.criteria, "score": score}
            if verification.skipped is not None:
                entry["skipped"] = verification.skipped
            entries = {self.verifier_id: entry}
        context.metadata.setdefault(VERIFICATIONS_KEY, {}).update(entries)
        if verification.raise_after is not None:
            raise verification.raise_after
        return context

    def verifier_settings(self, context: TaskStepContext) -> dict[str, Any]:
        """What this step grades with in ``context``'s run: its own fields, one over ``SETTING_INLINE_BYTES`` kept as its
        sha256 and size. A subclass adds what the run changes or decides, such as the ``step_params`` it applies or a
        judge's model. Pure, since it is also read at run start."""
        return {k: _compact(v) for k, v in self.to_dict().items() if k not in _NOT_SETTINGS}

    def config_problems(self) -> list[str]:
        problems = super().config_problems()
        if not 0.0 < self.min_passing_score <= 1.0:
            problems.append(f"min_passing_score must be above 0 and at most 1, not {self.min_passing_score}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        if self.verifier_id != self.id:
            base["verifier_id"] = self.verifier_id
        if self.score_aggregator is not self.DEFAULT_SCORE_AGGREGATOR:
            base["score_aggregator"] = self.score_aggregator.value
        if self.min_passing_score != DEFAULT_MIN_PASSING_SCORE:
            base["min_passing_score"] = self.min_passing_score
        return base

    @classmethod
    def _base_from_dict(cls, data: dict) -> dict:
        return {
            **super()._base_from_dict(data),
            "verifier_id": data.get("verifier_id"),
            "score_aggregator": data.get("score_aggregator"),
            "min_passing_score": data.get("min_passing_score"),
        }

    def _grade(self, context: TaskStepContext, status: VerifierStatus, score: float, criteria: list[dict[str, Any]],
               *, error: Optional[str] = None, skip: Optional[str] = None) -> Grade:
        passed = status is VerifierStatus.GRADED and score >= self.min_passing_score
        rows = [_row(i, row) for i, row in enumerate(criteria)] or [{
            "id": self.id, "score": score, "result": passed, "weight": 1.0, "skipped": status is VerifierStatus.SKIPPED,
        }]
        return Grade(
            name=self.id,
            score=score,
            min_passing_score=self.min_passing_score,
            passed=passed,
            verifier_status=status.value,
            verifier_error_reasoning=error,
            verifier_skip_reasoning=skip,
            verifier_output=None,
            verifier_criteria=rows,
            verifier_type=self.type,
            verifier_version=_providing_version(type(self).__module__),
            verifier_settings={"scoring_rules": SCORING_RULES, **self.verifier_settings(context)},
        )

    async def _output(self, context: TaskStepContext, output: Any) -> dict[str, Any]:
        """``verifier_output`` inline, or, over ``OUTPUT_INLINE_BYTES``, a ``verifier_output_ref`` to its copy in the
        object store, written on a worker thread; inline with ``verifier_output_offload_error`` when that write fails,
        which never fails the step."""
        if output is None:
            return {}
        body = json.dumps(output).encode()
        if len(body) <= OUTPUT_INLINE_BYTES:
            return {"verifier_output": output}
        try:
            url = await finish_on_thread(functools.partial(_store_output, context.instance_id, self.id, body),
                                         f"Uploading verifier {self.id}'s output")
        except Exception as exc:
            logger.warning("verifier %s: kept its %d-byte output inline, the object store refused it: %s",
                           self.id, len(body), exc)
            return {"verifier_output": output, "verifier_output_offload_error": f"{type(exc).__name__}: {exc}"}
        return {"verifier_output_ref": {"object_url": url, "bytes": len(body),
                                        "sha256": hashlib.sha256(body).hexdigest()}}

    def _write_grade(self, context: TaskStepContext, grade: Grade) -> None:
        context.metadata.setdefault(GRADES_KEY, {})[self.id] = grade
        logger.info("verifier %s: %s, score %s, %s", self.id, grade["verifier_status"], grade["score"],
                    "passed" if grade["passed"] else "not passed")


def _row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    """``row`` with the keys every grade's rows carry, each default chosen so ``aggregate_score`` reads the row as it
    reads the original: ``result`` is never filled in, since ALL_PASS counts only an explicit ``True``."""
    return {
        **row,
        "id": row.get("id", str(index)),
        "score": row.get("score", 1.0 if row.get("result") else 0.0),
        "result": row.get("result"),
        "weight": row.get("weight", 1.0),
        "skipped": row.get("skipped", False),
    }


def _compact(value: Any) -> Any:
    body = canonical_json(value)
    if len(body) <= SETTING_INLINE_BYTES:
        return value
    return {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def _store_output(instance_id: Optional[str], step_id: str, body: bytes) -> str:
    config = get_config()
    key = (f"{config.get_artifact_key_prefix()}verifier-outputs/instance_id={instance_id}/"
           f"{step_id}-{uuid.uuid4().hex}.json")
    return config.get_object_store().put(key, body, content_type="application/json")


@functools.cache
def _providing_version(module: str) -> Optional[str]:
    """The version of the installed package that holds ``module``: agentenv-framework's for agent_env's own; None when
    no one package does, as for a module on the path or an editable plugin."""
    if module.partition(".")[0] == "agent_env":
        return _requirements.installed(_FRAMEWORK)
    name = _requirements.distribution_holding(module, packages_distributions())
    return _requirements.installed(canonicalize_name(name)) if name else None
