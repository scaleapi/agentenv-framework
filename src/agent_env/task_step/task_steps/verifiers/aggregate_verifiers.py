"""Aggregate result rows across multiple upstream verifier steps into one unified score.

Reads `context.metadata["verifications"][<vid>]["results"]` for each id in
`verifier_ids`, concatenates the rows, drops skipped ones, and runs
`aggregate_score` once on the union — same math as the upstream steps, but
applied across their combined criterion set. A named verifier with no entry fails
the step (and, with fail_task_on_error, the task).

Use this to match Harbor's unified-aggregation behavior when filesystem
criteria (verify_sandbox) and response criteria (rubrics_verifier) live in
separate steps.

Two settings, both off by default:

- ``use_verifier_scores``: aggregate the named verifiers' own recorded scores instead of their
  rows, each verifier one row that passes when its score is 1.0. Use it when a verifier's pass
  rule isn't its rows' (a rubrics_verifier scored with ``mandatory_all_pass`` passes with optional
  rows failed): with ``all_pass``, the run passes when every named verifier passed by its own rule.
  A verifier whose entry records no score (None) is left out, as its rows (it has none) are when
  merging.
- ``fail_task_when_ungraded``: a run whose named verifiers weren't all graded fails the step with
  ``UngradedVerifierError``, an error rather than a verdict, so the run is run or graded again
  instead of scored: a verifier with no entry, an entry marked ``graded: false`` (a rubrics_verifier
  or agent_prompt_response_verifier that skipped a run whose prompt errored) or recording no score
  (None), or a step this one depends on that failed in this run and wasn't retried, such as a
  judge with fail_task_on_error off (without depends_on, any step that failed before it ran). It
  needs fail_task_on_error.
"""
from __future__ import annotations

import logging
import uuid
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score

logger = logging.getLogger(__name__)

ERROR_SHOWN_CHARS = 300  # of a failed step's error, in UngradedVerifierError's message


class UngradedVerifierError(RuntimeError):
    """A verifier ``aggregate_verifiers`` combines wasn't graded (``fail_task_when_ungraded``): the run has no verdict
    to score, and is to be run or graded again."""


class AggregateVerifiersTaskStep(TaskStep):
    type: ClassVar[str] = "aggregate_verifiers"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: Optional[int],
        verifier_ids: Optional[list[str]] = None,
        score_aggregator: Optional[ScoreAggregator] = None,
        verifier_id: Optional[str] = None,
        use_verifier_scores: bool = False,
        fail_task_when_ungraded: bool = False,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if fail_task_when_ungraded and not fail_task_on_error:
            raise ValueError(
                f"aggregate_verifiers '{id}': fail_task_when_ungraded needs fail_task_on_error: a tolerated failure "
                "lets the steps after it run as though the run were scored."
            )
        self.verifier_ids = list(verifier_ids or [])
        if isinstance(score_aggregator, str):
            score_aggregator = ScoreAggregator(score_aggregator)
        self.score_aggregator = score_aggregator or ScoreAggregator.WEIGHTED_AVERAGE
        self.verifier_id = verifier_id or uuid.uuid4().hex
        self.use_verifier_scores = use_verifier_scores
        self.fail_task_when_ungraded = fail_task_when_ungraded

    def config_problems(self) -> list[str]:
        """``use_verifier_scores`` with ``mandatory_all_pass``, which reads a flag a verifier's own row never has."""
        problems = super().config_problems()
        if self.use_verifier_scores and self.score_aggregator is ScoreAggregator.MANDATORY_ALL_PASS:
            problems.append(
                f"aggregate_verifiers '{self.id}': use_verifier_scores makes each verifier one row, which is never "
                "marked mandatory, so mandatory_all_pass could never pass; use all_pass to require every verifier "
                "to pass by its own rule")
        return problems

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["verifier_ids"] = self.verifier_ids
        base["score_aggregator"] = self.score_aggregator.value
        base["verifier_id"] = self.verifier_id
        if self.use_verifier_scores:
            base["use_verifier_scores"] = True
        if self.fail_task_when_ungraded:
            base["fail_task_when_ungraded"] = True
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "AggregateVerifiersTaskStep":
        raw_agg = data.get("score_aggregator")
        return cls(
            **cls._base_from_dict(data),
            verifier_ids=data.get("verifier_ids"),
            score_aggregator=ScoreAggregator(raw_agg) if raw_agg else None,
            verifier_id=data.get("verifier_id"),
            use_verifier_scores=data.get("use_verifier_scores", False),
            fail_task_when_ungraded=data.get("fail_task_when_ungraded", False),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        if problems := self.config_problems():
            raise ValueError("; ".join(problems))
        all_verifications = context.metadata.get("verifications", {})
        if self.fail_task_when_ungraded and (ungraded := self._ungraded(context, all_verifications)):
            raise UngradedVerifierError(
                f"AggregateVerifiers '{self.verifier_id}': not scoring a run whose verifiers weren't all graded "
                f"({'; '.join(ungraded)}); run or grade it again"
            )
        missing = [vid for vid in self.verifier_ids if vid not in all_verifications]
        if missing:
            raise RuntimeError(
                f"AggregateVerifiers: verifier_id(s) not found in "
                f"context.metadata.verifications: {missing}. "
                f"Available: {list(all_verifications.keys())}"
            )

        source_scores: dict[str, Optional[float]] = {}
        if self.use_verifier_scores:
            source_scores = {vid: all_verifications[vid].get("score") for vid in self.verifier_ids}
            rows = [{"id": vid, "score": float(score), "result": score >= 1}
                    for vid, score in source_scores.items() if score is not None]
            merged_count = len(rows)
        else:
            merged_results: list[dict] = []
            for vid in self.verifier_ids:
                merged_results.extend(all_verifications[vid].get("results", []))
            rows = [r for r in merged_results if not r.get("skipped")]
            merged_count = len(merged_results)
        score = aggregate_score(rows, self.score_aggregator)

        if "verifications" not in context.metadata:
            context.metadata["verifications"] = {}
        entry: dict = {
            "score": score,
            "source_verifier_ids": list(self.verifier_ids),
        }
        if self.use_verifier_scores:
            entry["source_scores"] = source_scores
        context.metadata["verifications"][self.verifier_id] = entry
        logger.info(
            f"AggregateVerifiers '{self.verifier_id}': merged {len(self.verifier_ids)} verifier(s), "
            f"{len(rows)}/{merged_count} {'scores' if self.use_verifier_scores else 'criteria'} evaluated, "
            f"score={score}"
        )
        return context

    def _ungraded(self, context: TaskStepContext, verifications: dict) -> list[str]:
        """Why the run's named verifiers weren't all graded, one reason each; empty when they were."""
        reasons = []
        for vid in self.verifier_ids:
            entry = verifications.get(vid)
            if entry is None:
                reasons.append(f"'{vid}' has no entry")
            elif entry.get("graded") is False:
                reasons.append(f"'{vid}' didn't grade the run")
            elif entry.get("score") is None:
                reasons.append(f"'{vid}' recorded no score")
        depends = None if self.depends_on is None else {d.task_step_id for d in self.depends_on}
        for failed in context.metadata.get("failed_steps") or []:
            if not isinstance(failed, dict) or failed.get("retried") or failed.get("step_id") == self.id:
                continue
            if depends is None or failed.get("step_id") in depends:
                error = str(failed.get("error"))[:ERROR_SHOWN_CHARS]
                reasons.append(f"step '{failed.get('step_id')}' failed: {error}")
        return reasons
