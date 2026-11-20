"""Combine other verifier steps' results into one grade.

An aggregate names its sources by step id (``sources``) and reads their named grades. ``combine`` says how:
``pooled``, the default, pools their criterion rows and aggregates them once with ``score_aggregator``;
``weighted_mean`` takes a weighted mean of their scores, equal unless ``weights`` gives one for every source.
``gates`` are steps whose grades must pass, each by its own rule, or the aggregate scores 0; with no ``sources``,
an aggregate of gates alone scores 1 when they all pass. A failed or missing source or gate fails the aggregate; a
skipped source is left out, and every source skipped, or a gate skipped, skips it.

``verifier_ids``, the older form, names its sources by their ``verifier_id`` and pools the rows of their legacy
``verifications`` entries, as it always has: a source whose entry has no rows adds nothing. It takes none of the
newer fields.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, ClassVar, Optional, Sequence

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from agent_env.task_step.task_steps.verifiers.verifier import (
    GRADES_KEY,
    Grade,
    Verification,
    VerifierStatus,
    VerifierTaskStep,
)

logger = logging.getLogger(__name__)

POOLED = "pooled"
WEIGHTED_MEAN = "weighted_mean"
COMBINES = (POOLED, WEIGHTED_MEAN)
_NEWER_FIELDS = ("sources", "gates", "combine", "weights")


class AggregateVerifiersTaskStep(VerifierTaskStep):
    type: ClassVar[str] = "aggregate_verifiers"
    entity_refs = ()
    DEFAULT_SCORE_AGGREGATOR: ClassVar[ScoreAggregator] = ScoreAggregator.WEIGHTED_AVERAGE

    def __init__(
        self,
        id: str,
        version: Optional[int],
        verifier_ids: Optional[list[str]] = None,
        score_aggregator: Optional[ScoreAggregator | str] = None,
        verifier_id: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        min_passing_score: Optional[float] = None,
        sources: Optional[list[str]] = None,
        combine: Optional[str] = None,
        weights: Optional[dict[str, float]] = None,
        gates: Optional[list[str]] = None,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         verifier_id=verifier_id or uuid.uuid4().hex, score_aggregator=score_aggregator,
                         min_passing_score=min_passing_score)
        self.verifier_ids = list(verifier_ids or [])
        self.sources, self.combine, self.weights, self.gates = sources, combine, weights, gates

    @property
    def names_steps(self) -> bool:
        """Whether this aggregate takes the newer fields, naming its sources by step id, not by ``verifier_ids``."""
        return any(getattr(self, name) is not None for name in _NEWER_FIELDS)

    def source_steps(self, steps: Sequence[TaskStep]) -> list[str]:
        """The ids of the steps in ``steps`` whose results this aggregate reads, sources then gates: for the older
        form, the steps whose ``verifier_id`` each of its ``verifier_ids`` names, a rubric's trajectory-mistakes entry
        resolving to its rubric. An id no step writes is left out, as are sources or gates not given as a list of step
        ids, which are its config problems."""
        if self.names_steps:
            named = [name for field in (self.sources, self.gates) if isinstance(field, list) for name in field]
            return list(dict.fromkeys(name for name in named if isinstance(name, str)))
        by_verifier_id = {}
        for step in steps:
            verifier_id = getattr(step, "verifier_id", None)
            if isinstance(verifier_id, str):
                by_verifier_id.setdefault(verifier_id, step.id)
                by_verifier_id.setdefault(f"{verifier_id}-trajectory-mistakes", step.id)
        return list(dict.fromkeys(by_verifier_id[vid] for vid in self.verifier_ids if vid in by_verifier_id))

    def to_dict(self) -> dict:
        base = super().to_dict()
        if not self.names_steps:
            base["verifier_ids"] = self.verifier_ids
        for name in _NEWER_FIELDS:
            if getattr(self, name) is not None:
                base[name] = getattr(self, name)
        base["score_aggregator"] = self.score_aggregator.value
        base["verifier_id"] = self.verifier_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "AggregateVerifiersTaskStep":
        return cls(**cls._base_from_dict(data), verifier_ids=data.get("verifier_ids"), sources=data.get("sources"),
                   combine=data.get("combine"), weights=data.get("weights"), gates=data.get("gates"))

    def criterion_rows(self) -> Optional[int]:
        """One, the row its grade holds for itself, as a pooled aggregate of grades reads it; an aggregate of
        ``verifier_ids`` reading this one's legacy entry finds none."""
        return 1

    def verifier_settings(self, context: TaskStepContext) -> dict[str, Any]:
        settings = super().verifier_settings(context)
        if self.combine == WEIGHTED_MEAN:
            del settings["score_aggregator"]
        return settings

    def config_problems(self) -> list[str]:
        """An aggregate a run can't combine as written: the older ``verifier_ids`` beside the newer fields, nothing to
        combine, a ``combine`` it doesn't know, sources or gates that aren't a list of distinct step ids other than its
        own, and ``weights`` that aren't a number of at least 0 for every source of a ``weighted_mean``, one of them
        above 0. The steps a source names are lint's to check, as they're the task's."""
        problems = super().config_problems()
        if not self.names_steps:
            return problems if self.verifier_ids else [*problems, "it combines nothing: name sources or gates"]
        if self.verifier_ids:
            problems.append("verifier_ids is the older form and takes none of sources, gates, combine or weights; "
                            "name its sources by step id in sources")
        for name in ("sources", "gates"):
            problems.extend(_ids_problems(name, getattr(self, name), self.id))
        if not self.sources and not self.gates:
            problems.append("it combines nothing: name sources or gates")
        if self.combine is not None and self.combine not in COMBINES:
            problems.append(f"combine must be one of {list(COMBINES)}, not {self.combine!r}")
        if self.combine == WEIGHTED_MEAN and self.score_aggregator is not self.DEFAULT_SCORE_AGGREGATOR:
            problems.append("score_aggregator aggregates pooled rows, so a weighted_mean takes none")
        if self.weights is not None:
            problems.extend(self._weights_problems())
        return problems

    def _weights_problems(self) -> list[str]:
        if self.combine != WEIGHTED_MEAN:
            return ["weights are for a weighted_mean; pooled rows carry their own"]
        if not isinstance(self.weights, dict):
            return [f"weights must map each source's step id to a number, not {self.weights!r}"]
        problems = [f"weights gives {key!r} {weight!r}; a weight is a number of at least 0"
                    for key, weight in self.weights.items()
                    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0]
        sources = self.sources if isinstance(self.sources, list) else []
        if unknown := [key for key in self.weights if key not in sources]:
            problems.append(f"weights names {unknown!r}, which aren't in sources")
        if missing := [source for source in sources if source not in self.weights]:
            problems.append(f"weights gives none for {missing!r}: give every source a weight, or none for an equal "
                            "mean")
        if not problems and not any(self.weights.values()):
            problems.append("every weight is 0, so nothing counts")
        return problems

    async def verify(self, context: TaskStepContext) -> Verification:
        if not self.names_steps:
            return self._pooled_entries(context)
        grades = context.metadata.get(GRADES_KEY, {})
        sources = {step: _grade(grades, step, "source") for step in self.sources or ()}
        gates = {step: _grade(grades, step, "gate") for step in self.gates or ()}
        output: dict[str, Any] = {
            "combine": self.combine or POOLED,
            "sources": {step: _summary(grade) for step, grade in sources.items()},
            "gates": {step: _summary(grade) for step, grade in gates.items()},
        }
        if skipped := [step for step, grade in gates.items() if grade["verifier_status"] == VerifierStatus.SKIPPED]:
            return Verification(skipped=f"gate {skipped[0]!r} was skipped, so whether it passes isn't known",
                                output=output)
        graded = {step: grade for step, grade in sources.items() if grade["verifier_status"] == VerifierStatus.GRADED}
        if self.sources:
            score = self._weighted_mean(graded) if self.combine == WEIGHTED_MEAN else self._pooled_grades(graded)
            if score is None:
                return Verification(skipped="every source it weighs was skipped", output=output)
        else:
            score = 1.0
        if failed := [step for step, grade in gates.items() if not grade["passed"]]:
            output.update(ungated_score=score, failed_gates=failed)
            score = 0.0
        logger.info(f"AggregateVerifiers '{self.id}': {output['combine']} over {list(graded)} of "
                    f"{list(sources)}, gates {list(gates)}, score={score}")
        return Verification(score=score, output=output)

    def _pooled_grades(self, graded: dict[str, Grade]) -> Optional[float]:
        """The graded sources' rows aggregated once, each row's ``passed`` read back as the ``result`` that ALL_PASS and
        ANY_PASS count; None when every source was skipped."""
        if not graded:
            return None
        rows = [{"score": row["score"], "weight": row["weight"], "result": row["passed"]}
                for grade in graded.values() for row in grade["verifier_criteria"] if not row.get("skipped")]
        return aggregate_score(rows, self.score_aggregator)

    def _weighted_mean(self, graded: dict[str, Grade]) -> Optional[float]:
        """The graded sources' scores, weighted as ``weights`` says, else equally, over the weight they carry between
        them; None when they carry none, every weighted source having been skipped."""
        weight = {step: float(self.weights[step]) if self.weights else 1.0 for step in graded}
        total = sum(weight.values())
        if not total:
            return None
        return sum(weight[step] * grade["score"] for step, grade in graded.items()) / total

    def _pooled_entries(self, context: TaskStepContext) -> Verification:
        all_verifications = context.metadata.get("verifications", {})
        missing = [vid for vid in self.verifier_ids if vid not in all_verifications]
        if missing:
            raise RuntimeError(
                f"AggregateVerifiers: verifier_id(s) not found in "
                f"context.metadata.verifications: {missing}. "
                f"Available: {list(all_verifications.keys())}"
            )

        merged_results: list[dict] = []
        for vid in self.verifier_ids:
            merged_results.extend(all_verifications[vid].get("results", []))

        non_skipped = [r for r in merged_results if not r.get("skipped")]
        score = aggregate_score(non_skipped, self.score_aggregator)
        logger.info(
            f"AggregateVerifiers '{self.verifier_id}': merged {len(self.verifier_ids)} verifier(s), "
            f"{len(non_skipped)}/{len(merged_results)} criteria evaluated, score={score}"
        )
        return Verification(score=score, output={"source_verifier_ids": list(self.verifier_ids)})


def _grade(grades: dict[str, Grade], step: str, role: str) -> Grade:
    """``step``'s grade, which a source or gate must have, graded or skipped: a missing or failed one fails the
    aggregate."""
    grade = grades.get(step)
    if grade is None:
        raise RuntimeError(f"{role} {step!r} has no grade: its step didn't run before the aggregate, or isn't a "
                           "verifier on the VerifierTaskStep base")
    if grade["verifier_status"] == VerifierStatus.FAILED:
        raise RuntimeError(f"{role} {step!r} failed, so the aggregate can't combine it: "
                           f"{grade.get('verifier_error_reasoning')}")
    return grade


def _summary(grade: Grade) -> dict[str, Any]:
    return {"verifier_status": grade["verifier_status"], "score": grade["score"], "passed": grade["passed"]}


def _ids_problems(name: str, ids: Any, own_id: str) -> list[str]:
    if ids is None:
        return []
    if not isinstance(ids, list) or not all(isinstance(step, str) and step for step in ids):
        return [f"{name} must be a list of step ids, not {ids!r}"]
    problems = []
    if twice := sorted({step for step in ids if ids.count(step) > 1}):
        problems.append(f"{name} names {twice!r} more than once")
    if own_id in ids:
        problems.append(f"{name} names the aggregate itself")
    return problems
