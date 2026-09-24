"""Aggregate result rows across multiple upstream verifier steps into one unified score.

Reads `context.metadata["verifications"][<vid>]["results"]` for each id in
`verifier_ids`, concatenates the rows, drops skipped ones, and runs
`aggregate_score` once on the union — same math as the upstream steps, but
applied across their combined criterion set.

Use this to match Harbor's unified-aggregation behavior when filesystem
criteria (verify_sandbox) and response criteria (rubrics_verifier) live in
separate steps.
"""
from __future__ import annotations

import logging
import uuid
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score

logger = logging.getLogger(__name__)


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
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.verifier_ids = list(verifier_ids or [])
        if isinstance(score_aggregator, str):
            score_aggregator = ScoreAggregator(score_aggregator)
        self.score_aggregator = score_aggregator or ScoreAggregator.WEIGHTED_AVERAGE
        self.verifier_id = verifier_id or uuid.uuid4().hex

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["verifier_ids"] = self.verifier_ids
        base["score_aggregator"] = self.score_aggregator.value
        base["verifier_id"] = self.verifier_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "AggregateVerifiersTaskStep":
        raw_agg = data.get("score_aggregator")
        return cls(
            **cls._base_from_dict(data),
            verifier_ids=data.get("verifier_ids"),
            score_aggregator=ScoreAggregator(raw_agg) if raw_agg else None,
            verifier_id=data.get("verifier_id"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
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

        if "verifications" not in context.metadata:
            context.metadata["verifications"] = {}
        context.metadata["verifications"][self.verifier_id] = {
            "score": score,
            "source_verifier_ids": list(self.verifier_ids),
        }
        logger.info(
            f"AggregateVerifiers '{self.verifier_id}': merged {len(self.verifier_ids)} verifier(s), "
            f"{len(non_skipped)}/{len(merged_results)} criteria evaluated, score={score}"
        )
        return context
