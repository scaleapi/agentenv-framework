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
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from agent_env.task_step.task_steps.verifiers.verifier import Verification, VerifierTaskStep

logger = logging.getLogger(__name__)


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
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         verifier_id=verifier_id or uuid.uuid4().hex, score_aggregator=score_aggregator,
                         min_passing_score=min_passing_score)
        self.verifier_ids = list(verifier_ids or [])

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["verifier_ids"] = self.verifier_ids
        base["score_aggregator"] = self.score_aggregator.value
        base["verifier_id"] = self.verifier_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "AggregateVerifiersTaskStep":
        return cls(**cls._base_from_dict(data), verifier_ids=data.get("verifier_ids"))

    async def verify(self, context: TaskStepContext) -> Verification:
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
