"""Verify response-side rubric criteria against an agent's prompt response text.

Deterministic checks that operate on the agent's final text response — no
sandbox connection, no LLM judge:
  - response_contains          — every needle in `needles` appears in the response
  - response_regex_present     — `pattern` (regex) matches the response (re.search)

Strict: unknown criterion types and criteria missing their required field
(empty `needles` for response_contains, empty `pattern` for
response_regex_present) raise ValueError. A verifier that can't actually
verify anything is a misconfiguration, not a runtime decision.

Result rows merge `{**criterion, **outcome}` and aggregated score is written
to `context.metadata["verifications"][verifier_id]`, matching the shape used
by `VerifySandboxTaskStep` and `RubricsVerifierTaskStep`.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score

logger = logging.getLogger(__name__)

_HANDLED_TYPES = {
    "response_contains",
    "response_regex_present",
}


class AgentPromptResponseVerifierTaskStep(TaskStep):
    type: ClassVar[str] = "agent_prompt_response_verifier"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: Optional[int],
        prompt_id: str,
        criteria: Optional[list[dict]] = None,
        score_aggregator: Optional[ScoreAggregator] = None,
        verifier_id: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.prompt_id = prompt_id
        self.criteria = list(criteria or [])
        if isinstance(score_aggregator, str):
            score_aggregator = ScoreAggregator(score_aggregator)
        self.score_aggregator = score_aggregator or ScoreAggregator.ALL_PASS
        self.verifier_id = verifier_id or uuid.uuid4().hex

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["prompt_id"] = self.prompt_id
        base["criteria"] = self.criteria
        base["score_aggregator"] = self.score_aggregator.value
        base["verifier_id"] = self.verifier_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "AgentPromptResponseVerifierTaskStep":
        raw_agg = data.get("score_aggregator")
        return cls(
            **cls._base_from_dict(data),
            prompt_id=data["prompt_id"],
            criteria=data.get("criteria"),
            score_aggregator=ScoreAggregator(raw_agg) if raw_agg else None,
            verifier_id=data.get("verifier_id"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        prompt_response = next(
            (pr for pr in context.prompt_responses if pr.prompt_id == self.prompt_id),
            None,
        )
        if prompt_response is None:
            raise RuntimeError(f"PromptResponse with prompt_id='{self.prompt_id}' not found in context")

        if prompt_response.error_type:
            logger.warning(
                f"Skipping verification '{self.verifier_id}': prompt had error_type={prompt_response.error_type}"
            )
            context.metadata.setdefault("verifications", {})[self.verifier_id] = {
                "results": [{
                    "id": "prompt_error",
                    "score": 0,
                    "result": False,
                    "message": f"Skipped: prompt had error_type={prompt_response.error_type}",
                }],
                "score": 0,
            }
            return context

        response = prompt_response.response or ""

        results: list[dict] = []
        for idx, criterion in enumerate(self.criteria):
            rtype = criterion.get("type")
            if rtype not in _HANDLED_TYPES:
                raise ValueError(
                    f"criterion #{idx} has unknown type {rtype!r}; "
                    f"handled types: {sorted(_HANDLED_TYPES)}"
                )
            outcome = self._eval_criterion(idx, response, criterion)
            results.append({
                **criterion,
                "criterion_index": idx,
                "score": float(outcome["score"]),
                "result": bool(outcome["passed"]),
                "justification": outcome["justification"],
            })

        score = aggregate_score(results, self.score_aggregator)

        context.metadata.setdefault("verifications", {})[self.verifier_id] = {
            "results": results,
            "score": score,
        }
        logger.info(
            f"Verification '{self.verifier_id}': "
            f"{len(results)} criteria evaluated, score={score}"
        )
        return context

    def _eval_criterion(self, idx: int, response: str, criterion: dict) -> dict:
        rtype = criterion["type"]
        if rtype == "response_contains":
            needles = criterion.get("needles") or []
            if not needles:
                raise ValueError(
                    f"criterion #{idx} (response_contains) requires non-empty `needles`"
                )
            missing = [n for n in needles if n not in response]
            passed = not missing
            return {
                "score": 1.0 if passed else 0.0,
                "passed": passed,
                "justification": "all needles present" if passed else f"missing: {missing}",
            }
        if rtype == "response_regex_present":
            pattern = criterion.get("pattern")
            if not pattern:
                raise ValueError(
                    f"criterion #{idx} (response_regex_present) requires non-empty `pattern`"
                )
            flags_raw = criterion.get("flags")
            if flags_raw is None:
                flags = re.DOTALL
            else:
                flags = 0
                for name in flags_raw:
                    flags |= getattr(re, name)
            passed = bool(re.search(pattern, response, flags))
            return {
                "score": 1.0 if passed else 0.0,
                "passed": passed,
                "justification": "pattern matched" if passed else f"no match for pattern {pattern!r}",
            }
        raise RuntimeError(f"unhandled criterion type {rtype}")
