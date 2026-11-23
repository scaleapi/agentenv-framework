"""Verify response-side rubric criteria against an agent's prompt response text.

Deterministic checks that operate on the agent's final text response — no
sandbox connection, no LLM judge:
  - response_contains          — every needle in `needles` appears in the response
  - response_regex_present     — `pattern` (regex) matches the response (re.search)

Strict: unknown criterion types and criteria missing their required field
(empty `needles` for response_contains, empty `pattern` for
response_regex_present) raise ValueError. A verifier that can't actually
verify anything is a misconfiguration, not a runtime decision.

Result rows merge `{**criterion, **outcome}`. On the `VerifierTaskStep` base, which
aggregates them into the step's grade and writes the
`context.metadata["verifications"][verifier_id]` entry, in the shape used by
`VerifySandboxTaskStep` and `RubricsVerifierTaskStep`.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any, ClassVar, Optional

from agent_env.task_step.context import StepContextError, TaskStepContext
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator
from agent_env.task_step.task_steps.verifiers.verifier import Verification, VerifierTaskStep

logger = logging.getLogger(__name__)

_HANDLED_TYPES = {
    "response_contains",
    "response_regex_present",
}


class AgentPromptResponseVerifierTaskStep(VerifierTaskStep):
    type: ClassVar[str] = "agent_prompt_response_verifier"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: Optional[int],
        prompt_id: str,
        criteria: Optional[list[dict]] = None,
        score_aggregator: Optional[ScoreAggregator | str] = None,
        verifier_id: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        min_passing_score: Optional[float] = None,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         verifier_id=verifier_id or uuid.uuid4().hex, score_aggregator=score_aggregator,
                         min_passing_score=min_passing_score)
        self.prompt_id = prompt_id
        self.criteria = list(criteria or [])

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["prompt_id"] = self.prompt_id
        base["criteria"] = self.criteria
        base["score_aggregator"] = self.score_aggregator.value
        base["verifier_id"] = self.verifier_id
        return base

    def criterion_rows(self) -> Optional[int]:
        return len(self.criteria)

    def config_problems(self) -> list[str]:
        """The criteria a run would refuse when it grades: an unknown type, an empty ``needles`` or ``pattern``, a
        pattern that doesn't compile, or a flag ``re`` doesn't have."""
        problems = super().config_problems()
        for idx, criterion in enumerate(self.criteria):
            problems.extend(f"criterion #{idx}: {problem}" for problem in _criterion_problems(criterion))
        return problems

    @classmethod
    def from_dict(cls, data: dict) -> "AgentPromptResponseVerifierTaskStep":
        return cls(**cls._base_from_dict(data), prompt_id=data["prompt_id"], criteria=data.get("criteria"))

    async def verify(self, context: TaskStepContext) -> Verification:
        prompt_response = next(
            (pr for pr in context.prompt_responses if pr.prompt_id == self.prompt_id),
            None,
        )
        if prompt_response is None:
            raise StepContextError(f"PromptResponse with prompt_id='{self.prompt_id}' not found in context")

        if prompt_response.error_type:
            logger.warning(
                f"Skipping verification '{self.verifier_id}': prompt had error_type={prompt_response.error_type}"
            )
            return Verification(criteria=[{
                "id": "prompt_error",
                "score": 0,
                "result": False,
                "message": f"Skipped: prompt had error_type={prompt_response.error_type}",
            }], score=0)

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
        return Verification(criteria=results)

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


def _criterion_problems(criterion: Any) -> list[str]:
    if not isinstance(criterion, dict):
        return [f"must be an object, not {type(criterion).__name__}"]
    rtype = criterion.get("type")
    if not isinstance(rtype, str) or rtype not in _HANDLED_TYPES:
        return [f"unknown type {rtype!r}; handled types: {sorted(_HANDLED_TYPES)}"]
    if rtype == "response_contains":
        needles = criterion.get("needles")
        if isinstance(needles, list) and needles and all(isinstance(needle, str) for needle in needles):
            return []
        return [f"response_contains needs non-empty needles, a list of strings, not {needles!r}"]
    pattern = criterion.get("pattern")
    if not (isinstance(pattern, str) and pattern):
        return [f"response_regex_present needs a non-empty pattern, a string, not {pattern!r}"]
    names = criterion.get("flags")
    if names is not None and not (isinstance(names, list) and all(isinstance(name, str) for name in names)):
        return [f"flags must be a list of re flag names, not {names!r}"]
    if unknown := [name for name in names or () if not isinstance(getattr(re, name, None), re.RegexFlag)]:
        return [f"flags {unknown!r} aren't re flags"]
    flags = re.DOTALL if names is None else 0
    for name in names or ():
        flags |= getattr(re, name)
    try:
        re.compile(pattern, flags)
    except re.error as e:
        return [f"pattern {pattern!r} doesn't compile: {e}"]
    return []
