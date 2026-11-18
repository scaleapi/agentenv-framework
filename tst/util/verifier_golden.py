"""The legacy ``verifications`` entries built-in verifiers left on main before they moved onto ``VerifierTaskStep``,
recorded under ``tst/data/verifier_golden``: a run must leave the same entries, number types included, the same
models record and the same error."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Optional

from agent_env.task_step.context import TaskStepContext, VerifierNotPassingError
from agent_env.task_step.task_steps.verifiers.verifier import VerifierTaskStep

GOLDEN = Path(__file__).parents[1] / "data" / "verifier_golden"


def observed(context: TaskStepContext, raised: Optional[BaseException]) -> dict[str, Any]:
    return {"verifications": context.metadata.get("verifications"), "models": context.metadata.get("models"),
            "raised": None if raised is None else f"{type(raised).__name__}: {raised}"}


@functools.cache
def _recorded(file: str) -> dict[str, Any]:
    return json.loads((GOLDEN / file).read_text())


def check(file: str, case: str, got: dict[str, Any]) -> None:
    expected = _recorded(file)[case]
    assert json.loads(json.dumps(got)) == expected
    assert json.dumps(got, sort_keys=True) == json.dumps(expected, sort_keys=True)


def check_grade(context: TaskStepContext, step: VerifierTaskStep, raised: Optional[BaseException]) -> None:
    """The step's named grade agrees with its legacy entry: failed when there is none, skipped when the entry is,
    else graded at the entry's score with its rows and the rest of the entry as its output; a gate fails the step
    only once both are written."""
    grade = context.metadata["grades"][step.id]
    entry = (context.metadata.get("verifications") or {}).get(step.verifier_id)
    if entry is None:
        assert (grade["verifier_status"], grade["score"], grade["passed"]) == ("failed", None, False)
        assert grade["verifier_error_reasoning"] == f"{type(raised).__name__}: {raised}"
        return
    assert grade["verifier_output"] == {k: v for k, v in entry.items() if k not in ("results", "score", "skipped")}
    if "skipped" in entry:
        assert (grade["verifier_status"], grade["score"], grade["passed"]) == ("skipped", None, False)
        assert grade["verifier_skip_reasoning"] == entry["skipped"]
        return
    assert grade["verifier_status"] == "graded" and grade["score"] == float(entry["score"])
    assert grade["passed"] is (entry["score"] >= step.min_passing_score)
    assert [row["id"] for row in grade["verifier_criteria"]] == [row["id"] for row in entry["results"]]
    if isinstance(raised, VerifierNotPassingError):
        assert not grade["passed"]
