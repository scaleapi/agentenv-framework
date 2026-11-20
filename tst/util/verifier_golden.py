"""Golden files under ``tst/data/verifier_golden``: the built-in verifiers' legacy ``verifications`` entries, first
recorded on main before the verifiers moved onto ``VerifierTaskStep``, and the requests the direct judge hands LiteLLM
(``judge_requests.json``), first recorded on main before its call moved onto ``config.model_call``; each the contract
since. A run must leave what the file holds, number types and key order included. After a deliberate change, re-record
with ``AGENTENV_TEST_RECORD_GOLDEN=1`` and review the files' diff: each check then writes what the run left instead."""

from __future__ import annotations

import fcntl
import functools
import json
import os
from pathlib import Path
from typing import Any, Optional

from agent_env.config import get_config
from agent_env.task_step.context import TaskStepContext, VerifierNotPassingError
from agent_env.task_step.task_steps.verifiers.verifier import VerifierTaskStep

GOLDEN = Path(__file__).parents[1] / "data" / "verifier_golden"
RECORDING = os.environ.get("AGENTENV_TEST_RECORD_GOLDEN") == "1"


def observed(context: TaskStepContext, raised: Optional[BaseException]) -> dict[str, Any]:
    return {"verifications": context.metadata.get("verifications"), "models": context.metadata.get("models"),
            "raised": None if raised is None else f"{type(raised).__name__}: {raised}"}


def normalised(value: Any) -> Any:
    """``value`` with the object store's urls, which differ from run to run, under ``<objects>/``."""
    objects = get_config().get_object_store().object_url("x")[:-1]
    return json.loads(json.dumps(value).replace(objects, "<objects>/"))


@functools.cache
def _recorded(file: str) -> dict[str, Any]:
    return json.loads((GOLDEN / file).read_text())


def check(file: str, case: str, got: dict[str, Any]) -> None:
    """``got`` is what ``file`` holds for ``case``: the same values, number types and key order; written there
    instead when re-recording."""
    if RECORDING:
        _record(file, case, got)
        return
    expected = _recorded(file)[case]
    assert json.loads(json.dumps(got)) == expected
    assert json.dumps(got) == json.dumps(expected)


def _record(file: str, case: str, got: dict[str, Any]) -> None:
    """``case`` written into ``file`` under a lock, so parallel workers each add theirs."""
    with open(GOLDEN / file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        cases = json.loads(f.read())
        cases[case] = json.loads(json.dumps(got))
        f.seek(0)
        f.write(json.dumps(cases, indent=1) + "\n")
        f.truncate()


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
