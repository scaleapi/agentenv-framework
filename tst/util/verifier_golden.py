"""The legacy ``verifications`` entries built-in verifiers left on main before they moved onto ``VerifierTaskStep``,
recorded under ``tst/data/verifier_golden``: a run must leave the same entries, number types included, the same
models record and the same error."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Optional

from agent_env.task_step.context import TaskStepContext

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
