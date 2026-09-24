"""Per-run criteria override in RubricsVerifierTaskStep._resolved_criteria.

A ``step_params["<id>"].criteria`` override replaces the stored criteria
(replace-on-presence, like ``load_artifact``).
"""

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

STORED = [{"id": "s1", "criterion": "stored 1", "weight": 1}]
INJECTED = [
    {"id": "i1", "criterion": "injected 1", "weight": 5},
    {"id": "i2", "criterion": "injected 2", "weight": 3},
]


def _verifier(criteria):
    return RubricsVerifierTaskStep(
        id="verify", version=1, criteria=criteria, prompt_id="p1",
        use_agent_judge=False, use_trajectory=False, verifier_id="vt",
    )


def _ctx(step_params_for_verify=None):
    md = {}
    if step_params_for_verify is not None:
        md = {"user_overrides": {"step_params": {"verify": step_params_for_verify}}}
    return TaskStepContext(metadata=md)


def test_no_override_uses_stored_criteria():
    assert _verifier(STORED)._resolved_criteria(_ctx()) == STORED


def test_override_replaces_stored_criteria():
    assert _verifier(STORED)._resolved_criteria(_ctx({"criteria": INJECTED})) == INJECTED


def test_empty_override_replaces_not_reverts():
    # presence, not truthiness: an explicit [] means "no criteria", NOT "fall back to stored".
    assert _verifier(STORED)._resolved_criteria(_ctx({"criteria": []})) == []


def test_override_does_not_mutate_self():
    v = _verifier(STORED)
    out = v._resolved_criteria(_ctx({"criteria": INJECTED}))
    out.append({"id": "x"})
    assert v.criteria == STORED
    assert v._resolved_criteria(_ctx()) == STORED


def test_override_keyed_by_other_step_id_is_ignored():
    v = _verifier(STORED)
    ctx = TaskStepContext(
        metadata={"user_overrides": {"step_params": {"some-other-step": {"criteria": INJECTED}}}}
    )
    assert v._resolved_criteria(ctx) == STORED
