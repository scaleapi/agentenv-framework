from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

CRITERIA = [{"id": "c1", "title": "t", "weight": 1}]


def _step(**kwargs) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(
        id="verify", version=None, criteria=CRITERIA, prompt_id="ask", **kwargs
    )


def test_judge_timeout_defaults_to_the_class_default():
    assert _step().judge_timeout_seconds == RubricsVerifierTaskStep.DEFAULT_JUDGE_TIMEOUT_SECONDS


def test_judge_timeout_override_round_trips():
    step = _step(judge_timeout_seconds=2400)
    assert step.judge_timeout_seconds == 2400
    assert step.to_dict()["judge_timeout_seconds"] == 2400
    assert RubricsVerifierTaskStep.from_dict(step.to_dict()).judge_timeout_seconds == 2400


def test_absent_key_falls_back_to_the_default():
    """Task docs written before this field existed must keep the old timeout."""
    data = {k: v for k, v in _step().to_dict().items() if k != "judge_timeout_seconds"}
    restored = RubricsVerifierTaskStep.from_dict(data)
    assert restored.judge_timeout_seconds == RubricsVerifierTaskStep.DEFAULT_JUDGE_TIMEOUT_SECONDS
