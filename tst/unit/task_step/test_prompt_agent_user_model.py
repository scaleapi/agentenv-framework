from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


def test_user_model_round_trips():
    step = PromptAgentTaskStep.from_dict(
        {"id": "s", "version": 1, "prompt": "hi", "user_model": "claude-sonnet-4-6"}
    )
    assert step.user_model == "claude-sonnet-4-6"
    assert step.to_dict()["user_model"] == "claude-sonnet-4-6"


def test_user_model_defaults_to_none():
    step = PromptAgentTaskStep.from_dict({"id": "s", "version": 1, "prompt": "hi"})
    assert step.user_model is None
