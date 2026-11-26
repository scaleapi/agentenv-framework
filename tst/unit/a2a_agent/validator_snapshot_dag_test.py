"""The validator's snapshot round-trip saves an agent no other check prompts.

A snapshot save waits out every turn its agent is running and captures that agent's workspace. When the token was
planted on the main agent, the modality probes' turns held the save back and their files rode in it: an agent that
installed a speech-to-text package for the audio probes made a workspace of gigabytes, and the save ran past the
capture's timeout.
"""

import pytest

from agent_env.a2a_agent import validator as validator_module
from agent_env.a2a_agent.validator import A2AAgentValidator, _UploadedFixtures
from agent_env.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.snapshot_agent_state import SnapshotAgentStateTaskStep


@pytest.fixture(autouse=True)
def no_validation_env_write(monkeypatch):
    monkeypatch.setattr(validator_module, "write_validation_env", lambda on_progress: None)


@pytest.mark.asyncio
async def test_the_snapshot_is_planted_on_and_saved_from_an_agent_of_its_own(monkeypatch):
    steps = await _validation_steps(monkeypatch)

    capture = next(s for s in steps if isinstance(s, SnapshotAgentStateTaskStep))
    plant = next(s for s in steps if isinstance(s, PromptAgentTaskStep) and s.prompt_id == capture.prompt_id)
    deploys = [s for s in steps if isinstance(s, DeployAgentTaskStep) and s.agent_name == capture.agent_name]
    prompted = [s for s in steps if isinstance(s, PromptAgentTaskStep) and s.agent_name == capture.agent_name]

    assert capture.agent_name != DeployAgentTaskStep.DEFAULT_AGENT_NAME
    assert plant.agent_name == capture.agent_name
    assert len(deploys) == 1 and [d.task_step_id for d in plant.depends_on] == [deploys[0].id]
    assert prompted == [plant]


async def _validation_steps(monkeypatch) -> list:
    """The steps ``A2AAgentValidator.validate`` builds, its modality probes included, with nothing uploaded or run."""
    captured: dict = {}

    class FakeTask:
        id = "validation-task"
        version = 1

        async def run(self, *, context: TaskStepContext, **_kwargs) -> TaskStepContext:
            return context

    class FakeAgent:
        id = "agent-x"
        version = 7
        VALIDATION_ENV_ID = "a2a-validation-env"

    def capture_task_put(**kwargs):
        captured["steps"] = kwargs["steps"]
        return FakeTask()

    async def noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(Task, "put", capture_task_put)
    monkeypatch.setattr(A2AAgentValidator, "_upload_skill_fixture", lambda *_args, **_kwargs: "s3://fixture/skill")
    monkeypatch.setattr(A2AAgentValidator, "_upload_probe_fixtures", lambda _agent, skill_object_url: _UploadedFixtures(
        skill_object_url=skill_object_url, png_object_uri="s3://fixture/red.png", mp4_object_uri="s3://fixture/clip.mp4"))
    monkeypatch.setattr(A2AAgentValidator, "_validate_agent_changelog", noop)
    monkeypatch.setattr(A2AAgentValidator, "_validate_clock", noop)
    monkeypatch.setattr(A2AAgentValidator, "_validate_install", noop)
    monkeypatch.setattr(A2AAgentValidator, "_cleanup_sandboxes", noop)

    await A2AAgentValidator.validate(FakeAgent())
    return captured["steps"]
