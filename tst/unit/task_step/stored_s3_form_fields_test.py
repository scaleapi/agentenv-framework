"""Stored step documents written before the S3 transfer forms were removed still load, and drop
the key."""

import pytest

from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_skill_config import (
    VerifyA2ASkillConfigStep,
)
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep


@pytest.mark.parametrize("step, key", [
    (DeployAgentTaskStep(id="d", version=None), "agent_changelog_s3_prefix"),
    (
        VerifyA2ASkillConfigStep(id="v", version=None, a2a_agent_id="a", rubric_verifier_id="r"),
        "skill_s3_url",
    ),
], ids=["deploy_agent", "verify_a2a_skill_config"])
def test_a_stored_step_with_an_s3_form_field_still_loads(step, key):
    document = step.to_dict()
    assert type(step).from_dict({**document, key: "s3://bucket/prefix/"}).to_dict() == document
