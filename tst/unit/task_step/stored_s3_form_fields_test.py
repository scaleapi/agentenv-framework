"""Stored step documents written before the S3 transfer forms were removed: a field that only
named an S3 location is dropped, an S3-form changelog source included (none is left in stored tasks)."""

import pytest

from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_skill_config import (
    VerifyA2ASkillConfigStep,
)
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep


def test_a_stored_skill_config_check_with_an_s3_probe_url_still_loads():
    step = VerifyA2ASkillConfigStep(id="v", version=None, a2a_agent_id="a", rubric_verifier_id="r")
    document = step.to_dict()
    assert VerifyA2ASkillConfigStep.from_dict({**document, "skill_s3_url": "s3://bucket/prefix/"}).to_dict() == document


@pytest.mark.parametrize("prefix", ["s3://bucket/agent_changelog/run-1/solver", None], ids=["set", "empty"])
def test_a_stored_deploy_step_with_an_s3_changelog_field_loads_without_it(prefix):
    step = DeployAgentTaskStep(id="d", version=None)
    document = step.to_dict()
    assert DeployAgentTaskStep.from_dict({**document, "agent_changelog_s3_prefix": prefix}).to_dict() == document
