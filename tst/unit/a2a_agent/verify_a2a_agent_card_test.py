"""Tests for A2A Agent Card validation."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_agent_card import (
    VerifyA2AAgentCardStep,
)


@pytest.mark.parametrize(
    ("include_skills", "skills", "expected_present"),
    [(True, [], True), (True, None, False), (False, None, False)],
)
def test_empty_skills_are_present_when_field_exists(
    include_skills: bool, skills: object, expected_present: bool
) -> None:
    card = {
        "name": "test-agent",
        "description": "test",
        "url": "/a2a",
        "version": "1",
        "capabilities": {"extensions": []},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
    }
    if include_skills:
        card["skills"] = skills

    context = TaskStepContext(
        deployed_agents=[
            DeployedAgent(
                agent_name="test-agent",
                api_url="http://agent.test",
                a2a_card=card,
            )
        ]
    )
    agent = MagicMock(metadata={})
    step = VerifyA2AAgentCardStep(
        id="verify-card",
        version=None,
        a2a_agent_id="test-agent",
    )

    with patch("agent_env.a2a_agent.A2AAgent.get", return_value=agent):
        asyncio.run(step.execute(context))

    validation = agent.update_metadata.call_args.args[0]["validated_agent_card"]
    assert validation["required_fields"]["skills"]["present"] is expected_present
