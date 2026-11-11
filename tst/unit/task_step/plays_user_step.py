"""A deploy step whose class plays the user (``plays_user``), registered under a test type, for tests of the lookups
that tell the agent under test from the user sim's."""

from __future__ import annotations

from typing import ClassVar

import pytest

from agent_env.config.runtime import Config
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep


class PlaysUser(DeployAgentTaskStep):
    type: ClassVar[str] = "test_plays_user"
    plays_user: ClassVar[bool] = True


@pytest.fixture
def plays_user_registered(monkeypatch):
    """The task step registry with ``PlaysUser`` registered under its type."""
    steps = {**Config().task_step_registry(), PlaysUser.type: PlaysUser}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    return PlaysUser
