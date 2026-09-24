"""install_agent must register the installed agent in the instance store —
context records with instance_id None broke every store-resolving consumer
(add_skills raised "missing instance_id"; the a2a-agent CLIs could not see
the agent at all)."""

import dataclasses
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from agent_env.a2a_agent.a2a_agent import DeployedA2AAgent
from agent_env.task_step.context import DeployedSandbox
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep


class _FakeInstanceStore:
    def __init__(self):
        self.created = []

    def create_instance(self, deployed: DeployedA2AAgent, ttl_seconds: int) -> DeployedA2AAgent:
        self.created.append((deployed, ttl_seconds))
        return dataclasses.replace(deployed, instance_id=f"{deployed.agent_id}-abcd1234")


def _step() -> InstallAgentTaskStep:
    return InstallAgentTaskStep(id="i", version=None, sandbox_name="sb", a2a_agent_id="claude-code-cli")


def _sandbox_record(expires_at_utc=None) -> DeployedSandbox:
    return DeployedSandbox(
        sandbox_name="sb",
        sandbox_id="sb-1",
        sandbox_mode="vm",
        sandbox_type="modal_vm",
        expires_at_utc=expires_at_utc,
    )


def test_installed_agent_is_registered_in_the_instance_store():
    agent = MagicMock()
    agent.id = "claude-code-cli"
    agent.version = 7
    card = {"name": "Claude Code"}
    store = _FakeInstanceStore()

    with patch(
        "agent_env.a2a_agent.store.get_a2a_agent_instance_store",
        return_value=store,
    ):
        deployed = _step()._register_instance(agent, "https://a2a.example", _sandbox_record(), card)

    assert deployed.instance_id == "claude-code-cli-abcd1234"
    (created, _ttl), = store.created
    assert created.agent_id == "claude-code-cli"
    assert created.agent_version == 7
    assert created.a2a_url == "https://a2a.example"
    assert created.sandbox_id == "sb-1"
    assert created.sandbox_type == "modal_vm"
    assert created.agent_card == card


def test_instance_ttl_mirrors_the_sandbox_expiry():
    expires = datetime.now(timezone.utc) + timedelta(minutes=90)
    ds = _sandbox_record(expires_at_utc=expires.strftime("%Y-%m-%d %H:%M UTC"))
    ttl = InstallAgentTaskStep._instance_ttl_seconds(ds)
    # strftime drops the seconds, so allow one minute of slack on either side.
    assert 90 * 60 - 60 <= ttl <= 90 * 60 + 60


def test_instance_ttl_defaults_when_the_sandbox_has_no_expiry():
    assert InstallAgentTaskStep._instance_ttl_seconds(_sandbox_record()) == TaskStep.DEFAULT_TTL_SECONDS


def test_instance_ttl_defaults_on_a_malformed_expiry():
    ds = _sandbox_record(expires_at_utc="not a timestamp")
    assert InstallAgentTaskStep._instance_ttl_seconds(ds) == TaskStep.DEFAULT_TTL_SECONDS


def test_instance_ttl_floors_at_sixty_seconds_for_an_expired_record():
    expired = datetime.now(timezone.utc) - timedelta(hours=2)
    ds = _sandbox_record(expires_at_utc=expired.strftime("%Y-%m-%d %H:%M UTC"))
    assert InstallAgentTaskStep._instance_ttl_seconds(ds) == 60
