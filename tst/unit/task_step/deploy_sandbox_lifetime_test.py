"""A deploy_sandbox step records when its sandbox was made and when its TTL runs out, the run's override of it
included: what install_agent puts on the sandbox lives as long as it does."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep


def _provider():
    sandbox = AsyncMock()
    sandbox.sandbox_id, sandbox.mode, sandbox.type = "sb-1", "vm", "modal_vm"
    sandbox.tunnel_urls, sandbox.vnc_url, sandbox.network_policy = {}, None, NetworkPolicy()
    provider = AsyncMock(ON_THIS_MACHINE=False)
    provider.links = (provider,)
    provider.create_vm.return_value = sandbox
    return provider


def _utc(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize("override, lifetime", [(None, 3600), (21600, 21600)], ids=["step", "run override"])
async def test_the_record_says_when_the_sandbox_runs_out(override, lifetime):
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="vm", sandbox_type="any",
                                 ttl_seconds=3600)
    metadata = {"user_overrides": {"ttl_seconds": override}} if override else {}
    before = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider",
               return_value=_provider()):
        context = await step.execute(TaskStepContext(metadata=metadata))

    [record] = context.deployed_sandboxes
    made = _utc(record.created_at_utc)
    assert before <= made <= datetime.now(timezone.utc)
    assert _utc(record.expires_at_utc) - made == timedelta(seconds=lifetime)
