"""Parity between the two deploy entry points for the env-state param (``env_state_type``).

The CLI (`env deploy --env-state-type`) and the Task/Temporal path (`DeployEnvTaskStep`, fed by
`task run --env-state-type` via `user_overrides`) must funnel the SAME value into `env.deploy(...)`.
Both paths are exercised for real here; only the `env.deploy` seam is mocked to capture kwargs.

(Data-load takes no env-state param — state is acquired at deploy and rehydrated on reattach — so
there is nothing to keep in parity there.)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from agent_env.cli.env import env as env_cli
from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep


def _fake_deployed() -> DeployedEnv:
    return DeployedEnv(
        env_id="e1", env_version=1, gateway_url="http://gw", mcp_url="http://mcp",
        db_web_url=None, sandbox_id="s1",
    )


def _cli_deploy_kwargs(extra_args: list[str]) -> dict:
    """Run the real ``env deploy`` click command; return the kwargs it passes to ``env.deploy``."""
    captured: dict = {}

    async def fake_deploy(**kwargs):
        captured.update(kwargs)
        return _fake_deployed()

    fake_env = MagicMock(id="e1", version=1, type="multi")
    fake_env.deploy = fake_deploy
    with patch("agent_env.cli.env.deploy.Env") as Env, \
         patch("agent_env.providers.build_sandbox_provider"):
        Env.get.return_value = fake_env
        res = CliRunner().invoke(env_cli, ["deploy", "--id", "e1", *extra_args])
    assert res.exit_code == 0, res.output
    return captured


async def _taskrun_deploy_kwargs(*, step_env_state_type=None, user_overrides=None) -> dict:
    """Run the real ``DeployEnvTaskStep.execute`` (the Task.run deploy unit); return env.deploy kwargs."""
    env = MagicMock()
    env.deploy = AsyncMock(return_value=_fake_deployed())
    step = DeployEnvTaskStep(
        id="t.deploy_env", version=None, env_id="e1", env_version=1,
        env_state_type=step_env_state_type,
    )
    ctx = TaskStepContext(metadata={"user_overrides": user_overrides} if user_overrides else {})
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(ctx)
    return env.deploy.await_args.kwargs


def test_env_state_type_parity_when_set():
    """--env-state-type on the CLI, the step's own field, and a task-run user_override all reach
    env.deploy with the identical value."""
    cli = _cli_deploy_kwargs(["--env-state-type", "remote_postgres"])
    task_field = asyncio.run(_taskrun_deploy_kwargs(step_env_state_type="remote_postgres"))
    # user_overrides["env_state_type"] is exactly what `task run --env-state-type` sets.
    task_override = asyncio.run(_taskrun_deploy_kwargs(user_overrides={"env_state_type": "remote_postgres"}))

    assert cli["env_state_type"] == "remote_postgres"
    assert cli["env_state_type"] == task_field["env_state_type"] == task_override["env_state_type"]


def test_env_state_type_parity_when_unset():
    """Unset on both paths => env.deploy never receives env_state_type (so envs that don't accept
    it aren't broken) — the default behavior stays in parity too."""
    cli = _cli_deploy_kwargs([])
    task = asyncio.run(_taskrun_deploy_kwargs())

    assert "env_state_type" not in cli
    assert "env_state_type" not in task
