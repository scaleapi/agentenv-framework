"""``capture_traffic``, off by default, from where a deploy is asked for to the provider: the deploy_env step and
``env deploy --capture-traffic`` pass it only when set, a website or multi env hands it to its provider, and a deploy
that can't capture is refused before anything is built: an env that serves no website, an env type whose deploy
doesn't take it, or a provider that doesn't."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_env.cli.env import env as env_cli
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.providers.env_providers.env_provider import EnvironmentProvider
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from tst.unit.task_step.deploy_env_test import _deployed_env, _env_with_fixed_deploy, _fake_env, _mcp_env


def _website(env_provider_type: str = "gateway") -> WebsiteEnv:
    return WebsiteEnv(id="notes-env", version=1, server_docker_image_artifact=MagicMock(), environment_name="notes",
                      env_provider_type=env_provider_type)


def _preflight(env, **options) -> list[str]:
    with patch("agent_env.env.env.Env.get", return_value=env):
        return DeployEnvTaskStep(id="d", version=None, env_id=env.id, **options).preflight()


@pytest.mark.parametrize("capture_traffic", [True, False], ids=["capturing", "plain"])
def test_capture_traffic_roundtrips_and_a_step_without_it_serializes_as_before(capture_traffic):
    step = DeployEnvTaskStep(id="t.deploy_env", version=None, env_id="notes-env", capture_traffic=capture_traffic)
    assert DeployEnvTaskStep.from_dict(step.to_dict()).capture_traffic is capture_traffic
    assert ("capture_traffic" in step.to_dict()) is capture_traffic
    assert DeployEnvTaskStep.from_dict({**step.to_dict(), "capture_traffic": None}).capture_traffic is False


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_traffic", [True, False], ids=["capturing", "plain"])
async def test_execute_asks_for_a_capture_only_for_a_capturing_step(capture_traffic):
    env = _fake_env(_deployed_env(None))
    step = DeployEnvTaskStep(id="t.deploy_env", version=None, env_id="art-1", capture_traffic=capture_traffic)
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(TaskStepContext())
    assert env.deploy.await_args.kwargs.get("capture_traffic") is (True if capture_traffic else None)


@pytest.mark.asyncio
async def test_execute_refuses_a_capture_for_an_env_whose_deploy_does_not_take_one():
    env = _env_with_fixed_deploy(_deployed_env(None))
    env.type = "fixed"
    step = DeployEnvTaskStep(id="d", version=None, env_id="art-1", capture_traffic=True)
    with patch("agent_env.env.env.Env") as Env, pytest.raises(ValueError, match="fixed envs take no capture_traffic"):
        Env.get.return_value = env
        await step.execute(TaskStepContext())


def test_preflight_passes_a_capture_of_a_website_env_behind_a_gateway():
    assert _preflight(_website(), capture_traffic=True) == []


def test_preflight_refuses_a_capture_of_an_env_that_serves_no_website():
    assert _preflight(_mcp_env("gateway"), capture_traffic=True) == [
        "deploy_env 'd': env 'mcp-email' serves no website, and capture_traffic records what a website's containers "
        "send and receive"]
    multi = MultiEnv(id="crm", version=1, mcp_server_envs=[_mcp_env("gateway")])
    assert "env 'crm' serves no website" in _preflight(multi, capture_traffic=True)[0]


class _NamedOptionsProvider(EnvironmentProvider):
    """A plugin's provider whose deploy names its options, and no capture_traffic among them."""

    type = "named-options"

    async def deploy(self, env, sandbox_provider, *, ttl_seconds: int = 10800, attribution=None):
        raise AssertionError("refused before deploying")

    async def close(self) -> None:
        pass


def test_preflight_refuses_a_capture_a_plugins_provider_does_not_take():
    with patch("agent_env.env.envs.website.provider_or_class", return_value=_NamedOptionsProvider):
        [refusal] = _preflight(_website("named-options"), capture_traffic=True)
    assert refusal.endswith("which doesn't take capture_traffic; name it in the provider's deploy() or take **options")


@pytest.mark.asyncio
@pytest.mark.parametrize("env", [_website(), MultiEnv(id="crm", version=1, mcp_server_envs=[], website_envs=[_website()])],
                         ids=["website", "multi"])
@pytest.mark.parametrize("capture_traffic", [True, False], ids=["capturing", "plain"])
async def test_a_website_or_multi_env_hands_a_capture_to_its_provider_only_when_asked(env, capture_traffic):
    module = "website" if isinstance(env, WebsiteEnv) else "multi_env"
    record = DeployedGatewayEnv(env_id=env.id, env_version=1, gateway_url="https://gw", sandbox_id="vm")
    with patch(f"agent_env.env.envs.{module}.deploy_through_provider", AsyncMock(return_value=record)) as through:
        await env.deploy(capture_traffic=capture_traffic)
    assert through.await_args.kwargs.get("capture_traffic") is (True if capture_traffic else None)


def _cli_deploy_kwargs(args: list[str]) -> dict:
    captured: dict = {}

    async def fake_deploy(**kwargs):
        captured.update(kwargs)
        return DeployedGatewayEnv(env_id="e1", env_version=1, gateway_url="http://gw", mcp_url="http://mcp",
                                  db_web_url=None, sandbox_id="s1")

    fake_env = MagicMock(id="e1", version=1, type="website")
    fake_env.deploy = fake_deploy
    with patch("agent_env.cli.env.deploy.Env") as Env, patch("agent_env.providers.build_sandbox_provider"):
        Env.get.return_value = fake_env
        result = CliRunner().invoke(env_cli, ["deploy", "--id", "e1", *args])
    assert result.exit_code == 0, result.output
    return captured


def test_env_deploy_captures_traffic_only_with_the_flag():
    assert _cli_deploy_kwargs(["--capture-traffic"])["capture_traffic"] is True
    assert "capture_traffic" not in _cli_deploy_kwargs([])
