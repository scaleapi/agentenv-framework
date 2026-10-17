"""Idle timeout resolution: the run's ``user_overrides['idle_timeout_seconds']`` wins over the step's own value, else
falls back to it, and an unset one is never passed on, so every sandbox keeps today's no-idle-limit default. An env that
doesn't take one deploys without, with a warning; an env provider is held to the usual rule for the options it names."""

from __future__ import annotations

import dataclasses
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent, DeployedA2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.envs._deployment import provider_options
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.providers.env_providers.env_provider import EnvironmentProvider
from agent_env.providers.sandbox_providers.sandbox_provider import Accepts
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep


class _Stop(Exception):
    pass


_RESOLUTIONS = [
    pytest.param(600, {"idle_timeout_seconds": 900}, 900, id="override-wins"),
    pytest.param(600, {}, 600, id="step-value"),
    pytest.param(None, {"idle_timeout_seconds": 900}, 900, id="override-only"),
    pytest.param(None, {}, None, id="unset"),
]


def _record() -> DeployedGatewayEnv:
    return DeployedGatewayEnv(env_id="art-1", env_version=2, gateway_url="http://gw", mcp_url="http://mcp",
                              db_web_url=None, sandbox_id="s1", metadata=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("step_idle, overrides, expected", _RESOLUTIONS)
async def test_deploy_env_hands_the_env_the_resolved_idle_timeout(step_idle, overrides, expected, caplog):
    env = MagicMock()
    env.deploy = AsyncMock(return_value=_record())
    step = DeployEnvTaskStep(id="d", version=None, env_id="art-1", env_version=2, idle_timeout_seconds=step_idle)
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(TaskStepContext(metadata={"user_overrides": overrides}))
    assert env.deploy.await_args.kwargs.get("idle_timeout_seconds") == expected
    assert ("idle_timeout_seconds" in env.deploy.await_args.kwargs) is (expected is not None)
    assert (f"idle_timeout requested={expected} applied=None" in caplog.text) is (expected is not None)


@pytest.mark.asyncio
async def test_deploy_env_records_and_logs_the_idle_timeout_its_env_applied(caplog):
    env = MagicMock()
    env.deploy = AsyncMock(return_value=dataclasses.replace(_record(), idle_timeout_seconds=900))
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        context = await DeployEnvTaskStep(id="d", version=None, env_id="art-1", idle_timeout_seconds=900).execute(
            TaskStepContext())
    assert context.deployed_envs[0].idle_timeout_seconds == 900
    assert "Env 'art-1' idle_timeout requested=900 applied=900" in caplog.text


class _EnvWithoutIdleTimeout:
    """An env whose deploy() names its options and predates the idle timeout, as plugin envs do."""

    def __init__(self):
        self.options: dict = {}

    async def deploy(self, ttl_seconds: int = 10800, disk_size_gb: float = 30, gateway_mode=None, cpu=None, memory_mb=None,
                     *, attribution=None):
        self.options = dict(ttl_seconds=ttl_seconds, attribution=attribution)
        return _record()


@pytest.mark.asyncio
async def test_deploy_env_deploys_an_env_that_takes_no_idle_timeout_without_one_and_warns(caplog):
    env = _EnvWithoutIdleTimeout()
    step = DeployEnvTaskStep(id="d", version=None, env_id="art-1", idle_timeout_seconds=900)
    with patch("agent_env.env.env.Env") as Env, \
         patch("agent_env.task_step.task_steps.deploy_env.update_env_instance_metadata"):
        Env.get.return_value = env
        await step.execute(TaskStepContext())
    assert env.options["ttl_seconds"] == step.ttl_seconds
    assert "doesn't take idle_timeout_seconds; it deploys without" in caplog.text


@pytest.mark.parametrize("step", [
    DeployEnvTaskStep(id="e", version=None, env_id="art-1", idle_timeout_seconds=900),
    DeployAgentTaskStep(id="a", version=None, idle_timeout_seconds=900),
    DeploySandboxTaskStep(id="s", version=None, sandbox_name="sbx", sandbox_mode="vm", idle_timeout_seconds=900),
], ids=["deploy_env", "deploy_agent", "deploy_sandbox"])
def test_a_steps_idle_timeout_round_trips_and_a_stored_step_without_one_has_none(step):
    assert type(step).from_dict(step.to_dict()).idle_timeout_seconds == 900
    stored = {k: v for k, v in step.to_dict().items() if k != "idle_timeout_seconds"}
    assert type(step).from_dict(stored).idle_timeout_seconds is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["vm", "container"])
@pytest.mark.parametrize("step_idle, overrides, expected", _RESOLUTIONS)
async def test_deploy_sandbox_creates_its_sandbox_with_the_resolved_idle_timeout(mode, step_idle, overrides, expected):
    sb = MagicMock(sandbox_id="s1", mode=mode, type="remote", tunnel_urls={}, vnc_url=None, network_policy=None,
                   idle_timeout=expected)
    provider = MagicMock(links=(), create_vm=AsyncMock(return_value=sb), create_container=AsyncMock(return_value=sb))
    provider.links = (provider,)
    step = DeploySandboxTaskStep(id="d", version=None, sandbox_name="sbx", sandbox_mode=mode, image="public.ecr.aws/x/y:1",
                                 port=8080, idle_timeout_seconds=step_idle)
    with patch("agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        context = await step.execute(TaskStepContext(metadata={"user_overrides": overrides}))
    create = provider.create_vm if mode == "vm" else provider.create_container
    assert create.await_args.kwargs.get("idle_timeout") == expected
    assert ("idle_timeout" in create.await_args.kwargs) is (expected is not None)
    assert context.deployed_sandboxes[0].idle_timeout_seconds == expected


@pytest.mark.asyncio
async def test_deploy_sandbox_records_none_when_its_provider_applied_no_idle_timeout(caplog):
    sb = MagicMock(sandbox_id="s1", mode="vm", type="remote", tunnel_urls={}, vnc_url=None, network_policy=None,
                   idle_timeout=None)
    provider = MagicMock(create_vm=AsyncMock(return_value=sb))
    step = DeploySandboxTaskStep(id="d", version=None, sandbox_name="sbx", sandbox_mode="vm", idle_timeout_seconds=900)
    with patch("agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        context = await step.execute(TaskStepContext())
    assert provider.create_vm.await_args.kwargs["idle_timeout"] == 900
    assert context.deployed_sandboxes[0].idle_timeout_seconds is None
    assert "Sandbox 'sbx' idle_timeout requested=900 applied=None" in caplog.text


def test_the_recorded_idle_timeouts_survive_a_stored_context():
    context = TaskStepContext(
        deployed_agents=[DeployedAgent(agent_name="a", api_url="https://agent.example", idle_timeout_seconds=900)],
        deployed_sandboxes=[DeployedSandbox(sandbox_name="sbx", sandbox_id="s1", sandbox_mode="vm", idle_timeout_seconds=900)],
    )
    restored = TaskStepContext.from_dict(context.to_dict())
    assert (restored.deployed_agents[0].idle_timeout_seconds, restored.deployed_sandboxes[0].idle_timeout_seconds) == (900, 900)


def test_an_agents_recorded_idle_timeout_survives_its_stored_record():
    deployed = DeployedA2AAgent(agent_id="a", agent_version=1, a2a_url="https://agent.example", sandbox_id="sb",
                                agent_card={}, idle_timeout_seconds=900)
    assert DeployedA2AAgent.from_dict(dataclasses.asdict(deployed)).idle_timeout_seconds == 900


@pytest.mark.asyncio
@pytest.mark.parametrize("step_idle, overrides, expected", _RESOLUTIONS)
async def test_deploy_agent_deploys_the_agent_with_the_resolved_idle_timeout(step_idle, overrides, expected):
    agent = MagicMock(metadata={})
    agent.deploy = AsyncMock(side_effect=_Stop)
    step = DeployAgentTaskStep(id="ag", version=None, a2a_agent_id="a", idle_timeout_seconds=step_idle)
    with patch.object(A2AAgent, "get", return_value=agent), pytest.raises(_Stop):
        await step.execute(TaskStepContext(metadata={"user_overrides": overrides}))
    assert agent.deploy.await_args.kwargs.get("idle_timeout_seconds") == expected
    assert ("idle_timeout_seconds" in agent.deploy.await_args.kwargs) is (expected is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize("applied", [900, None], ids=["applied", "not-applied"])
async def test_deploy_agent_records_the_idle_timeout_its_agent_applied(applied, caplog):
    agent = MagicMock(metadata={})
    agent.deploy = AsyncMock(return_value=MagicMock(a2a_url="https://agent.example", agent_card={}, sandbox_id="sb",
                                                    sandbox_type=None, instance_id="i", network_policy=None,
                                                    idle_timeout_seconds=applied))
    with patch.object(A2AAgent, "get", return_value=agent), \
         patch("agent_env.task_step.task_steps.deploy_agent.get_config") as get_config:
        get_config.return_value.get_model_for_role.return_value = None
        context = await DeployAgentTaskStep(id="ag", version=None, a2a_agent_id="a", idle_timeout_seconds=900).execute(
            TaskStepContext())
    assert context.deployed_agents[0].idle_timeout_seconds == applied
    assert f"idle_timeout requested=900 applied={applied}" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("applied", [900, None], ids=["applied", "not-applied"])
async def test_an_agents_record_carries_the_idle_timeout_its_sandbox_applied(applied):
    sandbox = MagicMock(sandbox_id="sb", mode="container", type="modal", tunnel_urls={8000: "https://agent.example"},
                        network_policy=None, idle_timeout=applied)
    provider = MagicMock(ON_THIS_MACHINE=False, SANDBOX_ACCEPTS=Accepts.NAME, url_from_sandbox=lambda url: url,
                         prepare_image=AsyncMock(), create_sandbox=AsyncMock(return_value=sandbox),
                         sandbox_problem=MagicMock(return_value=None))
    provider.links = (provider,)
    agent = A2AAgent(id="a", version=1, docker_image_artifact=DockerImageArtifact(
        id="img", description="t", image_name="public.ecr.aws/x/agent:1"))
    store = MagicMock(create_instance=lambda deployed, ttl: deployed)
    with patch("agent_env.providers.sandbox_providers.sandbox_provider.get_agent_sandbox_provider", return_value=provider), \
         patch.object(A2AAgent, "_wait_for_agent_card", AsyncMock(return_value={})), \
         patch("agent_env.a2a_agent.store.get_a2a_agent_instance_store", return_value=store):
        deployed = await agent.deploy(a2a_port=8000, idle_timeout_seconds=900,
                                      env_vars={"LITELLM_API_KEY": "k", "LITELLM_BASE_URL": "http://model"})
    assert deployed.idle_timeout_seconds == applied


@pytest.mark.asyncio
@pytest.mark.parametrize("idle", [900, None], ids=["set", "unset"])
async def test_an_agent_creates_its_sandbox_with_the_idle_timeout(idle):
    provider = MagicMock(ON_THIS_MACHINE=False, SANDBOX_ACCEPTS=Accepts.NAME, url_from_sandbox=lambda url: url,
                         prepare_image=AsyncMock(), create_sandbox=AsyncMock(side_effect=_Stop))
    provider.links = (provider,)
    agent = A2AAgent(id="a", version=None, docker_image_artifact=DockerImageArtifact(
        id="img", description="t", image_name="public.ecr.aws/x/agent:1"))
    with patch("agent_env.providers.sandbox_providers.sandbox_provider.get_agent_sandbox_provider", return_value=provider), \
         pytest.raises(_Stop):
        await agent.deploy(env_vars={"LITELLM_API_KEY": "k", "LITELLM_BASE_URL": "http://model"}, idle_timeout_seconds=idle)
    assert provider.create_sandbox.await_args.kwargs.get("idle_timeout") == idle
    assert ("idle_timeout" in provider.create_sandbox.await_args.kwargs) is (idle is not None)


class _NamedOptionsProvider(EnvironmentProvider):
    """A plugin's env provider whose deploy() names its options and predates the idle timeout."""

    type = "named_options"

    async def deploy(self, env, sandbox_provider, *, ttl_seconds: int = 10800, attribution: Optional[dict] = None):
        raise NotImplementedError

    async def close(self) -> None:
        pass


def _env() -> MCPServerEnv:
    return MCPServerEnv(id="mcp-email", version=1, docker_image_artifact=MagicMock(), environment_name="email",
                        env_provider_type="named_options")


def test_a_provider_that_names_its_options_and_not_the_idle_timeout_is_refused_it():
    with pytest.raises(ValueError, match=r"doesn't take idle_timeout_seconds; name it in the provider's deploy\(\)"):
        provider_options(_env(), _NamedOptionsProvider(), ttl_seconds=60, idle_timeout_seconds=900)


@pytest.mark.parametrize("make, attr, value", [
    (lambda: DeployEnvTaskStep("d", None, "art-1", 2, 3600, 20), "disk_size_gb", 20),
    (lambda: DeployAgentTaskStep("a", None, *([None] * 12), 3600, 2.0), "cpu", 2.0),
    (lambda: DeploySandboxTaskStep("s", None, "sbx", "vm", None, 1.0, 2048, 10, 3600, [8000]), "exposed_ports", [8000]),
    (lambda: DeployedAgent("a", "https://agent.example", None, None, None, None, None, None, None, True), "on_host", True),
], ids=["deploy_env", "deploy_agent", "deploy_sandbox", "deployed_agent"])
def test_the_idle_timeout_is_appended_so_a_positional_caller_keeps_its_meaning(make, attr, value):
    made = make()
    assert (getattr(made, attr), made.idle_timeout_seconds) == (value, None)
