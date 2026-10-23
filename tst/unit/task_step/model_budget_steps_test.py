"""``model_budget_usd`` on ``deploy_agent`` and ``install_agent``: the agent gets a key of its own in place of the
shared one, the context records the budget, and a key made for an agent that fails to come up is deleted."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import agent_env.a2a_agent as a2a_agent_package
from agent_env.a2a_agent import A2AAgent
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep
from tst.util.fake_litellm import serve


@pytest.fixture
def litellm(monkeypatch, local_stores):
    return serve(monkeypatch, local_stores)


class _Agent:
    metadata: dict = {}
    default_env_vars: dict = {}

    def __init__(self) -> None:
        self.closed = 0
        self.env_vars: dict | None = None

    async def deploy(self, **kwargs):
        self.env_vars = kwargs["env_vars"]
        return SimpleNamespace(
            a2a_url="https://agent.example.test", agent_card={}, sandbox_id="sb-1", sandbox_type="fake",
            instance_id="i-1", network_policy=None,
        )

    async def close(self) -> None:
        self.closed += 1


@pytest.fixture
def agent(monkeypatch):
    agent = _Agent()
    monkeypatch.setattr(a2a_agent_package.A2AAgent, "get", classmethod(lambda cls, *a, **k: agent))
    return agent


# ---------------------------------------------------------------- deploy_agent


@pytest.mark.asyncio
async def test_a_budgeted_agent_is_deployed_with_a_key_of_its_own_and_its_budget_recorded(litellm, agent):
    step = DeployAgentTaskStep(id="deploy", version=None, a2a_agent_id="a", agent_name="solver", model_budget_usd=3)
    context = await step.execute(TaskStepContext(instance_id="run-1"))

    [made] = litellm.keys.values()
    assert agent.env_vars["LITELLM_API_KEY"] == made.key != "sk-shared"
    assert made.max_budget == 3.0 and made.metadata["run_id"] == "run-1"
    budget = context.metadata["model_budgets"]["solver"]
    assert (budget["max_usd"], budget["key_alias"], budget["key_blocked"]) == (3.0, made.alias, False)
    assert made.key not in str(context.to_safe_dict()) and made.key not in str(step.to_dict())


@pytest.mark.asyncio
async def test_an_agent_without_a_budget_gets_the_shared_key_as_before(litellm, agent):
    step = DeployAgentTaskStep(id="deploy", version=None, a2a_agent_id="a", agent_name="solver")
    context = await step.execute(TaskStepContext())

    assert litellm.requests == [] and "model_budgets" not in context.metadata
    assert "LITELLM_API_KEY" not in (agent.env_vars or {})  # A2AAgent.deploy adds the configured one


@pytest.mark.asyncio
async def test_the_key_of_an_agent_that_fails_to_be_configured_is_deleted(litellm, agent, monkeypatch):
    async def unloadable(self, *args, **kwargs):
        raise RuntimeError("the agent could not be reached")

    monkeypatch.setattr(DeployAgentTaskStep, "_load_snapshot", unloadable)
    step = DeployAgentTaskStep(id="deploy", version=None, a2a_agent_id="a", agent_name="solver",
                               agent_snapshot_files_artifact_id="u", model_budget_usd=3)
    context = TaskStepContext()

    with pytest.raises(RuntimeError, match="could not be reached"):
        await step.execute(context)
    assert litellm.keys == {} and agent.closed == 1
    assert "model_budgets" not in context.metadata and context.deployed_agents == []


@pytest.mark.asyncio
async def test_a_budgeted_agent_pointed_at_another_endpoint_is_not_deployed(litellm, agent):
    step = DeployAgentTaskStep(id="deploy", version=None, a2a_agent_id="a", agent_name="solver",
                               litellm_base_url="https://elsewhere.test", model_budget_usd=3)

    with pytest.raises(ValueError, match="configured model endpoint"):
        await step.execute(TaskStepContext())
    assert litellm.requests == [] and agent.env_vars is None


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["env_vars", "the agent's defaults"])
async def test_a_budgeted_agent_whose_env_points_at_another_endpoint_is_not_deployed(litellm, agent, where):
    elsewhere = {"LITELLM_BASE_URL": "https://elsewhere.test"}
    if where == "the agent's defaults":
        agent.default_env_vars = elsewhere
    step = DeployAgentTaskStep(id="deploy", version=None, a2a_agent_id="a", agent_name="solver",
                               env_vars=elsewhere if where == "env_vars" else None, model_budget_usd=3)

    with pytest.raises(ValueError, match="configured model endpoint"):
        await step.execute(TaskStepContext())
    assert litellm.requests == [] and agent.env_vars is None


def test_deploy_agent_round_trips_its_budget_and_refuses_a_key_beside_it():
    step = DeployAgentTaskStep(id="deploy", version=None, model_budget_usd=1.5)
    assert DeployAgentTaskStep.from_dict(step.to_dict()).model_budget_usd == 1.5
    assert DeployAgentTaskStep.from_dict({"id": "deploy", "type": "deploy_agent"}).model_budget_usd is None
    with pytest.raises(ValueError, match="env_vars may not set LITELLM_API_KEY"):
        DeployAgentTaskStep(id="deploy", version=None, env_vars={"LITELLM_API_KEY": "sk-x"}, model_budget_usd=1)
    with pytest.raises(ValueError, match="positive number"):
        DeployAgentTaskStep(id="deploy", version=None, model_budget_usd=0)


# ---------------------------------------------------------------- install_agent


class _InstallSandbox:
    def __init__(self):
        self.scripts: list[str] = []

    def scoped_name(self, name):
        return name

    def host_port(self, port):
        return port

    async def load_docker_images(self, images):
        return None

    async def load_object_file(self, url, destination):
        return None

    async def exec_script(self, script):
        self.scripts.append(script)
        return ""


def _resolved(value):
    return lambda *args, **kwargs: asyncio.sleep(0, result=value)


def _install(monkeypatch, *, required_params=("litellm_api_key",), card=None, expires_at_utc=None):
    sandbox = _InstallSandbox()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda name: SimpleNamespace(get_sandbox=_resolved(sandbox)))
    image = SimpleNamespace(image_name="img", build_context_object_url="s3://bucket/ctx.tar.gz")
    monkeypatch.setattr(A2AAgent, "get", staticmethod(lambda *args: SimpleNamespace(
        id="agent-x", version=1, metadata={}, docker_image_artifact=image)))
    step = InstallAgentTaskStep(id="i", version=None, sandbox_name="vm", a2a_agent_id="agent-x", agent_name="solver",
                                model_budget_usd=2)
    monkeypatch.setattr(step, "_extract_install_extension", _resolved({"params": {
        "install_commands_host": ["start --key {litellm_api_key}"],
        "required_params_host": list(required_params), "a2a_port": 8000}}))
    monkeypatch.setattr(step, "_wait_for_agent_card", card or _resolved({}))
    monkeypatch.setattr(step, "_register_instance", lambda *args: SimpleNamespace(instance_id="solver-1"))
    context = TaskStepContext(deployed_sandboxes=[DeployedSandbox(
        sandbox_name="vm", sandbox_id="sb-1", sandbox_mode="vm", sandbox_type="local",
        tunnel_urls={"8000": "http://127.0.0.1:8000"}, expires_at_utc=expires_at_utc)])
    return step, sandbox, context


@pytest.mark.asyncio
async def test_an_installed_agent_with_a_budget_is_installed_with_its_own_key(litellm, monkeypatch):
    step, sandbox, context = _install(monkeypatch)

    await step.execute(context)

    [made] = litellm.keys.values()
    assert [s for s in sandbox.scripts if s.startswith("start ")] == [f"start --key {made.key}"]
    assert context.metadata["model_budgets"]["solver"]["max_usd"] == 2.0


@pytest.mark.asyncio
async def test_an_installed_agents_key_lasts_as_long_as_its_sandbox_and_an_hour(litellm, monkeypatch):
    expires = datetime.now(timezone.utc) + timedelta(hours=6)
    step, _, context = _install(monkeypatch, expires_at_utc=expires.strftime("%Y-%m-%d %H:%M UTC"))

    await step.execute(context)

    [made] = litellm.keys.values()
    assert 7 * 3600 - 120 <= int(made.duration.removesuffix("s")) <= 7 * 3600


@pytest.mark.asyncio
async def test_an_agent_that_takes_no_key_cannot_be_given_a_budget(litellm, monkeypatch):
    step, sandbox, context = _install(monkeypatch, required_params=())

    with pytest.raises(RuntimeError, match="takes no litellm_api_key"):
        await step.execute(context)
    assert litellm.requests == [] and sandbox.scripts == []


@pytest.mark.asyncio
async def test_the_key_of_an_install_that_never_answers_is_deleted(litellm, monkeypatch):
    async def silent(*args, **kwargs):
        raise RuntimeError("Agent card did not respond")

    step, _, context = _install(monkeypatch, card=silent)

    with pytest.raises(RuntimeError, match="did not respond"):
        await step.execute(context)
    assert len(litellm.calls("/key/generate")) == 1 and litellm.keys == {}
    assert "model_budgets" not in context.metadata


def test_install_agent_round_trips_its_budget():
    step = InstallAgentTaskStep(id="i", version=None, sandbox_name="vm", model_budget_usd=0.5)
    assert InstallAgentTaskStep.from_dict(step.to_dict()).model_budget_usd == 0.5
    with pytest.raises(ValueError, match="positive number"):
        InstallAgentTaskStep(id="i", version=None, sandbox_name="vm", model_budget_usd=-2)
