"""sync_env_clock, modify_env_tool_access and register_env_triggers act on the deployment their
env_step_id names when one task deploys the same env more than once. Only the
gateway HTTP boundary is mocked."""

from __future__ import annotations

import httpx
import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.env.gateway.constants import GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.modify_env_tool_access import ModifyEnvToolAccessStep
from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep

ENV_ID = "env-x"


def _sync_clock(env_step_id):
    return SyncEnvClockTaskStep(id="clock", version=None, env_id=ENV_ID, virtual_time="2025-03-14T09:00:00Z",
                                env_step_id=env_step_id)


def _tool_access(env_step_id):
    return ModifyEnvToolAccessStep(id="tools", version=None, env_id=ENV_ID, action="disable", role="*",
                                   tools=["world_tool"], env_step_id=env_step_id)


def _triggers(env_step_id):
    return RegisterEnvTriggersStep(id="triggers", version=None, env_id=ENV_ID, triggers=[{"id": "t1"}],
                                   env_step_id=env_step_id)


STEPS = pytest.mark.parametrize("make_step", [_sync_clock, _tool_access, _triggers],
                                ids=["sync_env_clock", "modify_env_tool_access", "register_env_triggers"])


@STEPS
def test_env_step_id_round_trips(make_step):
    step = make_step("deploy-env-b")
    data = step.to_dict()
    assert data["env_step_id"] == "deploy-env-b"
    assert type(step).from_dict(data).env_step_id == "deploy-env-b"
    assert type(step).from_dict({k: v for k, v in data.items() if k != "env_step_id"}).env_step_id is None


@pytest.mark.asyncio
@STEPS
async def test_each_step_targets_the_deployment_its_env_step_id_names(monkeypatch, make_step):
    sent = _mock_gateways(monkeypatch)
    context = TaskStepContext(deployed_envs=[_lane("a"), _lane("b")], metadata={})

    await _run(monkeypatch, make_step("deploy-env-b"), context)

    assert sent and {r.url.host for r in sent} == {"gw-b"}


@pytest.mark.asyncio
@STEPS
async def test_env_step_id_naming_no_deployment_raises_before_any_request(monkeypatch, make_step):
    sent = _mock_gateways(monkeypatch)
    context = TaskStepContext(deployed_envs=[_lane("a"), _lane("b")], metadata={})

    with pytest.raises(RuntimeError, match="env_step_id 'deploy-env-typo' names no deployment"):
        await _run(monkeypatch, make_step("deploy-env-typo"), context)
    assert sent == []


@pytest.mark.asyncio
@STEPS
async def test_several_deployments_without_env_step_id_raise(monkeypatch, make_step):
    sent = _mock_gateways(monkeypatch)
    context = TaskStepContext(deployed_envs=[_lane("a"), _lane("b")], metadata={})

    with pytest.raises(RuntimeError, match="2 deployments of env 'env-x'"):
        await _run(monkeypatch, make_step(None), context)
    assert sent == []


@pytest.mark.asyncio
@STEPS
async def test_a_single_deployment_resolves_without_env_step_id(monkeypatch, make_step):
    sent = _mock_gateways(monkeypatch)
    context = TaskStepContext(deployed_envs=[_lane("a")], metadata={})

    await _run(monkeypatch, make_step(None), context)

    assert sent and {r.url.host for r in sent} == {"gw-a"}


@pytest.mark.asyncio
@STEPS
async def test_an_env_not_deployed_still_raises_not_found(monkeypatch, make_step):
    _mock_gateways(monkeypatch)
    with pytest.raises(RuntimeError, match="Env 'env-x' not found in context.deployed_envs"):
        await _run(monkeypatch, make_step("deploy-env-a"), TaskStepContext(metadata={}))


async def _run(monkeypatch, step: TaskStep, context: TaskStepContext) -> None:
    if isinstance(step, SyncEnvClockTaskStep):
        monkeypatch.setattr(step, "_backing_servers", _no_backing_servers)
    await step.execute(context)


async def _no_backing_servers(deployed) -> list:
    return []


def _lane(lane: str) -> DeployedEnv:
    gateway = f"http://gw-{lane}:18765"
    return DeployedGatewayEnv(env_id=ENV_ID, env_version=1, gateway_url=gateway, mcp_url="", db_web_url=None,
                              sandbox_id=f"sb-{lane}", environment_card_url=f"{gateway}{WELL_KNOWN_PATH}",
                              environment_card={"name": "gw", "capabilities": {"extensions": GATEWAY_EXTENSIONS}},
                              metadata={"deploy_step_id": f"deploy-env-{lane}"})


def _mock_gateways(monkeypatch) -> list[httpx.Request]:
    """Answer every gateway request with an empty object; return the requests."""
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent
