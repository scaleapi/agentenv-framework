"""Behavior tests for RegisterEnvTriggersStep: executor-name resolution, the POST body/audit, and 4xx
propagation. Only the gateway HTTP boundary is mocked; the step logic runs real."""

from __future__ import annotations

import json

import httpx
import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, EnvCapabilityUnsupported
from agent_env.env.gateway.constants import GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep

_GATEWAY = "http://gw:18765"


def _context(*, with_executor: bool = False):
    env = _record()
    agents = []
    if with_executor:
        agents.append(DeployedAgent(agent_name="exec-1", api_url="http://exec:9000",
                                    a2a_url="http://exec:9000/a2a-base", role=with_executor
                                    if isinstance(with_executor, str) else "executor"))
    return TaskStepContext(deployed_envs=[env], deployed_agents=agents, metadata={})


_TRIGGERS = [{"id": "t1", "when": {"type": "action", "tool": "gdocs_create_document"},
              "actions": [{"type": "permission", "action": "enable", "role": "default",
                           "tools": ["snowflake_submit_query"]}]}]


def _step(**overrides):
    base = {"id": "reg-1", "version": None, "env_id": "env-x", "triggers": _TRIGGERS}
    base.update(overrides)
    return RegisterEnvTriggersStep(**base)


def test_round_trips_through_to_dict_from_dict():
    step = _step(watch_roles=["default"], executor_agent_name="exec-1", executor_timeout_seconds=90)
    rebuilt = RegisterEnvTriggersStep.from_dict(step.to_dict())
    assert rebuilt.env_id == "env-x"
    assert rebuilt.triggers == _TRIGGERS
    assert rebuilt.watch_roles == ["default"]
    assert rebuilt.executor_agent_name == "exec-1"
    assert rebuilt.executor_timeout_seconds == 90


def test_bad_triggers_rejected_at_construction():
    with pytest.raises(ValueError, match="triggers must be a list"):
        _step(triggers="nope")


@pytest.mark.asyncio
async def test_registers_and_audits(monkeypatch):
    sent = _mock_gateway(monkeypatch, {"ok": True, "added": ["t1"], "all": ["t1"]})
    out = await _step(watch_roles=["default"]).execute(_context())

    [request] = sent
    assert (request.method, str(request.url)) == ("POST", f"{_GATEWAY}/triggers/register")
    assert json.loads(request.content) == {"triggers": _TRIGGERS, "watch_roles": ["default"]}
    audit = out.metadata["env_trigger_registrations"]
    assert len(audit) == 1
    assert audit[0]["env_id"] == "env-x"
    assert audit[0]["added"] == ["t1"]


@pytest.mark.asyncio
async def test_resolves_executor_agent_to_a2a_url(monkeypatch):
    sent = _mock_gateway(monkeypatch, {"ok": True, "added": ["t1"], "all": ["t1"]})
    await _step(executor_agent_name="exec-1", executor_timeout_seconds=45,
                watch_roles=["default"]).execute(_context(with_executor=True))

    assert json.loads(sent[0].content)["executor"] == {
        "a2a_url": "http://exec:9000/a2a-base", "timeout_seconds": 45, "role": "executor"}


@pytest.mark.asyncio
@pytest.mark.parametrize("in_our_sandbox, expected", [
    (True, "http://host.docker.internal:41234"), (False, "http://127.0.0.1:41234"),
], ids=["local-gateway", "env-outside-our-sandboxes"])
async def test_a_local_gateway_reaches_a_local_executor_through_the_host(monkeypatch, in_our_sandbox, expected):
    sent = _mock_gateway(monkeypatch, {"ok": True, "added": ["t1"], "all": ["t1"]})
    env = _record()
    if in_our_sandbox:
        env.sandbox_type = "local"
    else:
        env = DeployedEnv(env_id=env.env_id, env_version=1, mcp_url="", environment_card_url=env.environment_card_url,
                          environment_card=env.environment_card)
    executor = DeployedAgent(agent_name="exec-1", api_url="http://127.0.0.1:41234", a2a_url="http://127.0.0.1:41234",
                             sandbox_id="local-a", sandbox_type="local", role="executor")
    context = TaskStepContext(deployed_envs=[env], deployed_agents=[executor], metadata={})

    await _step(executor_agent_name="exec-1", watch_roles=["default"]).execute(context)

    assert json.loads(sent[0].content)["executor"]["a2a_url"] == expected


@pytest.mark.asyncio
async def test_missing_executor_agent_raises():
    ctx = _context()  # no agents deployed
    with pytest.raises(RuntimeError, match="Executor agent 'exec-1' not found"):
        await _step(executor_agent_name="exec-1").execute(ctx)


@pytest.mark.asyncio
async def test_executor_without_role_raises():
    ctx = TaskStepContext(
        deployed_envs=[_record()],
        deployed_agents=[DeployedAgent(agent_name="exec-1", api_url="http://e", role=None)],
        metadata={})
    with pytest.raises(RuntimeError, match="deployed without an explicit role"):
        await _step(executor_agent_name="exec-1", watch_roles=["default"]).execute(ctx)


@pytest.mark.asyncio
async def test_executor_role_in_watch_roles_raises():
    ctx = _context(with_executor="default")  # executor deployed under the watched role
    with pytest.raises(RuntimeError, match="is in watch_roles"):
        await _step(executor_agent_name="exec-1", watch_roles=["default"]).execute(ctx)


@pytest.mark.asyncio
async def test_missing_env_raises():
    ctx = _context()
    with pytest.raises(RuntimeError, match="Env 'other-env' not found"):
        await _step(env_id="other-env").execute(ctx)


@pytest.mark.asyncio
async def test_gateway_4xx_raises(monkeypatch):
    _mock_gateway(monkeypatch, {"ok": False, "error": "trigger 't1': when.tool: bad"}, status=400)
    with pytest.raises(RuntimeError, match=r"trigger registration failed \(HTTP 400\): .*when.tool: bad"):
        await _step().execute(_context())


@pytest.mark.asyncio
async def test_a_card_without_triggers_raises_before_any_request(monkeypatch):
    sent = _mock_gateway(monkeypatch, {})
    ctx = TaskStepContext(deployed_envs=[_record(extensions=[])], deployed_agents=[], metadata={})
    with pytest.raises(EnvCapabilityUnsupported, match="does not offer 'register' on urn:agentenv:triggers/v1"):
        await _step().execute(ctx)
    assert sent == []


def _record(extensions: list = GATEWAY_EXTENSIONS) -> DeployedEnv:
    """A gateway deployment whose stored card advertises `extensions` (the gateway's own, by default)."""
    return DeployedGatewayEnv(env_id="env-x", env_version=1, gateway_url=_GATEWAY, mcp_url="", db_web_url=None, sandbox_id="sb",
                       environment_card_url=f"{_GATEWAY}{WELL_KNOWN_PATH}",
                       environment_card={"name": "gw", "capabilities": {"extensions": extensions}})


def _mock_gateway(monkeypatch, body: dict, status: int = 200) -> list[httpx.Request]:
    """Answer every request the protocol client sends with `body`; return the requests."""
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, json=body)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent
