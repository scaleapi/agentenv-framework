"""Behavior tests for RegisterEnvTriggersStep: executor-name resolution, the POST body/audit, and 4xx
propagation. Only the gateway HTTP boundary is mocked; the step logic runs real."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep

_GATEWAY = "http://gw:18765"


def _resp(status: int, url: str, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", url))


def _context(*, with_executor: bool = False):
    env = DeployedEnv(env_id="env-x", env_version=1, gateway_url=_GATEWAY, mcp_url="",
                      db_web_url=None, sandbox_id="sb")
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
async def test_registers_and_audits():
    captured = {}

    async def fake_post(self, url, json, timeout):
        captured["url"] = url
        captured["body"] = json
        return _resp(200, url, {"ok": True, "added": ["t1"], "all": ["t1"]})

    ctx = _context()
    with patch.object(httpx.AsyncClient, "post", fake_post):
        out = await _step(watch_roles=["default"]).execute(ctx)

    assert captured["url"] == f"{_GATEWAY}/triggers/register"
    assert captured["body"] == {"triggers": _TRIGGERS, "watch_roles": ["default"]}
    assert "executor" not in captured["body"]
    audit = out.metadata["env_trigger_registrations"]
    assert len(audit) == 1
    assert audit[0]["env_id"] == "env-x"
    assert audit[0]["added"] == ["t1"]


@pytest.mark.asyncio
async def test_resolves_executor_agent_to_a2a_url():
    captured = {}

    async def fake_post(self, url, json, timeout):
        captured["body"] = json
        return _resp(200, url, {"ok": True, "added": ["t1"], "all": ["t1"]})

    ctx = _context(with_executor=True)
    with patch.object(httpx.AsyncClient, "post", fake_post):
        await _step(executor_agent_name="exec-1", executor_timeout_seconds=45,
                    watch_roles=["default"]).execute(ctx)

    assert captured["body"]["executor"] == {
        "a2a_url": "http://exec:9000/a2a-base", "timeout_seconds": 45, "role": "executor"}


@pytest.mark.asyncio
async def test_missing_executor_agent_raises():
    ctx = _context()  # no agents deployed
    with pytest.raises(RuntimeError, match="Executor agent 'exec-1' not found"):
        await _step(executor_agent_name="exec-1").execute(ctx)


@pytest.mark.asyncio
async def test_executor_without_role_raises():
    from agent_env.env.env import DeployedEnv
    env = DeployedEnv(env_id="env-x", env_version=1, gateway_url=_GATEWAY, mcp_url="",
                      db_web_url=None, sandbox_id="sb")
    ctx = TaskStepContext(
        deployed_envs=[env],
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
async def test_gateway_4xx_raises():
    async def fake_post(self, url, json, timeout):
        return _resp(400, url, {"ok": False, "error": "trigger 't1': when.tool: bad"})

    ctx = _context()
    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(RuntimeError, match="trigger registration failed"):
            await _step().execute(ctx)
