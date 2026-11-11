"""ModifyEnvToolAccessStep calls the gateway's enable/disable-tool extension as the stored env card
declares it, records the role's resulting state, and refuses before any request when the card
lacks the method. Only the gateway HTTP boundary is mocked."""

from __future__ import annotations

import json

import httpx
import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, EnvCapabilityUnsupported
from agent_env.env.gateway.constants import EXT_DISABLE_TOOL_URI, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.modify_env_tool_access import ModifyEnvToolAccessStep

_GATEWAY = "http://gw:18765"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["disable", "enable"])
async def test_posts_the_action_as_the_card_declares_it_and_records_the_role_state(monkeypatch, action):
    sent = _mock_gateway(monkeypatch, {"ok": True, "disabled": ["slack_send"], "allowed": []})
    ctx = TaskStepContext(deployed_envs=[_record()], metadata={})

    out = await _step(action).execute(ctx)

    [request] = sent
    assert (request.method, str(request.url)) == ("POST", f"{_GATEWAY}/tools/{action}")
    assert json.loads(request.content) == {"role": "default", "tools": ["slack_send"]}
    [change] = out.metadata["tool_access_changes"]
    assert change["action"] == action
    assert change["role_state_after"] == {"disabled": ["slack_send"], "allowed": []}


@pytest.mark.asyncio
async def test_a_card_without_the_action_raises_before_any_request(monkeypatch):
    sent = _mock_gateway(monkeypatch, {})
    only_disable = [e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_DISABLE_TOOL_URI]
    ctx = TaskStepContext(deployed_envs=[_record(only_disable)], metadata={})

    with pytest.raises(EnvCapabilityUnsupported, match="does not offer 'enable' on urn:agentenv:enable-tool/v1"):
        await _step("enable").execute(ctx)
    assert sent == []


@pytest.mark.asyncio
async def test_a_gateway_error_raises(monkeypatch):
    _mock_gateway(monkeypatch, {"ok": False}, status=500)
    with pytest.raises(httpx.HTTPStatusError):
        await _step("disable").execute(TaskStepContext(deployed_envs=[_record()], metadata={}))


@pytest.mark.asyncio
@pytest.mark.parametrize("step_params, expected", [
    ({}, ["slack_send"]),
    ({"tools-1": {"tools": ["mail_send", "mail_delete"]}}, ["mail_send", "mail_delete"]),
    ({"tools-1": {"tools": []}}, []),
    ({"other-step": {"tools": ["mail_send"]}}, ["slack_send"]),
])
async def test_a_tools_step_override_replaces_the_stored_tools(monkeypatch, step_params, expected):
    sent = _mock_gateway(monkeypatch, {"ok": True, "disabled": expected, "allowed": []})
    ctx = TaskStepContext(deployed_envs=[_record()], metadata={"user_overrides": {"step_params": step_params}})
    step = _step("disable")

    out = await step.execute(ctx)

    [request] = sent
    assert json.loads(request.content) == {"role": "default", "tools": expected}
    assert out.metadata["tool_access_changes"][0]["tools_requested"] == expected
    assert step.tools == ["slack_send"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["mail_send", ["mail_send", 3], None, {"mail_send": True}])
async def test_a_tools_override_that_is_not_a_list_of_strings_raises_before_any_request(monkeypatch, bad):
    sent = _mock_gateway(monkeypatch, {})
    ctx = TaskStepContext(deployed_envs=[_record()], metadata={"user_overrides": {"step_params": {"tools-1": {"tools": bad}}}})

    with pytest.raises(ValueError, match="step override tools must be a list of strings"):
        await _step("disable").execute(ctx)
    assert sent == []


def _step(action: str) -> ModifyEnvToolAccessStep:
    return ModifyEnvToolAccessStep(id="tools-1", version=None, env_id="env-x", action=action, role="default",
                                   tools=["slack_send"])


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
