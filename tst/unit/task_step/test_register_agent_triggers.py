"""Behavior tests for RegisterAgentTriggersStep: card-driven endpoint resolution, the POST body/audit,
fail-loud on a missing/non-triggers agent and dangling env refs, and 4xx propagation. Only the agent
HTTP boundary is mocked; the step logic runs real."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.register_agent_triggers import RegisterAgentTriggersStep

_A2A = "http://stakeholder:8000"


def _card(uri: str = A2AAgent.EXT_TRIGGERS, endpoint: str = "/ext/triggers") -> dict:
    return {"capabilities": {"extensions": [
        {"uri": uri, "params": {"endpoint": endpoint,
                                "methods": {"register": {"method": "POST"},
                                            "decide": {"method": "POST", "endpoint": "/ext/triggers/decide"},
                                            "state": {"method": "GET"}}}}]}}


def _resp(status: int, url: str, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", url))


def _context(*, card: dict | None = None, env_ids: tuple[str, ...] = ("env-x",), with_agent: bool = True,
             registered_env_triggers: dict[str, list[str]] | None = None):
    if registered_env_triggers is None:
        registered_env_triggers = {"env-x": ["v6-rate"]}
    envs = [DeployedEnv(env_id=e, env_version=1, gateway_url="http://gw", mcp_url="",
                        db_web_url=None, sandbox_id="sb") for e in env_ids]
    agents = []
    if with_agent:
        agents.append(DeployedAgent(agent_name="stake-1", api_url="http://stakeholder:8000/api",
                                    a2a_url=_A2A, a2a_card=_card() if card is None else card, role="stakeholder"))
    trigger_registrations = [{"step_id": f"reg-{e}", "env_id": e, "added": list(tids), "executor_agent_name": None}
                             for e, tids in registered_env_triggers.items()]
    return TaskStepContext(deployed_envs=envs, deployed_agents=agents,
                           metadata={"env_trigger_registrations": trigger_registrations})


_TRIGGERS = [
    {"id": "open", "when": {"type": "step", "turn": 1}, "actions": [{"type": "say", "text": "[Julia]: hi"}]},
    {"id": "re-sweep", "when": {"type": "env_trigger", "env_id": "env-x", "trigger_id": "v6-rate", "status": "fired"},
     "actions": [{"type": "say", "text": "[Julia]: sweep"}]},
]


def _step(**overrides):
    base = {"id": "reg-1", "version": None, "agent_name": "stake-1", "triggers": _TRIGGERS}
    base.update(overrides)
    return RegisterAgentTriggersStep(**base)


def test_round_trips_through_to_dict_from_dict():
    step = _step()
    rebuilt = RegisterAgentTriggersStep.from_dict(step.to_dict())
    assert rebuilt.agent_name == "stake-1"
    assert rebuilt.triggers == _TRIGGERS


def test_bad_triggers_rejected_at_construction():
    with pytest.raises(ValueError, match="triggers must be a list"):
        _step(triggers="nope")


@pytest.mark.asyncio
async def test_registers_and_audits_with_card_endpoint():
    captured = {}

    async def fake_post(self, url, json, timeout):
        captured["url"] = url
        captured["body"] = json
        return _resp(200, url, {"ok": True, "added": ["open", "re-sweep"], "all": ["open", "re-sweep"]})

    ctx = _context()
    with patch.object(httpx.AsyncClient, "post", fake_post):
        out = await _step().execute(ctx)

    assert captured["url"] == f"{_A2A}/ext/triggers"
    assert captured["body"] == {"triggers": _TRIGGERS}
    audit = out.metadata["agent_trigger_registrations"]
    assert len(audit) == 1
    assert audit[0]["agent_name"] == "stake-1"
    assert audit[0]["added"] == ["open", "re-sweep"]


@pytest.mark.asyncio
async def test_custom_endpoint_from_card_is_honored():
    captured = {}

    async def fake_post(self, url, json, timeout):
        captured["url"] = url
        return _resp(200, url, {"ok": True, "added": []})

    ctx = _context(card=_card(endpoint="/custom/triggers"))
    with patch.object(httpx.AsyncClient, "post", fake_post):
        await _step().execute(ctx)

    assert captured["url"] == f"{_A2A}/custom/triggers"


@pytest.mark.asyncio
async def test_missing_agent_raises():
    ctx = _context(with_agent=False)
    with pytest.raises(RuntimeError, match="Agent 'stake-1' not found"):
        await _step().execute(ctx)


@pytest.mark.asyncio
async def test_agent_without_triggers_extension_raises():
    ctx = _context(card=_card(uri="urn:agentenv:agent-config/v1"))
    with pytest.raises(RuntimeError, match="does not advertise urn:agentenv:triggers/v1"):
        await _step().execute(ctx)


@pytest.mark.asyncio
async def test_dangling_env_trigger_ref_raises():
    ctx = _context(env_ids=("some-other-env",))  # 're-sweep' references env-x, not deployed
    with pytest.raises(RuntimeError, match="references env_trigger env_id 'env-x'"):
        await _step().execute(ctx)


@pytest.mark.asyncio
async def test_env_trigger_id_not_registered_raises():
    # env-x is deployed but no 'v6-rate' env trigger was registered on it (per the task-run context)
    ctx = _context(registered_env_triggers={"env-x": []})
    with pytest.raises(RuntimeError, match="env_trigger 'v6-rate' not registered on env 'env-x'"):
        await _step().execute(ctx)


@pytest.mark.asyncio
async def test_agent_4xx_raises():
    async def fake_post(self, url, json, timeout):
        return _resp(400, url, {"ok": False, "error": "trigger 'open': bad when"})

    ctx = _context()
    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(RuntimeError, match="agent trigger registration failed"):
            await _step().execute(ctx)
