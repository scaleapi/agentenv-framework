"""ConfigureWakeupsTaskStep posts the env's wakeups/v1 config as the stored card declares it and records it, and
refuses before any request on a gateway without wakeups/v1. Only the gateway HTTP boundary is mocked."""

from __future__ import annotations

import json

import httpx
import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.env.gateway.constants import EXT_WAKEUPS_URI, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.configure_wakeups import ConfigureWakeupsTaskStep

_GATEWAY = "http://gw:18765"


@pytest.mark.asyncio
async def test_posts_the_config_and_records_what_the_gateway_applied(monkeypatch):
    sent = _mock_gateway(monkeypatch, {"scheduled_checks": True, "max_active": 5, "min_interval": "PT1H"})
    out = await _step(max_active=5).execute(TaskStepContext(deployed_envs=[_record()], metadata={}))
    [request] = sent
    assert (request.method, str(request.url)) == ("POST", f"{_GATEWAY}/wakeups/config")
    assert json.loads(request.content) == {"scheduled_checks": True, "max_active": 5, "min_interval": "PT1H"}
    [entry] = out.metadata["wakeup_configurations"]
    assert entry["applied"] == {"scheduled_checks": True, "max_active": 5, "min_interval": "PT1H"}


@pytest.mark.asyncio
async def test_a_gateway_without_wakeups_refuses_before_any_request(monkeypatch):
    sent = _mock_gateway(monkeypatch, {})
    older = [e for e in GATEWAY_EXTENSIONS if e["uri"] != EXT_WAKEUPS_URI]
    with pytest.raises(RuntimeError, match="without urn:agentenv:wakeups/v1"):
        await _step().execute(TaskStepContext(deployed_envs=[_record(older)], metadata={}))
    assert sent == []


@pytest.mark.asyncio
async def test_a_refused_config_raises_with_the_gateway_s_reason(monkeypatch):
    _mock_gateway(monkeypatch, {"ok": False, "error": "['schedule_check'] already registered by a backing server"},
                  status=409)
    with pytest.raises(RuntimeError, match="HTTP 409.*schedule_check"):
        await _step().execute(TaskStepContext(deployed_envs=[_record()], metadata={}))


def test_round_trips_and_is_registered():
    step = _step(max_active=3, min_interval="PT2H")
    again = ConfigureWakeupsTaskStep.from_dict(step.to_dict())
    assert (again.max_active, again.min_interval, again.scheduled_checks) == (3, "PT2H", True)
    assert get_task_step_registry()["configure_wakeups"] is ConfigureWakeupsTaskStep


@pytest.mark.parametrize("kwargs", [{"max_active": 0}, {"scheduled_checks": "yes"}])
def test_bad_settings_are_refused_when_the_step_is_built(kwargs):
    with pytest.raises(ValueError):
        _step(**kwargs)


def _step(**kwargs) -> ConfigureWakeupsTaskStep:
    return ConfigureWakeupsTaskStep(id="wakeups-1", version=None, env_id="env-x", **kwargs)


def _record(extensions: list = GATEWAY_EXTENSIONS) -> DeployedEnv:
    return DeployedGatewayEnv(env_id="env-x", env_version=1, gateway_url=_GATEWAY, mcp_url="", db_web_url=None,
                              sandbox_id="sb", environment_card_url=f"{_GATEWAY}{WELL_KNOWN_PATH}",
                              environment_card={"name": "gw", "capabilities": {"extensions": extensions}})


def _mock_gateway(monkeypatch, body: dict, status: int = 200) -> list[httpx.Request]:
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, json=body)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent
