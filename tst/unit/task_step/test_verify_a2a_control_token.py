"""The validator's control-token check: a read the SDK serves on every agent, mcp-config's list, is called without the
agent's token and with it; the agent passes when the first is refused with a 401 and the second is not."""

import httpx
import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.a2a_agent_validator import verify_a2a_control_token as module
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_control_token import VerifyA2AControlTokenStep

_CARD = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_MCP_CONFIG, "params": {"endpoint": "/ext/mcp-config"}}]}}


def _serve(monkeypatch, *, guarded: bool) -> list[tuple[str, str | None]]:
    calls: list[tuple[str, str | None]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        token = request.headers.get("AgentEnv-Control-Token")
        calls.append((request.url.path, token))
        if guarded and token != "s3cret":
            return httpx.Response(401, json={"detail": "this route takes the agent's control token"})
        return httpx.Response(200, json={"mcp_servers": {}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    return calls


async def _check(card: dict = _CARD, token: str | None = "s3cret") -> dict:
    context = TaskStepContext(deployed_agents=[DeployedAgent(
        agent_name="default-agent", api_url="http://agent", a2a_url="http://agent", a2a_card=card, control_token=token)])
    step = VerifyA2AControlTokenStep(id="v", version=None, a2a_agent_id="agent-x")
    return (await step.execute(context)).metadata["verifications"]["a2a_control_token"]


@pytest.mark.asyncio
async def test_an_agent_that_refuses_a_call_without_its_token_and_answers_one_with_it_passes(monkeypatch):
    calls = _serve(monkeypatch, guarded=True)

    entry = await _check()

    assert entry == {"route": "/ext/mcp-config", "refuses_without_token": True, "accepts_with_token": True,
                     "passed": True}
    assert calls == [("/ext/mcp-config", None), ("/ext/mcp-config", "s3cret")]


@pytest.mark.asyncio
async def test_an_agent_that_answers_without_its_token_fails(monkeypatch):
    _serve(monkeypatch, guarded=False)

    entry = await _check()

    assert entry["refuses_without_token"] is False and entry["accepts_with_token"] is True
    assert entry["passed"] is False


@pytest.mark.asyncio
async def test_an_agent_with_no_guarded_read_or_no_token_on_its_record_fails_without_a_call(monkeypatch):
    calls = _serve(monkeypatch, guarded=True)

    no_read = await _check(card={})
    no_token = await _check(token=None)

    assert no_read["route"] is None and no_read["passed"] is False
    assert no_token["passed"] is False and calls == []
