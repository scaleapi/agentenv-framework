"""The validator's builtin_tools check: an agent whose agent-config doesn't take the field is recorded as not
supporting it and nothing is deployed; otherwise a second agent, deployed with {"disable": "*"} and no envs, must
refuse an unknown tool, read the selection back, and make no tool call when asked to run a shell command, or say it
has no tool when it reports no count."""

import json

import httpx
import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.context import DeployedAgent, PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.a2a_agent_validator import verify_a2a_builtin_tools as module
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_builtin_tools import NO_SUCH_TOOL, VerifyA2ABuiltinToolsStep


def _card(*supported: str) -> dict:
    return {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
        "endpoint": "/ext/agent-config",
        "methods": {"set": {"request": {"supported": list(supported)}}, "get": {"method": "GET"}}}}]}}


class _Agent:
    """The limited agent's agent-config: a selection it was deployed with, and the 400 an unknown tool gets."""

    def __init__(self):
        self.config, self.deploys = {}, []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            # As the protocol's readback writes a selection: with the field it leaves unset as null.
            return httpx.Response(200, json={"config": {**self.config, "builtin_tools": {
                "disable": None, "only": None, **self.config.get("builtin_tools", {})}}})
        body = json.loads(request.content)
        if NO_SUCH_TOOL in str(body):
            return httpx.Response(400, json={"detail": "not one of the agent's tools"})
        self.config.update(body)
        return httpx.Response(200, json={})


@pytest.fixture
def wired(monkeypatch):
    def wire(tool_call_count, reply="Linux box 6.8.0"):
        agent = _Agent()
        real_client = httpx.AsyncClient
        monkeypatch.setattr(module.httpx, "AsyncClient",
                            lambda **kw: real_client(transport=httpx.MockTransport(agent.handle), **kw))
        monkeypatch.setattr(A2AAgent, "get", lambda *a, **kw: type("Stored", (), {
            "metadata": {}, "update_metadata": lambda self, m: None})())

        async def deploy(step, context):
            agent.deploys.append(step.builtin_tools)
            agent.config["builtin_tools"] = step.builtin_tools
            context.deployed_agents.append(DeployedAgent(agent_name=step.agent_name, api_url="http://limited",
                                                         a2a_url="http://limited", a2a_card={}))
            return context

        async def prompt(step, context):
            context.prompt_responses.append(PromptResponse(prompt_id=step.prompt_id, response=reply,
                                                           tool_call_count=tool_call_count))
            return context

        monkeypatch.setattr(module.DeployAgentTaskStep, "execute", deploy)
        monkeypatch.setattr(module.PromptAgentTaskStep, "execute", prompt)
        return agent

    return wire


def _context(card: dict) -> TaskStepContext:
    context = TaskStepContext(instance_id="ti-1")
    context.deployed_agents.append(DeployedAgent(agent_name="solver", api_url="http://main", a2a_url="http://main",
                                                 a2a_card=card))
    return context


async def _verify(card: dict) -> dict:
    step = VerifyA2ABuiltinToolsStep(id="builtin-tools", version=None, a2a_agent_id="agent-1")
    return (await step.execute(_context(card))).metadata["verifications"]["a2a_builtin_tools"]


@pytest.mark.asyncio
async def test_an_agent_whose_config_takes_no_builtin_tools_is_recorded_as_not_supporting_it(wired):
    agent = wired(0)

    assert await _verify(_card("name", "role")) == {"supported": False,
                                                     "reason": "agent-config takes no builtin_tools"}
    assert agent.deploys == []


@pytest.mark.asyncio
async def test_an_agent_that_honours_it_passes(wired):
    agent = wired(0)

    verified = await _verify(_card("name", "builtin_tools"))

    assert agent.deploys == [{"disable": "*"}]
    assert {k: verified[k] for k in ("supported", "passed", "unknown_tool_refused", "read_back", "no_tool_calls")} == {
        "supported": True, "passed": True, "unknown_tool_refused": True, "read_back": True, "no_tool_calls": True}


@pytest.mark.asyncio
async def test_an_agent_whose_limited_turn_calls_a_tool_fails(wired):
    wired(2)

    verified = await _verify(_card("name", "builtin_tools"))

    assert (verified["passed"], verified["no_tool_calls"], verified["tool_call_count"]) == (False, False, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply, passed", [("NO-TOOLS", True), ("Linux box 6.8.0", False)])
async def test_an_agent_that_reports_no_count_must_say_it_has_no_tool(wired, reply, passed):
    wired(None, reply)

    assert (await _verify(_card("name", "builtin_tools")))["passed"] is passed
