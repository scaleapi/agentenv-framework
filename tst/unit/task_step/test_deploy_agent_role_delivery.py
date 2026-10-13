"""deploy_agent delivers the step's role to the agent, or fails the step before any request reaches it; it never
records a role the agent did not get."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import deploy_agent
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.mcp_cli_builder.codegen import ROLE_ENV_VAR

_LOGGER = "agent_env.task_step.task_steps.deploy_agent"
_GATEWAY_MCP_URL = "https://gw.example/mcp"
_LIVE_MCP_URL = "https://ext.example/mcp"


def _mcp_config(add_request: dict) -> dict:
    return {
        "uri": A2AAgent.EXT_MCP_CONFIG,
        "params": {"endpoint": "/ext/mcp-config", "methods": {"add": {"method": "POST", "request": add_request}}},
    }


def _agent_config(supported: list[str]) -> dict:
    return {
        "uri": A2AAgent.EXT_AGENT_CONFIG,
        "params": {"endpoint": "/ext/agent-config", "methods": {"set": {"method": "POST", "request": {"supported": supported}}}},
    }


def _card(*extensions: dict) -> dict:
    return {"capabilities": {"extensions": list(extensions)}}


HEADERS_MCP = _mcp_config({"required": ["url"], "optional": ["headers", "name"]})  # as the protocol renders it
LEGACY_MCP = _mcp_config({"required": ["url"]})
ROLE_CONFIG = _agent_config(["name", "description", "role"])
ROLELESS_CONFIG = _agent_config(["name", "description"])


@pytest.mark.asyncio
async def test_a_card_with_both_paths_gets_the_role_on_both(wiring, caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        context = await wiring.deploy(_card(HEADERS_MCP, ROLE_CONFIG), role="executor")

    assert wiring.bodies("/ext/mcp-config") == [
        {"url": _GATEWAY_MCP_URL, "headers": {AGENT_ENV_ROLE_HEADER: "executor"}, "name": "crm"}
    ]
    assert wiring.bodies("/ext/agent-config") == [{"name": "solver", "role": "executor"}]
    assert not [m for m in caplog.messages if "role" in m]
    assert [a.role for a in context.deployed_agents] == ["executor"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "card", [_card(HEADERS_MCP), _card(HEADERS_MCP, ROLELESS_CONFIG)], ids=["no-agent-config", "roleless-agent-config"]
)
async def test_a_header_only_card_gets_the_role_on_its_mcp_registration(wiring, card):
    context = await wiring.deploy(card, role="executor")

    assert wiring.bodies("/ext/mcp-config") == [
        {"url": _GATEWAY_MCP_URL, "headers": {AGENT_ENV_ROLE_HEADER: "executor"}, "name": "crm"}
    ]
    assert all("role" not in body for body in wiring.bodies("/ext/agent-config"))
    assert [a.role for a in context.deployed_agents] == ["executor"]


@pytest.mark.asyncio
async def test_a_config_only_card_gets_the_role_through_agent_config(wiring):
    context = await wiring.deploy(_card(LEGACY_MCP, ROLE_CONFIG), role="executor")

    assert wiring.bodies("/ext/mcp-config") == [{"url": _GATEWAY_MCP_URL}]
    assert wiring.bodies("/ext/agent-config") == [{"name": "solver", "role": "executor"}]
    assert [a.role for a in context.deployed_agents] == ["executor"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "card",
    [_card(LEGACY_MCP, ROLELESS_CONFIG), _card(LEGACY_MCP), {}],
    ids=["legacy-mcp-and-roleless-config", "legacy-mcp-alone", "empty-card"],
)
async def test_a_card_with_neither_path_fails_before_any_request(wiring, card):
    with pytest.raises(RuntimeError, match="cannot carry role 'grader'"):
        await wiring.deploy(card, role="grader")

    assert wiring.requests == []
    assert wiring.agent.close.await_count == 1
    assert wiring.context.deployed_agents == []


@pytest.mark.asyncio
async def test_an_env_that_is_not_behind_a_gateway_gets_the_header_too(wiring):
    """deploy_agent does not judge what the env does with the role: a protocol server reads the header itself."""
    context = await wiring.deploy(_card(HEADERS_MCP), role="grader", env_ids=["ext"])

    assert wiring.bodies("/ext/mcp-config") == [{"url": _LIVE_MCP_URL, "headers": {AGENT_ENV_ROLE_HEADER: "grader"}}]
    assert wiring.bodies("/ext/agent-config") == []
    assert [a.role for a in context.deployed_agents] == ["grader"]


@pytest.mark.asyncio
async def test_the_header_lands_on_every_mcp_registration(wiring):
    context = await wiring.deploy(_card(HEADERS_MCP), role="alice@example.com", env_ids=["crm", "ext"])

    assert wiring.bodies("/ext/mcp-config") == [
        {"url": _GATEWAY_MCP_URL, "headers": {AGENT_ENV_ROLE_HEADER: "alice@example.com"}, "name": "crm"},
        {"url": _LIVE_MCP_URL, "headers": {AGENT_ENV_ROLE_HEADER: "alice@example.com"}},
    ]
    assert wiring.bodies("/ext/agent-config") == []
    assert [a.role for a in context.deployed_agents] == ["alice@example.com"]


@pytest.mark.asyncio
async def test_the_role_header_joins_the_sandbox_headers_without_touching_them(wiring, monkeypatch):
    sandbox_headers = {"Modal-Key": "k"}
    monkeypatch.setattr(deploy_agent, "sandbox_request_headers_for_url", lambda url: sandbox_headers)

    await wiring.deploy(_card(HEADERS_MCP), role="executor")

    assert wiring.bodies("/ext/mcp-config")[0]["headers"] == {"Modal-Key": "k", AGENT_ENV_ROLE_HEADER: "executor"}
    assert sandbox_headers == {"Modal-Key": "k"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "add_request, takes_headers",
    [
        ({"required": ["url"]}, False),
        ({"required": ["url"], "optional": None}, False),
        ({"required": ["url"], "optional": ["headers", "name"]}, True),
        ({"supported": ["url", "headers"]}, True),
        ({"required": ["url", "name"], "optional": ["headers"]}, True),
        ({"required": ["url", "name"]}, False),
    ],
    ids=["legacy", "optional-none", "protocol-rendered", "hand-written-supported", "name-required", "name-required-no-headers"],
)
async def test_the_card_decides_whether_the_registration_carries_the_role(wiring, add_request, takes_headers):
    await wiring.deploy(_card(_mcp_config(add_request), ROLE_CONFIG), role="executor")

    body = wiring.bodies("/ext/mcp-config")[0]
    assert body.get("headers", {}).get(AGENT_ENV_ROLE_HEADER) == ("executor" if takes_headers else None)


@pytest.mark.asyncio
async def test_no_role_leaves_every_request_as_before(wiring, caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        context = await wiring.deploy(_card(HEADERS_MCP, ROLE_CONFIG), role=None)

    assert wiring.bodies("/ext/mcp-config") == [{"url": _GATEWAY_MCP_URL, "name": "crm"}]
    assert wiring.bodies("/ext/agent-config") == [{"name": "solver"}]
    assert not [m for m in caplog.messages if "role" in m]
    assert wiring.env_vars is None
    assert [a.role for a in context.deployed_agents] == [None]


@pytest.mark.asyncio
async def test_an_empty_role_is_no_role(wiring):
    assert DeployAgentTaskStep(id="ag", version=None, role="").role is None
    assert DeployAgentTaskStep.from_dict({"id": "ag", "type": "deploy_agent", "version": 1, "role": ""}).role is None

    context = await wiring.deploy(_card(LEGACY_MCP), role="")

    assert wiring.bodies("/ext/mcp-config") == [{"url": _GATEWAY_MCP_URL}]
    assert wiring.env_vars is None
    assert [a.role for a in context.deployed_agents] == [None]


@pytest.mark.asyncio
async def test_the_role_is_also_the_generated_clis_role(wiring):
    await wiring.deploy(_card(HEADERS_MCP), role="executor")
    assert wiring.env_vars == {ROLE_ENV_VAR: "executor"}


@pytest.mark.asyncio
async def test_an_explicit_cli_role_env_var_is_kept(wiring):
    await wiring.deploy(_card(HEADERS_MCP), role="executor", env_vars={ROLE_ENV_VAR: "cli"})
    assert wiring.env_vars == {ROLE_ENV_VAR: "cli"}


class _Wiring:
    """One deploy_agent run: the requests the agent received, the agent double, and the step's context."""

    def __init__(self, monkeypatch):
        self.requests: list[httpx.Request] = []
        self.agent = MagicMock(metadata={})
        self.agent.close = AsyncMock()
        self.context = TaskStepContext(deployed_envs=[_gateway_env()])
        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(self._handle), **kw)
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"ok": True})

    async def deploy(self, card: dict, *, role: str | None, env_ids=("crm",), **step_kwargs) -> TaskStepContext:
        self.agent.deploy = AsyncMock(
            return_value=MagicMock(a2a_url="https://agent.example", agent_card=card, sandbox_type=None)
        )
        step = DeployAgentTaskStep(
            id="ag", version=None, env_ids=list(env_ids), agent_name="solver", role=role, **step_kwargs
        )
        with (
            patch.object(A2AAgent, "get", return_value=self.agent),
            patch.object(deploy_agent.Env, "get", return_value=SimpleNamespace(mcp_url=_LIVE_MCP_URL)),
            patch("agent_env.task_step.task_steps.deploy_agent.get_config") as get_config,
        ):
            get_config.return_value.get_model_for_role.return_value = None
            return await step.execute(self.context)

    def bodies(self, path: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]

    @property
    def env_vars(self) -> dict | None:
        return self.agent.deploy.await_args.kwargs["env_vars"]


@pytest.fixture
def wiring(monkeypatch) -> _Wiring:
    return _Wiring(monkeypatch)


def _gateway_env() -> DeployedGatewayEnv:
    return DeployedGatewayEnv(
        env_id="crm", env_version=1, gateway_url="https://gw.example", mcp_url=_GATEWAY_MCP_URL, db_web_url=None,
        sandbox_id="sb-1", environment_card_url="https://gw.example/.well-known/agent-env.json",
        environment_card={"name": "crm"},
    )
