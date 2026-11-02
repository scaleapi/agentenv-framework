"""agent-env finds cdp-config/v1 on a card the protocol SDK builds, and where to call each of its operations."""

from starlette.testclient import TestClient

from agent_env.a2a_agent import A2AAgent
from agentenv_protocol.a2a_agent import (
    CDP_CONFIG_V1,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    a2a_agent,
)


@a2a_agent(identity=AgentIdentity(name="cdp-agent", description="test", version="1"), extensions=(CDP_CONFIG_V1,))
class CdpAgent(AgentEnvAgent):
    async def run(self, request: TaskRequest) -> TaskResult:
        return TaskResult.text("ok")


def test_each_operation_resolves_to_its_method_and_path():
    with TestClient(CdpAgent().create_app()) as client:
        card = client.get("/.well-known/agent-card.json").json()
    extension = A2AAgent.find_extension(card, A2AAgent.EXT_CDP_CONFIG)

    assert A2AAgent.EXT_CDP_CONFIG == CDP_CONFIG_V1.uri
    resolved = {name: A2AAgent.operation(extension, name) for name in ("add", "list", "remove")}
    assert {name: (method["method"], path) for name, (method, path) in resolved.items()} == {
        "add": ("POST", "/ext/cdp-config"),
        "list": ("GET", "/ext/cdp-config"),
        "remove": ("DELETE", "/ext/cdp-config"),
    }


def test_a_card_without_it_has_no_cdp_operations():
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_MCP_CONFIG, "params": {}}]}}

    assert A2AAgent.find_extension(card, A2AAgent.EXT_CDP_CONFIG) is None
    assert A2AAgent.extension_method(card, A2AAgent.EXT_CDP_CONFIG, "add") is None
