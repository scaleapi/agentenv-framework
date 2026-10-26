"""mcp-config add keeps the name the caller sends, and mints mcp_<hex> when it sends none. The SDK's peer_agents
server is listed beside them."""

from tst.unit.agents.claude_code.agent import a2a_server


MCP_CONFIG_PATH = a2a_server.MCP_CONFIG_V1.endpoint


def test_add_mcp_uses_the_callers_name_and_mints_one_otherwise(client):
    named = client.post(MCP_CONFIG_PATH, json={"url": "https://gw/mcp", "name": "env"}).json()
    assert named["name"] == "env"
    minted = client.post(MCP_CONFIG_PATH, json={"url": "https://gw2/mcp"}).json()
    assert minted["name"].startswith("mcp_") and len(minted["name"]) == 12
    assert set(client.get(MCP_CONFIG_PATH).json()["mcp_servers"]) == {"env", minted["name"], "peer_agents"}


def test_a_second_server_under_the_same_name_is_refused(client):
    assert client.post(MCP_CONFIG_PATH, json={"url": "https://gw/mcp", "name": "env"}).status_code == 200
    assert client.post(MCP_CONFIG_PATH, json={"url": "https://other/mcp", "name": "env"}).status_code == 409
    assert client.get(MCP_CONFIG_PATH).json()["mcp_servers"]["env"]["url"] == "https://gw/mcp"
