"""deploy_agent relays the env card's name as the MCP server alias to agents that accept one."""

import httpx
import pytest

from agent_env.task_step.task_steps.deploy_agent import _card_name, _mcp_add_body

_LEGACY_CARD_EXT = {"params": {"endpoint": "/ext/mcp-config", "methods": {"add": {"method": "POST", "request": {"required": ["url"]}}}}}
_NAMING_CARD_EXT = {"params": {"endpoint": "/ext/mcp-config", "methods": {"add": {"method": "POST", "request": {"required": ["url"], "optional": ["name"]}}}}}


def test_agents_that_do_not_advertise_name_get_the_legacy_body():
    assert _mcp_add_body(_LEGACY_CARD_EXT, "https://gw/mcp", None, "env") == {"url": "https://gw/mcp"}
    assert _mcp_add_body(_LEGACY_CARD_EXT, "https://gw/mcp", {"Authorization": "Bearer t"}, "env") == {
        "url": "https://gw/mcp", "headers": {"Authorization": "Bearer t"},
    }


def test_the_card_name_is_relayed_verbatim_whatever_it_is():
    for name in ("env", "crm", "AgentEnvGateway"):  # default, declared, and an env on an older gateway image
        assert _mcp_add_body(_NAMING_CARD_EXT, "https://gw/mcp", None, name) == {"url": "https://gw/mcp", "name": name}


def test_no_card_name_means_the_agent_mints_its_own():
    assert _mcp_add_body(_NAMING_CARD_EXT, "https://gw/mcp", None, None) == {"url": "https://gw/mcp"}
    assert "name" not in _mcp_add_body({"params": {"methods": {"add": {"request": {"optional": None}}}}}, "https://gw/mcp", None, "env")


@pytest.mark.asyncio
async def test_card_name_is_read_from_the_environment_card_and_absent_on_any_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/ok"):
            return httpx.Response(200, json={"name": "env", "protocolVersion": "1"})
        if request.url.path.endswith("/nameless"):
            return httpx.Response(200, json={"protocolVersion": "1"})
        if request.url.path.endswith("/text"):
            return httpx.Response(200, text="not json")
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await _card_name(client, "https://gw/.well-known/ok") == "env"
        assert await _card_name(client, "https://gw/.well-known/nameless") is None
        assert await _card_name(client, "https://gw/.well-known/text") is None
        assert await _card_name(client, "https://gw/.well-known/missing") is None
        assert await _card_name(client, None) is None
