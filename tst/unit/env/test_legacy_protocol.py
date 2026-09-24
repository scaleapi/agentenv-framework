"""Unit tests for the legacy MCP data-plane reset protocol."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env import legacy_protocol


@pytest.mark.asyncio
async def test_reset_via_rest_posts_and_returns_json():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"ok": True})
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("agent_env.env.legacy_protocol.httpx.AsyncClient", return_value=client):
        result = await legacy_protocol.reset_via_rest("http://gw/svc/mcp-email", "/data/seed.json")

    assert result == {"ok": True}
    resp.raise_for_status.assert_called_once()
    url, kwargs = client.post.call_args.args[0], client.post.call_args.kwargs
    assert url == "http://gw/svc/mcp-email/api/reset"
    assert kwargs["json"] == {"mock_data_path": "/data/seed.json"}
    assert kwargs["timeout"] == 30


@pytest.mark.asyncio
async def test_reset_uses_rest_on_success(monkeypatch):
    rest = AsyncMock(return_value={"ok": True})
    mcp = AsyncMock()
    monkeypatch.setattr(legacy_protocol, "reset_via_rest", rest)
    monkeypatch.setattr(legacy_protocol, "reset_via_mcp_tool", mcp)

    result = await legacy_protocol.reset("http://gw", "email", "/data/seed.json")

    assert result == {"ok": True}
    rest.assert_awaited_once_with("http://gw/svc/mcp-email", "/data/seed.json")
    mcp.assert_not_called()


@pytest.mark.asyncio
async def test_reset_falls_back_to_mcp_tool(monkeypatch):
    rest = AsyncMock(side_effect=RuntimeError("rest down"))
    mcp = AsyncMock()
    monkeypatch.setattr(legacy_protocol, "reset_via_rest", rest)
    monkeypatch.setattr(legacy_protocol, "reset_via_mcp_tool", mcp)

    result = await legacy_protocol.reset("http://gw", "email", "/data/seed.json", max_retries=2)

    assert result is None
    mcp.assert_awaited_once_with("http://gw/mcp", "email", "/data/seed.json", max_retries=2)


def test_service_base_url_mcp_vs_website():
    assert legacy_protocol.environment_base_url("http://gw", "email", mcp=True) == "http://gw/svc/mcp-email"
    assert legacy_protocol.environment_base_url("http://gw", "slack", mcp=False) == "http://gw/svc/slack"


def _fake_client(resp):
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.mark.asyncio
async def test_reset_via_rest_no_body_when_no_seed():
    resp = MagicMock(); resp.raise_for_status = MagicMock(); resp.json = MagicMock(return_value={"ok": True})
    client = _fake_client(resp)
    with patch("agent_env.env.legacy_protocol.httpx.AsyncClient", return_value=client) as ctor:
        await legacy_protocol.reset_via_rest("http://gw/svc/slack", timeout=60)

    ctor.assert_called_once_with(verify=True)
    assert client.post.call_args.args[0] == "http://gw/svc/slack/api/reset"
    assert client.post.call_args.kwargs["json"] is None
    assert client.post.call_args.kwargs["timeout"] == 60


@pytest.mark.asyncio
async def test_add_via_rest_posts_file_path():
    resp = MagicMock(); resp.raise_for_status = MagicMock(); resp.json = MagicMock(return_value={"ok": True})
    client = _fake_client(resp)
    with patch("agent_env.env.legacy_protocol.httpx.AsyncClient", return_value=client) as ctor:
        r = await legacy_protocol.add_via_rest("http://gw/svc/slack", "/tmp/data/seed.json")

    assert r == {"ok": True}
    ctor.assert_called_once_with(verify=True)
    assert client.post.call_args.args[0] == "http://gw/svc/slack/api/add"
    assert client.post.call_args.kwargs["json"] == {"file_path": "/tmp/data/seed.json"}


@pytest.mark.asyncio
async def test_export_state_gets_mcp_endpoint_and_returns_json():
    resp = MagicMock(); resp.raise_for_status = MagicMock(); resp.json = MagicMock(return_value={"emails": []})
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch("agent_env.env.legacy_protocol.httpx.AsyncClient", return_value=client):
        r = await legacy_protocol.export_state("http://gw", "email")

    assert r == {"emails": []}
    assert client.get.call_args.args[0] == "http://gw/svc/mcp-email/export-state"
    assert client.get.call_args.kwargs["timeout"] == 60
