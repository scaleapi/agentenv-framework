"""mcp_compat's client session: mcp's own HTTP client and its timeouts unless the caller gives one."""
from __future__ import annotations

import contextlib

import pytest

pytest.importorskip("mcp")

import mcp.client.streamable_http as streamable_http  # noqa: E402

from agentenv_protocol.mcp_compat import client_session  # noqa: E402


class _Stop(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(("kwargs", "connect", "read"), [({}, 30, 300), ({"timeout": 5, "read_timeout": 60}, 5, 60)])
async def test_a_session_keeps_mcps_client_timeouts_unless_given_its_own(monkeypatch, kwargs, connect, read):
    seen = []

    @contextlib.asynccontextmanager
    async def streams(url, *, http_client):
        seen.append(http_client)
        raise _Stop
        yield

    monkeypatch.setattr(streamable_http, "streamable_http_client", streams)
    with pytest.raises(_Stop):
        async with client_session("http://127.0.0.1:1/mcp", headers={"AgentEnv-Role": "grader"}, **kwargs):
            pass

    [client] = seen
    assert (client.timeout.connect, client.timeout.read) == (connect, read)
    assert client.headers["AgentEnv-Role"] == "grader"
