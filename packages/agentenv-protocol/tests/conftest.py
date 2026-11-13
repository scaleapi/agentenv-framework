"""No env clock unless a test gives one: a turn with MCP servers registered otherwise asks each for get_time
(``agentenv_protocol.a2a_agent.env_clock``), and the servers these tests register don't exist."""

import pytest


@pytest.fixture(autouse=True)
def no_env_clock(monkeypatch):
    from agentenv_protocol.a2a_agent import framework

    async def _none(servers, timeout_s=None):
        return None

    monkeypatch.setattr(framework, "find_env_clock", _none)
