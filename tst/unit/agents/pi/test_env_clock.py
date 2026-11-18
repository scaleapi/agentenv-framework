"""pi on the env's clock (FD-5916): while the clock page is armed the CLI goes out through the image's egress proxy,
and to the hosts NO_PROXY names, the env's MCP servers and a plain-HTTP model endpoint directly; unarmed, it goes out
as before. What every clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import pytest

from tst.unit.agents.pi.agent import pi_agent

PROXY = "http://127.0.0.1:3128"


@pytest.fixture(autouse=True)
def _model_hosts_resolve_publicly(monkeypatch):
    """A model host these tests name over HTTPS resolves to a public address, with no lookup leaving the machine."""
    monkeypatch.setattr(pi_agent, "_GLOBAL_HOSTS", {})
    monkeypatch.setattr(pi_agent.socket, "getaddrinfo", lambda host, port: [(None, None, None, "", ("93.184.216.34", 0))])
ENDPOINT = "https://litellm.example.com"


def _direct(env: dict[str, str]) -> set[str]:
    return set(env["NO_PROXY"].split(","))


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy():
    env = pi_agent.cli_env({pi_agent.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal", "KEPT": "1"}, ENDPOINT,
                           clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert env["NODE_USE_ENV_PROXY"] == "1" and env["KEPT"] == "1"
    assert {"localhost", "host.docker.internal", "registry.internal"} <= _direct(env)


@pytest.mark.parametrize("environ, armed", [({"KEPT": "1"}, True), ({pi_agent.EGRESS_PROXY_ENV: PROXY, "KEPT": "1"}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    env = pi_agent.cli_env(environ, ENDPOINT, {"env": {"url": "http://10.0.0.7:8080/mcp"}}, clock_armed=armed)

    assert env == environ


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = pi_agent.cli_env({pi_agent.EGRESS_PROXY_ENV: PROXY}, base_url, clock_armed=True)

    assert (base_url.split("//")[1].split(":")[0] in _direct(env)) is direct


def test_the_envs_mcp_servers_which_the_cli_calls_itself_are_reached_directly_on_an_armed_clock():
    servers = {"env": {"url": "http://gateway.internal:8080/mcp"}, "files": {"url": "https://mcp.example.com/mcp"},
               pi_agent.PEER_MCP_NAME: {"url": pi_agent.PEER_MCP_URL}}

    env = pi_agent.cli_env({pi_agent.EGRESS_PROXY_ENV: PROXY}, ENDPOINT, servers, clock_armed=True)

    assert {"gateway.internal", "mcp.example.com", "127.0.0.1"} <= _direct(env)
    assert "litellm.example.com" not in _direct(env)
