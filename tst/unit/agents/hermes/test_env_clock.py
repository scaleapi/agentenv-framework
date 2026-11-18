"""Hermes on the env's clock (FD-5916): while the clock page is armed the turn's process goes out through the image's
egress proxy, and to the hosts NO_PROXY names, a plain-HTTP model endpoint and the turn's MCP servers, which Hermes
connects to itself, directly; unarmed, it goes out as before. What every clocked image shares is checked in
tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from tst.unit.agents.hermes.agent import hermes_bridge

PROXY = "http://127.0.0.1:3128"


@pytest.fixture(autouse=True)
def _model_hosts_resolve_publicly(monkeypatch):
    """A model host these tests name over HTTPS resolves to a public address, with no lookup leaving the machine."""
    monkeypatch.setattr(hermes_bridge, "_GLOBAL_HOSTS", {})
    monkeypatch.setattr(hermes_bridge.socket, "getaddrinfo", lambda host, port: [(None, None, None, "", ("93.184.216.34", 0))])
HOME = Path("/data/.hermes/profiles/ctx-0")


def test_on_an_armed_clock_the_turn_goes_out_through_the_proxy():
    env = hermes_bridge.cli_env({hermes_bridge.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"}, HOME,
                                events_fd=7, clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert {"localhost", "host.docker.internal", "registry.internal"} <= set(env["NO_PROXY"].split(","))
    assert (env["HERMES_HOME"], env["AGENTENV_EVENTS_FD"]) == (str(HOME), "7")


@pytest.mark.parametrize("environ, armed", [({}, True), ({hermes_bridge.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_turn_goes_out_directly_as_before(environ, armed):
    env = hermes_bridge.cli_env(environ, HOME, events_fd=7, clock_armed=armed)

    assert env == {**environ, "HERMES_HOME": str(HOME), "AGENTENV_EVENTS_FD": "7"}


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = hermes_bridge.cli_env({hermes_bridge.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": base_url}, HOME,
                                clock_armed=True)
    assert (base_url.split("//")[1].split(":")[0] in env["NO_PROXY"].split(",")) is direct


def test_the_turns_mcp_servers_are_reached_directly_on_an_armed_clock():
    servers = {"env": {"url": "http://gateway.internal:8080/mcp"}, "docs": {"url": "https://10.1.2.3/mcp"}}
    env = hermes_bridge.cli_env({hermes_bridge.EGRESS_PROXY_ENV: PROXY}, HOME, mcp_servers=servers, clock_armed=True)

    assert {"gateway.internal", "10.1.2.3"} <= set(env["NO_PROXY"].split(","))
