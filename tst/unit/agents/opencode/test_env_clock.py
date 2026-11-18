"""OpenCode on the env's clock (FD-5916): while the clock page is armed the CLI goes out through the image's egress
proxy, and to the hosts NO_PROXY names, the env's MCP servers, which it connects to itself, and a plain-HTTP model
endpoint directly; unarmed, it goes out as before. What every clocked image shares is checked in
tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import pytest

from tst.unit.agents.opencode.agent import opencode_bridge

PROXY = "http://127.0.0.1:3128"


@pytest.fixture(autouse=True)
def _model_hosts_resolve_publicly(monkeypatch):
    """A model host these tests name over HTTPS resolves to a public address, with no lookup leaving the machine."""
    monkeypatch.setattr(opencode_bridge.socket, "getaddrinfo", lambda host, port: [(None, None, None, "", ("93.184.216.34", 0))])


@pytest.fixture(autouse=True)
def _sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(opencode_bridge, "SESSIONS_ROOT", str(tmp_path))


def _direct(env: dict[str, str]) -> set[str]:
    return set(env["NO_PROXY"].split(","))


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy():
    env = opencode_bridge.cli_env("ctx", {opencode_bridge.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"},
                                  clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert {"localhost", "host.docker.internal", "registry.internal"} <= _direct(env)
    # Still the context's own session store.
    assert env["XDG_DATA_HOME"] == str(opencode_bridge.context_data_dir("ctx") / "share")


@pytest.mark.parametrize("environ, armed", [({}, True), ({opencode_bridge.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    env = opencode_bridge.cli_env("ctx", environ, clock_armed=armed)

    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(env)
    assert set(env) == {*environ, "XDG_DATA_HOME", "XDG_STATE_HOME"}


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = opencode_bridge.cli_env("ctx", {opencode_bridge.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": base_url},
                                  clock_armed=True)
    assert (base_url.split("//")[1].split(":")[0] in _direct(env)) is direct


def test_the_envs_mcp_servers_are_reached_directly_on_an_armed_clock():
    servers = {"env": {"url": "http://host.docker.internal:8080/mcp"},
               "gateway": {"url": "https://mcp.gateway.internal/mcp", "headers": {"x": "y"}}}
    env = opencode_bridge.cli_env("ctx", {opencode_bridge.EGRESS_PROXY_ENV: PROXY}, mcp_servers=servers,
                                  clock_armed=True)

    assert {"host.docker.internal", "mcp.gateway.internal"} <= _direct(env)
