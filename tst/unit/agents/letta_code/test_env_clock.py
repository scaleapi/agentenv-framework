"""Letta Code on the env's clock (FD-5916): while the clock page is armed the CLI goes out through the image's egress
proxy, and to the hosts NO_PROXY names, the model relay on loopback and the env's MCP servers, which the CLI connects to
itself, directly; unarmed, it goes out as before. What every clocked image shares is checked in
tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import pytest

from tst.unit.agents.letta_code.agent import letta_code_bridge as bridge

PROXY = "http://127.0.0.1:3128"


@pytest.fixture(autouse=True)
def _sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path))


def _env(monkeypatch, environ: dict[str, str], *, direct=(), armed: bool) -> dict[str, str]:
    for name in (bridge.EGRESS_PROXY_ENV, "NO_PROXY", "no_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY",
                 "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    return bridge._context_env("ctx", direct, armed=armed)


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy(monkeypatch):
    env = _env(monkeypatch, {bridge.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"}, armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    # The model relay the CLI is pointed at is on loopback.
    assert {"localhost", "127.0.0.1", "host.docker.internal", "registry.internal"} <= set(env["NO_PROXY"].split(","))
    assert env["NODE_USE_ENV_PROXY"] == "1"


@pytest.mark.parametrize("environ, armed", [({}, True), ({bridge.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(monkeypatch, environ, armed):
    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(_env(monkeypatch, environ, armed=armed))


def test_the_envs_mcp_servers_are_reached_directly_on_an_armed_clock(monkeypatch):
    hosts = bridge._mcp_hosts({"env": {"url": "http://10.0.0.5:8080/mcp"},
                               "peers": {"url": "https://gateway.example.com/mcp"}})
    env = _env(monkeypatch, {bridge.EGRESS_PROXY_ENV: PROXY}, direct=hosts, armed=True)

    assert {"10.0.0.5", "gateway.example.com"} <= set(env["NO_PROXY"].split(","))
