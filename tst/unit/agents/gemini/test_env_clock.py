"""Gemini on the env's clock (FD-5916): while the clock page is armed the commands the model runs go out through the
image's egress proxy, by ALL_PROXY, and to the hosts NO_PROXY names, the env's MCP servers and a plain-HTTP model
endpoint directly; the CLI itself, which sends everything through a proxy its HTTP(S)_PROXY names, NO_PROXY or not, is
handed none, so it reaches the attribution proxy and the MCP servers directly. Unarmed, it all goes out as before. What
every clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from tst.unit.agents.gemini.agent import gemini_bridge

PROXY = "http://127.0.0.1:3128"
SETTINGS = Path("/tmp/settings.json")
ATTRIBUTION_PROXY = "http://127.0.0.1:8000/litellm-gemini/token"


def _env(environ, **kwargs):
    return gemini_bridge.cli_env(environ, system_settings=SETTINGS, base_url=ATTRIBUTION_PROXY, **kwargs)


def test_on_an_armed_clock_the_models_commands_go_out_through_the_proxy_and_the_cli_directly():
    env = _env({gemini_bridge.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"}, clock_armed=True)

    assert env["ALL_PROXY"] == env["all_proxy"] == PROXY
    assert not {"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"} & set(env)
    assert {"localhost", "127.0.0.1", "host.docker.internal", "registry.internal"} <= set(env["NO_PROXY"].split(","))
    assert env["GOOGLE_GEMINI_BASE_URL"] == ATTRIBUTION_PROXY


@pytest.mark.parametrize("environ, armed", [({}, True), ({gemini_bridge.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    assert not {"ALL_PROXY", "HTTPS_PROXY", "NO_PROXY"} & set(_env(environ, clock_armed=armed))


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    for name in ("LITELLM_BASE_URL", "GOOGLE_GEMINI_BASE_URL"):
        env = _env({gemini_bridge.EGRESS_PROXY_ENV: PROXY, name: base_url}, clock_armed=True)
        assert (base_url.split("//")[1].split(":")[0] in env["NO_PROXY"].split(",")) is direct


def test_the_envs_mcp_servers_are_reached_directly_on_an_armed_clock():
    env = _env({gemini_bridge.EGRESS_PROXY_ENV: PROXY}, clock_armed=True, direct_hosts=["gateway.internal", ""])

    assert "gateway.internal" in env["NO_PROXY"].split(",")
    assert "" not in env["NO_PROXY"].split(",")
