"""Codex on the env's clock (FD-5916): while the clock page is armed the CLI goes out through the image's egress proxy,
and to the hosts NO_PROXY names and a plain-HTTP model endpoint directly; unarmed, it goes out as before. What every
clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import pytest

from tst.unit.agents.codex.agent import codex_bridge

PROXY = "http://127.0.0.1:3128"



def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy():
    env = codex_bridge.cli_env({codex_bridge.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"},
                               clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert {"localhost", "host.docker.internal", "registry.internal"} <= set(env["NO_PROXY"].split(","))


@pytest.mark.parametrize("environ, armed", [({}, True), ({codex_bridge.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(codex_bridge.cli_env(environ, clock_armed=armed))


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = codex_bridge.cli_env({codex_bridge.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": base_url}, clock_armed=True)
    assert (base_url.split("//")[1].split(":")[0] in env["NO_PROXY"].split(",")) is direct
