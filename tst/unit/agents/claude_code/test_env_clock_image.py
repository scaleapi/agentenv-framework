"""Claude Code's bridge on the env's clock (FD-5916): while the clock page is armed, the CLI goes out through the
egress proxy, and to the env's MCP servers, the hosts NO_PROXY names and a plain-HTTP model endpoint directly; unarmed,
it goes out as before. What every clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import json
import types
from urllib.parse import urlsplit

import pytest

from tst.unit.agents.claude_code.agent import AGENT_DIR, claude_bridge
from tst.util.build_context import sent

PROXY = "http://127.0.0.1:3128"


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy_and_to_the_envs_servers_directly():
    env = claude_bridge.cli_env({claude_bridge.EGRESS_PROXY_ENV: PROXY}, direct_hosts=["gateway", "", "gateway"],
                                clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert env["NO_PROXY"] == env["no_proxy"] == "localhost,127.0.0.1,::1,host.docker.internal,gateway"
    assert env["NODE_USE_ENV_PROXY"] == "1"


def test_hosts_the_agent_was_deployed_to_reach_directly_stay_direct_on_an_armed_clock():
    # A private registry the agent's NO_PROXY names worked before the clock was armed; the proxy would refuse it.
    env = claude_bridge.cli_env({claude_bridge.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal, .corp",
                                 "no_proxy": "mirror.internal"}, direct_hosts=["gateway"], clock_armed=True)

    hosts = env["NO_PROXY"].split(",")
    assert env["no_proxy"] == env["NO_PROXY"]
    assert {"registry.internal", ".corp", "mirror.internal", "gateway", "localhost"} <= set(hosts)
    assert len(hosts) == len(set(hosts)) and "" not in hosts


@pytest.mark.parametrize("base_url, direct", [
    ("http://10.0.0.5:4000", True),  # plain HTTP, private: nothing to check at the env's date, and the proxy refuses it
    ("https://10.0.0.5:4000", True),  # private, over TLS too: the proxy refuses an address that isn't public
    ("https://[fd00::5]:4000", True),
    ("https://litellm.internal:4000", True),  # a name that resolves to a private address
    ("https://8.8.8.8", False),  # TLS to a public address: its certificate is checked through the proxy
    ("https://litellm.example.com", False),  # a name that resolves to a public one

])
def test_a_model_endpoint_over_plain_http_or_at_a_private_address_is_reached_directly_on_an_armed_clock(
        monkeypatch, base_url, direct):
    resolves = {"litellm.internal": "10.1.2.3", "litellm.example.com": "93.184.216.34"}
    monkeypatch.setattr(claude_bridge.socket, "getaddrinfo",
                        lambda host, port: [(None, None, None, "", (resolves[host], 0))])
    env = claude_bridge.cli_env({claude_bridge.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": base_url,
                                 "LITELLM_API_KEY": "k"}, clock_armed=True)
    assert (urlsplit(base_url).hostname in env["NO_PROXY"].split(",")) is direct and env["HTTPS_PROXY"] == PROXY


@pytest.mark.parametrize("environ, armed", [({}, True), ({claude_bridge.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    # A model endpoint at a private address, which the proxy refuses, keeps working until a task arms the clock.
    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(claude_bridge.cli_env(environ, clock_armed=armed))


def _clockpage():
    """The image's clock page writer, whose encoding libfaketime and the egress proxy read too."""
    # Run from its source, so no __pycache__ lands in agent-env's sources.
    module = types.ModuleType("clockpage")
    path = sent(AGENT_DIR, "clock/clockpage.py")
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def _page(armed: int, seq: int = 2) -> bytes:
    return _clockpage().encode(seq, 1_900_000_000 * 10**9 if armed else 0, 10**18 if armed else 0, 1.0, armed, 3, 0, 0)


@pytest.mark.parametrize("page, armed", [
    (_page(1), True),
    (_page(0), False),
    # A write under way (seq odd) that never finishes: the armed flag it left is the best guess.
    (_page(1, seq=3), True),
    (_page(1)[:40], False),  # not a page
    (b"\0" * 80, False),  # not one yet: no magic
    (None, False),  # no page: an image without the clock, or install_agent's target
])
def test_the_clock_is_armed_when_the_page_libfaketime_reads_says_so(tmp_path, page, armed):
    path = tmp_path / "agentenv-clock"
    if page is not None:
        path.write_bytes(page)
    assert claude_bridge.clock_armed(str(path)) is armed


@pytest.mark.parametrize("status", [
    {"connected": False, "error": "URLError: gateway restarting"},  # the follower lost its gateway
    {"connected": False, "waiting": "no config yet"},  # the follower restarted, before its config is read
    {"connected": True, "armed": False, "version": None},  # ... and before its first watch answers
])
def test_while_the_follower_reconnects_or_restarts_the_page_still_routes_the_cli_through_the_proxy(tmp_path, status):
    # The page keeps the last armed clock in both windows, so the CLI reads the env's time and must go through egress.
    (tmp_path / "status.json").write_text(json.dumps(status))
    page = tmp_path / "agentenv-clock"
    page.write_bytes(_page(1))
    env = claude_bridge.cli_env({claude_bridge.EGRESS_PROXY_ENV: PROXY}, clock_armed=claude_bridge.clock_armed(str(page)))
    assert env["HTTPS_PROXY"] == PROXY
