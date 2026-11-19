"""Claude Code's bridge on the env's clock (FD-5916): while the clock page is armed, the CLI goes out through the
egress proxy, env MCP servers included, and to loopback, the hosts NO_PROXY names and a plain-HTTP model endpoint
directly; unarmed, it goes out as before. What every clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import asyncio
import json
import types
from urllib.parse import urlsplit

import pytest

from tst.unit.agents.claude_code.agent import AGENT_DIR, claude_bridge
from tst.util.build_context import sent

PROXY = "http://127.0.0.1:3128"


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy_and_to_loopback_directly():
    env = claude_bridge.cli_env({claude_bridge.EGRESS_PROXY_ENV: PROXY}, clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert env["NO_PROXY"] == env["no_proxy"] == "localhost,127.0.0.1,::1,host.docker.internal"
    assert env["NODE_USE_ENV_PROXY"] == "1"


def _spawned_env(monkeypatch, tmp_path, armed: bool) -> dict[str, str]:
    """The environment _spawn_claude starts the CLI with, for an env MCP server and the peer-agents one."""
    spawned = {}

    async def fake_exec(*cmd, env, **_):
        spawned.update(env)
        raise RuntimeError("not starting the CLI")

    monkeypatch.setattr(claude_bridge.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(claude_bridge, "clock_armed", lambda: armed)
    monkeypatch.setattr(claude_bridge.os, "environ", {claude_bridge.EGRESS_PROXY_ENV: PROXY})
    monkeypatch.setattr(claude_bridge, "WORKSPACE_DIR", tmp_path)
    servers = {"env": {"url": "https://abc-123.w.modal.host/mcp"}, "peers": {"url": "http://127.0.0.1:8000/peers/mcp"}}
    with pytest.raises(RuntimeError, match="not starting the CLI"):
        asyncio.run(claude_bridge._spawn_claude("hi", "s", False, mcp_servers=servers))
    return spawned


def test_on_an_armed_clock_an_env_mcp_server_goes_through_the_proxy_and_a_loopback_one_stays_direct(monkeypatch, tmp_path):
    # Its certificate would be checked against the env's date, which can be before it was issued.
    env = _spawned_env(monkeypatch, tmp_path, armed=True)

    hosts = env["NO_PROXY"].split(",")
    assert env["HTTPS_PROXY"] == env["https_proxy"] == PROXY
    assert "abc-123.w.modal.host" not in hosts and "127.0.0.1" in hosts


def test_unarmed_the_cli_reaches_every_mcp_server_directly(monkeypatch, tmp_path):
    env = _spawned_env(monkeypatch, tmp_path, armed=False)

    assert not {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"} & set(env)


def test_hosts_the_agent_was_deployed_to_reach_directly_stay_direct_on_an_armed_clock():
    # A private registry the agent's NO_PROXY names worked before the clock was armed; the proxy would refuse it.
    env = claude_bridge.cli_env({claude_bridge.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal, .corp",
                                 "no_proxy": "mirror.internal"}, clock_armed=True)

    hosts = env["NO_PROXY"].split(",")
    assert env["no_proxy"] == env["NO_PROXY"]
    assert {"registry.internal", ".corp", "mirror.internal", "localhost"} <= set(hosts)
    assert len(hosts) == len(set(hosts)) and "" not in hosts


@pytest.mark.parametrize("base_url, direct", [
    ("http://10.0.0.5:4000", True),  # plain HTTP: nothing to check at the env's date
    # TLS, at a private address or a public one: its certificate is checked through the proxy, on real time, and the
    # agent's proxy reaches private networks (root.conf's AGENTENV_EGRESS_PRIVATE)
    ("https://10.0.0.5:4000", False),
    ("https://[fd00::5]:4000", False),
    ("https://litellm.internal:4000", False),
    ("https://litellm.example.com", False),
])
def test_a_model_endpoint_over_plain_http_is_reached_directly_and_one_over_tls_through_the_proxy_on_an_armed_clock(
        base_url, direct):
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

