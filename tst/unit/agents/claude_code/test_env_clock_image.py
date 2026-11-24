"""Claude Code's bridge on the env's clock: it hands the shared egress rule (tst/unit/agents/test_agent_egress.py) the
model endpoint's URL and every MCP server's, so on an armed clock an env MCP server over TLS goes through the egress
proxy and loopback and plain-HTTP hosts go direct; unarmed, the CLI goes out as before. Every run keeps its transcripts,
which the CLI's own cleanup would delete on a clock ahead of the files' real times. What every clocked image shares is
checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import asyncio
import json
import re
from urllib.parse import urlsplit

import pytest

from tst.unit.agents.claude_code.agent import AGENT_DIR, agent_egress, claude_bridge
from tst.util.build_context import sent

PROXY = "http://127.0.0.1:3128"
# How many days a JavaScript date can be from 1970 either way (ECMAScript's time value range).
JS_DATE_RANGE_DAYS = 100_000_000
ARMED_PAGE = agent_egress.clockpage.encode(2, 1_900_000_000 * 10**9, 10**18, 1.0, 1, 3, 0, 0)


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy_and_to_loopback_directly():
    env = claude_bridge.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY}, clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert env["NO_PROXY"] == env["no_proxy"] == "localhost,127.0.0.1,::1,host.docker.internal"
    assert env["NODE_USE_ENV_PROXY"] == "1"


@pytest.mark.parametrize("resume", [False, True])
def test_every_cli_run_keeps_transcripts_whatever_date_the_clock_shows(resume):
    command = claude_bridge._command("sess", resume, model=None, allowed_tools=None, mcp_servers=None,
                                     system_prompt=None, output_format=None, role=None, effort=None, max_turns=None)

    assert command[command.index("--setting-sources") + 1] == "user"
    assert json.loads(command[command.index("--settings") + 1])["cleanupPeriodDays"] >= JS_DATE_RANGE_DAYS


def _spawned_env(monkeypatch, tmp_path, armed: bool) -> dict[str, str]:
    """The environment _spawn_claude starts the CLI with, for an env MCP server over TLS, one over plain HTTP and the
    peer-agents one."""
    spawned = {}

    async def fake_exec(*cmd, env, **_):
        spawned.update(env)
        raise RuntimeError("not starting the CLI")

    monkeypatch.setattr(claude_bridge.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(claude_bridge, "clock_armed", lambda: armed)
    monkeypatch.setattr(claude_bridge.os, "environ", {agent_egress.EGRESS_PROXY_ENV: PROXY})
    monkeypatch.setattr(claude_bridge, "WORKSPACE_DIR", tmp_path)
    servers = {"env": {"url": "https://abc-123.w.modal.host/mcp"}, "local": {"url": "http://gateway:8080/mcp"},
               "peers": {"url": "http://127.0.0.1:8000/peers/mcp"}}
    with pytest.raises(RuntimeError, match="not starting the CLI"):
        asyncio.run(claude_bridge._spawn_claude("hi", "s", False, mcp_servers=servers))
    return spawned


def test_on_an_armed_clock_an_env_mcp_server_over_tls_goes_through_the_proxy_and_the_others_direct(monkeypatch,
                                                                                                  tmp_path):
    env = _spawned_env(monkeypatch, tmp_path, armed=True)

    hosts = env["NO_PROXY"].split(",")
    assert env["HTTPS_PROXY"] == env["https_proxy"] == PROXY
    assert "abc-123.w.modal.host" not in hosts and {"gateway", "127.0.0.1"} <= set(hosts)


def test_unarmed_the_cli_reaches_every_mcp_server_directly(monkeypatch, tmp_path):
    env = _spawned_env(monkeypatch, tmp_path, armed=False)

    assert not {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"} & set(env)


@pytest.mark.parametrize("base_url, direct", [
    ("http://10.0.0.5:4000", True),
    ("https://10.0.0.5:4000", False),
    ("https://[fd00::5]:4000", False),
    ("https://litellm.internal:4000", False),
    ("https://litellm.example.com", False),
])
def test_a_model_endpoint_over_plain_http_is_reached_directly_and_one_over_tls_through_the_proxy_on_an_armed_clock(
        base_url, direct):
    env = claude_bridge.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": base_url,
                                 "LITELLM_API_KEY": "k"}, clock_armed=True)
    assert (urlsplit(base_url).hostname in env["NO_PROXY"].split(",")) is direct and env["HTTPS_PROXY"] == PROXY


@pytest.mark.parametrize("environ, armed", [({}, True), ({agent_egress.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(claude_bridge.cli_env(environ, clock_armed=armed))


@pytest.mark.parametrize("status", [
    {"connected": False, "error": "URLError: gateway restarting"},
    {"connected": False, "waiting": "no config yet"},
    {"connected": True, "armed": False, "version": None},
], ids=["lost-its-gateway", "restarted-before-its-config", "restarted-before-its-first-watch"])
def test_while_the_follower_reconnects_or_restarts_the_page_still_routes_the_cli_through_the_proxy(tmp_path, status):
    (tmp_path / "status.json").write_text(json.dumps(status))
    page = tmp_path / "agentenv-clock"
    page.write_bytes(ARMED_PAGE)
    env = claude_bridge.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY},
                                clock_armed=agent_egress.clock_armed(str(page)))
    assert env["HTTPS_PROXY"] == PROXY


def test_an_installed_agent_has_the_egress_rule_beside_its_bridge_as_the_image_does():
    listed = re.search(r'^AGENT_FILES="(.+)"$', (AGENT_DIR / "install.sh").read_text(), re.MULTILINE).group(1).split()

    assert {"agent_clock/agent_egress.py", "clock/clockpage.py"} <= set(listed)
    assert all(sent(AGENT_DIR, name).is_file() for name in listed)
