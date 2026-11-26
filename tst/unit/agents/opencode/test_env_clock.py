"""OpenCode on the env's clock: its bridge hands the shared egress rule (tst/unit/agents/test_agent_egress.py) the
model endpoint's URL and every MCP server's, which the CLI connects to itself, so on an armed clock it goes out through
the image's egress proxy, and to loopback, the hosts NO_PROXY names and plain-HTTP hosts directly; unarmed, it goes out
as before. Where the image follows the clock, the CLI itself runs on real time and each turn's relay tells the model the
env's date (test_opencode_date.py). What every clocked image shares is checked in
tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tst.unit.agents.opencode.agent import agent_egress, opencode_bridge, opencode_date

PROXY = "http://127.0.0.1:3128"


@pytest.fixture(autouse=True)
def _sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(opencode_bridge, "SESSIONS_ROOT", str(tmp_path))


def _direct(env: dict[str, str]) -> set[str]:
    return set(env["NO_PROXY"].split(","))


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy():
    env = opencode_bridge.cli_env("ctx", {agent_egress.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"},
                                  clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert {"localhost", "host.docker.internal", "registry.internal"} <= _direct(env)
    assert env["XDG_DATA_HOME"] == str(opencode_bridge.context_data_dir("ctx") / "share")


@pytest.mark.parametrize("environ, armed", [({}, True), ({agent_egress.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    env = opencode_bridge.cli_env("ctx", environ, clock_armed=armed)

    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(env)
    assert set(env) == {*environ, "XDG_DATA_HOME", "XDG_STATE_HOME"}


@pytest.mark.parametrize("follows", [True, False])
def test_where_the_image_follows_the_clock_the_cli_starts_on_real_time_so_its_sessions_keep_their_order(
        monkeypatch, follows):
    """libfaketime takes the marker out of the CLI's environment, so what the CLI starts reads the env's time; where
    nothing follows the clock, as on an install target, the CLI starts as before."""
    monkeypatch.setattr(opencode_bridge, "follows", lambda: follows)

    for armed in (True, False):
        env = opencode_bridge.cli_env("ctx", {agent_egress.EGRESS_PROXY_ENV: PROXY}, clock_armed=armed)
        assert env.get("AGENTENV_REAL_TIME") == ("1" if follows else None)


def test_every_turns_relay_puts_the_env_date_in_its_model_calls(monkeypatch):
    seen = {}

    @contextlib.asynccontextmanager
    async def relay(upstream, calls, **options):
        seen.update(options)
        yield "http://relay"

    async def turn(prompt, context_id, **kwargs):
        seen["base_url"] = kwargs["base_url"]
        return opencode_bridge.OpencodeResult(response="", tool_call_count=0, session_id=None)

    monkeypatch.setattr(opencode_bridge, "model_relay", relay)
    monkeypatch.setattr(opencode_bridge, "_run_opencode", turn)
    asyncio.run(opencode_bridge.run_opencode("go", "ctx", web_tools=True))

    assert seen == {"rewrite": opencode_date.env_date_rewrite, "base_url": "http://relay"}


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = opencode_bridge.cli_env("ctx", {agent_egress.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": base_url},
                                  clock_armed=True)
    assert (base_url.split("//")[1].split(":")[0] in _direct(env)) is direct


def test_the_envs_plain_http_mcp_servers_are_reached_directly_and_https_ones_through_the_proxy_on_an_armed_clock():
    """Bun checks an HTTPS server's certificate against the env's date, which the proxy, on real time, doesn't."""
    servers = {"env": {"url": "http://gateway:8080/mcp"},
               "remote": {"url": "https://abc-123.w.modal.host/mcp", "headers": {"x": "y"}}}
    env = opencode_bridge.cli_env("ctx", {agent_egress.EGRESS_PROXY_ENV: PROXY}, mcp_servers=servers,
                                  clock_armed=True)

    assert "gateway" in _direct(env) and "abc-123.w.modal.host" not in _direct(env)
