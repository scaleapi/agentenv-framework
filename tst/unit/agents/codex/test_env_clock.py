"""Codex on the env's clock: its bridge hands the shared egress rule (tst/unit/agents/test_agent_egress.py) every MCP
server's URL, which the CLI's MCP clients connect to themselves, so on an armed clock what the CLI starts goes out
through the image's egress proxy, and to loopback, the hosts NO_PROXY names and plain-HTTP hosts directly; unarmed, it
goes out as before. The CLI, a static binary, stays on real time, and each turn's relay tells the model the env's date
(test_codex_date.py). What every clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import asyncio
import contextlib
import tomllib

import pytest

from tst.unit.agents.codex.agent import agent_egress, codex_bridge, codex_config, codex_date

PROXY = "http://127.0.0.1:3128"
SERVERS = {"env": {"url": "http://gateway:8080/mcp"}, "remote": {"url": "https://abc-123.w.modal.host/mcp"}}


def _direct(env: dict[str, str]) -> set[str]:
    return set(env["NO_PROXY"].split(","))


def test_on_an_armed_clock_what_the_cli_starts_goes_out_through_the_proxy():
    env = codex_bridge.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"}, armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert env["NODE_USE_ENV_PROXY"] == "1"
    assert {"localhost", "127.0.0.1", "host.docker.internal", "registry.internal"} <= _direct(env)


@pytest.mark.parametrize("environ, armed", [({}, True), ({agent_egress.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    env = codex_bridge.cli_env(environ, urls=[server["url"] for server in SERVERS.values()], armed=armed)

    assert not {"HTTPS_PROXY", "NO_PROXY", "NODE_USE_ENV_PROXY"} & set(env)
    assert env == codex_bridge.cli_env(environ)


def test_the_envs_plain_http_mcp_servers_are_reached_directly_and_https_ones_through_the_proxy_on_an_armed_clock():
    """mcp-remote checks an HTTPS server's certificate against the env's date, which the proxy, on real time, doesn't."""
    env = codex_bridge.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY},
                               urls=[server["url"] for server in SERVERS.values()], armed=True)

    assert "gateway" in _direct(env) and "abc-123.w.modal.host" not in _direct(env)


def test_mcp_remote_is_handed_the_proxy_and_told_to_use_it(monkeypatch, tmp_path):
    """Codex hands a stdio server only the variables its entry names, and mcp-remote's own HTTP client ignores a proxy
    unless told to take it; with no proxy variable set, as off the clock, it connects directly."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    asyncio.run(codex_config.write_mcp_servers(SERVERS))

    entries = tomllib.loads((tmp_path / "config.toml").read_text())["mcp_servers"]
    for entry in entries.values():
        assert "--enable-proxy" in entry["args"]
        assert set(agent_egress.egress_env({agent_egress.EGRESS_PROXY_ENV: PROXY}, armed=True)) <= set(entry["env_vars"])


def test_each_cli_run_starts_with_the_clock_as_it_is_then(monkeypatch):
    seen = []

    async def fake_exec(*cmd, env, **kwargs):
        seen.append(env)
        raise RuntimeError("stop here")

    monkeypatch.setenv(agent_egress.EGRESS_PROXY_ENV, PROXY)
    monkeypatch.setattr(codex_bridge.asyncio, "create_subprocess_exec", fake_exec)
    for armed in (False, True):
        monkeypatch.setattr(codex_bridge, "clock_armed", lambda armed=armed: armed)
        with pytest.raises(RuntimeError, match="stop here"):
            asyncio.run(codex_bridge._run_codex_once("go", None, mcp_urls=["https://abc-123.w.modal.host/mcp"]))

    assert ["HTTPS_PROXY" in env for env in seen] == [False, True]
    assert "abc-123.w.modal.host" not in _direct(seen[1])


def _cli_command(monkeypatch, relay) -> list[str]:
    """The command one CLI run starts with, the relay a stand-in for ``model_relay``."""
    seen = []

    async def fake_exec(*cmd, **kwargs):
        seen.append(list(cmd))
        raise RuntimeError("stop here")

    monkeypatch.setattr(codex_bridge, "model_relay", relay)
    monkeypatch.setattr(codex_bridge.asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(RuntimeError, match="stop here"):
        asyncio.run(codex_bridge._run_codex_once("go", None, config_overrides=["features.goals=false"]))
    [cmd] = seen
    return cmd


def test_each_cli_run_calls_the_endpoint_through_a_relay_that_puts_the_env_date_in_its_model_calls(monkeypatch):
    opened, closed = [], []

    @contextlib.asynccontextmanager
    async def relay(upstream, calls, **options):
        opened.append((upstream, options))
        yield "http://127.0.0.1:9/token"
        closed.append(upstream)

    monkeypatch.setenv("LITELLM_BASE_URL", "https://llm.example.com/v1/")
    cmd = _cli_command(monkeypatch, relay)

    assert opened == [("https://llm.example.com", {"rewrite": codex_date.env_date_rewrite})]
    assert closed == ["https://llm.example.com"]  # however the run ends
    assert cmd[-4:] == ["-c", "features.goals=false",
                        "-c", 'model_providers.litellm.base_url="http://127.0.0.1:9/token/openai/v1"']


def test_without_an_endpoint_there_is_no_provider_to_relay_and_the_cli_runs_as_before(monkeypatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)

    assert _cli_command(monkeypatch, None)[-2:] == ["-c", "features.goals=false"]
