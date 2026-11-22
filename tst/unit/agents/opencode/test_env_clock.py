"""OpenCode on the env's clock: its bridge hands the shared egress rule (tst/unit/agents/test_agent_egress.py) the
model endpoint's URL and every MCP server's, which the CLI connects to itself, so on an armed clock it goes out through
the image's egress proxy, and to loopback, the hosts NO_PROXY names and plain-HTTP hosts directly; unarmed, it goes out
as before. Where the image follows the clock, the CLI itself runs on real time and its model relay puts the env's date,
read from a command as the model's own commands read it, where the CLI put its own. What every clocked image shares is
checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import time

import pytest

from tst.unit.agents.opencode.agent import AGENT_DIR, agent_egress, opencode_bridge

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
    asyncio.run(opencode_bridge.run_opencode("go", "ctx"))

    assert seen == {"rewrite": opencode_bridge.env_date_rewrite, "base_url": "http://relay"}
    assert "plugin" not in opencode_bridge._build_config("m", None, {}, None)


# OpenCode's system prompt as its SystemPrompt.environment writes it (opencode-ai 1.17.18, src/session/system.ts).
ENV_BLOCK = ("Here is some useful information about the environment you are running in:\n<env>\n"
             "  Working directory: /app\n  Workspace root folder: /app\n  Is directory a git repo: no\n"
             "  Platform: linux\n  Today's date: Sat Nov 21 2026\n</env>")
SYSTEM = (f"You are OpenCode, the best coding agent on the planet.\n{ENV_BLOCK}\n"
          "The task's notes say: Today's date: as written")
SEARCH = "Search the web.\n\nThe current year is 2026. You MUST use this year when searching for recent information."


def _call(system=SYSTEM, *, parts=False, tools=True, user=SYSTEM) -> dict:
    content = [{"type": "text", "text": "header"}, {"type": "text", "text": system}] if parts else system
    call = {"model": "m", "stream": True, "messages": [{"role": "system", "content": content},
                                                       {"role": "user", "content": user}]}
    if tools:
        call["tools"] = [{"type": "function", "function": {"name": "websearch", "description": SEARCH}},
                         {"type": "function", "function": {"name": "bash", "description": "The current year is 2026."}}]
    return call


@pytest.fixture
def env_clock(tmp_path, monkeypatch):
    """An armed clock, in UTC, whose time a stand-in ``date`` prints (``FAKE_DATE_OUT``; it fails without one),
    recording the environment it got."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "date").write_text('#!/bin/sh\nenv > "$FAKE_DATE_ENV"\n[ "$1" = "+%s %N" ] || exit 2\n'
                                  '[ -n "$FAKE_DATE_SLEEP" ] && sleep "$FAKE_DATE_SLEEP"\n'
                                  '[ -n "$FAKE_DATE_OUT" ] || { echo no clock >&2; exit 1; }\necho "$FAKE_DATE_OUT"\n')
    (bin_dir / "date").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DATE_ENV", str(tmp_path / "date.env"))
    monkeypatch.setenv("AGENTENV_REAL_TIME", "1")
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    monkeypatch.setattr(opencode_bridge, "clock_armed", lambda: True)
    yield tmp_path / "date.env"
    monkeypatch.undo()
    time.tzset()


def _rewrite(call: dict) -> dict:
    return json.loads(asyncio.run(opencode_bridge.env_date_rewrite(json.dumps(call).encode())))


@pytest.mark.parametrize("date_out, today, year", [
    ("1801742400 000000000", "Thu Feb 04 2027", 2027),
    ("7258161600 250000000", "Wed Jan 01 2200", 2200),
    ("-631108800 000000000", "Sun Jan 01 1950", 1950),
    # Half a second before 1950: date prints the second below and a positive fraction.
    ("-631152001 500000000", "Sat Dec 31 1949", 1949),
])
@pytest.mark.parametrize("parts", [False, True])
def test_on_an_armed_clock_the_model_is_told_the_env_date_a_command_reads(env_clock, monkeypatch, date_out, today,
                                                                         year, parts):
    monkeypatch.setenv("FAKE_DATE_OUT", date_out)
    call = _rewrite(_call(parts=parts))

    system = call["messages"][0]["content"][1]["text"] if parts else call["messages"][0]["content"]
    assert system == SYSTEM.replace("Today's date: Sat Nov 21 2026", f"Today's date: {today}")
    assert call["messages"][1]["content"] == SYSTEM
    assert call["tools"][0]["function"]["description"] == SEARCH.replace("2026", str(year))
    assert call["tools"][1]["function"]["description"] == "The current year is 2026."
    assert "AGENTENV_REAL_TIME" not in env_clock.read_text() and "TZ=UTC" in env_clock.read_text()


def test_the_date_is_written_in_the_zone_the_cli_writes_its_own_in(env_clock, monkeypatch):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801785600 000000000")  # 2027-02-05T00:00Z, still the 4th in New York
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()

    assert "Today's date: Thu Feb 04 2027" in _rewrite(_call())["messages"][0]["content"]


@pytest.mark.parametrize("armed", [False, True])
def test_a_call_with_nothing_to_rewrite_or_off_the_clock_goes_as_written_and_reads_no_date(env_clock, monkeypatch,
                                                                                          armed):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")
    monkeypatch.setattr(opencode_bridge, "clock_armed", lambda: armed)
    calls = [json.dumps(_call(), indent=1).encode()] if not armed else []
    calls += [json.dumps(_call("A title for this conversation", tools=False, user="hi")).encode(), b"not json"]

    for body in calls:
        assert asyncio.run(opencode_bridge.env_date_rewrite(body)) == body
    assert not env_clock.exists()


@pytest.mark.parametrize("environ", [{}, {"FAKE_DATE_OUT": "soon"}, {"FAKE_DATE_OUT": "1 0", "FAKE_DATE_SLEEP": "5"}])
def test_a_date_that_cant_be_read_refuses_the_call_rather_than_tell_the_model_real_time(env_clock, monkeypatch,
                                                                                         environ):
    monkeypatch.setattr(opencode_bridge, "_DATE_TIMEOUT_S", 0.5)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(opencode_bridge.Refusal, match="the env's date couldn't be read"):
        _rewrite(_call())


def test_an_env_block_without_a_date_line_this_knows_refuses_the_call(env_clock, monkeypatch):
    monkeypatch.setenv("FAKE_DATE_OUT", "1801742400 000000000")

    with pytest.raises(opencode_bridge.Refusal, match="no date line"):
        _rewrite(_call(SYSTEM.replace("  Today's date:", "  Date:")))


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


def test_start_agent_keeps_a_providers_ca_bundle_trusted_by_the_server_and_the_proxy(tmp_path):
    """Sail points SSL_CERT_FILE and the like at the bundle it signs model requests under; root.conf points the server
    at agent-env's bundles, so start-agent adds the provider's to them and to the proxy's roots, once each."""
    start = (AGENT_DIR / "start-agent").read_text()
    merge = start[start.index('for bundle in "${SSL_CERT_FILE'):start.index("exec supervisord")]
    targets = {"/etc/agentenv/ca-bundle.pem": tmp_path / "ca-bundle.pem", "/etc/agentenv/ca.pem": tmp_path / "ca.pem",
               "/usr/local/lib/agentenv-egress/roots.pem": tmp_path / "roots.pem"}
    for real, fake in targets.items():
        fake.write_text("SYSTEM ROOTS\n")
        merge = merge.replace(real, str(fake))
    sail = tmp_path / "sailbox-ca-bundle.crt"
    sail.write_text("-----BEGIN CERTIFICATE-----\nSAIL\n-----END CERTIFICATE-----\n")

    subprocess.run(["sh", "-c", merge], check=True, env={"PATH": "/usr/bin:/bin", "SSL_CERT_FILE": str(sail),
                   "NODE_EXTRA_CA_CERTS": str(sail), "REQUESTS_CA_BUNDLE": "/etc/agentenv/ca-bundle.pem"})

    for fake in targets.values():
        text = fake.read_text()
        assert text.startswith("SYSTEM ROOTS\n") and text.count("SAIL") == 1, (fake.name, text)

