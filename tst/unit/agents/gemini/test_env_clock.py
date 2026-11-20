"""Gemini on the env's clock. On an armed clock the CLI itself stays on real time and goes out directly: started as its
script run by node, with no relaunch of its own, so the real-time marker reaches the process that runs it, it compares
its locks', sessions', tool outputs' and logs' real file times, its credentials' expiry and its MCP servers'
certificates with real time. The commands the model runs follow the clock and go out by the shared egress rule
(tst/unit/agents/test_agent_egress.py), from a file every bash the CLI starts reads first, which the CLI never reads
itself, as it would send every request through a proxy it's handed, NO_PROXY or not. The model's date line names the
env's date (test_litellm_proxy.py), and the changelog's capture and its tar run on real time. Unarmed, the CLI runs
exactly as before. What every clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tst.unit.agents.gemini.agent import AGENT_DIR, a2a_server, agent_egress, gemini_bridge, gemini_config
from tst.unit.agents.gemini.fake_cli import FakeCLI
from tst.unit.agents.gemini.support import namespace_grant, send, settings as user_settings

PROXY = "http://127.0.0.1:3128"
ATTRIBUTION_PROXY = "http://127.0.0.1:8000/litellm-gemini/token"
PROXY_VARS = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY",
              "no_proxy", "NODE_USE_ENV_PROXY"}
MCP_URLS = ["http://gateway:8080/mcp", "https://abc-123.w.modal.host/mcp"]
ENV_TODAY = "Tuesday, March 3, 2031"
CHANGELOG_PATH = a2a_server.SNAPSHOT_V1.endpoint + "/changelog"


def _page(armed: int) -> bytes:
    return agent_egress.clockpage.encode(2, 1_900_000_000 * 10**9, 10**18, 1.0, armed, 3, 0, 0)


@pytest.fixture
def run_settings(tmp_path) -> Path:
    path = tmp_path / "agentenv-runs" / "run.json"
    path.parent.mkdir()
    path.write_text("{}")
    return path


def _env(environ, run_settings, **kwargs) -> dict[str, str]:
    return gemini_bridge.cli_env(environ, system_settings=run_settings, base_url=ATTRIBUTION_PROXY, **kwargs)


def _commands_env(env: dict[str, str]) -> dict[str, str]:
    """The environment of a command the model runs: `bash -c`, as the CLI starts each, in the CLI's environment."""
    script = f"{shlex.quote(sys.executable)} -c 'import json, os; print(json.dumps(dict(os.environ)))'"
    return json.loads(subprocess.run(["bash", "-c", script], env=env, capture_output=True, check=True).stdout)


def test_on_an_armed_clock_the_cli_stays_on_real_time_with_no_relaunch_and_no_proxy_of_its_own(run_settings):
    env = _env({agent_egress.EGRESS_PROXY_ENV: PROXY}, run_settings, clock_armed=True, mcp_urls=MCP_URLS)

    assert (env["AGENTENV_REAL_TIME"], env["GEMINI_CLI_NO_RELAUNCH"]) == ("1", "true")
    assert not PROXY_VARS & set(env)


def test_on_an_armed_clock_the_models_commands_go_out_by_the_shared_egress_rule(run_settings):
    """An env MCP server over TLS goes through the proxy, since the model's curl to it checks its certificate against
    the env's date; plain-HTTP and loopback hosts and the deploy's NO_PROXY go direct."""
    environ = {agent_egress.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal",
               "LITELLM_BASE_URL": "https://litellm.example.com"}
    env = _env(environ, run_settings, clock_armed=True, mcp_urls=[*MCP_URLS, ""])

    seen = _commands_env(env)
    rule = agent_egress.egress_env(environ, [ATTRIBUTION_PROXY, environ["LITELLM_BASE_URL"], *MCP_URLS], armed=True)
    assert {name: seen[name] for name in rule} == rule
    direct = seen["NO_PROXY"].split(",")
    assert {"127.0.0.1", "gateway", "registry.internal"} <= set(direct) and "abc-123.w.modal.host" not in direct
    assert env["BASH_ENV"] == str(gemini_config.commands_env_path(run_settings))


@pytest.mark.parametrize("name", ["LITELLM_BASE_URL", "GOOGLE_GEMINI_BASE_URL"])
@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_the_models_commands_reach_a_model_endpoint_over_plain_http_directly(run_settings, name, base_url, direct):
    env = _env({agent_egress.EGRESS_PROXY_ENV: PROXY, name: base_url}, run_settings, clock_armed=True)

    assert (urlsplit(base_url).hostname in _commands_env(env)["NO_PROXY"].split(",")) is direct


@pytest.mark.parametrize("environ, added", [
    ({}, {"GOOGLE_GEMINI_BASE_URL": ATTRIBUTION_PROXY}),
    ({agent_egress.EGRESS_PROXY_ENV: PROXY, "LITELLM_API_KEY": "k", "LITELLM_BASE_URL": "https://gw",
      "NO_PROXY": "registry.internal", "HTTPS_PROXY": "http://corporate:8080"},
     {"GEMINI_API_KEY": "k", "GOOGLE_GEMINI_BASE_URL": ATTRIBUTION_PROXY}),
    ({"GEMINI_API_KEY": "own", "GOOGLE_GEMINI_BASE_URL": "https://own", "LITELLM_API_KEY": "k"}, {}),
], ids=["bare", "image", "own-endpoint"])
def test_unarmed_the_cli_runs_exactly_as_before(run_settings, environ, added):
    """The environment the CLI had before it could follow the clock, this process's with its key, its endpoint and its
    settings and nothing else; started as `gemini`, with no file written for its commands."""
    env = _env(environ, run_settings, clock_armed=False, mcp_urls=MCP_URLS)

    assert env == {**environ, **added, gemini_config.SYSTEM_SETTINGS_ENV: str(run_settings)}
    assert gemini_bridge.cli_program(env) == ["gemini"]
    assert not gemini_config.commands_env_path(run_settings).exists()


def test_the_envs_date_is_worded_as_the_clis_date_line_words_it():
    """The CLI's line is Node's toLocaleDateString in en-US, which the image's locale (C.UTF-8) gives: the weekday,
    the month, the day and the year. Here, with no clock to follow, `date` reads real time."""
    before = datetime.now()
    today = gemini_bridge.env_today()

    assert today in {f"{day:%A}, {day:%B} {day.day}, {day.year}" for day in (before, datetime.now())}


def test_the_commands_environment_goes_with_the_runs_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_CLI_HOME", str(tmp_path))
    with gemini_config.run_settings({}, None, None) as path:
        commands = Path(_env({agent_egress.EGRESS_PROXY_ENV: PROXY}, path, clock_armed=True)["BASH_ENV"])
        assert commands.is_file() and commands.parent == path.parent

    assert list((tmp_path / ".gemini" / "agentenv-runs").iterdir()) == []


def _installed_like_the_image(root: Path) -> tuple[Path, Path]:
    """node, and `gemini` on PATH as a link to the CLI's script, through node_modules/.bin, as the image has them."""
    script = root / "node_modules" / "@google" / "gemini-cli" / "bundle" / "gemini.js"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env node\n")
    (root / "node_modules" / ".bin").mkdir()
    (root / "node_modules" / ".bin" / "gemini").symlink_to("../@google/gemini-cli/bundle/gemini.js")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (bin_dir / "gemini").symlink_to(root / "node_modules" / ".bin" / "gemini")
    node = bin_dir / "node"
    node.write_text("#!/bin/sh\n")
    for path in (script, node):
        path.chmod(0o755)
    return node, script


@pytest.mark.parametrize("armed", [False, True])
def test_a_cli_on_real_time_is_started_as_its_script_run_by_node_so_the_marker_reaches_node(
        monkeypatch, tmp_path, run_settings, armed):
    """`gemini` starts with `#!/usr/bin/env node`: env, started with the marker, would stay on real time itself and
    start node without it."""
    node, script = _installed_like_the_image(tmp_path)
    env = _env({"PATH": str(node.parent), agent_egress.EGRESS_PROXY_ENV: PROXY}, run_settings, clock_armed=armed)
    started = []

    async def fake_exec(*cmd, **kwargs):
        started.append((list(cmd), kwargs["env"]))
        return FakeCLI(b'{"type":"result","status":"success","stats":{}}\n')

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(gemini_bridge.run_gemini(prompt="hi", session_id=None, env=env))

    ((cmd, spawned),) = started
    program = [str(node), os.path.realpath(script)] if armed else ["gemini"]
    assert cmd == [*program, "-p", "hi", "--output-format", "stream-json", "--approval-mode", "yolo"]
    assert spawned.get("AGENTENV_REAL_TIME") == ("1" if armed else None)


@pytest.mark.parametrize("armed", [False, True])
def test_a_turn_on_an_armed_clock_starts_its_cli_on_real_time_and_names_the_envs_date(
        monkeypatch, tmp_path, home, agent, client, armed):
    """The call site: the turn reads the clock page, and on an armed one hands the CLI the marker, its commands the
    proxy for the env's MCP server over TLS, and its model calls the env's date for the whole turn."""
    page = tmp_path / "agentenv-clock"
    page.write_bytes(_page(int(armed)))
    monkeypatch.setattr(agent_egress.clockpage, "PAGE", str(page))
    monkeypatch.setattr(a2a_server, "env_today", lambda: ENV_TODAY)
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(agent_egress.EGRESS_PROXY_ENV, PROXY)
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm.example.com")
    for name, url in zip(("modal", "gateway"), MCP_URLS):
        client.post(a2a_server.MCP_CONFIG_V1.endpoint, json={"url": url, "name": name})
    seen = {}

    async def fake_run_gemini(**kwargs):
        env = kwargs["env"]
        seen.update(env=env, route=agent._routes[env["GOOGLE_GEMINI_BASE_URL"].rsplit("/", 1)[1]],
                    commands=_commands_env(env))
        return gemini_bridge.GeminiResult(response="ok", tool_call_count=0, session_id=None)

    monkeypatch.setattr(a2a_server, "run_gemini", fake_run_gemini)
    send(client, "go")

    env, commands = seen["env"], seen["commands"]
    assert not PROXY_VARS & set(env)
    if not armed:
        assert "AGENTENV_REAL_TIME" not in env and "BASH_ENV" not in env and seen["route"].today is None
        assert not PROXY_VARS & set(commands)
        return
    assert (env["AGENTENV_REAL_TIME"], seen["route"].today) == ("1", ENV_TODAY)
    direct = commands["NO_PROXY"].split(",")
    assert commands["HTTPS_PROXY"] == PROXY and "gateway" in direct and "abc-123.w.modal.host" not in direct
    assert not Path(env["BASH_ENV"]).exists()


def test_the_changelogs_capture_runs_on_real_time_as_its_grants_expiry_is(home, changelog_dir, client):
    """The CLI starts it in bash as a child, which follows the env's clock: on one ahead of real time, the grant would
    look expired to it, and it would refuse every upload before sending it."""
    client.post(CHANGELOG_PATH, json={"write_namespace": namespace_grant(), "roots": [str(home)]})

    ((command,),) = [[h["command"] for h in e["hooks"]] for e in user_settings(home)["hooks"]["AfterTool"]]
    assert shlex.split(command) == ["AGENTENV_REAL_TIME=1", sys.executable, str(changelog_dir / "capture.py")]
    assert a2a_server.is_missing_capture_command(command) is False


def _recording_tar(directory: Path) -> Path:
    """A tar on PATH that records the marker it was started with, then fails, so a capture uploads nothing."""
    directory.mkdir()
    log = directory / "markers"
    tar = directory / "tar"
    tar.write_text(f'#!/bin/sh\necho "${{AGENTENV_REAL_TIME:-}}" >> {shlex.quote(str(log))}\nexit 2\n')
    tar.chmod(0o755)
    return log


def test_the_changelogs_tar_runs_on_real_time_for_its_baseline_and_each_capture(monkeypatch, tmp_path, home,
                                                                                changelog_dir, client):
    """Its --listed-incremental snapshot keeps the time it ran: on an env clock ahead of real time that would be after
    the next tool call's writes, which the next capture would then miss. The capture's Python, started with the marker,
    hands it to no child, so it hands it to tar itself."""
    markers = _recording_tar(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{markers.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("AGENTENV_REAL_TIME", raising=False)
    client.post(CHANGELOG_PATH, json={"write_namespace": namespace_grant(), "roots": [str(home)]})

    subprocess.run([sys.executable, str(changelog_dir / "capture.py")], input='{"tool_name": "write_file"}',
                   text=True, capture_output=True, check=True, timeout=60)

    assert markers.read_text().split() == ["1", "1"]


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
