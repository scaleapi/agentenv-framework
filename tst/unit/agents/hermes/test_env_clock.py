"""Hermes on the env's clock. Unarmed, a turn runs exactly as before the clock: ``hermes agentenv turn`` with the same
flags and environment. On an armed clock Hermes stays on real time, run as the command its launcher runs with the
real-time marker, so its own records, leases and cleanups keep real times while every command it starts follows the
env's clock; the bridge hands the shared egress rule (tst/unit/agents/test_agent_egress.py) the model endpoint's URL and
every MCP server's, which Hermes connects to itself, and names its variables for execute_code's scripts too; and the
plugin shows the model the env's date. What every clocked image shares is checked in
tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import shlex
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tst.unit.agents.hermes.agent import AGENT_DIR, agent_egress, hermes_bridge
from tst.unit.agents.hermes.conftest import FAKE_HERMES_BOOTSTRAP

PROXY = "http://127.0.0.1:3128"
HOME = Path("/data/.hermes/profiles/ctx-0")
# The server's environment as root.conf gives it, and the endpoint agent-env gives the agent.
ENVIRON = {agent_egress.EGRESS_PROXY_ENV: PROXY, "SSL_CERT_FILE": "/etc/agentenv/ca-bundle.pem",
           "REQUESTS_CA_BUNDLE": "/etc/agentenv/ca-bundle.pem", "NODE_EXTRA_CA_CERTS": "/etc/agentenv/ca.pem",
           "LITELLM_BASE_URL": "https://litellm.example.com", "LITELLM_API_KEY": "k"}
MCP = {"env": {"url": "https://abc-123.w.modal.host/mcp"},
       "peer_agents": {"url": "http://127.0.0.1:8000/ext/peer-agents/mcp"}}


def test_on_an_armed_clock_the_turn_runs_on_real_time_and_goes_out_through_the_proxy():
    env = hermes_bridge.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"}, HOME,
                                events_fd=7, clock_armed=True)

    assert env["AGENTENV_REAL_TIME"] == "1"
    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert {"localhost", "host.docker.internal", "registry.internal"} <= set(env["NO_PROXY"].split(","))
    assert (env["HERMES_HOME"], env["AGENTENV_EVENTS_FD"]) == (str(HOME), "7")


@pytest.mark.parametrize("environ", [{}, {agent_egress.EGRESS_PROXY_ENV: PROXY}])
def test_unarmed_the_turns_environment_is_this_processs_with_its_hermes_home_and_event_pipe(environ):
    env = hermes_bridge.cli_env(environ, HOME, events_fd=7, mcp_servers=MCP, clock_armed=False)

    assert env == {**environ, "HERMES_HOME": str(HOME), "AGENTENV_EVENTS_FD": "7"}


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = hermes_bridge.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": base_url}, HOME,
                                clock_armed=True)
    assert (base_url.split("//")[1].split(":")[0] in env["NO_PROXY"].split(",")) is direct


def test_the_turns_plain_http_mcp_servers_are_reached_directly_and_https_ones_through_the_proxy_on_an_armed_clock():
    """The proxy checks an HTTPS server's certificate on real time, and reaches a private address in this image; a host
    reached over HTTPS too stays behind it, as NO_PROXY goes by host."""
    servers = {"env": {"url": "http://gateway.internal:8080/mcp"}, "docs": {"url": "https://10.1.2.3/mcp"},
               "remote": {"url": "https://abc-123.w.modal.host/mcp"}, "both": {"url": "http://abc-123.w.modal.host/sse"}}
    env = hermes_bridge.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY}, HOME, mcp_servers=servers, clock_armed=True)

    direct = set(env["NO_PROXY"].split(","))
    assert "gateway.internal" in direct and not {"10.1.2.3", "abc-123.w.modal.host"} & direct


def _spawned(monkeypatch, tmp_path, armed: bool, environ: dict = ENVIRON) -> dict:
    """What run_hermes starts the turn's process with, and the profile config it writes for it, for a resumed turn
    with an effort and the env's MCP servers, from a server whose environment is ``environ``."""
    spawned = {}

    async def fake_exec(*cmd, env, pass_fds, **_):
        spawned.update(cmd=list(cmd), env=env, fd=pass_fds[0])
        raise RuntimeError("not starting hermes")

    monkeypatch.setattr(hermes_bridge.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(hermes_bridge, "clock_armed", lambda: armed)
    monkeypatch.setattr(hermes_bridge.os, "environ", dict(environ))
    with pytest.raises(RuntimeError, match="not starting hermes"):
        asyncio.run(hermes_bridge.run_hermes("hi", "ctx-0", session_id="s-1", model="m", effort="high",
                                             mcp_servers=MCP, skills_dir=tmp_path / "skills",
                                             workspace=str(tmp_path)))
    cmd = spawned["cmd"]
    spawned["files"] = (cmd[cmd.index("--prompt-file") + 1], cmd[cmd.index("--usage-file") + 1])
    spawned["config"] = json.loads((hermes_bridge.profile_dir("ctx-0") / "config.yaml").read_text())
    return spawned


def test_unarmed_the_bridge_starts_hermes_with_the_command_and_environment_it_always_has(monkeypatch, fake_hermes,
                                                                                         tmp_path):
    spawned = _spawned(monkeypatch, tmp_path, armed=False)

    prompt, usage = spawned["files"]
    assert spawned["cmd"] == [hermes_bridge.HERMES_BIN, "agentenv", "turn", "--prompt-file", prompt, "--provider",
                              "litellm", "--model", "m", "--usage-file", usage, "--resume", "s-1", "--reasoning", "high"]
    assert spawned["env"] == {**ENVIRON, "HERMES_HOME": str(hermes_bridge.profile_dir("ctx-0")),
                              "AGENTENV_EVENTS_FD": str(spawned["fd"])}
    assert spawned["config"] == {
        "model": {"provider": "litellm", "default": "m"},
        "providers": {"litellm": {"api": "https://litellm.example.com/v1", "key_env": "LITELLM_API_KEY",
                                  "transport": "chat_completions"}},
        "skills": {"external_dirs": [str(tmp_path / "skills")]}, "plugins": {"enabled": ["agentenv"]},
        "auxiliary": {"background_review": {"enabled": False}},
        "mcp_servers": {name: {"url": info["url"]} for name, info in MCP.items()}}


def test_the_clock_adds_only_execute_codes_passthrough_to_the_profile_config(monkeypatch, fake_hermes, tmp_path):
    unarmed = _spawned(monkeypatch, tmp_path, armed=False)["config"]
    armed = _spawned(monkeypatch, tmp_path, armed=True)["config"]

    assert {key: value for key, value in armed.items() if key != "terminal"} == unarmed
    assert list(armed["terminal"]) == ["env_passthrough"]


def test_on_an_armed_clock_the_bridge_starts_hermes_own_python_on_real_time_behind_the_proxy(monkeypatch, fake_hermes,
                                                                                            tmp_path):
    """Not the launcher: its shell would take the marker, and the Python it starts would follow the env's clock."""
    spawned = _spawned(monkeypatch, tmp_path, armed=True)

    prompt, usage = spawned["files"]
    assert spawned["cmd"] == [sys.executable, "-I", "-c", FAKE_HERMES_BOOTSTRAP, "agentenv", "turn", "--prompt-file",
                              prompt, "--provider", "litellm", "--model", "m", "--usage-file", usage, "--resume", "s-1",
                              "--reasoning", "high", "--env-clock"]
    env = spawned["env"]
    assert env["AGENTENV_REAL_TIME"] == "1" and env["HTTPS_PROXY"] == env["https_proxy"] == PROXY
    assert "abc-123.w.modal.host" not in env["NO_PROXY"].split(",") and "127.0.0.1" in env["NO_PROXY"].split(",")


@pytest.mark.parametrize("zone", [None, "America/Chicago"])
def test_on_an_armed_clock_execute_codes_scripts_get_the_proxy_its_ca_and_the_zone_as_the_terminals_commands_do(
        monkeypatch, fake_hermes, tmp_path, zone):
    """execute_code passes its scripts a fixed few variables and those terminal.env_passthrough names; it sets TZ only
    from Hermes' own timezone setting, not from the TZ sync_env_clock's ``timezone`` sets in the server's
    environment."""
    spawned = _spawned(monkeypatch, tmp_path, armed=True, environ={**ENVIRON, **({"TZ": zone} if zone else {})})

    passed = set(spawned["config"]["terminal"]["env_passthrough"])
    egress = {name for name in spawned["env"] if name.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}}
    named = {"NODE_USE_ENV_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS", *(["TZ"] if zone else [])}
    assert egress | named == passed
    assert len(egress) == 8 and spawned["env"].get("TZ") == zone


@pytest.mark.asyncio
async def test_an_armed_turn_runs_to_its_end_as_the_launchers_command(monkeypatch, fake_hermes, folders):
    monkeypatch.setattr(hermes_bridge, "clock_armed", lambda: True)
    monkeypatch.setenv(agent_egress.EGRESS_PROXY_ENV, PROXY)

    result = await hermes_bridge.run_hermes(prompt="what is 6*7?", context_id="ctx-a", workspace=str(folders))

    ran = json.loads(fake_hermes.read_text())
    assert (result.error, result.response) == (None, "42")
    assert ran["args"][:2] == ["agentenv", "turn"] and ran["args"][-1] == "--env-clock"
    assert (ran["env"]["AGENTENV_REAL_TIME"], ran["env"]["HTTPS_PROXY"]) == ("1", PROXY)


def test_the_launchers_command_is_read_as_hermes_installer_writes_it(tmp_path):
    """``#!/bin/sh`` and ``exec`` with the Python, -I and a bootstrap of many lines, quoted by ``shlex.join``, then
    ``"$@"``; /usr/local/bin/hermes links to it."""
    python = str(tmp_path / "tools" / "python 3.14" / "bin" / "python3")
    script = "import os, sys\n# the repair's own stdlib: it's bound first\nsys.exit(main())\n"
    launcher = tmp_path / "hermes"
    launcher.write_text(f'#!/bin/sh\nexec {shlex.join([python, "-I", "-c", script])} "$@"\n')
    launcher.chmod(0o755)
    link = tmp_path / "bin" / "hermes"
    link.parent.mkdir()
    link.symlink_to(launcher)

    assert hermes_bridge.launcher_command(str(link)) == [python, "-I", "-c", script]


@pytest.mark.parametrize("body", [
    "#!/usr/bin/env python3\nfrom hermes_cli.main import main\nmain()\n",
    "#!/bin/sh\nexec python3 -m hermes_cli.main\n",
    # Hermes' forwarder to its own launcher, and an interpreter behind env: sh or env would take the marker.
    '#!/bin/sh\nexec /opt/hermes/.hermes/bin/hermes "$@"\n',
    '#!/bin/sh\nexec env FOO=1 /opt/hermes/python -I -c "import hermes" "$@"\n',
    '#!/bin/sh\nexec /opt/hermes/python -E -c "import hermes" "$@"\n',
    '#!/bin/sh\nexec /opt/hermes/python -I -c "import hermes" --flag "$@"\n',
])
def test_a_launcher_that_isnt_hermes_installers_is_refused(tmp_path, body):
    launcher = tmp_path / "hermes"
    launcher.write_text(body)
    launcher.chmod(0o755)

    with pytest.raises(ValueError, match="doesn't exec Hermes' interpreter"):
        hermes_bridge.launcher_command(str(launcher))


# The two Hermes functions --env-clock reaches, as the pinned release has them.
HERMES_TIME = """
from datetime import datetime

def get_timezone():
    return TZ

def now():
    tz = get_timezone()
    return datetime.now(tz) if tz is not None else datetime.now().astimezone()
"""
HERMES_STATE_IDS = """
import uuid
from datetime import datetime

def new_session_id(now=None, *, hex_len=6):
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:hex_len]}"
"""


def _module(name: str, source: str = "", **values) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(values)
    exec(source, module.__dict__)
    return module


@pytest.fixture
def plugin(monkeypatch):
    """The plugin as Hermes loads it, over stand-ins for the Hermes modules it reads and the one-shot runner it ends
    in, which records each turn's message."""
    ran: list[str] = []
    hermes_time = _module("hermes_time", HERMES_TIME, TZ=timezone.utc)
    ids = _module("hermes_state_ids", HERMES_STATE_IDS)
    for module in (hermes_time, ids, _module("hermes_cli"),
                   _module("hermes_cli.main", _run_and_exit_oneshot=lambda message, **_: ran.append(message))):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location("hermes_agentenv_plugin", AGENT_DIR / "agentenv_plugin" / "__init__.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return SimpleNamespace(module=loaded, hermes_time=hermes_time, ids=ids, ran=ran)


def _turn(plugin, tmp_path, *flags: str) -> None:
    """``hermes agentenv turn`` as the bridge runs it, through the command the plugin registers."""
    parser = argparse.ArgumentParser()
    plugin.module._setup(parser)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hi")
    plugin.module._handle(parser.parse_args(["turn", "--prompt-file", str(prompt), *flags]))


@pytest.mark.parametrize("env_time", ["2031-03-04T05:06:07+00:00", "3026-11-20T12:00:00+00:00",
                                      "1950-01-01T00:00:00+00:00"])
@pytest.mark.parametrize("loaded", [{}, {"agent.agent_init": "new_session_id",
                                         "agent.conversation_compression": "mint_session_id"}])
def test_with_env_clock_hermes_dates_its_prompt_and_session_by_the_envs_clock(plugin, monkeypatch, tmp_path, env_time,
                                                                             loaded):
    """The date line's date and the session ids' stamp, which its start date is read from, as a command started from
    Hermes reads the time: Hermes runs on real time, what it starts on the env's clock. Modules that bound
    new_session_id by import before the turn, as the agent and compaction do, each under its own name for it, mint by
    the env's clock too; the plugin imports neither."""
    epoch = int(datetime.fromisoformat(env_time).timestamp())
    monkeypatch.setattr(plugin.module, "ENV_TIME_COMMAND", (sys.executable, "-c", f"print({epoch})"))
    for module, name in loaded.items():
        monkeypatch.setitem(sys.modules, module, _module(module, **{name: plugin.ids.new_session_id}))

    _turn(plugin, tmp_path, "--env-clock")

    stamp = datetime.fromtimestamp(epoch).strftime("%Y%m%d_%H%M%S_")
    assert plugin.hermes_time.now() == datetime.fromisoformat(env_time)
    assert plugin.ids.new_session_id(datetime(2026, 11, 20)).startswith(stamp)
    for module, name in loaded.items():
        assert getattr(sys.modules[module], name)().startswith(stamp), module
    assert {"agent.agent_init", "agent.conversation_compression"} & set(sys.modules) == set(loaded)
    assert plugin.ran == ["hi"]


def test_without_env_clock_hermes_reads_its_own_clock(plugin, monkeypatch, tmp_path):
    monkeypatch.setattr(plugin.module, "ENV_TIME_COMMAND", (sys.executable, "-c", "raise SystemExit(1)"))
    minted = plugin.ids.new_session_id

    _turn(plugin, tmp_path)

    assert plugin.hermes_time.datetime is datetime and plugin.ids.new_session_id is minted
    assert abs(plugin.hermes_time.now().timestamp() - datetime.now().timestamp()) < 60
    assert plugin.ran == ["hi"]


def test_the_env_time_command_prints_whole_seconds_since_the_epoch(plugin):
    """Whole seconds: ``%N`` beside a negative ``%s`` reads a second off before 1970."""
    out = subprocess.run(plugin.module.ENV_TIME_COMMAND, capture_output=True, text=True, check=True).stdout

    assert abs(int(out) - datetime.now().timestamp()) < 60


def test_the_names_env_clock_replaces_were_checked_against_the_hermes_the_image_installs(plugin):
    pinned = re.search(r"^ARG HERMES_COMMIT=(\w+)$", (AGENT_DIR / "Dockerfile").read_text(), re.MULTILINE).group(1)

    assert plugin.module.ENV_CLOCK_HERMES_COMMIT == pinned


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
