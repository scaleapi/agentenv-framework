"""Letta Code on the env's clock: its bridge hands the shared egress rule (tst/unit/agents/test_agent_egress.py) every
MCP server's URL, which the CLI connects to itself, so on an armed clock every ``letta`` command of a turn goes out
through the image's egress proxy, and to loopback, the model relay's among it, the hosts NO_PROXY names and plain-HTTP
hosts directly; unarmed, a turn runs the commands main ran, with the environment main gave them. The workspace
changelog's tar runs on real time whatever the clock. What every clocked image shares is checked in
tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import asyncio
import json
import subprocess
from contextlib import asynccontextmanager

import pytest

from tst.unit.agents.letta_code.agent import a2a_server, agent_egress
from tst.unit.agents.letta_code.agent import letta_code_bridge as bridge

PROXY = "http://127.0.0.1:3128"
RELAY = "http://127.0.0.1:41000/v1"
PROXY_VARS = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
              "NODE_USE_ENV_PROXY"}


@pytest.fixture(autouse=True)
def _sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path))


def _env(monkeypatch, environ: dict[str, str], *, urls=(), armed: bool) -> dict[str, str]:
    for name in (agent_egress.EGRESS_PROXY_ENV, *PROXY_VARS):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    return bridge._context_env("ctx", urls, armed=armed)


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy(monkeypatch):
    env = _env(monkeypatch, {agent_egress.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"}, armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    # The model relay the CLI is pointed at is on loopback.
    assert {"localhost", "127.0.0.1", "host.docker.internal", "registry.internal"} <= set(env["NO_PROXY"].split(","))
    assert env["NODE_USE_ENV_PROXY"] == "1"


@pytest.mark.parametrize("environ, armed", [({}, True), ({agent_egress.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(monkeypatch, environ, armed):
    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(_env(monkeypatch, environ, armed=armed))


def test_the_envs_plain_http_mcp_servers_are_reached_directly_and_https_ones_through_the_proxy_on_an_armed_clock(
        monkeypatch):
    """Node checks an HTTPS server's certificate against the env's date, which the proxy, on real time, doesn't."""
    urls = ["http://10.0.0.5:8080/mcp", "https://abc-123.w.modal.host/mcp"]
    direct = set(_env(monkeypatch, {agent_egress.EGRESS_PROXY_ENV: PROXY}, urls=urls, armed=True)["NO_PROXY"].split(","))

    assert "10.0.0.5" in direct and "abc-123.w.modal.host" not in direct


class _Proc:
    """A ``letta`` command that answers as the CLI does: an agent's id to ``agents create``, an empty tool catalog to
    ``mcp tools``, a successful result to the turn's own run."""

    pid = None
    returncode = 0

    def __init__(self, cmd: tuple[str, ...]):
        if "create" in cmd:
            self.out = b'{"id": "agent-1"}'
        elif "tools" in cmd:
            self.out = b"[]"
        elif "-p" in cmd:
            self.out = json.dumps({"type": "result", "subtype": "success", "result": "ok",
                                   "conversation_id": "local-conv-1"}).encode() + b"\n"
        else:
            self.out = b""
        self.stdout, self.stderr = asyncio.StreamReader(), asyncio.StreamReader()
        self.stdout.feed_data(self.out)
        for stream in (self.stdout, self.stderr):
            stream.feed_eof()

    async def communicate(self):
        return self.out, b""

    async def wait(self):
        return 0


def _turn(monkeypatch, tmp_path, *, armed: bool, environ: dict[str, str]) -> list[tuple[tuple[str, ...], dict]]:
    """Every command one turn of the bridge starts, with the environment each starts with, for an env MCP server over
    TLS, one over plain HTTP and the peer-agents one."""
    spawned = []

    async def spawn(*cmd, env, **_):
        spawned.append((cmd, dict(env)))
        return _Proc(cmd)

    @asynccontextmanager
    async def relay(*_args, **_kwargs):
        yield RELAY.removesuffix("/v1")

    monkeypatch.setattr(bridge.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(bridge, "model_relay", relay)
    monkeypatch.setattr(bridge, "_configure_agent", lambda *args: None)
    monkeypatch.setattr(bridge, "clock_armed", lambda: armed)
    monkeypatch.setattr(bridge.os, "environ", dict(environ))
    servers = {"env": {"url": "https://abc-123.w.modal.host/mcp"}, "local": {"url": "http://gateway:8080/mcp"},
               "peers": {"url": "http://127.0.0.1:8000/peers/mcp"}}
    result = asyncio.run(bridge.run_letta_code("hi", "ctx", model="m", mcp_servers=servers, workspace=str(tmp_path)))
    assert result.error is None and result.session_id == "local-conv-1"
    return spawned


IMAGE_ENV = {agent_egress.EGRESS_PROXY_ENV: PROXY, "PATH": "/usr/local/bin:/usr/bin:/bin",
             "NODE_EXTRA_CA_CERTS": "/etc/agentenv/ca.pem", "LITELLM_BASE_URL": "https://litellm.example.com"}


def test_on_an_armed_clock_every_command_of_a_turn_goes_out_through_the_proxy_but_to_loopback_and_plain_http(
        monkeypatch, tmp_path):
    spawned = _turn(monkeypatch, tmp_path, armed=True, environ={**IMAGE_ENV, "HOME": str(tmp_path / "data")})

    assert [cmd[3] for cmd, _ in spawned] == ["connect", "agents", "mcp", "-p"]
    for _, env in spawned:
        hosts = env["NO_PROXY"].split(",")
        assert env["HTTPS_PROXY"] == env["https_proxy"] == PROXY and env["NODE_USE_ENV_PROXY"] == "1"
        assert "abc-123.w.modal.host" not in hosts and {"gateway", "127.0.0.1"} <= set(hosts)


def test_unarmed_a_turn_runs_mains_commands_with_mains_environment(monkeypatch, tmp_path):
    """With the clock off, the bridge adds nothing to what the CLI ran with before the clock but the turn's marker: the
    image's own variables (the proxy's address, agent-env's CA bundles) pass through untouched, as every other variable
    of the server's."""
    environ = {**IMAGE_ENV, "HOME": str(tmp_path / "data")}
    spawned = _turn(monkeypatch, tmp_path, armed=False, environ=environ)
    home = bridge.context_data_dir("ctx")
    base = {**environ, "HOME": str(home), "LETTA_LOCAL_BACKEND_DIR": str(home / ".letta/lc-local-backend"),
            "LETTA_LOCAL_BACKEND_EXPERIMENTAL": "1", bridge._TURN_MARKER: spawned[0][1][bridge._TURN_MARKER]}
    handle = "openai-compatible/m"

    assert spawned == [
        (("letta", "--backend", "local", "connect", "openai-compatible", "--base-url", RELAY, "--api-key", "relay"),
         base),
        (("letta", "--backend", "local", "agents", "create", "--model", handle), base),
        (("letta", "--backend", "local", "mcp", "tools", "--full", "--agent", "agent-1"), base),
        (("letta", "--backend", "local", "-p", "hi", "--output-format", "stream-json", "--yolo", "--model", handle,
          "--toolset", "default", "--agent", "agent-1", "--new"), {**base, "LETTA_AGENT_ID": "agent-1"}),
    ]


def test_the_workspace_changelogs_tar_runs_on_real_time(monkeypatch, tmp_path):
    """tar's dump time goes into the snapshot file, and the next increment takes only files changed after it: a tar on
    an env clock ahead of real time would miss every change, as file times stay real."""
    runs = []

    def run(cmd, **kwargs):
        runs.append((cmd, kwargs.get("env")))
        return subprocess.CompletedProcess(cmd, 0)

    class Uploader:
        async def upload(self, name, path):
            pass

    changelog = tmp_path / "changelog"
    monkeypatch.setattr(a2a_server, "_CHANGELOG_DIR", changelog)
    monkeypatch.setattr(a2a_server, "_CHANGELOG_SNAR", changelog / "state.snar")
    monkeypatch.setattr(a2a_server, "_CHANGELOG_META_DIR", tmp_path / "meta")
    monkeypatch.setattr(a2a_server.subprocess, "run", run)
    workspace = tmp_path / "app"
    workspace.mkdir()
    a2a_server._baseline_changelog([str(workspace)])
    (changelog / "state.snar").write_bytes(b"")
    agent = a2a_server.LettaCodeChangelogAgent()
    agent._changelog = {"roots": [str(workspace)], "sequence": 0, "present": [str(workspace)], "uploader": Uploader()}
    asyncio.run(agent._capture_changelog())

    assert [cmd[1].split("=")[0] for cmd, _ in runs] == ["--listed-incremental"] * 2
    assert all(env == {**a2a_server.os.environ, "AGENTENV_REAL_TIME": "1"} for _, env in runs), runs
