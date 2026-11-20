"""Codex on the env's clock. The CLI is a static binary agent-env's libfaketime doesn't reach, so it stays on real time,
checks its own certificates on it and goes out directly, its environment as it was. On an armed clock each CLI run
keeps mcp-remote on real time too, started by Node itself with the real-time marker, gives the model's commands, which
follow the env's clock, the egress proxy's variables from the shared rule (tst/unit/agents/test_agent_egress.py), and
tells the model the env's date, which the CLI's own context gives as the host's. Unarmed, or where the image doesn't
start the server as the user libfaketime fakes, a run is what it was before the image followed a clock. What every
clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from tst.unit.agents.codex.agent import AGENT_DIR, agent_egress, codex_bridge, codex_config

PROXY = "http://127.0.0.1:3128"
PROXY_NAMES = {name for base in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
               for name in (base, base.lower())} | {"NODE_USE_ENV_PROXY"}
MODEL = "https://litellm.example.com"
SERVERS = {"env": {"url": "https://abc-123.w.modal.host/mcp",
                   "headers": {"Authorization": "Bearer t", "AgentEnv-Role": "user"}},
           "local": {"url": "http://gateway:8080/mcp", "headers": None}}
URLS = [info["url"] for info in SERVERS.values()]


def _overrides(cmd: list[str]) -> list[str]:
    return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-c"]


def _config(overrides: list[str]) -> dict:
    """The overrides as the CLI lays them over config.toml: each a TOML key and value."""
    return tomllib.loads("\n".join(overrides))


@pytest.fixture
def image_path(tmp_path, monkeypatch) -> Path:
    """Node and mcp-remote on PATH as the image has them: mcp-remote a link, through npm's .bin, to its script, which
    starts `#!/usr/bin/env node`. The script, resolved."""
    root = tmp_path / "image"
    dist, npm_bin, path = root / "node_modules/mcp-remote/dist", root / "node_modules/.bin", root / "bin"
    for folder in (dist, npm_bin, path):
        folder.mkdir(parents=True)
    script, node = dist / "proxy.js", path / "node"
    script.write_text("#!/usr/bin/env node\n")
    node.write_bytes(b"\x7fELF")
    for program in (script, node):
        program.chmod(0o755)
    (npm_bin / "mcp-remote").symlink_to("../mcp-remote/dist/proxy.js")
    (path / "mcp-remote").symlink_to(npm_bin / "mcp-remote")
    monkeypatch.setenv("PATH", str(path))
    return script.resolve()


def _starts_on_real_time(server: dict, script: Path, info: dict) -> None:
    """An MCP server the CLI starts as the program that reads the marker, Node, not through a `#!` line, whose
    interpreter would take the marker out of what it hands Node; with mcp-remote's script and config.toml's
    arguments."""
    first = Path(shutil.which(server["command"])).read_bytes()
    assert not first.startswith(b"#!"), server["command"]
    assert server["args"] == [str(script), *codex_config._server_entry(info["url"], info["headers"])["args"]]
    assert server["env"] == {codex_bridge.REAL_TIME_ENV: "1"}


def test_on_an_armed_clock_the_models_commands_go_out_through_the_proxy(image_path):
    environ = {agent_egress.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": MODEL, "NO_PROXY": "registry.internal"}

    shell = _config(codex_bridge.clock_overrides(environ, SERVERS, armed=True))["shell_environment_policy"]["set"]

    assert shell == agent_egress.egress_env(environ, [MODEL, *URLS], armed=True)
    assert {shell[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    direct = set(shell["NO_PROXY"].split(","))
    assert {"localhost", "127.0.0.1", "host.docker.internal", "registry.internal", "gateway"} <= direct
    assert not {"abc-123.w.modal.host", "litellm.example.com"} & direct


def test_on_an_armed_clock_each_mcp_server_is_started_by_node_itself_with_the_real_time_marker(image_path):
    environ = {agent_egress.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": MODEL}

    servers = _config(codex_bridge.clock_overrides(environ, SERVERS, armed=True))["mcp_servers"]

    assert servers.keys() == SERVERS.keys()
    for name, info in SERVERS.items():
        _starts_on_real_time(servers[name], image_path, info)


def test_unarmed_a_cli_run_takes_no_override():
    assert codex_bridge.clock_overrides({agent_egress.EGRESS_PROXY_ENV: PROXY}, SERVERS, armed=False) == []


class _Stream:
    def __init__(self, data: bytes = b""):
        self.data = data

    async def read(self, n: int = -1) -> bytes:
        chunk, self.data = (self.data, b"") if n < 0 else (self.data[:n], self.data[n:])
        return chunk


class _Stdin:
    def __init__(self):
        self.data = b""

    def write(self, data: bytes):
        self.data += data

    async def drain(self):
        return None

    def close(self):
        return None


class _CLI:
    """A CLI run that answers and completes its turn."""

    returncode = 0

    def __init__(self):
        events = [{"type": "turn.started"}, {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
                  {"type": "turn.completed"}]
        self.stdin, self.stderr = _Stdin(), _Stream()
        self.stdout = _Stream(b"".join(json.dumps(event).encode() + b"\n" for event in events))

    async def wait(self):
        return 0


class _Date:
    async def communicate(self):
        return b"2031-03-04 (Tuesday)\n", None


@pytest.fixture
def spawned(monkeypatch):
    """Every process the bridge starts: `date`, answered with the env's date, and the CLI."""
    runs: list[dict] = []

    async def fake_exec(*cmd, **kwargs):
        proc = _Date() if cmd[0] == "date" else _CLI()
        runs.append({"cmd": list(cmd), "kwargs": kwargs, "proc": proc})
        return proc

    monkeypatch.setattr(codex_bridge.asyncio, "create_subprocess_exec", fake_exec)
    for name in PROXY_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in {agent_egress.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": MODEL, "LITELLM_API_KEY": "k",
                        "TZ": "America/Chicago"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(codex_bridge.MODEL_HEADERS_ENV, raising=False)
    return runs


def _turn() -> None:
    asyncio.run(codex_bridge.run_codex(prompt="hi", thread_id=None, model="gpt-x", project_id="p-1", task_id="t-1",
                                       mcp_servers=SERVERS, config_overrides=["features.goals=false"]))


def _main_cmd() -> list[str]:
    """The CLI's command line as it was before the image followed a clock."""
    return ["codex", "exec", "--cd", codex_bridge.WORKSPACE_DIR, "-", "--json",
            "--dangerously-bypass-approvals-and-sandbox", "-c", 'model="gpt-x"', "-c", "features.goals=false"]


def _main_env() -> dict[str, str]:
    """The CLI's environment as it was before the image followed a clock."""
    return {**os.environ, "OPENAI_API_KEY": "k", "LITELLM_CUSTOMER_ID": "p-1",
            "LITELLM_TAGS": "projectId:p-1,taskId:t-1"}


def test_unarmed_the_cli_runs_as_it_did_before_the_image_followed_a_clock(spawned, monkeypatch):
    monkeypatch.setattr(codex_bridge, "clock_armed", lambda: False)

    _turn()

    [cli] = spawned
    assert cli["cmd"] == _main_cmd()
    assert cli["kwargs"]["env"] == _main_env()
    assert cli["proc"].stdin.data == b"hi"


def test_an_armed_page_beside_a_server_the_image_doesnt_start_leaves_the_cli_run_as_before(spawned, monkeypatch):
    """An agent installed with install/v1 runs as root in a computer's container, whose commands libfaketime doesn't
    fake: the computer's armed page tells nothing of their time."""
    monkeypatch.setattr(codex_bridge, "clock_armed", lambda: True)
    monkeypatch.delenv(agent_egress.EGRESS_PROXY_ENV)

    _turn()

    [cli] = spawned
    assert cli["cmd"] == _main_cmd()
    assert cli["proc"].stdin.data == b"hi"


def test_on_an_armed_clock_the_cli_keeps_its_environment_its_children_get_the_clocks_routes_and_the_model_its_date(
        spawned, image_path, monkeypatch):
    monkeypatch.setattr(codex_bridge, "clock_armed", lambda: True)

    _turn()

    date, cli = spawned
    assert date["cmd"] == ["date", "+%Y-%m-%d (%A)"]
    overrides = _overrides(cli["cmd"])
    assert overrides[:2] == ['model="gpt-x"', "features.goals=false"]
    config = _config(overrides[2:])
    for name, info in SERVERS.items():
        _starts_on_real_time(config["mcp_servers"][name], image_path, info)
    assert config["shell_environment_policy"]["set"]["HTTPS_PROXY"] == PROXY
    assert cli["kwargs"]["env"] == _main_env() and not PROXY_NAMES & set(cli["kwargs"]["env"])
    prompt = cli["proc"].stdin.data.decode()
    assert prompt.startswith("<environment_clock>\nToday is 2031-03-04 (Tuesday) in America/Chicago on this task's "
                             "clock, which `date` and every command you run read.")
    assert "current_date in <environment_context>" in prompt and prompt.endswith("</environment_clock>\n\nhi")


def test_a_task_on_an_armed_clock_reaches_the_cli_with_its_mcp_servers_on_real_time(codex, spawned, image_path,
                                                                                    monkeypatch):
    monkeypatch.setattr(codex_bridge, "clock_armed", lambda: True)

    async def exercise():
        async with codex.app.router.lifespan_context(codex.app), codex.client() as client:
            for name, info in SERVERS.items():
                await client.post("/ext/mcp-config", json={"name": name, **{k: v for k, v in info.items() if v}})
            await codex.send(client, "hi")

    asyncio.run(exercise())

    config = _config(_overrides(spawned[-1]["cmd"]))
    for name, info in SERVERS.items():
        _starts_on_real_time(config["mcp_servers"][name], image_path, info)
    direct = config["shell_environment_policy"]["set"]["NO_PROXY"].split(",")
    assert "gateway" in direct and "abc-123.w.modal.host" not in direct


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
