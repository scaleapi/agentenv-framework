"""Codex on the env's clock. The CLI is a static binary agent-env's libfaketime doesn't reach, so it stays on real time,
checks its own certificates on it and goes out directly, its environment as it was. On an armed clock each CLI run
keeps mcp-remote on real time too, gives the model's commands, which follow the env's clock, the egress proxy's
variables from the shared rule (tst/unit/agents/test_agent_egress.py), and tells the model the env's date, which the
CLI's own context gives as the host's. Unarmed, a run is what it was before the image followed a clock. What every
clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tomllib

import pytest

from tst.unit.agents.codex.agent import AGENT_DIR, agent_egress, codex_bridge

PROXY = "http://127.0.0.1:3128"
PROXY_NAMES = {name for base in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
               for name in (base, base.lower())} | {"NODE_USE_ENV_PROXY"}
MODEL = "https://litellm.example.com"
SERVERS = {"env": "https://abc-123.w.modal.host/mcp", "local": "http://gateway:8080/mcp"}
REAL_TIME = {f'mcp_servers.{name}.env.AGENTENV_REAL_TIME="1"' for name in SERVERS}
SHELL = "shell_environment_policy.set."


def _overrides(cmd: list[str]) -> list[str]:
    return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-c"]


def _shell(overrides: list[str]) -> dict[str, str]:
    """The variables the shell_environment_policy.set overrides give the model's commands, each value read as the CLI
    reads an override's: as TOML."""
    return {key[len(SHELL):]: tomllib.loads(f"v = {value}")["v"]
            for key, _, value in (o.partition("=") for o in overrides) if key.startswith(SHELL)}


def test_on_an_armed_clock_the_models_commands_go_out_through_the_proxy_and_mcp_remote_stays_on_real_time():
    environ = {agent_egress.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": MODEL, "NO_PROXY": "registry.internal"}

    overrides = codex_bridge.clock_overrides(environ, SERVERS, armed=True)

    shell = _shell(overrides)
    assert shell == agent_egress.egress_env(environ, [MODEL, *SERVERS.values()], armed=True)
    assert {shell[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    direct = set(shell["NO_PROXY"].split(","))
    assert {"localhost", "127.0.0.1", "host.docker.internal", "registry.internal", "gateway"} <= direct
    assert not {"abc-123.w.modal.host", "litellm.example.com"} & direct
    assert {o for o in overrides if not o.startswith(SHELL)} == REAL_TIME


def test_unarmed_a_cli_run_takes_no_override():
    assert codex_bridge.clock_overrides({agent_egress.EGRESS_PROXY_ENV: PROXY}, SERVERS, armed=False) == []


def test_without_the_egress_proxy_an_armed_run_keeps_mcp_remote_on_real_time_and_sets_no_proxy():
    assert set(codex_bridge.clock_overrides({}, SERVERS, armed=True)) == REAL_TIME


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


def _main_env() -> dict[str, str]:
    """The CLI's environment as it was before the image followed a clock."""
    return {**os.environ, "OPENAI_API_KEY": "k", "LITELLM_CUSTOMER_ID": "p-1", "LITELLM_TAGS": "projectId:p-1,taskId:t-1"}


def test_unarmed_the_cli_runs_as_it_did_before_the_image_followed_a_clock(spawned, monkeypatch):
    monkeypatch.setattr(codex_bridge, "clock_armed", lambda: False)

    _turn()

    [cli] = spawned
    assert cli["cmd"] == ["codex", "exec", "--cd", codex_bridge.WORKSPACE_DIR, "-", "--json",
                          "--dangerously-bypass-approvals-and-sandbox", "-c", 'model="gpt-x"', "-c", "features.goals=false"]
    assert cli["kwargs"]["env"] == _main_env()
    assert cli["proc"].stdin.data == b"hi"


def test_on_an_armed_clock_the_cli_keeps_its_environment_its_children_get_the_clocks_routes_and_the_model_its_date(
        spawned, monkeypatch):
    monkeypatch.setattr(codex_bridge, "clock_armed", lambda: True)

    _turn()

    date, cli = spawned
    # A child of this real-time server, without the real-time marker, so it shows the env's clock.
    assert date["cmd"] == ["date", "+%Y-%m-%d (%A)"]
    assert codex_bridge.REAL_TIME_ENV not in (date["kwargs"].get("env") or os.environ)
    overrides = _overrides(cli["cmd"])
    assert overrides[:2] == ['model="gpt-x"', "features.goals=false"]
    assert REAL_TIME <= set(overrides) and _shell(overrides)["HTTPS_PROXY"] == PROXY
    assert cli["kwargs"]["env"] == _main_env() and not PROXY_NAMES & set(cli["kwargs"]["env"])
    prompt = cli["proc"].stdin.data.decode()
    assert prompt.startswith("<environment_clock>\nToday is 2031-03-04 (Tuesday) in America/Chicago on this task's "
                             "clock, which `date` and every command you run read.")
    assert "current_date in <environment_context>" in prompt and prompt.endswith("</environment_clock>\n\nhi")


def test_a_task_on_an_armed_clock_reaches_the_cli_with_its_mcp_servers_on_real_time(codex, spawned, monkeypatch):
    monkeypatch.setattr(codex_bridge, "clock_armed", lambda: True)

    async def exercise():
        async with codex.app.router.lifespan_context(codex.app), codex.client() as client:
            for name, url in SERVERS.items():
                await client.post("/ext/mcp-config", json={"url": url, "name": name})
            await codex.send(client, "hi")

    asyncio.run(exercise())

    overrides = _overrides(spawned[-1]["cmd"])
    assert REAL_TIME <= set(overrides)
    direct = _shell(overrides)["NO_PROXY"].split(",")
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
