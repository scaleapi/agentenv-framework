"""pi on the env's clock: where the image runs the clock, its bridge starts pi with the real-time marker, which
agent-env's libfaketime takes out of pi's environment, so pi stays on real time while every command it runs follows the
clock; and it hands the shared egress rule (tst/unit/agents/test_agent_egress.py) the model endpoint's URL and every MCP
server's, so on an armed clock pi and its commands go out through the image's egress proxy, and to loopback, the hosts
NO_PROXY names and plain-HTTP hosts directly. With the clock off, pi runs with the flags and environment it had before.
What every clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from tst.unit.agents.pi.agent import AGENT_DIR, agent_egress, pi_agent

PROXY = "http://127.0.0.1:3128"
ENDPOINT = "https://litellm.example.com"
FAKE_PI = Path(__file__).with_name("fake_pi.py")
ARMED_PAGE = agent_egress.clockpage.encode(2, 1_900_000_000 * 10**9, 10**18, 1.0, 1, 3, 0, 0)
UNARMED_PAGE = agent_egress.clockpage.encode(2, 0, 0, 1.0, 0, 3, 0, 0)
ROUTING = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
           "NODE_USE_ENV_PROXY"}
RUN_OWN = {"PI_CODING_AGENT_DIR", "PI_OFFLINE", "PI_SKIP_VERSION_CHECK", "PI_TELEMETRY", "AGENTENV_MCP_0_HEADER_0"}


def _direct(env: dict[str, str]) -> set[str]:
    return set(env["NO_PROXY"].split(","))


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy():
    env = pi_agent.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal", "KEPT": "1"}, ENDPOINT,
                           armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert env["NODE_USE_ENV_PROXY"] == "1" and env["KEPT"] == "1"
    assert {"localhost", "host.docker.internal", "registry.internal"} <= _direct(env)


@pytest.mark.parametrize("armed", [False, True])
def test_pi_stays_on_real_time_wherever_the_image_runs_the_clock_and_the_commands_it_runs_follow_it(armed):
    """pi compares its own clock with times it didn't make on the env's (before 1970 its compaction's UUIDv7 throws;
    its file locks compare its clock with their modification times), so it runs on real time, armed or not, as a clock
    may arm mid-turn; libfaketime takes the marker out of pi's environment, so what pi starts follows the clock."""
    env = pi_agent.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY}, ENDPOINT, armed=armed)

    assert env[pi_agent.REAL_TIME_ENV] == "1"


@pytest.mark.parametrize("armed", [False, True])
def test_an_installed_agent_whose_target_runs_no_clock_starts_pi_as_before(armed):
    environ = {"KEPT": "1"}

    assert pi_agent.cli_env(environ, ENDPOINT, {"env": {"url": "http://10.0.0.7:8080/mcp"}}, armed=armed) == environ


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = pi_agent.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY}, base_url, armed=True)

    assert (base_url.split("//")[1].split(":")[0] in _direct(env)) is direct


def test_the_envs_plain_http_mcp_servers_are_reached_directly_and_https_ones_through_the_proxy_on_an_armed_clock():
    servers = {"env": {"url": "http://gateway.internal:8080/mcp"}, "files": {"url": "https://abc-123.w.modal.host/mcp"},
               pi_agent.PEER_MCP_NAME: {"url": pi_agent.PEER_MCP_URL}}

    env = pi_agent.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY}, ENDPOINT, servers, armed=True)

    # A command pi runs checks an HTTPS server's certificate against the env's date, which the proxy, on real time,
    # doesn't; pi itself, on real time, is fine either way.
    assert {"gateway.internal", "127.0.0.1"} <= _direct(env) and "abc-123.w.modal.host" not in _direct(env)
    assert "litellm.example.com" not in _direct(env)


@dataclass
class Spawned:
    """How the agent started pi: the argv and environment it passed, and what the fake CLI recorded of them."""

    environ: dict[str, str]
    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    record: dict = field(default_factory=dict)


@pytest.fixture
def run_task(tmp_path, monkeypatch):
    """Runs a task through the agent's A2A route with the fake CLI, on a clock page armed or not, in an image that runs
    the clock (its server's environment names the egress proxy) or, ``clocked=False``, an installed agent."""
    real_exec = pi_agent.asyncio.create_subprocess_exec
    record = tmp_path / "record.json"

    def run(*, armed: bool, clocked: bool = True) -> Spawned:
        page = tmp_path / "agentenv-clock"
        page.write_bytes(ARMED_PAGE if armed else UNARMED_PAGE)
        spawned = Spawned(environ={
            "PATH": "/usr/bin:/bin", "FAKE_PI_RECORD": str(record), "LITELLM_BASE_URL": "http://litellm.test/v1",
            "LITELLM_API_KEY": "sk-test", "NO_PROXY": "registry.internal", "TZ": "America/Chicago",
            **({agent_egress.EGRESS_PROXY_ENV: PROXY} if clocked else {})})
        monkeypatch.setattr(agent_egress.clockpage, "PAGE", str(page))
        monkeypatch.setattr(pi_agent.os, "environ", dict(spawned.environ))

        async def recording_exec(*argv, env, **kwargs):
            spawned.argv, spawned.env = list(argv), dict(env)
            return await real_exec(*argv, env=env, **kwargs)

        monkeypatch.setattr(pi_agent.asyncio, "create_subprocess_exec", recording_exec)
        agent = pi_agent.PiAgent(home=tmp_path / "home", workspace=tmp_path / "ws",
                                 pi_command=(sys.executable, str(FAKE_PI)))
        with TestClient(agent.create_app(), client=("127.0.0.1", 50000)) as client:
            client.post("/ext/agent-config", json={"model": "anthropic/claude-sonnet", "effort": "high"})
            client.post("/ext/mcp-config", json={"name": "env", "url": "https://abc-123.w.modal.host/mcp",
                                                 "headers": {"Authorization": "Bearer t"}})
            reply = client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
                "message": {"kind": "message", "messageId": "m-1", "role": "user", "contextId": "ctx-1",
                            "parts": [{"kind": "text", "text": "what's the date?"}]},
                "configuration": {"blocking": True}}}).json()
        assert reply["result"]["status"]["state"] == "completed", reply
        spawned.record = json.loads(record.read_text())
        return spawned

    return run


def test_on_an_armed_clock_the_agent_starts_pi_on_real_time_through_the_proxy(run_task):
    seen = run_task(armed=True).record["env"]

    assert seen[pi_agent.REAL_TIME_ENV] == "1"
    assert seen["HTTPS_PROXY"] == seen["HTTP_PROXY"] == PROXY and seen["NODE_USE_ENV_PROXY"] == "1"
    direct = set(seen["NO_PROXY"].split(","))
    assert {"litellm.test", "registry.internal", "127.0.0.1"} <= direct and "abc-123.w.modal.host" not in direct


def test_with_the_clock_off_the_agent_starts_pi_with_the_flags_and_environment_it_had_before(run_task):
    """Off, pi's environment is the server's, unchanged, with the run's own variables and the real-time marker, which
    libfaketime takes out before pi reads it; arming the clock changes the proxy variables alone, and no flag."""
    off, on = run_task(armed=False), run_task(armed=True)

    assert set(off.env) - set(off.environ) == {pi_agent.REAL_TIME_ENV, *RUN_OWN}
    assert {name: off.env[name] for name in off.environ} == off.environ
    assert off.argv == on.argv
    assert {name for name in {*off.env, *on.env} if off.env.get(name) != on.env.get(name)} == {*ROUTING,
                                                                                              "PI_CODING_AGENT_DIR"}


@pytest.mark.parametrize("armed", [False, True])
def test_an_installed_agent_whose_target_runs_no_clock_starts_pi_with_the_runs_own_variables_alone(run_task, armed):
    spawned = run_task(armed=armed, clocked=False)

    assert set(spawned.env) - set(spawned.environ) == RUN_OWN
    assert {name: spawned.env[name] for name in spawned.environ} == spawned.environ


def test_start_agent_keeps_a_providers_ca_bundle_trusted_by_the_server_and_the_proxy(tmp_path):
    # Sail points SSL_CERT_FILE and the like at the bundle it signs model requests under; root.conf points the server
    # at agent-env's bundles, so start-agent adds the provider's to them and to the proxy's roots, once each.
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
