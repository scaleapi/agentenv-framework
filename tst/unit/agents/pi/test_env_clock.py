"""pi on the env's clock: its bridge hands the shared egress rule (tst/unit/agents/test_agent_egress.py) the model
endpoint's URL and every MCP server's, which the CLI calls itself, so on an armed clock it goes out through the image's
egress proxy, and to loopback, the hosts NO_PROXY names and plain-HTTP hosts directly; unarmed, it goes out as before. What every clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import subprocess
import sys
import tarfile

import pytest

from agent_env.utils.docker.context import BuildContext
from tst.unit.agents.pi.agent import AGENT_DIR, agent_egress, pi_agent

PROXY = "http://127.0.0.1:3128"

ENDPOINT = "https://litellm.example.com"


def _direct(env: dict[str, str]) -> set[str]:
    return set(env["NO_PROXY"].split(","))


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy():
    env = pi_agent.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal", "KEPT": "1"}, ENDPOINT,
                           clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert env["NODE_USE_ENV_PROXY"] == "1" and env["KEPT"] == "1"
    assert {"localhost", "host.docker.internal", "registry.internal"} <= _direct(env)


@pytest.mark.parametrize("environ, armed", [({"KEPT": "1"}, True), ({agent_egress.EGRESS_PROXY_ENV: PROXY, "KEPT": "1"}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    env = pi_agent.cli_env(environ, ENDPOINT, {"env": {"url": "http://10.0.0.7:8080/mcp"}}, clock_armed=armed)

    assert env == environ


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = pi_agent.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY}, base_url, clock_armed=True)

    assert (base_url.split("//")[1].split(":")[0] in _direct(env)) is direct


def test_the_envs_plain_http_mcp_servers_are_reached_directly_and_https_ones_through_the_proxy_on_an_armed_clock():
    servers = {"env": {"url": "http://gateway.internal:8080/mcp"}, "files": {"url": "https://mcp.example.com/mcp"},
               pi_agent.PEER_MCP_NAME: {"url": pi_agent.PEER_MCP_URL}}

    env = pi_agent.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY}, ENDPOINT, servers, clock_armed=True)

    # Node checks an HTTPS server's certificate against the env's date, which the proxy, on real time, doesn't.
    assert {"gateway.internal", "127.0.0.1"} <= _direct(env) and "mcp.example.com" not in _direct(env)
    assert "litellm.example.com" not in _direct(env)


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


def test_an_installed_agent_imports_the_shared_egress_rule_from_the_build_context_it_unpacks(tmp_path):
    """install/v1 unpacks the build context, which keeps the shared modules in the folders it takes them as, and
    start.sh puts those folders on the server's path."""
    archive = tmp_path / "ctx.tar.gz"
    BuildContext.of(AGENT_DIR, AGENT_DIR / "Dockerfile").write(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "pi-a2a", filter="data")
    start = (AGENT_DIR / "start.sh").read_text()
    [path] = [line for line in start.splitlines() if line.startswith("export PYTHONPATH=")]
    script = f'here={tmp_path / "pi-a2a"}; {path}; exec "$0" -c "import agent_egress, clockpage"'

    subprocess.run(["sh", "-c", script, sys.executable], check=True, env={"PATH": "/usr/bin:/bin"})
