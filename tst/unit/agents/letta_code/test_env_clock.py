"""Letta Code on the env's clock: its bridge hands the shared egress rule (tst/unit/agents/test_agent_egress.py) every
MCP server's URL, which the CLI connects to itself, so on an armed clock the CLI goes out through the image's egress
proxy, and to loopback, the model relay's among it, the hosts NO_PROXY names and plain-HTTP hosts directly; unarmed, it
goes out as before. What every clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import subprocess

import pytest

from tst.unit.agents.letta_code.agent import AGENT_DIR, agent_egress
from tst.unit.agents.letta_code.agent import letta_code_bridge as bridge

PROXY = "http://127.0.0.1:3128"


@pytest.fixture(autouse=True)
def _sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path))


def _env(monkeypatch, environ: dict[str, str], *, urls=(), armed: bool) -> dict[str, str]:
    for name in (agent_egress.EGRESS_PROXY_ENV, "NO_PROXY", "no_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY",
                 "http_proxy", "ALL_PROXY", "all_proxy"):
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
