"""Codex on the env's clock: its bridge hands the shared egress rule (tst/unit/agents/test_agent_egress.py) the model
endpoint's URL, so on an armed clock the CLI goes out through the image's egress proxy, and to loopback, the hosts
NO_PROXY names and a plain-HTTP model endpoint directly; unarmed, it goes out as before. What every clocked image shares
is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import subprocess

import pytest

from tst.unit.agents.codex.agent import AGENT_DIR, agent_egress, codex_bridge

PROXY = "http://127.0.0.1:3128"


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy():
    env = codex_bridge.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"},
                               clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert {"localhost", "host.docker.internal", "registry.internal"} <= set(env["NO_PROXY"].split(","))


@pytest.mark.parametrize("environ, armed", [({}, True), ({agent_egress.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(codex_bridge.cli_env(environ, clock_armed=armed))


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = codex_bridge.cli_env({agent_egress.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": base_url}, clock_armed=True)
    assert (base_url.split("//")[1].split(":")[0] in env["NO_PROXY"].split(",")) is direct


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
