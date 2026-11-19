"""Gemini on the env's clock: its bridge hands the shared egress rule (tst/unit/agents/test_agent_egress.py) the model
endpoint's URL and every MCP server's, so on an armed clock the commands the model runs go out through the image's
egress proxy, by ALL_PROXY, and to loopback, the hosts NO_PROXY names and plain-HTTP hosts directly; the CLI itself, which sends everything through a proxy its HTTP(S)_PROXY names, NO_PROXY or not, is
handed none, so it reaches the attribution proxy and the MCP servers directly. Unarmed, it all goes out as before. What
every clocked image shares is checked in tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tst.unit.agents.gemini.agent import AGENT_DIR, agent_egress, gemini_bridge

PROXY = "http://127.0.0.1:3128"

SETTINGS = Path("/tmp/settings.json")
ATTRIBUTION_PROXY = "http://127.0.0.1:8000/litellm-gemini/token"


def _env(environ, **kwargs):
    return gemini_bridge.cli_env(environ, system_settings=SETTINGS, base_url=ATTRIBUTION_PROXY, **kwargs)


def test_on_an_armed_clock_the_models_commands_go_out_through_the_proxy_and_the_cli_directly():
    env = _env({agent_egress.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"}, clock_armed=True)

    assert env["ALL_PROXY"] == env["all_proxy"] == PROXY
    assert not {"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"} & set(env)
    assert {"localhost", "127.0.0.1", "host.docker.internal", "registry.internal"} <= set(env["NO_PROXY"].split(","))
    assert env["GOOGLE_GEMINI_BASE_URL"] == ATTRIBUTION_PROXY


@pytest.mark.parametrize("environ, armed", [({}, True), ({agent_egress.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    assert not {"ALL_PROXY", "HTTPS_PROXY", "NO_PROXY"} & set(_env(environ, clock_armed=armed))


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    for name in ("LITELLM_BASE_URL", "GOOGLE_GEMINI_BASE_URL"):
        env = _env({agent_egress.EGRESS_PROXY_ENV: PROXY, name: base_url}, clock_armed=True)
        assert (base_url.split("//")[1].split(":")[0] in env["NO_PROXY"].split(",")) is direct


def test_the_envs_plain_http_mcp_servers_are_reached_directly_and_https_ones_through_the_proxy_on_an_armed_clock():
    """The CLI's MCP client checks an HTTPS server's certificate against the env's date, which the proxy, on real time,
    doesn't."""
    env = _env({agent_egress.EGRESS_PROXY_ENV: PROXY}, clock_armed=True,
               mcp_urls=["http://gateway:8080/mcp", "https://abc-123.w.modal.host/mcp", ""])

    direct = env["NO_PROXY"].split(",")
    assert "gateway" in direct and "abc-123.w.modal.host" not in direct and "" not in direct
    assert env["ALL_PROXY"] == PROXY


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
