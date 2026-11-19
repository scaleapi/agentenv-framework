"""OpenCode on the env's clock (FD-5916): while the clock page is armed the CLI goes out through the image's egress
proxy, and to the hosts NO_PROXY names, the env's MCP servers, which it connects to itself, and a plain-HTTP model
endpoint directly; unarmed, it goes out as before. What every clocked image shares is checked in
tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import pytest

from tst.unit.agents.opencode.agent import opencode_bridge

PROXY = "http://127.0.0.1:3128"



@pytest.fixture(autouse=True)
def _sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(opencode_bridge, "SESSIONS_ROOT", str(tmp_path))


def _direct(env: dict[str, str]) -> set[str]:
    return set(env["NO_PROXY"].split(","))


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy():
    env = opencode_bridge.cli_env("ctx", {opencode_bridge.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"},
                                  clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert {"localhost", "host.docker.internal", "registry.internal"} <= _direct(env)
    # Still the context's own session store.
    assert env["XDG_DATA_HOME"] == str(opencode_bridge.context_data_dir("ctx") / "share")


@pytest.mark.parametrize("environ, armed", [({}, True), ({opencode_bridge.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    env = opencode_bridge.cli_env("ctx", environ, clock_armed=armed)

    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(env)
    assert set(env) == {*environ, "XDG_DATA_HOME", "XDG_STATE_HOME"}


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = opencode_bridge.cli_env("ctx", {opencode_bridge.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": base_url},
                                  clock_armed=True)
    assert (base_url.split("//")[1].split(":")[0] in _direct(env)) is direct


def test_the_envs_plain_http_mcp_servers_are_reached_directly_and_https_ones_through_the_proxy_on_an_armed_clock():
    servers = {"env": {"url": "http://host.docker.internal:8080/mcp"},
               "gateway": {"url": "https://mcp.gateway.internal/mcp", "headers": {"x": "y"}}}
    env = opencode_bridge.cli_env("ctx", {opencode_bridge.EGRESS_PROXY_ENV: PROXY}, mcp_servers=servers,
                                  clock_armed=True)

    # Bun checks an HTTPS server's certificate against the env's date, which the proxy, on real time, doesn't.
    assert "host.docker.internal" in _direct(env) and "mcp.gateway.internal" not in _direct(env)


def test_start_agent_keeps_a_providers_ca_bundle_trusted_by_the_server_and_the_proxy(tmp_path):
    # Sail points SSL_CERT_FILE and the like at the bundle it signs model requests under; root.conf points the server
    # at agent-env's bundles, so start-agent adds the provider's to them and to the proxy's roots, once each.
    import subprocess
    from pathlib import Path

    start = (Path(__file__).parents[4] / "src/agent_env/agents/agents/opencode/start-agent").read_text()
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
