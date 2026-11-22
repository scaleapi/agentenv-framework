"""OpenCode on the env's clock: its bridge hands the shared egress rule (tst/unit/agents/test_agent_egress.py) the
model endpoint's URL and every MCP server's, which the CLI connects to itself, so on an armed clock it goes out through
the image's egress proxy, and to loopback, the hosts NO_PROXY names and plain-HTTP hosts directly; unarmed, it goes out
as before. The CLI itself runs on real time and its plugin (env_date.js) tells the model the env's date, which it reads
from a command as the model's own commands do. What every clocked image shares is checked in
tst/unit/agents/test_agent_clock_images.py."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from tst.unit.agents.opencode.agent import AGENT_DIR, agent_egress, opencode_bridge

PROXY = "http://127.0.0.1:3128"


@pytest.fixture(autouse=True)
def _sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(opencode_bridge, "SESSIONS_ROOT", str(tmp_path))


def _direct(env: dict[str, str]) -> set[str]:
    return set(env["NO_PROXY"].split(","))


def test_on_an_armed_clock_the_cli_goes_out_through_the_proxy():
    env = opencode_bridge.cli_env("ctx", {agent_egress.EGRESS_PROXY_ENV: PROXY, "NO_PROXY": "registry.internal"},
                                  clock_armed=True)

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert {"localhost", "host.docker.internal", "registry.internal"} <= _direct(env)
    assert env["XDG_DATA_HOME"] == str(opencode_bridge.context_data_dir("ctx") / "share")


@pytest.mark.parametrize("environ, armed", [({}, True), ({agent_egress.EGRESS_PROXY_ENV: PROXY}, False)])
def test_without_the_proxy_or_an_armed_clock_the_cli_goes_out_directly_as_before(environ, armed):
    env = opencode_bridge.cli_env("ctx", environ, clock_armed=armed)

    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(env)
    assert set(env) == {*environ, "XDG_DATA_HOME", "XDG_STATE_HOME", "AGENTENV_REAL_TIME"}


@pytest.mark.parametrize("armed", [True, False])
def test_the_cli_starts_on_real_time_so_its_sessions_keep_their_order_at_any_env_date(armed):
    """libfaketime takes the marker out of the CLI's environment, so what the CLI starts reads the env's time."""
    env = opencode_bridge.cli_env("ctx", {agent_egress.EGRESS_PROXY_ENV: PROXY}, clock_armed=armed)

    assert env["AGENTENV_REAL_TIME"] == "1"


def test_every_turn_loads_the_date_plugin_which_the_image_and_an_install_carry_beside_the_bridge():
    config = opencode_bridge._build_config("m", None, {}, None)

    assert config["plugin"] == [(AGENT_DIR / "env_date.js").as_uri()]
    assert opencode_bridge.ENV_DATE_PLUGIN == AGENT_DIR / "env_date.js" and opencode_bridge.ENV_DATE_PLUGIN.is_file()
    assert re.search(r"^COPY .*\benv_date\.js\b.* \./$", (AGENT_DIR / "Dockerfile").read_text(), re.MULTILINE)
    assert "env_date.js" in re.search(r'^AGENT_FILES="([^"]+)"$', (AGENT_DIR / "install.sh").read_text(),
                                      re.MULTILINE).group(1).split()


# OpenCode's system prompt as its SystemPrompt.environment writes it (opencode-ai 1.17.18, src/session/system.ts).
SYSTEM = ("You are OpenCode, the best coding agent on the planet.\n"
          "You are powered by the model named m. The exact model ID is litellm/m\n"
          "Here is some useful information about the environment you are running in:\n<env>\n"
          "  Working directory: /app\n  Workspace root folder: /app\n  Is directory a git repo: no\n"
          "  Platform: linux\n  Today's date: Sat Nov 21 2026\n</env>\n"
          "The task's notes say: Today's date: as written")
HARNESS = """
const plugin = (await import(process.argv[1])).default
const hooks = await plugin.server({})
const out = { system: JSON.parse(process.argv[2]) }
const search = { description: "Searches the web. The current year is 2026. You MUST use this year.", parameters: {} }
const other = { description: "The current year is 2026.", parameters: {} }
try {
  await hooks["experimental.chat.system.transform"]({ model: {} }, out)
  await hooks["tool.definition"]({ toolID: "websearch" }, search)
  await hooks["tool.definition"]({ toolID: "bash" }, other)
} catch (error) {
  console.log(JSON.stringify({ error: String(error.message) }))
  process.exit(0)
}
console.log(JSON.stringify({ id: plugin.id, system: out.system, search: search.description, other: other.description }))
"""


def _plugin(tmp_path, date_out: str | None, system: list[str] = [SYSTEM]) -> tuple[dict, str]:
    """The plugin run by node as the CLI's Bun runs it, with a ``date`` that prints ``date_out`` (fails when None)
    and records the environment it got."""
    node = shutil.which("node")
    assert node, "node runs the CLI plugin's test"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "date").write_text('#!/bin/sh\nenv > "$FAKE_DATE_ENV"\n[ "$1" = "+%s %N" ] || exit 2\n'
                                  '[ -n "$FAKE_DATE_OUT" ] || exit 1\necho "$FAKE_DATE_OUT"\n')
    (bin_dir / "date").chmod(0o755)
    plugin = tmp_path / "env_date.mjs"
    shutil.copy(AGENT_DIR / "env_date.js", plugin)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "TZ": "UTC", "AGENTENV_REAL_TIME": "1",
           "FAKE_DATE_ENV": str(tmp_path / "date.env"), **({"FAKE_DATE_OUT": date_out} if date_out else {})}
    done = subprocess.run([node, "--input-type=module", "-e", HARNESS, plugin.as_uri(), json.dumps(system)],
                          capture_output=True, text=True, env=env, timeout=30, check=True)
    return json.loads(done.stdout), (tmp_path / "date.env").read_text()


@pytest.mark.parametrize("date_out, today, year", [
    ("1801742400 000000000", "Thu Feb 04 2027", 2027),
    ("7258161600 250000000", "Wed Jan 01 2200", 2200),
    ("-631108800 000000000", "Sun Jan 01 1950", 1950),
    # Half a second before 1950: date prints the second below and a positive fraction.
    ("-631152001 500000000", "Sat Dec 31 1949", 1949),
])
def test_the_plugin_tells_the_model_the_env_date_a_command_reads(tmp_path, date_out, today, year):
    result, child_env = _plugin(tmp_path, date_out)

    assert result["id"] == "agentenv-env-date"
    assert result["system"] == [SYSTEM.replace("Today's date: Sat Nov 21 2026", f"Today's date: {today}")]
    assert result["search"] == f"Searches the web. The current year is {year}. You MUST use this year."
    assert result["other"] == "The current year is 2026."
    assert "AGENTENV_REAL_TIME" not in child_env and "TZ=UTC" in child_env


def test_the_plugin_leaves_a_prompt_without_the_env_block_alone(tmp_path):
    system = ["You are a subagent.\nToday's date: whatever the task says"]

    assert _plugin(tmp_path, "1801742400 000000000", system)[0]["system"] == system


@pytest.mark.parametrize("date_out", [None, "not a time"])
def test_a_date_the_plugin_cant_read_fails_the_turn_rather_than_show_real_time(tmp_path, date_out):
    result, _ = _plugin(tmp_path, date_out)

    assert result["error"]


@pytest.mark.parametrize("base_url, direct", [("http://10.0.0.5:4000", True), ("https://litellm.example.com", False)])
def test_a_model_endpoint_over_plain_http_is_reached_directly_on_an_armed_clock(base_url, direct):
    env = opencode_bridge.cli_env("ctx", {agent_egress.EGRESS_PROXY_ENV: PROXY, "LITELLM_BASE_URL": base_url},
                                  clock_armed=True)
    assert (base_url.split("//")[1].split(":")[0] in _direct(env)) is direct


def test_the_envs_plain_http_mcp_servers_are_reached_directly_and_https_ones_through_the_proxy_on_an_armed_clock():
    """Bun checks an HTTPS server's certificate against the env's date, which the proxy, on real time, doesn't."""
    servers = {"env": {"url": "http://gateway:8080/mcp"},
               "remote": {"url": "https://abc-123.w.modal.host/mcp", "headers": {"x": "y"}}}
    env = opencode_bridge.cli_env("ctx", {agent_egress.EGRESS_PROXY_ENV: PROXY}, mcp_servers=servers,
                                  clock_armed=True)

    assert "gateway" in _direct(env) and "abc-123.w.modal.host" not in _direct(env)


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

