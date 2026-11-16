"""Claude Code on the env's clock (FD-5916): the image runs the computer base's clock follower and egress proxy, each
as its own user, and agent-env's libfaketime for the agent's user, so the CLI and every command the model runs read
the env's time once sync_env_clock has the agent follow; the A2A server stays on real time. The CLI goes out through
the egress proxy, and to the env's MCP servers directly."""

from __future__ import annotations

import re

import pytest

from tst.unit.agents.claude_code.agent import AGENT_DIR, claude_bridge

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()
ROOT_CONF = (AGENT_DIR / "root.conf").read_text()
BASE = AGENT_DIR.parents[2] / "computers" / "artifacts" / "ubuntu-24.04"
PROXY = "http://127.0.0.1:3128"


@pytest.mark.parametrize("folder", ["clock", "egress"])
def test_the_clock_pieces_are_the_computer_bases_until_the_clock_library_image_ships_them(folder):
    copied = sorted(p.relative_to(AGENT_DIR / folder) for p in (AGENT_DIR / folder).rglob("*") if p.is_file())

    assert copied
    assert [str(p) for p in copied if (BASE / folder / p).read_bytes() != (AGENT_DIR / folder / p).read_bytes()] == []


def _program(name: str) -> dict[str, str]:
    section = re.search(rf"^\[program:{name}\]\n(.*?)(?=^\[|\Z)", ROOT_CONF, re.MULTILINE | re.DOTALL).group(1)
    return dict(line.split("=", 1) for line in section.splitlines() if "=" in line and not line.startswith(";"))


def test_the_follower_and_the_proxy_run_as_their_own_users_and_the_a2a_server_as_the_agents_on_real_time():
    assert (_program("clock")["user"], _program("egress")["user"]) == ("agentenv-clock", "agentenv-egress")
    agent = _program("agent")
    assert agent["user"] == "agent"
    # uvicorn itself, so the marker reaches the server and not a shell that drops it on exec.
    assert agent["command"].startswith("/usr/local/bin/python3 -m uvicorn a2a_server:app")
    assert 'AGENTENV_REAL_TIME="1"' in agent["environment"]
    assert f'AGENTENV_EGRESS_PROXY="{PROXY}"' in agent["environment"]
    assert 'NODE_EXTRA_CA_CERTS="/etc/agentenv/ca.pem"' in agent["environment"]


def test_the_image_preloads_the_library_for_the_agents_user_and_the_follower_alone_reads_its_config():
    assert "> /etc/ld.so.preload" in DOCKERFILE and "libfaketimeMT.so.1" in DOCKERFILE
    assert 's/^AGENTENV_SESSION_UID=.*/AGENTENV_SESSION_UID=$(id -u agent)/' in DOCKERFILE
    assert "install -d -o agent -g agentenv-clock -m 2750 /etc/agentenv/clock" in DOCKERFILE
    assert not re.search(r"^USER ", DOCKERFILE, re.MULTILINE)


def test_started_as_another_user_the_image_runs_the_server_alone_as_before():
    start = (AGENT_DIR / "start-agent").read_text()
    assert 'if [ "$(id -u)" != 0 ]; then\n    exec /opt/agent/entrypoint.sh' in start
    assert "AGENTENV_SESSION_USER=agent AGENTENV_EGRESS_NSS=off" in start


def test_where_the_image_runs_the_proxy_the_cli_goes_out_through_it_and_to_the_envs_servers_directly():
    env = claude_bridge.cli_env({claude_bridge.EGRESS_PROXY_ENV: PROXY}, direct_hosts=["gateway", "", "gateway"])

    assert {env[name] for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy")} == {PROXY}
    assert env["NO_PROXY"] == env["no_proxy"] == "localhost,127.0.0.1,::1,host.docker.internal,gateway"
    assert env["NODE_USE_ENV_PROXY"] == "1"


def test_without_the_proxy_the_cli_is_run_as_before():
    assert not {"HTTPS_PROXY", "NO_PROXY"} & set(claude_bridge.cli_env({}))
