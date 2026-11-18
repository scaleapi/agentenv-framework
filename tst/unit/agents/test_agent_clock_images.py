"""Every agent image on the env's clock (FD-5916), whichever agent it is: each folder with ``clock/`` copies the
computer base's clock follower and egress proxy (``clock/``, ``egress/``) until the clock library image (FD-5700)
ships them, with libfaketime's init guard after the base's patch; runs the follower and the proxy, each as its own
user, and the A2A server as the agent's user on real time, under root's supervisord; preloads agent-env's libfaketime
for the agent's user alone; starts as another user as it did before; and its bridge decides the CLI's proxy with the
same ``clock_armed`` and ``egress_env`` as every other agent's."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

AGENTS_DIR = Path(__file__).parents[3] / "src" / "agent_env" / "agents" / "agents"
BASE = AGENTS_DIR.parents[1] / "computers" / "artifacts" / "ubuntu-24.04"
CLOCKED = sorted(p.parent.name for p in AGENTS_DIR.glob("*/clock") if p.is_dir())
PROXY = "http://127.0.0.1:3128"


def test_claude_code_follows_the_clock():
    assert "claude-code" in CLOCKED


@pytest.mark.parametrize("agent", CLOCKED)
@pytest.mark.parametrize("folder", ["clock", "egress"])
def test_the_clock_pieces_are_the_computer_bases_until_the_clock_library_image_ships_them(agent, folder):
    here = AGENTS_DIR / agent / folder
    copied = sorted(p.relative_to(here) for p in here.rglob("*") if p.is_file())

    assert copied
    assert [str(p) for p in copied if str(p) != "libfaketime.patch"
            and (BASE / folder / p).read_bytes() != (here / p).read_bytes()] == []


@pytest.mark.parametrize("agent", CLOCKED)
def test_the_agents_libfaketime_patch_is_the_bases_with_the_init_guard_after_it(agent):
    # A CLI that reads the clock while libfaketime's init runs (an allocator, as dlsym allocates) has pthread_once wait
    # on itself: Claude Code's `--version` never returned. The guard has a clock call made during the init read the
    # kernel's clock. The base doesn't take it yet, so the agent's copy is the base's patch, then the guard.
    base = (BASE / "clock" / "libfaketime.patch").read_text()
    patch = (AGENTS_DIR / agent / "clock" / "libfaketime.patch").read_text()
    guard = patch.removeprefix(base)

    assert patch.startswith(base) and guard.startswith("diff -ruN a/src/libfaketime.c b/src/libfaketime.c\n")
    assert "static __thread int ftpl_initializing = 0;" in guard
    assert "if (ftpl_initializing) return syscall(SYS_clock_gettime, clk_id, tp);" in guard
    assert "pthread_once(&initialized_once_control, ftpl_really_init_guarded);" in guard


def _program(agent: str, name: str) -> dict[str, str]:
    conf = (AGENTS_DIR / agent / "root.conf").read_text()
    section = re.search(rf"^\[program:{name}\]\n(.*?)(?=^\[|\Z)", conf, re.MULTILINE | re.DOTALL).group(1)
    return dict(line.split("=", 1) for line in section.splitlines() if "=" in line and not line.startswith(";"))


@pytest.mark.parametrize("agent", CLOCKED)
def test_the_follower_and_the_proxy_run_as_their_own_users_and_the_a2a_server_as_the_agents_on_real_time(agent):
    assert (_program(agent, "clock")["user"], _program(agent, "egress")["user"]) == ("agentenv-clock", "agentenv-egress")
    server = _program(agent, "agent")
    assert server["user"] == "agent"
    # The server's own interpreter, so the marker reaches it, not a shell that drops it on exec.
    assert server["command"].split()[0].rsplit("/", 1)[-1].startswith("python")
    assert 'AGENTENV_REAL_TIME="1"' in server["environment"]
    assert f'AGENTENV_EGRESS_PROXY="{PROXY}"' in server["environment"]


@pytest.mark.parametrize("agent", CLOCKED)
def test_the_image_preloads_the_library_for_the_agents_user_and_the_follower_alone_reads_its_config(agent):
    dockerfile = (AGENTS_DIR / agent / "Dockerfile").read_text()
    assert "> /etc/ld.so.preload" in dockerfile and "libfaketimeMT.so.1" in dockerfile
    assert 's/^AGENTENV_SESSION_UID=.*/AGENTENV_SESSION_UID=$(id -u agent)/' in dockerfile
    assert "install -d -o agent -g agentenv-clock -m 2750 /etc/agentenv/clock" in dockerfile
    assert 'CMD ["/usr/local/bin/start-agent"]' in dockerfile
    assert not re.search(r"^USER ", dockerfile, re.MULTILINE)


@pytest.mark.parametrize("agent", CLOCKED)
def test_started_as_another_user_the_image_runs_the_server_alone_as_before(agent):
    start = (AGENTS_DIR / agent / "start-agent").read_text()
    assert 'if [ "$(id -u)" != 0 ]; then\n    exec ' in start
    assert "AGENTENV_SESSION_USER=agent AGENTENV_EGRESS_NSS=off" in start


def _clock_helpers(agent: str) -> dict[str, str]:
    """The bridge's clock helpers, as source: the one module in the agent's folder that defines ``egress_env``."""
    [module] = [p for p in (AGENTS_DIR / agent).glob("*.py") if "def egress_env(" in p.read_text()]
    tree = ast.parse(module.read_text())
    names = {"clock_armed", "egress_env"}
    found = {node.name: ast.unparse(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    assert set(found) == names, (module, sorted(found))
    return found


@pytest.mark.parametrize("agent", CLOCKED)
def test_every_bridge_decides_the_clis_proxy_as_claude_codes_does(agent):
    assert _clock_helpers(agent) == _clock_helpers("claude-code")
