"""Every agent image on the env's clock, whichever agent it is: each folder whose ``shared_sources.toml`` takes the
clock takes the computer bases' clock library and egress proxy from agent-env's one copy, whose libfaketime patch has
the init guard; runs the follower and the proxy, each as its own user, and the A2A server as the agent's user on real
time, under root's supervisord; preloads agent-env's libfaketime for the agent's user alone; starts as another user as
it did before; stops when the A2A server exits, as the image did when the server was its main process, while the
follower and the proxy restart in place; and its bridge takes ``clock_armed`` and ``egress_env`` from the one shared
``agent_egress``, in a folder only agent images take, which its image copies beside it."""

from __future__ import annotations

import ast
import configparser
import json
import posixpath
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

AGENTS_DIR = Path(__file__).parents[3] / "src" / "agent_env" / "agents" / "agents"
SOURCES = AGENTS_DIR.parents[1]
CLOCKED = sorted(p.parent.name for p in AGENTS_DIR.glob("*/shared_sources.toml")
                 if "clock" in tomllib.loads(p.read_text()))
PROXY = "http://127.0.0.1:3128"


def test_claude_code_follows_the_clock():
    assert "claude-code" in CLOCKED


@pytest.mark.parametrize("agent", CLOCKED)
@pytest.mark.parametrize(("part", "shared"), [("clock", "utils/faketime_clock"), ("egress", "utils/egress_proxy")])
def test_the_clock_pieces_are_taken_from_the_one_copy_the_computer_bases_take(agent, part, shared):
    taken = tomllib.loads((AGENTS_DIR / agent / "shared_sources.toml").read_text())

    assert taken[part] == shared
    assert not (AGENTS_DIR / agent / part).exists(), f"{agent}/{part} is a copy: take it in shared_sources.toml instead"


@pytest.mark.parametrize("agent", CLOCKED)
def test_the_egress_rule_is_taken_from_the_one_copy_only_agent_images_take(agent):
    """A computer base that took it would be rebuilt, and every computer composed on it, whenever the rule changes."""
    taken = tomllib.loads((AGENTS_DIR / agent / "shared_sources.toml").read_text())
    others = [p for p in SOURCES.glob("*/*/*/shared_sources.toml") if AGENTS_DIR not in p.parents]

    assert taken["agent_clock"] == "utils/agent_clock"
    assert not (AGENTS_DIR / agent / "agent_clock").exists()
    assert others and not [p for p in others if "utils/agent_clock" in tomllib.loads(p.read_text()).values()]


def test_the_libfaketime_every_clocked_image_builds_has_the_init_guard():
    # A CLI that reads the clock while libfaketime's init runs (an allocator, as dlsym allocates) has pthread_once wait
    # on itself: Claude Code's `--version` never returned. The guard has a clock call made during the init read the
    # kernel's clock.
    patch = (SOURCES / "utils" / "faketime_clock" / "libfaketime.patch").read_text()

    assert "static __thread int ftpl_initializing = 0;" in patch
    assert "if (ftpl_initializing) return syscall(SYS_clock_gettime, clk_id, tp);" in patch
    assert "pthread_once(&initialized_once_control, ftpl_really_init_guarded);" in patch


def _conf(agent: str) -> configparser.RawConfigParser:
    """The image's root.conf as supervisord reads it, with its inline comments."""
    conf = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"))
    conf.read_string((AGENTS_DIR / agent / "root.conf").read_text())
    return conf


def _program(agent: str, name: str) -> dict[str, str]:
    return dict(_conf(agent)[f"program:{name}"])


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
def test_the_a2a_server_is_never_restarted_and_the_follower_and_the_proxy_are(agent):
    """A server restarted in place would have lost the env's MCP servers, its config, its peers and its running tasks."""
    assert (_program(agent, "agent")["autorestart"], _program(agent, "agent")["startretries"]) == ("false", "0")
    assert _program(agent, "clock")["autorestart"] == _program(agent, "egress")["autorestart"] == "true"


# What supervisord sends a listener: each event's header line, then its payload of len bytes.
LISTENER_HOST = r"""
import json, signal, subprocess, sys
argv, events = json.loads(sys.argv[1]), json.loads(sys.argv[2])
terminated = []
signal.signal(signal.SIGTERM, lambda *_: terminated.append(1))
listener = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
after = []
for serial, (name, payload) in enumerate(events):
    assert listener.stdout.readline() == b"READY\n"
    body = payload.encode()
    listener.stdin.write(f"ver:3.0 server:supervisor serial:{serial} pool:p poolserial:{serial} eventname:{name} "
                         f"len:{len(body)}\n".encode() + body)
    listener.stdin.flush()
    assert listener.stdout.readline() == b"RESULT 2\n" and listener.stdout.read(2) == b"OK"
    after.append(len(terminated))
listener.stdin.close()
print(json.dumps({"terminated_after": after, "stderr": listener.stderr.read().decode(), "exit": listener.wait()}))
"""


@pytest.mark.parametrize("agent", CLOCKED)
def test_the_a2a_servers_exit_stops_supervisord_and_the_followers_or_the_proxys_doesnt(agent):
    """The listener, run as supervisord runs it, under a stand-in for supervisord that counts the SIGTERMs it gets."""
    [section] = [name for name in _conf(agent).sections() if name.startswith("eventlistener:")]
    listener = _conf(agent)[section]
    assert set(listener["events"].split(",")) == {"PROCESS_STATE_EXITED", "PROCESS_STATE_FATAL"}
    assert "%" not in listener["command"]
    events = [("PROCESS_STATE_EXITED", "processname:clock groupname:clock from_state:RUNNING expected:0 pid:7"),
              ("PROCESS_STATE_FATAL", "processname:egress groupname:egress from_state:BACKOFF"),
              ("PROCESS_STATE_EXITED", "processname:agent-exit groupname:agent-exit from_state:RUNNING expected:1 pid:8"),
              ("PROCESS_STATE_EXITED", "processname:agent groupname:agent from_state:RUNNING expected:0 pid:9"),
              ("PROCESS_STATE_FATAL", "processname:agent groupname:agent from_state:BACKOFF")]

    host = subprocess.run([sys.executable, "-I", "-c", LISTENER_HOST, json.dumps(shlex.split(listener["command"])),
                           json.dumps(events)], capture_output=True, text=True, timeout=30, check=True)

    result = json.loads(host.stdout)
    assert result["terminated_after"] == [0, 0, 0, 1, 2]
    assert result["stderr"].count("the A2A server exited") == 2 and result["exit"] == 0


@pytest.mark.parametrize("agent", CLOCKED)
def test_the_agents_proxy_reaches_private_networks_where_its_model_endpoint_may_be_and_a_computers_doesnt(agent):
    # The agent's processes reach those directly anyway; its model endpoint, over TLS, goes through the proxy.
    assert 'AGENTENV_EGRESS_PRIVATE="allow"' in _program(agent, "egress")["environment"].split(",")
    base = (SOURCES / "computers" / "artifacts" / "ubuntu-24.04" / "root.conf").read_text()
    assert "AGENTENV_EGRESS_PRIVATE" not in base


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


SHARED = ("clock_armed", "egress_env")
SHARED_FILES = ("agent_clock/agent_egress.py", "clock/clockpage.py")


def _modules(agent: str) -> dict[str, ast.Module]:
    return {path.name: ast.parse(path.read_text()) for path in sorted((AGENTS_DIR / agent).glob("*.py"))}


def _bridges(agent: str) -> list[str]:
    """The agent's modules that take the egress rule from the shared module."""
    return [name for name, tree in _modules(agent).items()
            if any(isinstance(node, ast.ImportFrom) and node.module == "agent_egress"
                   and set(SHARED) <= {alias.name for alias in node.names} for node in tree.body)]


@pytest.mark.parametrize("agent", CLOCKED)
def test_every_bridge_takes_the_egress_rule_from_the_shared_module_and_defines_none_of_it(agent):
    assert _bridges(agent), f"no module of {agent}'s imports {SHARED} from agent_egress"
    for name, tree in _modules(agent).items():
        defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        defined |= {target.id for node in tree.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}
        own = defined & {*SHARED, "EGRESS_PROXY_ENV"}
        assert not own, (name, sorted(own))


def _copied_to(dockerfile: str) -> dict[str, set[str]]:
    """The folders the image's last stage copies each file of its build context into, by the file's name there."""
    workdir, folders = "/", {}
    stage = re.split(r"^FROM .*$", dockerfile, flags=re.MULTILINE)[-1]
    for instruction, args in re.findall(r"^(WORKDIR|COPY) (.+)$", stage, re.MULTILINE):
        if instruction == "WORKDIR":
            workdir = posixpath.join(workdir, args)
        elif not re.search(r"(^| )--from=", args):
            *sources, dest = [arg for arg in args.split() if not arg.startswith("--")]
            dest = posixpath.join(workdir, dest)
            folder = posixpath.normpath(dest if dest.endswith("/") or len(sources) > 1 else posixpath.dirname(dest))
            for source in sources:
                folders.setdefault(source, set()).add(folder)
    return folders


@pytest.mark.parametrize("agent", CLOCKED)
def test_every_clocked_image_copies_the_shared_module_and_the_page_format_beside_its_bridge(agent):
    folders = _copied_to((AGENTS_DIR / agent / "Dockerfile").read_text())

    assert set.intersection(*(folders.get(name, set()) for name in (*_bridges(agent), *SHARED_FILES))), folders
