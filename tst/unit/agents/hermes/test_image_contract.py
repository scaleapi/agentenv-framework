"""The agent's folder is its build context: the Dockerfile copies only files the context sends, the image carries every
module the server imports in the folder it runs the server from, its build fails, rather than a turn on an armed
clock, when the pinned Hermes' launcher isn't one the bridge reads, and the A2A server's exit stops the container, as
when the server was the container's process, while the follower and the proxy restart in place."""

from __future__ import annotations

import ast
import configparser
import os
import re
import select
import shlex
import signal
import subprocess
import sys
import time

import pytest

from agent_env.utils.docker.context import BuildContext
from tst.unit.agents.hermes.agent import AGENT_DIR
from tst.unit.agents.test_agent_clock_images import _copied_to

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()
SENT = {entry.name: entry.path for entry in BuildContext.of(AGENT_DIR, AGENT_DIR / "Dockerfile").entries}
SERVER_DIR = "/opt/agent"


def _imported(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_every_file_the_image_copies_is_in_its_build_context():
    copied = list(_copied_to(DOCKERFILE))

    assert copied and all(name in SENT or any(sent.startswith(name.rstrip("/") + "/") for sent in SENT)
                          for name in copied), copied


def test_the_image_carries_every_module_the_server_imports_beside_it():
    """The folder's own modules and those the context takes from agent-env's sources, the egress rule and the clock
    page's format among them, each copied into the folder uvicorn imports the server from."""
    modules = {name.rsplit("/", 1)[-1].removesuffix(".py"): name for name in SENT
               if name.endswith(".py") and name.count("/") <= 1 and not name.startswith("agentenv_plugin/")}
    needed, pending = set(), ["a2a_server"]
    while pending:
        if (name := modules[pending.pop()]) not in needed:
            needed.add(name)
            pending += _imported(SENT[name].read_text()) & set(modules)
    folders = _copied_to(DOCKERFILE)

    assert {"hermes_bridge.py", "hermes_kit.py", "agent_clock/agent_egress.py", "clock/clockpage.py"} <= needed
    assert all(SERVER_DIR in folders.get(name, set()) for name in needed), (needed, folders)
    # Beside the bridge too, where its PLUGIN_DIR is: each profile's plugins/agentenv links to it.
    assert SERVER_DIR in folders["agentenv_plugin"]


def test_the_build_reads_the_pinned_hermes_launcher_as_an_armed_turn_does():
    assert re.search(r"^RUN .*python3 -c 'import hermes_bridge; hermes_bridge\.launcher_command\(\)'$", DOCKERFILE,
                     re.MULTILINE)


def _root_conf() -> configparser.RawConfigParser:
    """root.conf as supervisord reads it: ';' or '#' after whitespace starts a comment, and a continuation line is a
    line of its value."""
    conf = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    conf.read_string((AGENT_DIR / "root.conf").read_text())
    return conf


ROOT_CONF = _root_conf()
LISTENER = ROOT_CONF["eventlistener:agent-exit"]


def test_the_server_isnt_started_again_and_a_listener_hears_it_exit_while_the_follower_and_the_proxy_restart():
    server = ROOT_CONF["program:agent"]

    assert (server["autorestart"], server["startretries"]) == ("false", "0")
    assert [ROOT_CONF[f"program:{name}"]["autorestart"] for name in ("clock", "egress")] == ["true", "true"]
    assert set(LISTENER["events"].split(",")) == {"PROCESS_STATE_EXITED", "PROCESS_STATE_FATAL"}
    assert "user" not in LISTENER and ROOT_CONF["supervisord"]["user"] == "root"


def _read(stream, size: int, timeout_s: float = 10) -> bytes:
    data, deadline = b"", time.monotonic() + timeout_s
    while len(data) < size:
        ready, _, _ = select.select([stream], [], [], max(0.0, deadline - time.monotonic()))
        assert ready, f"the listener said {data!r}, short of {size} bytes"
        chunk = os.read(stream.fileno(), size - len(data))
        assert chunk, f"the listener closed its stdout after {data!r}"
        data += chunk
    return data


def _event(name: str, process: str, from_state: str) -> bytes:
    payload = f"processname:{process} groupname:{process} from_state:{from_state} expected:0 pid:4242".encode()
    header = f"ver:3.0 server:supervisor serial:7 pool:agent-exit poolserial:7 eventname:{name} len:{len(payload)}\n"
    return header.encode() + payload


@pytest.mark.parametrize(("name", "from_state"),
                         [("PROCESS_STATE_EXITED", "RUNNING"), ("PROCESS_STATE_FATAL", "BACKOFF")])
def test_the_listener_stops_supervisord_when_the_server_exits_and_lets_the_followers_and_the_proxys_exits_pass(
        name, from_state):
    """The listener's command as supervisord runs it, under a stand-in for supervisord, its parent, that holds its
    stdin and stdout as supervisord does."""
    argv = shlex.split(LISTENER["command"] % {})
    assert argv[0] == "/usr/local/bin/python3"
    supervisord = subprocess.Popen(
        [sys.executable, "-c", "import subprocess, sys, time; subprocess.Popen(sys.argv[1:]); time.sleep(60)",
         sys.executable, *argv[1:]], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert _read(supervisord.stdout, 6) == b"READY\n"
        for other in (_event("PROCESS_STATE_EXITED", "clock", "RUNNING"),
                      _event("PROCESS_STATE_EXITED", "egress", "RUNNING"),
                      _event("PROCESS_STATE_FATAL", "clock", "BACKOFF")):
            supervisord.stdin.write(other)
            supervisord.stdin.flush()
            assert _read(supervisord.stdout, 17) == b"RESULT 2\nOKREADY\n"
        with pytest.raises(subprocess.TimeoutExpired):
            supervisord.wait(0.5)

        supervisord.stdin.write(_event(name, "agent", from_state))
        supervisord.stdin.flush()

        assert supervisord.wait(10) == -signal.SIGTERM
    finally:
        supervisord.kill()
        supervisord.wait()
        supervisord.stdin.close()
        supervisord.stdout.close()
