"""The A2A server is the container, as it was before the image ran supervisord: root.conf restarts the clock follower
and the egress proxy in place, never the server, and its event listener stops supervisord, so the container ends, once
the server exits or fails to start. A restarted server would have lost its MCP servers, peers and tasks while the dead
one's CLI went on in the workspace."""

from __future__ import annotations

import configparser
import json
import shlex
import subprocess
import sys

from tst.unit.agents.letta_code.agent import AGENT_DIR

EVENTS = {"PROCESS_STATE_EXITED", "PROCESS_STATE_FATAL"}


def _conf() -> configparser.RawConfigParser:
    """root.conf as supervisord reads it: a ';' or '#' after whitespace starts a comment."""
    conf = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    conf.read_string((AGENT_DIR / "root.conf").read_text())
    return conf


def _listener() -> configparser.SectionProxy:
    listeners = [name for name in _conf().sections() if name.startswith("eventlistener:")]
    assert len(listeners) == 1, listeners
    return _conf()[listeners[0]]


def test_the_server_is_never_restarted_and_the_follower_and_the_proxy_are():
    conf = _conf()

    assert conf["program:agent"]["autorestart"] == "false"
    assert conf["program:clock"]["autorestart"] == conf["program:egress"]["autorestart"] == "true"


def test_one_listener_hears_every_way_a_program_ends_on_its_own_as_root_which_may_stop_supervisord():
    """A stop, the container's or supervisorctl's, ends a program as STOPPED, which it doesn't hear."""
    listener = _listener()

    assert set(listener["events"].split(",")) == EVENTS
    assert "user" not in listener


# Plays supervisord to the listener, as its parent, so the listener's signal comes here: each event as the eventlistener
# protocol sends it, then whether the listener sent SIGTERM before it answered.
SUPERVISORD = r"""
import json, signal, subprocess, sys
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
listener = subprocess.Popen(sys.argv[1:], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
stopped = []
for name, event, state in json.loads(sys.stdin.read()):
    assert listener.stdout.readline() == b"READY\n"
    payload = f"processname:{name} groupname:{name} {state}".encode()
    header = f"ver:3.0 server:supervisor serial:1 pool:agent-exit poolserial:1 eventname:{event} len:{len(payload)}\n"
    listener.stdin.write(header.encode() + payload)
    listener.stdin.flush()
    assert listener.stdout.read(11) == b"RESULT 2\nOK"
    stopped.append(signal.SIGTERM in signal.sigpending())
    if stopped[-1]:
        signal.sigwait({signal.SIGTERM})
listener.kill()
print(json.dumps(stopped))
"""


def test_the_listener_stops_supervisord_when_the_server_exits_or_fails_to_start_and_for_nothing_else():
    exited, fatal = "from_state:RUNNING expected:0 pid:7", "from_state:BACKOFF"
    events = [("clock", "PROCESS_STATE_EXITED", exited), ("egress", "PROCESS_STATE_FATAL", fatal),
              ("agent", "PROCESS_STATE_EXITED", exited), ("clock", "PROCESS_STATE_FATAL", fatal),
              ("agent", "PROCESS_STATE_FATAL", fatal)]
    # supervisord expands '%(name)s' in a command, so a bare '%' would break it.
    command = shlex.split(_listener()["command"] % {})
    assert command[0] == "/usr/local/bin/python3"

    result = subprocess.run([sys.executable, "-c", SUPERVISORD, sys.executable, *command[1:]],
                            input=json.dumps(events), capture_output=True, text=True, timeout=60, check=True)

    assert json.loads(result.stdout) == [False, False, True, False, True]
