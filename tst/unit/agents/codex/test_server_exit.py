"""The A2A server's exit ends the image's container, as it did when the server was the image's only process: root's
supervisord never starts it again, and its agent-exit listener has supervisord shut down when it exits or can't start,
while the clock follower and the egress proxy keep restarting in place. root.conf is read as supervisord reads it, and
the listener is run as written there, under a stand-in supervisord that speaks its protocol."""

from __future__ import annotations

import configparser
import json
import shlex
import subprocess
import sys

import pytest

from tst.unit.agents.codex.agent import AGENT_DIR

CONF = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
CONF.read_string((AGENT_DIR / "root.conf").read_text())
LISTENER = "eventlistener:agent-exit"

# Sends the listener each event as supervisord does, waiting for its READY and its RESULT, and after each says whether
# the listener had sent this process, its parent, a SIGTERM; blocked, so it stays pending to be seen.
SUPERVISORD = """
import json, signal, subprocess, sys
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
listener = subprocess.Popen(json.loads(sys.argv[2]), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
terminated = []
for serial, (name, payload) in enumerate(json.loads(sys.argv[1]), 1):
    assert listener.stdout.readline() == b"READY\\n"
    header = f"ver:3.0 server:supervisor serial:{serial} pool:agent-exit poolserial:{serial} eventname:{name} len:{len(payload)}\\n"
    listener.stdin.write(header.encode() + payload.encode())
    listener.stdin.flush()
    assert listener.stdout.readline() == b"RESULT 2\\n" and listener.stdout.read(2) == b"OK"
    terminated.append(signal.SIGTERM in signal.sigpending())
listener.kill()
print(json.dumps(terminated))
"""


def _exited(name: str) -> tuple[str, str]:
    return "PROCESS_STATE_EXITED", f"processname:{name} groupname:{name} from_state:RUNNING expected:0 pid:4242"


def _fatal(name: str) -> tuple[str, str]:
    return "PROCESS_STATE_FATAL", f"processname:{name} groupname:{name} from_state:BACKOFF"


def _terminated(events: list[tuple[str, str]]) -> list[bool]:
    _, *args = shlex.split(CONF.get(LISTENER, "command"))
    done = subprocess.run([sys.executable, "-c", SUPERVISORD, json.dumps(events), json.dumps([sys.executable, *args])],
                          capture_output=True, timeout=30, check=True)
    return json.loads(done.stdout)


def test_the_server_is_never_started_again():
    assert CONF.get("program:agent", "autorestart") == "false"
    assert CONF.get("program:agent", "startretries") == "0"


def test_the_follower_and_the_proxy_restart_in_place():
    assert CONF.get("program:clock", "autorestart") == CONF.get("program:egress", "autorestart") == "true"


def test_the_listener_hears_the_servers_exit_and_failed_start_as_root():
    assert set(CONF.get(LISTENER, "events").split(",")) == {"PROCESS_STATE_EXITED", "PROCESS_STATE_FATAL"}
    # supervisord runs as root: a listener as another user couldn't signal it.
    assert not CONF.has_option(LISTENER, "user")


@pytest.mark.parametrize("event", [_exited, _fatal])
def test_the_servers_exit_shuts_supervisord_down(event):
    assert _terminated([event("agent")]) == [True]


def test_the_followers_or_the_proxys_exit_leaves_it_running():
    assert _terminated([_exited("clock"), _fatal("egress"), _exited("agent-exit"), _exited("agent")]) == [
        False, False, False, True]
