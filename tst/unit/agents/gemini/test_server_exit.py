"""When the image's A2A server exits, the container exits with it, as when the server was the container's process:
root.conf never starts it again, and its event listener, run here as supervisord runs it, stops supervisord, its parent,
on the server's EXITED or FATAL alone, while the follower and the proxy keep restarting in place."""

from __future__ import annotations

import configparser
import json
import shlex
import subprocess
import sys

import pytest

from tst.unit.agents.gemini.agent import AGENT_DIR

ACK = "RESULT 2\nOKREADY\n"

# The listener's parent, standing in for supervisord: SIGTERM blocked, so one the listener sends is pending by the time
# its acknowledgement, written after the signal, is read.
PARENT = """
import json, signal, subprocess, sys
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
listener = subprocess.Popen(json.loads(sys.argv[1]), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
ready, acks = listener.stdout.read(6).decode(), []
for name, payload in json.loads(sys.argv[2]):
    header = f"ver:3.0 server:supervisor serial:1 pool:agent-exit poolserial:1 eventname:{name} len:{len(payload)}\\n"
    listener.stdin.write((header + payload).encode())
    listener.stdin.flush()
    acks.append([listener.stdout.read(17).decode(), signal.SIGTERM in signal.sigpending()])
listener.stdin.close()
listener.wait()
print(json.dumps({"ready": ready, "acks": acks}))
"""


def _conf() -> configparser.RawConfigParser:
    """root.conf as supervisord reads it: ';' or '#' after whitespace starts a comment, and a continuation line is a
    line of its value."""
    conf = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    conf.read_string((AGENT_DIR / "root.conf").read_text())
    return conf


def _event(name: str, program: str, from_state: str) -> list[str]:
    extra = " expected:0 pid:4242" if name == "PROCESS_STATE_EXITED" else ""
    return [name, f"processname:{program} groupname:{program} from_state:{from_state}{extra}"]


def _listen(events: list[list[str]]) -> dict:
    command = shlex.split(_conf()["eventlistener:agent-exit"]["command"])
    assert command[0] == "/usr/local/bin/python3"
    argv = [sys.executable, "-c", PARENT, json.dumps([sys.executable, *command[1:]]), json.dumps(events)]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_server_is_never_started_again_and_the_follower_and_the_proxy_are():
    conf = _conf()

    assert (conf["program:agent"]["autorestart"], conf["program:agent"]["startretries"]) == ("false", "0")
    assert conf["program:clock"]["autorestart"] == conf["program:egress"]["autorestart"] == "true"


def test_the_listener_hears_the_server_exit_or_give_up_starting_and_is_ready_before_it_starts():
    conf = _conf()
    listener = conf["eventlistener:agent-exit"]

    assert set(listener["events"].split(",")) == {"PROCESS_STATE_EXITED", "PROCESS_STATE_FATAL"}
    assert "%" not in listener["command"]
    assert int(listener["priority"]) < int(conf["program:agent"]["priority"])


@pytest.mark.parametrize(("name", "from_state"),
                         [("PROCESS_STATE_EXITED", "RUNNING"), ("PROCESS_STATE_FATAL", "BACKOFF")])
def test_the_listener_stops_supervisord_when_the_server_exits_and_lets_the_followers_and_proxys_exits_pass(
        name, from_state):
    others = [_event("PROCESS_STATE_EXITED", "clock", "RUNNING"), _event("PROCESS_STATE_EXITED", "egress", "RUNNING"),
              _event("PROCESS_STATE_FATAL", "clock", "BACKOFF")]

    heard = _listen([*others, _event(name, "agent", from_state)])

    assert heard["ready"] == "READY\n"
    assert heard["acks"] == [[ACK, False]] * len(others) + [[ACK, True]]
