"""Root's supervisord never restarts the A2A server, which holds what the run gave it in memory: once the server exits,
or fails to start, supervisord stops and the container exits with it, as it did when the server was the container's
process. The follower and the egress proxy keep their restarts."""

from __future__ import annotations

import configparser
import json
import shlex
import subprocess
import sys

from tst.unit.agents.pi.agent import AGENT_DIR

CONF = configparser.RawConfigParser()
CONF.read_string((AGENT_DIR / "root.conf").read_text())
LISTENERS = [section for section in CONF.sections() if section.startswith("eventlistener:")]

# Plays supervisord to the listener: SIGTERM blocked, so a signal the listener sends its parent waits, pending, to be
# seen after each event's result.
PARENT = """
import json, signal, subprocess, sys
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
listener = subprocess.Popen(json.loads(sys.argv[1]), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
stopped = []
for event, name in json.loads(sys.argv[2]):
    assert listener.stdout.readline() == "READY\\n"
    payload = f"processname:{name} groupname:{name} from_state:RUNNING expected:0 pid:42"
    listener.stdin.write(f"ver:3.0 server:supervisor serial:1 pool:agent-exit poolserial:1 eventname:{event} "
                         f"len:{len(payload)}\\n{payload}")
    listener.stdin.flush()
    assert listener.stdout.readline() == "RESULT 2\\n" and listener.stdout.read(2) == "OK"
    stopped.append(signal.SIGTERM in signal.sigpending())
    if stopped[-1]:
        signal.sigwait({signal.SIGTERM})
listener.kill()
print(json.dumps(stopped))
"""


def test_the_server_is_never_restarted_and_the_follower_and_the_proxy_are():
    assert (CONF.get("program:agent", "autorestart"), CONF.get("program:agent", "startretries")) == ("false", "0")
    assert CONF.get("program:clock", "autorestart") == CONF.get("program:egress", "autorestart") == "true"


def test_one_listener_hears_the_servers_exit_and_its_failure_to_start_as_root():
    assert len(LISTENERS) == 1
    events = {event.strip() for event in CONF.get(LISTENERS[0], "events").split(",")}
    assert events == {"PROCESS_STATE_EXITED", "PROCESS_STATE_FATAL"}
    assert CONF.get(LISTENERS[0], "user") == "root"


def test_the_listener_stops_supervisord_on_the_servers_exit_alone():
    python, *args = shlex.split(CONF.get(LISTENERS[0], "command"))
    events = [["PROCESS_STATE_EXITED", "clock"], ["PROCESS_STATE_FATAL", "egress"], ["PROCESS_STATE_EXITED", "agent"],
              ["PROCESS_STATE_EXITED", "clock"], ["PROCESS_STATE_FATAL", "agent"]]

    out = subprocess.run([sys.executable, "-c", PARENT, json.dumps([sys.executable, *args]), json.dumps(events)],
                         capture_output=True, text=True, timeout=30, check=True).stdout

    assert python == "/usr/bin/python3"
    assert json.loads(out) == [False, False, True, False, True]
