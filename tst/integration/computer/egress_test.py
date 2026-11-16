"""The ubuntu-24.04 base's egress proxy, on real time, which every session program goes out through: the session user
can neither read its CA's key nor stop, replace or retrust it, and a killed one comes back; it refuses every
destination that isn't global; with the universe 182 days back, Java's HttpClient, an official Node build's fetch and
curl reach a live site through it by the session's settings alone, where Java and Node, which check certificates
against the universe's time, don't without it; and fresh handshakes, made all through a run of week-long jumps with
the universe a month on, are none of them aborted. Needs Docker, the local registry on localhost:5000, and the
internet."""

import asyncio
import json
import subprocess
import time
import uuid

import pytest

from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from tst.integration.computer.conftest import (EGRESS, EGRESS_USER, FOLLOWER_BACK_S, UniverseClock, computer_container,
                                               docker, kill_program)

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

DAY = 86400
LIVE = "https://example.com/"
APP = "/opt/agentenv/apps/clients"
CLIENTS = {"java": f"java {APP}/Fetch.java {LIVE}", "node": f"node {APP}/fetch.mjs {LIVE}",
           "curl": f"curl -sS -o /dev/null -w '%{{http_code}}' {LIVE}"}
#: What a client that ignores the session's proxy settings would do.
NO_PROXY = "env -u HTTPS_PROXY -u https_proxy -u ALL_PROXY -u all_proxy -u JAVA_TOOL_OPTIONS -u NODE_USE_ENV_PROXY"
#: Private, local and special-purpose destinations, by address, each of which the proxy refuses.
REFUSED = ["10.0.0.1", "10.255.255.254", "172.16.0.1", "172.31.255.1", "192.168.1.1", "127.0.0.1", "127.1.2.3",
           "0.0.0.0", "169.254.169.254", "169.254.1.1", "100.64.0.1", "100.127.255.254", "192.0.2.1", "198.18.0.1",
           "224.0.0.1", "255.255.255.255", "::1", "::", "fd00::1", "fe80::1", "::ffff:10.0.0.1", "2001:db8::1"]
#: Asks the proxy to CONNECT to each address given, printing the status line it answers each with.
CONNECT_EACH = """import json, socket, sys
out = {}
for address in sys.argv[1:]:
    host = f"[{address}]" if ":" in address else address
    with socket.create_connection(("127.0.0.1", 3128), timeout=10) as s:
        s.sendall(f"CONNECT {host}:443 HTTP/1.1\\r\\nHost: {host}:443\\r\\n\\r\\n".encode())
        out[address] = s.recv(64).split(b"\\r\\n")[0].decode()
print(json.dumps(out))
"""
#: As the session user, each a change the egress proxy's user or root alone may make.
ATTEMPTS = {
    "read the CA's key": "cat /etc/agentenv/egress/ca.key",
    "list the CA's folder": "ls /etc/agentenv/egress",
    "kill the proxy": "kill -9 $(pgrep -u agentenv-egress -f 'egress serve')",
    "stop it through root's supervisord": "supervisorctl -c /etc/supervisor/root.conf stop egress",
    "replace the proxy": "cp /bin/true /usr/local/lib/agentenv-egress/egress",
    "add to the roots it verifies sites against": "echo x >> /usr/local/lib/agentenv-egress/roots.pem",
    "add to agentenv's trust files": "echo x >> /etc/agentenv/ca-bundle.pem",
    "replace the public CA": "echo x > /etc/agentenv/egress-ca.pem",
}


def _serving(name: str) -> str:
    """The egress proxy's pid once it answers on its port."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        up = subprocess.run(["docker", "exec", name, "python3", "-c",
                             "import socket; socket.create_connection(('127.0.0.1', 3128), timeout=1)"],
                            capture_output=True)
        pid = subprocess.run(["docker", "exec", name, "pgrep", "-u", EGRESS_USER, "-f", f"{EGRESS} serve"],
                             capture_output=True, text=True).stdout.strip()
        if up.returncode == 0 and pid:
            return pid
        time.sleep(0.5)
    raise AssertionError(subprocess.run(["docker", "logs", name], capture_output=True, text=True).stdout[-2000:])


def test_the_session_user_can_neither_read_its_key_nor_stop_replace_or_retrust_it_and_a_killed_one_comes_back(base):
    name = f"agentenv-test-egress-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, base)
    try:
        pid = _serving(name)
        attempts = {what: subprocess.run(["docker", "exec", "-u", "user", name, "sh", "-c", command],
                                         capture_output=True, text=True).returncode
                    for what, command in ATTEMPTS.items()}
        same = _serving(name)
        killed, back, took = kill_program(name, f"{EGRESS} serve", EGRESS_USER)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    assert [what for what, code in attempts.items() if code == 0] == [], attempts
    assert same == pid
    assert killed == pid and back not in ("-", killed) and took < FOLLOWER_BACK_S, (killed, back, took)


def test_every_destination_that_isnt_global_is_refused(base):
    name = f"agentenv-test-egress-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, base)
    try:
        _serving(name)
        answered = json.loads(subprocess.run(["docker", "exec", "-u", "user", name, "python3", "-c", CONNECT_EACH,
                                              *REFUSED], check=True, capture_output=True, text=True).stdout)
        log = subprocess.run(["docker", "exec", name, "cat", "/var/log/agentenv-root/egress.log"],
                             capture_output=True, text=True).stdout
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    assert len(REFUSED) == 22
    assert {address: status for address, status in answered.items() if "403" not in status} == {}, answered
    refused = [json.loads(line) for line in log.splitlines() if '"refused"' in line]
    assert len(refused) == 22 and all("isn't public" in row["error"] for row in refused), refused


async def _status(deployed, command: str) -> str:
    ran = await inside.run(deployed, ["sh", "-c", command], timeout_s=120)
    lines = ran["stdout"].strip().splitlines()
    return lines[-1] if ran["exit_code"] == 0 and lines else f"exit {ran['exit_code']}: {ran.get('stderr', '')[-300:]}"


def _egress_rows(deployed) -> list[dict]:
    log = subprocess.run(["docker", "exec", computer_container(deployed.sandbox_id), "cat",
                          "/var/log/agentenv-root/egress.log"], capture_output=True, text=True).stdout
    return [json.loads(line) for line in log.splitlines() if line.startswith("{")]


@pytest.mark.asyncio
async def test_java_node_and_curl_reach_a_live_site_through_it_with_the_universe_182_days_back(egress_desktop):
    with namespace_routing():
        deployed = await Env.get(egress_desktop.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            UniverseClock(deployed).set(time.time() - 182 * DAY, 1)
            await asyncio.sleep(2)  # each program maps the page the next time it reads the clock, within a second
            through = {client: await _status(deployed, command) for client, command in CLIENTS.items()}
            direct = {client: await _status(deployed, f"{NO_PROXY} {command}") for client, command in CLIENTS.items()
                      if client != "curl"}
            rows = _egress_rows(deployed)
        finally:
            report = await teardown_run(context)

    assert through == {"java": "200", "node": "200", "curl": "200"}, through
    # Java and Node check the site's certificate against the universe's time, so without the proxy they refuse it.
    assert all(not status.isdigit() for status in direct.values()), direct
    intercepted = [row for row in rows if row.get("host") == "example.com" and row.get("mode") == "intercept"]
    assert len(intercepted) >= 3 and all(row["verified"] for row in intercepted), rows[-6:]
    assert not report.failed


#: Fetches each URL given with a fresh Node process, one after another, until ~/.egress-stop appears, printing a line
#: per fetch: the status, or FAIL and the URL.
FETCH_LOOP = f"""while [ ! -e ~/.egress-stop ]; do
  for url in "$@"; do
    node {APP}/fetch.mjs "$url" 2>>~/.egress-errors || echo "FAIL $url"
  done
done > ~/.egress-loop
echo DONE >> ~/.egress-loop"""


@pytest.mark.asyncio
async def test_fresh_handshakes_through_week_jumps_with_the_universe_a_month_on_are_none_aborted(egress_desktop):
    sites = [LIVE, "https://www.wikipedia.org/"]
    with namespace_routing():
        deployed = await Env.get(egress_desktop.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            clock = UniverseClock(deployed)
            clock.set(time.time() + 30 * DAY, 1)
            await asyncio.sleep(2)
            await inside.run(deployed, ["sh", "-c", f"setsid sh -c '{FETCH_LOOP}' loop {' '.join(sites)} "
                                        "> /dev/null 2>&1 < /dev/null &"], timeout_s=30)
            for _ in range(12):
                await asyncio.sleep(1.5)
                clock.advance(7 * DAY)
            await asyncio.sleep(3)
            await inside.run(deployed, ["sh", "-c", "touch ~/.egress-stop"], timeout_s=30)
            loop = ""
            for _ in range(60):
                loop = (await inside.run(deployed, ["sh", "-c", "cat ~/.egress-loop"], timeout_s=30))["stdout"]
                if "DONE" in loop:
                    break
                await asyncio.sleep(1)
            errors = (await inside.run(deployed, ["sh", "-c", "cat ~/.egress-errors"], timeout_s=30))["stdout"]
            rows = _egress_rows(deployed)
        finally:
            report = await teardown_run(context)

    fetches = [line for line in loop.splitlines() if line != "DONE"]
    assert "DONE" in loop and len(fetches) >= 40, loop[-500:]
    assert [line for line in fetches if line != "200"] == [], errors[-1500:]
    modes = {row.get("mode") for row in rows if row.get("host") in ("example.com", "www.wikipedia.org")}
    assert "intercept" in modes, rows[-6:]
    assert not report.failed
