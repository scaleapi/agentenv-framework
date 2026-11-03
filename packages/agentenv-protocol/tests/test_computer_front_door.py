"""A computer app alone behind its front door, run as its container runs it, with a fake browser that answers CDP on its
pipe: the app's card with computer-ready/v1 and no interface for agents; every route but the card taking the control
token, which no program it runs is given; requests sent on to the app's server, a large answer streamed whole; a killed
app server started again, which launches the browser again; a browser that ends on its own failing the env for good,
with its signal; a killed site proxy started again; an app server that can't start started again later and later,
saying why; requests while the app's server restarts answering or failing, never hanging; stopping it stopping
everything it runs; and its needing only the app extra, as the app's venv has."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("uvicorn")

import httpx  # noqa: E402

from agentenv_protocol import client  # noqa: E402
from agentenv_protocol.computer import front_door  # noqa: E402
from agentenv_protocol.computer.operations import COMPUTER_READY  # noqa: E402
from agentenv_protocol.types import CONTROL_TOKEN_ENV, WELL_KNOWN_PATH  # noqa: E402
from agentenv_protocol.website_browser import ALONE_ENV, BROWSER, SITE_PROXY_ENV  # noqa: E402
from test_website_browser import FAKE_BROWSER  # noqa: E402

TOKEN = "front-door-token"
BEARER = {"authorization": f"Bearer {TOKEN}"}
#: How long it may take to start, or to recover; how often it's asked meanwhile.
WAIT_S, POLL_S = 30.0, 0.2
#: How large an answer streams through whole.
LARGE_BYTES = 20_000_000
SLEEPER = f"{sys.executable} -c 'import time; time.sleep(600)'"
# The app, as its folder holds it: a website browser whose launch runs the fake browser, on Linux, its launches marked
# in a folder of the test's own, with an extension that says whether its server was given the control token and one
# that answers with a large body.
APP = f"""import os, sys
from pathlib import Path
from agentenv_protocol.agent_env_environment import extension
from agentenv_protocol.computer import Launch, Ready, app, app_runtime, linux
from agentenv_protocol.computer.platform import Platform
from agentenv_protocol.website_browser import WebsiteBrowserEnvironment

app.detect = lambda: Platform(os="linux")
app_runtime.LAUNCHED = Path(os.environ["TEST_LAUNCHED"])


@linux(launch=Launch(command=[sys.executable, str(Path(__file__).with_name("fake_browser.py"))],
                     ready=Ready(command=["true"])))
class Browser(WebsiteBrowserEnvironment):
    @extension("urn:test:probe/v1", op="token", method="GET")
    async def token(self) -> dict:
        return {{"given": "{CONTROL_TOKEN_ENV}" in os.environ}}

    @extension("urn:test:probe/v1", op="large", method="GET")
    async def large(self) -> dict:
        return {{"data": "x" * {LARGE_BYTES}}}
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Door:
    """A front door running as its container's command, on a port of its own, with the app in a short folder, since a
    Unix socket's path is short."""

    def __init__(self, served: str = "browser.py:Browser", site_proxy: str | None = None) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="fd-", dir="/tmp"))
        (self.folder / "browser.py").write_text(APP)
        (self.folder / "fake_browser.py").write_text(FAKE_BROWSER)
        (self.folder / "home").mkdir()
        (self.folder / ".agentenv").mkdir()
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = {key: value for key, value in os.environ.items() if key != "DISPLAY"}
        env.update({"HOME": str(self.folder / "home"), "ENVIRONMENT_NAME": "browser", ALONE_ENV: "1",
                    SITE_PROXY_ENV: "http://127.0.0.1:9", "TEST_LAUNCHED": str(self.folder / "launched"),
                    CONTROL_TOKEN_ENV: TOKEN})
        argv = [sys.executable, "-m", "agentenv_protocol.computer.front_door", str(self.folder), served,
                "--port", str(self.port), *(["--site-proxy", site_proxy] if site_proxy else [])]
        self.log = (self.folder / "front-door.log").open("w")
        self.process = subprocess.Popen(argv, env=env, stdout=self.log, stderr=subprocess.STDOUT)

    def children(self, matching: str) -> list[int]:
        found = subprocess.run(["pgrep", "-P", str(self.process.pid), "-f", matching], capture_output=True, text=True)
        return [int(pid) for pid in found.stdout.split()]

    def launches(self) -> list[dict]:
        try:
            lines = (self.folder / ".agentenv" / "launches").read_text().splitlines()
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in lines if line]

    def logged(self) -> str:
        return (self.folder / "front-door.log").read_text()

    async def status(self) -> dict:
        async with httpx.AsyncClient(base_url=self.url, timeout=10) as http:
            return (await http.get(front_door.STATUS_PATH, headers=BEARER)).json()

    async def until(self, holds, what: str):
        """``holds``' answer once it's truthy, within ``WAIT_S``."""
        last = None
        deadline = time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            with contextlib.suppress(httpx.HTTPError, ValueError, OSError):
                last = await holds()
                if last:
                    return last
            await asyncio.sleep(POLL_S)
        raise AssertionError(f"{what} within {WAIT_S:g} s; last {last!r}; its log:\n{self.logged()}")

    async def ready(self) -> dict:
        return await self.until(lambda: self._ready(), "ready")

    async def _ready(self):
        status = await self.status()
        return status if status["ready"] else None

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(15)
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
        for entry in self.launches():
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(entry["pid"], signal.SIGKILL)
        self.log.close()
        shutil.rmtree(self.folder, ignore_errors=True)


@pytest.fixture
def doors():
    opened: list[Door] = []

    def open_door(**kwargs) -> Door:
        door = Door(**kwargs)
        opened.append(door)
        return door

    yield open_door
    for door in opened:
        door.close()


def _browser(door: Door) -> int:
    """The pid of the browser the app's server launched last."""
    return [entry for entry in door.launches() if entry["event"] == "launch"][-1]["pid"]


@pytest.mark.asyncio
async def test_its_card_is_the_apps_with_computer_ready_and_no_interface_for_agents_and_only_the_card_is_open(doors):
    door = doors(site_proxy=SLEEPER)
    status = await door.ready()
    async with httpx.AsyncClient(base_url=door.url, timeout=10) as http:
        card = (await http.get(WELL_KNOWN_PATH)).json()
        ready = client.find_extension_method(card, "urn:agentenv:computer-app/v1", "ready")["endpoint"]
        unguarded = [(await http.get(path)).status_code for path in (front_door.STATUS_PATH, ready)]
        wrong = (await http.get(ready, headers={"authorization": "Bearer nope"})).status_code
        app_ready = (await http.get(ready, headers=BEARER)).json()
        token = (await http.get(client.find_extension_method(card, "urn:test:probe/v1", "token")["endpoint"],
                                headers=BEARER)).json()

    assert card["name"] == "browser" and card["additionalInterfaces"] == []
    assert client.find_extension(card, BROWSER) is not None
    assert client.find_extension_method(card, COMPUTER_READY, "status") == {
        "method": "GET", "endpoint": front_door.STATUS_PATH, "request": {"type": "object", "properties": {}}}
    assert status == {"ready": True, "checks": [{"name": "app browser", "ok": True, "detail": "ready"},
                                                {"name": "site proxy", "ok": True, "detail": "running"}]}
    assert unguarded == [401, 401] and wrong == 401
    assert app_ready["ready"] is True and token == {"given": False}
    assert [entry["event"] for entry in door.launches()] == ["launch"]


@pytest.mark.asyncio
async def test_a_large_answer_streams_through_whole(doors):
    door = doors()
    await door.ready()
    async with httpx.AsyncClient(base_url=door.url, timeout=60) as http:
        card = (await http.get(WELL_KNOWN_PATH)).json()
        answer = await http.get(client.find_extension_method(card, "urn:test:probe/v1", "large")["endpoint"],
                                headers=BEARER)

    assert answer.status_code == 200 and len(answer.json()["data"]) == LARGE_BYTES


@pytest.mark.asyncio
async def test_a_killed_app_server_starts_again_and_launches_the_browser_again_on_a_new_pipe(doors):
    door = doors()
    await door.ready()
    first_browser = _browser(door)
    (server,) = door.children("serve_app")
    os.kill(server, signal.SIGKILL)

    async def relaunched():
        return door.launches()[-1].get("cause") == "server_restart" and (await door.status())["ready"]

    await door.until(relaunched, "the browser launched again and ready")
    (restarted,) = door.children("serve_app")

    # The browser ends with its pipe, so the server finds nothing to stop, and no one logs it as its own exit.
    assert restarted != server and [entry["event"] for entry in door.launches()] in (
        ["launch", "launch"], ["launch", "stop", "launch"])
    assert door.launches()[0]["cause"] == "boot" and _browser(door) != first_browser
    assert "the app's server ended" in door.logged() and "signal SIGKILL" in door.logged()


@pytest.mark.asyncio
async def test_a_browser_that_ends_on_its_own_fails_the_env_for_good_with_its_signal(doors):
    door = doors()
    await door.ready()
    os.killpg(_browser(door), signal.SIGKILL)

    async def failed():
        status = await door.status()
        return status if "failed" in status else None

    status = await door.until(failed, "failed")
    (server,) = door.children("serve_app")
    os.kill(server, signal.SIGKILL)

    async def relaunched():
        return [entry["event"] for entry in door.launches()][-1] == "launch" and len(door.launches()) >= 3

    await door.until(relaunched, "the browser launched again by the restarted server")
    await asyncio.sleep(1)
    after = await door.status()

    assert status["ready"] is False and "by signal SIGKILL" in status["failed"]
    assert "an infrastructure failure" in status["failed"]
    assert status["checks"][0] == {"name": "app browser", "ok": False, "detail": status["failed"]}
    assert [entry["event"] for entry in door.launches()][:2] == ["launch", "exit"]
    assert door.launches()[1]["signal"] == "SIGKILL"
    assert after["ready"] is False and after["failed"] == status["failed"]


@pytest.mark.asyncio
async def test_a_killed_site_proxy_starts_again(doors):
    door = doors(site_proxy=SLEEPER)
    await door.ready()
    (proxy,) = door.children("time.sleep")
    os.kill(proxy, signal.SIGKILL)

    async def restarted():
        return [pid for pid in door.children("time.sleep") if pid != proxy]

    (again,) = await door.until(restarted, "the site proxy started again")
    status = await door.ready()

    assert again != proxy and status["checks"][1] == {"name": "site proxy", "ok": True, "detail": "running"}
    assert "the site proxy ended" in door.logged()


@pytest.mark.asyncio
async def test_an_app_server_that_cant_start_is_started_again_later_and_later_and_says_why(doors):
    door = doors(served="browser.py:Missing")

    async def said_why():
        status = await door.status()
        return status if "failed to start" in status["checks"][0]["detail"] else None

    status = await door.until(said_why, "the server's failure")
    await door.until(lambda: asyncio.sleep(0, "starting it again in 2 s" in door.logged()), "a 2 s delay")

    assert status["ready"] is False and "Missing" in status["checks"][0]["detail"]
    assert "failed" not in status
    delays = [line.rsplit(" in ", 1)[1] for line in door.logged().splitlines() if "starting it again in" in line]
    assert delays[:3] == ["0.5 s", "1 s", "2 s"]


@pytest.mark.asyncio
async def test_requests_while_its_app_server_restarts_answer_or_fail_and_never_hang(doors):
    door = doors()
    await door.ready()
    (server,) = door.children("serve_app")
    async with httpx.AsyncClient(base_url=door.url, timeout=WAIT_S) as http:
        card = (await http.get(WELL_KNOWN_PATH)).json()
        ready = client.find_extension_method(card, "urn:agentenv:computer-app/v1", "ready")["endpoint"]
        os.kill(server, signal.SIGKILL)
        answers = await asyncio.gather(*(http.get(ready, headers=BEARER) for _ in range(30)),
                                       *(http.get(WELL_KNOWN_PATH) for _ in range(10)))
    await door.ready()

    assert {answer.status_code for answer in answers} <= {200, 502}
    assert all(answer.json()["name"] == "browser" for answer in answers[30:])


@pytest.mark.asyncio
async def test_stopping_it_stops_every_program_it_runs(doors):
    door = doors(site_proxy=SLEEPER)
    await door.ready()
    programs = door.children("serve_app") + door.children("time.sleep")
    door.process.send_signal(signal.SIGTERM)
    code = await asyncio.to_thread(door.process.wait, WAIT_S)

    def gone(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False

    # uvicorn ends with the signal it shut down on, once it has.
    assert code in (0, -signal.SIGTERM) and len(programs) == 2 and all(gone(pid) for pid in programs)


def test_it_needs_only_the_app_extra():
    blocked = """
import importlib.abc, sys
class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("av", "PIL"):
            raise ImportError(name)
sys.meta_path.insert(0, Blocked())
import agentenv_protocol.computer.front_door
"""
    done = subprocess.run([sys.executable, "-c", blocked], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr


def test_it_sends_on_no_hop_or_token_header():
    headers = [("authorization", "Bearer x"), ("Host", "a"), ("content-type", "application/json"),
               ("Transfer-Encoding", "chunked"), ("x-forwarded-host", "b"), ("Connection", "close")]

    assert front_door._forwarded(headers) == [("content-type", "application/json"), ("x-forwarded-host", "b")]
