"""The shipped chromium alone behind its front door, in the chromium test bundle's composed image with the front door as
its command in place of the desktop's, as the browser provider will run it: headless, out only through a stand-in site
proxy; its card and browser/v1's record on the front door's port, with the control token, which no program it runs is
given, and no TCP port listening but the front door's and the proxy's; its killed app server started again, which
launches Chromium again on a new pipe; Chromium killed failing the env for good, with its signal, which a restarted app
server doesn't undo; and its killed site proxy started again. Needs Docker and the local registry on localhost:5000."""

import asyncio
import json
import platform
import shlex
import time
import uuid

import subprocess

import httpx
import pytest

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import front_door
from agentenv_protocol.computer.operations import COMPUTER_READY
from agentenv_protocol.types import CONTROL_TOKEN_ENV, WELL_KNOWN_PATH
from agentenv_protocol.website_browser import ALONE_ENV, BROWSER, SITE_PROXY_ENV

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
APP = "/opt/agentenv/apps/chromium"
PROXY_PORT = 3128
SITE = "http://site.test/"
TOKEN = uuid.uuid4().hex
BEARER = {"authorization": f"Bearer {TOKEN}"}
#: How long the browser may take to be ready, or to recover; how often it's asked meanwhile.
WAIT_S, POLL_S = 120.0, 1.0
# A stand-in for the site proxy: any page asked for through it is one that names its host.
SITE_PROXY = f"""import http.server, urllib.parse
class Proxy(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        host = urllib.parse.urlsplit(self.path).hostname or ""
        body = f"<title>proxied {{host}}</title>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
http.server.ThreadingHTTPServer(("127.0.0.1", {PROXY_PORT}), Proxy).serve_forever()
"""
# Every TCP port something in the container listens on.
LISTENING = """ports = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    for line in open(table).read().splitlines()[1:]:
        fields = line.split()
        if fields[3] == "0A":
            ports.add(int(fields[1].rsplit(":", 1)[1], 16))
print(sorted(ports))
"""


def docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=True, capture_output=True)


class Alone:
    """One container of the composed image running chromium alone behind its front door."""

    def __init__(self, image: str) -> None:
        conf = docker("run", "--rm", "--entrypoint", "cat", image,
                      "/etc/supervisor/conf.d/agentenv-app-chromium.conf").stdout.decode()
        served = next(line for line in conf.splitlines() if "serve_app" in line).rsplit(" ", 1)[1]
        settings = json.dumps({"chromium": {"start_urls": [SITE]}})
        self.name = f"front-door-{uuid.uuid4().hex[:8]}"
        docker("run", "-d", "--name", self.name, "-p", "127.0.0.1::8000", "-e", "DISPLAY=",
               "-e", f"{ALONE_ENV}=1", "-e", f"{SITE_PROXY_ENV}=http://127.0.0.1:{PROXY_PORT}",
               "-e", "ENVIRONMENT_NAME=chromium", "-e", f"{CONTROL_TOKEN_ENV}={TOKEN}",
               "-e", f"AGENTENV_APP_SETTINGS={settings}", image,
               f"{APP}/venv/bin/python", "-m", "agentenv_protocol.computer.front_door", APP, served,
               "--site-proxy", shlex.join(["python3", "-c", SITE_PROXY]))
        port = docker("port", self.name, "8000").stdout.decode().split()[0].rsplit(":", 1)[1]
        self.url = f"http://127.0.0.1:{port}"

    def run(self, *argv: str) -> str:
        return docker("exec", self.name, *argv).stdout.decode()

    def pids(self, pattern: str) -> list[int]:
        """The front door's programs whose command line has ``pattern``."""
        door = "$(pgrep -n -f '[a]gentenv_protocol.computer.front_door')"
        found = self.run("sh", "-c", f"pgrep -P {door} -f -- {shlex.quote(pattern)} || true")
        return [int(pid) for pid in found.split()]

    def launches(self) -> list[dict]:
        log = self.run("sh", "-c", f"cat {APP}/.agentenv/launches 2>/dev/null || true")
        return [json.loads(line) for line in log.splitlines() if line]

    async def get(self, path: str, **kwargs) -> httpx.Response:
        async with httpx.AsyncClient(base_url=self.url, timeout=30) as http:
            return await http.get(path, **kwargs)

    async def status(self) -> dict:
        return (await self.get(front_door.STATUS_PATH, headers=BEARER)).json()

    async def record(self) -> dict:
        card = (await self.get(WELL_KNOWN_PATH)).json()
        path = protocol_v1.find_extension_method(card, BROWSER, "record")["endpoint"]
        return (await self.get(path, headers=BEARER)).json()

    def logs(self) -> str:
        logged = docker("logs", self.name)
        return (logged.stdout + logged.stderr).decode()[-4000:]

    async def until(self, holds, what: str):
        """``holds``' answer once it's truthy, within ``WAIT_S``, while the container runs."""
        last, deadline = None, time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            if docker("inspect", "-f", "{{.State.Running}}", self.name).stdout.decode().strip() != "true":
                raise AssertionError(f"the container ended before {what}; its logs:\n{self.logs()}")
            try:
                last = await holds()
                if last:
                    return last
            except (httpx.HTTPError, ValueError, KeyError):
                pass
            await asyncio.sleep(POLL_S)
        raise AssertionError(f"{what} within {WAIT_S:g} s; last {last!r}; its logs:\n{self.logs()}")

    async def ready(self) -> dict:
        async def ready():
            status = await self.status()
            return status if status["ready"] else None
        return await self.until(ready, "ready")

    async def on_the_site(self) -> dict:
        async def loaded():
            record = await self.record()
            return record if [tab["title"] for tab in record["tabs"]] == ["proxied site.test"] else None
        return await self.until(loaded, "its start tab loaded through the site proxy")

    def remove(self) -> None:
        docker("rm", "-f", self.name)


@pytest.fixture
def alone(chromium_desktop):
    image = chromium_desktop.metadata["composed"]["container"][ARCH]["image"]
    started = []

    def start() -> Alone:
        started.append(Alone(image))
        return started[-1]

    yield start
    for container in started:
        container.remove()


def _browser(container: Alone) -> int:
    return [entry for entry in container.launches() if entry["event"] == "launch"][-1]["pid"]


@pytest.mark.asyncio
async def test_alone_it_serves_its_card_status_and_record_on_its_port_with_the_token_and_nothing_else_listens(alone):
    container = alone()
    status = await container.ready()
    record = await container.on_the_site()
    card = (await container.get(WELL_KNOWN_PATH)).json()
    record_path = protocol_v1.find_extension_method(card, BROWSER, "record")["endpoint"]
    refused = [(await container.get(path)).status_code for path in (front_door.STATUS_PATH, record_path)]
    wrong = (await container.get(record_path, headers={"authorization": "Bearer nope"})).status_code
    listening = json.loads(container.run("python3", "-c", LISTENING))
    (server,) = container.pids("serve_app")
    environments = [container.run("sh", "-c", f"tr '\\0' '\\n' < /proc/{pid}/environ")
                    for pid in (server, _browser(container))]
    flags = container.run("sh", "-c", f"tr '\\0' ' ' < /proc/{_browser(container)}/cmdline")

    assert card["name"] == "chromium" and card["additionalInterfaces"] == []
    assert protocol_v1.find_extension_method(card, COMPUTER_READY, "status")["endpoint"] == front_door.STATUS_PATH
    assert [check["name"] for check in status["checks"]] == ["app chromium", "site proxy"]
    assert [(tab["url"], tab["active"]) for tab in record["tabs"]] == [(SITE, True)]
    assert [(entry["event"], entry["cause"]) for entry in record["launches"]] == [("launch", "boot")]
    assert refused == [401, 401] and wrong == 401
    assert listening == sorted([8000, PROXY_PORT])
    assert all(f"{CONTROL_TOKEN_ENV}=" not in environment for environment in environments)
    assert "--headless=new" in flags and "--remote-debugging-pipe" in flags
    assert f"--proxy-server=http://127.0.0.1:{PROXY_PORT}" in flags


@pytest.mark.asyncio
async def test_its_killed_app_server_starts_again_and_launches_chromium_again_on_a_new_pipe(alone):
    container = alone()
    await container.ready()
    await container.on_the_site()
    first = _browser(container)
    (server,) = container.pids("serve_app")
    container.run("kill", "-9", str(server))

    async def relaunched():
        return container.launches()[-1].get("cause") == "server_restart" and (await container.status())["ready"]

    await container.until(relaunched, "Chromium launched again and ready")
    record = await container.on_the_site()
    (restarted,) = container.pids("serve_app")

    assert restarted != server and _browser(container) != first
    assert [entry["event"] for entry in container.launches()] in (["launch", "launch"], ["launch", "stop", "launch"])
    assert "exit" not in [entry["event"] for entry in record["launches"]]


@pytest.mark.asyncio
async def test_chromium_killed_fails_the_env_for_good_with_its_signal_even_once_its_server_restarts(alone):
    container = alone()
    await container.ready()
    container.run("kill", "-9", str(_browser(container)))

    async def failed():
        status = await container.status()
        return status if "failed" in status else None

    status = await container.until(failed, "failed")
    record = await container.record()
    (server,) = container.pids("serve_app")
    container.run("kill", "-9", str(server))

    async def relaunched():
        return container.launches()[-1].get("cause") == "server_restart"

    await container.until(relaunched, "Chromium launched again by the restarted server")
    after = await container.status()

    assert status["ready"] is False and "by signal SIGKILL" in status["failed"]
    assert status["checks"][0]["ok"] is False and status["checks"][0]["detail"] == status["failed"]
    assert [entry["event"] for entry in record["launches"]] == ["launch", "exit"]
    assert record["launches"][1]["signal"] == "SIGKILL"
    assert [tab["closed_by"] for tab in record["closed_tabs"]] == ["exit"]
    assert after["ready"] is False and after["failed"] == status["failed"]


@pytest.mark.asyncio
async def test_its_killed_site_proxy_starts_again(alone):
    container = alone()
    await container.ready()
    (proxy,) = container.pids("ThreadingHTTPServer")
    container.run("kill", "-9", str(proxy))

    async def restarted():
        return [pid for pid in container.pids("ThreadingHTTPServer") if pid != proxy]

    (again,) = await container.until(restarted, "the site proxy started again")
    status = await container.ready()
    fetched = container.run("python3", "-c", "import urllib.request; print(urllib.request.build_opener("
                            f"urllib.request.ProxyHandler({{'http': 'http://127.0.0.1:{PROXY_PORT}'}}))"
                            f".open('{SITE}').read().decode())")

    assert again != proxy and status["checks"][1] == {"name": "site proxy", "ok": True, "detail": "running"}
    assert fetched.strip() == "<title>proxied site.test</title>"
