"""The ubuntu-24.04 base's clock follower, run as it runs on a computer, against a stand-in for the gateway's watch that
holds a watch while the version it names is current, as the gateway does: it follows nothing until it has a config,
then writes each state the gateway answers to the clock page and acknowledges it with its next watch; it follows a
changed config at once, during a held watch too; and one it can't reach is named in its status, which holds no token."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

import agent_env
from agentenv_protocol.computer import clock as protocol_clock

CLOCK = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04" / "clock"
FOLLOWER = CLOCK / "clock-follower"



@pytest.fixture(autouse=True)
def _disable_network():
    """The loopback stand-in for the gateway is the point here, so the unit tier's socket block is lifted."""
    yield


def _clockpage(page: Path):
    spec = importlib.util.spec_from_file_location("clockpage_under_test", CLOCK / "clockpage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PAGE = str(page)
    return module


class Gateway:
    """A watch that holds while ``after`` is the current version (``hold`` seconds), answers the clock's state with
    its receive and send stamps, and records each watch's version, follower and token."""

    def __init__(self, hold: float = 5.0):
        self.hold, self.watches = hold, []
        self.state = {"armed": False, "version": "boot:0"}
        self.changed = threading.Condition()
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                received = time.time()
                query = {key: values[0] for key, values in parse_qs(urlsplit(self.path).query).items()}
                gateway.watches.append((query.get("after", ""), query.get("follower"),
                                        self.headers.get("Authorization")))
                with gateway.changed:
                    if query.get("after", "") == gateway.state["version"]:
                        gateway.changed.wait(gateway.hold)
                    body = json.dumps({**gateway.state, "received": received, "sent": time.time()}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/clock/watch"

    def set(self, version: str, universe: datetime, rate: float) -> None:
        with self.changed:
            self.state = {"armed": True, "version": version, "virtual_time": universe.isoformat(),
                          "virtual_seconds_per_real_second": rate}
            self.changed.notify_all()

    def close(self) -> None:
        with self.changed:
            self.changed.notify_all()
        self.server.shutdown()


@pytest.fixture
def follower(tmp_path):
    """The follower started with its config, status, page and file under ``tmp_path``; yields a namespace with
    ``configure(config)``, ``status()`` and ``universe()`` (the page's time now)."""
    paths = {"AGENTENV_CLOCK_CONFIG": tmp_path / "follower.json", "AGENTENV_CLOCK_STATUS": tmp_path / "status.json",
             "AGENTENV_CLOCK_PAGE": tmp_path / "page", "AGENTENV_CLOCK_FILE": tmp_path / "clock"}
    process = subprocess.Popen([sys.executable, "-I", str(FOLLOWER)], env={**os.environ, **{k: str(v) for k, v in paths.items()}},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    page = _clockpage(paths["AGENTENV_CLOCK_PAGE"])

    class Running:
        def configure(self, config: dict) -> None:
            staged = tmp_path / "follower.json.staged"
            staged.write_text(json.dumps(config))
            staged.replace(paths["AGENTENV_CLOCK_CONFIG"])

        def status(self) -> dict:
            try:
                return json.loads(paths["AGENTENV_CLOCK_STATUS"].read_text())
            except (OSError, ValueError):
                return {}

        def universe(self) -> float:
            with open(page.PAGE, "rb") as f:
                fields = page.LAYOUT.unpack(f.read(page.LAYOUT.size))
            _, _, _, universe_ns, anchor_ns, rate, armed, *_ = fields
            return page._universe_at(armed, universe_ns, anchor_ns, rate, time.time_ns()) / 1e9

        def until(self, check, within: float = 5.0):
            deadline = time.monotonic() + within
            while time.monotonic() < deadline:
                if result := check():
                    return result
                time.sleep(0.02)
            raise AssertionError(f"not within {within} s; status {self.status()}; process alive: {process.poll() is None}")

    yield Running()
    process.kill()
    process.wait()


def test_the_followers_paths_are_the_ones_core_and_the_driver_use():
    source = FOLLOWER.read_text()
    assert f'"{protocol_clock.CONFIG}"' in source and f'"{protocol_clock.STATUS}"' in source


def test_it_follows_the_gateway_its_config_names_and_acknowledges_each_version_with_its_next_watch(follower):
    gateway = Gateway()
    try:
        assert follower.until(lambda: follower.status().get("waiting"))
        gateway.set("boot:1", datetime(2020, 1, 1, tzinfo=timezone.utc), 1.0)
        follower.configure({"watch_url": gateway.url, "name": "desktop", "token": "read-token"})

        follower.until(lambda: follower.status().get("version") == "boot:1")
        assert abs(follower.universe() - datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()) < 2
        follower.until(lambda: ("boot:1", "desktop", "Bearer read-token") in gateway.watches)

        frozen = datetime(2021, 6, 1, 12, tzinfo=timezone.utc)
        gateway.set("boot:2", frozen, 0.0)
        follower.until(lambda: follower.status().get("version") == "boot:2", within=2)
        assert follower.universe() == frozen.timestamp()
        follower.until(lambda: ("boot:2", "desktop", "Bearer read-token") in gateway.watches)
        assert "read-token" not in json.dumps(follower.status())
    finally:
        gateway.close()


def test_a_changed_config_is_followed_at_once_during_a_held_watch(follower):
    first, second = Gateway(hold=30), Gateway(hold=30)
    try:
        first.set("a:1", datetime(2020, 1, 1, tzinfo=timezone.utc), 1.0)
        second.set("b:1", datetime(2030, 1, 1, tzinfo=timezone.utc), 1.0)
        follower.configure({"watch_url": first.url, "name": "desktop", "token": "t"})
        follower.until(lambda: ("a:1", "desktop", "Bearer t") in first.watches)  # held now

        changed = time.monotonic()
        follower.configure({"watch_url": second.url, "name": "desktop", "token": "t"})
        follower.until(lambda: follower.status().get("version") == "b:1", within=2)

        assert time.monotonic() - changed < 1.5
        assert abs(follower.universe() - datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp()) < 3
    finally:
        first.close()
        second.close()


def test_a_gateway_it_cant_reach_is_named_in_its_status_and_it_keeps_trying(follower):
    follower.configure({"watch_url": "http://127.0.0.1:9/clock/watch", "name": "desktop", "token": "t"})

    status = follower.until(lambda: follower.status() if follower.status().get("error") else None)

    assert status["connected"] is False and status["url"] == "http://127.0.0.1:9/clock/watch"
    assert "Connection refused" in status["error"] or "URLError" in status["error"]
    follower.until(lambda: follower.status().get("errors", 0) >= 2)


def test_a_config_that_isnt_one_is_named_and_a_good_one_then_followed(follower):
    gateway = Gateway()
    try:
        follower.configure({"watch_url": gateway.url, "name": "desktop"})
        follower.until(lambda: "isn't one" in (follower.status().get("error") or ""))

        gateway.set("boot:1", datetime(2020, 1, 1, tzinfo=timezone.utc), 1.0)
        follower.configure({"watch_url": gateway.url, "name": "desktop", "token": "t"})
        follower.until(lambda: follower.status().get("connected"))
    finally:
        gateway.close()


def test_the_drivers_read_of_the_gateways_version_names_no_follower_so_it_acknowledges_nothing():
    gateway = Gateway()
    try:
        gateway.set("boot:7", datetime(2020, 1, 1, tzinfo=timezone.utc), 1.0)

        assert protocol_clock.gateway_version(gateway.url) == "boot:7"
        assert gateway.watches == [("", None, None)]
    finally:
        gateway.close()
