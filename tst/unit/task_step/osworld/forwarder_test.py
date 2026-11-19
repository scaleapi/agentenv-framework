"""Upstream's calls reach the guest server through loopback forwarders of their own: each call, method, path, query,
body and all, goes on under the guest server's prefix with the env's control headers, which win over the caller's; an
unreachable guest is 502; the Chrome DevTools and VLC ports refuse, and say which a failure needed; the ports close
once the guest is left."""
from __future__ import annotations

import json

import httpx
import pytest
from pytest_socket import enable_socket, socket_allow_hosts
import requests

from agent_env.task_step.task_steps.computer.osworld import forwarder
from agent_env.task_step.task_steps.computer.osworld.forwarder import CHROMIUM, VLC, Guest

BASE = "https://computer.example/svc/osworld-server/osworld"
HEADERS = {"Authorization": "Bearer control"}


@pytest.fixture(autouse=True)
def _loopback():
    """The forwarders listen on loopback; nothing else is reached."""
    enable_socket()
    socket_allow_hosts(["127.0.0.1"], allow_unix_socket=True)


@pytest.fixture
def seen(monkeypatch):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/down"):
            raise httpx.ConnectError("down")
        return httpx.Response(201, content=b"from guest", headers={"x-guest": "1", "content-encoding": "identity"})

    real = httpx.Client
    monkeypatch.setattr(forwarder.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return calls


def test_a_call_goes_on_under_the_prefix_with_the_control_headers(seen):
    with Guest(BASE, HEADERS) as guest:
        answer = requests.post(f"http://127.0.0.1:{guest.port}/setup/upload?x=1&x=2", data=b"payload",
                               headers={"Authorization": "Bearer agent", "X-Kept": "yes"}, timeout=10)
    assert (answer.status_code, answer.content, answer.headers["x-guest"]) == (201, b"from guest", "1")
    [call] = seen
    assert str(call.url) == f"{BASE}/setup/upload?x=1&x=2" and call.method == "POST" and call.content == b"payload"
    assert call.headers["authorization"] == "Bearer control" and call.headers["x-kept"] == "yes"


def test_an_unreachable_guest_is_502(seen):
    with Guest(BASE, HEADERS) as guest:
        assert requests.get(f"http://127.0.0.1:{guest.port}/down", timeout=10).status_code == 502


def test_the_ports_the_app_doesnt_expose_refuse_and_say_which_was_needed(seen):
    with Guest(BASE, HEADERS) as guest:
        count = guest.mark_count()
        assert guest.needed_since(count) is None
        answer = requests.get(f"http://127.0.0.1:{guest.ports[CHROMIUM]}/json/version", timeout=10)
        requests.post(f"http://127.0.0.1:{guest.ports[VLC]}/requests/status.xml", data=json.dumps({}), timeout=10)
        requests.get(f"http://127.0.0.1:{guest.ports[CHROMIUM]}/json", timeout=10)
        why = guest.needed_since(count)
    assert answer.status_code == 501 and seen == []
    assert why == ("it needs the computer's Chrome DevTools port 9222 and VLC's HTTP port 8080, which the "
                   "osworld-server app doesn't expose; not supported by the osworld_* steps")


def test_once_stopped_the_forwarder_refuses_every_call(seen):
    with Guest(BASE, HEADERS) as guest:
        guest.stop.set()
        answer = requests.post(f"http://127.0.0.1:{guest.port}/execute", json={"command": "rm -rf ~"}, timeout=10)
    assert answer.status_code == 503 and seen == []


def test_the_ports_close_once_the_guest_is_left(seen):
    with Guest(BASE, HEADERS) as guest:
        port = guest.port
    with pytest.raises(requests.ConnectionError):
        requests.get(f"http://127.0.0.1:{port}/screenshot", timeout=5)
