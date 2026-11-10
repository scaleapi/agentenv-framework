"""A browser alone on the open web: the headless base's stand-in proxy on 3128, its route out with no site routing,
forwards any host, HTTP and CONNECT alike, without decrypting and refusing nothing, the sandbox's network policy
deciding where it can go; it bounds every wait, and logs each connection, and each one that fails, as a JSON line. The
proxy is a network program, so these tests open sockets, on loopback alone."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import resource
import socket
import threading
import time
from pathlib import Path

import pytest

import agent_env

pytestmark = pytest.mark.usefixtures("socket_enabled")

STAND_IN = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04-headless" / "site-proxy.py"
GITHUB = "140.82.112.3"


def _stand_in():
    spec = importlib.util.spec_from_file_location("headless_site_proxy", STAND_IN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def proxy(monkeypatch):
    with _serving(monkeypatch, _stand_in()) as serving:
        yield serving


@contextlib.contextmanager
def _serving(monkeypatch, module):
    """``module``'s stand-in serving on a free port, with ``site.test`` resolving to one of GitHub's addresses, and
    connections to an address off loopback landing on a local upstream that answers each request; yields the port,
    each address the proxy connected to, what the upstream read, and the upstream's own port."""
    real_resolve, real_connect = socket.getaddrinfo, socket.create_connection
    upstream, read, opened = socket.create_server(("127.0.0.1", 0)), [], []

    def answer():
        while True:
            try:
                conn, _ = upstream.accept()
            except OSError:
                return
            with conn:
                read.append(conn.recv(65536))
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello")

    def getaddrinfo(host, port, *args, **kwargs):
        if host == "site.test":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (GITHUB, port))]
        return real_resolve(host, port, *args, **kwargs)

    def create_connection(address, *args, **kwargs):
        if address[0] in ("127.0.0.1", "localhost"):
            return real_connect(address, *args, **kwargs)
        opened.append(address)
        return real_connect(upstream.getsockname(), *args, **kwargs)

    monkeypatch.setattr(module.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(module.socket, "create_connection", create_connection)
    server = module.Server(("127.0.0.1", 0), module.Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=answer, daemon=True).start()
    try:
        yield server.server_address[1], opened, read, upstream.getsockname()[1]
    finally:
        server.shutdown()
        upstream.close()


def _ask(port: int, request: bytes, then: bytes = b"") -> bytes:
    """What the proxy answers ``request``, and in a tunnel it opened, ``then``."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
        conn.sendall(request)
        answer = conn.recv(65536)
        if then and answer.startswith(b"HTTP/1.1 200"):
            conn.sendall(then)
        while chunk := conn.recv(65536):
            answer += chunk
        return answer


def _logged(text: str, kind: str) -> list[dict]:
    return [json.loads(line)[kind] for line in text.splitlines() if line.startswith(f'{{"{kind}"')]


def test_the_stand_in_tunnels_and_forwards_a_public_host_and_logs_each_connection(proxy, capsys):
    port, opened, read, _ = proxy

    tunnelled = _ask(port, b"CONNECT site.test:443 HTTP/1.1\r\nHost: site.test:443\r\n\r\n",
                     then=b"GET / HTTP/1.1\r\nHost: site.test\r\n\r\n")
    forwarded = _ask(port, b"GET http://site.test/a?b=1 HTTP/1.1\r\nHost: site.test\r\nProxy-Connection: keep-alive"
                           b"\r\n\r\n")

    assert tunnelled.startswith(b"HTTP/1.1 200 Connection established") and tunnelled.endswith(b"hello"), tunnelled
    assert forwarded.startswith(b"HTTP/1.1 200 OK") and forwarded.endswith(b"hello"), forwarded
    assert opened == [(GITHUB, 443), (GITHUB, 80)]
    assert read[0] == b"GET / HTTP/1.1\r\nHost: site.test\r\n\r\n"
    assert read[1].startswith(b"GET /a?b=1 HTTP/1.1\r\n") and b"Host: site.test" in read[1]
    assert b"Proxy-Connection" not in read[1] and b"Connection: close" in read[1]
    logged = _logged(capsys.readouterr().out, "connection")
    assert [(entry["host"], entry["port"], entry["address"]) for entry in logged] == [
        ("site.test", 443, GITHUB), ("site.test", 80, GITHUB)]
    assert all(entry["bytes_sent"] > 0 and entry["bytes_received"] > 0 for entry in logged), logged
    assert all(set(entry) >= {"started", "duration_s"} for entry in logged)


def test_the_stand_in_refuses_nothing_it_reaches_the_computers_own_loopback_and_an_internal_address_alike(proxy):
    port, opened, read, upstream_port = proxy

    own = _ask(port, f"GET http://127.0.0.1:{upstream_port}/page HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    private = _ask(port, b"GET http://10.0.0.1/ HTTP/1.1\r\nHost: x\r\n\r\n")

    assert own.startswith(b"HTTP/1.1 200 OK") and private.startswith(b"HTTP/1.1 200 OK"), (own, private)
    assert read[0].startswith(b"GET /page HTTP/1.1") and opened == [("10.0.0.1", 80)]


def test_the_stand_in_answers_a_name_that_doesnt_resolve_at_once_and_logs_it(proxy, capsys):
    port, opened, _, _ = proxy

    answer = _ask(port, b"CONNECT nowhere.invalid:443 HTTP/1.1\r\nHost: nowhere.invalid:443\r\n\r\n")

    assert answer.startswith(b"HTTP/1.1 502 ") and b"nowhere.invalid doesn't resolve" in answer, answer
    assert b"X-Agentenv-Site-Proxy: failed" in answer and opened == []
    assert _logged(capsys.readouterr().out, "failed")[0]["reason"].startswith("doesn't resolve")


@pytest.mark.parametrize("first", ["refuses", "hangs"])
def test_the_stand_in_tries_a_names_addresses_in_turn_within_one_deadline_for_them_all(monkeypatch, capsys, first):
    module = _stand_in()
    monkeypatch.setattr(module, "CONNECT_TIMEOUT_S", 0.5)
    real_resolve, real_connect = socket.getaddrinfo, socket.create_connection
    upstream, tried = socket.create_server(("127.0.0.1", 0)), []

    def answer_once():
        with contextlib.suppress(OSError), upstream.accept()[0] as conn:
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")

    threading.Thread(target=answer_once, daemon=True).start()

    def create_connection(address, timeout=None, *args, **kwargs):
        if address[0] == "127.0.0.1":
            return real_connect(address, timeout, *args, **kwargs)
        tried.append(address[0])
        if address[0] == GITHUB:
            if first == "refuses":
                raise ConnectionRefusedError(address)
            time.sleep(timeout)
            raise TimeoutError(address)
        return real_connect(upstream.getsockname(), timeout, *args, **kwargs)

    monkeypatch.setattr(module.socket, "getaddrinfo", lambda host, port, *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in (GITHUB, "140.82.112.4")
    ] if host == "site.test" else real_resolve(host, port, *a, **k))
    monkeypatch.setattr(module.socket, "create_connection", create_connection)
    server = module.Server(("127.0.0.1", 0), module.Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        began = time.monotonic()
        answer = _ask(server.server_address[1], b"GET http://site.test/ HTTP/1.1\r\nHost: site.test\r\n\r\n")
        took = time.monotonic() - began
    finally:
        server.shutdown()
        upstream.close()

    if first == "refuses":
        assert answer.startswith(b"HTTP/1.1 200 OK") and tried == [GITHUB, "140.82.112.4"], (answer, tried)
    else:
        assert answer.startswith(b"HTTP/1.1 502 ") and tried == [GITHUB] and took < 1.5, (answer, tried, took)
        [failed] = _logged(capsys.readouterr().out, "failed")
        assert failed["reason"] == "no answer" and failed["addresses"] == [GITHUB, "140.82.112.4"]


def test_the_stand_in_relays_a_connection_whose_sockets_are_numbered_past_1023(proxy):
    port, opened, read, _ = proxy
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 1400:
        pytest.skip(f"this machine allows {hard} open files, too few to number a socket past 1023")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 1400) if hard == resource.RLIM_INFINITY else
                                                min(max(soft, 1400), hard), hard))
    held = [os.pipe() for _ in range(520)]
    try:
        answer = _ask(port, b"CONNECT site.test:443 HTTP/1.1\r\nHost: site.test:443\r\n\r\n",
                      then=b"GET / HTTP/1.1\r\nHost: site.test\r\n\r\n")
    finally:
        for pair in held:
            os.close(pair[0])
            os.close(pair[1])
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    assert answer.startswith(b"HTTP/1.1 200 Connection established") and answer.endswith(b"hello"), answer
    assert opened == [(GITHUB, 443)]


def test_the_stand_in_ends_a_tunnel_whose_server_stopped_reading_instead_of_waiting_on_it_for_ever(monkeypatch, capsys):
    module = _stand_in()
    monkeypatch.setattr(module, "IDLE_S", 1)
    sink, accepted = socket.create_server(("127.0.0.1", 0)), []
    threading.Thread(target=lambda: accepted.append(sink.accept()), daemon=True).start()  # never reads
    real_resolve, real_connect = socket.getaddrinfo, socket.create_connection
    monkeypatch.setattr(module.socket, "getaddrinfo", lambda host, port, *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (GITHUB, port))] if host == "site.test" else real_resolve(
        host, port, *a, **k))
    monkeypatch.setattr(module.socket, "create_connection", lambda address, *a, **k: real_connect(
        sink.getsockname() if address[0] == GITHUB else address, *a, **k))
    server = module.Server(("127.0.0.1", 0), module.Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with socket.create_connection(server.server_address, timeout=20) as conn:
            conn.sendall(b"CONNECT site.test:443 HTTP/1.1\r\nHost: site.test:443\r\n\r\n")
            assert conn.recv(4096).startswith(b"HTTP/1.1 200")
            began, ended = time.monotonic(), None
            try:
                while time.monotonic() - began < 20:
                    conn.sendall(b"x" * 65536)
            except OSError:
                ended = time.monotonic() - began
    finally:
        server.shutdown()
        sink.close()

    assert ended is not None and ended < 15, ended
    [logged] = _logged(capsys.readouterr().out, "connection")
    assert logged["host"] == "site.test" and logged["bytes_sent"] > 0
