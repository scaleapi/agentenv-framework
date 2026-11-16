"""The headless base's stand-in proxy on 3128, a browser alone's route out with no site routing: it forwards any
host, HTTP and CONNECT alike, without decrypting and refusing nothing, the sandbox's network policy deciding where it
can go; it bounds every wait, and logs each connection, and each one that fails, as a JSON line. The proxy is a
network program, so these tests open sockets, on loopback alone."""

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
from types import SimpleNamespace

import pytest

import agent_env

pytestmark = pytest.mark.usefixtures("socket_enabled")

STAND_IN = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04-browser" / "site-proxy.py"
GITHUB = "140.82.112.3"
OK = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello"


def _stand_in():
    spec = importlib.util.spec_from_file_location("browser_site_proxy", STAND_IN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _accepting(server: socket.socket, each) -> None:
    """``each`` run on every connection ``server`` accepts, on a thread of its own, until it closes."""
    def run():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            each(conn)

    threading.Thread(target=run, daemon=True).start()


@contextlib.contextmanager
def _serving(monkeypatch, names: dict | None = None, acts: dict | None = None, **settings):
    """The stand-in serving on a free port, with its module's ``settings`` set first. Each of ``names`` resolves to the
    addresses it gives (none: it doesn't resolve), ``site.test`` to one of GitHub's unless given. A connection to an
    address off loopback acts as ``acts`` says, ``"refuses"``, ``"hangs"`` (for its whole timeout) or ``"never
    reads"``, else lands on a local upstream that answers each request with ``OK``. Yields the port, each address and
    port the proxy connected to, what the upstream read, and the upstream's own port."""
    module = _stand_in()
    for name, value in settings.items():
        monkeypatch.setattr(module, name, value)
    names, acts = {"site.test": [GITHUB], **(names or {})}, acts or {}
    real_resolve, real_connect = socket.getaddrinfo, socket.create_connection
    upstream, sink = socket.create_server(("127.0.0.1", 0)), socket.create_server(("127.0.0.1", 0))
    read, opened, held = [], [], []

    def answer(conn):
        with conn:
            read.append(conn.recv(65536))
            conn.sendall(OK)

    def getaddrinfo(host, port, *args, **kwargs):
        if host not in names:
            return real_resolve(host, port, *args, **kwargs)
        if not names[host]:
            raise socket.gaierror(-2, "Name or service not known")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in names[host]]

    def create_connection(address, *args, **kwargs):
        if address[0] in ("127.0.0.1", "localhost"):
            return real_connect(address, *args, **kwargs)
        opened.append(address)
        if (act := acts.get(address[0])) == "refuses":
            raise ConnectionRefusedError(f"{address[0]} refused")
        if act == "hangs":
            time.sleep(kwargs.get("timeout") or 0)
            raise TimeoutError("timed out")
        return real_connect((sink if act == "never reads" else upstream).getsockname(), *args, **kwargs)

    monkeypatch.setattr(module.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(module.socket, "create_connection", create_connection)
    _accepting(upstream, answer)
    _accepting(sink, held.append)
    server = module.Server(("127.0.0.1", 0), module.Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield SimpleNamespace(port=server.server_address[1], opened=opened, read=read,
                              upstream_port=upstream.getsockname()[1], module=module)
    finally:
        server.shutdown()
        upstream.close()
        sink.close()


@pytest.fixture
def proxy(monkeypatch):
    with _serving(monkeypatch) as serving:
        yield serving


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


def _logged(text: str, kind: str | None = None) -> list[dict]:
    """A proxy's log, every line of which is JSON: each line's record, or, given ``kind`` (``connection``,
    ``failed``, ...), the records of that kind."""
    entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    return entries if kind is None else [entry[kind] for entry in entries if kind in entry]


def test_the_stand_in_tunnels_and_forwards_a_public_host_and_logs_each_connection(proxy, capsys):
    tunnelled = _ask(proxy.port, b"CONNECT site.test:443 HTTP/1.1\r\nHost: site.test:443\r\n\r\n",
                     then=b"GET / HTTP/1.1\r\nHost: site.test\r\n\r\n")
    forwarded = _ask(proxy.port, b"GET http://site.test/a?b=1 HTTP/1.1\r\nHost: site.test\r\n"
                                 b"Proxy-Connection: keep-alive\r\n\r\n")

    assert tunnelled.startswith(b"HTTP/1.1 200 Connection established") and tunnelled.endswith(b"hello"), tunnelled
    assert forwarded.startswith(b"HTTP/1.1 200 OK") and forwarded.endswith(b"hello"), forwarded
    assert proxy.opened == [(GITHUB, 443), (GITHUB, 80)]
    assert proxy.read[0] == b"GET / HTTP/1.1\r\nHost: site.test\r\n\r\n"
    assert proxy.read[1].startswith(b"GET /a?b=1 HTTP/1.1\r\n") and b"Host: site.test" in proxy.read[1]
    assert b"Proxy-Connection" not in proxy.read[1] and b"Connection: close" in proxy.read[1]
    logged = _logged(capsys.readouterr().out, "connection")
    assert [(entry["host"], entry["port"], entry["address"]) for entry in logged] == [
        ("site.test", 443, GITHUB), ("site.test", 80, GITHUB)]
    assert all(entry["bytes_sent"] > 0 and entry["bytes_received"] > 0 for entry in logged), logged
    assert all(set(entry) >= {"started", "duration_s"} for entry in logged)


def test_the_stand_in_refuses_nothing_it_reaches_the_computers_own_loopback_and_an_internal_address_alike(proxy):
    own = _ask(proxy.port, f"GET http://127.0.0.1:{proxy.upstream_port}/page HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    private = _ask(proxy.port, b"GET http://10.0.0.1/ HTTP/1.1\r\nHost: x\r\n\r\n")

    assert own.startswith(b"HTTP/1.1 200 OK") and private.startswith(b"HTTP/1.1 200 OK"), (own, private)
    assert proxy.read[0].startswith(b"GET /page HTTP/1.1") and proxy.opened == [("10.0.0.1", 80)]


def test_the_stand_in_answers_a_name_that_doesnt_resolve_at_once_and_logs_it(monkeypatch, capsys):
    with _serving(monkeypatch, names={"nowhere.test": []}) as proxy:
        answer = _ask(proxy.port, b"CONNECT nowhere.test:443 HTTP/1.1\r\nHost: nowhere.test:443\r\n\r\n")

    assert answer.startswith(b"HTTP/1.1 502 ") and b"nowhere.test doesn't resolve" in answer, answer
    assert b"X-Agentenv-Site-Proxy: failed" in answer and proxy.opened == []
    [failed] = _logged(capsys.readouterr().out, "failed")
    assert (failed["reason"], failed["error"]) == ("doesn't resolve", "gaierror: [Errno -2] Name or service not known")


@pytest.mark.parametrize("first", ["refuses", "hangs"])
def test_the_stand_in_tries_a_names_addresses_in_turn_within_one_deadline_for_them_all(monkeypatch, capsys, first):
    with _serving(monkeypatch, names={"site.test": [GITHUB, "140.82.112.4"]}, acts={GITHUB: first},
                  CONNECT_TIMEOUT_S=0.5) as proxy:
        began = time.monotonic()
        answer = _ask(proxy.port, b"GET http://site.test/ HTTP/1.1\r\nHost: site.test\r\n\r\n")
        took = time.monotonic() - began

    if first == "refuses":
        assert answer.startswith(b"HTTP/1.1 200 OK") and proxy.opened == [(GITHUB, 80), ("140.82.112.4", 80)], answer
    else:
        assert answer.startswith(b"HTTP/1.1 502 ") and b"site.test:80 didn't answer" in answer, answer
        assert proxy.opened == [(GITHUB, 80)] and took < 1.5, (proxy.opened, took)
        [failed] = _logged(capsys.readouterr().out, "failed")
        assert (failed["reason"], failed["error"]) == ("no answer", "TimeoutError: timed out")


def test_the_stand_in_relays_a_connection_whose_sockets_are_numbered_past_1023(proxy):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 1400:
        pytest.skip(f"this machine allows {hard} open files, too few to number a socket past 1023")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 1400) if hard == resource.RLIM_INFINITY else
                                                min(max(soft, 1400), hard), hard))
    held = [os.pipe() for _ in range(520)]
    try:
        answer = _ask(proxy.port, b"CONNECT site.test:443 HTTP/1.1\r\nHost: site.test:443\r\n\r\n",
                      then=b"GET / HTTP/1.1\r\nHost: site.test\r\n\r\n")
    finally:
        for pair in held:
            os.close(pair[0])
            os.close(pair[1])
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    assert answer.startswith(b"HTTP/1.1 200 Connection established") and answer.endswith(b"hello"), answer
    assert proxy.opened == [(GITHUB, 443)]


def test_the_stand_in_ends_a_tunnel_whose_server_stopped_reading_instead_of_waiting_on_it_for_ever(monkeypatch, capsys):
    with _serving(monkeypatch, acts={GITHUB: "never reads"}, IDLE_S=1) as proxy:
        with socket.create_connection(("127.0.0.1", proxy.port), timeout=20) as conn:
            conn.sendall(b"CONNECT site.test:443 HTTP/1.1\r\nHost: site.test:443\r\n\r\n")
            assert conn.recv(4096).startswith(b"HTTP/1.1 200")
            began, ended = time.monotonic(), None
            try:
                while time.monotonic() - began < 20:
                    conn.sendall(b"x" * 65536)
            except OSError:
                ended = time.monotonic() - began

    assert ended is not None and ended < 15, ended
    [logged] = _logged(capsys.readouterr().out, "connection")
    assert logged["host"] == "site.test" and logged["bytes_sent"] > 0
