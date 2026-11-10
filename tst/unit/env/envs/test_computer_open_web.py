"""A browser alone on the open web: the headless base's stand-in proxy on 3128, its route out with no site routing,
forwards any host, HTTP and CONNECT alike, without decrypting, but only to an address it resolved and checked itself,
refusing loopback, private, link-local and the other addresses that aren't globally routable; and it logs each
connection and refusal as a JSON line. The shared site proxy judges the addresses of a live site's hosts the same way.
The proxy is a network program, so these tests open sockets, on loopback alone."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import resource
import socket
import threading
from pathlib import Path

import pytest

import agent_env
from agent_env.env.envs.computer.utils import site_proxy

pytestmark = pytest.mark.usefixtures("socket_enabled")

STAND_IN = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04-headless" / "site-proxy.py"
GITHUB = "140.82.112.3"
REFUSED = ["127.0.0.1", "127.8.9.1", "10.0.0.1", "172.16.5.4", "192.168.1.1", "169.254.169.254", "100.64.0.1",
           "0.0.0.0", "224.0.0.1", "255.255.255.255", "192.0.2.1", "::1", "::", "fe80::1", "fe80::1%eth0", "fc00::1",
           "fd12:3456::1", "ff02::1", "::ffff:10.0.0.1", "::ffff:127.0.0.1", "64:ff9b::a00:1", "2002:a9fe:a9fe::1"]
ALLOWED = [GITHUB, "8.8.8.8", "2606:4700:4700::1111", "::ffff:140.82.112.3", "64:ff9b::8c52:7003"]


def _stand_in():
    spec = importlib.util.spec_from_file_location("headless_site_proxy", STAND_IN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("judge", ["stand-in", "site proxy"])
def test_both_proxies_refuse_every_address_that_isnt_globally_routable_and_take_the_rest(judge):
    allowed = _stand_in().allowed if judge == "stand-in" else site_proxy.allowed

    assert [address for address in REFUSED if allowed(address)] == []
    assert [address for address in ALLOWED if not allowed(address)] == []


@pytest.fixture
def proxy(monkeypatch):
    with _serving(monkeypatch, _stand_in()) as serving:
        yield serving


@contextlib.contextmanager
def _serving(monkeypatch, module):
    """``module``'s stand-in serving on a free port, with ``site.test`` resolving to one of GitHub's addresses and
    ``rebound.test`` to an internal one, and connections to an address off loopback landing on a local upstream that
    answers each request; yields the port, each address the proxy connected to, what the upstream read, and the
    upstream's own port."""
    real_resolve, real_connect = socket.getaddrinfo, socket.create_connection
    names = {"site.test": [GITHUB], "rebound.test": ["10.0.0.7"], "mixed.test": ["192.168.0.2", "140.82.112.4"]}
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
        if host in names:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in names[host]]
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
        else:
            while chunk := conn.recv(65536):
                answer += chunk
        return answer


def _logged(text: str, kind: str) -> list[dict]:
    return [json.loads(line)[kind] for line in text.splitlines() if line.startswith(f'{{"{kind}"')]


@pytest.mark.parametrize("request_line", [
    b"CONNECT 127.0.0.1:18765 HTTP/1.1", b"CONNECT [::1]:443 HTTP/1.1", b"CONNECT 169.254.169.254:80 HTTP/1.1",
    b"GET http://10.0.0.1/ HTTP/1.1", b"GET http://169.254.169.254/latest/meta-data/ HTTP/1.1",
    b"GET http://rebound.test/ HTTP/1.1", b"CONNECT rebound.test:443 HTTP/1.1", b"GET http://localhost:18765/ HTTP/1.1",
], ids=["connect-loopback", "connect-ipv6-loopback", "connect-metadata", "url-private", "url-metadata",
        "name-resolving-internal", "connect-name-resolving-internal", "url-localhost"])
def test_the_stand_in_refuses_an_internal_address_at_once_whoever_names_it_and_logs_why(proxy, capsys, request_line):
    port, opened, read, _ = proxy

    answer = _ask(port, request_line + b"\r\nHost: x\r\n\r\n")

    assert answer.startswith(b"HTTP/1.1 403 ") and b"X-Agentenv-Site-Proxy: failed" in answer, answer
    assert b"is an internal address, which the browser never reaches" in answer
    assert opened == [] and read == []
    [refused] = _logged(capsys.readouterr().out, "refused")
    assert refused["reason"] == "internal address" and refused["addresses"]


def test_the_stand_in_tunnels_and_forwards_a_public_host_at_the_address_it_checked_and_logs_each_connection(
        proxy, capsys):
    port, opened, read, _ = proxy

    tunnelled = _ask(port, b"CONNECT site.test:443 HTTP/1.1\r\nHost: site.test:443\r\n\r\n",
                     then=b"GET / HTTP/1.1\r\nHost: site.test\r\n\r\n")
    forwarded = _ask(port, b"GET http://mixed.test/a?b=1 HTTP/1.1\r\nHost: mixed.test\r\nProxy-Connection: keep-alive"
                           b"\r\n\r\n")

    assert tunnelled.startswith(b"HTTP/1.1 200 Connection established") and tunnelled.endswith(b"hello"), tunnelled
    assert forwarded.startswith(b"HTTP/1.1 200 OK") and forwarded.endswith(b"hello"), forwarded
    assert opened == [(GITHUB, 443), ("140.82.112.4", 80)]
    assert read[0] == b"GET / HTTP/1.1\r\nHost: site.test\r\n\r\n"
    assert read[1].startswith(b"GET /a?b=1 HTTP/1.1\r\n") and b"Host: mixed.test" in read[1]
    assert b"Proxy-Connection" not in read[1] and b"Connection: close" in read[1]
    logged = _logged(capsys.readouterr().out, "connection")
    assert [(entry["host"], entry["port"], entry["address"]) for entry in logged] == [
        ("site.test", 443, GITHUB), ("mixed.test", 80, "140.82.112.4")]
    assert all(entry["bytes_sent"] > 0 and entry["bytes_received"] > 0 for entry in logged), logged
    assert all(set(entry) >= {"started", "duration_s"} for entry in logged)


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


def test_the_stand_in_answers_a_name_that_doesnt_resolve_at_once(proxy, capsys):
    port, opened, _, _ = proxy

    answer = _ask(port, b"CONNECT nowhere.invalid:443 HTTP/1.1\r\nHost: nowhere.invalid:443\r\n\r\n")

    assert answer.startswith(b"HTTP/1.1 502 ") and b"nowhere.invalid doesn't resolve" in answer, answer
    assert opened == [] and _logged(capsys.readouterr().out, "refused")[0]["reason"].startswith("doesn't resolve")


def test_with_its_own_loopback_let_in_the_stand_in_reaches_this_computers_servers_and_nothing_else_internal(
        monkeypatch, capsys):
    monkeypatch.setenv("AGENTENV_SETTING_OWN_LOOPBACK", "True")
    with _serving(monkeypatch, _stand_in()) as (port, opened, read, upstream_port):
        own = _ask(port, f"GET http://127.0.0.1:{upstream_port}/page HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        private = _ask(port, b"GET http://10.0.0.1/ HTTP/1.1\r\nHost: x\r\n\r\n")

    assert own.startswith(b"HTTP/1.1 200 OK") and read[0].startswith(b"GET /page HTTP/1.1"), own
    assert private.startswith(b"HTTP/1.1 403 "), private
