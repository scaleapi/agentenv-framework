"""The browser base's two site proxies hand a live site to the egress proxy alike: the stand-in, which serves the
browser computer deployed alone and has only the standard library in the image, and core's, which serves it in a multi
env. Each takes the same addresses for public, sends the same CONNECT, and takes the same answers for established,
refused, closed or endless, so a fix to one that the other lacks fails here. Both reach a stand-in egress proxy on
loopback."""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import socket
import threading
from pathlib import Path

import pytest

import agent_env
from agent_env.env.envs.computer.utils import site_proxy

pytestmark = pytest.mark.usefixtures("socket_enabled")

STAND_IN_FILE = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04-browser" / "site-proxy.py"
ESTABLISHED = b"HTTP/1.1 200 Connection established\r\n\r\n"
ANSWERS = {"refused": b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n", "closed": b"",
           "endless": b"HTTP/1.1 200 OK\r\nX-Pad: " + b"a" * (2 << 20)}
REFUSALS = {"refused": "the egress proxy answered 'HTTP/1.1 403 Forbidden'",
            "closed": "the egress proxy closed the tunnel before answering it",
            "endless": "the egress proxy's answer has no end to its head"}


def _load_stand_in():
    spec = importlib.util.spec_from_file_location("stand_in_hand_off", STAND_IN_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAND_IN = _load_stand_in()


@contextlib.contextmanager
def _egress(answer: bytes = ESTABLISHED):
    """An egress proxy on loopback that records each CONNECT head, answers it with ``answer`` and, once established,
    echoes what comes through the tunnel; yields its address and the heads."""
    server, heads = socket.create_server(("127.0.0.1", 0)), []

    def serve(conn: socket.socket) -> None:
        with conn, contextlib.suppress(OSError):
            head = b""
            while not head.endswith(b"\r\n\r\n") and (byte := conn.recv(1)):
                head += byte
            heads.append(head)
            conn.sendall(answer)
            if answer == ESTABLISHED:
                conn.sendall(conn.recv(65536))

    def accept() -> None:
        with contextlib.suppress(OSError):
            while True:
                threading.Thread(target=serve, args=(server.accept()[0],), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    try:
        yield server.getsockname()[:2], heads
    finally:
        server.close()


def _through_stand_in(egress: tuple[str, int], host: str, port: int, payload: bytes) -> bytes:
    with STAND_IN._through_egress(egress, host, port, 5) as tunnel:
        tunnel.sendall(payload)
        return tunnel.recv(65536)


def _through_core(egress: tuple[str, int], host: str, port: int, payload: bytes) -> bytes:
    async def run() -> bytes:
        reader, writer = await site_proxy.through_egress(egress, host, port)
        try:
            writer.write(payload)
            await writer.drain()
            return await asyncio.wait_for(reader.read(65536), 5)
        finally:
            writer.close()

    return asyncio.run(run())


HAND_OFFS = {"stand-in": _through_stand_in, "core": _through_core}
PUBLICS = {"stand-in": STAND_IN._public, "core": site_proxy.public}
hand_off = pytest.mark.parametrize("through", HAND_OFFS.values(), ids=list(HAND_OFFS))


def test_both_read_the_egress_proxy_from_the_same_variable():
    assert STAND_IN.EGRESS_PROXY_ENV == site_proxy.EGRESS_PROXY_ENV


@pytest.mark.parametrize("is_public", PUBLICS.values(), ids=list(PUBLICS))
@pytest.mark.parametrize(("addresses", "expected"), [
    (["140.82.112.3"], True), (["2606:4700::1111"], True), (["140.82.112.3", "2606:4700::1111"], True),
    ([], False), (["10.0.0.5"], False), (["127.0.0.1"], False), (["::1"], False), (["fe80::1%eth0"], False),
    (["100.64.0.1"], False), (["140.82.112.3", "192.168.1.2"], False)])
def test_both_take_the_same_addresses_for_public(is_public, addresses, expected):
    assert is_public(addresses) is expected


@hand_off
@pytest.mark.parametrize(("host", "authority"), [("github.com", b"github.com:443"), ("2606:4700::1111",
                                                                                     b"[2606:4700::1111]:443")])
def test_both_send_the_same_connect_and_carry_the_tunnel_once_its_established(through, host, authority):
    with _egress() as (egress, heads):
        echoed = through(egress, host, 443, b"hello")

    assert echoed == b"hello"
    assert heads == [b"CONNECT " + authority + b" HTTP/1.1\r\nHost: " + authority + b"\r\n\r\n"]


@hand_off
@pytest.mark.parametrize("answer", ANSWERS, ids=list(ANSWERS))
def test_both_refuse_an_answer_that_isnt_established_naming_it_alike(through, answer):
    with _egress(ANSWERS[answer]) as (egress, _), pytest.raises(ConnectionError) as refused:
        through(egress, "github.com", 443, b"hello")

    assert type(refused.value).__name__ == "EgressRefused" and str(refused.value) == REFUSALS[answer]
