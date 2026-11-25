"""Exercise RPC routing through HTTPX down to the TCP and TLS handshakes."""

import errno
import socket
import ssl

import httpcore
import pytest
from httpcore._backends.auto import AutoBackend

from agentenv_protocol import client


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme,proxies,via_proxy,refuse_interval", [
    ("http", {}, False, False),
    ("https", {}, False, False),
    ("https", {}, False, True),
    ("http", {"HTTP_PROXY": "http://proxy.test:8080"}, True, False),
    ("https", {"HTTPS_PROXY": "http://proxy.test:8080"}, True, False),
    ("http", {"ALL_PROXY": "http://proxy.test:8080"}, True, False),
    ("https", {"HTTPS_PROXY": "http://proxy.test:8080", "NO_PROXY": ".env.test"}, False, False),
    ("http", {"ALL_PROXY": "http://proxy.test:8080", "NO_PROXY": "*"}, False, False),
])
@pytest.mark.parametrize("verify", [True, False])
async def test_rpc_socket_options_and_routing(monkeypatch, scheme, proxies, via_proxy, refuse_interval, verify):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "REQUEST_METHOD"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "unused.test")
    for name, value in proxies.items():
        monkeypatch.setenv(name, value)

    connections = []
    handshakes = []
    options = []
    interval = getattr(socket, "TCP_KEEPINTVL", 12345)
    if refuse_interval:
        monkeypatch.setattr(socket, "TCP_KEEPINTVL", interval, raising=False)

    class Socket:
        def setsockopt(self, level, option, value):
            options.append((level, option, value))
            if refuse_interval and option == interval:
                raise OSError(errno.ENOPROTOOPT, "interval unavailable")

    raw = Socket()
    body = b'{"jsonrpc":"2.0","id":1,"result":{}}'
    responses = [b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body]
    if via_proxy and scheme == "https":
        responses.insert(0, b"HTTP/1.1 200 Connection established\r\n\r\n")

    class Stream(httpcore.AsyncMockStream):
        def get_extra_info(self, info):
            return raw if info == "socket" else super().get_extra_info(info)

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            handshakes.append((server_hostname, ssl_context.verify_mode))
            return self

    async def connect_tcp(self, host, port, socket_options, **kwargs):
        connections.append((host, port))
        for option in socket_options or ():
            raw.setsockopt(*option)
        return Stream(responses)

    monkeypatch.setattr(AutoBackend, "connect_tcp", connect_tcp)
    await client.add_data(f"{scheme}://worker.env.test", [], verify=verify)

    (host, port), = connections
    target = ("proxy.test", 8080) if via_proxy else ("worker.env.test", 443 if scheme == "https" else 80)
    assert (host, port) == target
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options
    idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    if idle is not None:
        assert (socket.IPPROTO_TCP, idle, 30) in options
    for name, value in (("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 3)):
        if hasattr(socket, name):
            assert (socket.IPPROTO_TCP, getattr(socket, name), value) in options
    expected = ssl.CERT_REQUIRED if verify else ssl.CERT_NONE
    assert handshakes == ([("worker.env.test", expected)] if scheme == "https" else [])
