"""HTTP client for data-plane RPCs that may wait a long time for response headers."""

import socket

import httpx
from httpx._utils import get_environment_proxies


def rpc_client(verify: bool) -> httpx.AsyncClient:
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    if idle is not None:
        options.append((socket.IPPROTO_TCP, idle, 30))
    for name, value in (("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 3)):
        option = getattr(socket, name, None)
        if option is not None:
            options.append((socket.IPPROTO_TCP, option, value))

    # A custom default transport disables HTTPX's environment proxy discovery.
    # Its NO_PROXY entries map to None, which selects our direct transport.
    mounts = {
        pattern: httpx.AsyncHTTPTransport(proxy=proxy, verify=verify) if proxy is not None else None
        for pattern, proxy in get_environment_proxies().items()
    }
    direct = httpx.AsyncHTTPTransport(verify=verify, socket_options=options)
    return httpx.AsyncClient(transport=direct, mounts=mounts)
