"""HTTP client for data-plane RPCs that may wait a long time for response headers."""

import socket
from contextlib import suppress

import httpx
from httpx._utils import get_environment_proxies


class _DirectTransport(httpx.AsyncHTTPTransport):
    _KEEPALIVE_IDLE_SECONDS = 30
    _KEEPALIVE_PROBE_INTERVAL_SECONDS = 10
    _KEEPALIVE_PROBE_COUNT = 3

    def _enable_keepalive(self, sock: socket.socket) -> None:
        options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
        if idle is not None:
            options.append((socket.IPPROTO_TCP, idle, self._KEEPALIVE_IDLE_SECONDS))
        for name, value in (
            ("TCP_KEEPINTVL", self._KEEPALIVE_PROBE_INTERVAL_SECONDS),
            ("TCP_KEEPCNT", self._KEEPALIVE_PROBE_COUNT),
        ):
            option = getattr(socket, name, None)
            if option is not None:
                options.append((socket.IPPROTO_TCP, option, value))
        for level, option, value in options:
            with suppress(OSError):
                sock.setsockopt(level, option, value)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        extensions = request.extensions
        previous = extensions.get("trace")

        async def trace(event, info):
            if event == "connection.connect_tcp.complete":
                sock = info["return_value"].get_extra_info("socket")
                if sock is not None:
                    self._enable_keepalive(sock)
            if previous is not None:
                await previous(event, info)

        # Scope tuning to the direct hop. HTTPX can reuse the original request's extensions on a redirect.
        request.extensions = {**extensions, "trace": trace}
        try:
            return await super().handle_async_request(request)
        finally:
            request.extensions = extensions


def rpc_client(verify: bool) -> httpx.AsyncClient:
    # A custom default transport disables HTTPX's environment proxy discovery.
    # Its NO_PROXY entries map to None, which selects our direct transport.
    mounts = {
        pattern: httpx.AsyncHTTPTransport(proxy=proxy, verify=verify) if proxy is not None else None
        for pattern, proxy in get_environment_proxies().items()
    }
    direct = _DirectTransport(verify=verify)
    return httpx.AsyncClient(transport=direct, mounts=mounts)
