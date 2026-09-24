"""The /svc and /website proxy over real loopback sockets: what h11 actually frames for the client, end to end.

The mocked-transport tests prove streaming and header handling inside the ASGI app; these prove the wire: a
forwarded Content-Length that matches the bytes, HEAD, a chunked upstream, a mid-body upstream failure seen by
the client as a cut body, and the request-target as uvicorn really decodes and forwards it."""

from __future__ import annotations

import socket
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from pytest_socket import enable_socket
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from agent_env.env.gateway.gateway import Gateway

BODY = bytes(range(256)) * 256  # 64 KiB covering every byte value


@pytest.fixture(autouse=True)
def _disable_network():
    """Override the suite-wide socket block (tst/unit/conftest.py) for this module: loopback is the point here."""
    enable_socket()
    yield


async def _content_length(_: Request) -> Response:
    return Response(BODY, media_type="application/octet-stream")


async def _chunked(_: Request) -> Response:
    async def parts():
        for i in range(0, len(BODY), 4096):
            yield BODY[i : i + 4096]

    return StreamingResponse(parts(), media_type="application/octet-stream")


async def _cut(_: Request) -> Response:
    async def parts():  # declares the full length, then dies after 1 KiB
        yield BODY[:1024]
        raise RuntimeError("upstream died mid-body")

    return StreamingResponse(parts(), headers={"content-length": str(len(BODY))})


async def _echo_target(request: Request) -> Response:
    return Response(request.scope["raw_path"] + b"?" + request.scope["query_string"])


def _upstream_app() -> Starlette:
    return Starlette(routes=[
        Route("/cl", _content_length), Route("/chunked", _chunked), Route("/cut", _cut),
        Route("/echo/{rest:path}", _echo_target),
    ])


async def _serve(app) -> tuple[uvicorn.Server, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.set_inheritable(True)
    config = uvicorn.Config(app, lifespan="off", log_level="critical", access_log=False)
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)  # what Server.serve() does before startup(); no signal handlers here
    await server.startup(sockets=[sock])
    return server, f"http://127.0.0.1:{sock.getsockname()[1]}"


@asynccontextmanager
async def _proxy():
    """A real upstream and the real gateway app on loopback uvicorn servers; yields the gateway's base URL."""
    upstream, upstream_url = await _serve(_upstream_app())
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                 rest_proxy_urls={"svc": upstream_url}, website_urls={"web": upstream_url})
    gateway, gateway_url = await _serve(gw._mcp.streamable_http_app())
    try:
        yield gateway_url
    finally:
        await gateway.shutdown()
        await upstream.shutdown()


@pytest.mark.asyncio
async def test_content_length_body_is_forwarded_and_matches_the_bytes():
    async with _proxy() as gateway_url, httpx.AsyncClient() as client:
        r = await client.get(f"{gateway_url}/svc/svc/cl")
    assert r.status_code == 200 and r.content == BODY
    assert r.headers["content-length"] == str(len(BODY)) and "transfer-encoding" not in r.headers


@pytest.mark.asyncio
async def test_head_carries_the_length_with_no_body():
    async with _proxy() as gateway_url, httpx.AsyncClient() as client:
        r = await client.head(f"{gateway_url}/svc/svc/cl")
    assert r.status_code == 200 and r.content == b""
    assert r.headers["content-length"] == str(len(BODY))


@pytest.mark.asyncio
async def test_chunked_upstream_is_re_chunked_by_the_gateway():
    async with _proxy() as gateway_url, httpx.AsyncClient() as client:
        r = await client.get(f"{gateway_url}/website/web/chunked")
    assert r.status_code == 200 and r.content == BODY
    assert "content-length" not in r.headers and r.headers["transfer-encoding"] == "chunked"


@pytest.mark.asyncio
async def test_mid_body_upstream_failure_reaches_the_client_as_a_cut_body(caplog):
    async with _proxy() as gateway_url, httpx.AsyncClient() as client:
        with pytest.raises(httpx.RemoteProtocolError):
            await client.get(f"{gateway_url}/svc/svc/cut")
    assert any("failed after" in rec.getMessage() and "RemoteProtocolError" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_percent_encoded_request_target_reaches_the_upstream_intact():
    async with _proxy() as gateway_url, httpx.AsyncClient() as client:
        r = await client.get(f"{gateway_url}/svc/svc/echo/a%2Fb%3Fc%23d%C3%A9?q=%2F&q=x")
    assert r.status_code == 200 and r.content == b"/echo/a%2Fb%3Fc%23d%C3%A9?q=%2F&q=x"


@pytest.mark.asyncio
async def test_escaped_slash_in_the_route_prefix_is_refused_over_a_real_socket():
    async with _proxy() as gateway_url, httpx.AsyncClient() as client:
        r = await client.get(f"{gateway_url}/svc/svc%2Fx/y")
    assert r.status_code == 400 and "/svc/svc%2Fx/y" in r.text
