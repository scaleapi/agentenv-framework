"""traffic/v1 on the gateway: a deploy that captures its websites' traffic mounts their pcap files into the gateway,
which lists them oldest first and streams each, only to a caller carrying its control token, which it takes off every
request before anything is proxied, so no site ever reads it; a site's route never reaches the captures. A gateway
without a capture folder serves and advertises none of it."""

from __future__ import annotations

import os

import httpx
import pytest
from agentenv_protocol import traffic as protocol_traffic

from agent_env.env.gateway import constants as gw_constants
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.constants import EXT_TRAFFIC_URI, GATEWAY_EXTENSIONS, PCAP_MEDIA_TYPE, WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import Gateway

TOKEN = "control-secret"
_RealAsyncClient = httpx.AsyncClient
_PCAP_HEADER = bytes.fromhex("d4c3b2a1020004000000000000000000ffff000001000000")


@pytest.fixture
def traffic_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    folder = tmp_path / "traffic"
    folder.mkdir()
    return folder


def _gateway(traffic_dir=None, control_token=TOKEN, **kw) -> Gateway:
    return Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                   traffic_dir=str(traffic_dir) if traffic_dir else None, control_token=control_token, **kw)


def _write(folder, name: str, body: bytes, mtime: int) -> None:
    (folder / name).write_bytes(body)
    os.utime(folder / name, (mtime, mtime))


async def _get(gw: Gateway, path: str, headers=None, params=None) -> httpx.Response:
    transport = httpx.ASGITransport(app=gw._asgi_app())
    async with _RealAsyncClient(transport=transport, base_url="http://gateway") as client:
        return await client.get(path, headers=headers, params=params)


def _bearer(token: str = TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_list_names_each_capture_by_its_container_oldest_first_and_get_streams_it(traffic_dir):
    _write(traffic_dir, "notes-website-backend.pcap1", _PCAP_HEADER + b"second", 2000)
    _write(traffic_dir, "notes-website-backend.pcap0", _PCAP_HEADER + b"first", 1000)
    _write(traffic_dir, "notes-website-frontend.pcap0", _PCAP_HEADER, 1500)
    _write(traffic_dir, "stray.txt", b"not a capture", 1000)
    gw = _gateway(traffic_dir)

    listing = (await _get(gw, "/traffic/captures", headers=_bearer())).json()
    got = await _get(gw, "/traffic/capture", headers=_bearer(), params={"name": "notes-website-backend.pcap0"})

    assert listing["format"] == "pcap"
    assert [(c["name"], c["container"], c["bytes"]) for c in listing["captures"]] == [
        ("notes-website-backend.pcap0", "notes-website-backend", len(_PCAP_HEADER) + 5),
        ("notes-website-backend.pcap1", "notes-website-backend", len(_PCAP_HEADER) + 6),
        ("notes-website-frontend.pcap0", "notes-website-frontend", len(_PCAP_HEADER)),
    ]
    assert listing["captures"][0]["modified_at"].startswith("1970-01-01T00:16:40")
    assert (got.status_code, got.headers["content-type"], got.content) == (200, PCAP_MEDIA_TYPE, _PCAP_HEADER + b"first")


@pytest.mark.asyncio
@pytest.mark.parametrize("name, status", [
    ("../trajectory.jsonl", 400), ("notes-website-backend.pcap0/../../x", 400), ("stray.txt", 400),
    (".pcap0", 400), ("", 400), ("gone-website-backend.pcap0", 404),
])
async def test_get_serves_only_a_capture_file_of_the_folder(traffic_dir, name, status):
    (traffic_dir / "stray.txt").write_bytes(b"x")
    response = await _get(_gateway(traffic_dir), "/traffic/capture", headers=_bearer(), params={"name": name})
    assert response.status_code == status, response.text


@pytest.mark.asyncio
async def test_an_empty_or_missing_folder_lists_no_captures(traffic_dir, tmp_path):
    assert (await _get(_gateway(traffic_dir), "/traffic/captures", headers=_bearer())).json()["captures"] == []
    missing = _gateway(tmp_path / "never-mounted")
    assert (await _get(missing, "/traffic/captures", headers=_bearer())).json()["captures"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [None, _bearer("guess"), {"Authorization": TOKEN}, {"AgentEnv-Role-Key": TOKEN}],
                         ids=["none", "wrong", "not-bearer", "role-key-header"])
async def test_each_traffic_route_refuses_a_caller_without_the_control_token(traffic_dir, headers):
    _write(traffic_dir, "notes-website-backend.pcap0", _PCAP_HEADER, 1000)
    gw = _gateway(traffic_dir)
    listed = await _get(gw, "/traffic/captures", headers=headers)
    got = await _get(gw, "/traffic/capture", headers=headers, params={"name": "notes-website-backend.pcap0"})
    assert (listed.status_code, got.status_code) == (401, 401)
    assert listed.json()["error"]["code"] == "unauthorized" and _PCAP_HEADER not in got.content


@pytest.mark.asyncio
async def test_the_inner_app_alone_refuses_the_traffic_routes_of_a_gateway_with_a_token(traffic_dir):
    """Only the served app takes the token off a request and vouches for it, so the routes refuse anything else."""
    transport = httpx.ASGITransport(app=_gateway(traffic_dir)._mcp.streamable_http_app())
    async with _RealAsyncClient(transport=transport, base_url="http://gateway") as client:
        assert (await client.get("/traffic/captures", headers=_bearer())).status_code == 401


@pytest.mark.asyncio
async def test_a_gateway_with_no_token_serves_its_captures_to_anyone(traffic_dir):
    assert (await _get(_gateway(traffic_dir, control_token=None), "/traffic/captures")).status_code == 200


@pytest.mark.asyncio
async def test_the_card_advertises_traffic_only_with_a_capture_folder(traffic_dir):
    capturing = (await _get(_gateway(traffic_dir), "/.well-known/agent-env.json")).json()
    plain = _gateway(None, control_token=None)
    card = (await _get(plain, "/.well-known/agent-env.json")).json()

    [traffic] = [e for e in capturing["capabilities"]["extensions"] if e["uri"] == EXT_TRAFFIC_URI]
    assert {name: (op["method"], op["endpoint"]) for name, op in traffic["params"]["methods"].items()} == {
        "list": ("GET", "/traffic/captures"), "get": ("GET", "/traffic/capture")}
    assert plain._gateway_extensions() is GATEWAY_EXTENSIONS
    assert EXT_TRAFFIC_URI not in [e["uri"] for e in card["capabilities"]["extensions"]]
    assert (await _get(plain, "/traffic/captures")).status_code == 404


@pytest.mark.asyncio
async def test_the_control_token_never_reaches_a_site_and_a_sites_route_never_reaches_the_captures(traffic_dir,
                                                                                                    monkeypatch):
    _write(traffic_dir, "notes-website-backend.pcap0", _PCAP_HEADER, 1000)
    seen: list[httpx.Request] = []

    async def body():
        yield b"the site's own page"

    def site(request: httpx.Request) -> httpx.Response:
        if request.url.path == WELL_KNOWN_PATH:
            return httpx.Response(404)
        seen.append(request)
        return httpx.Response(200, content=body())

    transport = httpx.MockTransport(site)
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(traffic_dir, website_urls={"notes": "http://notes-website-backend:8000"},
                  rest_proxy_urls={"notes": "http://notes-website-backend:8000"})

    for path in ("/website/notes/traffic/captures", "/website/notes/traffic/capture", "/svc/notes/api/x"):
        response = await _get(gw, path, headers={**_bearer(), "Authorization-Extra": "kept"},
                              params={"name": "notes-website-backend.pcap0"})
        assert (response.status_code, response.content) == (200, b"the site's own page"), path
    assert [request.url.path for request in seen] == ["/traffic/captures", "/traffic/capture", "/api/x"]
    assert all("authorization" not in request.headers and request.headers["authorization-extra"] == "kept"
               for request in seen)


@pytest.mark.asyncio
async def test_a_header_that_isnt_the_token_still_reaches_the_site(traffic_dir, monkeypatch):
    seen: list[httpx.Request] = []

    async def body():
        yield b"{}"

    def site(request: httpx.Request) -> httpx.Response:
        if request.url.path == WELL_KNOWN_PATH:
            return httpx.Response(404)
        seen.append(request)
        return httpx.Response(200, content=body())

    transport = httpx.MockTransport(site)
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(traffic_dir, website_urls={"notes": "http://notes"})
    for authorization in ("Bearer user-token", f"Bearer {TOKEN}"):
        await _get(gw, "/website/notes/api", headers={"Authorization": authorization})
    assert [request.headers.get("authorization") for request in seen] == ["Bearer user-token", None]


@pytest.mark.asyncio
async def test_a_link_in_the_folder_is_neither_listed_nor_served(traffic_dir, tmp_path):
    secret = tmp_path / "secret"
    secret.write_bytes(b"not traffic")
    (traffic_dir / "notes-website-backend.pcap0").symlink_to(secret)
    (traffic_dir / "notes-website-backend.pcap1").mkdir()
    gw = _gateway(traffic_dir)
    assert (await _get(gw, "/traffic/captures", headers=_bearer())).json()["captures"] == []
    for name in ("notes-website-backend.pcap0", "notes-website-backend.pcap1"):
        response = await _get(gw, "/traffic/capture", headers=_bearer(), params={"name": name})
        assert response.status_code == 404 and b"not traffic" not in response.content


def test_the_gateways_traffic_constants_mirror_the_protocols():
    """The gateway's container can't import the protocol, so it keeps its own copies of traffic/v1's names."""
    assert (gw_constants.EXT_TRAFFIC_URI, gw_constants.TRAFFIC_LIST_OP, gw_constants.TRAFFIC_GET_OP,
            gw_constants.PCAP_MEDIA_TYPE, gw_constants.TRAFFIC_EXTENSION["params"]["format"]) == (
        protocol_traffic.TRAFFIC_EXTENSION_URI, protocol_traffic.TRAFFIC_LIST_OP, protocol_traffic.TRAFFIC_GET_OP,
        protocol_traffic.PCAP_MEDIA_TYPE, protocol_traffic.TRAFFIC_FORMAT)
    assert protocol_traffic.serves_traffic({"capabilities": {"extensions": [gw_constants.TRAFFIC_EXTENSION]}})
