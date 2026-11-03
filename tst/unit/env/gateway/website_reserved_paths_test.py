"""A site's route never reaches its data plane or its env card, nor any path its card advertises for agentenv (its
state's among them), however the path is spelled, so an agent browsing the site can't read what grades it; agentenv
still reaches them through /svc/<name>/."""

import httpx
import pytest

from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.constants import EXT_WEBSITE_URI, WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import reserved_site_path, site_reserved_paths
from tst.unit.env.gateway.changes_test import _Changelog, _gateway, _http, _RealAsyncClient


@pytest.mark.parametrize(("path", "reserved"), [
    ("/agentenv", True), ("/agentenv/", True), ("/agentenv/data", True), ("//agentenv", True),
    ("/x/../agentenv", True), ("/./.well-known/agent-env.json", True), ("/.well-known/agent-env.json", True),
    ("/agentenv-help", False), ("/api/agentenv", False), ("/.well-known/other.json", False), ("/", False),
])
def test_the_data_plane_and_card_are_reserved_however_their_path_is_spelled(path, reserved):
    assert reserved_site_path(path) is reserved


@pytest.mark.asyncio
async def test_a_sites_route_refuses_its_reserved_paths_and_svc_still_reaches_them(monkeypatch):
    reached = []

    async def body():
        yield b"ok"

    def site(request):
        reached.append(request.url.path)
        return httpx.Response(200, content=body())

    transport = httpx.MockTransport(site)
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(_Changelog(), website_urls={"mail": "http://mail-website-backend:8000"},
                  rest_proxy_urls={"mail": "http://mail-website-backend:8000"})

    assert (await _http(gw, "GET", "/website/mail/.well-known/agent-env.json"))[0] == 404
    assert (await _http(gw, "POST", "/website/mail/agentenv", body=b"{}"))[0] == 404
    assert (await _http(gw, "POST", "/website/mail/%61gentenv", body=b"{}"))[0] == 404
    for spelled in ("public%2Fx/../agentenv", "public%2Fx/../.well-known/agent-env.json", "a%2F..%2Fagentenv",
                    "a/%2E%2E/agentenv", "x/..%2Fagentenv"):
        assert (await _http(gw, "POST", f"/website/mail/{spelled}", body=b"{}"))[0] == 404, spelled
    assert (await _http(gw, "GET", "/website/mail/inbox"))[0] == 200
    assert (await _http(gw, "GET", "/svc/mail/.well-known/agent-env.json"))[0] == 200
    assert reached == ["/.well-known/agent-env.json", "/inbox", "/.well-known/agent-env.json"]


async def _streamed():
    yield b"ok"


_CARD = {"name": "mail", "url": "/agentenv", "capabilities": {"extensions": [
    {"uri": EXT_WEBSITE_URI, "params": {"state": {"path": "/go"}, "methods": {
        "state": {"method": "GET", "endpoint": "/agentenv/ext/website/state"},
        "grade": {"method": "GET", "endpoint": "/grading/now"}}}},
    {"uri": "urn:example:hooks/v1", "params": {"endpoint": "/hooks"}},
    {"uri": "urn:example:root/v1", "params": {"endpoint": "/"}}]}}


def test_a_sites_reserved_paths_are_the_ones_its_card_advertises_each_a_path_of_its_own():
    assert site_reserved_paths(_CARD) == {"/agentenv", "/agentenv/ext/website/state", "/grading/now", "/hooks", "/go"}
    assert site_reserved_paths({"name": "x", "url": "http://elsewhere/agentenv"}) == frozenset()
    reserved = site_reserved_paths(_CARD)
    for path in ("/go", "/go/", "/x/../go", "/grading/now", "/hooks/a"):
        assert reserved_site_path(path, reserved), path
    for path in ("/gold", "/grading", "/hooksy", "/"):
        assert not reserved_site_path(path, reserved), path


@pytest.mark.asyncio
async def test_a_sites_route_refuses_what_its_card_advertises_reading_the_card_once(monkeypatch):
    reached = []

    def site(request):
        reached.append(request.url.path)
        if request.url.path == WELL_KNOWN_PATH:
            return httpx.Response(200, json=_CARD)
        return httpx.Response(200, content=_streamed())

    transport = httpx.MockTransport(site)
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(_Changelog(), website_urls={"mail": "http://mail-website-backend:8000"},
                  rest_proxy_urls={"mail": "http://mail-website-backend:8000"})

    for path in ("go?sid=default", "go", "x/../go", "grading/now", "hooks/a"):
        assert (await _http(gw, "GET", f"/website/mail/{path}"))[0] == 404, path
    assert (await _http(gw, "GET", "/website/mail/gold"))[0] == 200
    assert (await _http(gw, "GET", "/svc/mail/go?sid=default"))[0] == 200
    assert reached == [WELL_KNOWN_PATH, "/gold", "/go"]


@pytest.mark.asyncio
async def test_a_card_that_cant_be_read_is_read_again_and_one_that_isnt_served_is_not(monkeypatch):
    answers = {"mail": [httpx.Response(503), httpx.Response(200, json=_CARD)], "notes": [httpx.Response(404)]}
    reached = []

    def site(request):
        app = request.url.host
        reached.append((app, request.url.path))
        if request.url.path == WELL_KNOWN_PATH:
            return answers[app].pop(0)
        return httpx.Response(200, content=_streamed())

    transport = httpx.MockTransport(site)
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(_Changelog(), website_urls={"mail": "http://mail", "notes": "http://notes"})

    assert (await _http(gw, "GET", "/website/mail/go"))[0] == 200
    assert (await _http(gw, "GET", "/website/mail/go"))[0] == 404
    assert (await _http(gw, "GET", "/website/notes/go"))[0] == 200
    assert (await _http(gw, "GET", "/website/notes/go"))[0] == 200
    assert reached == [("mail", WELL_KNOWN_PATH), ("mail", "/go"), ("mail", WELL_KNOWN_PATH),
                       ("notes", WELL_KNOWN_PATH), ("notes", "/go"), ("notes", "/go")]


def test_an_sdk_sites_card_reserves_its_state_where_the_protocol_serves_it(tmp_path):
    from agentenv_protocol.client import find_extension_method
    from agentenv_protocol.website import STATE_PATH, WEBSITE_EXTENSION_URI, WEBSITE_STATE_OP, WebsiteEnvironment, website
    from starlette.testclient import TestClient

    (tmp_path / "index.html").write_text("<html></html>")

    @website(domains=["x.test"], static=str(tmp_path))
    class Site(WebsiteEnvironment):
        pass

    card = TestClient(Site().create_app()).get(WELL_KNOWN_PATH).json()
    assert EXT_WEBSITE_URI == WEBSITE_EXTENSION_URI
    reserved = site_reserved_paths(card)
    assert {STATE_PATH, find_extension_method(card, WEBSITE_EXTENSION_URI, WEBSITE_STATE_OP)["endpoint"]} <= reserved
    assert reserved_site_path(f"{STATE_PATH}", reserved) and not reserved_site_path("/", reserved)
