"""A site's route never reaches its data plane or its env card, however the path is spelled, so an agent browsing the
site can't read what grades it; agentenv still reaches both through /svc/<name>/."""

import httpx
import pytest

from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.gateway import reserved_site_path
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
    assert (await _http(gw, "GET", "/website/mail/inbox"))[0] == 200
    assert (await _http(gw, "GET", "/svc/mail/.well-known/agent-env.json"))[0] == 200
    assert reached == ["/inbox", "/.well-known/agent-env.json"]
