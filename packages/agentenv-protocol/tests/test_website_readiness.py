"""How a website env's health is determined: WebsiteReadiness, and website_readiness and env_readiness reading it
through cards."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from agentenv_protocol import ExtensionCallError
from agentenv_protocol.website import (
    WEBSITE_EXTENSION_URI,
    WEBSITE_READY_OP,
    WebsiteEnvironment,
    WebsiteReadiness,
    env_readiness,
    ready_check,
    website,
    website_readiness,
)

_GW = "http://gw"


def _site_card(name: str, *, ready: bool = True) -> dict:
    methods = {"ready": {"method": "GET", "endpoint": f"/svc/{name}/agentenv/ext/website/ready"}} if ready else {}
    return {"name": name, "capabilities": {"extensions": [
        {"uri": WEBSITE_EXTENSION_URI, "params": {"health": "/api/health", "methods": methods}}]}}


def _routed(handler):
    """Every httpx.AsyncClient the helpers open answers through ``handler(request) -> httpx.Response``."""
    real = httpx.AsyncClient
    return patch.object(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handler)))


def test_a_website_readiness_names_no_problem_unless_told_and_ignores_fields_it_doesnt_know() -> None:
    assert WebsiteReadiness.model_validate({"ready": True}) == WebsiteReadiness(ready=True, problems=[])
    assert WebsiteReadiness.model_validate({"ready": False, "problems": ["seeded"], "since": "boot"}).model_dump(
        exclude_none=True) == {"ready": False, "problems": ["seeded"]}
    assert WebsiteReadiness.model_validate({"ready": False, "waited_s": 3, "timeout_s": 60}).model_dump(
        exclude_none=True) == {"ready": False, "problems": [], "waited_s": 3.0, "timeout_s": 60.0}


@pytest.mark.parametrize("answer", [{}, {"ready": "soon"}, {"ready": "true"}, {"ready": 1}, {"ready": True, "problems": "none"},
                                    {"ready": False, "problems": [1]}, ["ready"], None])
def test_an_answer_without_a_boolean_ready_and_a_list_of_problems_is_no_website_readiness(answer: object) -> None:
    with pytest.raises(ValidationError):
        WebsiteReadiness.model_validate(answer)


def test_the_ready_op_is_the_one_the_sdk_advertises(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    (tmp_path / "index.html").write_text("<html></html>")

    @website(domains=["x.test"], static=str(tmp_path))
    class Site(WebsiteEnvironment):
        pass

    card = TestClient(Site().create_app()).get("/.well-known/agent-env.json").json()
    ext = next(e for e in card["capabilities"]["extensions"] if e["uri"] == WEBSITE_EXTENSION_URI)
    assert WEBSITE_READY_OP == "ready" and WEBSITE_READY_OP in ext["params"]["methods"]


@pytest.mark.asyncio
async def test_website_readiness_reads_an_sdk_sites_ready_through_its_card(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html></html>")

    @website(domains=["x.test"], static=str(tmp_path))
    class Site(WebsiteEnvironment):
        up = False

        @ready_check
        def database_up(self) -> bool:
            return self.up

    env = Site()
    app, real = env.create_app(), httpx.AsyncClient
    async with real(transport=httpx.ASGITransport(app=app), base_url="http://site") as client:
        card = (await client.get("/.well-known/agent-env.json")).json()
    with patch.object(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.ASGITransport(app=app))):
        waiting = await website_readiness("http://site", card)
        assert (waiting.ready, waiting.problems, waiting.timeout_s) == (False, ["@ready_check database_up returned False"], 300)
        env.up = True
        assert (await website_readiness("http://site", card)).ready is True


@pytest.mark.asyncio
async def test_website_readiness_is_none_for_a_card_without_ready_and_asks_nothing() -> None:
    def site(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"asked {request.url}")

    with _routed(site):
        assert await website_readiness(_GW, _site_card("shop", ready=False)) is None
        assert await website_readiness(_GW, {"name": "mcp"}) is None


@pytest.mark.asyncio
async def test_website_readiness_sends_the_headers_it_is_given() -> None:
    seen = []

    def site(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"ready": True, "problems": []})

    with _routed(site):
        assert (await website_readiness(_GW, _site_card("mail"), headers={"Authorization": "Bearer t"})).ready
    assert seen == ["Bearer t"]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [{"ok": True}, {"ready": "yes"}, ["ready"]])
async def test_website_readiness_refuses_an_answer_that_isnt_a_website_readiness(answer: object) -> None:
    with _routed(lambda request: httpx.Response(200, json=answer)), pytest.raises(ValidationError):
        await website_readiness(_GW, _site_card("mail"))


@pytest.mark.asyncio
async def test_website_readiness_raises_when_the_site_cant_be_asked() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _routed(down), pytest.raises(httpx.ConnectError):
        await website_readiness(_GW, _site_card("mail"))
    with _routed(lambda request: httpx.Response(502, text="bad gateway")), pytest.raises(ExtensionCallError):
        await website_readiness(_GW, _site_card("mail"))
    with _routed(lambda request: httpx.Response(200, text="<html>")), pytest.raises(ValueError):
        await website_readiness(_GW, _site_card("mail"))


def _composed(*children: dict) -> dict:
    return {"name": "gateway", "capabilities": {"extensions": []}, "children_environments": list(children)}


@pytest.mark.asyncio
async def test_env_readiness_asks_every_site_of_a_composed_card_and_names_each_ones_problems() -> None:
    asked = []

    def gateway(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        if request.url.path == "/svc/mail/agentenv/ext/website/ready":
            return httpx.Response(200, json={"ready": True, "problems": []})
        if request.url.path == "/svc/shop/agentenv/ext/website/ready":
            return httpx.Response(200, json={"ready": False, "problems": ["seeded", "indexed"]})
        if request.url.path == "/svc/docs/agentenv/ext/website/ready":
            return httpx.Response(200, json={"ready": False})
        raise AssertionError(f"asked {request.url}")

    card = _composed(_site_card("mail"), _site_card("shop"), _site_card("docs"), _site_card("old", ready=False),
                     {"name": "mcp-items", "capabilities": {"tools": []}})
    with _routed(gateway):
        readiness = await env_readiness(_GW, card)
    assert readiness == WebsiteReadiness(ready=False, problems=[
        "shop: seeded", "shop: indexed", "docs: ready says not ready"])
    assert sorted(asked) == ["/svc/docs/agentenv/ext/website/ready", "/svc/mail/agentenv/ext/website/ready",
                             "/svc/shop/agentenv/ext/website/ready"]


@pytest.mark.asyncio
async def test_env_readiness_counts_a_site_that_cant_be_asked_or_answers_no_readiness_as_a_problem() -> None:
    def gateway(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/svc/mail/"):
            raise httpx.ConnectError("refused", request=request)
        if request.url.path.startswith("/svc/shop/"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json={"ready": True})

    with _routed(gateway):
        readiness = await env_readiness(_GW, _composed(_site_card("mail"), _site_card("shop"), _site_card("docs")))
    assert not readiness.ready
    assert [p.split(" could not be read: ")[0] for p in readiness.problems] == ["mail: ready", "shop: ready"]
    assert "ConnectError: refused" in readiness.problems[0] and "ValidationError" in readiness.problems[1]


@pytest.mark.asyncio
async def test_env_readiness_asks_the_sites_at_once() -> None:
    arrived, both = [], asyncio.Event()

    async def gateway(request: httpx.Request) -> httpx.Response:
        arrived.append(request.url.path)
        if len(arrived) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), timeout=5)
        return httpx.Response(200, json={"ready": True, "problems": []})

    with _routed(gateway):
        assert await env_readiness(_GW, _composed(_site_card("mail"), _site_card("shop"))) == WebsiteReadiness(ready=True)


@pytest.mark.asyncio
async def test_env_readiness_of_a_single_sites_card_is_that_sites_own() -> None:
    card = {"name": "mail", "capabilities": {"extensions": [{"uri": WEBSITE_EXTENSION_URI, "params": {
        "methods": {"ready": {"method": "GET", "endpoint": "/agentenv/ext/website/ready"}}}}]}}

    with _routed(lambda request: httpx.Response(200, json={"ready": False, "problems": ["seeded"]})):
        assert await env_readiness("http://mail", card) == WebsiteReadiness(ready=False, problems=["seeded"])


@pytest.mark.asyncio
async def test_env_readiness_of_an_env_with_no_site_serving_ready_is_ready_with_no_problems() -> None:
    def gateway(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"asked {request.url}")

    with _routed(gateway):
        assert await env_readiness(_GW, _composed({"name": "mcp-items"}, _site_card("old", ready=False))) == \
            WebsiteReadiness(ready=True, problems=[])
        assert await env_readiness(_GW, {"name": "computer"}) == WebsiteReadiness(ready=True)
