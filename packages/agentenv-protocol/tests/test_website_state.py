"""What a website env's task state is: WebsiteState, the SDK's website/v1 state, and website_state and website_sites
reading it through cards."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from agentenv_protocol import DataPart
from agentenv_protocol.agent_env_environment import get_data
from agentenv_protocol.client import find_extension_method
from agentenv_protocol.website import (
    STATE_PATH,
    WEBSITE_EXTENSION_URI,
    WEBSITE_STATE_OP,
    WebsiteEnvironment,
    WebsiteState,
    api,
    website,
    website_sites,
    website_state,
)

_GW = "http://gw"


def _site_card(name: str, *, state: bool = True) -> dict:
    methods = {"state": {"method": "GET", "endpoint": f"/svc/{name}/agentenv/ext/website/state"}} if state else {}
    return {"name": name, "url": f"/svc/{name}/agentenv", "capabilities": {"extensions": [
        {"uri": WEBSITE_EXTENSION_URI, "params": {"methods": methods}}]}}


def _routed(handler):
    real = httpx.AsyncClient
    return patch.object(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handler)))


@pytest.fixture
def shop(tmp_path: Path) -> WebsiteEnvironment:
    (tmp_path / "index.html").write_text("<html></html>")

    @website(domains=["shop.test"], static=str(tmp_path))
    class Shop(WebsiteEnvironment):
        @api.post("/api/orders")
        def place(self, order: dict) -> dict:
            return self.store.insert("orders", order)

    env = Shop()
    env.store.load({"orders": [{"id": 1, "sku": "pen"}]})
    env.store.mark_initial()
    return env


def test_a_website_state_ignores_fields_it_doesnt_know_and_refuses_a_state_that_isnt_an_object() -> None:
    assert WebsiteState.model_validate({"current_state": {"a": 1}, "seen_at": "now"}).model_dump() == {
        "initial_state": None, "current_state": {"a": 1}, "state_diff": None}
    with pytest.raises(ValidationError):
        WebsiteState.model_validate({"current_state": [1]})


def test_an_sdk_site_advertises_state_and_answers_its_stores_state_there_and_at_the_path_it_names(
        shop: WebsiteEnvironment) -> None:
    client = TestClient(shop.create_app())
    card = client.get("/.well-known/agent-env.json").json()
    op = find_extension_method(card, WEBSITE_EXTENSION_URI, WEBSITE_STATE_OP)
    params = next(e for e in card["capabilities"]["extensions"] if e["uri"] == WEBSITE_EXTENSION_URI)["params"]
    assert (op["method"], op["endpoint"], params["state"]) == ("GET", "/agentenv/ext/website/state", {"path": STATE_PATH})
    client.post("/api/orders", json={"sku": "mug"})
    answer = client.get(op["endpoint"], params={"sid": "default"}).json()
    assert answer == client.get(STATE_PATH).json() == {**shop.store.state(), "changed_sessions": []}
    assert answer["state_diff"]["orders"]["added"][0]["sku"] == "mug"


def test_an_sdk_sites_state_op_answers_the_session_its_sid_names(shop: WebsiteEnvironment) -> None:
    client = TestClient(shop.create_app())
    client.post("/api/orders", params={"sid": "task-1"}, json={"sku": "mug"})
    named = client.get("/agentenv/ext/website/state", params={"sid": "task-1"}).json()
    assert named == client.get(STATE_PATH, params={"sid": "task-1"}).json()
    assert named["state_diff"]["orders"]["added"][0]["sku"] == "mug" and named["changed_sessions"] == ["task-1"]
    default = client.get("/agentenv/ext/website/state").json()
    assert WebsiteState.model_validate(default).state_diff == shop.store.state()["state_diff"]
    assert default["current_state"]["orders"] == [{"id": 1, "sku": "pen"}]


def test_a_site_with_data_hooks_of_its_own_advertises_no_state(tmp_path: Path) -> None:
    @website(domains=["own.test"])
    class Own(WebsiteEnvironment):
        @get_data
        def export(self) -> list:
            return [DataPart(data={"mine": True})]

    client = TestClient(Own().create_app())
    card = client.get("/.well-known/agent-env.json").json()
    params = next(e for e in card["capabilities"]["extensions"] if e["uri"] == WEBSITE_EXTENSION_URI)["params"]
    assert find_extension_method(card, WEBSITE_EXTENSION_URI, WEBSITE_STATE_OP) is None and "state" not in params
    assert client.get("/agentenv/ext/website/state").status_code == 404


def test_website_state_asks_the_cards_state_of_the_session_with_the_headers_given() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.url.params), request.headers.get("authorization")))
        return httpx.Response(200, json={"initial_state": {}, "current_state": {"n": 1}, "state_diff": {"n": {}}})

    with _routed(handler):
        state = asyncio.run(website_state(_GW, _site_card("mail"), sid="s1", headers={"Authorization": "Bearer t"}))
        assert asyncio.run(website_state(_GW, _site_card("mail", state=False))) is None
    assert state == WebsiteState(initial_state={}, current_state={"n": 1}, state_diff={"n": {}})
    assert seen == [("/svc/mail/agentenv/ext/website/state", {"sid": "s1"}, "Bearer t")]


def test_an_answer_that_isnt_a_website_state_is_refused() -> None:
    with _routed(lambda request: httpx.Response(200, json={"current_state": "full"})):
        with pytest.raises(ValueError):
            asyncio.run(website_state(_GW, _site_card("mail")))


def test_website_sites_are_a_single_sites_own_card_or_a_composed_cards_website_children() -> None:
    mail, calendar = _site_card("mail"), _site_card("calendar", state=False)
    mcp = {"name": "mail", "url": "/svc/mcp-mail/agentenv", "capabilities": {"extensions": []}}
    composed = {"name": "gw", "children_environments": [mail, mcp, calendar, "junk"]}
    assert [(site.name, site.url) for site in website_sites(composed)] == [
        ("mail", "/svc/mail/agentenv"), ("calendar", "/svc/calendar/agentenv")]
    assert [site.card for site in website_sites(mail)] == [mail]
    assert website_sites({"name": "plain", "children_environments": [mcp]}) == []
