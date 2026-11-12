"""A live site on WebsiteEnvironment: a real site, such as github.com, whose server serves none of its pages, only the
card, with website/v1's live params, a data plane, ready and the account's session from its ``@session_state`` method;
and a site's ``Settings``, which a deploy hands it in ``WEBSITE_SETTINGS_ENV``."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel, SecretStr
from starlette.testclient import TestClient

from agentenv_protocol.website import (
    WEBSITE_ALLOW_HOSTS_PARAM,
    WEBSITE_EXTENSION_URI,
    WEBSITE_LIVE_PARAM,
    WEBSITE_SETTINGS_ENV,
    WebsiteEnvironment,
    session_state,
    website,
)


@website(domains=["github.com"], live=True, allow_hosts=["*.githubassets.com", "api.github.com"])
class GitHub(WebsiteEnvironment):
    class Settings(BaseModel):
        token: SecretStr
        org: str = "octo"

    @session_state
    async def account(self, user: str | None) -> dict:
        value = f"{user}:{self.settings.token.get_secret_value()}"
        return {"cookies": [{"name": "user_session", "value": value, "path": "/", "max_age": 60}], "local_storage": []}


@pytest.fixture
def github(monkeypatch) -> GitHub:
    monkeypatch.setenv(WEBSITE_SETTINGS_ENV, json.dumps({"token": "t0"}))
    return GitHub()


def _params(client: TestClient) -> dict:
    card = client.get("/.well-known/agent-env.json").json()
    return next(e for e in card["capabilities"]["extensions"] if e["uri"] == WEBSITE_EXTENSION_URI)["params"]


def test_a_live_sites_card_names_the_real_site_and_the_hosts_its_pages_reach_and_it_serves_no_pages(github) -> None:
    client = TestClient(github.create_app())

    params = _params(client)

    assert {key: params[key] for key in ("domains", WEBSITE_LIVE_PARAM, WEBSITE_ALLOW_HOSTS_PARAM)} == {
        "domains": ["github.com"], WEBSITE_LIVE_PARAM: True,
        WEBSITE_ALLOW_HOSTS_PARAM: ["*.githubassets.com", "api.github.com"]}
    assert not {"routes", "session_cookie", "session_path"} & params.keys()
    assert params["methods"]["session"]["endpoint"] == "/agentenv/ext/website/session"
    assert (client.get("/").status_code, client.get("/api/session").status_code) == (404, 404)


def test_a_live_sites_session_is_its_accounts_as_its_session_state_method_says(github) -> None:
    client = TestClient(github.create_app())

    state = client.post("/agentenv/ext/website/session", json={"user": "alex"}).json()

    assert state == {"cookies": [{"name": "user_session", "value": "alex:t0", "path": "/", "max_age": 60}],
                     "local_storage": []}


def test_a_sites_settings_arrive_from_the_deploy_and_a_secret_stays_one(github, monkeypatch) -> None:
    assert (github.settings.org, github.settings.token.get_secret_value()) == ("octo", "t0")
    assert "t0" not in repr(github.settings)
    monkeypatch.delenv(WEBSITE_SETTINGS_ENV)

    @website(domains=["shop.test"])
    class Shop(WebsiteEnvironment):
        pass

    assert Shop().settings.model_dump() == {}


def test_a_live_site_marking_no_session_state_method_cant_sign_a_browser_in() -> None:
    @website(domains=["github.com"], live=True)
    class Bare(WebsiteEnvironment):
        pass

    with pytest.raises(TypeError, match="marks no @session_state method"):
        asyncio.run(Bare()._website_session("alex"))


@pytest.mark.parametrize(("kwargs", "problem"), [
    ({"live": True}, "names the domains the real site is at"),
    ({"domains": ["github.com"], "live": True, "static": "dist/"}, "serves no pages"),
    ({"domains": ["github.com"], "allow_hosts": ["api.github.com"]}, "declare live=True"),
    ({"domains": ["github.com"], "live": True, "allow_hosts": "api.github.com"}, "not the string"),
    ({"domains": ["github.com"], "live": True, "allow_hosts": ["https://api.github.com"]}, "not a host name"),
    ({"domains": ["github.com"], "live": "yes"}, "True or False"),
], ids=["no-domains", "static", "allow-hosts-not-live", "allow-hosts-string", "allow-hosts-url", "live-not-bool"])
def test_website_refuses_a_live_declaration_that_cant_be_one(kwargs, problem) -> None:
    with pytest.raises(ValueError, match=problem):
        website(**kwargs)
