"""Live websites: a live site serves none of its pages, declares live and allow_hosts on its card, and its @session
method's storage state comes back from website/v1's session in browser/v1's shapes, given its domain; a session is
checked without naming a cookie's value."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from agentenv_protocol import DataPart, add_data, get_data, reset_data
from agentenv_protocol.client import find_extension
from agentenv_protocol.website import (WEBSITE_EXTENSION_URI, WebsiteEnvironment, api, ready_check, session,
                                       session_state, website)

SESSION = "/agentenv/ext/website/session"


@website(domains=["github.com"], live=True, allow_hosts=["*.githubusercontent.com", "api.github.com"])
class GitHub(WebsiteEnvironment):
    """A live site: its data hooks would call GitHub's API with the token setting."""

    def __init__(self) -> None:
        super().__init__()
        self.repos: list = []

    @ready_check
    def has_token(self) -> bool:
        return bool(self.setting("GITHUB_TOKEN"))

    @session
    async def sign_in(self, user: str | None) -> list[dict]:
        return [{"name": "user_session", "value": f"{self.setting('GITHUB_TOKEN')}:{user}", "secure": True,
                 "httpOnly": True, "expires": 1_900_000_000},
                {"name": "logged_in", "value": "yes", "domain": ".github.com", "max_age": 3600}]

    @reset_data
    async def reset(self) -> None:
        self.repos = []

    @add_data
    async def load(self, parts: list) -> None:
        self.repos += [repo for part in parts for repo in part.data["repos"]]

    @get_data
    async def export(self) -> list:
        return [DataPart(data={"repos": self.repos})]


def ready(client: TestClient) -> dict:
    card = client.get("/.well-known/agent-env.json").json()
    return client.get(find_extension(card, WEBSITE_EXTENSION_URI)["params"]["methods"]["ready"]["endpoint"]).json()


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    return TestClient(GitHub().create_app())


def test_a_live_site_serves_no_pages_only_its_data_plane_and_card(client):
    for path in ("/", "/login", "/api/session"):
        assert client.get(path).status_code == 404, path
    card = client.get("/.well-known/agent-env.json").json()
    params = find_extension(card, WEBSITE_EXTENSION_URI)["params"]
    assert params["live"] is True and params["signs_in"] is True
    assert params["domains"] == ["github.com"] and params["routes"] == []
    assert params["allow_hosts"] == ["*.githubusercontent.com", "api.github.com"]
    assert "session_cookie" not in params and "session_path" not in params
    assert ready(client)["ready"] is True


def test_a_live_sites_readiness_says_whether_its_account_works(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert ready(TestClient(GitHub().create_app()))["ready"] is False


def test_the_session_operation_returns_the_session_methods_cookies_for_the_sites_domain(client):
    assert client.post(SESSION, json={"user": "octocat"}).json() == {"cookies": [
        {"name": "user_session", "value": "ghp_secret:octocat", "domain": "github.com", "expires": 1_900_000_000,
         "httpOnly": True, "secure": True},
        {"name": "logged_in", "value": "yes", "domain": ".github.com", "max_age": 3600, "httpOnly": False,
         "secure": False}], "local_storage": []}


def test_a_live_site_without_a_session_method_says_so():
    @website(domains=["example.com"], live=True)
    class Plain(WebsiteEnvironment):
        pass

    client = TestClient(Plain().create_app())
    params = find_extension(client.get("/.well-known/agent-env.json").json(), WEBSITE_EXTENSION_URI)["params"]
    assert params["signs_in"] is False
    answer = client.post(SESSION, json={})
    assert answer.status_code == 501 and answer.json()["error"]["code"] == "no_session"


def test_a_session_method_returning_a_bad_cookie_is_refused_without_its_value():
    @website(domains=["example.com"], live=True)
    class Broken(WebsiteEnvironment):
        @session
        def sign_in(self, user):
            return {"cookies": [{"name": "sid", "value": "hunter2", "sameSite": "sometimes"}], "local_storage": []}

    answer = TestClient(Broken().create_app()).post(SESSION, json={})
    assert answer.status_code == 500 and "cookies[0].sameSite" in answer.text and "hunter2" not in answer.text


def test_a_synthetic_site_keeps_its_signed_session_and_may_override_it():
    @website(domains=["shop.example"])
    class Shop(WebsiteEnvironment):
        pass

    @website(domains=["*.persona.example"])
    class Persona(WebsiteEnvironment):
        @session
        def as_persona(self, user):
            return {"cookies": [{"name": "persona", "value": user or "guest"}],
                    "local_storage": [{"origin": "https://app.persona.example",
                                       "localStorage": [{"name": "theme", "value": "dark"}]}]}

    signed = TestClient(Shop().create_app()).post(SESSION, json={"user": "ann"}).json()
    assert signed["cookies"][0]["name"] == "agentenv_session_Shop"
    persona = TestClient(Persona().create_app()).post(SESSION, json={}).json()
    assert persona == {"cookies": [{"name": "persona", "value": "guest", "domain": ".persona.example",
                                    "httpOnly": False, "secure": False}],
                       "local_storage": [{"origin": "https://app.persona.example",
                                          "localStorage": [{"name": "theme", "value": "dark"}]}]}


@pytest.mark.parametrize("declared, problem", [
    (dict(live=True), "names its domains"),
    (dict(domains=["a.example"], live=True, static="dist"), "takes no static"),
    (dict(domains=["a.example"], allow_hosts=["cdn.example"]), "allow_hosts is for a live site"),
    (dict(domains=["https://a.example"]), "not a host name"),
    (dict(domains=["a.example"], live=True, allow_hosts=["*"]), "not a host name"),
    (dict(domains="a.example"), "not the string"),
])
def test_a_live_declaration_is_checked(declared, problem):
    with pytest.raises(ValueError, match=problem):
        website(**declared)


def test_a_live_site_serves_no_api():
    with pytest.raises(ValueError, match="serves no @api route"):
        @website(domains=["a.example"], live=True)
        class WithApi(WebsiteEnvironment):
            @api.get("/api/x")
            def x(self) -> dict:
                return {}


def test_a_session_state_is_checked_by_index_and_field_never_by_value():
    with pytest.raises(ValueError) as e:
        session_state([{"name": "a", "value": "s3cret", "domain": "a.example", "sameSite": "sometimes"}, "x",
                       {"name": "b", "value": "s3cret"}], [{"origin": "not an origin", "localStorage": []}])
    message = str(e.value)
    assert "cookies[0].sameSite" in message and "cookies[1] is not an object" in message
    assert "cookies[2].cookie" in message and "local_storage[0.origin]" in message and "s3cret" not in message
    with pytest.raises(ValueError, match="list"):
        session_state({"name": "a"}, [])
    assert session_state([{"name": "b", "value": "v", "url": "https://b.example/"}], [], ["a.example"])["cookies"] == [
        {"name": "b", "value": "v", "url": "https://b.example/", "httpOnly": False, "secure": False}]
