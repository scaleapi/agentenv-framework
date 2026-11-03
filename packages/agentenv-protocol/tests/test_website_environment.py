"""WebsiteEnvironment as a site's server: static pages, health, the card and its OpenAPI document, the default store
and its data plane, sessions, the universe's clock, and a mounted app."""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from agentenv_protocol import DataPart, get_data
from agentenv_protocol.website import (
    SESSION_COOKIE,
    UNIVERSE_NOW_HEADER,
    WEBSITE_EXTENSION_URI,
    WEBSITE_READY_OP,
    ApiContext,
    Collections,
    WebsiteEnvironment,
    WebsiteReadiness,
    api,
    ready_check,
    website,
)
from agentenv_protocol.website import clock as clock_module
from agentenv_protocol.website.clock import parse_time

RPC = "/agentenv"
COOKIE = "agentenv_session_Shop"


def rpc(client: TestClient, method: str, params: dict | None = None) -> dict:
    body = client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).json()
    assert "error" not in body, body
    return body["result"]


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<html><head><title>Shop</title></head><body>home</body></html>")
    (root / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("outside")
    return root


def site(dist: Path, **declared) -> type[WebsiteEnvironment]:
    @website(domains=["shop.example", "www.shop.example"], static=str(dist), **declared)
    class Shop(WebsiteEnvironment):
        @api.get("/api/orders", name="list_orders")
        def list_orders(self, ctx: ApiContext) -> list:
            """List the signed-in user's orders."""
            return self.store.find("orders", user=ctx.user)

        @api.post("/api/orders", name="place_order", roles=["default", "agent"])
        def place_order(self, order: dict, ctx: ApiContext) -> dict:
            return self.store.insert("orders", {**order, "user": ctx.user, "at": self.now().isoformat()})

        @api.post("/api/sign-in")
        def sign_in(self, body: dict, ctx: ApiContext) -> dict:
            ctx.sign_in(body["user"])
            return {"user": ctx.user}

    return Shop


SEED = {"current_user": "alice", "users": [{"id": "alice"}, {"id": "bob"}], "orders": [{"id": 1, "user": "bob"}]}


@pytest.fixture
def shop(dist: Path) -> WebsiteEnvironment:
    env = site(dist)()
    env.store.load(SEED)
    return env


@pytest.fixture
def client(shop: WebsiteEnvironment) -> TestClient:
    return TestClient(shop.create_app())


def test_static_files_and_index_fallback_answer_for_any_host(client: TestClient) -> None:
    for host in ("testserver", "shop.example", "10.0.0.7:8000", "anything.invalid"):
        page = client.get("/", headers={"Host": host})
        assert page.status_code == 200 and "home" in page.text and page.headers["content-type"].startswith("text/html")
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert "home" in client.get("/orders/42/edit").text
    assert client.get("/assets/missing.js").status_code == 404
    assert client.get("/api/nothing-here").json() == {"error": "not found", "code": "not_found"}
    assert client.get("/../secret.txt").status_code == 404
    assert client.post("/orders").status_code == 404


def test_behind_a_forwarded_prefix_the_site_serves_its_own_paths_and_the_fallback_still_guards_reserved_ones(
        client: TestClient) -> None:
    behind = {"X-Forwarded-Prefix": "/website/shop"}
    assert "home" in client.get("/", headers=behind).text
    assert client.get("/assets/app.js", headers=behind).text == "console.log(1)"
    assert "home" in client.get("/orders/42/edit", headers=behind).text
    assert client.get("/api/orders", headers=behind).json() == []
    assert client.get("/.well-known/agent-env.json", headers=behind).json()["name"] == "Shop"
    assert client.get("/website/shop/assets/app.js", headers=behind).status_code == 404
    assert client.post("/agentenv/x", headers=behind).status_code in (404, 405)


def test_a_link_out_of_the_static_folder_is_never_served(dist: Path) -> None:
    secret = dist.parent / "secret.txt"
    (dist / "leak.txt").symlink_to(secret)
    (dist / "linked").mkdir()
    (dist / "linked" / "index.html").symlink_to(secret)
    client = TestClient(site(dist)().create_app())
    assert "outside" not in client.get("/leak.txt").text
    assert "outside" not in client.get("/linked").text
    assert "outside" not in client.get("/linked/").text


def test_the_fallback_never_shadows_api_health_or_reserved_paths(client: TestClient) -> None:
    assert client.get("/api/orders").json() == []
    assert client.put("/api/orders").status_code == 405
    assert client.get("/api/health").json() == {"status": "ok"}
    for path in ("/agentenv", "/agentenv/anything", "/agentenv/ext/website/nope", "/agentenv/../agentenv/x"):
        assert client.get(path).status_code in (404, 405), path
    assert client.get("/.well-known/agent-env.json").json()["name"] == "Shop"


def test_health_waits_for_every_ready_check(dist: Path) -> None:
    class Slow(site(dist)):
        ready = False

        @ready_check
        def database_up(self) -> bool:
            return self.ready

        @ready_check
        async def cache_warm(self) -> bool:
            return True

    env = Slow()
    client = TestClient(env.create_app())
    waiting = client.get("/api/health")
    assert waiting.status_code == 503 and waiting.json() == {"status": "starting", "waiting_for": ["database_up"]}
    env.ready = True
    assert client.get("/api/health").status_code == 200


def test_the_cards_website_v1_ready_says_whether_the_site_is_usable_by_its_ready_checks(dist: Path) -> None:
    class Slow(site(dist)):
        ready = False

        @ready_check
        def database_up(self) -> bool:
            return self.ready

    env = Slow()
    client = TestClient(env.create_app())
    card = client.get("/.well-known/agent-env.json").json()
    ready = _extension(card, WEBSITE_EXTENSION_URI)["params"]["methods"][WEBSITE_READY_OP]
    assert (ready["method"], ready["endpoint"]) == ("GET", "/agentenv/ext/website/ready")
    assert client.get(ready["endpoint"]).json() == {"ready": False, "problems": ["database_up"]}
    assert WebsiteReadiness.model_validate(client.get(ready["endpoint"]).json()) == WebsiteReadiness(
        ready=False, problems=["database_up"])
    env.ready = True
    assert client.get(ready["endpoint"]).json() == {"ready": True, "problems": []}



def test_a_declared_health_path_is_served_there(dist: Path) -> None:
    client = TestClient(site(dist, health="/healthz")().create_app())
    assert client.get("/healthz").json() == {"status": "ok"}
    card = client.get("/.well-known/agent-env.json").json()
    assert _extension(card, WEBSITE_EXTENSION_URI)["params"]["health"] == "/healthz"


def _extension(card: dict, uri: str) -> dict:
    return next(e for e in card["capabilities"]["extensions"] if e["uri"] == uri)


def test_the_card_declares_website_v1_its_routes_clock_and_data_plane(client: TestClient) -> None:
    card = client.get("/.well-known/agent-env.json").json()
    assert card["capabilities"]["operations"] == ["data/reset", "data/add", "data/get"]
    params = _extension(card, WEBSITE_EXTENSION_URI)["params"]
    assert params["domains"] == ["shop.example", "www.shop.example"]
    assert params["health"] == "/api/health"
    assert params["routes"] == [
        {"method": "GET", "path": "/api/orders", "name": "list_orders", "roles": None},
        {"method": "POST", "path": "/api/orders", "name": "place_order", "roles": ["default", "agent"]},
        {"method": "POST", "path": "/api/sign-in", "name": "sign_in", "roles": None},
    ]
    assert params["methods"]["openapi"] == {"method": "GET", "endpoint": "/agentenv/ext/website/openapi",
                                            "request": {"type": "object", "properties": {}}}
    assert params["methods"]["session"]["endpoint"] == "/agentenv/ext/website/session"
    clock = _extension(card, "urn:agentenv:clock/v1")["params"]
    assert clock["endpoint"] == "/agentenv/ext/sync_time"


def test_the_openapi_document_names_each_route(client: TestClient) -> None:
    document = client.get("/agentenv/ext/website/openapi").json()
    assert document["openapi"].startswith("3.") and document["info"]["title"] == "Shop"
    listed = document["paths"]["/api/orders"]["get"]
    assert listed["operationId"] == "list_orders" and listed["summary"] == "List the signed-in user's orders."
    placed = document["paths"]["/api/orders"]["post"]
    assert placed["x-agentenv-roles"] == ["default", "agent"]
    assert placed["requestBody"]["content"]["application/json"]["schema"]["type"] == "object"
    assert listed["parameters"] == []


def test_the_default_store_round_trips_through_the_data_plane(client: TestClient, shop: WebsiteEnvironment) -> None:
    assert rpc(client, "data/get")["parts"][0]["data"] == SEED
    rpc(client, "data/reset")
    assert rpc(client, "data/get")["parts"][0]["data"] == {}
    as_file = base64.b64encode(json.dumps({"orders": [{"id": 7, "user": "carol"}]}).encode()).decode()
    rpc(client, "data/add", {"parts": [{"kind": "data", "data": {"current_user": "carol", "users": [{"id": "carol"}]}},
                                       {"kind": "file", "file": {"bytes": as_file, "name": "orders.json"}},
                                       {"kind": "text", "text": json.dumps({"notes": [{"id": "n1"}]})}]})
    assert rpc(client, "data/get")["parts"][0]["data"] == {
        "current_user": "carol", "users": [{"id": "carol"}], "orders": [{"id": 7, "user": "carol"}],
        "notes": [{"id": "n1"}]}
    assert shop.acting_user() == "carol"


def test_a_sites_data_is_read_through_the_data_plane_alone(client: TestClient) -> None:
    client.post("/api/orders", json={"sku": "mug"})
    assert rpc(client, "data/get")["parts"][0]["data"]["orders"][-1]["sku"] == "mug"
    assert "home" in client.get("/go").text
    params = client.get("/.well-known/agent-env.json").json()["capabilities"]["extensions"]
    website_params = next(e for e in params if e["uri"] == "urn:agentenv:website/v1")["params"]
    assert "state" not in website_params and "state" not in website_params.get("methods", {})


def test_each_sid_keeps_a_state_of_its_own_started_from_the_initial_state(client: TestClient,
                                                                          shop: WebsiteEnvironment) -> None:
    client.post("/api/orders", json={"sku": "mug"})
    assert TestClient(shop.asgi).get("/go", params={"sid": "task-7"}).json() == {
        "initial_state": SEED, "current_state": SEED, "state_diff": {}, "changed_sessions": []}
    page = client.get("/", params={"sid": "task-7"})
    assert page.headers["set-cookie"].startswith("agentenv_sid_Shop=task-7;")
    assert "set-cookie" not in client.get("/", params={"sid": "task-7"}).headers
    client.post("/api/orders", json={"sku": "pen"})
    client.post("/api/orders", params={"sid": "other"}, json={"sku": "cup"})
    added = {sid: [o.get("sku") for d in [client.get("/go", params={"sid": sid}).json()["state_diff"]]
                   for o in d.get("orders", {}).get("added", [])] for sid in ("default", "task-7", "other")}
    assert added == {"default": ["mug"], "task-7": ["pen"], "other": ["cup"]}
    assert client.get("/go").json()["state_diff"]["orders"]["added"][0]["sku"] == "cup"
    client.get("/", params={"sid": "default"})
    assert client.get("/go").json()["state_diff"]["orders"]["added"][0]["sku"] == "mug"
    assert rpc(client, "data/get")["parts"][0]["data"]["orders"][-1]["sku"] == "mug"
    assert sorted(shop._store.sessions()) == ["other", "task-7"]
    assert sorted(client.get("/go").json()["changed_sessions"]) == ["other", "task-7"]
    rpc(client, "data/reset")
    assert shop._store.sessions() == [] and client.get("/go", params={"sid": "task-7"}).json()["current_state"] == {}


def test_a_session_id_is_sanitised_as_cua_gym_sites_do() -> None:
    from agentenv_protocol.website.store import session_id

    assert [session_id(v) for v in (None, "", "default", "../a b", "x" * 200)] == [None, None, None, "ab", "x" * 128]
    store = Collections({"items": [{"id": 1}]})
    store.session("s")["items"].append({"id": 2})
    assert store.session("default") is store and store.export() == {"items": [{"id": 1}]}
    assert store.session("s").diff() == {"items": {"added": [{"id": 2}], "removed": [], "changed": []}}


def test_broad_routes_leave_the_built_in_ones_answering_their_paths(dist: Path) -> None:
    class Broad(site(dist)):
        @api.get("/{slug}")
        def page(self, slug: str) -> dict:
            return {"page": slug}

        @api.get("/api/{name}")
        def named(self, name: str) -> dict:
            return {"named": name}

    env = Broad()
    env.store.load(SEED)
    client = TestClient(env.create_app())
    assert client.get("/api/session").json()["user"] == "alice"
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/about").json() == {"page": "about"} and client.get("/api/other").json() == {"named": "other"}


def test_a_broad_route_never_serves_a_page_at_the_data_planes_or_cards_paths(dist: Path) -> None:
    class Catchall(site(dist)):
        @api.get("/{rest:path}")
        def page(self, rest: str) -> dict:
            return {"page": rest}

    client = TestClient(Catchall().create_app())
    for path in ("/agentenv/anything", "/agentenv/", "/x/../agentenv/y", "/.well-known/agent-env.json/"):
        assert client.get(path).status_code == 404, path
    assert client.get("/.well-known/agent-env.json").status_code == 200
    assert client.get("/docs/a").json() == {"page": "docs/a"}


def test_a_site_with_its_own_hooks_keeps_them(dist: Path) -> None:
    class Own(site(dist)):
        @get_data
        def export(self) -> list:
            return [DataPart(data={"mine": True})]

    client = TestClient(Own().create_app())
    assert rpc(client, "data/get")["parts"][0]["data"] == {"mine": True}
    assert client.get("/.well-known/agent-env.json").json()["capabilities"]["operations"] == ["data/reset", "data/add", "data/get"]


def test_a_seed_data_file_is_the_sites_data_to_start(tmp_path: Path, dist: Path) -> None:
    (tmp_path / "seed.json").write_text(json.dumps(SEED))
    env = site(dist, data=str(tmp_path / "seed.json"))()
    assert env.store.export() == SEED and env.acting_user() == "alice"


def test_the_acting_user_is_signed_in_until_a_session_says_else(client: TestClient) -> None:
    assert client.get("/api/orders").json() == []
    first = client.get("/api/session")
    assert first.json() == {"user": "alice", "role": "default", "signed_in": True}
    cookie = first.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE}=") and "domain" not in cookie.lower() and "httponly" in cookie.lower()
    client.post("/api/sign-in", json={"user": "bob"})
    assert client.get("/api/orders").json() == [{"id": 1, "user": "bob"}]
    assert client.get("/api/session").json()["user"] == "bob"
    assert client.delete("/api/session").json() == {"user": None, "role": "default", "signed_in": False}
    assert client.get("/api/session").json()["user"] is None
    # The jar holds the site's signed-out cookie under its own domain; a second cookie of the name would race it.
    client.cookies.clear()
    client.cookies.set(COOKIE, "forged.signature")
    assert client.get("/api/session").json()["user"] == "alice"


def test_the_session_cookie_expires_when_the_session_does(client: TestClient) -> None:
    from email.utils import parsedate_to_datetime

    cookie = client.get("/api/session").headers["set-cookie"]
    expires = next(a.split("=", 1)[1] for a in cookie.split(";") if a.strip().lower().startswith("expires="))
    ahead = parsedate_to_datetime(expires) - datetime.datetime.now(datetime.timezone.utc)
    assert datetime.timedelta(days=7) - datetime.timedelta(minutes=1) < ahead <= datetime.timedelta(days=7)


def test_the_session_operation_mints_cookies_for_a_user_or_signed_out(shop: WebsiteEnvironment) -> None:
    client = TestClient(shop.create_app())
    minted = client.post("/agentenv/ext/website/session", json={"user": "bob"}).json()
    assert minted["user"] == "bob" and minted["expires_at"].endswith("Z")
    (cookie,) = minted["cookies"]
    assert cookie["name"] == COOKIE and "domain" not in cookie and cookie["path"] == "/"
    browser = TestClient(shop.asgi, cookies={COOKIE: cookie["value"]})
    assert browser.get("/api/session").json()["user"] == "bob"
    signed_out = client.post("/agentenv/ext/website/session", json={}).json()["cookies"][0]["value"]
    assert TestClient(shop.asgi, cookies={COOKIE: signed_out}).get("/api/session").json()["user"] is None


def test_each_site_keeps_its_own_session_cookie_and_reads_the_legacy_one(dist: Path, monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT_NAME", "shop a/1")
    first = site(dist)()
    monkeypatch.setenv("ENVIRONMENT_NAME", "shop-b")
    second = site(dist)()
    for env in (first, second):
        env.store.load(SEED)
    named = "agentenv_session_shop_a_1~030f5785d67a"
    assert (first.session_cookie, second.session_cookie) == (named, "agentenv_session_shop-b")
    browser = TestClient(first.create_app())
    browser.post("/api/sign-in", json={"user": "bob"})
    other = TestClient(second.create_app(), cookies=browser.cookies)
    other.post("/api/sign-in", json={"user": "alice"})
    assert set(other.cookies.keys()) == {named, "agentenv_session_shop-b"}
    assert TestClient(first.asgi, cookies=other.cookies).get("/api/session").json()["user"] == "bob"
    card = browser.get("/.well-known/agent-env.json").json()
    assert _extension(card, WEBSITE_EXTENSION_URI)["params"]["session_cookie"] == named
    legacy = first._session_signer().encode("bob")[0]
    assert TestClient(first.asgi, cookies={SESSION_COOKIE: legacy}).get("/api/orders").json() == [{"id": 1, "user": "bob"}]


def test_no_site_s_session_choice_is_another_site_s_sign_in(dist: Path, monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT_NAME", "notes")
    notes = site(dist)()
    monkeypatch.setenv("ENVIRONMENT_NAME", "notes_sid")
    notes_sid = site(dist)()
    browser = TestClient(notes.create_app())
    browser.post("/api/orders", params={"sid": "task-1"}, json={"sku": "mug"})
    other = TestClient(notes_sid.create_app(), cookies=browser.cookies)
    other.post("/api/sign-in", json={"user": "bob"})
    back = TestClient(notes.asgi, cookies=other.cookies)
    assert [o["sku"] for o in back.get("/go").json()["current_state"]["orders"]] == ["mug"]
    assert TestClient(notes_sid.asgi, cookies=back.cookies).get("/api/session").json()["user"] == "bob"


def test_sites_whose_names_spell_alike_keep_their_own_cookies(dist: Path, monkeypatch) -> None:
    apps = []
    for name in ("shop a/1", "shop_a_1"):
        monkeypatch.setenv("ENVIRONMENT_NAME", name)
        env = site(dist)()
        env.store.load(SEED)
        apps.append(env.create_app())
    browser = TestClient(apps[0])
    browser.post("/api/sign-in", json={"user": "bob"})
    browser.get("/", params={"sid": "task-1"})
    other = TestClient(apps[1], cookies=browser.cookies)
    other.post("/api/sign-in", json={"user": "alice"})
    other.get("/", params={"sid": "task-2"})
    assert len(other.cookies.keys()) == 4
    back = TestClient(apps[0], cookies=other.cookies)
    assert back.get("/api/session").json()["user"] == "bob"
    assert back.cookies.get("agentenv_sid_shop_a_1~030f5785d67a") == "task-1"


def test_every_answer_carries_the_universe_time(client: TestClient) -> None:
    for response in (client.get("/"), client.get("/api/orders"), client.get("/nope.js"),
                     client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": "data/get"}),
                     client.get("/.well-known/agent-env.json")):
        now = parse_time(response.headers[UNIVERSE_NOW_HEADER])
        assert abs((now - datetime.datetime.now(datetime.timezone.utc)).total_seconds()) < 60


class _Clock:
    """A gateway-shaped /clock/time whose time a test sets."""

    def __init__(self, virtual: str, rate: object = None) -> None:
        self.virtual, self.rate = virtual, rate
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                answer = {"virtual_time": outer.virtual}
                if outer.rate is not None:
                    answer["virtual_seconds_per_real_second"] = outer.rate
                body = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/clock/time"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def gateway_clock():
    clock = _Clock("2031-01-01T00:00:00Z")
    yield clock
    clock.close()


def test_sync_time_ties_now_and_the_header_to_the_gateway_clock(client: TestClient, shop: WebsiteEnvironment,
                                                               gateway_clock: _Clock, monkeypatch) -> None:
    synced = client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url}).json()
    assert synced == {"env_get_time_url": gateway_clock.url, "gateway_virtual_time": "2031-01-01T00:00:00.000Z"}
    assert shop.now().year == 2031
    assert client.get("/").headers[UNIVERSE_NOW_HEADER].startswith("2031-01-01T00:00:0")
    order = client.post("/api/orders", json={"sku": "pen"}).json()
    assert order["at"].startswith("2031-01-01")
    gateway_clock.virtual = "2031-06-01T12:00:00Z"
    monkeypatch.setattr(clock_module, "REFRESH_S", 0.0)
    assert client.get("/").headers[UNIVERSE_NOW_HEADER].startswith("2031-06-01T12:00")


def test_sync_time_takes_the_universe_s_time_zone_and_a_later_sync_without_one_drops_it(
        client: TestClient, shop: WebsiteEnvironment, gateway_clock: _Clock) -> None:
    method = _extension(client.get("/.well-known/agent-env.json").json(), "urn:agentenv:clock/v1")["params"]["methods"]["sync_time"]
    assert "timezone" in method["request"]["properties"] and method["request"]["required"] == ["env_get_time_url"]
    zoned = client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url, "timezone": " America/Chicago "})
    assert zoned.json()["timezone"] == "America/Chicago" == shop.timezone
    assert shop.now().tzinfo == datetime.timezone.utc
    unzoned = client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url}).json()
    assert "timezone" not in unzoned and shop.timezone is None


def test_a_clock_that_cannot_be_read_is_refused_and_a_lost_one_extrapolates(client: TestClient,
                                                                           shop: WebsiteEnvironment,
                                                                           gateway_clock: _Clock, monkeypatch) -> None:
    refused = client.post("/agentenv/ext/sync_time", json={"env_get_time_url": "http://127.0.0.1:9/clock/time"})
    assert refused.status_code == 500 and shop.now().year == datetime.datetime.now().year
    client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url})
    gateway_clock.close()
    monkeypatch.setattr(clock_module, "REFRESH_S", 0.0)
    assert client.get("/").headers[UNIVERSE_NOW_HEADER].startswith("2031-01-01")


def test_clock_patch_puts_the_clock_script_in_html_only(dist: Path, gateway_clock: _Clock) -> None:
    client = TestClient(site(dist, clock_patch=True)().create_app())
    client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url})
    page = client.get("/")
    head = page.text.index("<head>") + len("<head>")
    assert page.text[head:].startswith("<script>") and "window.Date=U" in page.text
    assert str(int(datetime.datetime(2031, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000))[:8] in page.text
    assert int(page.headers["content-length"]) == len(page.content)
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert "<script>" not in TestClient(site(dist)().create_app()).get("/").text
    partial = client.get("/index.html", headers={"Range": "bytes=0-5"})
    assert (partial.status_code, partial.content) == (206, b"<html>")


def test_clock_patch_streams_a_streamed_page_holding_only_what_comes_before_its_head(dist: Path) -> None:
    from starlette.responses import StreamingResponse

    later = asyncio.Event()

    async def chunks():
        yield b"<html><he"
        yield b"ad><title>t</title>"
        await later.wait()
        yield b"<body>late</body></html>"

    async def page(request):
        return StreamingResponse(chunks(), media_type="text/html")

    class Streams(site(dist, clock_patch=True)):
        app = Starlette(routes=[Route("/live", page)])

    app = Streams().create_app()
    sent: list[dict] = []

    async def receive() -> dict:
        await later.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)
        if message.get("body", b"").endswith(b"<title>t</title>"):
            later.set()

    async def run() -> None:
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
                 "path": "/live", "raw_path": b"/live", "root_path": "", "query_string": b"", "headers": [],
                 "client": ("t", 1), "server": ("t", 80)}
        await asyncio.wait_for(app(scope, receive, send), 5)

    asyncio.run(run())
    bodies = [m for m in sent if m["type"] == "http.response.body" and m.get("body")]
    first = bodies[0]["body"]
    assert first.startswith(b"<html><head><script>") and first.endswith(b"<title>t</title>") and bodies[0]["more_body"]
    assert b"".join(m["body"] for m in bodies).endswith(b"<body>late</body></html>")
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert b"content-length" not in dict(start["headers"])


NODE_DATE_CHECKS = """globalThis.window = globalThis;
eval(require("fs").readFileSync(0, "utf8").replace(/<\\/?script>/g, ""));
class Later extends Date {}
console.log(JSON.stringify({
  called: typeof Date() === "string" && Date().includes("2031"),
  constructed: new Date().getUTCFullYear(), given: new Date(0).getTime(), parts: new Date(2020, 0, 2).getDate(),
  now: new Date(Date.now()).getUTCFullYear(), parse: Date.parse("2020-01-01T00:00:00Z"), utc: Date.UTC(2020, 0, 1),
  instance: new Date() instanceof Date && new Later() instanceof Date && new Date().constructor === Date,
  subclass: new Later().getUTCFullYear(),
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_clock_script_s_date_answers_called_and_constructed_at_the_universe_s_time() -> None:
    script = clock_module.clock_patch_script(datetime.datetime(2031, 6, 1, tzinfo=datetime.timezone.utc))
    out = subprocess.run(["node", "-e", NODE_DATE_CHECKS], input=script, capture_output=True, check=True).stdout
    assert json.loads(out) == {"called": True, "constructed": 2031, "given": 0, "parts": 2, "now": 2031,
                               "parse": 1577836800000, "utc": 1577836800000, "instance": True, "subclass": 2031}


def test_a_mounted_app_answers_what_api_and_static_do_not(dist: Path) -> None:
    async def stream(request):
        return PlainTextResponse("streamed")

    class WithApp(site(dist)):
        app = Starlette(routes=[Route("/upload", stream, methods=["POST"]), Route("/api/raw", stream)])

    client = TestClient(WithApp().create_app())
    assert client.post("/upload").text == "streamed"
    assert client.get("/api/raw").text == "streamed"
    assert client.get("/api/raw").headers[UNIVERSE_NOW_HEADER]
    assert "home" in client.get("/somewhere").text
    assert client.get("/api/missing").status_code == 404
    assert client.get("/api/orders").json() == []


def test_serve_runs_uvicorn_on_port(monkeypatch, dist: Path) -> None:
    import uvicorn

    ran = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: ran.update(app=app, **options))
    monkeypatch.setenv("PORT", "8123")
    env = site(dist)()
    env.serve()
    assert ran["port"] == 8123 and ran["host"] == "0.0.0.0" and ran["app"] is env.asgi
    monkeypatch.delenv("PORT")
    with pytest.raises(RuntimeError, match="already mounted"):
        env.create_app()


def test_the_environment_reads_its_database_url(monkeypatch, dist: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://db/site")
    assert site(dist)().database_url == "postgresql://db/site"


def test_a_subclass_init_must_call_super(dist: Path) -> None:
    class Forgot(site(dist)):
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError, match="super"):
        TestClient(Forgot().create_app()).get("/")


def test_collections_insert_update_delete_and_find() -> None:
    store = Collections({"todos": [{"id": 1, "title": "a"}, {"id": 2, "title": "b"}]})
    assert store.insert("todos", {"title": "c"})["id"] == 3
    store.update("todos", 1, title="A")
    assert store.delete("todos", 2) and not store.delete("todos", 9)
    assert store.find("todos", title="A") == [{"id": 1, "title": "A"}]
    assert store.export() == {"todos": [{"id": 1, "title": "A"}, {"id": 3, "title": "c"}]}
    with pytest.raises(ValueError):
        store.load({"todos": "nope"})


def test_the_clock_extrapolates_at_a_rate_two_intervals_agree_on_and_takes_a_set_for_a_jump(monkeypatch) -> None:
    asyncio.run(_clock_rates(monkeypatch))


async def _clock_rates(monkeypatch) -> None:
    from agentenv_protocol.website import UniverseClock

    real, virtual = [0.0], [datetime.datetime(2031, 1, 1, tzinfo=datetime.timezone.utc)]

    async def read(url: str) -> tuple[datetime.datetime, None]:
        return virtual[0], None

    monkeypatch.setattr(clock_module.time, "monotonic", lambda: real[0])
    monkeypatch.setattr(clock_module, "_read", read)
    clock = UniverseClock()
    await clock.sync("http://gateway/clock/time")

    def tick(seconds: float, virtual_seconds: float) -> None:
        real[0] += seconds
        virtual[0] += datetime.timedelta(seconds=virtual_seconds)

    tick(1, 100)
    await clock.refresh()
    assert clock._rate == 1.0
    tick(1, 100)
    await clock.refresh()
    assert clock._rate == 100.0
    real[0] += 0.5
    assert clock.now() == virtual[0] + datetime.timedelta(seconds=50)
    real[0] -= 0.5
    tick(0.4, 0)
    await clock.refresh()
    assert clock._rate == 100.0
    tick(1, 86400 * 30)
    await clock.refresh()
    assert clock._rate == 100.0 and clock.now() == virtual[0]


def test_the_clock_extrapolates_at_the_rate_the_clock_answers_with(monkeypatch) -> None:
    asyncio.run(_answered_rates(monkeypatch))


async def _answered_rates(monkeypatch) -> None:
    from agentenv_protocol.website import UniverseClock

    gateway = _Clock("2031-01-01T00:00:00Z", rate=60)
    try:
        assert await clock_module._read(gateway.url) == (parse_time("2031-01-01T00:00:00Z"), 60.0)
        for unusable in ("fast", True, -1, float("inf")):
            gateway.rate = unusable
            assert (await clock_module._read(gateway.url))[1] is None
    finally:
        gateway.close()
    real, answers = [0.0], [(parse_time("2031-01-01T00:00:00Z"), 60.0)]

    async def read(url: str) -> tuple[datetime.datetime, float | None]:
        return answers[0]

    monkeypatch.setattr(clock_module.time, "monotonic", lambda: real[0])
    monkeypatch.setattr(clock_module, "_read", read)
    clock = UniverseClock()
    await clock.sync("http://gateway/clock/time")
    real[0] += 0.5
    assert clock._rate == 60.0 and clock.now() == answers[0][0] + datetime.timedelta(seconds=30)
    real[0] += 0.5
    answers[0] = (parse_time("2031-02-01T00:00:00Z"), 0.0)
    await clock.refresh()
    real[0] += 10
    assert clock._rate == 0.0 and clock.now() == answers[0][0]


def test_a_refresh_of_the_old_clock_never_overwrites_a_sync_to_a_new_one(monkeypatch) -> None:
    asyncio.run(_refresh_during_sync(monkeypatch))


async def _refresh_during_sync(monkeypatch) -> None:
    from agentenv_protocol.website import UniverseClock

    old, new = (datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc) for year in (2031, 2040))
    release, held = asyncio.Event(), []

    async def read(url: str) -> tuple[datetime.datetime, None]:
        if url == "http://old/clock/time":
            if held:
                await release.wait()
            return old, None
        return new, None

    monkeypatch.setattr(clock_module, "_read", read)
    monkeypatch.setattr(clock_module, "REFRESH_S", 0.0)
    clock = UniverseClock()
    await clock.sync("http://old/clock/time")
    held.append(True)
    refreshing = asyncio.create_task(clock.refresh())
    await asyncio.sleep(0)
    syncing = asyncio.create_task(clock.sync("http://new/clock/time"))
    await asyncio.sleep(0)
    release.set()
    await refreshing
    await syncing
    assert clock.url == "http://new/clock/time" and clock._rate == 1.0
    assert abs((clock.now() - new).total_seconds()) < 5
