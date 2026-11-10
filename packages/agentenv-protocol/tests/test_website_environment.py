"""WebsiteEnvironment as a site's server: static pages, readiness, the card and its OpenAPI document, a site's own data
plane, sessions, the universe's clock read per request, and a mounted app."""

from __future__ import annotations

import asyncio
import copy
import datetime
import json
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from agentenv_protocol import DataPart, add_data, get_data, reset_data
from agentenv_protocol.website import (
    SESSION_COOKIE,
    SESSION_SECRET_ENV,
    SITE_READY_TIMEOUT_S,
    UNIVERSE_NOW_HEADER,
    WEBSITE_EXTENSION_URI,
    WEBSITE_READY_OP,
    ApiContext,
    Unauthorized,
    WebsiteEnvironment,
    WebsiteReadiness,
    api,
    ready_check,
    website,
)
from agentenv_protocol.website import clock as clock_module
from agentenv_protocol.website import environment as environment_module
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
        def __init__(self) -> None:
            super().__init__()
            self.data: dict = {}

        def acting_user(self) -> str | None:
            return self.data.get("current_user")

        @reset_data
        def reset(self) -> None:
            self.data = {}

        @add_data
        def add(self, parts: list) -> None:
            for part in parts:
                self.data.update(part.data)

        @get_data
        def export(self) -> list:
            return [DataPart(data=self.data)]

        @api.get("/api/orders", name="list_orders")
        def list_orders(self, ctx: ApiContext) -> list:
            """List the signed-in user's orders."""
            return [order for order in self.data.get("orders", []) if order["user"] == ctx.user]

        @api.post("/api/orders", name="place_order", roles=["default", "agent"])
        def place_order(self, order: dict, ctx: ApiContext) -> dict:
            placed = {**order, "user": ctx.user, "at": self.now().isoformat()}
            self.data.setdefault("orders", []).append(placed)
            return placed

        @api.post("/api/sign-in")
        def sign_in(self, body: dict, ctx: ApiContext) -> dict:
            ctx.sign_in(body["user"])
            return {"user": ctx.user}

    return Shop


SEED = {"current_user": "alice", "users": [{"id": "alice"}, {"id": "bob"}], "orders": [{"id": 1, "user": "bob"}]}


@pytest.fixture
def shop(dist: Path) -> WebsiteEnvironment:
    env = site(dist)()
    env.data = copy.deepcopy(SEED)
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


def test_a_path_segment_longer_than_a_file_name_can_be_is_a_page_not_a_500(client: TestClient) -> None:
    assert client.get("/" + "a" * 300).status_code == 200
    assert client.get("/" + "a" * 300 + ".js").status_code == 404


def test_the_fallback_never_shadows_api_or_reserved_paths(client: TestClient) -> None:
    assert client.get("/api/orders").json() == []
    assert client.put("/api/orders").status_code == 405
    for path in ("/agentenv", "/agentenv/anything", "/agentenv/ext/website/nope", "/agentenv/../agentenv/x"):
        assert client.get(path).status_code in (404, 405), path
    assert client.get("/.well-known/agent-env.json").json()["name"] == "Shop"


def test_ready_checks_run_at_once_each_bounded_and_say_what_they_found(dist: Path, monkeypatch) -> None:
    monkeypatch.setattr(environment_module, "CHECK_TIMEOUT_S", 0.3)

    class Slow(site(dist)):
        @ready_check
        def stuck(self) -> bool:
            time.sleep(1)
            return True

        @ready_check
        async def waiting(self) -> bool:
            await asyncio.sleep(1)
            return True

        @ready_check
        async def broken(self) -> bool:
            raise RuntimeError("database locked")

        @ready_check
        def seeded(self) -> int:
            return 0

    async def ask() -> tuple[dict, float]:
        started = time.monotonic()
        return await Slow()._website_ready(), time.monotonic() - started

    answer, took = asyncio.run(ask())
    assert answer["problems"] == ["@ready_check broken: RuntimeError: database locked", "@ready_check seeded returned 0",
                                  "@ready_check stuck ran past 0.3 s", "@ready_check waiting ran past 0.3 s"]
    assert answer["ready"] is False and took < 0.9


def test_a_plain_def_check_past_its_bound_runs_once_however_often_ready_is_asked(dist: Path, monkeypatch) -> None:
    monkeypatch.setattr(environment_module, "CHECK_TIMEOUT_S", 0.2)
    released, calls = threading.Event(), []

    class Syncing(site(dist)):
        @ready_check
        def synced(self) -> bool:
            calls.append(len(calls))
            return released.wait(5)

    async def ask() -> tuple[list, dict]:
        env = Syncing()
        stuck = [(await env._website_ready())["problems"] for _ in range(3)]
        released.set()
        while not env._checking["synced"].done():
            await asyncio.sleep(0.01)
        return stuck, await env._website_ready()

    stuck, after = asyncio.run(ask())
    assert stuck == [["@ready_check synced ran past 0.2 s"]] * 3
    assert after["ready"] is True and calls == [0, 1]


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
    answer = client.get(ready["endpoint"]).json()
    assert (answer["ready"], answer["problems"]) == (False, ["@ready_check database_up returned False"])
    assert answer["timeout_s"] == SITE_READY_TIMEOUT_S and 0 <= answer["waited_s"] < 60
    assert WebsiteReadiness.model_validate(answer).problems == ["@ready_check database_up returned False"]
    env.ready = True
    assert {k: v for k, v in client.get(ready["endpoint"]).json().items() if k in ("ready", "problems")} == {
        "ready": True, "problems": []}



def test_a_site_says_how_long_it_may_take_to_be_ready_and_serves_no_health_path(dist: Path) -> None:
    client = TestClient(site(dist, ready_timeout_s=45)().create_app())
    assert client.get("/agentenv/ext/website/ready").json()["timeout_s"] == 45
    card = client.get("/.well-known/agent-env.json").json()
    assert "health" not in _extension(card, WEBSITE_EXTENSION_URI)["params"]
    assert client.get("/api/health").status_code == 404


def _extension(card: dict, uri: str) -> dict:
    return next(e for e in card["capabilities"]["extensions"] if e["uri"] == uri)


def test_the_card_declares_website_v1_its_routes_clock_and_data_plane(client: TestClient) -> None:
    card = client.get("/.well-known/agent-env.json").json()
    assert card["capabilities"]["operations"] == ["data/reset", "data/add", "data/get"]
    params = _extension(card, WEBSITE_EXTENSION_URI)["params"]
    assert params["domains"] == ["shop.example", "www.shop.example"]
    assert "health" not in params
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


def test_a_sites_own_hooks_are_its_data_plane(client: TestClient, shop: WebsiteEnvironment) -> None:
    assert rpc(client, "data/get")["parts"][0]["data"] == SEED
    rpc(client, "data/reset")
    assert rpc(client, "data/get")["parts"][0]["data"] == {}
    rpc(client, "data/add", {"parts": [{"kind": "data", "data": {"current_user": "carol", "users": [{"id": "carol"}]}}]})
    assert rpc(client, "data/get")["parts"][0]["data"] == {"current_user": "carol", "users": [{"id": "carol"}]}
    assert shop.acting_user() == "carol"


def test_a_site_without_data_hooks_serves_no_data_plane_and_signs_nobody_in(dist: Path) -> None:
    @website(static=str(dist))
    class Bare(WebsiteEnvironment):
        pass

    client = TestClient(Bare().create_app())
    assert not client.get("/.well-known/agent-env.json").json()["capabilities"].get("operations")
    answer = client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": "data/get", "params": {}}).json()
    assert "error" in answer and "result" not in answer
    assert client.get("/api/session").json() == {"user": None, "role": "default", "signed_in": False}


def test_a_sites_data_is_read_through_the_data_plane_alone(client: TestClient) -> None:
    client.post("/api/orders", json={"sku": "mug"})
    assert rpc(client, "data/get")["parts"][0]["data"]["orders"][-1]["sku"] == "mug"
    assert "home" in client.get("/go").text
    params = client.get("/.well-known/agent-env.json").json()["capabilities"]["extensions"]
    website_params = next(e for e in params if e["uri"] == "urn:agentenv:website/v1")["params"]
    assert "state" not in website_params and "state" not in website_params.get("methods", {})


def test_broad_routes_leave_the_built_in_ones_answering_their_paths(dist: Path) -> None:
    class Broad(site(dist)):
        @api.get("/{slug}")
        def page(self, slug: str) -> dict:
            return {"page": slug}

        @api.get("/api/{name}")
        def named(self, name: str) -> dict:
            return {"named": name}

    env = Broad()
    env.data = copy.deepcopy(SEED)
    client = TestClient(env.create_app())
    assert client.get("/api/session").json()["user"] == "alice"
    assert client.get("/api/health").json() == {"named": "health"}
    assert client.get("/about").json() == {"page": "about"} and client.get("/api/other").json() == {"named": "other"}


def test_a_broad_route_never_serves_a_page_at_the_data_planes_or_cards_paths(dist: Path) -> None:
    class Catchall(site(dist)):
        @api.get("/{rest:path}")
        def page(self, rest: str) -> dict:
            return {"page": rest}

    client = TestClient(Catchall().create_app())
    for path in ("/agentenv/anything", "/agentenv/", "/x/../agentenv/y", "/.well-known/agent-env.json/"):
        assert client.get(path).status_code == 404, path
        assert client.get(path, headers={"X-Forwarded-Prefix": "/svc/shop"}).status_code == 404, f"behind /svc: {path}"
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


def test_the_session_cookie_lasts_a_max_age_not_until_an_absolute_time(client: TestClient) -> None:
    attributes = [a.strip().lower() for a in client.get("/api/session").headers["set-cookie"].split(";")[1:]]
    assert f"max-age={7 * 24 * 3600}" in attributes and not any(a.startswith("expires") for a in attributes)


def test_a_site_restarted_under_its_deploys_secret_keeps_its_sessions(dist: Path, monkeypatch) -> None:
    monkeypatch.setenv(SESSION_SECRET_ENV, "the-deploys-secret")
    before, after = (site(dist)() for _ in range(2))
    before.data = after.data = copy.deepcopy(SEED)
    browser = TestClient(before.create_app())
    browser.post("/api/sign-in", json={"user": "bob"})
    restarted = TestClient(after.create_app(), cookies={COOKIE: browser.cookies[COOKIE]})
    assert restarted.get("/api/session").json()["user"] == "bob"


def test_the_session_operation_mints_cookies_for_a_user_or_signed_out(shop: WebsiteEnvironment) -> None:
    client = TestClient(shop.create_app())
    minted = client.post("/agentenv/ext/website/session", json={"user": "bob"}).json()
    assert minted.keys() == {"cookies", "local_storage"} and minted["local_storage"] == []
    (cookie,) = minted["cookies"]
    assert cookie["name"] == COOKIE and cookie["path"] == "/" and cookie["max_age"] == 7 * 24 * 3600
    assert "domain" not in cookie and "expires" not in cookie
    browser = TestClient(shop.asgi, cookies={COOKIE: cookie["value"]})
    assert browser.get("/api/session").json()["user"] == "bob"
    signed_out = client.post("/agentenv/ext/website/session", json={}).json()["cookies"][0]["value"]
    assert TestClient(shop.asgi, cookies={COOKIE: signed_out}).get("/api/session").json()["user"] is None


def test_each_site_keeps_its_own_session_cookie(dist: Path, monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT_NAME", "shop a/1")
    first = site(dist)()
    monkeypatch.setenv("ENVIRONMENT_NAME", "shop-b")
    second = site(dist)()
    for env in (first, second):
        env.data = copy.deepcopy(SEED)
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
    shared = first._session_signer().encode("bob")
    assert TestClient(first.asgi, cookies={SESSION_COOKIE: shared}).get("/api/session").json()["user"] == "alice"


def test_sites_whose_names_spell_alike_keep_their_own_cookies(dist: Path, monkeypatch) -> None:
    apps = []
    for name in ("shop a/1", "shop_a_1"):
        monkeypatch.setenv("ENVIRONMENT_NAME", name)
        env = site(dist)()
        env.data = copy.deepcopy(SEED)
        apps.append(env.create_app())
    browser = TestClient(apps[0])
    browser.post("/api/sign-in", json={"user": "bob"})
    other = TestClient(apps[1], cookies=browser.cookies)
    other.post("/api/sign-in", json={"user": "alice"})
    assert len(other.cookies.keys()) == 2
    back = TestClient(apps[0], cookies=other.cookies)
    assert back.get("/api/session").json()["user"] == "bob"


def test_every_answer_carries_the_universe_time(client: TestClient) -> None:
    for response in (client.get("/"), client.get("/api/orders"), client.get("/nope.js"),
                     client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": "data/get"}),
                     client.get("/.well-known/agent-env.json")):
        now = parse_time(response.headers[UNIVERSE_NOW_HEADER])
        assert abs((now - datetime.datetime.now(datetime.timezone.utc)).total_seconds()) < 60


class _Clock:
    """A gateway-shaped /clock/time whose time a test sets; None answers 404, as an unarmed clock does. It counts the
    reads it answers, each after ``delay_s``."""

    def __init__(self, virtual: str | None) -> None:
        self.virtual = virtual
        self.hits = 0
        self.delay_s = 0.0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.hits += 1
                time.sleep(outer.delay_s)
                if outer.virtual is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps({"virtual_time": outer.virtual}).encode()
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


def test_concurrent_requests_share_one_clock_read_and_every_one_answers(shop: WebsiteEnvironment,
                                                                        gateway_clock: _Clock) -> None:
    gateway_clock.delay_s = 0.05

    async def burst() -> tuple[list[httpx.Response], int]:
        transport = httpx.ASGITransport(app=shop.create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://shop.example") as client:
            await client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url})
            before = gateway_clock.hits
            answers = await asyncio.gather(*(client.get("/api/orders") for _ in range(200)))
            return answers, gateway_clock.hits - before

    answers, reads = asyncio.run(burst())
    assert [a.status_code for a in answers] == [200] * 200
    assert {a.headers[UNIVERSE_NOW_HEADER] for a in answers} == {"2031-01-01T00:00:00.000Z"}
    assert reads < 20


def test_sync_time_ties_now_and_the_header_to_the_gateway_clock_read_per_request(client: TestClient,
                                                                                shop: WebsiteEnvironment,
                                                                                gateway_clock: _Clock) -> None:
    synced = client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url}).json()
    assert synced == {"env_get_time_url": gateway_clock.url, "gateway_virtual_time": "2031-01-01T00:00:00.000Z"}
    assert shop.now().year == 2031
    assert client.get("/").headers[UNIVERSE_NOW_HEADER] == "2031-01-01T00:00:00.000Z"
    order = client.post("/api/orders", json={"sku": "pen"}).json()
    assert order["at"] == "2031-01-01T00:00:00+00:00"
    gateway_clock.virtual = "2031-06-01T12:00:00Z"
    for _ in range(2):
        assert client.get("/").headers[UNIVERSE_NOW_HEADER] == "2031-06-01T12:00:00.000Z"


def test_an_unarmed_gateway_clock_reads_as_wall_time(client: TestClient, gateway_clock: _Clock) -> None:
    client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url})
    gateway_clock.virtual = None
    now = parse_time(client.get("/").headers[UNIVERSE_NOW_HEADER])
    assert abs((now - datetime.datetime.now(datetime.timezone.utc)).total_seconds()) < 60


def test_sync_time_with_an_empty_url_goes_back_to_wall_time(client: TestClient, shop: WebsiteEnvironment,
                                                            gateway_clock: _Clock) -> None:
    client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url, "timezone": "America/Chicago"})
    unsynced = client.post("/agentenv/ext/sync_time", json={"env_get_time_url": ""}).json()
    assert unsynced["env_get_time_url"] == "" and not shop.clock.synced and shop.timezone is None
    now = parse_time(client.get("/").headers[UNIVERSE_NOW_HEADER])
    assert abs((now - datetime.datetime.now(datetime.timezone.utc)).total_seconds()) < 60


def test_sync_time_takes_the_universe_s_time_zone_and_a_later_sync_without_one_drops_it(
        client: TestClient, shop: WebsiteEnvironment, gateway_clock: _Clock) -> None:
    method = _extension(client.get("/.well-known/agent-env.json").json(), "urn:agentenv:clock/v1")["params"]["methods"]["sync_time"]
    assert "timezone" in method["request"]["properties"] and method["request"]["required"] == ["env_get_time_url"]
    zoned = client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url, "timezone": " America/Chicago "})
    assert zoned.json()["timezone"] == "America/Chicago" == shop.timezone
    assert shop.now().tzinfo == datetime.timezone.utc
    unzoned = client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url}).json()
    assert "timezone" not in unzoned and shop.timezone is None


def test_a_clock_that_cannot_be_read_is_refused_and_a_lost_one_fails_the_sites_answers_but_not_agentenvs(
        client: TestClient, shop: WebsiteEnvironment, gateway_clock: _Clock) -> None:
    refused = client.post("/agentenv/ext/sync_time", json={"env_get_time_url": "http://127.0.0.1:9/clock/time"})
    assert refused.status_code == 500 and not shop.clock.synced
    client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url})
    gateway_clock.close()
    for path in ("/", "/api/orders", "/api/missing"):
        answer = client.get(path)
        assert answer.status_code == 503 and answer.json()["code"] == "clock_unavailable", path
    card = client.get("/.well-known/agent-env.json")
    assert card.status_code == 200 and UNIVERSE_NOW_HEADER not in card.headers
    assert rpc(client, "data/get")["parts"][0]["data"] == SEED
    behind = {"X-Forwarded-Prefix": "/svc/shop"}
    assert client.get("/", headers=behind).status_code == 503
    assert client.get("/.well-known/agent-env.json", headers=behind).status_code == 200
    answer = client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": "data/get", "params": {}}, headers=behind)
    assert answer.status_code == 200 and answer.json()["result"]["parts"][0]["data"] == SEED
    assert client.post("/agentenv/ext/sync_time", json={"env_get_time_url": ""}, headers=behind).status_code == 200
    assert client.get("/", headers=behind).status_code == 200


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


def test_the_clock_script_goes_after_head_not_header_and_after_the_doctype_without_a_head(tmp_path: Path,
                                                                                          gateway_clock: _Clock) -> None:
    root = tmp_path / "pages"
    root.mkdir()
    (root / "index.html").write_text("<!doctype html><html><body><header>top</header></body></html>")
    (root / "headed.html").write_text("<html><header>x</header><head><title>t</title></head></html>")

    @website(static=str(root), clock_patch=True)
    class Patched(WebsiteEnvironment):
        pass

    client = TestClient(Patched().create_app())
    client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url})
    page = client.get("/").text
    assert page.startswith("<!doctype html><html><script>") and page.endswith("<body><header>top</header></body></html>")
    headed = client.get("/headed.html").text
    assert headed.index("<script>") == headed.index("<head>") + len("<head>")


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


def test_dotfiles_and_paths_that_cant_be_files_are_never_served(dist: Path) -> None:
    (dist / ".env").write_text("SECRET=1")
    (dist / ".git").mkdir()
    (dist / ".git" / "config").write_text("SECRET=2")
    client = TestClient(site(dist)().create_app())
    for path in ("/.env", "/.git/config", "/assets/../.env"):
        assert "SECRET" not in client.get(path).text, path
    assert client.get("/.env").status_code == 404
    assert client.get("/%00").status_code != 500


def test_a_request_that_fails_sets_no_session_cookie(dist: Path) -> None:
    class Fails(site(dist)):
        @api.post("/api/refused")
        def refused(self, ctx: ApiContext) -> dict:
            ctx.sign_in("mallory")
            raise Unauthorized("no")

        @api.post("/api/crashes")
        def crashes(self, ctx: ApiContext) -> dict:
            ctx.sign_in("mallory")
            raise RuntimeError("boom")

    client = TestClient(Fails().create_app())
    for path, status in (("/api/refused", 401), ("/api/crashes", 500), ("/api/sign-in", 200)):
        answer = client.post(path, json={"user": "bob"})
        assert answer.status_code == status and ("set-cookie" in answer.headers) is (status == 200), path


def test_a_plain_def_route_runs_in_a_worker_thread_and_an_async_one_on_the_loop(dist: Path) -> None:
    class Threads(site(dist)):
        @api.get("/api/plain")
        def plain(self) -> dict:
            return {"thread": threading.get_ident()}

        @api.get("/api/awaited")
        async def awaited(self) -> dict:
            return {"thread": threading.get_ident()}

    # One session keeps the loop's thread alive for both requests; a thread id is reused once its thread ends.
    with TestClient(Threads().create_app()) as client:
        assert client.get("/api/plain").json()["thread"] != client.get("/api/awaited").json()["thread"]


def test_a_route_whose_type_hints_cant_be_resolved_fails_the_apps_build(dist: Path) -> None:
    class Unresolved(site(dist)):
        @api.post("/api/notes")
        def add_note(self, note: MissingModel) -> dict:  # noqa: F821
            return {}

    with pytest.raises(TypeError, match="add_note: its type hints can't be resolved"):
        Unresolved().create_app()


@pytest.mark.parametrize(("value", "microseconds"), [("2031-01-01T00:00:00.12Z", 120000),
                                                     ("2031-01-01T00:00:00.123456789Z", 123456),
                                                     ("2031-01-01T00:00:00Z", 0),
                                                     ("2031-01-01T02:00:00.5+02:00", 500000)])
def test_a_time_parses_whatever_its_fraction_of_a_second(value: str, microseconds: int) -> None:
    moment = parse_time(value)
    assert moment.astimezone(datetime.timezone.utc).replace(microsecond=0) == datetime.datetime(
        2031, 1, 1, tzinfo=datetime.timezone.utc) and moment.microsecond == microseconds
