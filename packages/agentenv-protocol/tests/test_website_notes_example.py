"""The Notes example site (examples/website/notes) as website/v1's reference site: it passes the conformance checker,
and its own behavior holds: the acting user's auto sign-in and signing in as another user, notes kept per user, the
admin-only route, timestamps from the universe's clock, and its seed loaded, exported and reset through the data
plane, readiness following it."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import pytest
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

from agentenv_protocol.website import (ROLE_KEY_ENV, ROLE_KEY_HEADER, UNIVERSE_NOW_HEADER, WebsiteEnvironment,
                                       conformance, website)

NOTES_DIR = Path(__file__).resolve().parents[1] / "examples" / "website" / "notes"
SEED = json.loads((NOTES_DIR / "data" / "seed.json").read_text(encoding="utf-8"))
RPC = "/agentenv"


def _load_site():
    """The example's ``site.py``, imported under a name of its own: ``site`` is the standard library's."""
    spec = importlib.util.spec_from_file_location("notes_example_site", NOTES_DIR / "site.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


notes_site = _load_site()


def rpc(client: TestClient, method: str, params: dict | None = None) -> dict:
    body = client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).json()
    assert "error" not in body, body
    return body["result"]


@pytest.fixture
def notes_class(tmp_path: Path) -> type[WebsiteEnvironment]:
    """The example's class, serving a stand-in page instead of ``web/dist``, which only a Vite build makes."""
    (tmp_path / "index.html").write_text("<!doctype html><html><head><title>Notes</title></head><body></body></html>")
    declared = notes_site.Notes.declaration()._replace(static=str(tmp_path))
    return website(**declared._asdict())(type("Notes", (notes_site.Notes,), {}))


@pytest.fixture
def env(notes_class: type[WebsiteEnvironment]) -> WebsiteEnvironment:
    return notes_class()


@pytest.fixture
def app(env: WebsiteEnvironment):
    return env.create_app()


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


class _Clock:
    """A gateway-shaped /clock/time whose time a test sets."""

    def __init__(self, virtual: str) -> None:
        self.virtual = virtual
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
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
    clock = _Clock("2040-02-03T04:05:06Z")
    yield clock
    clock.close()


def ready(client: TestClient) -> dict:
    """website/v1's ready, where the card says it is."""
    card = client.get("/.well-known/agent-env.json").json()
    site = next(e for e in card["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:website/v1")
    return client.get(site["params"]["methods"]["ready"]["endpoint"]).json()


def test_the_example_passes_every_conformance_check(client: TestClient) -> None:
    results = conformance.check("http://testserver", client=client)
    assert {r.name: r.status for r in results} == {**dict.fromkeys(
        ["card", "any_host", "health", "universe_now", "reserved_paths", "data_plane", "clock", "forwarded",
         "cookies"], "passed"), "mcp": "skipped"}, results


def test_the_card_declares_the_site_and_its_routes_by_name_and_it_is_ready_through_website_v1(
        client: TestClient) -> None:
    card = client.get("/.well-known/agent-env.json").json()
    site = next(e for e in card["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:website/v1")["params"]
    assert card["name"] == "notes"
    assert site["domains"] == ["notes.test"] and "health" not in site
    assert [r["name"] for r in site["routes"]] == ["list_users", "sign_in", "list_notes", "create_note", "get_note",
                                                   "edit_note", "toggle_note", "delete_note", "all_notes"]
    assert next(r for r in site["routes"] if r["name"] == "all_notes")["roles"] == ["admin"]
    assert ready(client)["ready"] is True


def test_a_browser_is_signed_in_as_the_acting_user_and_can_sign_in_as_another(client: TestClient) -> None:
    session = client.get("/api/session")
    assert session.json() == {"user": "ann", "role": "default", "signed_in": True}
    assert "agentenv_session_notes" in session.cookies
    assert [n["title"] for n in client.get("/api/notes").json()] == ["Buy milk", "Book dentist"]
    assert client.post("/api/sign-in", json={"user": "ben"}).json() == {"user": "ben"}
    assert client.get("/api/session").json()["user"] == "ben"
    assert [n["title"] for n in client.get("/api/notes").json()] == ["Draft the quarterly report"]
    refused = client.post("/api/sign-in", json={"user": "mallory"})
    assert refused.status_code == 404 and refused.json()["code"] == "not_found"


def test_notes_are_created_edited_toggled_and_deleted_by_their_owner_only(app, client: TestClient) -> None:
    created = client.post("/api/notes", json={"title": "  Call Sam  ", "body": "about the trip"})
    assert created.status_code == 201
    note = created.json()
    assert (note["owner"], note["title"], note["done"]) == ("ann", "Call Sam", False)
    assert note["created_at"] == note["updated_at"]

    ben = TestClient(app)
    ben.post("/api/sign-in", json={"user": "ben"})
    assert "Call Sam" not in [n["title"] for n in ben.get("/api/notes").json()]
    for response in (ben.get(f"/api/notes/{note['id']}"), ben.patch(f"/api/notes/{note['id']}", json={"done": True}),
                     ben.post(f"/api/notes/{note['id']}/toggle"), ben.delete(f"/api/notes/{note['id']}")):
        assert response.status_code == 404, response.text

    edited = client.patch(f"/api/notes/{note['id']}", json={"body": "about the trip, before Friday"}).json()
    assert (edited["title"], edited["body"]) == ("Call Sam", "about the trip, before Friday")
    assert client.post(f"/api/notes/{note['id']}/toggle").json()["done"] is True
    assert [n["title"] for n in client.get("/api/notes", params={"done": "true"}).json()] == ["Call Sam", "Book dentist"]
    assert client.patch(f"/api/notes/{note['id']}", json={"title": " "}).json()["code"] == "empty_title"
    assert client.post("/api/notes", json={"body": "no title"}).status_code == 400
    assert client.delete(f"/api/notes/{note['id']}").json() == {"deleted": note["id"]}
    assert client.get(f"/api/notes/{note['id']}").status_code == 404


def test_concurrent_creates_each_get_a_note_of_their_own(env: WebsiteEnvironment) -> None:
    barrier = threading.Barrier(8)
    real_max = max

    def slow_max(*args, **kwargs):
        """Hold every create's id choice open together, as worker threads racing would."""
        found = real_max(*args, **kwargs)
        try:
            barrier.wait(timeout=0.2)
        except threading.BrokenBarrierError:
            pass
        return found

    notes_site.max = slow_max
    try:
        with ThreadPoolExecutor(8) as pool:
            made = list(pool.map(lambda i: env.create_note(notes_site.NewNote(title=f"n{i}"), _Signed("ann")),
                                 range(8)))
    finally:
        del notes_site.max
    ids = [note["id"] for note in made]
    assert len(set(ids)) == 8, ids
    assert len([n for n in env.data["notes"] if n["owner"] == "ann" and n["title"].startswith("n")]) == 8


class _Signed:
    """An ``ApiContext`` stand-in signed in as ``user``."""

    def __init__(self, user: str) -> None:
        self.user = user


def test_every_users_notes_are_listed_for_an_admin_caller_only(client: TestClient,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ROLE_KEY_ENV, "k3y")
    for headers in ({}, {"AgentEnv-Role": "agent", ROLE_KEY_HEADER: "k3y"}, {"AgentEnv-Role": "admin"},
                    {"AgentEnv-Role": "admin", ROLE_KEY_HEADER: "guess"}):
        refused = client.get("/api/admin/notes", headers=headers)
        assert refused.status_code == 403, headers
    every = client.get("/api/admin/notes", headers={"AgentEnv-Role": "admin", ROLE_KEY_HEADER: "k3y"}).json()
    assert {n["owner"] for n in every} == {"ann", "ben"}


def test_timestamps_follow_the_universes_clock_once_synced(client: TestClient, gateway_clock: _Clock) -> None:
    before = client.post("/api/notes", json={"title": "before the sync"}).json()
    assert not before["created_at"].startswith("2040")
    client.post("/agentenv/ext/sync_time", json={"env_get_time_url": gateway_clock.url})
    after = client.post("/api/notes", json={"title": "after the sync"})
    assert after.json()["created_at"].startswith("2040-02-03T04:05")
    assert after.headers[UNIVERSE_NOW_HEADER].startswith("2040-02-03T04:05")
    toggled = client.post(f"/api/notes/{before['id']}/toggle").json()
    assert toggled["created_at"] == before["created_at"] and toggled["updated_at"].startswith("2040-02-03T04:05")


def test_the_seed_loads_exports_and_resets_through_the_data_plane(client: TestClient) -> None:
    assert rpc(client, "data/get")["parts"][0]["data"] == SEED
    rpc(client, "data/reset")
    assert rpc(client, "data/get")["parts"][0]["data"] == {}
    assert client.get("/api/notes").status_code == 401
    assert ready(client)["ready"] is False

    seed_file = {"kind": "file", "file": {"uri": (NOTES_DIR / "data" / "seed.json").as_uri(),
                                          "mimeType": "application/json"}}
    rpc(client, "data/add", {"parts": [seed_file]})
    assert ready(client)["ready"] is True
    assert rpc(client, "data/get")["parts"][0]["data"] == SEED
    assert client.get("/api/session").json()["user"] == "ann"

    client.post("/api/notes/1/toggle")
    toggled = rpc(client, "data/get")["parts"][0]["data"]["notes"]
    assert next(note for note in toggled if note["id"] == 1)["done"] is True



GATEWAY_PREFIX = "/website/notes"


def _behind_gateway_prefix(app):
    """The gateway's ``/website/<name>/`` route: a request under the prefix reaches the site with the prefix taken off,
    and any other one answers the gateway's own 404."""

    async def gateway(scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith(GATEWAY_PREFIX + "/"):
            path = scope["path"][len(GATEWAY_PREFIX):]
            return await app({**scope, "path": path, "raw_path": path.encode()}, receive, send)
        if scope["type"] != "http":
            return await app(scope, receive, send)
        await PlainTextResponse("no gateway route", status_code=404)(scope, receive, send)

    return gateway


@pytest.mark.skipif(not (NOTES_DIR / "web" / "dist" / "index.html").is_file(), reason="web/dist needs a Vite build")
def test_the_built_page_loads_its_assets_and_api_under_the_gateways_prefix() -> None:
    page_url = f"http://gateway.test{GATEWAY_PREFIX}/"
    browser = TestClient(_behind_gateway_prefix(notes_site.Notes().create_app()), base_url="http://gateway.test")
    page = browser.get(page_url)
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert page.text.index('location.pathname.endsWith("/")') < page.text.index("<script type=\"module\"")

    assets = [urljoin(page_url, ref) for ref in re.findall(r'(?:src|href)="([^"]+)"', page.text)]
    assert assets and all(urlsplit(url).path.startswith(GATEWAY_PREFIX + "/") for url in assets), assets
    scripts = []
    for url in assets:
        fetched = browser.get(url)
        assert fetched.status_code == 200, url
        if url.endswith(".js"):
            scripts.append(fetched.text)

    called = {path for script in scripts for path in re.findall(r'["`](/?api/[\w-]+)', script)}
    assert {"api/session", "api/users", "api/notes", "api/sign-in"} <= called
    assert not [path for path in called if path.startswith("/")], called
    assert browser.get(urljoin(page_url, "api/session")).json()["user"] == "ann"
    assert [u["id"] for u in browser.get(urljoin(page_url, "api/users")).json()] == ["ann", "ben"]
