"""Capture and replay: the WARC a capture writes reads back as it was answered, a capture follows same-site links to
its depth and loads each page's assets, and ReplayWebsite serves a capture for any host, matched by method, path and
normalised query, with @api routes winning over it, their state its own data plane's, and a capture loaded
through data/add until data/reset; the replay example serves its capture and its synthetic sign-up."""

from __future__ import annotations

import base64
import gzip
import importlib.util
import io
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from agentenv_protocol import DataPart, get_data
from agentenv_protocol.website import WebsiteEnvironment, api, conformance, website
from agentenv_protocol.website.replay import Archive, Entry, ReplayWebsite, WarcWriter, capture, main, request_key

RPC = "/agentenv"
EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "website" / "replay"


def rpc(client: TestClient, method: str, params: dict | None = None) -> dict:
    body = client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).json()
    assert "error" not in body, body
    return body["result"]


def warc(*exchanges: tuple, gzipped: bool = True) -> bytes:
    """A WARC of ``(url, status, headers, body)`` answers to GETs, or ``(url, status, headers, body, method)``."""
    out = io.BytesIO()
    writer = WarcWriter(out, gzipped)
    writer.warcinfo({"software": "test"})
    for url, status, headers, body, *method in exchanges:
        writer.exchange(url, method[0] if method else "GET", status, "OK", headers, body)
    return out.getvalue()


CAPTURE = warc(
    ("https://shop.example/", 200, [("Content-Type", "text/html")], b"<html><body>home</body></html>"),
    ("https://shop.example/search?q=lamp&page=2", 200, [("Content-Type", "application/json")], b'{"hits": 2}'),
    ("https://shop.example/old", 301, [("Location", "https://shop.example/new?x=1")], b""),
    ("https://shop.example/login", 200, [("Set-Cookie", "sid=abc; Domain=shop.example; Path=/; HttpOnly"),
                                         ("Strict-Transport-Security", "max-age=1")], b"ok"),
    ("https://shop.example/api/price", 200, [("Content-Type", "application/json")], b'{"price": 10}'),
    ("https://shop.example/api/session", 200, [("Content-Type", "application/json")], b'{"captured": true}'),
    ("https://shop.example/", 200, [("Content-Type", "text/html")], b"a second capture of home"),
)


@website(domains=["shop.example"])
class Shop(ReplayWebsite):
    def __init__(self, capture=None) -> None:
        super().__init__(capture)
        self.cart: list = []

    def add_site_data(self, parts: list) -> None:
        self.cart += [item for part in parts for item in part.data.get("cart", [])]

    def reset_site_data(self) -> None:
        self.cart = []

    @get_data
    def export(self) -> list:
        return [DataPart(data={"cart": list(self.cart)} if self.cart else {})]

    @api.get("/api/price")
    def price(self) -> dict:
        return {"price": 7, "overridden": True}

    @api.post("/api/cart")
    def add_to_cart(self, body: dict) -> dict:
        self.cart.append({**body, "id": len(self.cart) + 1})
        return self.cart[-1]


@pytest.fixture
def client() -> TestClient:
    return TestClient(Shop(CAPTURE).create_app())


def test_a_request_key_normalises_method_path_and_query():
    assert request_key("head", "", "") == ("GET", "/", "")
    assert request_key("GET", "/a/%7e b", "b=2&a=1&a=0") == ("GET", "/a/~%20b", "a=0&a=1&b=2")
    assert request_key("GET", "/x", "utm_source=mail&fbclid=1&_=123&keep=&q=1") == ("GET", "/x", "keep=&q=1")
    assert request_key("GET", "/a%2Fb", "")[1] != request_key("GET", "/a/b", "")[1]
    assert request_key("GET", "/x", "_=1", ignore_query=())[2] == "_=1"


@pytest.mark.parametrize("gzipped", [True, False])
def test_a_warc_reads_back_each_answer_and_the_method_its_request_record_names(gzipped):
    raw = warc(("https://a.example/x", 200, [("Content-Type", "text/plain"), ("Content-Encoding", "gzip"),
                                             ("Transfer-Encoding", "chunked")], b"hello"),
               ("https://a.example/form", 201, [], b"made", "POST"), gzipped=gzipped)
    assert (raw[:2] == b"\x1f\x8b") is gzipped
    archive = Archive.read(raw)
    get = archive.find("GET", "/x")
    assert (get.status, get.body) == (200, b"hello")
    assert ("Content-Length", "5") in get.headers
    assert not any(name.lower() in ("content-encoding", "transfer-encoding") for name, _ in get.headers)
    assert archive.find("POST", "/form").body == b"made" and archive.find("GET", "/form") is None
    assert b"WARC/1.1" in (gzip.decompress(raw) if gzipped else raw)


def test_a_warc_another_tool_wrote_with_a_chunked_body_reads_dechunked(tmp_path):
    http = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
    record = (f"WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: http://b.example/\r\n"
              f"Content-Type: application/http;msgtype=response\r\nContent-Length: {len(http)}\r\n\r\n").encode()
    (tmp_path / "other.warc").write_bytes(record + http + b"\r\n\r\n")
    entry = Archive.read(tmp_path / "other.warc").find("GET", "/")
    assert entry.body == b"hello world" and entry.headers == ()


class _Site:
    """A small site to capture: pages, a redirect, an asset, a link to another site and one too deep to follow."""

    PAGES = {
        "/": (200, "text/html", b'<html><head><link rel="stylesheet" href="/s.css"></head><body>'
                                b'<a href="/a?utm_source=x">a</a><a href="/moved">moved</a>'
                                b'<a href="https://elsewhere.example/">away</a><img srcset="/i.png 2x"></body></html>'),
        "/a": (200, "text/html", b'<a href="/deep">deep</a><a href="#top">self</a>'),
        "/moved": (302, "text/html", b""),
        "/b": (200, "text/html", b"b"),
        "/deep": (200, "text/html", b"too deep at depth 1"),
        "/s.css": (200, "text/css", b"body{}"),
        "/i.png": (200, "image/png", b"\x89PNG"),
    }

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict]] = []
        pages, seen = self.PAGES, self.seen

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                seen.append((self.path, dict(self.headers)))
                status, kind, body = pages.get(self.path.split("?")[0], (404, "text/plain", b"no"))
                self.send_response(status)
                self.send_header("Content-Type", kind)
                if status == 302:
                    self.send_header("Location", "/b")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def live_site():
    site = _Site()
    yield site
    site.close()


def test_a_capture_follows_same_site_links_to_its_depth_with_each_pages_assets_and_redirects(live_site, tmp_path):
    out = io.BytesIO()
    result = capture([live_site.url + "/"], out, depth=1)
    archive = Archive.read(out.getvalue())
    assert {key[1] for key in archive.entries} == {"/", "/a", "/moved", "/b", "/s.css", "/i.png"}
    assert archive.find("GET", "/a").body.startswith(b'<a href="/deep">')
    assert archive.find("GET", "/moved").status == 302
    assert result.captured == 6 and result.skipped == 1 and not result.failed
    assert all(path != "/deep" for path, _ in live_site.seen)

    path = tmp_path / "site.warc.gz"
    assert main(["capture", live_site.url + "/", "--out", str(path), "--depth", "0", "--no-assets"]) == 0
    assert {key[1] for key in Archive.read(path).entries} == {"/"}


def test_the_request_records_of_a_capture_name_no_header_but_host(live_site):
    out = io.BytesIO()
    capture([live_site.url + "/b"], out, depth=0)
    requests = [block for block in gzip.decompress(out.getvalue()).split(b"WARC/1.1\r\n")
                if b"msgtype=request" in block]
    assert len(requests) == 1
    request = requests[0].split(b"\r\n\r\n")[1]
    assert request == f"GET /b HTTP/1.1\r\nHost: {live_site.url.split('//')[1]}".encode()


def test_a_start_url_that_isnt_http_is_refused():
    with pytest.raises(ValueError, match="http or https"):
        capture(["file:///etc/passwd"], io.BytesIO())


def test_a_replay_serves_its_capture_for_any_host_and_404s_what_it_doesnt_hold(client):
    for host in ("shop.example", "anything.test"):
        home = client.get("/", headers={"Host": host})
        assert (home.status_code, home.text) == (200, "<html><body>home</body></html>")
    assert client.get("/search?page=2&q=lamp&utm_campaign=x").json() == {"hits": 2}
    assert client.get("/search?q=lamp").status_code == 404
    assert client.get("/nowhere").status_code == 404
    assert client.post("/").status_code == 404
    head = client.head("/")
    assert head.status_code == 200 and head.content == b"" and head.headers["content-length"] == "30"


def test_a_replay_keeps_cookies_and_redirects_on_its_own_host(client):
    login = client.get("/login")
    assert login.headers["set-cookie"] == "sid=abc; Path=/; HttpOnly"
    assert "strict-transport-security" not in login.headers
    moved = client.get("/old", follow_redirects=False)
    assert (moved.status_code, moved.headers["location"]) == (301, "/new?x=1")


def test_api_routes_win_over_the_capture_and_the_capture_over_built_in_routes(client):
    assert client.get("/api/price").json() == {"price": 7, "overridden": True}
    assert client.get("/api/session").json() == {"captured": True}
    assert client.get("/agentenv/ext/website/openapi").status_code == 200
    assert client.get("/.well-known/agent-env.json").json()["name"]
    assert client.get("/agentenv/../agentenv").status_code in (404, 405)


def test_the_data_plane_exports_the_overrides_state(client):
    assert client.post("/api/cart", json={"sku": "lamp"}).json() == {"sku": "lamp", "id": 1}
    assert rpc(client, "data/get")["parts"][0]["data"] == {"cart": [{"sku": "lamp", "id": 1}]}
    rpc(client, "data/reset")
    assert rpc(client, "data/get")["parts"][0]["data"] == {}


def test_a_cookie_scoped_to_a_path_holds_at_that_path_behind_the_gateways_prefix():
    site = TestClient(Shop(warc(("https://shop.example/account/sign-in", 200, [
        ("Set-Cookie", "sid=abc; Domain=shop.example; path=/account; Secure"),
        ("Set-Cookie", "theme=dark"),
        ("Set-Cookie", "note=1; Path=relative")], b"ok"))).create_app())
    behind = {"X-Forwarded-Prefix": "/website/shop"}
    assert site.get("/account/sign-in", headers=behind).headers.get_list("set-cookie") == [
        "sid=abc; path=/website/shop/account; Secure", "theme=dark", "note=1; Path=relative"]
    assert site.get("/account/sign-in").headers.get_list("set-cookie") == [
        "sid=abc; path=/account; Secure", "theme=dark", "note=1; Path=relative"]


def test_a_replay_behind_the_gateways_prefix_matches_the_sites_own_path(client):
    behind = {"X-Forwarded-Prefix": "/website/shop"}
    assert client.get("/", headers=behind).text == "<html><body>home</body></html>"
    assert client.get("/search?q=lamp&page=2", headers=behind).json() == {"hits": 2}
    assert client.get("/api/session", headers=behind).json() == {"captured": True}
    assert client.get("/api/price", headers=behind).json() == {"price": 7, "overridden": True}
    assert client.get("/nowhere", headers=behind).status_code == 404
    moved = client.get("/old", headers=behind, follow_redirects=False)
    assert moved.headers["location"] == "/website/shop/new?x=1"
    assert client.get("/login", headers=behind).headers["set-cookie"] == "sid=abc; Path=/website/shop/; HttpOnly"

    relative = TestClient(Shop(warc(("https://shop.example/a", 302, [("Location", "/b")], b""),
                                    ("https://shop.example/c", 302, [("Location", "d")], b""))).create_app())
    assert relative.get("/a", headers=behind, follow_redirects=False).headers["location"] == "/website/shop/b"
    assert relative.get("/c", headers=behind, follow_redirects=False).headers["location"] == "d"
    assert relative.get("/a", follow_redirects=False).headers["location"] == "/b"

    spaced = TestClient(Shop(warc(("https://shop.example/a%2Fb/x%20y", 200, [], b"escaped"))).create_app())
    assert spaced.get("/a%2Fb/x%20y", headers=behind).text == "escaped"
    assert spaced.get("/a/b/x%20y", headers=behind).status_code == 404


def test_a_capture_loaded_through_the_data_plane_is_served_until_a_reset(client, tmp_path):
    other = warc(("https://shop.example/", 200, [("Content-Type", "text/plain")], b"the other capture"))
    by_bytes = {"kind": "file", "file": {"bytes": base64.b64encode(other).decode(), "name": "other.warc.gz"}}
    rpc(client, "data/add", {"parts": [by_bytes, {"kind": "data", "data": {"cart": [{"id": 9}]}}]})
    assert client.get("/").text == "the other capture"
    assert client.get("/search?q=lamp&page=2").status_code == 404
    assert rpc(client, "data/get")["parts"][0]["data"] == {"cart": [{"id": 9}]}
    rpc(client, "data/reset")
    assert client.get("/").text == "<html><body>home</body></html>"

    (tmp_path / "c.warc").write_bytes(warc(("https://x.example/", 200, [], b"from a file"), gzipped=False))
    rpc(client, "data/add", {"parts": [{"kind": "file", "file": {"uri": (tmp_path / "c.warc").as_uri(),
                                                                 "mimeType": "application/warc"}}]})
    assert client.get("/").text == "from a file"


def test_a_capture_loaded_by_a_signed_https_url_is_fetched_within_its_limit(client, monkeypatch):
    import httpx

    from agentenv_protocol.website import replay

    other = warc(("https://shop.example/", 200, [], b"from a signed url"))
    asked = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        return httpx.Response(200, content=other) if request.url.path == "/c.warc.gz" else httpx.Response(403)

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(answer), **kw))

    def load(url: str) -> dict:
        part = {"kind": "file", "file": {"uri": url, "name": "c.warc.gz"}}
        return client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": "data/add",
                                      "params": {"parts": [part]}}).json()

    assert "error" not in load("https://bucket.example/c.warc.gz?X-Amz-Signature=abc")
    assert asked == ["https://bucket.example/c.warc.gz?X-Amz-Signature=abc"]
    assert client.get("/").text == "from a signed url"
    refused = load("https://bucket.example/gone.warc.gz?X-Amz-Signature=abc")
    assert "error" in refused and "X-Amz-Signature" not in str(refused)
    monkeypatch.setattr(replay, "MAX_CAPTURE_URL_BYTES", 10)
    assert "error" in load("https://bucket.example/c.warc.gz?X-Amz-Signature=abc")
    assert "error" in load("http://bucket.example/c.warc.gz")


def test_the_built_in_routes_follow_the_capture_being_served(client):
    def load(capture: bytes) -> None:
        rpc(client, "data/add", {"parts": [{"kind": "file", "file": {"bytes": base64.b64encode(capture).decode(),
                                                                     "name": "c.warc.gz"}}]})

    assert client.get("/api/session").json() == {"captured": True}
    load(warc(("https://shop.example/", 200, [], b"no session here")))
    assert client.get("/api/session").json()["signed_in"] is False
    rpc(client, "data/reset")
    assert client.get("/api/session").json() == {"captured": True}


def test_a_replay_with_a_static_folder_is_refused(tmp_path):
    @website(domains=["x.example"], static=str(tmp_path))
    class Static(ReplayWebsite):
        pass

    with pytest.raises(ValueError, match="no static folder"):
        Static(CAPTURE).create_app()


def test_a_replay_passes_the_conformance_checks(client):
    results = {r.name: r.status for r in conformance.check("http://testserver", client=client)}
    assert {name for name, status in results.items() if status == "failed"} == set(), results


def _example():
    spec = importlib.util.spec_from_file_location("replay_example_site", EXAMPLE_DIR / "site.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_example_serves_its_capture_and_its_sign_up():
    example = _example()
    assert issubclass(example.WidgetDocs, WebsiteEnvironment)
    client = TestClient(example.WidgetDocs().create_app())
    assert "<title>Widget Docs</title>" in client.get("/", headers={"Host": "docs.widgets.test"}).text
    assert client.get("/guide/usage.html").status_code == 200
    assert client.get("/guide/usage.html?utm_source=home").status_code == 200
    assert client.get("/guide/advanced.html").status_code == 404
    assert client.get("/style.css").headers["content-type"].startswith("text/css")
    made = client.post("/api/subscribe", json={"email": "ann@example.com"})
    assert made.status_code == 201
    assert client.post("/api/subscribe", json={"email": "ann@example.com"}).status_code == 409
    assert client.post("/api/subscribe", json={"email": "nobody"}).status_code == 400
    exported = rpc(client, "data/get")["parts"][0]["data"]
    assert [s["email"] for s in exported["subscriptions"]] == ["ann@example.com"]

def test_an_answer_kept_once_still_records_each_host_it_was_captured_from() -> None:
    archive = Archive([Entry("GET", "https://example.com/", 200, (), b"apex"),
                       Entry("GET", "https://www.example.com/", 200, (), b"www")])
    assert archive.find("GET", "/").body == b"apex" and archive.hosts == {"example.com", "www.example.com"}



def test_an_entry_is_kept_once_per_request():
    archive = Archive([Entry("GET", "https://a.example/x?b=1&a=2", 200, (), b"first"),
                       Entry("GET", "https://a.example/x?a=2&b=1", 200, (), b"second")])
    assert len(archive) == 1 and archive.find("GET", "/x", "a=2&b=1").body == b"first"


def test_a_capture_that_expands_past_its_limit_is_refused_while_it_is_read(client, tmp_path, monkeypatch):
    from agentenv_protocol.website import replay

    big = warc(("https://shop.example/", 200, [], b"x" * 100_000))
    assert len(big) < 1_000
    monkeypatch.setattr(replay, "MAX_CAPTURE_BYTES", 50_000)
    with pytest.raises(ValueError, match="over 50000 bytes expanded"):
        Archive.read(big)
    (tmp_path / "big.warc.gz").write_bytes(big)
    with pytest.raises(ValueError, match="expanded"):
        Archive.read(tmp_path / "big.warc.gz")
    (tmp_path / "big.warc").write_bytes(gzip.decompress(big))
    with pytest.raises(ValueError, match="expanded"):
        Archive.read(tmp_path / "big.warc")
    inline = {"kind": "file", "file": {"bytes": base64.b64encode(big).decode(), "name": "big.warc.gz"}}
    by_file = {"kind": "file", "file": {"uri": (tmp_path / "big.warc.gz").as_uri(), "name": "big.warc.gz"}}
    for part in (inline, by_file):
        answer = client.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": "data/add",
                                        "params": {"parts": [part]}}).json()
        assert "error" in answer
    assert client.get("/").text == "<html><body>home</body></html>"
    assert Archive.read(warc(("https://shop.example/", 200, [], b"x" * 40_000))).find("GET", "/").body == b"x" * 40_000


def test_many_small_compressed_answers_are_refused_once_their_decoded_bodies_together_pass_the_limit(monkeypatch):
    from agentenv_protocol.website import replay

    def answers(count: int) -> bytes:
        records = b""
        for n in range(count):
            body = gzip.compress(b"x" * 20_000)
            http = (f"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
            records += (f"WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: https://shop.example/{n}\r\n"
                        f"Content-Type: application/http;msgtype=response\r\nContent-Length: {len(http)}\r\n\r\n"
                        ).encode() + http + b"\r\n\r\n"
        return records

    monkeypatch.setattr(replay, "MAX_CAPTURE_BYTES", 50_000)
    assert len(answers(3)) < 50_000 and len(Archive.read(answers(2))) == 2
    with pytest.raises(ValueError, match="answers are over 50000 bytes decoded"):
        Archive.read(answers(3))


class _StyledSite(_Site):
    """A page whose stylesheet imports another and loads an image, a font, a data URI and another site's image."""

    PAGES = {
        "/": (200, "text/html", b'<link rel="stylesheet" href="/s.css"><p>styled</p>'),
        "/s.css": (200, "text/css", b"@import '/t.css'; body{background:url( \"bg.png\" )}"
                                    b"i{background:url(https://elsewhere.example/x.png)}"
                                    b"b{background:url(data:image/png;base64,AA)}"),
        "/t.css": (200, "text/css; charset=utf-8", b"@font-face{src:URL(/f.woff2) format('woff2')}"),
        "/bg.png": (200, "image/png", b"\x89PNG"),
        "/f.woff2": (200, "font/woff2", b"wOF2"),
    }


def test_a_capture_follows_a_stylesheets_imports_and_assets_on_the_site():
    site = _StyledSite()
    try:
        out = io.BytesIO()
        result = capture([site.url + "/"], out, depth=0)
        assert {key[1] for key in Archive.read(out.getvalue()).entries} == {"/", "/s.css", "/t.css", "/bg.png",
                                                                             "/f.woff2"}
        assert result.captured == 5 and not result.failed
        bare = io.BytesIO()
        capture([site.url + "/"], bare, depth=0, assets=False)
        assert {key[1] for key in Archive.read(bare.getvalue()).entries} == {"/"}
        limited = io.BytesIO()
        assert capture([site.url + "/"], limited, depth=0, max_requests=3).captured == 3
    finally:
        site.close()


class _InlineStyledSite(_Site):
    """A page naming its assets only in a ``style`` value, a ``<style>`` element and a video's ``poster``."""

    PAGES = {
        "/": (200, "text/html", b'<style>@import "/t.css"; h1{background:url(/h.png)}</style>'
                                b'<div style="background:url(\'/d.png\')"></div><video poster="/p.jpg"></video>'
                                b'<i style="background:url(data:image/png;base64,AA)"></i>'),
        "/t.css": (200, "text/css", b"p{}"),
        "/h.png": (200, "image/png", b"\x89PNG"),
        "/d.png": (200, "image/png", b"\x89PNG"),
        "/p.jpg": (200, "image/jpeg", b"\xff\xd8"),
    }


def test_a_capture_loads_what_a_pages_style_values_style_elements_and_posters_name():
    site = _InlineStyledSite()
    try:
        out = io.BytesIO()
        result = capture([site.url + "/"], out, depth=0)
        assert {key[1] for key in Archive.read(out.getvalue()).entries} == {"/", "/t.css", "/h.png", "/d.png", "/p.jpg"}
        assert result.captured == 5 and not result.failed
    finally:
        site.close()


def test_an_imported_warcs_compressed_page_is_read_decoded_so_its_links_stay_on_the_replay(tmp_path):
    page = b'<a href="/a">a</a><link rel="stylesheet" href="https://shop.example/s.css">'
    css = b"b{background:url(/b.png)}"
    records = b""
    for url, kind, body, coding in (("https://shop.example/", "text/html", gzip.compress(page), "gzip"),
                                    ("https://shop.example/s.css", "text/css", zlib.compress(css), "deflate")):
        http = (f"HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Encoding: {coding}\r\n"
                f"Content-Length: {len(body)}\r\n\r\n").encode() + body
        records += (f"WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: {url}\r\n"
                    f"Content-Type: application/http;msgtype=response\r\nContent-Length: {len(http)}\r\n\r\n"
                    ).encode() + http + b"\r\n\r\n"
    (tmp_path / "other.warc").write_bytes(records)
    site = TestClient(Shop(tmp_path / "other.warc").create_app())
    behind = {"X-Forwarded-Prefix": "/website/shop"}
    served = site.get("/", headers=behind)
    assert "content-encoding" not in served.headers
    assert served.content == b'<a href="/website/shop/a">a</a><link rel="stylesheet" href="/website/shop/s.css">'
    assert site.get("/s.css", headers=behind).content == b"b{background:url(/website/shop/b.png)}"


def test_an_imported_answer_in_a_coding_the_replay_cant_undo_is_served_as_captured():
    http = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Encoding: br\r\n\r\n\x0b\x02\x80"
    record = (f"WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: https://shop.example/\r\n"
              f"Content-Type: application/http;msgtype=response\r\nContent-Length: {len(http)}\r\n\r\n").encode()
    entry = Archive.read(record + http + b"\r\n\r\n").find("GET", "/")
    assert entry.body == b"\x0b\x02\x80" and ("Content-Encoding", "br") in entry.headers


def test_a_replay_drops_a_link_tags_integrity_so_its_rewritten_stylesheet_still_loads():
    page = (b'<link rel="stylesheet" href="/s.css" integrity="sha384-abc" crossorigin>'
            b'<script src="/a.js" integrity="sha384-def"></script>')
    site = TestClient(Shop(warc(("https://shop.example/", 200, [("Content-Type", "text/html")], page))).create_app())
    assert site.get("/").content == (b'<link rel="stylesheet" href="/s.css" crossorigin>'
                                     b'<script src="/a.js" integrity="sha384-def"></script>')


def test_a_replay_serves_no_captured_content_security_policy_so_its_rewritten_page_still_loads():
    page = (b'<head><meta http-equiv="Content-Security-Policy" content="style-src \'sha256-abc\'">'
            b'<meta charset="utf-8"><style>b{background:url(/b.png)}</style></head>')
    headers = [("Content-Type", "text/html"), ("Content-Security-Policy", "style-src 'sha256-abc'"),
               ("Content-Security-Policy-Report-Only", "default-src https://shop.example"), ("X-Kept", "1")]
    site = TestClient(Shop(warc(("https://shop.example/", 200, headers, page))).create_app())
    served = site.get("/", headers={"X-Forwarded-Prefix": "/website/shop"})
    assert "content-security-policy" not in served.headers
    assert "content-security-policy-report-only" not in served.headers
    assert served.headers["x-kept"] == "1"
    assert served.content == b'<head><meta charset="utf-8"><style>b{background:url(/website/shop/b.png)}</style></head>'


def test_a_replay_keeps_a_pages_and_a_stylesheets_same_site_links_on_the_replay():
    page = ('<html><head><link rel="stylesheet" href="/style.css"><style>b{background:url("/b.png")}</style>'
            "<script>var u = '/api/x'; location.href = \"https://shop.example/js\";</script></head><body>"
            "<a href='https://shop.example/a?x=1&amp;y=2'>a</a><a href=//shop.example>home</a>"
            '<a href="https://elsewhere.example/z">away</a><a href="rel/c">c</a><a href="#top">top</a>'
            '<img src=/i.png srcset="/i.png 1x, https://shop.example/i2.png 2x" alt="/not-a-link">'
            '<form action="/search"><button formaction="/go2">go</button></form>'
            '<div style="background: url(\'/d.png\')" data-href="/kept">é</div></body></html>').encode()
    css = b"@import '/t.css'; a{background:url(https://shop.example/bg.png)} b{src:url(f.woff2)}"
    site = TestClient(Shop(warc(("https://shop.example/", 200, [("Content-Type", "text/html; charset=utf-8")], page),
                                ("https://shop.example/style.css", 200, [("Content-Type", "text/css")], css),
                                ("https://shop.example/app.js", 200, [("Content-Type", "text/javascript")],
                                 b"fetch('/api/x')"))).create_app())
    behind = {"X-Forwarded-Prefix": "/website/shop"}
    served = site.get("/", headers=behind)
    assert served.content == page.replace(b'"/style.css"', b'"/website/shop/style.css"').replace(
        b'url("/b.png")', b'url("/website/shop/b.png")').replace(
        b"'https://shop.example/a?x=1&amp;y=2'", b"'/website/shop/a?x=1&amp;y=2'").replace(
        b"href=//shop.example>", b"href=/website/shop/>").replace(b"src=/i.png", b"src=/website/shop/i.png").replace(
        b'"/i.png 1x, https://shop.example/i2.png 2x"', b'"/website/shop/i.png 1x, /website/shop/i2.png 2x"').replace(
        b'"/search"', b'"/website/shop/search"').replace(b'"/go2"', b'"/website/shop/go2"').replace(
        b"url('/d.png')", b"url('/website/shop/d.png')")
    assert b"var u = '/api/x'; location.href = \"https://shop.example/js\"" in served.content
    assert served.headers["content-length"] == str(len(served.content))
    assert site.head("/", headers=behind).headers["content-length"] == str(len(served.content))
    assert site.get("/style.css", headers=behind).content == (
        b"@import '/website/shop/t.css'; a{background:url(/website/shop/bg.png)} b{src:url(f.woff2)}")
    assert site.get("/app.js", headers=behind).content == b"fetch('/api/x')"

    direct = site.get("/").content
    assert b"'/a?x=1&amp;y=2'" in direct and b'"/style.css"' in direct and b"https://elsewhere.example/z" in direct
