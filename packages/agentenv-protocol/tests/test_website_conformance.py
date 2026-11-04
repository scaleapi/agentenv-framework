"""The website/v1 conformance checker against an SDK site, a hand-written server that meets the contract, and servers
that don't."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

from agentenv_protocol import DataPart, add_data, get_data, reset_data
from agentenv_protocol.website import WebsiteEnvironment, api, website
from agentenv_protocol.website import conformance
from agentenv_protocol.website.clock import UNIVERSE_NOW_HEADER, iso

CHECKS = ["card", "any_host", "health", "universe_now", "data_plane", "clock", "forwarded", "cookies"]


def statuses(results: list[conformance.Result]) -> dict[str, str]:
    return {r.name: r.status for r in results}


@pytest.fixture
def notes(tmp_path: Path) -> WebsiteEnvironment:
    (tmp_path / "index.html").write_text("<html><head></head><body>hi</body></html>")

    @website(domains=["notes.example"], static=str(tmp_path))
    class Notes(WebsiteEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.data: dict = {"notes": [{"id": 1, "text": "hello"}]}

        def acting_user(self) -> str | None:
            return "ann"

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

        @api.get("/api/notes")
        def notes(self) -> list:
            return self.data.get("notes", [])

    return Notes()


@pytest.fixture
def sdk_site(notes: WebsiteEnvironment) -> TestClient:
    return TestClient(notes.create_app())


def test_an_sdk_site_passes_every_check(sdk_site: TestClient) -> None:
    results = conformance.check("http://testserver", client=sdk_site)
    assert [r.name for r in results] == CHECKS
    assert all(r.status == "passed" for r in results), results
    assert sdk_site.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"}).json()["result"][
        "parts"][0]["data"] == {"notes": [{"id": 1, "text": "hello"}]}


def _handwritten(*, leak_host: bool = False, domain_cookie: bool = False, dated_cookie: bool = False, clock: bool = True,
                 clock_ops_first: tuple[str, ...] = (),
                 leak_redirect: bool = False, stamp_rpc: bool = True, reset_keeps: bool = False,
                 get_fails_after_reset: bool = False, rpc_reply: tuple[int, dict] | None = None,
                 ready_answer: object = None, ready_status: int = 200, rpc_path: str | None = None,
                 ready_endpoint: str = "/ready", card_url: str | None = None) -> Starlette:
    """A server written without the SDK: the contract by hand, optionally broken. Its state is its ``state.data``; with
    ``ready_answer`` its card advertises ``website/v1``'s ``ready``, which answers it with ``ready_status``. With ``rpc_path`` its card's
    ``url`` names its data plane there, not at ``/agentenv``; ``card_url`` puts what it says in the card's ``url`` alone."""
    state = {"items": ["a"], "url": None, "at": None, "reset": False}

    def now() -> datetime.datetime:
        return state["at"] or datetime.datetime.now(datetime.timezone.utc)

    def stamp(response: Response) -> Response:
        response.headers[UNIVERSE_NOW_HEADER] = iso(now())
        return response

    async def page(request: Request) -> Response:
        host = request.headers["host"] if leak_host else request.headers.get("x-forwarded-host", request.headers["host"])
        response = stamp(HTMLResponse(f'<html><a href="https://{host}/x">x</a></html>'))
        response.set_cookie("s", "1", domain="example.com" if domain_cookie else None, expires=3600 if dated_cookie else None)
        return response

    async def health(request: Request) -> Response:
        return stamp(JSONResponse({"ok": True}))

    async def ready(request: Request) -> Response:
        return stamp(JSONResponse(ready_answer, status_code=ready_status))

    async def items(request: Request) -> Response:
        return stamp(JSONResponse(state["items"]))

    async def items_slash(request: Request) -> Response:
        host = request.headers["host"] if leak_redirect else request.headers.get("x-forwarded-host", request.headers["host"])
        return stamp(Response(status_code=307, headers={"Location": f"https://{host}/items"}))

    async def card(request: Request) -> Response:
        extensions = [{"uri": "urn:agentenv:website/v1",
                       "params": {"domains": ["h.example"], "health": "/healthz",
                                  "routes": [{"method": "GET", "path": "/items", "name": "items"}]}}]
        if ready_answer is not None:
            extensions[0]["params"]["methods"] = {"ready": {"method": "GET", "endpoint": ready_endpoint}}
        if clock:
            extensions.append({"uri": "urn:agentenv:clock/v1", "params": {
                "endpoint": "/agentenv/ext/sync_time", "methods": {
                    **{op: {"method": "GET", "endpoint": f"/agentenv/ext/{op}"} for op in clock_ops_first},
                    "sync_time": {"method": "POST"}}}})
        url = card_url or rpc_path
        return stamp(JSONResponse({"name": "hand", **({"url": url} if url else {}),
                                   "capabilities": {"extensions": extensions,
                                                    "operations": ["data/reset", "data/add", "data/get"]}}))

    async def rpc(request: Request) -> Response:
        body = await request.json()
        method = body["method"]
        if rpc_reply is not None:
            return stamp(JSONResponse(rpc_reply[1], status_code=rpc_reply[0]))
        if method == "data/reset":
            state["reset"] = True
            if not reset_keeps:
                state["items"] = []
        elif method == "data/add":
            state["reset"] = False
            added = [item for part in body["params"]["parts"] for item in part["data"]["items"]]
            state["items"] = added if reset_keeps else state["items"] + added
        if method == "data/get" and get_fails_after_reset and state["reset"]:
            answer = {"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32603, "message": "down"}}
        else:
            result = {"parts": [{"kind": "data", "data": {"items": state["items"]}}]} if method == "data/get" else {}
            answer = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        return stamp(JSONResponse(answer)) if stamp_rpc else JSONResponse(answer)

    async def sync_time(request: Request) -> Response:
        import httpx

        url = (await request.json())["env_get_time_url"]
        state["synced"] = url
        if not url:
            state["at"] = None
            return stamp(JSONResponse({"env_get_time_url": url}))
        async with httpx.AsyncClient() as client:
            virtual = (await client.get(url)).json()["virtual_time"]
        state["at"] = datetime.datetime.fromisoformat(virtual.replace("Z", "+00:00"))
        return stamp(JSONResponse({"gateway_virtual_time": virtual}))

    app = Starlette(routes=[Route("/", page), Route("/healthz", health), Route("/ready", ready), Route("/items", items),
                            Route("/items/", items_slash),
                            Route("/.well-known/agent-env.json", card), Route(rpc_path or "/agentenv", rpc, methods=["POST"]),
                            Route("/agentenv/ext/sync_time", sync_time, methods=["POST"])])
    app.state.data = state
    return app


def test_a_handwritten_server_meeting_the_contract_passes() -> None:
    results = conformance.check("http://testserver", client=TestClient(_handwritten()))
    assert statuses(results) == dict.fromkeys(CHECKS, "passed"), results


@pytest.mark.parametrize(("answer", "status", "detail"), [
    ({"ready": True, "problems": [], "since": "boot"}, "passed", "website/v1 ready says it is ready"),
    ({"ready": False, "problems": ["seeded", "indexed"]}, "failed", "website/v1 ready says it is not ready: seeded, indexed"),
    ({"ready": False}, "failed", "website/v1 ready says it is not ready: no problem named"),
])
def test_health_reads_the_cards_ready_as_a_website_readiness(answer: dict, status: str, detail: str) -> None:
    results = {r.name: r for r in conformance.check("http://testserver", client=TestClient(_handwritten(ready_answer=answer)))}
    assert (results["health"].status, results["health"].detail) == (status, detail)


@pytest.mark.parametrize(("answer", "detail"), [
    ({"ready": True, "problems": []}, "website/v1 ready answered 503: no problem named"),
    ({"ready": False, "problems": ["seeding"]}, "website/v1 ready answered 503: seeding"),
])
def test_ready_answering_an_http_error_fails_health_whatever_its_body_says(answer: dict, detail: str) -> None:
    client = TestClient(_handwritten(ready_answer=answer, ready_status=503))
    health = next(r for r in conformance.check("http://testserver", client=client) if r.name == "health")
    assert (health.status, health.detail) == ("failed", detail)


@pytest.mark.parametrize(("answer", "why"), [
    ({"ok": True}, "ready: Field required"),
    ({"ready": "soon", "problems": []}, "ready: Input should be a valid boolean"),
    ({"ready": False, "problems": "seeded"}, "problems: Input should be a valid list"),
    (["ready"], "the answer: Input should be a valid dictionary or instance of WebsiteReadiness"),
])
def test_an_answer_to_ready_that_isnt_a_website_readiness_fails_health_saying_why(answer: object, why: str) -> None:
    health = next(r for r in conformance.check("http://testserver", client=TestClient(_handwritten(ready_answer=answer)))
                  if r.name == "health")
    assert health.status == "failed"
    assert health.detail.startswith(f"website/v1 ready answered {answer!r}, not a WebsiteReadiness: ") and why in health.detail


def test_a_server_breaking_the_contract_fails_those_checks() -> None:
    results = statuses(conformance.check("http://testserver",
                                         client=TestClient(_handwritten(leak_host=True, domain_cookie=True, clock=False))))
    assert results["forwarded"] == results["cookies"] == results["clock"] == "failed"
    assert results["card"] == results["health"] == results["data_plane"] == "passed"


def test_a_cookie_with_an_absolute_expires_warns_without_failing_the_run() -> None:
    cookies = {r.name: r for r in conformance.check("http://testserver", client=TestClient(_handwritten(dated_cookie=True)))}["cookies"]
    assert (cookies.status, cookies.passed) == ("warned", True) and cookies.detail.endswith(": s")


def test_the_forwarded_check_follows_no_redirect_and_reads_its_location(sdk_site: TestClient) -> None:
    forwarded = {r.name: r for r in conformance.check("http://testserver", client=sdk_site)}["forwarded"]
    assert forwarded.status == "passed" and "/api/notes/" in forwarded.detail
    leaked = {r.name: r for r in conformance.check("http://testserver",
                                                    client=TestClient(_handwritten(leak_redirect=True)))}["forwarded"]
    assert leaked.status == "failed" and "Location of /items/" in leaked.detail


def test_universe_now_is_checked_on_the_answers_of_every_check() -> None:
    result = {r.name: r for r in conformance.check("http://testserver",
                                                   client=TestClient(_handwritten(stamp_rpc=False)))}["universe_now"]
    assert result.status == "failed" and "/agentenv" in result.detail


def test_a_reset_that_clears_nothing_fails_the_data_plane_and_leaves_the_data() -> None:
    app = _handwritten(reset_keeps=True)
    result = {r.name: r for r in conformance.check("http://testserver", client=TestClient(app))}["data_plane"]
    assert result.status == "failed" and "data/reset" in result.detail
    assert app.state.data["items"] == ["a"]


def test_the_data_is_added_back_when_a_step_after_the_reset_fails() -> None:
    app = _handwritten(get_fails_after_reset=True)
    result = {r.name: r for r in conformance.check("http://testserver", client=TestClient(app))}["data_plane"]
    assert result.status == "failed" and "down" in result.detail
    assert app.state.data["items"] == ["a"]


@pytest.mark.parametrize("reply", [(500, {}), (200, {}), (500, {"jsonrpc": "2.0", "id": "x", "result": {}})])
def test_a_data_request_without_a_2xx_json_rpc_result_fails_the_data_plane(reply: tuple[int, dict]) -> None:
    app = _handwritten(rpc_reply=reply)
    result = {r.name: r for r in conformance.check("http://testserver", client=TestClient(app))}["data_plane"]
    assert result.status == "failed" and "no JSON-RPC result" in result.detail


def test_an_sdk_sites_data_round_trips_and_is_left_as_the_check_found_it_changed_or_not(
        notes: WebsiteEnvironment, sdk_site: TestClient) -> None:
    for _ in range(2):
        before = json.loads(json.dumps(notes.data))
        result = {r.name: r for r in conformance.check("http://testserver", client=sdk_site)}["data_plane"]
        assert result.status == "passed", result
        assert notes.data == before
        notes.data["notes"].append({"id": 2, "text": "new"})


def _hosts_asked(client: TestClient) -> list[str]:
    hosts: list[str] = []
    client.event_hooks = {"request": [lambda request: hosts.append(request.url.host)], "response": []}
    return hosts


def test_a_card_naming_another_server_for_its_data_plane_fails_it_with_nothing_sent_there() -> None:
    app = _handwritten(card_url="http://elsewhere.example/agentenv")
    client = TestClient(app)
    hosts = _hosts_asked(client)
    result = {r.name: r for r in conformance.check("http://testserver", client=client)}["data_plane"]
    assert result.status == "failed" and "outside the server under check" in result.detail
    assert set(hosts) == {"testserver"} and app.state.data["items"] == ["a"]


def test_a_card_path_that_would_name_another_host_stays_on_the_server() -> None:
    client = TestClient(_handwritten(card_url="//elsewhere.example/agentenv", ready_endpoint="@elsewhere.example/ready",
                                     ready_answer={"ready": True, "problems": []}))
    hosts = _hosts_asked(client)
    conformance.check("http://testserver", client=client)
    assert set(hosts) == {"testserver"}


def test_the_clock_check_leaves_the_server_back_on_wall_time() -> None:
    app = _handwritten()
    results = {r.name: r.status for r in conformance.check("http://testserver", client=TestClient(app))}
    assert results["clock"] == "passed" and app.state.data["synced"] == "" and app.state.data["at"] is None


def test_the_clock_check_leaves_an_sdk_site_unsynced(notes: WebsiteEnvironment, sdk_site: TestClient) -> None:
    results = {r.name: r.status for r in conformance.check("http://testserver", client=sdk_site)}
    assert results["clock"] == "passed" and not notes.clock.synced


def test_the_data_plane_is_reached_at_the_cards_url() -> None:
    app = _handwritten(rpc_path="/rpc")
    result = {r.name: r for r in conformance.check("http://testserver", client=TestClient(app))}["data_plane"]
    assert result.status == "passed", result
    assert app.state.data["items"] == ["a"]


def test_the_clock_check_calls_sync_time_whichever_operation_the_card_lists_first() -> None:
    results = {r.name: r.status for r in conformance.check(
        "http://testserver", client=TestClient(_handwritten(clock_ops_first=("time",))))}
    assert results["clock"] == "passed", results


def test_a_server_with_no_card_health_or_header_fails_them() -> None:
    bare = Starlette(routes=[Route("/", lambda request: JSONResponse({"not": "html"}))])
    results = statuses(conformance.check("http://testserver", client=TestClient(bare)))
    assert results["card"] == results["any_host"] == results["health"] == results["universe_now"] == "failed"
    assert results["data_plane"] == results["clock"] == "failed"
    assert results["cookies"] == "skipped"


def test_main_prints_each_check_and_exits_by_outcome(monkeypatch, capsys) -> None:
    monkeypatch.setattr(conformance, "check", lambda url, **options: [conformance.Result("card", "passed"),
                                                                       conformance.Result("cookies", "warned", "dated"),
                                                                       conformance.Result("clock", "failed", "late")])
    assert conformance.main(["http://site"]) == 1
    printed = capsys.readouterr().out
    assert "PASS card" in printed and "WARN cookies: dated" in printed and "FAIL clock: late" in printed
    assert "2 passed, warned or skipped, 1 failed" in printed


def test_main_exits_2_when_the_server_cannot_be_reached(capsys) -> None:
    assert conformance.main(["http://127.0.0.1:9", "--timeout", "2"]) == 2
    assert "couldn't reach" in capsys.readouterr().err
