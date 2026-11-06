"""A website env that is an MCP server too: its @tool methods served at /mcp, declared on its card, beside its pages,
API and data plane; a site without tools serving no /mcp; and conformance checking the endpoint a card declares."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

from agentenv_protocol import DataPart, ToolContext, add_data, environment_card, get_data, reset_data, tool
from agentenv_protocol.website import UNIVERSE_NOW_HEADER, WebsiteEnvironment, api, website
from agentenv_protocol.website import conformance

MCP_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def _shop(*, tools: bool = True, static: Path | None = None) -> WebsiteEnvironment:
    if static is not None:
        (static / "index.html").write_text("<html><head></head><body>shop</body></html>")
    declared = website(domains=["shop.test"], static=None if static is None else str(static))

    class Shop(WebsiteEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.orders: list[dict] = []

        @api.get("/api/orders", name="list_orders")
        def list_orders(self) -> list:
            return self.orders

        @reset_data
        def reset(self) -> None:
            self.orders = []

        @add_data
        def add(self, parts: list) -> None:
            for part in parts:
                self.orders += part.data.get("orders", [])

        @get_data
        def export(self) -> list:
            return [DataPart(data={"orders": self.orders})]

    if not tools:
        return environment_card(name="shop")(declared(Shop))()

    class ToolShop(Shop):
        @tool()
        def place_order(self, item: str, ctx: ToolContext) -> str:
            """Place an order for item, as the caller."""
            self.orders.append({"item": item, "by": ctx.role, "at": self.now().isoformat()})
            return f"order {len(self.orders)} placed"

    return environment_card(name="shop")(declared(ToolShop))()


def _answer(response: httpx.Response) -> dict:
    text = response.text
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        text = [line[len("data:"):] for line in text.splitlines() if line.startswith("data:")][-1]
    return json.loads(text)


def _mcp(client: TestClient) -> dict:
    """Headers of an initialized MCP session with the site."""
    first = client.post("/mcp", headers=MCP_HEADERS, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}})
    assert first.status_code == 200, first.text
    headers = {**MCP_HEADERS, "mcp-session-id": first.headers["mcp-session-id"]}
    assert client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"}).is_success
    return headers


def test_a_site_with_tools_declares_its_mcp_endpoint_and_tools_on_its_card() -> None:
    with TestClient(_shop().create_app()) as client:
        card = client.get("/.well-known/agent-env.json").json()
    assert card["additionalInterfaces"] == [{"url": "/mcp", "transport": "mcp"}]
    assert [t["name"] for t in card["capabilities"]["tools"]] == ["place_order"]
    assert "ctx" not in card["capabilities"]["tools"][0]["inputSchema"]["properties"]
    assert {e["uri"] for e in card["capabilities"]["extensions"]} >= {"urn:agentenv:website/v1", "urn:agentenv:clock/v1"}


def test_a_tool_call_acts_on_the_site_its_pages_and_data_plane_read() -> None:
    site = _shop()
    with TestClient(site.create_app()) as client:
        headers = _mcp(client)
        listed = _answer(client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
        assert [t["name"] for t in listed["result"]["tools"]] == ["place_order"]
        called = client.post("/mcp", headers={**headers, "AgentEnv-Role": "world"}, json={
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "place_order", "arguments": {"item": "milk"}}})
        assert _answer(called)["result"]["content"][0]["text"] == "order 1 placed"
        assert UNIVERSE_NOW_HEADER in called.headers
        assert [o["item"] for o in client.get("/api/orders").json()] == ["milk"]
        data = client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get"}).json()
    assert data["result"]["parts"][0]["data"]["orders"][0]["item"] == "milk"
    assert site.orders[0]["by"] == "world"


def test_a_site_without_tools_serves_no_mcp_endpoint() -> None:
    site = _shop(tools=False)
    assert not site.serves_mcp()
    with TestClient(site.create_app()) as client:
        card = client.get("/.well-known/agent-env.json").json()
        answer = client.post("/mcp", headers=MCP_HEADERS, json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert card["additionalInterfaces"] == []
    assert "tools" not in (card["capabilities"] or {}) or not card["capabilities"]["tools"]
    assert answer.status_code == 404
    assert site.mcp is None


def test_a_site_with_tools_passes_conformance_with_its_mcp_endpoint_checked(tmp_path: Path) -> None:
    with TestClient(_shop(static=tmp_path).create_app()) as client:
        results = {r.name: r for r in conformance.check("http://testserver", client=client)}
    assert results["mcp"].status == "passed", results["mcp"]
    assert "1 tool(s)" in results["mcp"].detail
    assert all(r.passed for r in results.values()), results


def test_a_site_without_tools_skips_the_mcp_check(tmp_path: Path) -> None:
    with TestClient(_shop(tools=False, static=tmp_path).create_app()) as client:
        results = {r.name: r for r in conformance.check("http://testserver", client=client)}
    assert results["mcp"].status == "skipped"


def _declaring(mcp: Route | None) -> Starlette:
    """A server whose card declares an MCP interface listing the tool ``a``, answering at /mcp as ``mcp`` does."""
    card = {"name": "x", "additionalInterfaces": [{"url": "/mcp", "transport": "mcp"}],
            "capabilities": {"tools": [{"name": "a", "inputSchema": {"type": "object"}}]}}
    routes = [Route("/.well-known/agent-env.json", lambda request: JSONResponse(card))]
    return Starlette(routes=routes + ([mcp] if mcp is not None else []))


def _listing(*pages: tuple[str, ...]) -> Route:
    """An MCP endpoint whose tools/list answers ``pages`` in turn, each pointing at the next by its cursor."""
    async def answer(request: Request) -> Response:
        body = await request.json()
        if body.get("method") == "initialize":
            return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "result": {"serverInfo": {"name": "x"}}})
        if body.get("method") == "tools/list":
            at = int((body.get("params") or {}).get("cursor") or 0)
            result = {"tools": [{"name": n} for n in pages[at]], **({"nextCursor": str(at + 1)} if at + 1 < len(pages)
                                                                    else {})}
            return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "result": result})
        return Response(status_code=202)

    return Route("/mcp", answer, methods=["POST"])


@pytest.mark.parametrize(("mcp", "status", "detail"), [
    (_listing(("a",)), "passed", "lists the card's 1 tool(s)"),
    (_listing((), ("a",)), "passed", "lists the card's 1 tool(s)"),
    (_listing(("a", "b")), "failed", "named ['b'], which the card doesn't list"),
    (_listing(()), "warned", "lists 0 of the card's tools, not ['a']: hidden by default, or missing"),
    (None, "failed", "initialize answered 404"),
])
def test_the_mcp_check_reads_the_endpoint_the_card_declares(mcp: Route | None, status: str, detail: str) -> None:
    client = TestClient(_declaring(mcp))
    card = client.get("/.well-known/agent-env.json").json()
    found = conformance._outcome(lambda: conformance._Checker("http://testserver", client, "127.0.0.1")._mcp(card))
    assert found[0] == status and detail in found[1], found


def test_a_hidden_by_default_tool_on_the_card_warns_without_failing_the_run(tmp_path: Path) -> None:
    site = _shop(static=tmp_path)

    class Gated(type(site)):
        tool_access = True

        @tool(hidden_by_default=True)
        def refund(self, item: str) -> str:
            """Refund an order."""
            return "refunded"

    gated = environment_card(name="shop")(Gated)()
    with TestClient(gated.create_app()) as client:
        results = {r.name: r for r in conformance.check("http://testserver", client=client)}
    assert results["mcp"].status == "warned" and "['refund']" in results["mcp"].detail
    assert all(r.passed for r in results.values()), results


@pytest.mark.parametrize("result", [None, {}, {"tools": "a"}, {"tools": None}])
def test_a_tools_list_answering_no_list_of_tools_fails_the_check(result: object) -> None:
    async def answer(request: Request) -> Response:
        body = await request.json()
        if body.get("method") == "initialize":
            return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "result": {"serverInfo": {"name": "x"}}})
        if body.get("method") == "tools/list":
            return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "result": result})
        return Response(status_code=202)

    client = TestClient(_declaring(Route("/mcp", answer, methods=["POST"])))
    card = client.get("/.well-known/agent-env.json").json()
    found = conformance._outcome(lambda: conformance._Checker("http://testserver", client, "127.0.0.1")._mcp(card))
    assert found[0] == "failed" and "answered no tools list" in found[1], found


def _streaming(*events: str) -> Route:
    """An MCP endpoint answering each request with an event stream of ``events``, ``{id}`` the request's id."""
    async def answer(request: Request) -> Response:
        body = await request.json()
        if "id" not in body:
            return Response(status_code=202)
        stream = "".join(event.replace("{id}", json.dumps(body["id"])) for event in events)
        return Response(stream, media_type="text/event-stream")

    return Route("/mcp", answer, methods=["POST"])


def test_the_mcp_check_reads_whole_events_and_the_reply_to_its_own_request() -> None:
    reply = '{"jsonrpc": "2.0", "id": {id},\n"result": {"serverInfo": {"name": "x"}, "tools": [{"name": "a"}]}}'
    mcp = _streaming('event: message\ndata: {"jsonrpc": "2.0", "method": "notifications/message", "params": {}}\n\n',
                     ": a comment\n\n",
                     "".join(f"data: {line}\n" for line in reply.split("\n")) + "\n",
                     'data: {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}\n\n')
    client = TestClient(_declaring(mcp))
    card = client.get("/.well-known/agent-env.json").json()
    found = conformance._outcome(lambda: conformance._Checker("http://testserver", client, "127.0.0.1")._mcp(card))
    assert found == ("passed", "x at /mcp lists the card's 1 tool(s)")


def test_an_mcp_interface_on_another_server_fails_the_check_with_nothing_sent_there() -> None:
    sent = []
    client = httpx.Client(transport=httpx.MockTransport(lambda request: sent.append(request) or httpx.Response(500)))
    card = {"additionalInterfaces": [{"url": "http://elsewhere.test/mcp", "transport": "mcp"}]}
    found = conformance._outcome(lambda: conformance._Checker("http://site.test", client, "127.0.0.1")._mcp(card))
    assert found[0] == "failed" and sent == []
