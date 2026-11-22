"""A computer's browser requests take the role of whoever drives the computer now, the last to make an input call to its
tools, and a computer's own calls claim none of the writes others make while they run."""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from mcp.types import CallToolResult, TextContent, Tool as MCPTool, ToolAnnotations

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.caller import Caller
from agent_env.env.gateway.constants import AGENT_ENV_COMPUTER_HEADER, AGENT_ENV_ROLE_HEADER, COMPUTER_TOKEN_HEADER
from agent_env.env.gateway.entrypoint import parse_computer_digests
from agent_env.env.gateway.gateway import Gateway
from agent_env.env.gateway.roles import digest
from tst.unit.env.gateway.changes_test import _Changelog, _http, _RealAsyncClient, _site

MAIL, DESKTOP = "http://mail/mcp", "http://desktop/mcp"
_SCHEMA = {"type": "object", "properties": {}}
SEND = MCPTool(name="mail_send_message", inputSchema=_SCHEMA)
CLICK, TYPE, SCREENSHOT = (
    MCPTool(name=f"desktop_computer_{op}", inputSchema=_SCHEMA, annotations=ToolAnnotations(readOnlyHint=read))
    for op, read in (("click", False), ("type", False), ("screenshot", True)))
FROM_DESKTOP = {AGENT_ENV_COMPUTER_HEADER: "desktop"}
DESKTOP_TOKEN = "attribution-token-of-the-desktop"
DIGESTS = {"desktop": digest(DESKTOP_TOKEN)}


class _Child:
    """A child server whose every call runs `effect` with the tool's name first."""

    def __init__(self, tools: list[MCPTool], effect=lambda name: None):
        self.tools, self.effect = tools, effect

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, arguments, *args, meta=None, **kwargs):
        self.effect(name)
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)


async def _office(changelog: _Changelog, monkeypatch, desktop_effect=lambda name: None, **gateway) -> Gateway:
    """A gateway in front of a mail server, its site, and the computer `desktop`, whose site route writes a draft on
    every POST; `gateway` adds to the gateway's arguments."""
    async def ok():
        yield b'{"ok": true}'

    def website(request: httpx.Request) -> httpx.Response:
        changelog.write("drafts", f"d{len(changelog.rows) + 1}")
        return httpx.Response(201, content=ok())

    transport = _site(website)
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = Gateway(host="127.0.0.1", port=0, server_name="t",
                 internal_mcp_servers=[InternalMCPServer("mail", MAIL), InternalMCPServer("desktop", DESKTOP)],
                 website_urls={"mail": "http://mail-frontend"}, computers=["desktop"], **gateway)
    children = {MAIL: _Child([SEND], lambda name: changelog.write("messages", f"m{len(changelog.rows) + 1}")),
                DESKTOP: _Child([CLICK, TYPE, SCREENSHOT], desktop_effect)}

    async def _open(server, stack, report):
        return children[server.mcp_url]

    gw._sessions.open_session = _open  # type: ignore[method-assign]
    gw._query_changelog_id = changelog.max_id  # type: ignore[method-assign]
    gw._db_conn = changelog
    await gw._ensure_tools_discovered()
    return gw


async def _step(gw: Gateway, tool: MCPTool, role: str) -> None:
    response = await gw._step_call_tool({"tool_name": tool.name, "arguments": {}}, role)
    assert response.status_code == 200, response.body


async def _post(gw: Gateway, headers: dict) -> None:
    status, _ = await _http(gw, "POST", "/website/mail/api/drafts", headers=headers)
    assert status == 201


async def _served_post(gw: Gateway, headers: dict) -> int:
    """A POST to the computer's site through the app ``Gateway.run`` serves, which may refuse it."""
    async with _RealAsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway") as client:
        return (await client.post("/website/mail/api/drafts", headers=headers)).status_code


async def _actors(gw: Gateway, *, how: bool = False) -> dict[str, list[tuple[str, ...]]]:
    _, body = await _http(gw, "GET", "/changes")
    return {c["row_id"]: [(a["kind"], a["role"], *((a["role_auth"],) if how else ())) for a in c["actors"]]
            for c in json.loads(body)["changes"]}


@pytest.mark.asyncio
async def test_a_computers_browser_writes_are_its_drivers_and_its_calls_claim_no_one_elses(monkeypatch):
    changelog = _Changelog()
    gw = await _office(changelog, monkeypatch, desktop_effect=lambda name: changelog.write(
        "messages", f"world{len(changelog.rows) + 1}") if name == CLICK.name else None)
    try:
        await _step(gw, CLICK, "person")
        await _post(gw, FROM_DESKTOP)
        await _step(gw, SCREENSHOT, "helper")
        await _post(gw, FROM_DESKTOP)
        await _step(gw, TYPE, "helper")
        await _post(gw, FROM_DESKTOP)
        ctx = contextvars.copy_context()
        ctx.run(Caller(role="carol").bind)
        await ctx.run(asyncio.create_task, gw._mcp._tool_manager.call_tool(CLICK.name, {}))
        await _post(gw, FROM_DESKTOP)
        await _post(gw, {})
        await _post(gw, {AGENT_ENV_COMPUTER_HEADER: "laptop"})
        await _post(gw, {**FROM_DESKTOP, AGENT_ENV_ROLE_HEADER: "auditor"})
        actors = await _actors(gw)
    finally:
        await gw._close_child_sessions()

    assert actors == {"world1": [], "d2": [("request", "person")], "d3": [("request", "person")],
                      "d4": [("request", "helper")], "world5": [], "d6": [("request", "carol")],
                      "d7": [("request", "default")], "d8": [("request", "default")], "d9": [("request", "auditor")]}


@pytest.mark.asyncio
async def test_a_trigger_actions_input_on_a_computer_drives_it_as_the_role_it_acts_as(monkeypatch):
    changelog = _Changelog()
    gw = await _office(changelog, monkeypatch, desktop_effect=lambda name: changelog.write(
        "messages", f"world{len(changelog.rows) + 1}") if name == TYPE.name else None)
    gw._trigger_engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "nora-types", "when": {"type": "action", "tool": SEND.name},
         "actions": [{"type": "tool", "tool": TYPE.name, "as": "nora", "args": {}}]}]})
    try:
        await _step(gw, SEND, "default")
        for _ in range(100):
            if gw._trigger_engine.state()["triggers"][0]["status"] == "fired":
                break
            await asyncio.sleep(0.02)
        await _post(gw, FROM_DESKTOP)
        actors = await _actors(gw)
    finally:
        await gw._close_child_sessions()

    assert actors == {"m1": [("tool_call", "default")], "world2": [], "d3": [("request", "nora")]}


@pytest.mark.asyncio
async def test_a_request_keeps_the_role_it_arrived_with_when_someone_else_drives_before_it_ends(monkeypatch):
    changelog = _Changelog()
    gw = await _office(changelog, monkeypatch)

    async def ok():
        yield b'{"ok": true}'

    def website(request: httpx.Request) -> httpx.Response:
        changelog.write("drafts", f"d{len(changelog.rows) + 1}")
        gw._drive("helper", TYPE.name)
        return httpx.Response(201, content=ok())

    transport = _site(website)
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw._trigger_engine.register({"watch_roles": ["person", "helper"], "triggers": [
        {"id": "drafted", "actions": [], "when": {"type": "website", "app": "mail", "method": "POST",
                                                  "path": {"regex": "^/?api/drafts$"}, "repeat": True}}]})
    try:
        await _step(gw, CLICK, "person")
        await _post(gw, FROM_DESKTOP)
        await _post(gw, FROM_DESKTOP)
        actors = await _actors(gw)
    finally:
        await gw._close_child_sessions()

    detected = [e["provoking"]["role"] for e in gw._trigger_engine.state()["events"] if e["kind"] == "detected"]
    assert detected == ["person", "helper"]
    assert actors == {"d1": [("request", "person")], "d2": [("request", "helper")]}


@pytest.mark.asyncio
async def test_without_digests_a_computer_claims_its_driver_only_while_the_drivers_role_is_open(monkeypatch):
    """A gateway started without computer digests honours AgentEnv-Computer on its own, as before, but not for a
    driver whose role is token-auth: anyone can send the header, so it falls to the plain rules instead."""
    changelog = _Changelog()
    gw = await _office(changelog, monkeypatch)
    try:
        await _step(gw, CLICK, "person")
        await _post(gw, FROM_DESKTOP)
        gw._roles.create("person", {"type": "token"})
        await _post(gw, FROM_DESKTOP)
        actors = await _actors(gw)
    finally:
        await gw._close_child_sessions()

    assert actors == {"d1": [("request", "person")], "d2": [("request", "default")]}


@pytest.mark.asyncio
async def test_with_digests_a_computer_takes_its_driver_only_with_its_token(monkeypatch, caplog):
    """A gateway started with computer digests believes AgentEnv-Computer only with that computer's attribution token:
    without it, or with a wrong one, the header is ignored (logged once per computer) and the request is default's."""
    caplog.set_level(logging.WARNING, logger="agent_env.env.gateway.gateway")
    changelog = _Changelog()
    gw = await _office(changelog, monkeypatch, computer_digests=DIGESTS)
    try:
        await _step(gw, CLICK, "person")
        await _post(gw, FROM_DESKTOP)
        await _post(gw, {**FROM_DESKTOP, COMPUTER_TOKEN_HEADER: "a-wrong-token"})
        await _post(gw, {AGENT_ENV_COMPUTER_HEADER: "laptop", COMPUTER_TOKEN_HEADER: DESKTOP_TOKEN})
        await _post(gw, {**FROM_DESKTOP, COMPUTER_TOKEN_HEADER: DESKTOP_TOKEN})
        actors = await _actors(gw, how=True)
    finally:
        await gw._close_child_sessions()

    assert actors == {"d1": [("request", "default", "default")], "d2": [("request", "default", "default")],
                      "d3": [("request", "default", "default")], "d4": [("request", "person", "computer_token")]}
    ignored = [r.getMessage() for r in caplog.records if r.getMessage().startswith("ignored")]
    assert ignored == ["ignored AgentEnv-Computer 'desktop': it came without that computer's attribution token",
                       "ignored AgentEnv-Computer 'laptop': it came without that computer's attribution token"]
    assert DESKTOP_TOKEN not in caplog.text


@pytest.mark.asyncio
async def test_a_computer_with_its_token_is_never_refused_even_when_its_drivers_role_is_token_auth(monkeypatch):
    changelog = _Changelog()
    gw = await _office(changelog, monkeypatch, computer_digests=DIGESTS)
    gw._roles.create("person", {"type": "token"})
    gw._roles.create("default", {"type": "token"})
    try:
        await _step(gw, CLICK, "person")
        statuses = [await _served_post(gw, headers) for headers in (
            {**FROM_DESKTOP, COMPUTER_TOKEN_HEADER: DESKTOP_TOKEN}, FROM_DESKTOP,
            {**FROM_DESKTOP, AGENT_ENV_ROLE_HEADER: "person"})]
        actors = await _actors(gw, how=True)
    finally:
        await gw._close_child_sessions()

    assert statuses == [201, 403, 403]
    assert actors == {"d1": [("request", "person", "computer_token")]}


@pytest.mark.asyncio
async def test_a_gateway_started_with_no_computer_digests_at_all_believes_no_computer(monkeypatch):
    changelog = _Changelog()
    gw = await _office(changelog, monkeypatch, computer_digests={})
    try:
        await _step(gw, CLICK, "person")
        await _post(gw, FROM_DESKTOP)
        actors = await _actors(gw)
    finally:
        await gw._close_child_sessions()
    assert actors == {"d1": [("request", "default")]}


def test_the_gateway_reads_its_computers_token_digests_from_its_environment():
    hexed = "ab" * 32
    assert parse_computer_digests(None) is None
    assert parse_computer_digests("") == {}
    assert parse_computer_digests(f" desktop={hexed.upper()} ,laptop={hexed},") == {"desktop": hexed, "laptop": hexed}
    for spec, fragment in (("desktop", "expected 'name=sha256 hex'"), (f"={hexed}", "expected"), ("desktop=", "expected"),
                           ("desktop=abc", "expected"), (f"desktop={hexed}0", "expected"),
                           (f"desktop={hexed},desktop={hexed}", "twice")):
        with pytest.raises(ValueError, match=fragment):
            parse_computer_digests(spec)


@pytest.mark.parametrize("entry", ["desktop:raw-attribution-token", "raw-attribution-token",
                                   "raw-attribution-token==", "desktop=raw-attribution-token"],
                         ids=["colon", "bare", "padded", "token-for-digest"])
def test_a_malformed_digest_entry_is_named_by_its_position_never_by_what_it_holds(entry):
    with pytest.raises(ValueError) as refused:
        parse_computer_digests(f"laptop={'ab' * 32},{entry}")
    assert str(refused.value) == "invalid COMPUTER_TOKEN_DIGESTS entry 2: expected 'name=sha256 hex'"
