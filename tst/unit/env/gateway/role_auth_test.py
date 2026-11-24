"""Who a request acts as on a gateway with roles: resolved once per request from the role header, the agent token and
the control token, refused with 403 only on the routes that act as the role (MCP, /step, /svc, /website, /browser),
and checked again on every tool call of an MCP session already open."""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, Tool as MCPTool

from agent_env.env.gateway import AGENT_ENV_ROLE_META_KEY, InternalMCPServer
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.caller import Caller
from agent_env.env.gateway.constants import AGENT_ENV_COMPUTER_HEADER, AGENT_ENV_ROLE_HEADER, AGENT_TOKEN_HEADER, \
    COMPUTER_TOKEN_HEADER
from agent_env.env.gateway.entrypoint import parse_roles
from agent_env.env.gateway.gateway import Gateway
from agent_env.env.gateway.roles import digest
from tst.unit.env.gateway.changes_test import _Changelog

CONTROL = "s3cret-control"
MEMBER = "agent-token-of-brookes-agent"
DENISES = "agent-token-of-denises-agent"
STRANGER = "agent-token-nobody-registered"
COMPUTER = "attribution-token-of-the-desktop"
BROOKE, DENISE = "brooke@example.com", "denise@example.com"
URL = "http://slack/mcp"
TOOL = MCPTool(name="slack_read_messages", description="", inputSchema={"type": "object", "properties": {}})
_RealAsyncClient = httpx.AsyncClient
INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "probe", "version": "1"}}}
MCP_HEADERS = {"accept": "application/json, text/event-stream"}
# Every kind of route that acts as the caller's role, as (method, path, JSON body).
ACTING = [("POST", "/mcp", INITIALIZE), ("POST", "/step", {"action": "list_tools"}),
          ("POST", "/step", {"action": "call_tool", "tool_name": TOOL.name, "arguments": {}}),
          ("POST", "/svc/shop/api/orders", {"sku": "x"}), ("GET", "/website/shop/index.html", None),
          ("GET", "/browser/web/cdp/t0k3n/json/version", None)]
ACTING_IDS = ["mcp-initialize", "step-list-tools", "step-call-tool", "svc", "website", "browser-discovery"]
# Routes that never take a role, each answering 200 on the gateway these tests build.
NEVER = [("GET", "/state"), ("GET", "/.well-known/agent-env.json"), ("GET", "/trajectory"), ("GET", "/changes"),
         ("GET", "/clock/time"), ("GET", "/triggers/state"), ("GET", "/notifications/state")]


class _Child:
    """The backing MCP server's session: records each call's role."""

    def __init__(self):
        self.roles: list[str] = []

    async def list_tools(self):
        return SimpleNamespace(tools=[TOOL])

    async def call_tool(self, name, arguments, *args, meta=None, **kwargs):
        self.roles.append((meta or {}).get(AGENT_ENV_ROLE_META_KEY))
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)


class _Upstream:
    """Every service behind /svc, /website and /browser: records what reaches it, serves no card, and writes a row of
    ``changelog`` for each request that isn't a GET."""

    def __init__(self, monkeypatch, changelog: _Changelog | None = None):
        self.requests: list[httpx.Request] = []
        self.changelog = changelog
        transport = httpx.MockTransport(self._answer)
        monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient",
                            lambda **kw: _RealAsyncClient(transport=transport, **kw))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/agent-env.json":
            return httpx.Response(404)
        self.requests.append(request)
        if self.changelog is not None and request.method != "GET":
            self.changelog.write("orders", f"o{len(self.changelog.rows) + 1}")

        async def body():
            yield b'{"ok": true}'

        return httpx.Response(200, headers={"content-type": "application/json"}, content=body())


def _gateway(tmp_path, monkeypatch, child: _Child | None = None, **kw) -> Gateway:
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[InternalMCPServer("slack", URL)],
                 rest_proxy_urls={"mcp-slack": "http://slack", "shop": "http://shop-backend", "web": "http://web"},
                 website_urls={"shop": "http://shop-backend"}, control_token=CONTROL, **kw)
    child = child or _Child()

    async def _open(server, stack, report):
        return child

    gw._sessions.open_session = _open  # type: ignore[method-assign]
    changelog = _Changelog()
    gw._query_changelog_id = changelog.max_id  # type: ignore[method-assign]
    gw._db_conn = changelog
    return gw


@asynccontextmanager
async def _client(gw: Gateway):
    async with _RealAsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway") as client:
        try:
            yield client
        finally:
            await gw._close_child_sessions()


async def _declare(client: httpx.AsyncClient, role: str, auth: str, *members: str) -> None:
    control = {"Authorization": f"Bearer {CONTROL}"}
    (await client.post("/roles", json={"role": role, "auth": {"type": auth}}, headers=control)).raise_for_status()
    for token in members:
        (await client.post("/roles/members", json={"role": role, "agent_token": token}, headers=control)).raise_for_status()


async def _act(client: httpx.AsyncClient, method: str, path: str, body: dict | None, headers: dict) -> httpx.Response:
    return await client.request(method, path, json=body, headers={**MCP_HEADERS, **headers})


async def _call(client: httpx.AsyncClient, headers: dict) -> httpx.Response:
    return await client.post("/step", json={"action": "call_tool", "tool_name": TOOL.name, "arguments": {}},
                             headers=headers)


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path, body", ACTING, ids=ACTING_IDS)
async def test_a_plain_header_naming_a_token_role_is_refused_on_every_route_that_acts_as_it(
        tmp_path, monkeypatch, caplog, method, path, body):
    upstream, child = _Upstream(monkeypatch), _Child()
    caplog.set_level(logging.WARNING, logger="agent_env.env.gateway.gateway")
    async with _client(_gateway(tmp_path, monkeypatch, child)) as client:
        await _declare(client, BROOKE, "token", MEMBER)
        refused = await _act(client, method, path, body, {AGENT_ENV_ROLE_HEADER: BROOKE})
    assert refused.status_code == 403
    assert refused.json() == {"ok": False, "error": f"refused: role '{BROOKE}' takes the agent token of one of its members"}
    assert (upstream.requests, child.roles) == ([], [])
    route = path.replace("t0k3n", "<redacted>")
    assert [r.getMessage() for r in caplog.records if r.getMessage().startswith("refused")] == [
        f"refused {method} {route} as '{BROOKE}': role '{BROOKE}' takes the agent token of one of its members"]
    assert "t0k3n" not in caplog.text


@pytest.mark.asyncio
async def test_a_members_token_plays_the_role_and_a_non_members_does_not(tmp_path, monkeypatch):
    upstream, child = _Upstream(monkeypatch), _Child()
    async with _client(_gateway(tmp_path, monkeypatch, child)) as client:
        await _declare(client, BROOKE, "token", MEMBER)
        await _declare(client, DENISE, "none", DENISES)
        played = await _call(client, {AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: MEMBER})
        listed = await client.post("/step", json={"action": "list_tools"},
                                   headers={AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: MEMBER})
        proxied = await client.post("/svc/shop/api/orders", json={},
                                    headers={AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: MEMBER})
        non_member = await _call(client, {AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: DENISES})
        stranger = await _call(client, {AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: STRANGER})

    assert (played.status_code, listed.status_code, proxied.status_code) == (200, 200, 200)
    assert TOOL.name in [t["name"] for t in listed.json()["tools"]]
    assert child.roles == [BROOKE]
    assert upstream.requests[0].headers.get_list("agentenv-role") == [BROOKE]
    assert non_member.status_code == 403 and "isn't a member" in non_member.json()["error"]
    assert stranger.status_code == 403 and "takes the agent token" in stranger.json()["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path, body", ACTING, ids=ACTING_IDS)
async def test_a_token_roles_name_in_another_case_is_refused_on_every_route_that_acts_as_it(
        tmp_path, monkeypatch, method, path, body):
    """The services behind the gateway match a persona's address case-insensitively, so another case of a protected
    persona's name must not pass as a role nobody declared."""
    upstream, child = _Upstream(monkeypatch), _Child()
    async with _client(_gateway(tmp_path, monkeypatch, child)) as client:
        await _declare(client, BROOKE, "token", MEMBER)
        answers = [(await _act(client, method, path, body, {AGENT_ENV_ROLE_HEADER: spelled})).status_code
                   for spelled in (BROOKE.upper(), "Brooke@Example.com")]
    assert answers == [403, 403]
    assert (upstream.requests, child.roles) == ([], [])


@pytest.mark.asyncio
async def test_a_members_token_plays_its_role_named_in_another_case_and_a_respelled_declare_conflicts(
        tmp_path, monkeypatch):
    child = _Child()
    async with _client(_gateway(tmp_path, monkeypatch, child)) as client:
        await _declare(client, BROOKE, "token", MEMBER)
        played = await _call(client, {AGENT_ENV_ROLE_HEADER: BROOKE.upper(), AGENT_TOKEN_HEADER: MEMBER})
        respelled = await client.post("/roles", json={"role": BROOKE.upper(), "auth": {"type": "token"}},
                                      headers={"Authorization": f"Bearer {CONTROL}"})
    assert played.status_code == 200 and child.roles == [BROOKE.upper()]
    assert respelled.status_code == 409
    assert respelled.json() == {"ok": False, "role": BROOKE, "auth": {"type": "token"},
                                "error": f"role '{BROOKE.upper()}' is already declared as '{BROOKE}' with auth 'token'"}


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path, body", ACTING[:5], ids=ACTING_IDS[:5])
async def test_with_default_token_auth_a_request_naming_no_role_and_carrying_no_token_is_refused(
        tmp_path, monkeypatch, method, path, body):
    _Upstream(monkeypatch)
    async with _client(_gateway(tmp_path, monkeypatch)) as client:
        await _declare(client, "default", "token")
        refused = await _act(client, method, path, body, {})
    assert refused.status_code == 403


@pytest.mark.asyncio
async def test_with_default_token_auth_the_routes_that_take_no_role_still_answer(tmp_path, monkeypatch):
    _Upstream(monkeypatch)
    async with _client(_gateway(tmp_path, monkeypatch)) as client:
        await _declare(client, "default", "token")
        await _declare(client, BROOKE, "token")
        (await client.put("/clock/set-time", json={"virtual_time": "2026-01-01T00:00:00Z"})).raise_for_status()
        answers = {path: (await client.request(method, path, headers={AGENT_ENV_ROLE_HEADER: BROOKE})).status_code
                   for method, path in NEVER}
    assert answers == {path: 200 for _, path in NEVER}


@pytest.mark.asyncio
async def test_an_undeclared_or_open_role_works_with_a_plain_header_and_an_unrecognised_token_changes_nothing(
        tmp_path, monkeypatch):
    child = _Child()
    async with _client(_gateway(tmp_path, monkeypatch, child)) as client:
        await _declare(client, DENISE, "none")
        await _declare(client, BROOKE, "token", MEMBER)
        answers = [(await _call(client, headers)).status_code for headers in (
            {AGENT_ENV_ROLE_HEADER: DENISE}, {AGENT_ENV_ROLE_HEADER: "never-declared"},
            {AGENT_ENV_ROLE_HEADER: DENISE, AGENT_TOKEN_HEADER: STRANGER}, {AGENT_TOKEN_HEADER: STRANGER},
            {AGENT_ENV_ROLE_HEADER: DENISE, AGENT_TOKEN_HEADER: ""})]
    assert answers == [200] * 5
    assert child.roles == [DENISE, "never-declared", DENISE, "default", DENISE]


@pytest.mark.asyncio
async def test_one_agent_in_two_roles_plays_either_and_without_a_header_its_only_role_else_default(
        tmp_path, monkeypatch):
    child = _Child()
    async with _client(_gateway(tmp_path, monkeypatch, child)) as client:
        await _declare(client, BROOKE, "token", MEMBER)
        await _call(client, {AGENT_TOKEN_HEADER: MEMBER})
        await _declare(client, DENISE, "token", MEMBER)
        answers = [(await _call(client, headers)).status_code for headers in (
            {AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: MEMBER},
            {AGENT_ENV_ROLE_HEADER: DENISE, AGENT_TOKEN_HEADER: MEMBER}, {AGENT_TOKEN_HEADER: MEMBER})]
    assert answers == [200] * 3
    assert child.roles == [BROOKE, BROOKE, DENISE, "default"]


@pytest.mark.asyncio
async def test_the_control_token_plays_any_role_and_without_a_header_is_default_never_refused(tmp_path, monkeypatch):
    child = _Child()
    control = {"Authorization": f"Bearer {CONTROL}"}
    async with _client(_gateway(tmp_path, monkeypatch, child)) as client:
        await _declare(client, BROOKE, "token", MEMBER)
        await _declare(client, "default", "token")
        answers = [(await _call(client, {**control, **headers})).status_code
                   for headers in ({AGENT_ENV_ROLE_HEADER: BROOKE}, {})]
    assert answers == [200, 200]
    assert child.roles == [BROOKE, "default"]


@pytest.mark.asyncio
async def test_a_session_already_open_is_refused_once_its_role_is_protected_or_its_membership_ends(
        tmp_path, monkeypatch):
    """The MCP session keeps the role its initialize took; each tool call checks it again, so protecting the role or
    ending the membership applies to it without a reconnect, whatever its later requests carry."""
    gw = _gateway(tmp_path, monkeypatch)
    gw._create_and_register_proxy_tool(URL, TOOL)
    gw._tool_server_urls[TOOL.name] = URL
    plain = Caller(BROOKE, "s1", "header")
    member = Caller(DENISE, "s2", "agent_token", digest(MEMBER))
    async with _client(gw) as client:
        await _declare(client, DENISE, "token", MEMBER)
        before = [await _session_call(gw, caller) for caller in (plain, member)]
        await _declare(client, BROOKE, "token")
        pinned = contextvars.copy_context()
        pinned.run(plain.bind)
        listed = pinned.run(gw._mcp._tool_manager.list_tools)
        state = (await client.get("/roles", headers={"Authorization": f"Bearer {CONTROL}"})).json()
        [denise] = [r for r in state["roles"] if r["role"] == DENISE]
        await client.request("DELETE", "/roles/members", json={"role": DENISE, "member_id": denise["members"][0]},
                             headers={"Authorization": f"Bearer {CONTROL}"})
        after = [await _session_call(gw, caller) for caller in (plain, member)]
    assert before == ["ok", "ok"]
    assert listed == []
    assert after == [f"refused: role '{BROOKE}' takes the agent token of one of its members",
                     f"refused: this agent token isn't a member of role '{DENISE}'"]


async def _session_call(gw: Gateway, caller: Caller, tool: str = TOOL.name) -> str:
    """One tools/call on a session pinned to ``caller``, as the session lifespan pins it."""
    ctx = contextvars.copy_context()
    ctx.run(caller.bind)
    try:
        result = await ctx.run(asyncio.create_task, gw._mcp._tool_manager.call_tool(tool, {}))
    except ToolError as e:
        return str(e)
    return result.content[0].text if isinstance(result, CallToolResult) else str(result)


@pytest.mark.asyncio
async def test_an_unrecognised_token_is_logged_once_by_a_digest_prefix_never_the_token(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="agent_env.env.gateway.gateway")
    async with _client(_gateway(tmp_path, monkeypatch)) as client:
        for _ in range(3):
            assert (await _call(client, {AGENT_TOKEN_HEADER: STRANGER})).status_code == 200
    lines = [r.getMessage() for r in caplog.records if "doesn't recognise" in r.getMessage()]
    assert lines == [f"an agent token the gateway doesn't recognise (sha256 {digest(STRANGER)[:12]}...) counts as none"]
    assert STRANGER not in caplog.text


@pytest.mark.asyncio
async def test_past_a_bound_made_up_tokens_and_computer_names_are_neither_logged_nor_kept(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="agent_env.env.gateway.gateway")
    monkeypatch.setattr(gw_mod, "_LOGGED_ONCE_MAX", 2)
    gw = _gateway(tmp_path, monkeypatch, computer_digests={"desktop": digest(COMPUTER)})
    async with _client(gw) as client:
        for n in range(5):
            await _call(client, {AGENT_TOKEN_HEADER: f"made-up-{n}", AGENT_ENV_COMPUTER_HEADER: f"made-up-{n}"})
    lines = [r.getMessage() for r in caplog.records]
    assert sum("doesn't recognise" in line for line in lines) == 2
    assert sum(line.startswith("ignored AgentEnv-Computer") for line in lines) == 2
    assert lines.count("further unrecognised agent tokens are not logged") == 1
    assert lines.count("further ignored AgentEnv-Computer names are not logged") == 1
    assert len(gw._unknown_agent_tokens) == len(gw._ignored_computers) == 3


# ---------------------------------------------------------------- startup roles


def test_startup_roles_are_a_json_list_of_roles_and_their_auth():
    assert parse_roles("") == parse_roles("  ") == {}
    assert parse_roles(json.dumps([{"role": f" {BROOKE} ", "auth": {"type": "token"}}, {"role": DENISE},
                                   {"role": "default", "auth": {"type": "none"}}])) == {
        BROOKE: "token", DENISE: "none", "default": "none"}


@pytest.mark.parametrize("spec, fragment", [
    ("[{", "is not JSON"), ('{"role": "x"}', "a JSON list"), ('["x"]', "a JSON list"),
    ('[{"role": "*"}]', "must name a role"), ('[{"role": ""}]', "must name a role"), ('[{}]', "must name a role"),
    ('[{"role": "x", "auth": {"type": "password"}}]', "auth must be"), ('[{"role": "x", "auth": "token"}]', "auth must be"),
    ('[{"role": "x"}, {"role": " x "}]', "twice"), ('[{"role": "x"}, {"role": "X"}]', r"names 'X' twice \(first as 'x'\)"),
    ('[{"role": "x", "members": ["t"]}]', "unknown field")],
    ids=["bad-json", "object", "strings", "wildcard", "empty", "no-role", "unknown-auth", "bare-auth", "twice",
         "twice-in-another-case", "extra-field"])
def test_startup_roles_refuse_what_the_roles_extension_refuses(spec, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_roles(spec)


@pytest.mark.asyncio
async def test_a_startup_token_role_refuses_a_plain_header_at_once_and_a_restart_brings_it_back_without_members(
        tmp_path, monkeypatch):
    spec = json.dumps([{"role": BROOKE, "auth": {"type": "token"}}])
    async with _client(_gateway(tmp_path, monkeypatch, roles=parse_roles(spec))) as client:
        first = await _call(client, {AGENT_ENV_ROLE_HEADER: BROOKE})
        await _declare(client, BROOKE, "token", MEMBER)
        played = await _call(client, {AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: MEMBER})
        await client.request("DELETE", "/roles", json={"role": BROOKE}, headers={"Authorization": f"Bearer {CONTROL}"})
        undeclared = await _call(client, {AGENT_ENV_ROLE_HEADER: BROOKE})
    async with _client(_gateway(tmp_path, monkeypatch, roles=parse_roles(spec))) as client:
        state = (await client.get("/roles", headers={"Authorization": f"Bearer {CONTROL}"})).json()
        after_restart = await _call(client, {AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: MEMBER})

    assert (first.status_code, played.status_code, undeclared.status_code) == (403, 200, 200)
    assert state == {"roles": [{"role": BROOKE, "auth": {"type": "token"}, "members": [], "startup": True}]}
    assert after_restart.status_code == 403


# ---------------------------------------------------------------- role_auth on the trajectory


def _events(tmp_path, kind: str) -> list[dict]:
    lines = (tmp_path / "trajectory.jsonl").read_text().splitlines()
    return [event for event in map(json.loads, lines) if event["event_type"] == kind]


@pytest.mark.asyncio
async def test_both_step_handlers_record_how_the_role_was_established(tmp_path, monkeypatch):
    async with _client(_gateway(tmp_path, monkeypatch)) as client:
        await _declare(client, BROOKE, "token", MEMBER)
        for headers in ({AGENT_TOKEN_HEADER: MEMBER}, {"Authorization": f"Bearer {CONTROL}", AGENT_ENV_ROLE_HEADER: DENISE},
                        {AGENT_ENV_ROLE_HEADER: DENISE}, {}):
            for tool in (TOOL.name, "list_website_urls"):
                called = await client.post("/step", json={"action": "call_tool", "tool_name": tool, "arguments": {}},
                                           headers=headers)
                assert called.status_code == 200, called.text
    assert [(e["tool_call"]["function_name"], e["role"], e["role_auth"]) for e in _events(tmp_path, "tool_call")] == [
        (TOOL.name, BROOKE, "agent_token"), ("list_website_urls", BROOKE, "agent_token"),
        (TOOL.name, DENISE, "control_token"), ("list_website_urls", DENISE, "control_token"),
        (TOOL.name, DENISE, "header"), ("list_website_urls", DENISE, "header"),
        (TOOL.name, "default", "default"), ("list_website_urls", "default", "default")]


@pytest.mark.asyncio
async def test_mcp_tool_calls_proxied_and_native_record_the_sessions_role_auth(tmp_path, monkeypatch):
    gw = _gateway(tmp_path, monkeypatch)
    gw._create_and_register_proxy_tool(URL, TOOL)
    gw._tool_server_urls[TOOL.name] = URL
    gw._roles.create(BROOKE, {"type": "token"})
    gw._roles.add_member(BROOKE, MEMBER)
    sessions = [Caller(BROOKE, "s1", "agent_token", digest(MEMBER)), Caller(DENISE, "s2", "control_token"),
                Caller(DENISE, "s3", "header"), Caller("default", "s4", "default")]
    try:
        for caller in sessions:
            for tool in (TOOL.name, "list_website_urls"):
                assert not (await _session_call(gw, caller, tool)).startswith("refused")
    finally:
        await gw._close_child_sessions()
    assert [(e["role"], e["role_auth"]) for e in _events(tmp_path, "tool_call")] == [
        (caller.role, caller.source) for caller in sessions for _ in range(2)]


class _SeesCaller(_Child):
    """A backing session that also records the caller each call was made under."""

    def __init__(self):
        super().__init__()
        self.callers: list[Caller] = []

    async def call_tool(self, name, arguments, *args, meta=None, **kwargs):
        self.callers.append(Caller.current())
        return await super().call_tool(name, arguments, *args, meta=meta, **kwargs)


@pytest.mark.asyncio
async def test_a_session_whose_token_joins_its_role_later_records_agent_token_and_is_refused_once_it_leaves(
        tmp_path, monkeypatch):
    """A session opened on an open role with a token that wasn't yet a member records each call by how the membership
    stands at that call: header before the token joins, agent_token once the role is protected and it has, and a
    refusal once it has left."""
    child = _SeesCaller()
    gw = _gateway(tmp_path, monkeypatch, child)
    gw._create_and_register_proxy_tool(URL, TOOL)
    gw._tool_server_urls[TOOL.name] = URL
    opened = gw._roles.resolve(BROOKE, digest(MEMBER))
    session = Caller(opened.role, "s1", opened.source, opened.member)
    tools = (TOOL.name, "list_website_urls")
    try:
        before = [await _session_call(gw, session, tool) for tool in tools]
        gw._roles.create(BROOKE, {"type": "token"})
        member_id = gw._roles.add_member(BROOKE, MEMBER)["member_id"]
        joined = [await _session_call(gw, session, tool) for tool in tools]
        gw._roles.remove_member(BROOKE, member_id)
        left = [await _session_call(gw, session, tool) for tool in tools]
    finally:
        await gw._close_child_sessions()

    assert opened.source == "header" and opened.refusal is None
    assert not any(answer.startswith("refused") for answer in before + joined)
    assert left == [f"refused: this agent token isn't a member of role '{BROOKE}'"] * 2
    assert [(e["tool_call"]["function_name"], e["role"], e["role_auth"]) for e in _events(tmp_path, "tool_call")] == [
        (TOOL.name, BROOKE, "header"), ("list_website_urls", BROOKE, "header"),
        (TOOL.name, BROOKE, "agent_token"), ("list_website_urls", BROOKE, "agent_token")]
    assert [(c.role, c.source, c.member) for c in child.callers] == [
        (BROOKE, "header", digest(MEMBER)), (BROOKE, "agent_token", digest(MEMBER))]


@pytest.mark.asyncio
async def test_svc_and_website_writes_record_how_the_role_was_established_and_changes_names_it(tmp_path, monkeypatch):
    gw = _gateway(tmp_path, monkeypatch)
    _Upstream(monkeypatch, changelog=gw._db_conn)
    who = [({AGENT_TOKEN_HEADER: MEMBER}, BROOKE, "agent_token"),
           ({"Authorization": f"Bearer {CONTROL}"}, "default", "control_token"),
           ({AGENT_ENV_ROLE_HEADER: DENISE}, DENISE, "header"), ({}, "default", "default")]
    async with _client(gw) as client:
        await _declare(client, BROOKE, "token", MEMBER)
        for path in ("/svc/shop/api/orders", "/website/shop/api/orders"):
            for headers, _, _ in who:
                assert (await client.post(path, json={}, headers=headers)).status_code == 200
        changes = (await client.get("/changes")).json()["changes"]

    expected = [(route, role, how) for route in ("svc", "website") for _, role, how in who]
    assert [(e["route"], e["role"], e["role_auth"]) for e in _events(tmp_path, "http_request")] == expected
    assert [(c["actors"][0]["request"]["route"], c["actors"][0]["role"], c["actors"][0]["role_auth"])
            for c in changes] == expected


@pytest.mark.asyncio
async def test_no_token_reaches_the_log_or_the_trajectory_on_any_path(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    gw = _gateway(tmp_path, monkeypatch, computer_digests={"desktop": digest(COMPUTER)})
    _Upstream(monkeypatch, changelog=gw._db_conn)
    control = {"Authorization": f"Bearer {CONTROL}"}
    callers = [{AGENT_TOKEN_HEADER: MEMBER}, {**control, AGENT_ENV_ROLE_HEADER: BROOKE},
               {AGENT_ENV_COMPUTER_HEADER: "desktop", COMPUTER_TOKEN_HEADER: COMPUTER},
               {AGENT_ENV_COMPUTER_HEADER: "desktop", COMPUTER_TOKEN_HEADER: STRANGER},
               {AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: STRANGER}]
    async with _client(gw) as client:
        await _declare(client, BROOKE, "token", MEMBER)
        for headers in callers:
            for tool in (TOOL.name, "list_website_urls"):
                await client.post("/step", json={"action": "call_tool", "tool_name": tool, "arguments": {}},
                                  headers=headers)
            await client.post("/step", json={"action": "list_tools"}, headers=headers)
            await client.post("/svc/shop/api/orders", json={}, headers=headers)
            await client.post("/website/shop/api/orders", json={}, headers=headers)
            await client.get("/browser/web/cdp/t0k3n/json/version", headers=headers)
        await client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, **callers[-1]})
    trajectory = (tmp_path / "trajectory.jsonl").read_text()
    assert {e["role_auth"] for e in json.loads(f"[{','.join(trajectory.splitlines())}]") if "role" in e} == {
        "agent_token", "control_token", "computer_token", "default"}
    for token in (MEMBER, CONTROL, COMPUTER, STRANGER):
        assert token not in caplog.text and token not in trajectory


# ---------------------------------------------------------------- an MCP client over loopback


async def _serve(app) -> tuple[uvicorn.Server, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, log_level="critical", access_log=False, timeout_graceful_shutdown=3)
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server, f"http://127.0.0.1:{sock.getsockname()[1]}"


@pytest.mark.asyncio
@pytest.mark.usefixtures("socket_enabled")
async def test_an_mcp_client_with_a_members_token_lists_and_calls_and_a_session_pinned_to_a_role_is_rechecked(
        tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    child = _Child()
    gw = _gateway(tmp_path, monkeypatch, child)
    server, url = await _serve(gw._asgi_app())
    control = {"Authorization": f"Bearer {CONTROL}"}
    try:
        async with _RealAsyncClient(base_url=url, headers=control) as harness:
            await _declare(harness, BROOKE, "token", MEMBER)
            async with (
                _RealAsyncClient(headers={AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: MEMBER},
                                 timeout=httpx.Timeout(10, read=60)) as http,
                streamable_http_client(f"{url}/mcp", http_client=http) as (read, write, _),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                listed = [tool.name for tool in (await session.list_tools()).tools]
                called = await session.call_tool(TOOL.name, {})
                state = (await harness.get("/roles")).json()
                await harness.request("DELETE", "/roles/members",
                                      json={"role": BROOKE, "member_id": state["roles"][0]["members"][0]})
                # its later requests name an open role, so only the session's own check stands in the way
                del http.headers[AGENT_TOKEN_HEADER]
                http.headers[AGENT_ENV_ROLE_HEADER] = DENISE
                refused = await session.call_tool(TOOL.name, {})
    finally:
        await server.shutdown()

    assert TOOL.name in listed and not called.isError and child.roles == [BROOKE]
    assert refused.isError and "isn't a member" in refused.content[0].text
    assert [(e["role"], e["role_auth"]) for e in _events(tmp_path, "tool_call")] == [(BROOKE, "agent_token")]
    trajectory = (tmp_path / "trajectory.jsonl").read_text()
    assert MEMBER not in caplog.text and MEMBER not in trajectory and CONTROL not in caplog.text


@pytest.mark.asyncio
@pytest.mark.usefixtures("socket_enabled")
async def test_a_session_opened_before_its_token_became_a_member_plays_the_role_once_it_is_one(tmp_path, monkeypatch):
    """The session keeps the digest of the token it presented, known then or not, so the order in which the harness
    protects the role, adds the member and the agent connects doesn't matter."""
    child = _Child()
    gw = _gateway(tmp_path, monkeypatch, child)
    server, url = await _serve(gw._asgi_app())
    try:
        async with (
            _RealAsyncClient(base_url=url, headers={"Authorization": f"Bearer {CONTROL}"}) as harness,
            _RealAsyncClient(headers={AGENT_ENV_ROLE_HEADER: BROOKE, AGENT_TOKEN_HEADER: MEMBER},
                             timeout=httpx.Timeout(10, read=60)) as http,
            streamable_http_client(f"{url}/mcp", http_client=http) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            while_open = await session.call_tool(TOOL.name, {})
            await _declare(harness, BROOKE, "token", MEMBER)
            once_a_member = await session.call_tool(TOOL.name, {})
            state = (await harness.get("/roles")).json()
            await harness.request("DELETE", "/roles/members",
                                  json={"role": BROOKE, "member_id": state["roles"][0]["members"][0]})
            # its later requests name an open role, so only the session's own check stands in the way
            http.headers[AGENT_ENV_ROLE_HEADER] = DENISE
            after_removal = await session.call_tool(TOOL.name, {})
    finally:
        await server.shutdown()

    assert not while_open.isError and not once_a_member.isError and child.roles == [BROOKE, BROOKE]
    assert [(e["role"], e["role_auth"]) for e in _events(tmp_path, "tool_call")] == [
        (BROOKE, "header"), (BROOKE, "agent_token")]
    assert after_removal.isError
    assert after_removal.content[0].text == f"refused: this agent token isn't a member of role '{BROOKE}'"


@pytest.mark.asyncio
@pytest.mark.usefixtures("socket_enabled")
async def test_a_non_ascii_role_header_is_decoded_once_as_latin_1_for_mcp_and_rest_alike(tmp_path, monkeypatch):
    """The served app decodes the role header once, as Starlette decodes headers, so an MCP session, /step and a
    service behind /svc act as the same role whatever bytes it holds."""
    raw = "zoë@example.com".encode()
    role = raw.decode("latin-1")
    child, upstream = _Child(), _Upstream(monkeypatch)
    gw = _gateway(tmp_path, monkeypatch, child)
    server, url = await _serve(gw._asgi_app())
    try:
        async with _RealAsyncClient(base_url=url, headers={AGENT_ENV_ROLE_HEADER: raw},
                                    timeout=httpx.Timeout(10, read=60)) as http:
            stepped = await _call(http, {})
            proxied = await http.post("/svc/shop/api/orders", json={})
            async with (streamable_http_client(f"{url}/mcp", http_client=http) as (read, write, _),
                        ClientSession(read, write) as session):
                await session.initialize()
                called = await session.call_tool(TOOL.name, {})
    finally:
        await server.shutdown()

    assert (stepped.status_code, proxied.status_code, called.isError) == (200, 200, False)
    assert child.roles == [role, role]
    assert upstream.requests[0].headers["agentenv-role"] == role
