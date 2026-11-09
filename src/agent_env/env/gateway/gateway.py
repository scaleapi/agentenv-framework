"""Gateway that aggregates MCP tools and REST proxies for website backends."""
from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import posixpath
import socket
import subprocess
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import parse_qsl, urljoin, urlsplit
from uuid import uuid4

import anyio
import httpx
import psycopg2
import uvicorn
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, ArgModelBase
from mcp.types import CallToolResult, TextContent, Tool as MCPTool
from pydantic import create_model
from starlette.requests import HTTPConnection, Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket

if TYPE_CHECKING:
    from . import InternalMCPServer

# Relative import — gateway runs as a standalone container with no agent_env.*
from .constants import (
    AGENT_ENV_HIDDEN_BY_DEFAULT_META_KEY,
    AGENT_ENV_ROLE_HEADER,
    AGENT_ENV_ROLE_META_KEY,  # noqa: F401  (the server module serves the whole vocabulary, used in this file or not)
    AGENT_ENV_SESSION_META_KEY,  # noqa: F401
    CARD_FETCH_TIMEOUT_S,
    CHANGES_PAGE_DEFAULT,
    CHANGES_PAGE_MAX,
    CHILD_DISCOVERY_TIMEOUT_S,
    DATA_PLANE_LOAD_TIMEOUT_MAX_S,
    DEFAULT_ROLE,
    EXT_WEBSITE_URI,
    GATEWAY_COMPOSE_HOST,
    GATEWAY_EXTENSIONS,
    GATEWAY_TRAJECTORY_FILE,
    MCP_TRANSPORT,
    METHOD_ADD,
    METHOD_GET,
    METHOD_RESET,
    PROTOCOL_VERSION,
    RPC_PATH,
    STATE_TOOL_NAME_CLASHES_KEY,
    STEP_SOURCE_POST_TURN_HOOK,
    TOOL_DISABLE_ACTION,
    TOOL_ENABLE_ACTION,
    WEBSITE_BROWSE_URL_PARAM,
    WELL_KNOWN_PATH,
    WILDCARD,
    GatewayMode,
    tool_name_clashes_text,
)
from .browser_proxy import PING_INTERVAL_S, BrowserRoutes, TokenRedaction
from .caller import Caller
from .changes import ChangeAttribution
from .child_sessions import ChildSessions, is_dead_session
from .clock import Clock, ClockError, _iso
from .get_time import GET_TIME_TOOL_NAME, build_get_time_tool
from .rest_proxy import (
    ROLE_HEADER_BYTES,
    UpstreamResponse,
    WebsiteRoute,
    fetch_child_card,
    forward,
    operations,
    refuse_websocket,
    relay_websocket,
    rewrite_child_card,
    route_name,
    upstream_url,
    website_routes,
)
from .triggers import TriggerEngine, TriggerError
from .notifications import CHECK_TOOL_NAMES, NotificationError, Notifications, build_check_tools

logger = logging.getLogger(__name__)

# RFC 9110's safe methods: a request using one asks for no change, so the proxy measures no write window for it.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
# A CDP address at the gateway: the website browser behind the /svc child <key>, discovery and its WebSocket.
BROWSER_ROUTE = "/browser/{key}/cdp/{token}/{path:path}"
# The loggers that write a request's path or a forwarded URL: uvicorn's, for HTTP requests and WebSocket handshakes,
# and httpx's, for each request the gateway forwards.
_TOKEN_LOGGERS = ("uvicorn.access", "uvicorn.error", "httpx")
_STEP_API = {"source": "step_api"}


class ToolNameClash(ValueError):
    """More than one source serves a tool under one name, so an agent couldn't tell which it calls: the gateway serves
    none of its servers' tools rather than a set missing some."""

    def __init__(self, clashes: dict[str, list[str]]):
        self.clashes = clashes
        super().__init__(f"Duplicate tool name{'s' if len(clashes) > 1 else ''} {tool_name_clashes_text(clashes)}")


class Gateway:
    """Gateway that aggregates MCP tools and REST proxies for website backends."""

    TOOL_CALL_TIMEOUT_S = 600.0
    STARTUP_DISCOVERY_TIMEOUT_S = 60.0  # startup discovery past it fails open, like a failed one; uvicorn must come up
    BARRIER_READ_AHEAD_MAX_BYTES = 8 * 1024 * 1024  # a larger sized body held by a barrier streams, its window to the last byte
    TRIGGER_BARRIER_TIMEOUT_S = 30.0  # a barrier may ask for less; past it the call is answered anyway
    # Must be >= the largest timeout any client can ask for, or the proxy severs a load the
    # client was still legitimately waiting on -- the ordering this module's constants file
    # documents (client add_data >= gateway REST proxy >= actual load time). Clients now
    # scale their own timeout by payload size, and the gateway can't cheaply know that size
    # per request, so it holds the ceiling for all of them.
    REST_PROXY_TIMEOUT_S = float(DATA_PLANE_LOAD_TIMEOUT_MAX_S)  # data-plane loads
    REST_PROXY_CONNECT_TIMEOUT_S = 10.0  # a service that never answers the TCP connect fails in seconds, not an hour
    REST_PROXY_TIMEOUT = httpx.Timeout(REST_PROXY_TIMEOUT_S, connect=REST_PROXY_CONNECT_TIMEOUT_S)

    def __init__(
        self,
        host: str,
        port: int,
        server_name: str,
        internal_mcp_servers: list[InternalMCPServer],
        website_urls: dict[str, str] | None = None,
        rest_proxy_urls: dict[str, str] | None = None,
        gateway_mode: GatewayMode = GatewayMode.PERFORMANCE,
        service_db_url: str | None = None,
        provider_headers: Iterable[str] = (),
    ):
        """`provider_headers` names the headers the sandbox provider in front of the gateway is sent, its credentials,
        which the gateway takes off every request it forwards to a service behind it."""
        self.host = host
        self.port = port
        self.server_name = server_name
        self.internal_mcp_servers = internal_mcp_servers
        self.website_urls = website_urls or {}
        self.rest_proxy_urls = rest_proxy_urls or {}
        self._site_mcp_endpoints = site_mcp_endpoints(internal_mcp_servers, self.website_urls, self.rest_proxy_urls)
        self._site_reserved: dict[str, frozenset[str]] = {}
        self._withheld = frozenset(name.strip().lower().encode() for name in provider_headers if name.strip())
        self._browsers = BrowserRoutes(self.rest_proxy_urls, self.REST_PROXY_TIMEOUT, self._log_event,
                                       self._attribution, self._request_role, withheld=self._withheld)
        for name in _TOKEN_LOGGERS:  # each names a CDP address's path, its token in it, at INFO
            log = logging.getLogger(name)
            if not any(isinstance(f, TokenRedaction) for f in log.filters):
                log.addFilter(TokenRedaction())
        self.gateway_mode = gateway_mode
        self._service_db_url = service_db_url
        self._website_routes: dict[str, list[WebsiteRoute] | None] = {}
        self._db_conn: psycopg2.extensions.connection | None = None
        if service_db_url:
            self._db_conn = psycopg2.connect(service_db_url)
            self._db_conn.autocommit = True
        # Each child's session, shared by the MCP proxy, /step and triggers (its locks are taken after _discover_lock)
        self._sessions = ChildSessions(self.internal_mcp_servers)
        self._discover_lock = asyncio.Lock()
        self._server_tools: dict[str, list[MCPTool]] = {}
        self._tools_discovered = False
        self._tool_name_clashes: dict[str, list[str]] = {}  # found once, by a discovery that then registered nothing
        self._tool_server_urls: dict[str, str] = {}
        self._tool_call_lock: asyncio.Lock | None = asyncio.Lock() if self.gateway_mode == GatewayMode.CONSISTENT else None
        self._role_rules: dict[str, dict[str, bool]] = {}
        self._hidden_by_default: set[str] = set()  # tools a server listed hidden: only rules naming them apply
        self._role_rules_lock = asyncio.Lock()

        self._event_counter = 0
        self._event_lock = asyncio.Lock()
        self._changes = ChangeAttribution()
        Path(GATEWAY_TRAJECTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
        self._trajectory_file = open(GATEWAY_TRAJECTORY_FILE, "a")
        atexit.register(self._close_trajectory_file)

        self._mcp = FastMCP(server_name, lifespan=self._session_lifespan)
        self._mcp.settings.host = host
        self._mcp.settings.port = port
        self._mcp.settings.transport_security.enable_dns_rebinding_protection = False
        self._trigger_engine = TriggerEngine(self)
        if self.website_urls:
            @self._mcp.tool(name="list_website_urls", description="List available website URLs that can be browsed.")
            async def list_website_urls() -> list[str]:
                return list(self.website_urls.values())
            self._server_tools.setdefault("gateway", []).append(
                MCPTool(
                    name="list_website_urls",
                    description="List available website URLs that can be browsed.",
                    inputSchema={"type": "object", "properties": {}},
                )
            )
        self._clock = Clock()
        self._notifications = Notifications(lambda: self._clock)
        self._env_get_time_url_cache: str | None = None
        self._install_role_filters()

        if self.rest_proxy_urls:
            @self._mcp.custom_route(
                "/svc/{service_name}/{path:path}",
                methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            )
            async def proxy_to_service(request: Request) -> Response:
                return await self._proxy_rest_request(request)

            @self._mcp.custom_route(BROWSER_ROUTE, methods=["GET", "HEAD"])
            async def browser_discovery(request: Request) -> Response:
                return await self._browsers.discovery(request)

        if self.website_urls:
            @self._mcp.custom_route(
                "/website/{service_name}/{path:path}",
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            )
            async def proxy_to_frontend(request: Request) -> Response:
                return await self._proxy_website_request(request)

            @self._mcp.custom_route(
                "/website/{service_name}",
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            )
            async def proxy_to_frontend_root(request: Request) -> Response:
                return await self._proxy_website_request(request)

            self._mcp._custom_starlette_routes += [
                WebSocketRoute("/website/{service_name}/{path:path}", self._proxy_website_socket),
                WebSocketRoute("/website/{service_name}", self._proxy_website_socket),
            ]

        @self._mcp.custom_route(f"/tools/{TOOL_DISABLE_ACTION}", methods=["POST"])
        async def tools_disable(request: Request) -> Response:
            return await self._handle_role_mutation(request, action=TOOL_DISABLE_ACTION)

        @self._mcp.custom_route(f"/tools/{TOOL_ENABLE_ACTION}", methods=["POST"])
        async def tools_enable(request: Request) -> Response:
            return await self._handle_role_mutation(request, action=TOOL_ENABLE_ACTION)

        @self._mcp.custom_route("/state", methods=["GET"])
        async def get_state(request: Request) -> Response:
            try:
                await self._ensure_tools_discovered()
            except Exception:
                pass  # already logged; return whatever state we have
            mcp_servers = [
                {
                    "name": server_name,
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        }
                        for tool in tools
                    ],
                }
                for server_name, tools in self._server_tools.items()
            ]
            changelog_id = self._query_changelog_id()
            roles = {role: self._format_role_rules(rules) for role, rules in self._role_rules.items()}
            return self._json_response({"mcp_servers": mcp_servers, "changelog_id": changelog_id, "roles": roles,
                                        STATE_TOOL_NAME_CLASHES_KEY: self._tool_name_clashes}, 200)

        @self._mcp.custom_route("/step", methods=["POST"])
        async def step_endpoint(request: Request) -> Response:
            return await self._handle_step(request)

        async def _trigger_write(request: Request, method, prepare=None) -> Response:
            """Shared body for the trigger write routes: parse JSON, await `prepare` with it, invoke the engine
            method, map a TriggerError (bad config) to a 400."""
            try:
                body = await request.json()
            except Exception:
                return self._json_response({"ok": False, "error": "Invalid JSON body"}, 400)
            if prepare is not None:
                await prepare(body)
            try:
                return self._json_response(method(body), 200)
            except TriggerError as e:
                return self._json_response({"ok": False, "error": str(e)}, 400)

        @self._mcp.custom_route("/triggers/register", methods=["POST"])
        async def triggers_register(request: Request) -> Response:
            return await _trigger_write(request, self._trigger_engine.register, prepare=self._read_trigger_website_routes)

        @self._mcp.custom_route("/triggers/remove", methods=["POST"])
        async def triggers_remove(request: Request) -> Response:
            return await _trigger_write(request, self._trigger_engine.remove)

        @self._mcp.custom_route("/triggers/clear", methods=["POST"])
        async def triggers_clear(request: Request) -> Response:
            return self._json_response(self._trigger_engine.clear(), 200)

        @self._mcp.custom_route("/triggers/state", methods=["GET"])
        async def triggers_state(request: Request) -> Response:
            return self._json_response(self._trigger_engine.state(), 200)

        @self._mcp.custom_route("/clock/set-time", methods=["PUT"])
        async def clock_set_time(request: Request) -> Response:
            try:
                body = await request.json()
            except Exception:
                return self._json_response({"ok": False, "error": "Invalid JSON body"}, 400)
            if GET_TIME_TOOL_NAME in self._tool_server_urls:
                return self._json_response(
                    {"ok": False, "error": f"'{GET_TIME_TOOL_NAME}' is already registered by a backing server"}, 409)
            try:
                state = self._clock.set_time(body.get("virtual_time"), body.get("virtual_seconds_per_real_second"))
            except ClockError as e:
                return self._json_response({"ok": False, "error": str(e)}, 400)
            self._register_get_time_tool()
            return self._json_response(state, 200)

        @self._mcp.custom_route("/clock/advance", methods=["PUT"])
        async def clock_advance(request: Request) -> Response:
            try:
                body = await request.json()
            except Exception:
                return self._json_response({"ok": False, "error": "Invalid JSON body"}, 400)
            if not isinstance(body, dict):
                return self._json_response({"ok": False, "error": "body must be a JSON object"}, 400)
            try:
                state = self._clock.advance(body.get("virtual_time"))
            except ClockError as e:
                return self._json_response({"ok": False, "error": str(e)}, 400)
            return self._json_response({**state, "due": self._trigger_engine.clock_advanced()}, 200)

        @self._mcp.custom_route("/clock/clear", methods=["POST"])
        async def clock_clear(request: Request) -> Response:
            self._unregister_get_time_tool()
            state = self._clock.clear()
            return self._json_response(state, 200)

        @self._mcp.custom_route("/clock/time", methods=["GET"])
        async def clock_time(request: Request) -> Response:
            try:
                return self._json_response(self._clock.read(), 200)
            except ClockError:
                return self._json_response({"ok": False, "error": "clock not armed"}, 404)

        @self._mcp.custom_route("/clock/state", methods=["GET"])
        async def clock_state(request: Request) -> Response:
            st = self._clock.state()
            if st.get("armed"):
                st["env_get_time_url"] = await self._env_get_time_url()
            return self._json_response(st, 200)

        async def _notifications_write(request: Request, method) -> Response:
            try:
                body = await request.json()
            except Exception:
                return self._json_response({"ok": False, "error": "Invalid JSON body"}, 400)
            try:
                return self._json_response(method(body), 200)
            except NotificationError as e:
                return self._json_response({"ok": False, "error": str(e)}, 400)

        @self._mcp.custom_route("/notifications/config", methods=["POST"])
        async def notifications_config(request: Request) -> Response:
            try:
                body = await request.json()
            except Exception:
                return self._json_response({"ok": False, "error": "Invalid JSON body"}, 400)
            # A config that names roles registers the tools, so the backing servers' tools must be known (or a later
            # discovery would replace ours unseen) and none may have their names. One that turns checks off
            # (roles: [], the default) registers nothing, so neither can block it.
            if not (isinstance(body, dict) and body.get("roles", []) == []):
                try:
                    await self._ensure_tools_discovered()
                except Exception as e:
                    return self._json_response(
                        {"ok": False, "error": f"tool discovery has not completed ({type(e).__name__}); retry"}, 503)
                taken = sorted(set(CHECK_TOOL_NAMES) & set(self._tool_server_urls))
                if taken:
                    return self._json_response(
                        {"ok": False, "error": f"{taken} already registered by a backing server"}, 409)
            try:
                state = self._notifications.configure(body)
            except NotificationError as e:
                return self._json_response({"ok": False, "error": str(e)}, 400)
            if self._notifications.enabled:
                self._register_check_tools()
            else:
                self._unregister_check_tools()
            return self._json_response(state, 200)

        @self._mcp.custom_route("/notifications/state", methods=["GET"])
        async def notifications_state(request: Request) -> Response:
            try:  # ?to=<role> for one recipient's; every recipient's without it
                return self._json_response(self._notifications.state(request.query_params.get("to")), 200)
            except NotificationError as e:
                return self._json_response({"ok": False, "error": str(e)}, 400)

        @self._mcp.custom_route("/notifications/take", methods=["POST"])
        async def notifications_take(request: Request) -> Response:
            return await _notifications_write(request, self._notifications.take)

        @self._mcp.custom_route("/notifications/clear", methods=["POST"])
        async def notifications_clear(request: Request) -> Response:
            self._unregister_check_tools()
            return self._json_response(self._notifications.clear(), 200)

        @self._mcp.custom_route("/trajectory", methods=["GET"])
        async def env_trajectory(request: Request) -> Response:
            return self._serve_trajectory()

        @self._mcp.custom_route("/changes", methods=["GET"])
        async def env_changes(request: Request) -> Response:
            return await self._serve_changes(request)

        @self._mcp.custom_route(WELL_KNOWN_PATH, methods=["GET"])
        async def get_env_card(request: Request) -> Response:
            return await self._serve_env_card()

        @self._mcp.custom_route(RPC_PATH, methods=["POST"])
        async def env_data_plane(request: Request) -> Response:
            return await self._serve_env_data_plane(request)

    def run(self) -> None:
        logger.info(f"Starting Gateway on {self.host}:{self.port} (mode={self.gateway_mode.value})")
        config = uvicorn.Config(
            self._asgi_app(),
            host=self._mcp.settings.host,
            port=self._mcp.settings.port,
            log_level=self._mcp.settings.log_level.lower(),
            ws_ping_interval=PING_INTERVAL_S,
            ws_ping_timeout=PING_INTERVAL_S,
        )
        server = uvicorn.Server(config)
        if self._resolve_i6pn_host():
            # On Modal's i6pn, serve both families: IPv4 for the orchestrator's HTTPS tunnel and
            # IPv6 for backing servers reaching the gateway over the i6pn mesh. A lone `::` bind
            # comes up IPv6-only on this image and starves the tunnel.
            anyio.run(server.serve, [self._listen_socket(socket.AF_INET, "0.0.0.0"),
                                     self._listen_socket(socket.AF_INET6, "::")])
        else:
            anyio.run(server.serve)

    # The three scopes a Caller is bound in, innermost first. Request: the served app binds the role the
    # AgentEnv-Role header named. Session: mcp's Server.run enters the session lifespan once per inbound MCP
    # session, in the session's server task before any handler task is spawned, so the key bound there is inherited
    # by every tools/call of that session (and the role is the one its initialize request carried, which that task
    # inherited). Process: child sessions and the time-trigger driver, outer to the MCP session manager's lifespan.

    def _asgi_app(self):
        """The served app: FastMCP's Starlette app with the request's caller bound around it, and the process
        lifespan wrapped around the MCP session manager's."""
        inner = self._mcp.streamable_http_app()
        inner.router.lifespan_context = self._process_lifespan(inner.router.lifespan_context)
        if self.rest_proxy_urls:
            inner.router.routes.append(WebSocketRoute(BROWSER_ROUTE, self._browsers.relay))

        async def app(scope, receive, send):
            if scope["type"] != "http":
                await inner(scope, receive, send)
                return
            role = DEFAULT_ROLE
            for name, value in scope.get("headers", ()):
                if name == ROLE_HEADER_BYTES:
                    role = value.decode(errors="ignore").strip() or DEFAULT_ROLE
                    break
            token = Caller(role=role, session=Caller.current().session).bind()
            try:
                await inner(scope, receive, send)
            finally:
                Caller.reset(token)

        return app

    @asynccontextmanager
    async def _session_lifespan(self, mcp: FastMCP):
        """Once per inbound MCP session: the session's key joins the caller, and a connect gets to heal a failed
        startup discovery (tools/list is synchronous and cannot). Nothing process-scoped lives here."""
        Caller(role=Caller.current().role, session=uuid4().hex).bind()
        try:
            await self._ensure_tools_discovered()
        except ToolNameClash as e:
            logger.error(f"MCP session started on an env serving none of its servers' tools: {e}")
        except Exception:
            logger.exception("Tool discovery failed at MCP session start; proxy tools stay unavailable until it succeeds")
        yield

    def _process_lifespan(self, mcp_lifespan):
        """Once per process (uvicorn's lifespan), outer to the MCP session manager's: at shutdown the MCP sessions end
        first, then the driver stops and the scheduled trigger actions are cancelled, then the child sessions close."""
        @asynccontextmanager
        async def lifespan(app):
            try:
                await asyncio.wait_for(self._ensure_tools_discovered(), timeout=self.STARTUP_DISCOVERY_TIMEOUT_S)
            except ToolNameClash:
                pass  # logged by discovery; the gateway stays up to report the clash on /state
            except Exception:
                # a raised startup makes uvicorn exit the process; discovery is retried on first use instead
                logger.exception("Tool discovery failed at startup; retrying on first use")
            self._trigger_engine.start_driver()
            try:
                async with mcp_lifespan(app):
                    yield
            finally:
                self._sessions.closing = True
                await self._trigger_engine.stop_driver()
                await self._trigger_engine.cancel_scheduled()
                await self._close_child_sessions()

        return lifespan

    def _listen_socket(self, family: int, host: str) -> socket.socket:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind((host, self.port))
        sock.listen()
        return sock

    def _is_disabled(self, role: str, name: str) -> bool:
        """Walk the rule chain (role first, then WILDCARD) returning the first concrete True/False; default False.
        A tool listed hidden by default skips "*"-tool rules: only rules naming it apply, and with none it is hidden.
        The scheduled-check tools exist only for the roles the notifications config names, whatever the rules say."""
        if name in CHECK_TOOL_NAMES and name not in self._tool_server_urls and role not in self._notifications.roles:
            return True
        hidden = name in self._hidden_by_default
        for r in (role, WILDCARD):
            rules = self._role_rules.get(r, {})
            if name in rules:
                return rules[name]
            if WILDCARD in rules and not hidden:
                return rules[WILDCARD]
        return hidden

    def _install_role_filters(self) -> None:
        """Wrap the FastMCP tool manager so MCP tools/list and tools/call honor the current caller's role."""
        tm = self._mcp._tool_manager
        original_list_tools = tm.list_tools
        original_call_tool = tm.call_tool

        def filtered_list_tools():
            role = Caller.current().role
            return [t for t in original_list_tools() if not self._is_disabled(role, t.name)]

        async def filtered_call_tool(name, arguments, context=None, convert_result=False):
            role = Caller.current().role
            if self._is_disabled(role, name):
                raise ToolError(f"Tool '{name}' is disabled for role '{role}'")
            native = name not in self._tool_server_urls
            event_id = await self._log_native_tool_call(name, arguments, role) if native else None
            result = await original_call_tool(name, arguments, context=context, convert_result=convert_result)
            if native:
                await self._log_native_tool_result(event_id, result)
            barrier_tasks, barrier_timeout_s = self._fire_triggers(role, name, arguments, result)
            await self._await_barriers(barrier_tasks, barrier_timeout_s)
            return result

        tm.list_tools = filtered_list_tools
        tm.call_tool = filtered_call_tool

    def _fire_triggers(self, role: str, name: str, arguments: dict | None,
                       result: Any) -> tuple[list[asyncio.Task], float | None]:
        """Feed a completed external tool call to the trigger engine (fail-open). Hooked at every
        external boundary (MCP filter + both `/step` handlers); the engine's `_internal_call` reads
        bypass these, so internal reads never re-trigger (no recursion). Only watched roles fire."""
        try:
            return self._trigger_engine.on_tool_call(role, name, arguments or {}, result)
        except Exception:
            logger.exception("Trigger evaluation failed (fail-open)")
            return [], None

    async def _await_barriers(self, pending: list[asyncio.Task], timeout_s: float | None = None) -> None:
        """Block until the pending tasks (i.e. barriers) finish, for a maximum of `timeout_s`."""
        if not pending:
            return
        timeout = timeout_s or self.TRIGGER_BARRIER_TIMEOUT_S
        try:
            _, still_running = await asyncio.wait(pending, timeout=timeout)
            if still_running:
                trigger_ids = sorted(t.get_name() for t in still_running)
                logger.warning(f"barrier fire(s) {trigger_ids} still running after {timeout}s; "
                               "answering the provoking call")
                await self._log_event({"event_type": "trigger_barrier_timeout", "source": "trigger_engine",
                                       "at": "provoking_call", "trigger_ids": trigger_ids, "timeout_s": timeout})
        except Exception:
            logger.exception("Awaiting trigger barriers failed (fail-open)")

    async def _env_get_time_url(self) -> str:
        """The gateway's own server-reachable /clock/time URL (cached, resolved off the event loop): modal i6pn if resolvable, else the compose service name."""
        if self._env_get_time_url_cache is None:
            host = await asyncio.to_thread(self._resolve_i6pn_host) or GATEWAY_COMPOSE_HOST
            self._env_get_time_url_cache = f"http://{host}:{self.port}/clock/time"
        return self._env_get_time_url_cache

    @staticmethod
    def _resolve_i6pn_host() -> str | None:
        """Resolve this container's own Modal i6pn IPv6 (bracketed for a URL), or None outside Modal."""
        try:
            out = subprocess.run(["getent", "ahostsv6", "i6pn.modal.local"],
                                 capture_output=True, text=True, timeout=5)
            addr = out.stdout.split()[0] if out.stdout.strip() else None
            return f"[{addr}]" if addr else None
        except Exception:
            return None

    @asynccontextmanager
    async def _tool_call_guard(self):
        if self._tool_call_lock is not None:
            async with self._tool_call_lock:
                yield
        else:
            yield

    def _query_changelog_id(self) -> int | None:
        if self._db_conn is None:
            return None
        try:
            with self._db_conn.cursor() as cur:
                cur.execute("SELECT MAX(id) FROM public._changelog")
                row = cur.fetchone()
                return row[0] if row else None
        except Exception:
            logger.debug("Changelog ID query failed", exc_info=True)
            try:
                self._db_conn = psycopg2.connect(self._service_db_url)
                self._db_conn.autocommit = True
            except Exception:
                self._db_conn = None
            return None

    async def _proxy_rest_request(self, request: Request, url_map: dict[str, str] | None = None) -> Response:
        """Reverse-proxy an HTTP request to an internal service, streaming the response through."""
        url_map = self.rest_proxy_urls if url_map is None else url_map
        service_name = request.path_params["service_name"]
        base_url = url_map.get(service_name)
        if base_url is None:
            return self._json_response({"error": f"Unknown service: {service_name}"}, 404)
        role = self._request_role(request)
        site_mcp = self._site_mcp_endpoints.get(service_name) if url_map is self.rest_proxy_urls else None
        mcp_server = url_map is self.rest_proxy_urls and (service_name.startswith("mcp-") or site_mcp is not None)
        mcp_endpoint = site_mcp or "mcp"
        if url_map is self.rest_proxy_urls and (cdp := await self._browsers.cdp_answer(request, base_url, role)) is not None:
            return cdp
        if request.method in _SAFE_METHODS:
            return await forward(request, base_url, role=role, timeout=self.REST_PROXY_TIMEOUT, mcp_server=mcp_server,
                                 withheld=self._withheld, mcp_endpoint=mcp_endpoint)
        changelog_before = self._query_changelog_id()
        attribution = self._attribution(role)  # at the start: a fire may be gone by the time the response ends

        async def log_write(status: int) -> None:
            changelog_after = self._query_changelog_id()
            if changelog_after is None or changelog_after <= (changelog_before or 0):
                return  # only a request that wrote changelog rows is a write window
            path = "/" + request.path_params.get("path", "")
            op = (self._website_op(service_name, request.method, site_request_path(request) or path)
                  if url_map is self.website_urls else None)
            await self._log_event({
                "event_type": "http_request",
                "route": "website" if url_map is self.website_urls else "svc",
                "service": service_name,
                "method": request.method,
                "path": path,
                **({"op": op} if op else {}),
                "status": status,
                **attribution,
                "changelog_id_before": changelog_before,
                "changelog_id_after": changelog_after,
            })

        return await forward(request, base_url, role=role, timeout=self.REST_PROXY_TIMEOUT, mcp_server=mcp_server,
                             on_done=log_write, withheld=self._withheld, mcp_endpoint=mcp_endpoint)

    async def _proxy_website_request(self, request: Request) -> Response:
        """Proxy a website request; with a website or state trigger registered, feed it to the engine and hold barriers.
        A site's data plane and card, and every path its card advertises for agentenv (``site_reserved_paths``), are
        never reached through its route, so an agent browsing the site can't read its data or what grades it; agentenv
        reaches them through /svc/<name>/."""
        reserved = await self._site_reserved_paths(request.path_params["service_name"])
        if reserved is None:
            return self._json_response({"error": "website card unavailable"}, 503)
        if reserved_site_request(request, reserved):
            return self._json_response({"error": "not found"}, 404)
        response = await self._proxy_rest_request(request, url_map=self.website_urls)
        app = request.path_params["service_name"]
        engine = self._trigger_engine
        if app in self.website_urls and (engine.has_website_triggers or engine.has_state_triggers):
            path = site_request_path(request) or "/" + request.path_params.get("path", "")
            if isinstance(response, UpstreamResponse):
                if not engine.may_hold_website_request(self._request_role(request), app, request.method, path,
                                                       response.status_code, self._website_op(app, request.method, path)):
                    response.after_body(lambda status: self._fire_website_triggers(request, app, status))  # answered at its last byte
                    return response
                await response.read_ahead_if_sized(self.BARRIER_READ_AHEAD_MAX_BYTES)  # its window ends before its triggers run
            barrier_tasks, barrier_timeout_s = await self._fire_website_triggers(request, app, response.status_code)
            await self._await_barriers(barrier_tasks, barrier_timeout_s)
        return response

    async def _proxy_website_socket(self, websocket: WebSocket) -> None:
        """Relay a WebSocket on a site's route to its server, as its HTTP requests are: its data plane and card, and
        every path its card advertises for agentenv (``site_reserved_paths``), are refused, and it goes upstream with
        the role, role key and forwarded address an HTTP request would carry."""
        reserved = await self._site_reserved_paths(websocket.path_params["service_name"])
        if reserved is None:
            await refuse_websocket(websocket, self._json_response({"error": "website card unavailable"}, 503))
            return
        if reserved_site_request(websocket, reserved):
            await refuse_websocket(websocket, self._json_response({"error": "not found"}, 404))
            return
        service_name = websocket.path_params["service_name"]
        base_url = self.website_urls.get(service_name)
        if base_url is None:
            await refuse_websocket(websocket, self._json_response({"error": f"Unknown service: {service_name}"}, 404))
            return
        await relay_websocket(websocket, base_url, role=self._request_role(websocket),
                              open_timeout=self.REST_PROXY_CONNECT_TIMEOUT_S, withheld=self._withheld)

    async def _fire_website_triggers(self, request: Request, app: str,
                                     status: int) -> tuple[list[asyncio.Task], float | None]:
        """Feed an answered website request to the trigger engine (fail-open)."""
        try:
            # Only a website trigger reads the request's args; a state re-check needs just the method and status.
            args = _request_args(request, await request.body()) if self._trigger_engine.has_website_triggers else {}
            path = site_request_path(request) or "/" + request.path_params.get("path", "")
            return self._trigger_engine.on_website_request(self._request_role(request), app, request.method, path,
                                                           status, args, self._website_op(app, request.method, path))
        except Exception:
            logger.exception("Trigger evaluation failed (fail-open)")
            return [], None

    @staticmethod
    def _request_role(request: HTTPConnection) -> str:
        """The role the gateway attributes a REST request to: the first AgentEnv-Role header, else the default role."""
        return request.headers.get(AGENT_ENV_ROLE_HEADER, "").strip() or DEFAULT_ROLE

    async def _read_trigger_website_routes(self, body: Any) -> None:
        """Read the routes of each website a website trigger in a register `body` names an `op` of, so registration
        can check the op and requests can be matched by it; one whose card could not be read before is read again."""
        triggers = body.get("triggers") if isinstance(body, dict) else None
        apps = {trigger["when"]["app"] for trigger in triggers if isinstance(trigger, dict)
                and isinstance(trigger.get("when"), dict) and trigger["when"].get("type") == "website"
                and trigger["when"].get("op") is not None and isinstance(trigger["when"].get("app"), str)
                and trigger["when"]["app"] in self.website_urls} if isinstance(triggers, list) else set()
        await asyncio.gather(*(self._read_website_routes(app, again=True) for app in apps))

    async def _read_website_routes(self, app: str, *, again: bool = False) -> None:
        """Read website `app`'s routes from its card once (`again`: once more, when its card could not be read). The
        env card's composition reads them too, so a deploy, which reads that card, leaves them read."""
        if app in self._website_routes and not (again and self._website_routes[app] is None):
            return
        async with httpx.AsyncClient() as client:
            card = await fetch_child_card(client, app, self.rest_proxy_urls.get(app) or self.website_urls[app])
        self._website_routes[app] = website_routes(card) if card is not None else None

    def website_route_names(self, app: str) -> set[str] | None:
        """The names of the routes website `app`'s card names, as last read; None when it has not been read."""
        routes = self._website_routes.get(app)
        return None if routes is None else {route.name for route in routes if route.name is not None}

    def _website_op(self, app: str, method: str, path: str) -> str | None:
        """The name of the route of website `app` a request reached, by the routes last read from its card; never a
        read of its own, so answering a request waits on no card."""
        return route_name(self._website_routes.get(app) or [], method, path)

    async def _site_reserved_paths(self, app: str) -> frozenset[str] | None:
        """The paths ``app``'s card advertises for agentenv (``site_reserved_paths``), read once from its server, the one
        its /svc/ route reaches (else the one its website route does), and kept; none for a site serving no card, one
        whose card path answers a 4xx or what isn't a JSON object, and None, read again next time, while its card can't
        be read, unreachable or answering a 5xx, so its route serves nothing until it knows what to refuse."""
        if app in self._site_reserved:
            return self._site_reserved[app]
        base_url = self.rest_proxy_urls.get(app) or self.website_urls.get(app)
        if base_url is None:
            return frozenset()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url.rstrip('/')}{WELL_KNOWN_PATH}", timeout=CARD_FETCH_TIMEOUT_S)
            if response.status_code >= 500:
                response.raise_for_status()
        except Exception as e:
            logger.warning(f"website {app}: its card could not be read ({type(e).__name__}: {e}), so its route answers "
                           "503 until it can be")
            return None
        try:
            card = response.json() if response.is_success else None
        except ValueError:
            card = None
        address = self.website_urls.get(app) or base_url
        self._site_reserved[app] = site_reserved_paths(card, address) if isinstance(card, dict) else frozenset()
        return self._site_reserved[app]

    def _backing_keys(self) -> list[str]:
        # rest_proxy_urls entries are the env's backing servers: mcp-<svc> (MCP) + <svc> (website
        # backends). Non-v1 backers (e.g. the injected website_browser) self-omit from the card via 404.
        return list(self.rest_proxy_urls)

    def _environment_name(self, key: str) -> str:
        """The name the env behind the backing server ``key`` is registered as (``route_environment_name``)."""
        return route_environment_name(key, self.rest_proxy_urls.get(key, ""), self.internal_mcp_servers,
                                      website=key in self.website_urls)

    def _rewrite_child_card(self, key: str, card: dict) -> dict:
        """One backing server's card rooted at the gateway (``rewrite_child_card``), stating the ``environment_name``
        the env behind it is registered as, which its own card's name needn't be; a website's ``website/v1`` params
        also name the route the gateway serves it to a browser at, ``/website/<key>/``, as ``browse_url``."""
        card = rewrite_child_card(key, card, self._gateway_extensions())
        card["environment_name"] = self._environment_name(key)
        if key in self.website_urls:
            for ext in (card.get("capabilities") or {}).get("extensions") or []:
                if isinstance(ext, dict) and ext.get("uri") == EXT_WEBSITE_URI:
                    params = ext.get("params") if isinstance(ext.get("params"), dict) else {}
                    ext["params"] = {**params, WEBSITE_BROWSE_URL_PARAM: f"/website/{key}/"}
        return card

    async def _fetch_child_card(self, client: httpx.AsyncClient, key: str, base_url: str) -> dict | None:
        """Fetch + rewrite one backing server's card; None (omit) on 404 / error / timeout / malformed JSON. A
        website's card read here also refreshes the routes its requests are named by."""
        child = await fetch_child_card(client, key, base_url)
        if child is not None and key in self.website_urls:
            self._website_routes[key] = website_routes(child)
        return None if child is None else self._rewrite_child_card(key, child)

    def _gateway_extensions(self) -> list[dict]:
        return GATEWAY_EXTENSIONS

    async def _serve_env_card(self) -> Response:
        """Compose the env card: the gateway's own extensions + each backing server's card nested
        under children_environments (endpoints rewritten to gateway-rooted)."""
        keys = self._backing_keys()
        children: list[dict] = []
        if keys:
            async with httpx.AsyncClient() as client:
                results = await asyncio.gather(
                    *(self._fetch_child_card(client, k, self.rest_proxy_urls[k]) for k in keys),
                    return_exceptions=True,
                )
            for key, result in zip(keys, results):
                if isinstance(result, BaseException):
                    logger.warning(f"env card: child {key} raised {result!r}; omitting")
                elif result is not None:
                    children.append(result)
        card = {
            "name": self.server_name,
            "protocolVersion": PROTOCOL_VERSION,
            "url": RPC_PATH,
            "preferredTransport": "JSONRPC",
            "additionalInterfaces": [{"url": self._mcp.settings.streamable_http_path, "transport": MCP_TRANSPORT}],
            "capabilities": {"extensions": self._gateway_extensions()},
            "children_environments": children,
        }
        return self._json_response(card, 200)

    def _jsonrpc_error(self, request_id: Any, code: int, message: str) -> Response:
        return self._json_response({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}, 200)

    async def _serve_env_data_plane(self, request: Request) -> Response:
        """Env-level v1 data plane: forward data/reset|add|get to the single backing MCP server
        (`-32000` unless exactly one). Multi-child fan-out is a MultiEnv follow-up."""
        try:
            body = await request.json()
        except Exception:
            return self._jsonrpc_error(None, -32700, "Parse error")
        if not isinstance(body, dict):
            return self._jsonrpc_error(None, -32600, "Invalid request")
        request_id = body.get("id")
        method = body.get("method")
        if method not in (METHOD_RESET, METHOD_ADD, METHOD_GET):
            return self._jsonrpc_error(request_id, -32601, f"Method not found: {method}")
        keys = self._backing_keys()
        if len(keys) != 1:
            return self._jsonrpc_error(request_id, -32000, f"env-level {method} requires exactly one backing server, found {len(keys)}")
        target_url = f"{self.rest_proxy_urls[keys[0]].rstrip('/')}{RPC_PATH}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(target_url, json=body, timeout=self.REST_PROXY_TIMEOUT_S)
        except Exception as e:
            logger.error(f"env data plane forward to {target_url} failed: {e}")
            return self._jsonrpc_error(request_id, -32000, f"forward failed: {e}")
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    async def _ensure_child_sessions(self) -> dict[str, ClientSession]:
        return await self._sessions.sessions()

    async def _close_child_sessions(self) -> None:
        await self._sessions.close()

    async def _call_child_tool(self, server_url: str, tool_name: str, arguments: dict, *,
                               caller: Caller | None = None) -> CallToolResult:
        """Every child tools/call (MCP proxy, /step, trigger engine), with the caller's `_meta` stamp (none for an
        unattributed trigger action), on its child's session (see child_sessions). Discovery is not redone on a
        reconnect; the registered proxies look their child up by URL."""
        meta = caller.meta() if caller is not None else None
        return await self._sessions.call(server_url, tool_name, arguments, meta, timeout=self.TOOL_CALL_TIMEOUT_S)

    async def _ensure_tools_discovered(self) -> None:
        """Idempotent discovery over the process's child sessions (startup, MCP session start, /step, /state, triggers)."""
        if self._tools_discovered:
            return
        async with self._discover_lock:
            if self._tools_discovered:
                return
            if self._tool_name_clashes:  # the servers' names don't change, so neither does a clash
                raise ToolNameClash(self._tool_name_clashes)
            sessions = await self._ensure_child_sessions()
            try:
                await self._discover_and_register_tools(sessions)
            except ToolNameClash as e:
                self._tool_name_clashes = e.clashes
                logger.error(f"Tool discovery refused: {e}. The env serves none of its servers' tools")
                raise
            except Exception:
                logger.exception("Tool discovery failed")
                for server in self.internal_mcp_servers:
                    self._server_tools.pop(server.name, None)  # the gateway's own entry stays
                self._tools_discovered = False
                raise

    async def _discover_and_register_tools(self, sessions: dict[str, ClientSession]) -> None:
        """Discover tools from all internal servers via persistent sessions and register proxies: for every tool once
        every server has listed and no two sources share a name (ToolNameClash), else for none."""
        logger.info(f"Discovering tools from {len(self.internal_mcp_servers)} internal server(s)...")
        listed: list[tuple[InternalMCPServer, list[MCPTool]]] = []
        for server in self.internal_mcp_servers:
            try:
                session = sessions[server.mcp_url]
                # bounded: this runs under _discover_lock, which startup and every MCP connect wait on
                tools_result = await asyncio.wait_for(session.list_tools(), timeout=CHILD_DISCOVERY_TIMEOUT_S)
            except Exception as e:
                logger.error(f"Failed to discover tools from {server.name}: {e}")
                child = self._sessions.current.get(server.mcp_url)
                if is_dead_session(e) and child is not None and child.session is session:
                    await self._sessions.replace(child)
                raise
            logger.info(f"Found {len(tools_result.tools)} tool(s) from {server.name}")
            listed.append((server, tools_result.tools))
        # get_time and the scheduled-check tools are left out: a backing server's tool of that name wins, and the
        # clock and the checks refuse to turn on over it
        sources: dict[str, list[str]] = {
            tool.name: ["gateway"] for tool in self._server_tools.get("gateway", [])
            if tool.name != GET_TIME_TOOL_NAME and tool.name not in CHECK_TOOL_NAMES}
        for server, tools in listed:
            for tool in tools:
                sources.setdefault(tool.name, []).append(server.name)
        clashes = {name: names for name, names in sources.items() if len(names) > 1}
        if clashes:
            raise ToolNameClash(clashes)
        proxies = [(server, tool, self._proxy_tool(server.mcp_url, tool)) for server, tools in listed for tool in tools]
        for server, tool, proxy in proxies:
            self._mcp._tool_manager._tools[tool.name] = proxy
            self._tool_server_urls[tool.name] = server.mcp_url
            logger.debug(f"Registered proxy tool: {tool.name}")
        for server, tools in listed:
            self._server_tools[server.name] = tools
        self._hidden_by_default = {
            tool.name for _, tools in listed for tool in tools
            if (tool.meta or {}).get(AGENT_ENV_HIDDEN_BY_DEFAULT_META_KEY) is True}
        self._tools_discovered = True
        if self._mirror_get_time_descriptor():
            logger.info(f"Re-mirrored '{GET_TIME_TOOL_NAME}' descriptor after tool discovery")
        if self._mirror_check_descriptors():
            logger.info("Re-mirrored the scheduled-check tool descriptors after tool discovery")
        # list_website_urls sits in _server_tools from construction, so a memo may predate the backing tools
        self._trigger_engine.invalidate_readonly_cache()

    @staticmethod
    def _json_response(data: dict | list, status_code: int) -> Response:
        return Response(content=json.dumps(data), status_code=status_code, media_type="application/json")

    @staticmethod
    def _format_role_rules(rules: dict[str, bool]) -> dict:
        return {
            "disabled": sorted(k for k, v in rules.items() if v),
            "allowed": sorted(k for k, v in rules.items() if not v),
        }

    def _apply_rule(self, role: str, tool: str, value: bool) -> None:
        if role == WILDCARD and tool == WILDCARD:
            self._role_rules.clear()
            self._role_rules[WILDCARD] = {WILDCARD: value}
        elif role == WILDCARD:
            for r in self._role_rules:
                self._role_rules[r][tool] = value
            self._role_rules.setdefault(WILDCARD, {})[tool] = value
        elif tool == WILDCARD:
            self._role_rules[role] = {WILDCARD: value}
        else:
            self._role_rules.setdefault(role, {})[tool] = value

    async def _handle_role_mutation(self, request: Request, action: str) -> Response:
        try:
            body = await request.json()
        except Exception:
            return self._json_response({"error": "Invalid JSON body"}, 400)
        role = body.get("role")
        if not isinstance(role, str) or not role:
            return self._json_response({"error": "Missing or invalid 'role' field"}, 400)
        tools = body.get("tools")
        if tools == WILDCARD:
            target_keys = [WILDCARD]
        elif isinstance(tools, list) and all(isinstance(t, str) for t in tools):
            target_keys = tools
        else:
            return self._json_response(
                {"error": "'tools' must be a list of strings or the string '*'"},
                400,
            )
        value = action == TOOL_DISABLE_ACTION
        async with self._role_rules_lock:
            for tool in target_keys:
                self._apply_rule(role, tool, value)
            snapshot = self._format_role_rules(self._role_rules.get(role, {}))
        return self._json_response({"role": role, **snapshot}, 200)

    async def _handle_step(self, request: Request) -> Response:
        try:
            await self._ensure_tools_discovered()
        except ToolNameClash as e:
            return self._json_response({"error": f"{e}, so the env serves none of its servers' tools."}, 503)
        except Exception:
            return self._json_response(
                {"error": "Tool discovery has not completed. Internal MCP servers may be unavailable."},
                503,
            )
        try:
            body = await request.json()
        except Exception:
            return self._json_response({"error": "Invalid JSON body"}, 400)
        action = body.get("action")
        role = self._request_role(request)
        source = _step_source(body)
        if source is None:
            return self._json_response({"error": f"'source' may only be '{STEP_SOURCE_POST_TURN_HOOK}', with a "
                                                 "non-empty 'hook_artifact_id' naming the hook"}, 400)
        if action == "list_tools":
            return self._step_list_tools(role)
        elif action == "call_tool":
            return await self._step_call_tool(body, role, source)
        else:
            return self._json_response({"error": f"Unknown action: {action}. Expected 'list_tools' or 'call_tool'."}, 400)

    def _step_list_tools(self, role: str) -> Response:
        """Return a flat list of all registered tools, filtered by role."""
        tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            }
            for server_tools in self._server_tools.values()
            for tool in server_tools
            if not self._is_disabled(role, tool.name)
        ]
        return self._json_response({"tools": tools}, 200)

    async def _step_call_tool(self, body: dict, role: str, source: dict = _STEP_API) -> Response:
        tool_name = body.get("tool_name")
        if not tool_name or not isinstance(tool_name, str):
            return self._json_response({"error": "Missing or invalid 'tool_name' field"}, 400)
        arguments = body.get("arguments", {})
        if not isinstance(arguments, dict):
            return self._json_response({"error": "'arguments' must be a JSON object"}, 400)

        if self._is_disabled(role, tool_name):
            return self._json_response(
                {"error": f"Tool '{tool_name}' is disabled for role '{role}'"},
                403,
            )

        server_url = self._tool_server_urls.get(tool_name)
        if server_url is None:
            return await self._step_call_gateway_tool(tool_name, arguments, role, source)
        tool_call_event_id = await self._log_event({
            "event_type": "tool_call",
            **source,
            "tool_call": {"function_name": tool_name, "arguments": arguments},
            **self._attribution(role),
        })
        try:
            async with self._tool_call_guard():
                changelog_before = self._query_changelog_id()
                try:
                    result = await self._call_child_tool(server_url, tool_name, arguments, caller=Caller(role=role))
                except Exception as e:
                    await self._log_failed_call(tool_call_event_id, changelog_before, e, **source)
                    raise
                changelog_id = self._query_changelog_id()

            await self._log_event({
                "event_type": "tool_call_result",
                **source,
                "tool_call_event_id": tool_call_event_id,
                "tool_call_result": result.model_dump(mode="json"),
                "changelog_id_before": changelog_before,
                "changelog_id": changelog_id,
            })
            barrier_tasks, barrier_timeout_s = self._fire_triggers(role, tool_name, arguments, result)
            await self._await_barriers(barrier_tasks, barrier_timeout_s)
            response_data = result.model_dump(mode="json", exclude_none=True)
            response_data["changelog_id"] = changelog_id
            return self._json_response(response_data, 200)
        except asyncio.TimeoutError:
            return self._json_response({"error": f"Tool '{tool_name}' timed out after {self.TOOL_CALL_TIMEOUT_S}s"}, 504)
        except Exception as e:
            logger.exception(f"/step call_tool '{tool_name}' failed")
            error_detail = str(e) or repr(e)
            return self._json_response({"error": f"Tool execution failed ({type(e).__name__}): {error_detail}"}, 500)

    async def _step_call_gateway_tool(self, tool_name: str, arguments: dict, role: str = DEFAULT_ROLE,
                                      source: dict = _STEP_API) -> Response:
        """Call a FastMCP-registered tool (e.g. list_website_urls) for /step."""
        tool = self._mcp._tool_manager._tools.get(tool_name)
        if tool is None:
            return self._json_response({"error": f"Unknown tool: {tool_name}"}, 404)
        tool_call_event_id = await self._log_event({
            "event_type": "tool_call",
            **source,
            "tool_call": {"function_name": tool_name, "arguments": arguments},
            **self._attribution(role),
        })
        try:
            async with self._tool_call_guard():
                # bound as the MCP path binds it, so a tool that acts for its caller (schedule_check) reads the role
                token = Caller(role=role, session=Caller.current().session).bind()
                try:
                    result = await tool.run(arguments)
                finally:
                    Caller.reset(token)
                changelog_id = self._query_changelog_id()
            if isinstance(result, CallToolResult):
                response_data = result.model_dump(mode="json", exclude_none=True)
            else:
                text = json.dumps(result) if not isinstance(result, str) else result
                result = CallToolResult(content=[TextContent(type="text", text=text)], isError=False)
                response_data = result.model_dump(mode="json", exclude_none=True)
            await self._log_event({
                "event_type": "tool_call_result",
                **source,
                "tool_call_event_id": tool_call_event_id,
                "tool_call_result": response_data,
                "changelog_id": changelog_id,
            })
            barrier_tasks, barrier_timeout_s = self._fire_triggers(role, tool_name, arguments, result)
            await self._await_barriers(barrier_tasks, barrier_timeout_s)
            response_data["changelog_id"] = changelog_id
            return self._json_response(response_data, 200)
        except Exception as e:
            logger.exception(f"/step call_tool '{tool_name}' failed")
            error_detail = str(e) or repr(e)
            return self._json_response({"error": f"Tool execution failed ({type(e).__name__}): {error_detail}"}, 500)

    def _register_get_time_tool(self) -> None:
        """Expose the agent-facing clock read (idempotent). Called on arm only."""
        if GET_TIME_TOOL_NAME in self._tool_server_urls:
            raise ValueError(f"'{GET_TIME_TOOL_NAME}' is already registered by a backing server")
        tools = self._mcp._tool_manager._tools
        if GET_TIME_TOOL_NAME in tools:
            return
        tool = build_get_time_tool(self._clock, self._create_arg_model_from_schema)
        tools[GET_TIME_TOOL_NAME] = tool
        if self._mirror_get_time_descriptor():
            self._trigger_engine.invalidate_readonly_cache()

    def _mirror_get_time_descriptor(self) -> bool:
        """Mirror the registered tool into `_server_tools` — what `/step list_tools`, `/state` and
        `TriggerEngine._is_readonly` read. Derived from the tool so the two cannot drift. Returns
        whether it appended."""
        if not self._tools_discovered or GET_TIME_TOOL_NAME in self._tool_server_urls:
            return False
        tool = self._mcp._tool_manager._tools.get(GET_TIME_TOOL_NAME)
        if tool is None:
            return False
        mirrored = self._server_tools.setdefault("gateway", [])
        if any(t.name == GET_TIME_TOOL_NAME for t in mirrored):
            return False
        mirrored.append(MCPTool(name=tool.name, description=tool.description,
                                inputSchema=tool.parameters, annotations=tool.annotations))
        return True

    def _unregister_get_time_tool(self) -> None:
        """Remove it on clear (idempotent). A backing server's tool of the same name is left alone."""
        tools = self._mcp._tool_manager._tools
        if GET_TIME_TOOL_NAME in self._tool_server_urls or GET_TIME_TOOL_NAME not in tools:
            return
        del tools[GET_TIME_TOOL_NAME]
        remaining = [t for t in self._server_tools.get("gateway", []) if t.name != GET_TIME_TOOL_NAME]
        if remaining:
            self._server_tools["gateway"] = remaining
        else:
            self._server_tools.pop("gateway", None)
        self._trigger_engine.invalidate_readonly_cache()

    def _register_check_tools(self) -> None:
        """Expose schedule_check and its siblings when a config names roles, rebuilt on each config so their
        descriptions carry its max_active and min_interval."""
        self._unregister_check_tools()
        tools = self._mcp._tool_manager._tools
        for tool in build_check_tools(self._notifications, self._create_arg_model_from_schema):
            if tool.name not in self._tool_server_urls:
                tools[tool.name] = tool
        if self._mirror_check_descriptors():
            self._trigger_engine.invalidate_readonly_cache()

    def _mirror_check_descriptors(self) -> bool:
        """Mirror the registered scheduled-check tools into `_server_tools`, as `_mirror_get_time_descriptor` does for
        get_time. Returns whether it appended."""
        if not self._tools_discovered or not self._notifications.enabled:
            return False
        mirrored = self._server_tools.setdefault("gateway", [])
        present = {t.name for t in mirrored}
        added = False
        for name in CHECK_TOOL_NAMES:
            tool = self._mcp._tool_manager._tools.get(name)
            if tool is None or name in present or name in self._tool_server_urls:
                continue
            mirrored.append(MCPTool(name=tool.name, description=tool.description,
                                    inputSchema=tool.parameters, annotations=tool.annotations))
            added = True
        return added

    def _unregister_check_tools(self) -> None:
        """Remove them when a task turns scheduled checks off or clears (idempotent); a backing server's are left."""
        tools = self._mcp._tool_manager._tools
        for name in CHECK_TOOL_NAMES:
            if name not in self._tool_server_urls:
                tools.pop(name, None)
        remaining = [t for t in self._server_tools.get("gateway", []) if t.name not in CHECK_TOOL_NAMES
                     or t.name in self._tool_server_urls]
        if remaining:
            self._server_tools["gateway"] = remaining
        else:
            self._server_tools.pop("gateway", None)
        self._trigger_engine.invalidate_readonly_cache()

    def _create_and_register_proxy_tool(self, server_url: str, tool: MCPTool) -> None:
        self._mcp._tool_manager._tools[tool.name] = self._proxy_tool(server_url, tool)

    def _proxy_tool(self, server_url: str, tool: MCPTool) -> FastMCPTool:
        """A proxy tool that forwards calls to an internal server via its persistent session."""
        tool_name = tool.name
        tool_description = tool.description or ""
        input_schema = tool.inputSchema or {"type": "object", "properties": {}}

        async def proxy(**kwargs) -> CallToolResult:
            # The session's own caller, never anything the inbound request wrote: the same value the disabled check
            # and trigger attribution already keyed this call on.
            caller = Caller.current()
            tool_call_event_id = await self._log_event({
                "event_type": "tool_call",
                "tool_call": {
                    "function_name": tool_name,
                    "arguments": kwargs,
                },
                **self._attribution(caller.role),
            })

            async with self._tool_call_guard():
                changelog_before = self._query_changelog_id()
                try:
                    result = await self._call_child_tool(server_url, tool_name, kwargs, caller=caller)
                except Exception as e:
                    await self._log_failed_call(tool_call_event_id, changelog_before, e)
                    raise
                changelog_id = self._query_changelog_id()

            await self._log_event({
                "event_type": "tool_call_result",
                "tool_call_event_id": tool_call_event_id,
                "tool_call_result": result.model_dump(mode="json"),
                "changelog_id_before": changelog_before,
                "changelog_id": changelog_id,
            })
            return result

        arg_model = self._create_arg_model_from_schema(tool_name, input_schema)
        return FastMCPTool(
            fn=proxy,
            name=tool_name,
            description=tool_description,
            parameters=input_schema,
            fn_metadata=FuncMetadata(arg_model=arg_model),
            is_async=True,
        )

    @staticmethod
    def _create_arg_model_from_schema(tool_name: str, schema: dict[str, Any]):
        fields = {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        for name in properties:
            if name in required:
                fields[name] = (Any, ...)
            else:
                fields[name] = (Any, None)
        if not fields:
            return create_model(f"{tool_name}Args", __base__=_ProxyArgModelBase)
        return create_model(f"{tool_name}Args", __base__=_ProxyArgModelBase, **fields)

    def _close_trajectory_file(self) -> None:
        if self._trajectory_file and not self._trajectory_file.closed:
            self._trajectory_file.close()

    async def _log_native_tool_call(self, name: str, arguments: dict | None, role: str) -> str | None:
        """Trajectory record for a gateway-owned tool on the MCP path (fail-open)."""
        try:
            return await self._log_event({
                "event_type": "tool_call",
                "tool_call": {"function_name": name, "arguments": arguments or {}},
                **self._attribution(role),
            })
        except Exception:
            logger.exception(f"Failed to log tool_call for gateway tool '{name}'")
            return None

    async def _log_native_tool_result(self, event_id: str | None, result: Any) -> None:
        if event_id is None:
            return
        try:
            await self._log_event({
                "event_type": "tool_call_result",
                "tool_call_event_id": event_id,
                "tool_call_result": result,
            })
        except Exception:
            logger.exception("Failed to log tool_call_result for a gateway tool")

    async def _log_failed_call(self, tool_call_event_id: str, changelog_before: int | None, error: Exception,
                               **source: str) -> None:
        """Close a failed call's write window, which may hold a write it committed first (fail-open)."""
        try:
            await self._log_event({"event_type": "tool_call_failed", **source, "tool_call_event_id": tool_call_event_id,
                                   "error": f"{type(error).__name__}: {str(error)[:300]}",
                                   "changelog_id_before": changelog_before, "changelog_id": self._query_changelog_id()})
        except Exception:
            logger.exception("Failed to log a failed tool call")

    def _attribution(self, role: str) -> dict:
        """Who made a tool call, for its trajectory event: the caller's role, and the trigger fires it was made for
        when that role is an executor realizing them."""
        causes = self._trigger_engine.caused_by(role)
        return {"role": role, **({"caused_by": causes} if causes else {})}

    async def _log_event(self, event: dict) -> str:
        async with self._event_lock:
            self._event_counter += 1
            event_id = f"event_{self._event_counter}"
            event["event_id"] = event_id
            event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            # Wall time says nothing under a fast clock; stamp the gateway clock too — omitted when unarmed (TriggerEngine._emit's idiom).
            virtual = self._clock.now()
            if virtual is not None:
                event["virtual_time"] = _iso(virtual)
            self._trajectory_file.write(json.dumps(
                event, default=lambda o: o.model_dump(mode="json") if hasattr(o, "model_dump") else str(o),
            ) + "\n")
            self._trajectory_file.flush()
            self._changes.observe(event)
        return event_id

    async def _serve_changes(self, request: Request) -> Response:
        """The changelog rows after ``after_id`` (at most ``limit``, oldest first), each with its actors (see ``changes``)."""
        try:
            after_id = int(request.query_params.get("after_id", 0))
            limit = int(request.query_params.get("limit", CHANGES_PAGE_DEFAULT))
        except ValueError:
            return self._json_response({"ok": False, "error": "after_id and limit must be integers"}, 400)
        if after_id < 0 or not 1 <= limit <= CHANGES_PAGE_MAX:
            return self._json_response(
                {"ok": False, "error": f"after_id must be >= 0 and limit between 1 and {CHANGES_PAGE_MAX}"}, 400)
        if self._db_conn is None:
            return self._json_response({"ok": False, "error": "this env keeps no changelog"}, 404)
        try:
            rows = await asyncio.to_thread(self._read_changelog, after_id, limit + 1)
        except Exception as e:
            logger.exception("Reading the changelog failed")
            return self._json_response({"ok": False, "error": f"changelog unavailable: {e}"}, 503)
        rows, has_more = rows[:limit], len(rows) > limit
        actors = self._changes.actors([row["id"] for row in rows])
        return self._json_response(
            {"changes": [{**row, "actors": who} for row, who in zip(rows, actors)], "has_more": has_more}, 200)

    def _read_changelog(self, after_id: int, count: int) -> list[dict]:
        with self._db_conn.cursor() as cur:
            cur.execute('SELECT id, "timestamp", schema_name, table_name, operation, row_id, summary, changed_fields '
                        "FROM public._changelog WHERE id > %s ORDER BY id LIMIT %s", (after_id, count))
            names = [column[0] for column in cur.description]
            rows = [dict(zip(names, values)) for values in cur.fetchall()]
        for row in rows:
            row["timestamp"] = row["timestamp"].isoformat()
        return rows

    def _serve_trajectory(self) -> Response:
        """Stream the whole env trajectory as JSONL; only a mid-write tail line can arrive torn."""
        try:
            total = os.path.getsize(GATEWAY_TRAJECTORY_FILE)
        except OSError as e:
            return self._json_response({"ok": False, "error": f"trajectory file unavailable: {e}"}, 404)

        def _iter():
            with open(GATEWAY_TRAJECTORY_FILE, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk

        return StreamingResponse(_iter(), media_type="application/jsonl", headers={"X-Trajectory-Total-Bytes": str(total)})


class _ProxyArgModelBase(ArgModelBase):
    """ArgModelBase subclass that excludes unset fields from model_dump_one_level().

    This prevents the gateway proxy from forwarding null values for optional
    parameters that the caller didn't provide. Pydantic's model_fields_set
    tracks which fields were actually present in the input, so explicitly
    provided null values are still forwarded correctly.
    """

    def model_dump_one_level(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for field_name, field_info in self.__class__.model_fields.items():
            if field_name not in self.model_fields_set:
                continue
            value = getattr(self, field_name)
            output_name = field_info.alias if field_info.alias else field_name
            kwargs[output_name] = value
        return kwargs


def _step_source(body: dict) -> dict | None:
    """The trajectory tag of a /step call: step_api, or the post-turn hook its body names; None for any other source."""
    if "source" not in body:
        return _STEP_API
    hook = body.get("hook_artifact_id")
    if body["source"] != STEP_SOURCE_POST_TURN_HOOK or not isinstance(hook, str) or not hook:
        return None
    return {"source": STEP_SOURCE_POST_TURN_HOOK, "hook_artifact_id": hook}


def _request_args(request: Request, body: bytes) -> dict:
    """A request's query parameters and form or JSON-object body fields (the body wins a clash; repeats become lists)."""
    args = _fields(parse_qsl(request.url.query, keep_blank_values=True))
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type == "application/x-www-form-urlencoded":
        args.update(_fields(parse_qsl(body.decode("utf-8", "replace"), keep_blank_values=True)))
    elif media_type == "application/json" or media_type.endswith("+json"):
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            args.update(parsed)
    return args


def _fields(pairs: list[tuple[str, str]]) -> dict:
    grouped: dict[str, list[str]] = {}
    for key, value in pairs:
        grouped.setdefault(key, []).append(value)
    return {key: values[0] if len(values) == 1 else values for key, values in grouped.items()}


def _normalized(path: str) -> str:
    return posixpath.normpath("/" + path.lstrip("/"))


def reserved_site_path(path: str, reserved: frozenset[str] = frozenset()) -> bool:
    """Whether ``path``, as a site's server would route it, is its data plane or its env card, or one of the paths
    ``reserved`` names (``site_reserved_paths``), or under one."""
    normalized = _normalized(path)
    return normalized == WELL_KNOWN_PATH or any(normalized == base or normalized.startswith(base.rstrip("/") + "/")
                                                for base in (RPC_PATH, *reserved))


def route_environment_name(key: str, base_url: str, servers: list[InternalMCPServer], *, website: bool) -> str:
    """The name the env behind the backing server ``key``, at ``base_url``, is registered as. A website's key is its
    name. Any other key is an MCP server's route, ``mcp-<name>``, or a child env's own name, a computer's, which may
    start ``mcp-`` too; the internal MCP server whose endpoint is at the route's backend says which: one under the
    route's base address first, else one on its host, as a server whose card is on another port than its MCP endpoint
    is. With no such server, an ``mcp-`` key is an MCP server's route."""
    stripped = key.removeprefix("mcp-")
    if website or stripped == key:
        return key
    base = base_url.rstrip("/")
    host = urlsplit(base).hostname

    def at_backend(name: str) -> int:
        urls = [server.mcp_url for server in servers if server.name == name]
        if any(url.startswith(f"{base}/") for url in urls) and base:
            return 2
        return 1 if host and any(urlsplit(url).hostname == host for url in urls) else 0

    route, own = at_backend(stripped), at_backend(key)
    return key if own > route else stripped


def site_mcp_endpoints(servers: list[InternalMCPServer], website_urls: dict[str, str],
                       rest_proxy_urls: dict[str, str]) -> dict[str, str]:
    """The MCP endpoint, as a path under its /svc/<name>/ base, of each site that is an internal MCP server too, by
    name: where a call through /svc/<name>/ gets the gateway's role in its JSON-RPC ``_meta``, as an MCP server's does."""
    endpoints = {}
    for server in servers:
        base = rest_proxy_urls.get(server.name, "")
        if server.name not in website_urls or not base:
            continue
        resolved = card_url(base, server.mcp_url)
        if resolved is not None and resolved.own_server and resolved.route_path is not None:
            endpoints[server.name] = resolved.route_path
    return endpoints


class CardURL(NamedTuple):
    """A URL a site's card names, as the site at an address serves it: its path and query; the path a request on the
    site's route reaches it at, its path under the address's (None when it isn't under it); and whether it is on the
    site's own server."""
    path: str
    query: str
    route_path: str | None
    own_server: bool


def card_url(base: str, url: str) -> CardURL | None:
    """``url``, which a site's card names, against the site's address ``base`` (a URL, or only its path): a path,
    absolute or relative, joined under the address's path, since a card's paths are relative to where it was read, and
    on the site's own server; a full URL as it is, on the site's own server when its scheme, host and port are the
    address's. The one reading of a card's URLs that discovery, /svc role stamping and a route's reserved paths share.
    None for one with no path, or that can't be read."""
    try:
        own, given = urlsplit(base.rstrip("/")), urlsplit(url)
        full = bool(given.scheme or given.netloc)
        same = not full or _address(given) == _address(own)
    except ValueError:
        return None
    under = own.path.rstrip("/")
    if full:
        path = given.path
    elif given.path.startswith("/"):
        path = under + given.path
    else:
        path = urljoin(under + "/", given.path) if given.path else ""
    if not path.startswith("/"):
        return None
    normalized = _normalized(path)
    route = (normalized[len(under):] or "/") if normalized == under or normalized.startswith(under + "/") else None
    return CardURL(path, given.query, route, same)


def _address(url) -> tuple:
    """A split URL's scheme, host and port, its scheme's default port when it names none."""
    return url.scheme, url.hostname, url.port or {"http": 80, "https": 443}.get(url.scheme)


def site_reserved_paths(card: dict, base: str = "") -> frozenset[str]:
    """The paths a site's card advertises for agentenv, which its route never reaches: its ``url``, every interface's,
    such as the MCP endpoint of a site that is an MCP server too, and every extension's endpoint and each of its
    methods'; each one a path of its own, not the site's root, read from a full URL or one with a query as well as from
    a bare path, as ``card_url`` reads them against the site's address ``base``. A full URL's path is reserved
    whatever its host, since a site is reached under names it never sees in advance: as its route reaches it, under the
    address's path, or as it is when it isn't under it."""
    urls = [card.get("url")]
    urls += [i.get("url") for i in card.get("additionalInterfaces") or [] if isinstance(i, dict)]
    for ext in (card.get("capabilities") or {}).get("extensions") or []:
        params = ext.get("params") if isinstance(ext, dict) else None
        if not isinstance(params, dict):
            continue
        urls += [endpoint for *_, endpoint in operations(params)]
        urls.append(params.get("endpoint"))
    resolved = (card_url(base, url) for url in urls if isinstance(url, str))
    paths = (_normalized(r.path if r.route_path is None else r.route_path) for r in resolved if r is not None)
    return frozenset(p for p in paths if p != "/")


def site_request_path(request: HTTPConnection) -> str | None:
    """The path a site's server receives for a request on its route: the raw path forwarded, its dot segments removed
    by httpx and then decoded. It differs from the path as routed here when an encoded slash sits beside a dot segment
    (``public%2Fx/../api`` reaches ``/api``). None for a request-target the proxy can't build a URL for: ``forward()``
    refuses it."""
    try:
        return httpx.URL(upstream_url("http://site", request)).path
    except (ValueError, httpx.InvalidURL):
        return None


def reserved_site_request(request: HTTPConnection, reserved: frozenset[str] = frozenset()) -> bool:
    """Whether a request or WebSocket on a site's route reaches its data plane or env card, or one of the paths
    ``reserved`` names (``site_reserved_paths``): by the path as routed here, or by the one its server receives
    (``site_request_path``). A request-target the proxy can't build a URL for isn't reserved here: ``forward()``
    refuses it."""
    if reserved_site_path("/" + request.path_params.get("path", ""), reserved):
        return True
    sent = site_request_path(request)
    return sent is not None and reserved_site_path(sent, reserved)
