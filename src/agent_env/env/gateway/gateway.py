"""Gateway that aggregates MCP tools and REST proxies for website backends."""
from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import logging
import os
import socket
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import anyio
import httpx
import psycopg2
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, ArgModelBase
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, CallToolResult, ErrorData, TextContent, Tool as MCPTool
from pydantic import create_model
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from anyio import ClosedResourceError

if TYPE_CHECKING:
    from . import InternalMCPServer

# Relative import — gateway runs as a standalone container with no agent_env.*
from .constants import (
    AGENT_ENV_ROLE_HEADER,
    AGENT_ENV_ROLE_META_KEY,  # noqa: F401  (the server module serves the whole vocabulary, used in this file or not)
    AGENT_ENV_SESSION_META_KEY,  # noqa: F401
    CARD_FETCH_TIMEOUT_S,
    CHILD_DISCOVERY_TIMEOUT_S,
    DATA_PLANE_LOAD_TIMEOUT_MAX_S,
    DEFAULT_ROLE,
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
    TOOL_DISABLE_ACTION,
    TOOL_ENABLE_ACTION,
    WELL_KNOWN_PATH,
    WILDCARD,
    GatewayMode,
    tool_name_clashes_text,
)
from .caller import Caller
from .child_transport import child_http_client
from .clock import Clock, ClockError, _iso
from .get_time import GET_TIME_TOOL_NAME, build_get_time_tool
from .rest_proxy import ROLE_HEADER_BYTES, fetch_child_card, forward, rewrite_child_card
from .triggers import TriggerEngine, TriggerError

logger = logging.getLogger(__name__)


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
    CHILD_CLOSE_TIMEOUT_S = 3.0  # wait for the owner task to exit a retired child-session generation; past it, cancel it
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
    ):
        self.host = host
        self.port = port
        self.server_name = server_name
        self.internal_mcp_servers = internal_mcp_servers
        self.website_urls = website_urls or {}
        self.rest_proxy_urls = rest_proxy_urls or {}
        self.gateway_mode = gateway_mode
        self._service_db_url = service_db_url
        self._db_conn: psycopg2.extensions.connection | None = None
        if service_db_url:
            self._db_conn = psycopg2.connect(service_db_url)
            self._db_conn.autocommit = True
        # The process's one generation of child sessions, shared by the MCP proxy, /step and triggers. The client
        # contexts are anyio task groups, so only the task that entered them may exit them: the owner task.
        self._child_sessions: dict[str, ClientSession] = {}
        self._child_owner: asyncio.Task | None = None
        self._child_close: asyncio.Event | None = None
        self._child_lock = asyncio.Lock()  # open / retire / close of a generation (taken after _discover_lock)
        self._closing = False  # set at process shutdown; no generation is opened past it
        self._retirements: set[asyncio.Task] = set()  # generations being retired on a reported transport failure
        self._retiring: set[int] = set()  # ids of generations a report has already retired, so the rest are no-ops
        self._discover_lock = asyncio.Lock()
        self._server_tools: dict[str, list[MCPTool]] = {}
        self._tools_discovered = False
        self._tool_name_clashes: dict[str, list[str]] = {}  # found once, by a discovery that then registered nothing
        self._tool_server_urls: dict[str, str] = {}
        self._tool_call_lock: asyncio.Lock | None = asyncio.Lock() if self.gateway_mode == GatewayMode.CONSISTENT else None
        self._role_rules: dict[str, dict[str, bool]] = {}
        self._role_rules_lock = asyncio.Lock()

        self._event_counter = 0
        self._event_lock = asyncio.Lock()
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
        self._env_get_time_url_cache: str | None = None
        self._install_role_filters()

        if self.rest_proxy_urls:
            @self._mcp.custom_route(
                "/svc/{service_name}/{path:path}",
                methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            )
            async def proxy_to_service(request: Request) -> Response:
                return await self._proxy_rest_request(request)

        if self.website_urls:
            @self._mcp.custom_route(
                "/website/{service_name}/{path:path}",
                methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            )
            async def proxy_to_frontend(request: Request) -> Response:
                return await self._proxy_rest_request(request, url_map=self.website_urls)

            @self._mcp.custom_route(
                "/website/{service_name}",
                methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            )
            async def proxy_to_frontend_root(request: Request) -> Response:
                return await self._proxy_rest_request(request, url_map=self.website_urls)

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

        async def _trigger_write(request: Request, method) -> Response:
            """Shared body for the trigger write routes: parse JSON, invoke the engine method,
            map a TriggerError (bad config) to a 400."""
            try:
                body = await request.json()
            except Exception:
                return self._json_response({"ok": False, "error": "Invalid JSON body"}, 400)
            try:
                return self._json_response(method(body), 200)
            except TriggerError as e:
                return self._json_response({"ok": False, "error": str(e)}, 400)

        @self._mcp.custom_route("/triggers/register", methods=["POST"])
        async def triggers_register(request: Request) -> Response:
            return await _trigger_write(request, self._trigger_engine.register)

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

        @self._mcp.custom_route("/trajectory", methods=["GET"])
        async def env_trajectory(request: Request) -> Response:
            return self._serve_trajectory()

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
                self._closing = True
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
        """Walk the rule chain (role first, then WILDCARD) returning the first concrete True/False; default False."""
        for r in (role, WILDCARD):
            rules = self._role_rules.get(r, {})
            if name in rules:
                return rules[name]
            if WILDCARD in rules:
                return rules[WILDCARD]
        return False

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
        return await forward(request, base_url, role=self._request_role(request), timeout=self.REST_PROXY_TIMEOUT,
                             mcp_server=url_map is self.rest_proxy_urls and service_name.startswith("mcp-"))

    @staticmethod
    def _request_role(request: Request) -> str:
        """The role the gateway attributes a REST request to: the first AgentEnv-Role header, else the default role."""
        return request.headers.get(AGENT_ENV_ROLE_HEADER, "").strip() or DEFAULT_ROLE

    def _backing_keys(self) -> list[str]:
        # rest_proxy_urls entries are the env's backing servers: mcp-<svc> (MCP) + <svc> (website
        # backends). Non-v1 backers (e.g. the injected website_browser) self-omit from the card via 404.
        return list(self.rest_proxy_urls)

    def _rewrite_child_card(self, key: str, card: dict) -> dict:
        return rewrite_child_card(key, card, self._gateway_extensions())

    async def _fetch_child_card(self, client: httpx.AsyncClient, key: str, base_url: str) -> dict | None:
        """Fetch + rewrite one backing server's card; None (omit) on 404 / error / timeout / malformed JSON."""
        child = await fetch_child_card(client, key, base_url)
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

    async def _open_persistent_sessions(self, exit_stack: contextlib.AsyncExitStack, max_retries: int = 10, retry_delay: float = 1.0) -> dict[str, ClientSession]:
        """Open a persistent MCP client session to each internal server, over a transport that reports a failed
        request to this generation so it is retired at once (see child_transport).

        Each attempt opens its contexts on a stack of its own. mcp's client runs its requests in an anyio task group
        whose cancel scope belongs to the task that entered it, this one: a request that fails outright (the child
        refusing the connection) cancels whatever this task awaits, and keeps cancelling it until that group is
        exited. So a failed attempt is exited before anything else is awaited, and reads as a failure to retry
        rather than as this task being cancelled; only a cancellation that is still pending once the attempt's
        group is gone came from outside."""
        sessions: dict[str, ClientSession] = {}

        def report(exc: httpx.TransportError) -> None:
            self._child_transport_failed(sessions, exc)

        for server in self.internal_mcp_servers:
            for attempt in range(max_retries):
                attempt_stack = contextlib.AsyncExitStack()
                try:
                    client = await attempt_stack.enter_async_context(child_http_client(report))
                    read_stream, write_stream, _ = await attempt_stack.enter_async_context(
                        streamable_http_client(server.mcp_url, http_client=client))
                    session = await attempt_stack.enter_async_context(ClientSession(read_stream, write_stream))
                    # a child that accepts the connection but never answers initialize is a failed attempt, not a hang
                    await asyncio.wait_for(session.initialize(), timeout=CARD_FETCH_TIMEOUT_S)
                except BaseException as e:
                    failure: BaseException = e
                    try:
                        await attempt_stack.aclose()
                    except BaseException as unwound:  # the client's task group names what failed
                        if isinstance(e, asyncio.CancelledError):
                            failure = unwound
                    if isinstance(e, asyncio.CancelledError) and asyncio.current_task().cancelling():
                        raise
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to connect to {server.name} (attempt {attempt + 1}/{max_retries}): {failure!r}")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to connect to {server.name} after {max_retries} attempts: {failure!r}")
                        if failure is e:
                            raise
                        raise RuntimeError(f"could not open a session to {server.name}: {failure!r}") from failure
                else:
                    exit_stack.push_async_callback(attempt_stack.aclose)
                    sessions[server.mcp_url] = session
                    logger.info(f"Opened persistent session to {server.name}")
                    break
        return sessions

    async def _ensure_child_sessions(self) -> dict[str, ClientSession]:
        """The process's one generation of child sessions, opened on first need. Liveness is the owner task's, not
        the dict's emptiness, so a gateway with no backing servers does not reopen on every call."""
        if self._child_owner is not None and not self._child_owner.done():
            return self._child_sessions
        async with self._child_lock:
            if self._child_owner is not None and not self._child_owner.done():
                return self._child_sessions
            if self._closing:
                raise RuntimeError("gateway is shutting down")
            await self._close_child_sessions()  # a dead or half-open generation
            loop = asyncio.get_running_loop()
            ready: asyncio.Future = loop.create_future()
            close = asyncio.Event()
            owner = loop.create_task(self._own_child_sessions(ready, close), name="gateway-child-sessions")
            owner.add_done_callback(self._log_owner_exit)
            try:
                sessions = await ready
            except BaseException:
                # Nobody will use this generation: an open still in progress is abandoned (its stack unwinds what
                # it entered), not finished in the background to hold connections nobody waits on.
                close.set()
                owner.cancel()
                raise
            self._child_sessions, self._child_owner, self._child_close = sessions, owner, close
            return sessions

    async def _own_child_sessions(self, ready: asyncio.Future, close: asyncio.Event) -> None:
        """Holds one generation open: opens the client contexts, reports through `ready`, parks until `close` is set
        (or it is cancelled) and exits them. Returns rather than raising on a failed open, which `ready` carries.
        `close` is set on the way out whatever the reason, so a call racing on it fails now, not at its timeout."""
        try:
            async with contextlib.AsyncExitStack() as stack:
                try:
                    sessions = await self._open_persistent_sessions(stack)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if not ready.done():
                        ready.set_exception(e)
                    return
                if ready.cancelled():
                    return  # nobody is waiting any more; the stack closes what was opened
                ready.set_result(sessions)
                await close.wait()
        except BaseExceptionGroup as e:
            # a request failing outright under a live generation cancels this task through mcp's client task group
            # (see _open_persistent_sessions); the stack has unwound, and the transport failure retires the generation
            logger.warning(f"child session generation ended: {e!r}")
        finally:
            close.set()

    @staticmethod
    def _log_owner_exit(owner: asyncio.Task) -> None:
        """Retrieve a generation's exception when it dies on its own (a transport crash), so it is logged now rather
        than as 'exception never retrieved' at garbage collection."""
        if not owner.cancelled() and owner.exception() is not None:
            logger.warning("child session generation ended", exc_info=owner.exception())

    async def _close_child_sessions(self) -> None:
        """Retire the current generation and wait (bounded) for its owner to exit it. Callers hold _child_lock,
        except the process lifespan, which runs after the last request."""
        owner, close = self._child_owner, self._child_close
        self._child_sessions, self._child_owner, self._child_close = {}, None, None
        if close is not None:
            close.set()
        if owner is not None and not owner.done():
            try:
                await asyncio.wait_for(owner, timeout=self.CHILD_CLOSE_TIMEOUT_S)
            except Exception:
                logger.warning("child sessions did not close cleanly", exc_info=True)

    async def _reset_child_sessions(self, stale: dict[str, ClientSession]) -> None:
        """Retire the generation a caller found dead; a concurrent caller that already did is a no-op."""
        async with self._child_lock:
            if self._child_sessions is stale:
                await self._close_child_sessions()

    def _child_transport_failed(self, sessions: dict[str, ClientSession], exc: httpx.TransportError) -> None:
        """A request to a child of ``sessions`` failed at the transport. mcp's client would keep the request
        pending, so the generation is retired now: every call it has in flight fails with CONNECTION_CLOSED instead
        of waiting out its timeout, and the next call reopens. Called from inside the client's own task, so the
        retirement is scheduled rather than awaited; a report for a generation already retired is nothing."""
        if self._closing or sessions is not self._child_sessions or id(sessions) in self._retiring:
            return  # every request in flight on a dying generation reports; the first one retires it
        self._retiring.add(id(sessions))
        logger.warning(f"child transport failed ({exc!r}); retiring the child session generation")
        task = asyncio.get_running_loop().create_task(self._reset_child_sessions(sessions), name="gateway-child-retire")
        self._retirements.add(task)
        task.add_done_callback(self._retirements.discard)
        task.add_done_callback(lambda _: self._retiring.discard(id(sessions)))

    async def _call_child_tool(self, server_url: str, tool_name: str, arguments: dict, *,
                               caller: Caller | None = None) -> CallToolResult:
        """Every child tools/call (MCP proxy, /step, trigger engine): one shared generation, the caller's `_meta`
        stamp (none for an unattributed trigger action), and one reconnect when the session turns out dead,
        retrying the call only if it never reached the child (a call the transport lost in flight may already have
        written; the next call reopens). Discovery is not redone on a reconnect; the registered proxies look their
        session up by URL."""
        meta = caller.meta() if caller is not None else None
        sessions = await self._ensure_child_sessions()
        closed = self._child_close  # this generation's; no await since the lookup, so they agree
        try:
            return await asyncio.wait_for(self._call_until_retired(sessions[server_url], closed, tool_name, arguments, meta),
                                          timeout=self.TOOL_CALL_TIMEOUT_S)
        except Exception as e:
            if not _is_dead_session(e):
                raise
            await self._reset_child_sessions(sessions)
            if not _call_never_reached_child(e):
                logger.warning(f"child session to {server_url} died with {tool_name} in flight ({e!r}); not retried")
                raise
            logger.warning(f"child session to {server_url} is dead ({e!r}), reconnecting...")
            sessions = await self._ensure_child_sessions()
            closed = self._child_close
            return await asyncio.wait_for(self._call_until_retired(sessions[server_url], closed, tool_name, arguments, meta),
                                          timeout=self.TOOL_CALL_TIMEOUT_S)

    @staticmethod
    async def _call_until_retired(session: ClientSession, closed: asyncio.Event | None, tool_name: str,
                                  arguments: dict, meta: dict | None) -> CallToolResult:
        """``session.call_tool``, unless its generation is retired first: then the call fails with CONNECTION_CLOSED
        the way a closed transport would fail it. mcp's client keeps a request whose transport failed pending, and
        exiting the session under it cancels the receive loop before that loop can fail the request, so the gateway
        fails it. The retirement event is this generation's; a generation already gone has none."""
        if closed is None:
            raise McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed"))
        call = asyncio.ensure_future(session.call_tool(tool_name, arguments, meta=meta))
        retired = asyncio.ensure_future(closed.wait())
        try:
            done, _ = await asyncio.wait({call, retired}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            retired.cancel()
            if not call.done():
                call.cancel()
            await asyncio.gather(call, retired, return_exceptions=True)  # the loser's cleanup ends before the caller moves on
        if call in done:
            return call.result()
        raise McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed"))

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
            except Exception as e:
                logger.exception("Tool discovery failed")
                for server in self.internal_mcp_servers:
                    self._server_tools.pop(server.name, None)  # the gateway's own entry stays
                self._tools_discovered = False
                if _is_dead_session(e):
                    await self._reset_child_sessions(sessions)
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
                raise
            logger.info(f"Found {len(tools_result.tools)} tool(s) from {server.name}")
            listed.append((server, tools_result.tools))
        # get_time is left out: a backing server's tool of that name wins, and the clock refuses to arm over it
        sources: dict[str, list[str]] = {
            tool.name: ["gateway"] for tool in self._server_tools.get("gateway", []) if tool.name != GET_TIME_TOOL_NAME}
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
        self._tools_discovered = True
        if self._mirror_get_time_descriptor():
            logger.info(f"Re-mirrored '{GET_TIME_TOOL_NAME}' descriptor after tool discovery")
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
        if action == "list_tools":
            return self._step_list_tools(role)
        elif action == "call_tool":
            return await self._step_call_tool(body, role)
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

    async def _step_call_tool(self, body: dict, role: str) -> Response:
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
            return await self._step_call_gateway_tool(tool_name, arguments, role)
        tool_call_event_id = await self._log_event({
            "event_type": "tool_call",
            "source": "step_api",
            "tool_call": {"function_name": tool_name, "arguments": arguments},
            **self._attribution(role),
        })
        try:
            async with self._tool_call_guard():
                result = await self._call_child_tool(server_url, tool_name, arguments, caller=Caller(role=role))
                changelog_id = self._query_changelog_id()

            await self._log_event({
                "event_type": "tool_call_result",
                "source": "step_api",
                "tool_call_event_id": tool_call_event_id,
                "tool_call_result": result.model_dump(mode="json"),
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

    async def _step_call_gateway_tool(self, tool_name: str, arguments: dict, role: str = DEFAULT_ROLE) -> Response:
        """Call a FastMCP-registered tool (e.g. list_website_urls) for /step."""
        tool = self._mcp._tool_manager._tools.get(tool_name)
        if tool is None:
            return self._json_response({"error": f"Unknown tool: {tool_name}"}, 404)
        tool_call_event_id = await self._log_event({
            "event_type": "tool_call",
            "source": "step_api",
            "tool_call": {"function_name": tool_name, "arguments": arguments},
            **self._attribution(role),
        })
        try:
            async with self._tool_call_guard():
                result = await tool.run(arguments)
                changelog_id = self._query_changelog_id()
            if isinstance(result, CallToolResult):
                response_data = result.model_dump(mode="json", exclude_none=True)
            else:
                text = json.dumps(result) if not isinstance(result, str) else result
                result = CallToolResult(content=[TextContent(type="text", text=text)], isError=False)
                response_data = result.model_dump(mode="json", exclude_none=True)
            await self._log_event({
                "event_type": "tool_call_result",
                "source": "step_api",
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
                result = await self._call_child_tool(server_url, tool_name, kwargs, caller=caller)
                changelog_id = self._query_changelog_id()

            await self._log_event({
                "event_type": "tool_call_result",
                "tool_call_event_id": tool_call_event_id,
                "tool_call_result": result.model_dump(mode="json"),
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
        return event_id

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


def _is_dead_session(exc: BaseException) -> bool:
    """Whether a child call failed because its session is gone, so the generation is retired: the stream closed under
    it, the child answered 404 for the session id (idle expiry or a restart, which the client reports as code 32600),
    or the transport died with the call in flight (CONNECTION_CLOSED)."""
    if isinstance(exc, (ClosedResourceError, anyio.BrokenResourceError)):
        return True
    return isinstance(exc, McpError) and exc.error.code in (32600, CONNECTION_CLOSED)


def _call_never_reached_child(exc: BaseException) -> bool:
    """Whether a dead-session failure may be retried: the stream was closed before the request was written, or the
    child rejected the session id at the transport. Not CONNECTION_CLOSED: the call was in flight when the transport
    died, and the child may already have performed its write."""
    if isinstance(exc, (ClosedResourceError, anyio.BrokenResourceError)):
        return True
    return isinstance(exc, McpError) and exc.error.code == 32600
