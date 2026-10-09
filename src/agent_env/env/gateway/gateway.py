"""Gateway that aggregates MCP tools and REST proxies for website backends."""
from __future__ import annotations

import asyncio
import atexit
import contextlib
import copy
import json
import logging
import os
import socket
import subprocess
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

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
from mcp.types import CallToolResult, TextContent, Tool as MCPTool
from pydantic import create_model
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send
from anyio import ClosedResourceError

if TYPE_CHECKING:
    from . import InternalMCPServer

# Relative import — gateway runs as a standalone container with no agent_env.*
from .constants import (
    AGENT_ENV_ROLE_HEADER,
    CARD_FETCH_TIMEOUT_S,
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
    TOOL_DISABLE_ACTION,
    TOOL_ENABLE_ACTION,
    WELL_KNOWN_PATH,
    WILDCARD,
    GatewayMode,
)
from .clock import Clock, ClockError, _iso
from .get_time import GET_TIME_TOOL_NAME, build_get_time_tool
from .triggers import TriggerEngine, TriggerError

logger = logging.getLogger(__name__)

_role_var: ContextVar[str] = ContextVar("agent_env_role", default=DEFAULT_ROLE)


class Gateway:
    """Gateway that aggregates MCP tools and REST proxies for website backends."""

    TOOL_CALL_TIMEOUT_S = 600.0
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
        self._mcp_sessions: dict[str, ClientSession] = {}
        self._server_tools: dict[str, list[MCPTool]] = {}
        self._tools_discovered = False
        self._tool_server_urls: dict[str, str] = {}
        self._step_sessions: dict[str, ClientSession] = {}
        self._step_exit_stack: contextlib.AsyncExitStack | None = None
        self._tool_call_lock: asyncio.Lock | None = asyncio.Lock() if self.gateway_mode == GatewayMode.CONSISTENT else None
        self._role_rules: dict[str, dict[str, bool]] = {}
        self._role_rules_lock = asyncio.Lock()

        self._init_lock = asyncio.Lock()
        self._event_counter = 0
        self._event_lock = asyncio.Lock()
        Path(GATEWAY_TRAJECTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
        self._trajectory_file = open(GATEWAY_TRAJECTORY_FILE, "a")
        atexit.register(self._close_trajectory_file)

        @asynccontextmanager
        async def lifespan(mcp: FastMCP):
            async with contextlib.AsyncExitStack() as stack:
                try:
                    self._mcp_sessions = await self._open_persistent_sessions(stack)
                    await self._discover_and_register_tools(self._mcp_sessions)
                except Exception:
                    logger.exception("Failed to discover tools during MCP lifespan startup")
                    self._mcp_sessions = {}
                    self._server_tools = {}
                    self._tools_discovered = False
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
                self._trigger_engine.start_driver()
                stack.push_async_callback(self._trigger_engine.stop_driver)
                yield

        self._mcp = FastMCP(server_name, lifespan=lifespan)
        self._mcp.settings.host = host
        self._mcp.settings.port = port
        self._mcp.settings.transport_security.enable_dns_rebinding_protection = False
        self._trigger_engine = TriggerEngine(self)
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
            return self._json_response({"mcp_servers": mcp_servers, "changelog_id": changelog_id, "roles": roles}, 200)

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
        inner = self._mcp.streamable_http_app()
        role_header_bytes = AGENT_ENV_ROLE_HEADER.lower().encode()

        async def app(scope, receive, send):
            if scope["type"] != "http":
                await inner(scope, receive, send)
                return
            role = DEFAULT_ROLE
            for name, value in scope.get("headers", ()):
                if name == role_header_bytes:
                    role = value.decode(errors="ignore").strip() or DEFAULT_ROLE
                    break
            token = _role_var.set(role)
            try:
                await inner(scope, receive, send)
            finally:
                _role_var.reset(token)

        config = uvicorn.Config(
            app,
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
        """Wrap the FastMCP tool manager so MCP tools/list and tools/call honor _role_var."""
        tm = self._mcp._tool_manager
        original_list_tools = tm.list_tools
        original_call_tool = tm.call_tool

        def filtered_list_tools():
            role = _role_var.get()
            return [t for t in original_list_tools() if not self._is_disabled(role, t.name)]

        async def filtered_call_tool(name, arguments, context=None, convert_result=False):
            role = _role_var.get()
            if self._is_disabled(role, name):
                raise ToolError(f"Tool '{name}' is disabled for role '{role}'")
            native = name not in self._tool_server_urls
            event_id = await self._log_native_tool_call(name, arguments) if native else None
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

        try:
            target_url = self._proxy_target_url(base_url, request)
        except ValueError as e:  # an escaped slash inside the route prefix; refuse rather than forward a wrong path
            return self._json_response({"error": str(e)}, 400)
        client = httpx.AsyncClient(timeout=self.REST_PROXY_TIMEOUT)
        upstream_request = client.build_request(
            request.method,
            target_url,
            content=await request.body(),  # /svc and /website request bodies are small (JSON-RPC, forms)
            headers=self._proxy_request_headers(request),
        )
        try:
            upstream = await client.send(upstream_request, stream=True)  # returns once the headers are in
        except Exception as e:
            await client.aclose()
            logger.error(f"Proxy to {target_url} failed: {e!r}")
            return self._json_response({"error": f"Proxy request failed: {e!r}"}, 502)
        return _UpstreamResponse(upstream, client, target_url)

    @staticmethod
    def _proxy_target_url(base_url: str, request: Request) -> str:
        # Forward the path as received: uvicorn decodes scope["path"] for routing, so Starlette's {path} param
        # would send %2F, %3F and %23 to the upstream as separators. latin-1 keeps one char per wire byte.
        url = base_url
        decoded_rest = request.path_params.get("path", "")
        if decoded_rest:
            raw_rest = _raw_remainder(request.scope["raw_path"], request.scope["path"], decoded_rest)
            url += "/" + raw_rest.decode("latin-1")
        query = request.scope.get("query_string", b"")
        if query:
            url += "?" + query.decode("latin-1")
        return url

    @staticmethod
    def _proxy_request_headers(request: Request) -> httpx.Headers:
        """One entry per raw header, so repeated fields (cookie, x-forwarded-for) keep their multiplicity."""
        headers = httpx.Headers([(k, v) for k, v in request.headers.raw if k.lower() not in _HOP_BY_HOP])
        # Bytes are forwarded undecoded, so only ask upstream for encodings the caller can take.
        headers.setdefault("accept-encoding", "identity")
        return headers

    def _backing_keys(self) -> list[str]:
        # rest_proxy_urls entries are the env's backing servers: mcp-<svc> (MCP) + <svc> (website
        # backends). Non-v1 backers (e.g. the injected website_browser) self-omit from the card via 404.
        return list(self.rest_proxy_urls)

    def _rewrite_child_card(self, key: str, card: dict) -> dict:
        """Rewrite every path the child names (its `url`, each extension's endpoint and each method's
        own) to the gateway-rooted `/svc/{key}/...`, so it resolves against the env's address. An
        operation the gateway's own extension with the same `uri` also offers (same method name, HTTP
        verb and endpoint, e.g. `disable`, POST `/tools/disable`) is the gateway's route and keeps its
        path; a child method falling back to that endpoint gets one of its own under the child's prefix. Backing
        servers are leaves, so any nested `children_environments` is dropped, and so are the child's
        interfaces: agents reach its tools through the gateway's aggregate MCP endpoint."""
        gateway_operations, gateway_requests, gateway_paths = set(), set(), set()
        for gateway_ext in self._gateway_extensions():
            for name, _, _, verb, path in self._operations(gateway_ext.get("params") or {}):
                gateway_operations.add((gateway_ext["uri"], name, verb, path))
                gateway_requests.add((gateway_ext["uri"], verb, path))
                gateway_paths.add((gateway_ext["uri"], path))

        def child_path(path: str) -> str:
            return f"/svc/{key}{path}" if path.startswith("/") else path

        card = copy.deepcopy(card)
        if card.get("children_environments"):
            logger.warning(f"env card: dropping nested children_environments from leaf {key} (multi-level composition unsupported)")
        card["children_environments"] = None
        card["additionalInterfaces"] = []
        url = card.get("url")
        if isinstance(url, str) and url:
            card["url"] = child_path(url.rstrip("/") or "/")
        for ext in (card.get("capabilities") or {}).get("extensions") or []:
            params = ext.get("params") if isinstance(ext, dict) else None
            if not isinstance(params, dict):
                continue
            uri = ext["uri"] if isinstance(ext.get("uri"), str) else None
            endpoint = params.get("endpoint") if isinstance(params.get("endpoint"), str) else None
            # A method is the gateway's when the gateway offers it under the same name; an extension
            # listing no methods is a bare request, the gateway's when the gateway serves it.
            operations = [(method, holder, (uri, name, verb, path) in gateway_operations
                           if name is not None else (uri, verb, path) in gateway_requests)
                          for name, method, holder, verb, path in self._operations(params)]
            # The extension's endpoint stays when a gateway operation falls back to it or, used by no
            # operation, when the gateway's extension with this uri names it.
            keep_endpoint = any(on_gateway and holder is params for _, holder, on_gateway in operations) or (
                all(holder is not params for _, holder, _ in operations) and (uri, endpoint) in gateway_paths)
            for method, holder, on_gateway in operations:
                if on_gateway:
                    continue
                if holder is not params:
                    holder["endpoint"] = child_path(holder["endpoint"])
                elif keep_endpoint and method is not None:
                    method["endpoint"] = child_path(params["endpoint"])
            if endpoint is not None and not keep_endpoint:
                params["endpoint"] = child_path(endpoint)
        return card

    @staticmethod
    def _operations(params: dict) -> list[tuple[str | None, dict | None, dict, str, str]]:
        """Each operation an extension offers, read the way `invoke_extension` reads it: (method name, method, the
        object naming its endpoint, HTTP verb, endpoint). A method without an endpoint of its own uses
        the extension's, its verb defaults to POST, and an extension listing no methods offers a POST
        to its endpoint. One without a text endpoint is skipped, so it stays as stored."""
        methods = params.get("methods")
        if isinstance(methods, dict) and methods:
            operations = [(name, method, method if method.get("endpoint") else params, str(method.get("method") or "POST").upper())
                          for name, method in methods.items() if isinstance(method, dict)]
        else:
            operations = [(None, None, params, "POST")]
        return [(name, method, holder, verb, holder["endpoint"]) for name, method, holder, verb in operations
                if isinstance(holder.get("endpoint"), str) and holder["endpoint"]]

    async def _fetch_child_card(self, client: httpx.AsyncClient, key: str, base_url: str) -> dict | None:
        """Fetch + rewrite one backing server's card; None (omit) on 404 / error / timeout / malformed JSON."""
        url = f"{base_url.rstrip('/')}{WELL_KNOWN_PATH}"
        try:
            response = await client.get(url, timeout=CARD_FETCH_TIMEOUT_S)
        except Exception as e:
            logger.warning(f"env card: fetching {key} failed ({type(e).__name__}: {e}); omitting")
            return None
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            logger.warning(f"env card: {key} returned HTTP {response.status_code}; omitting")
            return None
        try:
            child = response.json()
        except Exception:
            logger.warning(f"env card: {key} returned malformed JSON; omitting")
            return None
        return self._rewrite_child_card(key, child)

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
        """Open a persistent MCP client session to each internal server."""
        sessions: dict[str, ClientSession] = {}
        for server in self.internal_mcp_servers:
            for attempt in range(max_retries):
                try:
                    read_stream, write_stream, _ = await exit_stack.enter_async_context(streamable_http_client(server.mcp_url))
                    session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
                    await session.initialize()
                    sessions[server.mcp_url] = session
                    logger.info(f"Opened persistent session to {server.name}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to connect to {server.name} (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to connect to {server.name} after {max_retries} attempts: {e}")
                        raise
        return sessions

    async def _ensure_tools_discovered(self) -> None:
        """Lazy-init tool discovery for /step and /state endpoints."""
        if self._tools_discovered:
            return
        async with self._init_lock:
            if self._tools_discovered:
                return
            try:
                if self._step_exit_stack is None:
                    self._step_exit_stack = contextlib.AsyncExitStack()
                if not self._step_sessions:
                    self._step_sessions = await self._open_persistent_sessions(self._step_exit_stack)
                await self._discover_and_register_tools(self._step_sessions)
            except Exception:
                logger.exception("Tool discovery failed during lazy init")
                self._server_tools = {}
                self._tools_discovered = False
                self._step_sessions = {}
                if self._step_exit_stack is not None:
                    await self._step_exit_stack.aclose()
                    self._step_exit_stack = None
                raise

    async def _discover_and_register_tools(self, sessions: dict[str, ClientSession]) -> None:
        """Discover tools from all internal servers via persistent sessions and register proxies."""
        logger.info(f"Discovering tools from {len(self.internal_mcp_servers)} internal server(s)...")
        registered_tools: dict[str, str] = {}
        for server in self.internal_mcp_servers:
            try:
                session = sessions[server.mcp_url]
                tools_result = await session.list_tools()
                tools = tools_result.tools
                logger.info(f"Found {len(tools)} tool(s) from {server.name}")
                for tool in tools:
                    if tool.name in registered_tools:
                        raise ValueError(
                            f"Duplicate tool name '{tool.name}': already registered from "
                            f"'{registered_tools[tool.name]}', cannot register from '{server.name}'"
                        )
                    self._create_and_register_proxy_tool(server.mcp_url, tool)
                    self._tool_server_urls[tool.name] = server.mcp_url
                    registered_tools[tool.name] = server.name
                    logger.debug(f"Registered proxy tool: {tool.name}")
                self._server_tools[server.name] = tools
            except Exception as e:
                logger.error(f"Failed to discover tools from {server.name}: {e}")
                raise
        self._tools_discovered = True
        if self._mirror_get_time_descriptor():
            logger.info(f"Re-mirrored '{GET_TIME_TOOL_NAME}' descriptor after tool discovery")
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
        role = request.headers.get(AGENT_ENV_ROLE_HEADER, "").strip() or DEFAULT_ROLE
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

    async def _get_step_session(self, server_url: str) -> ClientSession:
        """Get or create a persistent MCP session for /step calls."""
        if not self._step_sessions:
            if self._step_exit_stack is None:
                self._step_exit_stack = contextlib.AsyncExitStack()
            self._step_sessions = await self._open_persistent_sessions(self._step_exit_stack)
        return self._step_sessions[server_url]

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
        })
        try:
            async with self._tool_call_guard():
                session = await self._get_step_session(server_url)
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments),
                        timeout=self.TOOL_CALL_TIMEOUT_S,
                    )
                except ClosedResourceError:
                    logger.warning(f"/step session to {server_url} closed, reconnecting...")
                    self._step_sessions.clear()
                    if self._step_exit_stack is not None:
                        await self._step_exit_stack.aclose()
                        self._step_exit_stack = None
                    session = await self._get_step_session(server_url)
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments),
                        timeout=self.TOOL_CALL_TIMEOUT_S,
                    )
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
        """Create a proxy tool that forwards calls to an internal server via its persistent session."""
        tool_name = tool.name
        tool_description = tool.description or ""
        input_schema = tool.inputSchema or {"type": "object", "properties": {}}

        async def proxy(**kwargs) -> CallToolResult:
            tool_call_event_id = await self._log_event({
                "event_type": "tool_call",
                "tool_call": {
                    "function_name": tool_name,
                    "arguments": kwargs,
                },
            })

            async with self._tool_call_guard():
                session = self._mcp_sessions[server_url]
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, kwargs),
                    timeout=self.TOOL_CALL_TIMEOUT_S,
                )
                changelog_id = self._query_changelog_id()

            await self._log_event({
                "event_type": "tool_call_result",
                "tool_call_event_id": tool_call_event_id,
                "tool_call_result": result.model_dump(mode="json"),
                "changelog_id": changelog_id,
            })
            return result

        arg_model = self._create_arg_model_from_schema(tool_name, input_schema)
        fastmcp_tool = FastMCPTool(
            fn=proxy,
            name=tool_name,
            description=tool_description,
            parameters=input_schema,
            fn_metadata=FuncMetadata(arg_model=arg_model),
            is_async=True,
        )
        self._mcp._tool_manager._tools[tool_name] = fastmcp_tool

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

    async def _log_native_tool_call(self, name: str, arguments: dict | None) -> str | None:
        """Trajectory record for a gateway-owned tool on the MCP path (fail-open)."""
        try:
            return await self._log_event({
                "event_type": "tool_call",
                "tool_call": {"function_name": name, "arguments": arguments or {}},
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


# HTTP (RFC 7230) hop-by-hop headers, never proxied in either direction.
_HOP_BY_HOP = frozenset({
    b"host", b"connection", b"keep-alive", b"transfer-encoding",
    b"te", b"trailer", b"upgrade", b"proxy-authorization", b"proxy-authenticate",
})


class _UpstreamResponse(StreamingResponse):
    """Streams an httpx response through untouched and closes it when the response ends, iterated or not.

    The close lives in `__call__` rather than the body generator because a client that leaves before the first
    chunk makes Starlette cancel before the generator's first step, and a never-started generator runs no `finally`."""

    _DROP = _HOP_BY_HOP | {b"date", b"server"}  # uvicorn adds the gateway's own; forwarding upstream's duplicates them

    def __init__(self, upstream: httpx.Response, client: httpx.AsyncClient, target_url: str) -> None:
        super().__init__(self._body(upstream, target_url), status_code=upstream.status_code)
        # Header block forwarded byte for byte: repeated names (Set-Cookie) stay separate, non-latin-1 values
        # never hit a str round-trip, and Content-Length/Content-Encoding stay valid for the undecoded body.
        self.raw_headers = [(k.lower(), v) for k, v in upstream.headers.raw if k.lower() not in self._DROP]
        self._upstream = upstream
        self._client = client

    @staticmethod
    async def _body(upstream: httpx.Response, target_url: str) -> AsyncIterator[bytes]:
        sent = 0
        try:
            async for chunk in upstream.aiter_raw():
                sent += len(chunk)
                yield chunk
        except Exception as e:  # past the headers a failure can only reach the client as a cut body; name it here
            logger.error(f"Proxy to {target_url} failed after {sent} bytes: {e!r}")
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:  # normal end, upstream error, or the downstream client going away
            await self._upstream.aclose()
            await self._client.aclose()


def _raw_remainder(raw_path: bytes, decoded_path: str, decoded_rest: str) -> bytes:
    """The still-encoded tail of `raw_path` that Starlette's decoded `{path}` param came from.

    >>> _raw_remainder(b"/svc/mcp-svc/x%2Fy%3Fz", "/svc/mcp-svc/x/y?z", "x/y?z")
    b'x%2Fy%3Fz'
    """
    prefix = decoded_path.removesuffix(decoded_rest)  # "/svc/mcp-svc/"
    rest = raw_path.split(b"/", prefix.count("/"))[-1]  # 3 slashes in both forms, so the tail starts after the 3rd
    if unquote(rest.decode("latin-1")) != decoded_rest:  # only false when an escaped slash hid inside the prefix
        raise ValueError(f"cannot route request-target {raw_path.decode('latin-1')!r}: percent-encoded slash in the route prefix")
    return rest
