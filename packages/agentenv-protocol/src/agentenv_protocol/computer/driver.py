"""ComputerDriverEnvironment: a driver writes screen operations, and the base class serves the computer contract
from them: the tools agents call, the framework extensions agentenv calls, the data plane and the card, which lists the
computer's apps as child environments, each reached at ``/svc/<name>/`` through the driver."""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Optional

import httpx
import websockets
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import ClientConnection, unix_connect

from ..agent_env_environment import (_CARD_CONFIG_ATTR, AgentEnvEnvironment, AgentEnvFastMCPApplication,
                                     _error_response, control_check, control_guard, environment_card)
from ..tool_context import ToolContext
from ..types import EnvironmentCard, EnvironmentExtension, EnvironmentTool
from . import _linux, apps, card
from .app import SOCKET
from .checks import check_features, check_names
from .dataplane import DataPlane
from .errors import DeliveryFailed, TargetUnavailable
from .extensions import Extensions
from .operations import ReadyCheck, check_class, default_for
from .platform import Platform, detect
from .recording import Recording
from .runtime import ComputerRuntime
from .settings import ComputerSettings, load_settings
from .tools import Tools

#: The caller every call comes from until agents carry an identity of their own.
DEFAULT_IDENTITY = "default"
#: The methods ``/svc/<name>/`` forwards to an app.
FORWARDED_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
#: A WebSocket close code that means the connection ended without one, which a close can't send on.
UNSENDABLE_CLOSES = (1005, 1006, 1015)


class DriverPlatform(BaseModel):
    """What a driver serves on one platform its ``supports`` names: the extensions it ``provides``, its ``tools`` and
    its computer/v1 ``features``."""

    model_config = ConfigDict(extra="forbid")

    platform: Platform
    provides: list[str]
    tools: list[str]
    features: dict[str, Any]


class DriverDeclaration(BaseModel):
    """What a ComputerDriverEnvironment class declares, as a ``computer_driver`` env stores it: its card's ``name``, the
    platforms it ``supports``, as predicates, and what it serves on each of them."""

    model_config = ConfigDict(extra="forbid")

    name: str
    supports: list[dict[str, Any]]
    platforms: list[DriverPlatform]


class ComputerDriverEnvironment(Tools, Extensions, DataPlane, AgentEnvEnvironment):
    """Subclass it and mark the methods that capture and send input (see ``operations``); the class is checked when
    it is defined. Features are class attributes, on the card. A subclass's ``__init__`` calls ``super().__init__()``.
    """

    buttons: ClassVar[tuple[str, ...]] = ("left",)
    max_click_count: ClassVar[int] = 1
    #: ``"all"`` for every W3C key, or the key values this computer can press.
    keys: ClassVar[Any] = "all"
    max_text_length: ClassVar[int] = 10_000
    max_wait_ms: ClassVar[int] = 60_000
    max_drag_points: ClassVar[int] = 2
    max_duration_ms: ClassVar[int] = 10_000
    max_key_repeat: ClassVar[int] = 100
    max_scroll_amount: ClassVar[int] = 100
    #: The (width, height) modes computer-display/v1 set takes; with ``@set_display`` only.
    display_modes: ClassVar[tuple] = ()
    layouts: ClassVar[tuple[str, ...]] = ("us",)
    #: The platforms the driver supports, as the driver env declares them: ``[{"os": {...}, "arch": [...]}]``.
    supports: ClassVar[list] = []
    #: computer-recording/v1, from ``@record`` if the driver has one, else by polling ``@capture``; ``None`` turns
    #: it off.
    recording: ClassVar[Recording | None] = Recording()
    #: What an agent should know about this driver's computer beyond what its card says, as markdown: the end of its skill.
    notes: ClassVar[str] = ""
    #: What the computer's screen shows, its card's ``screen``: a ``desktop``, or a website browser's ``page``, its
    #: active tab.
    screen_kind: ClassVar[Literal["desktop", "page"]] = "desktop"
    tool_access = True
    #: What the base class keeps for the driver, built by ``__init__``.
    _computer: ComputerRuntime | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        check_names(cls, ComputerDriverEnvironment)
        check_class(cls)
        check_features(cls)

    def __init__(self, settings: ComputerSettings | None = None, platform: Platform | None = None) -> None:
        self.settings = settings if settings is not None else load_settings()
        self.platform = platform if platform is not None else detect()
        self._computer = ComputerRuntime(self)

    @classmethod
    def declaration(cls, platform: Platform) -> EnvironmentCard:
        """The card this driver class serves on ``platform`` with default settings, under its own name, built without
        running its ``__init__``: what a ``computer_driver`` env records for each platform it supports. The card
        hooks may read the class and the base class's state, not what a subclass's ``__init__`` sets."""
        driver = cls.__new__(cls)
        ComputerDriverEnvironment.__init__(driver, settings=ComputerSettings(), platform=platform)
        config = dict(getattr(cls, _CARD_CONFIG_ATTR, None) or {})
        card = EnvironmentCard(**{**config, "name": config.get("name") or cls.__name__})
        return AgentEnvFastMCPApplication(card, driver).environment_card

    def has(self, capability: str) -> bool:
        """Whether an operation, or a whole group (``pointer``, ``keyboard``), works on this platform."""
        return self._computer.has(capability)

    def caller_identity(self, context: ToolContext) -> str:
        """Who a call comes from, for each caller's latest frame and others' input since a frame."""
        return DEFAULT_IDENTITY

    async def ready_checks(self) -> list[ReadyCheck]:
        """Checks of the driver's own, run with the base class's by computer-ready/v1; never sends input."""
        return []

    async def bundled_skill(self) -> tuple[str, str, dict[str, str]] | None:
        """The skill the software this driver drives ships for agents, as its description, its SKILL.md body without
        frontmatter, and its other files by relative path; the computer skill leads with it. None by default."""
        return None

    def create_app(self) -> Any:
        self._computer_require_init()
        return self._computer_front_apps(super().create_app())

    def mount(self, app: Any) -> Any:
        self._computer_require_init()
        return self._computer_front_apps(super().mount(app))

    async def children_environments(self) -> list[EnvironmentCard] | None:
        """The computer's apps, from the folders it composed them into."""
        return await apps.children(self._computer.apps) if self._computer.apps else None

    def _computer_front_apps(self, app: Any) -> Any:
        app.custom_route("/svc/{name}/{path:path}", methods=FORWARDED_METHODS)(control_guard()(self._computer_forward))
        _websocket_route(app, "/svc/{name}/{path:path}", self._computer_relay)
        return app

    def _computer_app_folder(self, name: str) -> Optional[Path]:
        return next((folder for folder in self._computer.apps if folder.name == name), None)

    async def _computer_forward(self, request: Request) -> Response:
        """A request under ``/svc/<name>/``, sent on to the app named so, its body and its answer's streamed: its own
        server answers it."""
        name = request.path_params["name"]
        folder = self._computer_app_folder(name)
        if folder is None:
            return _error_response(TargetUnavailable(f"this computer has no app named {name!r}"))
        headers = apps.to_app(request.headers.items(), request.headers.get("host", ""), request.url.scheme,
                              f"/svc/{name}")
        has_body = request.headers.get("content-length", "0") != "0" or "transfer-encoding" in request.headers
        try:
            answer, close = await apps.forward(folder, request.method, _target(request), headers,
                                               request.stream() if has_body else None)
        except (httpx.HTTPError, OSError) as e:
            return _error_response(DeliveryFailed(f"app {name}'s server isn't answering: {type(e).__name__}: {e}"))

        async def streamed() -> AsyncIterator[bytes]:
            try:
                async for chunk in answer.aiter_raw():
                    yield chunk
            finally:
                await close()

        response = StreamingResponse(streamed(), status_code=answer.status_code, background=BackgroundTask(close))
        response.raw_headers = [(key.encode("latin-1"), value.encode("latin-1"))
                                for key, value in apps.passed_on(answer.headers.multi_items())]
        return response

    async def _computer_relay(self, websocket: WebSocket) -> None:
        """A WebSocket under ``/svc/<name>/``, relayed both ways to the app named so until either side closes; refused,
        before it opens, without the control token or with no app or none answering."""
        name = websocket.path_params["name"]
        folder = self._computer_app_folder(name)
        if folder is None or not control_check()(websocket.headers):
            await websocket.close(code=1008)
            return
        host = websocket.headers.get("host", "")
        headers = [(key, value) for key, value in apps.to_app(
            websocket.headers.items(), host, "https" if websocket.url.scheme == "wss" else "http", f"/svc/{name}")
            if key.lower() != "host" and not key.lower().startswith("sec-websocket-")]
        offered = [p.strip() for p in websocket.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
        try:
            upstream = await unix_connect(str(folder / SOCKET), f"ws://{host or 'app'}{_target(websocket)}",
                                          additional_headers=headers, subprotocols=offered or None, max_size=None,
                                          compression=None, ping_interval=None)
        except (OSError, websockets.InvalidHandshake, asyncio.TimeoutError):
            await websocket.close(code=1011)
            return
        async with upstream:
            await websocket.accept(subprotocol=upstream.subprotocol)
            await _relay(websocket, upstream)

    # The card: what the driver's operations support.

    def served_tools(self, tools: list[EnvironmentTool]) -> list[EnvironmentTool]:
        return card.served_tools(self, tools)

    def served_extensions(self, extensions: list[EnvironmentExtension]) -> list[EnvironmentExtension]:
        return card.served_extensions(self, extensions)

    def served_operations(self, operations: list[str]) -> list[str]:
        return card.served_operations(self, operations)

    def extension_params(self, uri: str) -> dict:
        return card.extension_params(self, uri)

    def declared_extensions(self) -> list[EnvironmentExtension]:
        return card.declared_extensions(self)

    def _computer_require_init(self) -> None:
        if self._computer is None:
            raise TypeError(f"{type(self).__qualname__}.__init__ must call super().__init__()")

    # The Linux defaults for the OS operations.

    @default_for("run_in_session", os="linux")
    async def _computer_linux_run_in_session(self, argv: list, cwd: str | None, env: dict | None, timeout_s: float):
        return await _linux.run_in_session(argv, cwd, env, timeout_s, output_bytes=self.settings.exec_output_bytes)

    @default_for("write_files", os="linux")
    async def _computer_linux_write_files(self, files: dict) -> None:
        await asyncio.to_thread(_linux.write_files, files)


def _target(connection: Request | WebSocket) -> str:
    """The path and query a request under ``/svc/<name>/`` asks its app for."""
    query = connection.url.query
    return f"/{connection.path_params['path']}" + (f"?{query}" if query else "")


def _websocket_route(app: Any, path: str, endpoint: Callable) -> None:
    """``endpoint`` served at ``path`` as a WebSocket route of ``app``, an mcp or a standalone fastmcp FastMCP, which
    have no API for one."""
    for routes in ("_custom_starlette_routes", "_additional_http_routes"):
        if isinstance(getattr(app, routes, None), list):
            getattr(app, routes).append(WebSocketRoute(path, endpoint))
            return
    raise TypeError(f"a driver's app takes WebSocket routes on an mcp or a fastmcp FastMCP, not a {type(app).__name__}")


async def _relay(client: WebSocket, app: ClientConnection) -> None:
    """Each message from either side sent to the other until one side closes, then the other closed with its code."""
    async def inbound() -> None:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                code = message.get("code") or 1000
                await app.close(1000 if code in UNSENDABLE_CLOSES else code)
                return
            await app.send(message["text"] if message.get("text") is not None else message["bytes"])

    async def outbound() -> None:
        with contextlib.suppress(websockets.ConnectionClosed):
            async for message in app:
                if isinstance(message, str):
                    await client.send_text(message)
                else:
                    await client.send_bytes(message)
        code = app.close_code or 1000
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await client.close(1000 if code in UNSENDABLE_CLOSES else code)

    tasks = [asyncio.ensure_future(inbound()), asyncio.ensure_future(outbound())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def computer_driver(name: str | None = None, *, supports: list | None = None) -> Callable:
    """Name a driver, its card's name unless ENVIRONMENT_NAME gives one, and declare the platforms it supports."""
    def deco(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, ComputerDriverEnvironment)):
            raise TypeError("@computer_driver decorates a ComputerDriverEnvironment subclass")
        if supports is not None:
            cls.supports = list(supports)
        return environment_card(name=name)(cls) if name else cls
    return deco
