"""WebsiteBrowserEnvironment, its settings, and ``browser/v1`` on its card."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, ClassVar, Literal, Optional
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.websockets import WebSocket

from ..agent_env_environment import (OP_ADD_DATA, add_data, add_websocket_route, agent_skill, extension, get_data,
                                     reset_data)
from .cdp import ROUTE, Relay, Tokens, reached, under, websocket_form
from .log import (ACTOR, ERROR_CHARS, TOOL_CALL_HEADER, UNLOGGED_OPS, BrowserLog, op_acts, read_tool_call,
                  redact_text, redacted_args, redacted_op)
from .network import HAR_FILE, NetworkLog, log_path
from ..computer import AppDeclaration, ComputerAppEnvironment, Launch, install_step, ready_check
from ..computer import app as computer_app
from ..computer import app_data
from ..computer.app import SOCKET, agent_acted, forget_agent_acted
from ..computer.dataplane import file_parts
from ..computer.errors import DeviceUnavailable, InvalidArgument, PermissionDenied, TargetUnavailable
from ..tool_context import DEFAULT_ROLE, ROLE_HEADER, normalize_role
from ..transfers import WriteNamespaceGrant
from ..types import METHOD_ADD, RPC_PATH, EnvironmentExtension, IntakeDeclaration, IntakeFormat, intake_extension
from ..website.readiness import names_host
from . import data
from .contexts import CONTEXTS, TASK, Contexts, label_problem
from .scope import Scope
from .control import AGENT, DOWNLOADS, EXIT, LIVE_START_TAB_TIMEOUT_S, LOADED, STOP, UPLOADS, BrowserControl, _now
from .devtools import PIPE_IN, PIPE_OUT, DevToolsError, DevToolsPipe
from .recording import BROWSER_RECORDING, MAX_DURATION_S, MAX_FRAMES, BrowserRecordings
from .shapes import (BrowserRecord, BrowserState, ContextState, ContextToLoad, CookieToSet, OriginStorage, SiteNames,
                     StateToLoad)

logger = logging.getLogger(__name__)

SITE_PROXY_ENV = "AGENTENV_SITE_PROXY"
BROWSER = "urn:agentenv:browser/v1"
#: Where ``browser/v1``'s ops are served on its server.
OPS_PATH = f"{RPC_PATH}/ext/browser/"
#: The largest op body whose parameters its entry keeps; a longer one keeps only its length.
LOGGED_BODY_BYTES = 1 << 20
#: Where its server keeps the env's sites ``sites/set`` hands it, which outlive a relaunch or the server's restart.
SITES = SOCKET.parent / "sites.json"
#: The flags agentenv sets, by prefix, which a subclass's command may not carry: among them every way to open a debugging
#: port, since the browser is reached only over its pipe.
OWNED_FLAGS = ("--user-data-dir", "--remote-debugging-", "--remote-allow-origins", "--proxy-", "--no-proxy-server",
               "--headless", "--window-size", "--lang", "--force-webrtc-ip-handling-policy", "--disk-cache-dir",
               "--disable-dev-shm-usage", "--enable-unsafe-swiftshader")
#: Below this much /dev/shm, as Docker's 64 MiB default, the browser keeps its shared memory in files instead.
SHM_MIN_BYTES = 1 << 30
#: Chromium's content setting that denies a permission without a prompt.
BLOCK = 2
#: How often a call that waits for the browser to be ready checks it.
READY_POLL_S = 0.5
#: How much longer than its ready timeout, within which a launch's start tabs settle, a call waits for them.
START_TABS_MARGIN_S = 30.0
#: Alone, loopback goes through the proxy too. In a computer, the apps on loopback beside it stay reachable; ``<-loopback>``
#: removes Chromium's own rule, so ``*.localhost`` names still don't.
ALONE_BYPASS = "<-loopback>"
COMPUTER_BYPASS = "<-loopback>;127.0.0.1;[::1];localhost"
#: Managed policies for unattended use: no sign-in, sync, password manager, autofill, metrics, promotions, translation,
#: background mode, component updates, default-browser check, download prompt or warning bar for its flags.
POLICIES: dict[str, Any] = {
    "BrowserSignin": 0,
    "SyncDisabled": True,
    "PasswordManagerEnabled": False,
    "AutofillSettings": [{"url_pattern": "*", "blocked_types": ["all"]}],
    "MetricsReportingEnabled": False,
    "PromotionalTabsEnabled": False,
    "TranslateEnabled": False,
    "BackgroundModeEnabled": False,
    "ComponentUpdatesEnabled": False,
    "DefaultBrowserSettingEnabled": False,
    "PromptForDownloadLocation": False,
    "CommandLineFlagSecurityWarningsEnabled": False,
    "DownloadDirectory": "${user_home}/Downloads",
}
SKILL = f"""# Website browser

- Open sites by their real names, such as `https://slack.com`: the environment serves its sites under those names.
- Files to upload are in `~/{UPLOADS}`; downloads land in `~/{DOWNLOADS}`, and a leased context's in
  `~/{CONTEXTS}/<label>/{DOWNLOADS}`.
"""


class Window(BaseModel):
    width: int = Field(1280, gt=0)
    height: int = Field(800, gt=0)


class BrowserSettings(BaseModel):
    """The settings every website browser takes, per deploy; a browser's own ``Settings`` adds to them."""

    #: Opened in tabs at launch, the first active.
    start_urls: list[str] = Field(default_factory=lambda: ["about:blank"])
    #: Its window when it runs alone; in a computer the screen sets it.
    window: Window = Field(default_factory=Window)
    locale: str = "en-US"
    #: The IANA time zone its pages see.
    time_zone: str = "UTC"
    #: Whether a page asking for the location, or to show notifications, is denied without a prompt or asks.
    geolocation: Literal["deny", "ask"] = "deny"
    notifications: Literal["deny", "ask"] = "deny"
    #: Who answers a page's JavaScript dialogs: the agent, or agentenv accepting or dismissing each.
    dialogs: Literal["agent", "accept", "dismiss"] = "agent"
    #: Whether ``data/get`` exports its network log, ``network.har``, and ``browser/v1`` ``network`` reads it: off, a
    #: run's export holds none of the cookies and form bodies the log keeps.
    network: bool = True

    @field_validator("start_urls")
    @classmethod
    def _no_flags(cls, urls: list[str]) -> list[str]:
        if flags := [url for url in urls if url.startswith("-")]:
            raise ValueError(f"start_urls are URLs, not flags: {flags}")
        return urls


class WebsiteBrowserEnvironment(ComputerAppEnvironment):
    """Subclass it as any computer app, with ``@computer_app`` and ``@linux``, its launch running the browser itself,
    and set ``version``, its build, and ``policy_dir``, where that build reads managed policies. ``policies`` adds its
    own to agentenv's, and ``profile`` is its profile folder under the session user's home, holding what its install
    steps put there."""

    Settings = BrowserSettings
    version: ClassVar[str] = ""
    policy_dir: ClassVar[Path] = Path("/etc/chromium/policies/managed")
    policies: ClassVar[dict[str, Any]] = {}
    profile: ClassVar[str] = ".config/chromium"
    #: How long its ready check waits on the browser's answer over its pipe.
    devtools_timeout_s: ClassVar[float] = 5.0
    agent_routes = (ROUTE,)
    #: A load sets cookies and storage over its pipe, so the browser runs through it; its reset puts the profile
    #: back as installed, which its server keeps.
    data_live = frozenset({OP_ADD_DATA})
    data_keeps_paths = True

    def __init__(self) -> None:
        #: Its CDP connection, over the pipe its server launches it with.
        self.devtools = DevToolsPipe()
        self._following: Optional[asyncio.Future] = None
        #: The start tabs of each launch since this server started, newest last.
        self.start_tab_launches: list[dict] = []
        super().__init__()
        for folder in (UPLOADS, DOWNLOADS):
            (Path.home() / folder).mkdir(exist_ok=True)
        #: agentenv's control of it over that connection, and the record it keeps.
        self.control = BrowserControl(self.devtools, self.settings.dialogs,
                                      (self.settings.window.width, self.settings.window.height) if _alone()
                                      else None)
        #: Its log: an entry for each action agents take on it, and its tabs' events.
        self.log = BrowserLog.for_app(os.environ.get("ENVIRONMENT_NAME") or type(self).__name__.lower(),
                                      SOCKET.parent)
        self.control.log = self.log
        #: The contexts it leases beside its default one, which no relaunch outlives.
        self.contexts = Contexts(self.devtools, self.control)
        self.control.download_folders = self.contexts.holding
        #: Its network log, as the proxy it goes out through writes it; None when no proxy logs for it.
        self.network = NetworkLog(path) if (path := log_path()) is not None else None
        #: The CDP addresses it has handed out, and the relay agents' connections to them go through.
        self.cdp_tokens = Tokens(SOCKET.parent / "cdp-tokens.json")
        self.relay = Relay(self.devtools, dialogs_answered=self.settings.dialogs != AGENT, log=self.log,
                           control=self.control, contexts=self.contexts)
        self.contexts.on_end.append(self._lease_ended)
        #: Its recordings: frames of the tab an agent last acted in, and its log, uploaded under each one's grant.
        self.recordings = BrowserRecordings(self.log, self.control)
        self.devtools.on_close(self._pipe_closed)

    @classmethod
    def declaration(cls) -> AppDeclaration:
        """What the class declares, a website browser; ValueError for a launch whose command carries a flag agentenv
        owns."""
        for declared in cls._installs.values():
            if declared.launch is not None:
                _refuse_owned(cls.__name__, declared.launch)
        return super().declaration().model_copy(update={"browser": True})

    def profile_dir(self) -> Path:
        return Path.home() / self.profile

    def cache_dir(self) -> Path:
        """Its HTTP cache, kept out of its profile, which holds only its data."""
        return Path.home() / ".cache" / Path(self.profile).name

    def agentenv_flags(self) -> list[str]:
        """The flags agentenv owns. A browser alone goes out only through its site proxy, so it won't launch without
        one; in a computer it goes through one when the computer has one, loopback aside. Alone it runs headful on the
        X server its computer gives it, where WebGL draws with SwiftShader since no GPU is there, else headless."""
        settings, alone, proxy = self.settings, _alone(), os.environ.get(SITE_PROXY_ENV)
        flags = [f"--user-data-dir={self.profile_dir()}", f"--disk-cache-dir={self.cache_dir()}",
                 "--remote-debugging-pipe", "--no-first-run", "--no-default-browser-check", f"--lang={settings.locale}"]
        if (shm := _shm_bytes()) < SHM_MIN_BYTES:
            logger.info(f"/dev/shm holds {shm // 2**20} MiB, under {SHM_MIN_BYTES // 2**20} MiB: the browser keeps its "
                        "shared memory in files")
            flags.append("--disable-dev-shm-usage")
        if proxy:
            flags += [f"--proxy-server={proxy}", f"--proxy-bypass-list={ALONE_BYPASS if alone else COMPUTER_BYPASS}",
                      "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"]
        elif alone:
            raise ValueError(f"a website browser alone goes out only through its site proxy, and {SITE_PROXY_ENV} "
                             "names none")
        if alone:
            flags.append(f"--window-size={settings.window.width},{settings.window.height}")
            flags.append("--enable-unsafe-swiftshader" if _x_server() else "--headless=new")
        return flags

    def _launch_for(self, launch: Launch) -> Launch:
        """The subclass's launch with agentenv's flags and the start tabs after its command, and the time zone in its
        environment; alone, with no window to match or wait for, its ``ready`` keeping its port and command, and
        none when that leaves it nothing, so its ready checks say when it's ready."""
        _refuse_owned(type(self).__name__, launch)
        update: dict[str, Any] = {"command": [*launch.command, *self.agentenv_flags(), *self._command_line_tabs()],
                                  "env": {**(launch.env or {}), "TZ": self.settings.time_zone}}
        if _alone():
            ready = launch.ready
            waits = ready is not None and (ready.port is not None or bool(ready.command))
            update.update(window=None, ready=ready.model_copy(update={"window": False}) if waits else None)
        return launch.model_copy(update=update)

    def _on_desktop(self) -> bool:
        return not _alone()

    def _command_line_tabs(self) -> list[str]:
        """The tab its command line opens, blank: its start tabs open over its pipe, so each is held to its window and
        followed as it loads."""
        return ["about:blank"]

    async def _launch_fds(self) -> dict[int, int]:
        """Before each launch, an empty cache, no lease's folder, as a server killed with leases leaves them, and the
        deploy's prompt answers in its profile; and a new debugging pipe: the browser's ends as its fds 3 and 4, this
        server's ends its CDP connection, which follows the browser and then opens the start tabs its command line
        can't."""
        shutil.rmtree(self.cache_dir(), ignore_errors=True)
        self.contexts.ended()
        self._write_prompt_answers()
        to_browser_read, to_browser_write = os.pipe()
        from_browser_read, from_browser_write = os.pipe()
        await self.devtools.connect(to_browser_write, from_browser_read)
        self._following = asyncio.ensure_future(self._follow(self.settings.start_urls, _now(),
                                                             asyncio.get_running_loop().time()))
        self._following.add_done_callback(lambda done: done.cancelled() or done.exception())
        return {PIPE_IN: to_browser_read, PIPE_OUT: from_browser_write}

    def _pipe_closed(self) -> None:
        """The browser closed its pipe: its tabs closed by its exit when it ended on its own, else by its server's
        stop, and every lease with it."""
        self.control.ended(EXIT if self._computer_app.exited() else STOP)
        self.relay.ended()
        self.contexts.ended()

    async def _follow(self, urls: list[str], at: str, launched: float) -> None:
        """Its control attached, then its start tabs at ``urls`` loaded, the launch's outcomes kept."""
        await self.control.attach()
        await self._load_start_tabs(urls, at, launched)

    async def _load_start_tabs(self, urls: list[str], at: str, since: Optional[float] = None) -> list[dict]:
        """Its start tabs at ``urls``, as the control loads them, every one settled within the browser's ready timeout
        of ``since``, its launch's start on the loop's clock, else of now: a tab on one of the env's sites waited on
        what's left of it, any other on the live wait at most. A launch's setup counts against it, so its start tabs
        settle before anything waiting that timeout and a margin on it gives up. Their outcomes, kept as a launch's
        from ``at``."""
        sites, now = _read_sites(), asyncio.get_running_loop().time()
        left = max(0.0, self._computer_app.timeout_s - (now - (since if since is not None else now)))
        tabs = await self.control.load_start_tabs(urls, lambda url: _site_of(url, sites), left,
                                                  min(LIVE_START_TAB_TIMEOUT_S, left))
        self.start_tab_launches.append({"at": at, "tabs": tabs})
        return tabs

    def _write_prompt_answers(self) -> None:
        """Its profile's preferences, which it reads as it starts, denying location and notification prompts or
        leaving them to ask, as the deploy's settings say."""
        preferences = self.profile_dir() / "Default" / "Preferences"
        try:
            current = json.loads(preferences.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
        current = current if isinstance(current, dict) else {}
        defaults = current.setdefault("profile", {}).setdefault("default_content_setting_values", {})
        for name in ("geolocation", "notifications"):
            if getattr(self.settings, name) == "deny":
                defaults[name] = BLOCK
            else:
                defaults.pop(name, None)
        preferences.parent.mkdir(parents=True, exist_ok=True)
        preferences.write_text(json.dumps(current), encoding="utf-8")

    def _relaunch_on_start(self) -> bool:
        return True

    def _launch_running(self) -> Callable[[], Awaitable[None]]:
        return self._open_window

    async def _open_window(self) -> None:
        """Another window, which a launch of the browser while it runs opens over its pipe: a second copy would only
        hand its window to the first and exit, taking its new pipe with it."""
        await self.control.create_target({"url": "about:blank", "newWindow": True})

    @install_step(scope="system")
    def write_policies(self) -> None:
        """agentenv's unattended-use policies, with the subclass's over them, where its build reads them."""
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        (self.policy_dir / "agentenv.json").write_text(json.dumps({**POLICIES, **self.policies}, indent=1) + "\n",
                                                       encoding="utf-8")

    @ready_check
    async def devtools_answers(self) -> bool:
        """It answers over its pipe."""
        try:
            await self.devtools.call("Browser.getVersion", timeout_s=self.devtools_timeout_s)
        except (ConnectionError, DevToolsError, asyncio.TimeoutError):
            return False
        return True

    @ready_check
    async def start_tabs_loaded(self) -> bool:
        """Its control follows it and every start tab of its launch has settled: loaded, failed or waited on its
        time."""
        following = self._following
        return following is not None and following.done() and not following.cancelled() and \
            following.exception() is None

    @extension(BROWSER, op="record", method="GET", description=(
        "A website browser agentenv runs: its build, the clock its pages see and, alone, its window. record returns "
        "what agentenv's own connection to it saw: its open tabs, each with whether it's active, the tabs closed and "
        "who closed them, its downloads with their size and sha256, its JavaScript dialogs and who answered each, and "
        "each launch, stop and exit of it since the computer booted, with its cause. page/viewport, page/capture and "
        "page/input are its active tab's page as a computer's page driver sees it: its size in pixels, which headless "
        "is its window's, a PNG of it, and CDP input calls sent to it in order. navigate and history open a URL in "
        "the active tab, or go back, forward or reload; tabs/list, tabs/open, tabs/activate and tabs/close manage "
        "its tabs by their ids; dialog/answer answers the dialog open on the active tab; files/choose gives the file "
        "chooser open on it files in ~/Uploads, alone; and page/state is the active tab's id, URL and title and "
        "what is open on it. cookies/get, cookies/set and cookies/delete, and local_storage/get, local_storage/set and "
        "local_storage/clear, read and change its default context's cookies and local storage as Playwright's "
        "storageState has them, a cookie's max_age set as an expiry on the browser's clock; downloads/list and "
        "downloads/get are its downloads, get with its file, uploaded under a write_namespace grant or inline. "
        "contexts/lease leases a context beside the default one by a label, lowercase letters, digits and '-': its "
        "own cookies, local storage, tabs and downloads, in ~/Contexts/<label>/Downloads, the one leased under that "
        "label before when there is one; contexts/grant records an agent's user it's for, which nothing enforces yet; "
        "contexts/release ends it, closing its tabs and deleting its folder; and contexts/list lists them, each with "
        "who made it: the task, or the agent whose CDP connection made it, leased to it as agent-<user>-<n>. Naming "
        "a lease's label as context, the tabs, cookies, local_storage and downloads ops act on that context, else on "
        "the default one. A reset or relaunch of the browser ends every lease. cdp "
        "returns a user's CDP address, http_url to discover from and ws_url to connect to, under origin, where the "
        "caller's agent reaches the env, when it names one, else at the address the call reached the browser at, "
        "with a token agentenv mints in its path: each connection to it gets its own browser-level session behind a "
        "guard, and record lists them as cdp_connections, each with the commands the guard refused or changed. "
        "Naming a lease's label as context, cdp returns an address scoped to it, for a user it's granted to: its "
        "connections see that lease as the browser's default context, its targets alone, pages, iframes and workers, "
        "with the contexts their user made, and what they open, set or download lands in it; the address ends with "
        "the lease. An address with no context sees the default context and its user's own, not the leases. "
        "sites/set hands it the env's own sites, by name and domains: a start tab on one must load, its document's "
        "status 2xx, or its launch counts it failed, while one on any other site is waited on 30 s at most. record "
        "keeps each launch's start_tabs: the URL asked for and ended on, the status, any network error, the load time, "
        "the env's site and the outcome, loaded, failed or timed_out. log returns its log's lines after the sequence "
        "number after, in order: an action entry for each CDP command, MCP tool call and pixel action an agent took, "
        "marked acting or reading, with who, the tab and its URL and, acting, its parameters with secrets taken out; "
        "and a tab event each time a tab opened, navigated or closed, with what caused it. network returns its "
        "network log's lines after the sequence number after, in order, as the proxy it goes out through writes them: "
        "an open line once a request has gone upstream or a tunnel has opened and an end line, with the same id, once "
        "it's done or has failed. The shared site proxy writes a whole exchange, the request and the answer with their "
        "headers and bodies, the sign-in headers a site declares removed, and a WebSocket's message counts; the base's "
        "stand-in writes each connection, its host, address and bytes each way. With the network setting off, or on a "
        "computer with a desktop, where no proxy logs for it, network returns no lines."))
    async def browser_record(self) -> dict:
        return BrowserRecord(**await self.control.record(), launches=self._computer_app.launches(),
                             cdp_connections=list(self.relay.connections),
                             start_tabs=self.start_tab_launches).model_dump()

    @extension(BROWSER, op="log", method="GET")
    async def browser_log(self, after: int = 0) -> dict:
        return await self.log.read_async(after)

    @extension(BROWSER, op="network", method="GET")
    async def browser_network(self, after: int = 0) -> dict:
        if self.network is None or not self.settings.network:
            return {"lines": [], "last": after, "more": False}
        return await self.network.read_async(after)

    @extension(BROWSER, op="sites/set")
    async def browser_sites_set(self, sites: list[dict]) -> dict:
        """The env's own sites, each by its name and domains, which it keeps for each launch after: a start tab on one
        is waited on strictly."""
        named = _parsed(SiteNames, sites, "sites")
        SITES.parent.mkdir(parents=True, exist_ok=True)
        written = SITES.with_name(f"{SITES.name}.part")
        written.write_text(json.dumps([site.model_dump() for site in named]), encoding="utf-8")
        written.replace(SITES)
        return {}

    @extension(BROWSER, op="calls/begin")
    async def browser_calls_begin(self, user: str, request: Request, acts: bool = True, within_s: float = 600.0) -> dict:
        """An agent's tool call, from the tool-call header this is sent with, which a driver serves through ``user``'s
        CDP connections, begun as an entry in the log, by the caller's role: ``user``'s commands are credited to it
        until ``calls/end`` ends it, or ``within_s`` passes and it's written as never said to have ended. Its entry's
        sequence number, ``entry``."""
        call = read_tool_call(request.headers.get(TOOL_CALL_HEADER))
        if call is None:
            raise InvalidArgument(f"calls/begin is sent with the tool call it begins, in its {TOOL_CALL_HEADER} header")
        who = normalize_role(request.headers.get(ROLE_HEADER)) or DEFAULT_ROLE
        return {"entry": self.relay.begin_call(user, who, call, acts=acts, within_s=within_s)}

    @extension(BROWSER, op="calls/end")
    async def browser_calls_end(self, entry: int, ok: bool, error: Optional[str] = None) -> dict:
        """The tool call ``calls/begin`` began as ``entry`` ended, with whether it worked and, if not, why: its entry
        written, with the tab its commands acted on."""
        if not self.relay.end_call(entry, ok, error):
            raise InvalidArgument(f"no tool call is open as entry {entry}")
        return {}

    @extension(BROWSER_RECORDING, op="start")
    async def browser_recording_start(self, request_id: str, upload: dict, frames: bool = True) -> dict:
        """A recording of this browser, its frames and its log uploaded under the write namespace grant ``upload``:
        with ``frames``, a frame of the tab an agent last acted in at the start, after each burst of its actions
        settles and at the end. A start again with the same ``request_id`` answers the recording it began."""
        return await self.recordings.start(request_id, upload, frames)

    @extension(BROWSER_RECORDING, op="end")
    async def browser_recording_end(self, recording_id: str, upload: Optional[dict] = None,
                                    reason: str = "end") -> dict:
        """The recording ``recording_id`` ended, under a fresh grant ``upload`` when one is given: its last frames,
        its log's lines since its start and its manifest uploaded, and the manifest answered."""
        return await self.recordings.end(recording_id, upload, reason)

    @extension(BROWSER_RECORDING, op="end_all")
    async def browser_recording_end_all(self, reason: str = "teardown") -> dict:
        """Every open recording ended, under its start's grant, as the env closes or resets: their manifests."""
        return {"recordings": await self.recordings.end_all(reason)}

    @extension(BROWSER, op="cdp")
    async def browser_cdp(self, user: str, request: Request, origin: Optional[str] = None,
                          context: Optional[str] = None) -> dict:
        """``user``'s CDP address once the browser's ready, the same one each time it's asked for, as the server that
        minted it keeps it: under ``origin``, where the caller's agent reaches the env, when the caller names it, as a
        deploy names the env's other addresses, since a tunnel in front of the env may not say how it was reached;
        else at the address this call reached the browser at. Scoped to the lease ``context`` when it names one,
        which must be granted to ``user``, and which the address ends with."""
        base = under(_origin(origin), request) if origin is not None else reached(request)
        await self._until_ready()
        if context is not None:
            if (lease := self.contexts.get(context)) is None:
                raise TargetUnavailable(f"no context {context!r} is leased; contexts/list names the leased ones")
            if user not in lease.grants:
                raise PermissionDenied(f"the context {context!r} isn't granted to {user!r}: contexts/grant grants it")
        address = self.cdp_tokens.mint(user, base, named=origin is not None, context=context)
        return {"http_url": address.http_url, "ws_url": address.ws_url}

    def _lease_ended(self, lease: Any) -> None:
        """A lease ended: the addresses scoped to it end with it, and their connections are closed."""
        self.cdp_tokens.revoke(lease.label)
        self.relay.lease_ended(lease.id)

    def served_asgi(self, asgi: Any) -> Any:
        return _LoggedOps(super().served_asgi(asgi), self)

    def create_app(self) -> Any:
        app = super().create_app()
        for path in ("", "/", "/version", "/version/", "/list", "/list/"):
            app.custom_route(f"{ROUTE}{{token}}/json{path}", methods=["GET"])(self._cdp_discovery)
        for path in ("", "/{browser}"):
            add_websocket_route(app, f"{ROUTE}{{token}}/devtools/browser{path}", self._cdp_connect)
        return app

    async def _cdp_discovery(self, request: Request) -> Response:
        """``json/version`` and ``json/list`` under a CDP address, answered from the browser, with its WebSocket as
        minted when its caller named the origin, else at the address the client reached: a browser on a pipe has no
        endpoints of its own. 403 for a token agentenv didn't mint."""
        token = request.path_params["token"]
        if (address := self.cdp_tokens.address(token)) is None:
            return JSONResponse({"error": "agentenv didn't mint this CDP address"}, status_code=403)
        try:
            version = await self.devtools.call("Browser.getVersion")
            if request.url.path.rstrip("/").endswith("/json/version"):
                return JSONResponse({
                    "Browser": version.get("product"), "Protocol-Version": version.get("protocolVersion"),
                    "User-Agent": version.get("userAgent"), "V8-Version": version.get("jsVersion"),
                    "WebKit-Version": version.get("revision"),
                    "webSocketDebuggerUrl": address.ws_url if address.named else
                    f"{websocket_form(reached(request))}{ROUTE}{token}/devtools/browser"})
            tabs = [tab for tab in await self.control.tab_list(every=True)
                    if self._sees(address, self.control.tabs.get(tab["id"], {}).get("context_id"))]
        except (ConnectionError, DevToolsError, asyncio.TimeoutError) as e:
            return JSONResponse({"error": f"the browser isn't answering: {e}"}, status_code=503)
        return JSONResponse([{"id": tab["id"], "type": "page", "title": tab["title"], "url": tab["url"],
                              "description": "", "devtoolsFrontendUrl": ""} for tab in tabs])

    async def _cdp_connect(self, websocket: WebSocket) -> None:
        address = self.cdp_tokens.address(websocket.path_params["token"])
        if address is None:
            await websocket.close(code=1008)
            return
        await self.relay.serve(websocket, address.user, address.context)

    def _sees(self, address: Any, context_id: Optional[str]) -> bool:
        """Whether an address sees a tab in the browser context ``context_id``, as a connection to it does."""
        lease = self.contexts.get(address.context) if address.context is not None else None
        if address.context is not None and lease is None:
            return False
        return Scope(self.contexts, self.relay.target_contexts, address.user, lease.id if lease else None).sees(
            context_id)

    async def _until_ready(self) -> None:
        """Back once it's ready, as ``computer-app/v1``'s ``ready`` says, or DeviceUnavailable after its ready
        timeout."""
        timeout_s = self._computer_app.timeout_s
        deadline = asyncio.get_running_loop().time() + timeout_s
        while not (await self._computer_app.readiness()).ready:
            if asyncio.get_running_loop().time() > deadline:
                raise DeviceUnavailable(f"the browser isn't ready after {timeout_s:g} s")
            await asyncio.sleep(READY_POLL_S)

    @extension(BROWSER, op="page/viewport", method="GET")
    async def browser_page_viewport(self) -> dict:
        window = self.settings.window
        return {"width": window.width, "height": window.height}

    @extension(BROWSER, op="page/capture", method="GET")
    async def browser_page_capture(self) -> dict:
        return await self.control.capture_page()

    @extension(BROWSER, op="page/input")
    async def browser_page_input(self, events: list[dict]) -> dict:
        await self.control.dispatch(events)
        return {}

    @extension(BROWSER, op="page/state", method="GET")
    async def browser_page_state(self) -> dict:
        return await self.control.state()

    @extension(BROWSER, op="navigate")
    async def browser_navigate(self, url: str) -> dict:
        await self.control.navigate(url)
        return {}

    @extension(BROWSER, op="history")
    async def browser_history(self, direction: str) -> dict:
        await self.control.history(direction)
        return {}

    @extension(BROWSER, op="contexts/lease")
    async def browser_contexts_lease(self, label: str) -> dict:
        """The context leased as ``label``: the one leased under it before, as a retried step finds it, else a new
        one; its label and the browser's id for it."""
        lease = await self.contexts.lease(label)
        return {"context": lease.label, "id": lease.id}

    @extension(BROWSER, op="contexts/grant")
    async def browser_contexts_grant(self, context: str, user: str) -> dict:
        """The lease ``context`` granted to the agent ``user`` too, recorded, not enforced; who it's granted to."""
        lease = self.contexts.grant(context, user)
        return {"context": lease.label, "grants": list(lease.grants)}

    @extension(BROWSER, op="contexts/release")
    async def browser_contexts_release(self, context: str) -> dict:
        """The lease ``context`` ended, its tabs closed and its folder deleted; whether there was one."""
        return {"released": await self.contexts.release(context)}

    @extension(BROWSER, op="contexts/list", method="GET")
    async def browser_contexts_list(self) -> dict:
        """Every lease, a context made where agentenv didn't see leased to nobody first."""
        await self.contexts.find_unseen()
        return {"contexts": [lease.record() for lease in self.contexts]}

    @extension(BROWSER, op="tabs/list", method="GET")
    async def browser_tabs_list(self, context: Optional[str] = None, every: bool = False) -> dict:
        """The tabs of the lease ``context`` names, else of the default context; or with ``every`` all of them, each
        naming its context, the page's tab the active one."""
        if every and context is not None:
            raise InvalidArgument("tabs/list takes a context or every, not both")
        return {"tabs": await self.control.tab_list(self.contexts.id_of(context), every=every)}

    @extension(BROWSER, op="tabs/open")
    async def browser_tabs_open(self, url: Optional[str] = None, background: bool = False,
                                context: Optional[str] = None) -> dict:
        return {"tab": await self.control.open_tab(url, background, self.contexts.id_of(context))}

    @extension(BROWSER, op="tabs/activate")
    async def browser_tabs_activate(self, tab: str, context: Optional[str] = None) -> dict:
        await self.control.activate(tab, self.contexts.id_of(context))
        return {}

    @extension(BROWSER, op="tabs/close")
    async def browser_tabs_close(self, tab: str, context: Optional[str] = None) -> dict:
        await self.control.close(tab, self.contexts.id_of(context))
        return {}

    @extension(BROWSER, op="cookies/get")
    async def browser_cookies_get(self, urls: Optional[list[str]] = None, context: Optional[str] = None) -> dict:
        return {"cookies": await self.control.cookies(urls, self.contexts.id_of(context))}

    @extension(BROWSER, op="cookies/set")
    async def browser_cookies_set(self, cookies: list[dict], context: Optional[str] = None) -> dict:
        await self.control.set_cookies(_parsed(CookieToSet, cookies, "cookies"), self.contexts.id_of(context))
        return {}

    @extension(BROWSER, op="cookies/delete")
    async def browser_cookies_delete(self, name: Optional[str] = None, domain: Optional[str] = None,
                                     path: Optional[str] = None, context: Optional[str] = None) -> dict:
        return {"deleted": await self.control.delete_cookies(name, domain, path, self.contexts.id_of(context))}

    @extension(BROWSER, op="local_storage/get")
    async def browser_local_storage_get(self, origins: Optional[list[str]] = None,
                                        context: Optional[str] = None) -> dict:
        return {"origins": await self.control.local_storage(origins, self.contexts.id_of(context))}

    @extension(BROWSER, op="local_storage/set")
    async def browser_local_storage_set(self, origins: list[dict], context: Optional[str] = None) -> dict:
        await self.control.set_local_storage(_parsed(OriginStorage, origins, "origins"), self.contexts.id_of(context))
        return {}

    @extension(BROWSER, op="local_storage/clear")
    async def browser_local_storage_clear(self, origins: Optional[list[str]] = None,
                                          context: Optional[str] = None) -> dict:
        return {"origins": await self.control.clear_local_storage(origins, self.contexts.id_of(context))}

    @extension(BROWSER, op="downloads/list", method="GET")
    async def browser_downloads_list(self, context: Optional[str] = None) -> dict:
        return {"downloads": self.control.download_list(self.contexts.id_of(context))}

    @extension(BROWSER, op="downloads/get")
    async def browser_downloads_get(self, guid: str, write_namespace: Optional[dict] = None,
                                    context: Optional[str] = None) -> dict:
        """The download ``guid`` of the context ``context``, else of the default one, and, completed, its file:
        uploaded under ``write_namespace``, a grant, when given, else inline."""
        download, grant = self.control.download(guid, self.contexts.id_of(context)), _grant(write_namespace)
        home = Path.home().resolve()
        landed = Path(download["path"]).resolve() if download["state"] == "completed" and download["path"] else None
        parts = await file_parts(home, [landed.relative_to(home).as_posix()], grant) \
            if landed is not None and landed.is_relative_to(home) else []
        return {"download": download, "parts": [part.model_dump(mode="json", exclude_none=True) for part in parts]}

    @extension(BROWSER, op="dialog/answer")
    async def browser_dialog_answer(self, accept: bool, text: Optional[str] = None) -> dict:
        await self.control.answer_dialog(accept, text)
        return {}

    @extension(BROWSER, op="files/choose")
    async def browser_files_choose(self, paths: list[str]) -> dict:
        await self.control.choose_files(paths)
        return {}

    def _resets_failed_add(self) -> bool:
        """A load that fails is followed by a reset only before an agent has acted: after, the reset would end the
        agent's session over it."""
        return not agent_acted()

    def declared_extensions(self) -> list[EnvironmentExtension]:
        """Its data plane moves files by their path, a file tree: its state as ``browser-state.json``, its log, its
        network log as HAR, its profile, its downloads and files for agents to upload. A load takes its state as a data
        part too."""
        tree = IntakeDeclaration(get=[IntakeFormat(part="file", format="file-tree")],
                                 add=[IntakeFormat(part="file", format="file-tree", load="replace"),
                                      IntakeFormat(part="data", format="json", load="replace")])
        return [*super().declared_extensions(), intake_extension(tree, (
            f"data/get returns the browser's state as {data.STATE_FILE}: its cookies and local storage as Playwright's "
            f"storageState has them, its tabs open and closed, its downloads, its clock and, under contexts, each "
            f"lease's by its label; beside it, its log as {data.LOG_FILE} and, when a proxy logs for it and its "
            f"network setting is on, its network log as HAR 1.2 in {HAR_FILE}, an entry for each request and tunnel "
            f"in the order they started, both of which data/add passes over, its profile without caches, and its "
            f"finished downloads, a lease's under ~/{CONTEXTS}/<label>/{DOWNLOADS}. data/add applies such a state, "
            f"while the browser runs, leasing each context it names that isn't leased, and writes files by their "
            f"path: its profile, restored with the browser stopped, ~/{DOWNLOADS}, ~/{UPLOADS} and a lease's "
            f"downloads. data/reset ends every lease, puts the profile back as installed and relaunches it at its "
            f"start tabs."))]

    @get_data
    async def browser_get_data(self, write_namespace: Optional[WriteNamespaceGrant] = None) -> list:
        """Its state, as ``browser-state.json``, its log, its network log as HAR when it has one and its ``network``
        setting is on, its profile without caches, copied
        with the browser paused so its files are of one moment, and its finished downloads: each a file uploaded under
        ``write_namespace`` when there is one, else inline."""
        home = Path.home()
        await self.contexts.find_unseen()
        leases = list(self.contexts)
        state = BrowserState(cookies=await self.control.cookies(), local_storage=await self.control.local_storage(),
                             tabs=await self.control.tab_list(),
                             closed_tabs=(await self.control.record())["closed_tabs"],
                             downloads=self.control.download_list(), clock={"mode": "real", "now": _now()},
                             start_tabs=self.start_tab_launches[-1]["tabs"] if self.start_tab_launches else [],
                             contexts={lease.label: ContextState(
                                 made_by=lease.made_by, grants=lease.grants,
                                 cookies=await self.control.cookies(context=lease.id),
                                 local_storage=await self.control.local_storage(context=lease.id),
                                 tabs=await self.control.tab_list(lease.id),
                                 downloads=self.control.download_list(lease.id)) for lease in leases})
        written = {**state.model_dump(), "cookies": _written(state.cookies),
                   "contexts": {label: {**context.model_dump(), "cookies": _written(context.cookies)}
                                for label, context in state.contexts.items()}}
        (home / ".cache").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=home / ".cache") as staging:
            root = Path(staging)
            (root / data.STATE_FILE).write_text(json.dumps(written, indent=1), encoding="utf-8")
            (root / data.LOG_FILE).write_text(await self.log.complete_text(), encoding="utf-8")
            exported = [data.STATE_FILE, data.LOG_FILE]
            if self.network is not None and self.settings.network:
                (root / HAR_FILE).write_text(await self.network.har_text(), encoding="utf-8")
                exported.append(HAR_FILE)
            with self._computer_app.paused():
                await app_data.in_thread(data.stage_profile, home, self.profile, root)
            parts = await file_parts(root, [*exported, self.profile], write_namespace)
        downloads = [*data.finished_downloads(home),
                     *(path for lease in leases for path in data.finished_downloads(home, lease.under()))]
        return parts + await file_parts(home, downloads, write_namespace)

    @add_data
    async def browser_add_data(self, parts: list) -> None:
        """Its state, its profile and files, from ``parts``, while it runs: a profile restored with the browser
        stopped and launched again, then the files, then the cookies, the local storage and the tabs. A load that
        changes cookies or storage and names no tabs opens the start tabs again, so they show it; and once an agent has
        acted, a load can't change the profile or the tabs, which would end what it's doing."""
        load = await data.sorted_load(parts, self.profile)
        try:
            state = StateToLoad.model_validate(load.state or {})
        except ValidationError as e:
            raise InvalidArgument(f"{data.STATE_FILE}: " + "; ".join(
                f"{'.'.join(map(str, error['loc'])) or 'it'}: {error['msg']}"
                for error in e.errors(include_input=False, include_url=False)))
        if agent_acted() and (load.profile or state.tabs is not None or
                              any(context.tabs is not None for context in state.contexts.values())):
            raise InvalidArgument("an agent has acted in this computer, so a load can't change the browser's "
                                  "profile or tabs, which would end what it's doing; only its cookies, local storage "
                                  "and files")
        _check_contexts(state, load, set() if load.profile else {lease.label for lease in self.contexts})
        applied: list[str] = []
        try:
            if load.profile:
                await self._computer_app.stop_app()
                await app_data.in_thread(data.restore_profile, Path.home(), self.profile, load.profile)
                await self._computer_app.launch_app(METHOD_ADD)
                _refuse_failed(await self._start_tabs_settled())
                await self._until_ready()
                applied.append("the profile")
            if load.files:
                await app_data.in_thread(app_data.write_within, Path.home(), load.files, [DOWNLOADS, UPLOADS])
                applied.append("the files")
            if state.cookies:
                await self.control.set_cookies(state.cookies)
                applied.append("the cookies")
            if state.local_storage:
                await self.control.set_local_storage(state.local_storage)
                applied.append("the local storage")
            if state.tabs is not None:
                await self.control.replace_tabs(state.tabs)
                applied.append("the tabs")
            elif (state.cookies or state.local_storage) and not agent_acted():
                _refuse_failed(await self._load_start_tabs(self.settings.start_urls, _now()))
                applied.append("the start tabs again")
            for label in [*state.contexts, *(label for label in load.context_files if label not in state.contexts)]:
                await self._load_context(label, state.contexts.get(label), load.context_files.get(label, {}))
                applied.append(f"the context {label!r}")
        except Exception as e:
            done = ", ".join(applied) or "nothing"
            raise type(e)(f"{e} (the load had applied {done})") if isinstance(e, (ValueError, InvalidArgument)) \
                else RuntimeError(f"{type(e).__name__}: {e} (the load had applied {done})") from e

    async def _load_context(self, label: str, loaded: Optional[ContextToLoad], files: dict[str, bytes]) -> None:
        """The lease ``label``, leased as ``loaded`` says when it isn't, its downloads ``files`` written, then its
        cookies, local storage and tabs loaded."""
        lease = self.contexts.get(label) if loaded is None else \
            await self.contexts.lease(label, made_by=loaded.made_by, grants=tuple(loaded.grants))
        if files:
            await app_data.in_thread(app_data.write_within, Path.home(), files, [lease.under()])
        if loaded is None:
            return
        if loaded.cookies:
            await self.control.set_cookies(loaded.cookies, lease.id)
        if loaded.local_storage:
            await self.control.set_local_storage(loaded.local_storage, lease.id)
        if loaded.tabs is not None:
            await self.control.replace_tabs(loaded.tabs, lease.id)

    @reset_data
    async def browser_reset_data(self) -> None:
        """With the browser stopped: its addresses ended, every lease with it, its profile as installed, its
        downloads and uploads gone and a record that starts now. Its data plane launches it again after, at its start
        tabs."""
        self.relay.ended()
        self.contexts.ended()
        self.cdp_tokens.revoke_all()
        self.log.note("reset", {})
        await app_data.reset(self.paths)
        await app_data.in_thread(data.empty, Path.home(), DOWNLOADS, UPLOADS)
        self.control.forget()
        self.start_tab_launches = []
        forget_agent_acted()

    async def _relaunched(self, op: str) -> None:
        """After a reset relaunches it: back once it's ready, InvalidArgument naming a start tab on one of the env's
        sites that failed."""
        _refuse_failed(await self._start_tabs_settled())
        await self._until_ready()

    async def _start_tabs_settled(self) -> list[dict]:
        """The start tabs of its latest launch once they've settled, waited on its ready timeout and the margin its
        setup takes before they load; DeviceUnavailable if they haven't by then."""
        if (following := self._following) is not None:
            timeout_s = self._computer_app.timeout_s + START_TABS_MARGIN_S
            try:
                await asyncio.wait_for(asyncio.shield(following), timeout_s)
            except asyncio.TimeoutError:
                raise DeviceUnavailable(f"the browser's start tabs hadn't settled after {timeout_s:g} s") from None
        return self.start_tab_launches[-1]["tabs"] if self.start_tab_launches else []

    def extension_params(self, uri: str) -> dict:
        if uri == BROWSER_RECORDING:
            return {"max_duration_s": MAX_DURATION_S, "max_frames": MAX_FRAMES, "frames": True}
        if uri == BROWSER:
            params: dict[str, Any] = {"chromium": self.version, "clock": "real", "mode": _mode()}
            if _alone():
                params["window"] = self.settings.window.model_dump()
            return params
        return super().extension_params(uri)

    @agent_skill
    def browser_skill(self) -> tuple[str, str]:
        return "How to use this website browser.", SKILL


class _LoggedOps:
    """Its server's application, with each ``browser/v1`` op an entry in its log as it runs: every op a driver makes
    for an agent's tool call, which carries that call, and every acting op agentenv makes itself. While it runs, the op
    is the action the browser's control credits the tab events it causes to."""

    def __init__(self, app: Any, browser: WebsiteBrowserEnvironment) -> None:
        self.app, self.browser = app, browser

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        op = scope["path"][len(OPS_PATH):] if scope["type"] == "http" and scope["path"].startswith(OPS_PATH) else None
        if op is None or op in UNLOGGED_OPS:
            return await self.app(scope, receive, send)
        headers = {name.decode("latin-1").lower(): value.decode("latin-1") for name, value in scope.get("headers", [])}
        call, acting = read_tool_call(headers.get(TOOL_CALL_HEADER.lower())), op_acts(op)
        if call is None and not acting:
            return await self.app(scope, receive, send)
        chunks, more = [], True
        while more:
            message = await receive()
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False) and message["type"] == "http.request"
        body = b"".join(chunks)
        action: dict[str, Any] = {"via": "mcp" if call else "agentenv", "op": op, "acts": acting}
        if call is not None:
            action.update(who=normalize_role(headers.get(ROLE_HEADER.lower())) or DEFAULT_ROLE, tool=call["tool"],
                          call=call["call"])
        log = self.browser.log
        begun = log.begin(action)
        actor = {"seq": begun["seq"], "via": action["via"], "who": action.get("who"), "acts": acting, "tab": None,
                 "password": None}
        answered: dict[str, Any] = {"status": 500, "error": b""}
        replayed = False

        async def receiving() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        async def sending(message: dict) -> None:
            if message["type"] == "http.response.start":
                answered["status"] = message["status"]
            elif message["type"] == "http.response.body" and answered["status"] >= 400 and \
                    len(answered["error"]) < ERROR_CHARS * 4:
                answered["error"] += message.get("body", b"")
            await send(message)

        token = ACTOR.set(actor)
        try:
            await self.app(scope, receiving, sending)
        finally:
            ACTOR.reset(token)
            fields: dict[str, Any] = {"tab": actor["tab"], "url": self.browser.control.url_of(actor["tab"]),
                                      "ok": answered["status"] < 400}
            if answered["status"] >= 400:
                fields["error"] = _error_text(answered["error"])
            # What it typed is kept only once every check it made found no password field focused; one that failed,
            # or never got to check, keeps it withheld.
            withheld_typing = actor["password"] is not False or answered["status"] >= 400
            if acting and len(body) > LOGGED_BODY_BYTES:
                fields["params"] = {"too_long": True, "length": len(body)}
            elif acting:
                fields["params"] = redacted_op(op, _op_params(scope, body), password=withheld_typing)
                if call is not None:
                    fields["args"] = redacted_args(call["tool"], call["args"], password=withheld_typing)
            log.finish(begun, **fields)


def _op_params(scope: dict, body: bytes) -> dict:
    """An op's parameters: its query's for a GET, else its JSON body's."""
    if scope.get("method") == "GET":
        return dict(parse_qsl(scope.get("query_string", b"").decode("latin-1")))
    try:
        params = json.loads(body) if body else {}
    except ValueError:
        return {}
    return params if isinstance(params, dict) else {}


def _error_text(body: bytes) -> str:
    """What an op's error answer says, its message when it's agentenv's error shape."""
    text = body.decode(errors="replace")
    try:
        error = json.loads(text).get("error")
        text = str(error.get("message", error)) if isinstance(error, dict) else str(error or text)
    except (ValueError, AttributeError):
        pass
    return redact_text(text)[:ERROR_CHARS]


def _read_sites() -> list[SiteNames]:
    """The env's sites ``sites/set`` handed it, none before it did or for a file it can't read."""
    try:
        return [SiteNames.model_validate(site) for site in json.loads(SITES.read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError, ValidationError):
        return []


def _site_of(url: str, sites: list[SiteNames]) -> Optional[str]:
    """The name of the env's site among ``sites`` that serves ``url``'s host, None for any other."""
    host = (urlsplit(url).hostname or "").rstrip(".")
    return next((site.name for site in sites if any(names_host(domain.lower(), host) for domain in site.domains)), None)


def _refuse_failed(tabs: list[dict]) -> None:
    """InvalidArgument naming each start tab on one of the env's sites that didn't load."""
    if failed := [tab for tab in tabs if tab["site"] is not None and tab["outcome"] != LOADED]:
        raise InvalidArgument("; ".join(f"start tab {tab['url']} on the env's site {tab['site']} {start_tab_why(tab)}"
                                        for tab in failed))


def start_tab_why(tab: dict) -> str:
    """Why a start tab didn't load, as its outcome says."""
    if tab.get("error"):
        return f"didn't load: {tab['error']}"
    if tab.get("status") is not None and tab["status"] >= 400:
        return f"answered {tab['status']}"
    return f"was still loading after {tab['load_ms'] / 1000:g} s"


def _alone() -> bool:
    """Whether it runs on a computer with no display, a browser computer's, alone: its page is that computer's screen."""
    return computer_app.detect().display == "none"


def _x_server() -> bool:
    """Whether an X server is there for it, as a browser computer gives its browser one no agent sees."""
    return bool(os.environ.get("DISPLAY"))


def _mode() -> str:
    """How it runs: headless alone with no X server, else headful, on its own X server alone or on the desktop."""
    return "headless" if _alone() and not _x_server() else "headful"


def _origin(value: Any) -> str:
    """``value`` as an origin a CDP address is minted under, its scheme in lower case as the address's WebSocket form
    reads it: an http or https URL, a path allowed, with no query, fragment or credentials; InvalidArgument for anything
    else."""
    try:
        parts = urlsplit(value)
        valid = (value.isprintable() and parts.scheme in ("http", "https") and bool(parts.hostname)
                 and "@" not in parts.netloc and not parts.query and not parts.fragment
                 and (parts.port is None or 0 < parts.port))
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise InvalidArgument("origin is an http or https URL with no query, fragment or credentials")
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def _written(cookies: list) -> list[dict]:
    """``cookies`` as a snapshot keeps them, without the fields the browser left unset."""
    return [cookie.model_dump(exclude_none=True) for cookie in cookies]


def _check_contexts(state: StateToLoad, load: data.Load, leased: set[str]) -> None:
    """InvalidArgument, before a load applies anything, for a context it would lease by a label no lease can have, or
    downloads it writes for a context it neither leases nor finds leased: ``leased``, none once a profile it restores
    relaunches the browser."""
    for label, context in state.contexts.items():
        if label not in leased and (problem := label_problem(label, by_task=context.made_by == TASK)):
            raise InvalidArgument(f"{data.STATE_FILE}: contexts: {problem}")
    if orphans := sorted(set(load.context_files) - set(state.contexts) - leased):
        raise InvalidArgument(f"the load writes downloads for contexts it doesn't lease and that aren't leased: "
                              f"{', '.join(orphans)}")

def _parsed(model: type[BaseModel], items: Any, what: str) -> list:
    """Each of ``items`` as ``model``; InvalidArgument naming the fields that don't fit and never their values, which
    for a cookie are its credentials."""
    if not isinstance(items, list):
        raise InvalidArgument(f"{what} is a list")
    parsed = []
    for index, item in enumerate(items):
        try:
            parsed.append(model.model_validate(item))
        except ValidationError as e:
            reasons = "; ".join(f"{'.'.join(map(str, error['loc'])) or 'it'}: {error['msg']}"
                                for error in e.errors(include_input=False, include_url=False))
            raise InvalidArgument(f"{what}[{index}]: {reasons}")
    return parsed


def _grant(write_namespace: Optional[dict]) -> Optional[WriteNamespaceGrant]:
    """``write_namespace`` as a grant; InvalidArgument naming its fields only, since its values carry a signature."""
    if write_namespace is None:
        return None
    try:
        return WriteNamespaceGrant.model_validate(write_namespace)
    except ValidationError as e:
        fields = ", ".join(".".join(map(str, error["loc"])) or "write_namespace" for error in e.errors())
        raise InvalidArgument(f"write_namespace is not a valid namespace grant ({fields})")


def _shm_bytes(path: str = "/dev/shm") -> int:
    try:
        stats = os.statvfs(path)
    except OSError:
        return 0
    return stats.f_frsize * stats.f_blocks


def _refuse_owned(name: str, launch: Launch) -> None:
    if owned := [arg for arg in launch.command if arg.startswith(OWNED_FLAGS)]:
        raise ValueError(f"{name}'s launch sets {', '.join(owned)}, which agentenv sets for every website browser")
