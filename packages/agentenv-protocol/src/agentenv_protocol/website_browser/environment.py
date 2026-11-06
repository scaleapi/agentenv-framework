"""WebsiteBrowserEnvironment, its settings, and ``browser/v1`` on its card."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable, ClassVar, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.websockets import WebSocket

from ..agent_env_environment import add_websocket_route, agent_skill, extension
from ..browser_cdp import ROUTE, Relay, Tokens, reached, websocket_form
from ..computer import AppDeclaration, ComputerAppEnvironment, Launch, install_step, ready_check
from ..computer import app as computer_app
from ..computer.app import SOCKET
from ..computer.dataplane import file_parts
from ..computer.errors import DeviceUnavailable, InvalidArgument
from ..transfers import WriteNamespaceGrant
from .control import AGENT, DOWNLOADS, EXIT, STOP, UPLOADS, BrowserControl
from .devtools import PIPE_IN, PIPE_OUT, DevToolsError, DevToolsPipe
from .shapes import BrowserRecord, CookieToSet, OriginStorage

logger = logging.getLogger(__name__)

SITE_PROXY_ENV = "AGENTENV_SITE_PROXY"
BROWSER = "urn:agentenv:browser/v1"
#: The flags agentenv sets, by prefix, which a subclass's command may not carry: among them every way to open a debugging
#: port, since the browser is reached only over its pipe.
OWNED_FLAGS = ("--user-data-dir", "--remote-debugging-", "--remote-allow-origins", "--proxy-", "--no-proxy-server",
               "--headless", "--window-size", "--lang", "--force-webrtc-ip-handling-policy", "--disk-cache-dir",
               "--disable-dev-shm-usage")
#: Below this much /dev/shm, as Docker's 64 MiB default, the browser keeps its shared memory in files instead.
SHM_MIN_BYTES = 1 << 30
#: Chromium's content setting that denies a permission without a prompt.
BLOCK = 2
#: How often a call that waits for the browser to be ready checks it.
READY_POLL_S = 0.5
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
- Files to upload are in `~/{UPLOADS}`; downloads land in `~/{DOWNLOADS}`.
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

    def __init__(self) -> None:
        #: Its CDP connection, over the pipe its server launches it with.
        self.devtools = DevToolsPipe()
        self._following: Optional[asyncio.Future] = None
        super().__init__()
        for folder in (UPLOADS, DOWNLOADS):
            (Path.home() / folder).mkdir(exist_ok=True)
        #: agentenv's control of it over that connection, and the record it keeps.
        self.control = BrowserControl(self.devtools, self.settings.dialogs,
                                      (self.settings.window.width, self.settings.window.height) if _alone()
                                      else None)
        #: The CDP addresses it has handed out, and the relay agents' connections to them go through.
        self.cdp_tokens = Tokens(SOCKET.parent / "cdp-tokens.json")
        self.relay = Relay(self.devtools, dialogs_answered=self.settings.dialogs != AGENT)
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
        one; in a computer it goes through one when the computer has one, loopback aside."""
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
        if alone:
            flags.append("--headless=new")
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
        """The start tabs its command line opens: headless Chromium takes one, and the rest open over its pipe."""
        return self.settings.start_urls[:1] if _alone() else self.settings.start_urls

    async def _launch_fds(self) -> dict[int, int]:
        """Before each launch, an empty cache and the deploy's prompt answers in its profile; and a new debugging pipe:
        the browser's ends as its fds 3 and 4, this server's ends its CDP connection, which follows the browser and
        then opens the start tabs its command line can't."""
        shutil.rmtree(self.cache_dir(), ignore_errors=True)
        self._write_prompt_answers()
        to_browser_read, to_browser_write = os.pipe()
        from_browser_read, from_browser_write = os.pipe()
        await self.devtools.connect(to_browser_write, from_browser_read)
        self._following = asyncio.ensure_future(self._follow(self.settings.start_urls[len(self._command_line_tabs()):]))
        self._following.add_done_callback(lambda done: done.cancelled() or done.exception())
        return {PIPE_IN: to_browser_read, PIPE_OUT: from_browser_write}

    def _pipe_closed(self) -> None:
        """The browser closed its pipe: its tabs closed by its exit when it ended on its own, else by its server's
        stop."""
        self.control.ended(EXIT if self._computer_app.exited() else STOP)
        self.relay.ended()

    async def _follow(self, urls: list[str]) -> None:
        """Its control attached, then each of ``urls`` in a tab behind the active one, in order."""
        await self.control.attach()
        for url in urls:
            await self.devtools.call("Target.createTarget", {"url": url, "background": True})

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
        await self.devtools.call("Target.createTarget", {"url": "about:blank", "newWindow": True})

    @install_step(scope="system")
    def write_policies(self) -> None:
        """agentenv's unattended-use policies, with the subclass's over them, where its build reads them."""
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        (self.policy_dir / "agentenv.json").write_text(json.dumps({**POLICIES, **self.policies}, indent=1) + "\n",
                                                       encoding="utf-8")

    @ready_check
    async def devtools_answers(self) -> bool:
        """It answers over its pipe, its control follows it and it has opened its start tabs."""
        try:
            await self.devtools.call("Browser.getVersion", timeout_s=self.devtools_timeout_s)
        except (ConnectionError, DevToolsError, asyncio.TimeoutError):
            return False
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
        "chooser open on it files in ~/Uploads, headless; and page/state is the active tab's id, URL and title and "
        "what is open on it. cookies/get, cookies/set and cookies/delete, and local_storage/get, local_storage/set and "
        "local_storage/clear, read and change its default context's cookies and local storage as Playwright's "
        "storageState has them, a cookie's max_age set as an expiry on the browser's clock; downloads/list and "
        "downloads/get are its downloads, get with its file, uploaded under a write_namespace grant or inline. cdp "
        "returns a user's CDP address, http_url to discover from and ws_url to connect to, "
        "with a token agentenv mints in its path: each connection to it gets its own browser-level session behind a "
        "guard, and record lists them as cdp_connections, each with the commands the guard refused or changed."))
    async def browser_record(self) -> dict:
        return BrowserRecord(**await self.control.record(), launches=self._computer_app.launches(),
                             cdp_connections=list(self.relay.connections)).model_dump()

    @extension(BROWSER, op="cdp")
    async def browser_cdp(self, user: str, request: Request) -> dict:
        """``user``'s CDP address at the address this call reached the browser at, once it's ready: the same one each
        time it's asked for, as the server that minted it keeps it."""
        await self._until_ready()
        address = self.cdp_tokens.mint(user, reached(request))
        return {"http_url": address.http_url, "ws_url": address.ws_url}

    def create_app(self) -> Any:
        app = super().create_app()
        for path in ("", "/", "/version", "/version/", "/list", "/list/"):
            app.custom_route(f"{ROUTE}{{token}}/json{path}", methods=["GET"])(self._cdp_discovery)
        for path in ("", "/{browser}"):
            add_websocket_route(app, f"{ROUTE}{{token}}/devtools/browser{path}", self._cdp_connect)
        return app

    async def _cdp_discovery(self, request: Request) -> Response:
        """``json/version`` and ``json/list`` under a CDP address, answered from the browser, with its WebSocket at the
        address the client reached: a browser on a pipe has no endpoints of its own. 403 for a token agentenv didn't
        mint."""
        token = request.path_params["token"]
        if self.cdp_tokens.address(token) is None:
            return JSONResponse({"error": "agentenv didn't mint this CDP address"}, status_code=403)
        try:
            version = await self.devtools.call("Browser.getVersion")
            if request.url.path.rstrip("/").endswith("/json/version"):
                return JSONResponse({
                    "Browser": version.get("product"), "Protocol-Version": version.get("protocolVersion"),
                    "User-Agent": version.get("userAgent"), "V8-Version": version.get("jsVersion"),
                    "WebKit-Version": version.get("revision"),
                    "webSocketDebuggerUrl": f"{websocket_form(reached(request))}{ROUTE}{token}/devtools/browser"})
            tabs = await self.control.tab_list()
        except (ConnectionError, DevToolsError, asyncio.TimeoutError) as e:
            return JSONResponse({"error": f"the browser isn't answering: {e}"}, status_code=503)
        return JSONResponse([{"id": tab["id"], "type": "page", "title": tab["title"], "url": tab["url"],
                              "description": "", "devtoolsFrontendUrl": ""} for tab in tabs])

    async def _cdp_connect(self, websocket: WebSocket) -> None:
        address = self.cdp_tokens.address(websocket.path_params["token"])
        if address is None:
            await websocket.close(code=1008)
            return
        await self.relay.serve(websocket, address.user)

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

    @extension(BROWSER, op="tabs/list", method="GET")
    async def browser_tabs_list(self) -> dict:
        return {"tabs": await self.control.tab_list()}

    @extension(BROWSER, op="tabs/open")
    async def browser_tabs_open(self, url: Optional[str] = None) -> dict:
        return {"tab": await self.control.open_tab(url)}

    @extension(BROWSER, op="tabs/activate")
    async def browser_tabs_activate(self, tab: str) -> dict:
        await self.control.activate(tab)
        return {}

    @extension(BROWSER, op="tabs/close")
    async def browser_tabs_close(self, tab: str) -> dict:
        await self.control.close(tab)
        return {}

    @extension(BROWSER, op="cookies/get")
    async def browser_cookies_get(self, urls: Optional[list[str]] = None) -> dict:
        return {"cookies": await self.control.cookies(urls)}

    @extension(BROWSER, op="cookies/set")
    async def browser_cookies_set(self, cookies: list[dict]) -> dict:
        await self.control.set_cookies(_parsed(CookieToSet, cookies, "cookies"))
        return {}

    @extension(BROWSER, op="cookies/delete")
    async def browser_cookies_delete(self, name: Optional[str] = None, domain: Optional[str] = None,
                                     path: Optional[str] = None) -> dict:
        return {"deleted": await self.control.delete_cookies(name, domain, path)}

    @extension(BROWSER, op="local_storage/get")
    async def browser_local_storage_get(self, origins: Optional[list[str]] = None) -> dict:
        return {"origins": await self.control.local_storage(origins)}

    @extension(BROWSER, op="local_storage/set")
    async def browser_local_storage_set(self, origins: list[dict]) -> dict:
        await self.control.set_local_storage(_parsed(OriginStorage, origins, "origins"))
        return {}

    @extension(BROWSER, op="local_storage/clear")
    async def browser_local_storage_clear(self, origins: Optional[list[str]] = None) -> dict:
        return {"origins": await self.control.clear_local_storage(origins)}

    @extension(BROWSER, op="downloads/list", method="GET")
    async def browser_downloads_list(self) -> dict:
        return {"downloads": self.control.download_list()}

    @extension(BROWSER, op="downloads/get")
    async def browser_downloads_get(self, guid: str, write_namespace: Optional[dict] = None) -> dict:
        """The download ``guid`` and, completed, its file: uploaded under ``write_namespace``, a grant, when given,
        else inline."""
        download, grant = self.control.download(guid), _grant(write_namespace)
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

    def extension_params(self, uri: str) -> dict:
        if uri == BROWSER:
            params: dict[str, Any] = {"chromium": self.version, "clock": "real"}
            if _alone():
                params["window"] = self.settings.window.model_dump()
            return params
        return super().extension_params(uri)

    @agent_skill
    def browser_skill(self) -> tuple[str, str]:
        return "How to use this website browser.", SKILL


def _alone() -> bool:
    """Whether it runs on a computer with no display, a browser computer's, alone and headless."""
    return computer_app.detect().display == "none"


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
