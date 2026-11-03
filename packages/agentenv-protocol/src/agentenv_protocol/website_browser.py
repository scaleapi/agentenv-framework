"""WebsiteBrowserEnvironment: a website browser as a computer app, run inside a computer or alone. A subclass says how
its build installs and starts; this class adds what agentenv owns, the same in both: its flags after the subclass's
command, the unattended-use policies, a CDP connection over the browser's debugging pipe, the settings every browser
takes, a fresh cache and the deploy's prompt answers at every launch, another window for a launch while it runs, its
skill and ``browser/v1`` on its card. Whether it runs alone, on the open web and through which site proxy come from
variables agentenv sets, never from its settings."""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable, ClassVar, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .agent_env_environment import agent_skill
from .computer import AppDeclaration, ComputerAppEnvironment, Launch, install_step, ready_check
from .types import EnvironmentExtension

logger = logging.getLogger(__name__)

ALONE_ENV = "AGENTENV_BROWSER_ALONE"
OPEN_WEB_ENV = "AGENTENV_BROWSER_OPEN_WEB"
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
#: Where an agent's files to upload are, and where downloads land, under the session user's home.
UPLOADS = "Uploads"
DOWNLOADS = "Downloads"
SKILL = f"""# Website browser

- Open sites by their real names, such as `https://slack.com`: the environment serves its sites under those names.
- Files to upload are in `~/{UPLOADS}`; downloads land in `~/{DOWNLOADS}`.
"""
#: The fds a Chromium on ``--remote-debugging-pipe`` reads its CDP messages from and writes its answers to.
PIPE_IN, PIPE_OUT = 3, 4
#: The largest CDP message the pipe takes, such as a screenshot's.
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


class DevToolsError(RuntimeError):
    """A CDP call the browser answered with an error."""


class DevToolsPipe:
    """A CDP connection over a Chromium's debugging pipe, each message JSON ended by a NUL. It connects anew at every
    launch; a call with no browser connected, or whose browser exits before answering, raises ConnectionError."""

    #: How long a call waits for the browser's answer when it names no wait of its own.
    call_timeout_s: ClassVar[float] = 30.0

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self._waiting: dict[int, asyncio.Future] = {}
        self._writer: Optional[asyncio.WriteTransport] = None
        self._reading: Optional[asyncio.Task] = None

    async def connect(self, to_browser: int, from_browser: int) -> None:
        """Take over this end of a new pipe, ending any connection to a browser launched before."""
        self.close()
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=MAX_MESSAGE_BYTES)
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(from_browser, "rb", 0))
        self._writer, _ = await loop.connect_write_pipe(asyncio.Protocol, os.fdopen(to_browser, "wb", 0))
        self._reading = asyncio.ensure_future(self._read(reader))

    async def call(self, method: str, params: Optional[dict] = None, *, session_id: Optional[str] = None,
                   timeout_s: Optional[float] = None) -> dict:
        """``method``'s result, or DevToolsError with the browser's message."""
        if self._writer is None or self._writer.is_closing():
            raise ConnectionError("no browser is connected over its debugging pipe")
        message_id = next(self._ids)
        answer = asyncio.get_running_loop().create_future()
        self._waiting[message_id] = answer
        message = {"id": message_id, "method": method, "params": params or {},
                   **({"sessionId": session_id} if session_id else {})}
        self._writer.write(json.dumps(message).encode() + b"\0")
        try:
            answered = await asyncio.wait_for(answer, self.call_timeout_s if timeout_s is None else timeout_s)
        finally:
            self._waiting.pop(message_id, None)
        if "error" in answered:
            raise DevToolsError(f"{method}: {answered['error'].get('message', answered['error'])}")
        return answered.get("result", {})

    def close(self) -> None:
        if self._reading is not None:
            self._reading.cancel()
        if self._writer is not None:
            self._writer.close()
        self._fail_waiting()
        self._writer = self._reading = None

    async def _read(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                message = json.loads((await reader.readuntil(b"\0"))[:-1])
                answer = self._waiting.get(message.get("id"))
                if answer is not None and not answer.done():
                    answer.set_result(message)
        except (asyncio.IncompleteReadError, ConnectionError):
            if self._writer is not None:
                self._writer.close()
            self._fail_waiting()

    def _fail_waiting(self) -> None:
        for answer in self._waiting.values():
            if not answer.done():
                answer.set_exception(ConnectionError("the browser closed its debugging pipe"))


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

    def __init__(self) -> None:
        #: Its CDP connection, over the pipe its server launches it with.
        self.devtools = DevToolsPipe()
        self._start_tabs: Optional[asyncio.Future] = None
        super().__init__()
        for folder in (UPLOADS, DOWNLOADS):
            (Path.home() / folder).mkdir(exist_ok=True)

    @classmethod
    def declaration(cls) -> AppDeclaration:
        """What the class declares; ValueError for a launch whose command carries a flag agentenv owns."""
        for declared in cls._installs.values():
            if declared.launch is not None:
                _refuse_owned(cls.__name__, declared.launch)
        return super().declaration()

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
        if _headless():
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
        return self.settings.start_urls[:1] if _headless() else self.settings.start_urls

    async def _launch_fds(self) -> dict[int, int]:
        """Before each launch, an empty cache and the deploy's prompt answers in its profile; and a new debugging pipe:
        the browser's ends as its fds 3 and 4, this server's ends its CDP connection, over which the start tabs its
        command line can't open are opened."""
        shutil.rmtree(self.cache_dir(), ignore_errors=True)
        self._write_prompt_answers()
        to_browser_read, to_browser_write = os.pipe()
        from_browser_read, from_browser_write = os.pipe()
        await self.devtools.connect(to_browser_write, from_browser_read)
        rest = self.settings.start_urls[len(self._command_line_tabs()):]
        self._start_tabs = asyncio.ensure_future(self._open_tabs(rest)) if rest else None
        if self._start_tabs is not None:
            self._start_tabs.add_done_callback(lambda done: done.cancelled() or done.exception())
        return {PIPE_IN: to_browser_read, PIPE_OUT: from_browser_write}

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

    async def _open_tabs(self, urls: list[str]) -> None:
        """Each of ``urls`` in a tab behind the active one, in order."""
        for url in urls:
            await self.devtools.call("Target.createTarget", {"url": url, "background": True})

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
        """It answers over its pipe and has opened its start tabs."""
        try:
            await self.devtools.call("Browser.getVersion", timeout_s=self.devtools_timeout_s)
        except (ConnectionError, DevToolsError, asyncio.TimeoutError):
            return False
        tabs = self._start_tabs
        return tabs is None or (tabs.done() and not tabs.cancelled() and tabs.exception() is None)

    def declared_extensions(self) -> list[EnvironmentExtension]:
        params: dict[str, Any] = {"chromium": self.version, "clock": "real"}
        if _alone():
            params["window"] = self.settings.window.model_dump()
        browser = EnvironmentExtension(
            uri=BROWSER, params=params,
            description="A website browser agentenv runs: its build, the clock its pages see and, alone, its window.")
        return [*super().declared_extensions(), browser]

    @agent_skill
    def browser_skill(self) -> tuple[str, str]:
        return "How to use this website browser.", SKILL


def _alone() -> bool:
    return os.environ.get(ALONE_ENV) == "1"


def _open_web() -> bool:
    return os.environ.get(OPEN_WEB_ENV) == "1"


def _shm_bytes(path: str = "/dev/shm") -> int:
    try:
        stats = os.statvfs(path)
    except OSError:
        return 0
    return stats.f_frsize * stats.f_blocks


def _headless() -> bool:
    """Alone it runs headless, but headful on Xvfb for the open web, whose sites block headless browsers."""
    return _alone() and not _open_web()


def _refuse_owned(name: str, launch: Launch) -> None:
    if owned := [arg for arg in launch.command if arg.startswith(OWNED_FLAGS)]:
        raise ValueError(f"{name}'s launch sets {', '.join(owned)}, which agentenv sets for every website browser")
