"""WebsiteBrowserEnvironment: a website browser as a computer app, run inside a computer or alone. A subclass says how
its build installs and starts; this class adds what agentenv owns, the same in both: its flags after the subclass's
command, the unattended-use policies, the debugging address, the settings every browser takes, its skill and
``browser/v1`` on its card. Whether it runs alone, on the open web and through which site proxy come from variables
agentenv sets, never from its settings."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, ClassVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field, field_validator

from .agent_env_environment import agent_skill
from .computer import AppDeclaration, ComputerAppEnvironment, Launch, install_step, ready_check
from .types import EnvironmentExtension

ALONE_ENV = "AGENTENV_BROWSER_ALONE"
OPEN_WEB_ENV = "AGENTENV_BROWSER_OPEN_WEB"
SITE_PROXY_ENV = "AGENTENV_SITE_PROXY"
BROWSER = "urn:agentenv:browser/v1"
#: The flags agentenv sets, by prefix, which a subclass's command may not carry. agentenv never sets
#: ``--remote-allow-origins``, so the debugging port refuses a WebSocket that carries a page's Origin.
OWNED_FLAGS = ("--user-data-dir", "--remote-debugging-", "--remote-allow-origins", "--proxy-", "--no-proxy-server",
               "--headless", "--window-size", "--lang", "--force-webrtc-ip-handling-policy")
#: Alone, loopback goes through the proxy too, so no page reaches the debugging port. In a computer, the apps on
#: loopback beside it stay reachable; ``<-loopback>`` removes Chromium's own rule, so ``*.localhost`` names still don't.
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
_CDP_ATTR = "_aee_cdp"


def cdp(fn: Callable) -> Callable:
    """Mark the method that returns the browser's own debugging address, a ``ws://`` URL where it runs;
    ``devtools_address`` calls it each time, so the address is right after a relaunch."""
    setattr(fn, _CDP_ATTR, True)
    return fn


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

    def __init__(self) -> None:
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

    def agentenv_flags(self) -> list[str]:
        """The flags agentenv owns. A browser alone goes out only through its site proxy, so it won't launch without
        one; in a computer it goes through one when the computer has one, loopback aside."""
        settings, alone, proxy = self.settings, _alone(), os.environ.get(SITE_PROXY_ENV)
        flags = [f"--user-data-dir={self.profile_dir()}", "--remote-debugging-port=0", "--no-first-run",
                 "--no-default-browser-check", f"--lang={settings.locale}"]
        if proxy:
            flags += [f"--proxy-server={proxy}", f"--proxy-bypass-list={ALONE_BYPASS if alone else COMPUTER_BYPASS}",
                      "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"]
        elif alone:
            raise ValueError(f"a website browser alone goes out only through its site proxy, and {SITE_PROXY_ENV} "
                             "names none")
        if alone:
            flags.append(f"--window-size={settings.window.width},{settings.window.height}")
            if not _open_web():
                flags.append("--headless=new")
        return flags

    def _launch_for(self, launch: Launch) -> Launch:
        """The subclass's launch with agentenv's flags and the start tabs after its command, and the time zone in its
        environment; alone, with no window to match or wait for, so its ready checks say when it's ready."""
        _refuse_owned(type(self).__name__, launch)
        update: dict[str, Any] = {"command": [*launch.command, *self.agentenv_flags(), *self.settings.start_urls],
                                  "env": {**(launch.env or {}), "TZ": self.settings.time_zone}}
        if _alone():
            update.update(window=None, ready=None)
        return launch.model_copy(update=update)

    def _on_desktop(self) -> bool:
        return not _alone()

    @install_step(scope="system")
    def write_policies(self) -> None:
        """agentenv's unattended-use policies, with the subclass's over them, where its build reads them."""
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        (self.policy_dir / "agentenv.json").write_text(json.dumps({**POLICIES, **self.policies}, indent=1) + "\n",
                                                       encoding="utf-8")

    @cdp
    def devtools(self) -> str:
        """Chromium's own: the port it picked and the browser's path, as it writes them to ``DevToolsActivePort`` in
        its profile."""
        port, path = (self.profile_dir() / "DevToolsActivePort").read_text().split()[:2]
        return f"ws://127.0.0.1:{port}{path}"

    async def devtools_address(self) -> str:
        """What the method marked ``@cdp`` returns now; FileNotFoundError until the browser has written its address."""
        for klass in type(self).__mro__:
            for name, member in vars(klass).items():
                if getattr(member, _CDP_ATTR, False):
                    address = getattr(self, name)()
                    return await address if inspect.isawaitable(address) else address
        raise TypeError(f"{type(self).__name__} marks no method @cdp")

    @ready_check
    async def debugging_address_answers(self) -> bool:
        try:
            address = urlsplit(await self.devtools_address())
        except (FileNotFoundError, ValueError):
            return False
        try:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as http:
                return (await http.get(f"http://{address.netloc}/json/version")).status_code == 200
        except httpx.HTTPError:
            return False

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


def _refuse_owned(name: str, launch: Launch) -> None:
    if owned := [arg for arg in launch.command if arg.startswith(OWNED_FLAGS)]:
        raise ValueError(f"{name}'s launch sets {', '.join(owned)}, which agentenv sets for every website browser")
