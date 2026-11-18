"""browser_open_tab, browser_close_tab and browser_activate_tab: a tab opened in an env's website browser, or one open
there, picked by its URL, closed or brought to the front."""

from __future__ import annotations

import contextlib
import logging
from typing import ClassVar, Optional
from urllib.parse import urlsplit, urlunsplit

import idna

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.task_step.task_steps.browser.base import BrowserStep

logger = logging.getLogger(__name__)

_DEFAULT_PORTS = {"http": 80, "https": 443}


class BrowserOpenTabTaskStep(BrowserStep):
    """Open a tab at ``url``, in front, or with ``background`` behind the active tab, which stays active; the new tab's
    id is recorded. ``background`` is sent only when set, so a browser built before it opens tabs in front."""

    type: ClassVar[str] = "browser_open_tab"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, url: str, background: bool = False,
                 browser: Optional[str] = None, depends_on: Optional[list[TaskStepDependency]] = None,
                 fail_task_on_error: bool = True, context: Optional[str] = None):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         context=context)
        self.url, self.background = url, background

    def config_problems(self) -> list[str]:
        problems = super().config_problems() + _url_problems(self.id, self.url)
        if not isinstance(self.background, bool):
            problems.append(f"step {self.id!r}: background must be true or false")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "url": self.url, "background": self.background}

    @classmethod
    def from_dict(cls, data: dict) -> BrowserOpenTabTaskStep:
        return cls(**cls._in_context_from_dict(data), url=data["url"], background=data.get("background", False))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        params = {"url": self.url, **({"background": True} if self.background else {})}
        tab = (await self._call(context, "tabs/open", params))["tab"]
        logger.info(f"{self.id}: opened tab {tab} at {self.url}{' in the background' if self.background else ''} in "
                    f"the browser of env {self.env_id!r}")
        self._record(context, tab=tab)
        return context


class _TabAtUrlStep(BrowserStep):
    """A step on the open tab whose URL is ``url``, as the browser shows it now: the same after both are normalized as
    the browser normalizes a URL. No tab, or several, at that URL fails the step, naming the tabs open."""

    #: The browser/v1 operation called on the tab, and what the step says it did.
    op: ClassVar[str]
    did: ClassVar[str]

    def __init__(self, id: str, version: Optional[int], env_id: str, url: str, browser: Optional[str] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True,
                 context: Optional[str] = None):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         context=context)
        self.url = url

    def config_problems(self) -> list[str]:
        return super().config_problems() + _url_problems(self.id, self.url)

    def to_dict(self) -> dict:
        return {**super().to_dict(), "url": self.url}

    @classmethod
    def from_dict(cls, data: dict) -> _TabAtUrlStep:
        return cls(**cls._in_context_from_dict(data), url=data["url"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        tabs = (await self._call(context, "tabs/list", {}))["tabs"]
        wanted = normalized(self.url)
        matching = [tab for tab in tabs if normalized(tab["url"]) == wanted]
        if len(matching) != 1:
            raise RuntimeError(f"{self.type} {self.id!r}: {len(matching) or 'no'} tabs are at {self.url} in the browser "
                               f"of env {self.env_id!r}; the tabs open are at {[tab['url'] for tab in tabs]}")
        await self._call(context, self.op, {"tab": matching[0]["id"]})
        logger.info(f"{self.id}: {self.did} tab {matching[0]['id']} at {self.url} in the browser of env {self.env_id!r}")
        self._record(context, tab=matching[0]["id"])
        return context


class BrowserCloseTabTaskStep(_TabAtUrlStep):
    """Close the open tab at ``url``."""

    type: ClassVar[str] = "browser_close_tab"
    entity_refs = (EntityRef.env("env_id"),)
    op, did = "tabs/close", "closed"


class BrowserActivateTabTaskStep(_TabAtUrlStep):
    """Bring the open tab at ``url`` to the front, the active tab."""

    type: ClassVar[str] = "browser_activate_tab"
    entity_refs = (EntityRef.env("env_id"),)
    op, did = "tabs/activate", "activated"


def normalized(url: str) -> str:
    """``url`` as the browser shows it: its scheme and host lowercase, a host in another script mapped by UTS 46 without
    transitional mapping and its labels in punycode, as Chromium spells it (``faß`` stays apart from ``fass``, and ``☃``
    is kept though IDNA 2008 refuses it), its default port dropped, and an empty path as ``/``; its query and fragment as
    they are. A URL with no host, such as ``about:blank``, is left as it is."""
    parts = urlsplit(url)
    if not parts.netloc:
        return url
    scheme, host = parts.scheme.lower(), (parts.hostname or "")
    if ":" in host:
        host = f"[{host}]"
    elif not host.isascii():
        with contextlib.suppress(idna.IDNAError, UnicodeError):
            host = ".".join(label if label.isascii() else "xn--" + label.encode("punycode").decode("ascii")
                            for label in idna.uts46_remap(host, std3_rules=False, transitional=False).split("."))
    port = parts.port
    netloc = host if port is None or port == _DEFAULT_PORTS.get(scheme) else f"{host}:{port}"
    if parts.username is not None:
        netloc = f"{parts.username}{':' + parts.password if parts.password is not None else ''}@{netloc}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, parts.fragment))


def _url_problems(step_id: str, url: object) -> list[str]:
    if not isinstance(url, str) or not url:
        return [f"step {step_id!r}: url must be a non-empty string"]
    try:
        parts = urlsplit(url)
        _ = parts.port
    except ValueError as e:
        return [f"step {step_id!r}: url {url!r} isn't a URL: {e}"]
    if not parts.scheme or (parts.scheme.lower() in _DEFAULT_PORTS and not parts.netloc):
        return [f"step {step_id!r}: url {url!r} isn't absolute: give its scheme and host, such as https://shop.test/"]
    return []
