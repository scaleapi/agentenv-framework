"""What a CDP connection sees and acts in, its ``Scope``: a lease or the browser's default context, and every context
its user made. The relay keeps each target's context as Chromium reports it and asks a connection's scope about each
command it sends (``Scope.verdict``), each answer it gets (``Scope.answer``) and each event on its browser session
(``Scope.hears``): it hears only of the targets of the contexts it sees, pages, cross-origin iframes and workers alike;
its commands that act in a context and name none get its lease, and the ones that name a context or a target it
doesn't see are refused. A guardrail that keeps cooperating agents apart; the sandbox stays the boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .contexts import Contexts

#: The commands that act in a browser context, which name it as ``browserContextId``, the default without one.
ACTS_IN_CONTEXT = frozenset({"Target.createTarget", "Storage.getCookies", "Storage.setCookies", "Storage.clearCookies",
                             "Browser.grantPermissions", "Browser.resetPermissions", "Browser.setPermission",
                             "Browser.setDownloadBehavior"})
#: The commands on a target, which name it as ``targetId``.
ON_TARGET = frozenset({"Target.attachToTarget", "Target.activateTarget", "Target.closeTarget", "Target.getTargetInfo",
                       "Target.autoAttachRelated", "Browser.getWindowForTarget"})
#: The domains whose commands on a browser session act across every context, intercepting, tracing or trusting for all
#: of them, which a scoped connection may send only on its own pages' sessions.
BROWSER_WIDE = ("Fetch.", "Tracing.", "Security.")
#: The ``Storage`` commands a scoped connection may send on its browser session: the cookie ones, which get its lease.
COOKIES = frozenset({"Storage.getCookies", "Storage.setCookies", "Storage.clearCookies"})
#: The ``Browser`` commands on a window, which name it as ``windowId``.
ON_WINDOW = frozenset({"Browser.setWindowBounds", "Browser.getWindowBounds", "Browser.setContentsSize"})
#: The ``Target`` events about a target, which carry its ``targetInfo``, and those that carry only its ``targetId``.
WITH_INFO = frozenset({"Target.targetCreated", "Target.targetInfoChanged", "Target.attachedToTarget"})
WITH_ID = frozenset({"Target.targetDestroyed", "Target.targetCrashed"})
#: Chromium's answer to a command for a target it doesn't know, as a connection is answered for one it doesn't see.
NO_TARGET = {"code": -32602, "message": "No target with given id found"}
#: The commands that make and end a browser context, which agentenv records as leases.
MAKES_CONTEXT, ENDS_CONTEXT = "Target.createBrowserContext", "Target.disposeBrowserContext"


class Targets:
    """The browser context of each target Chromium has reported, by its id: kept for the browser's life, as each
    connection hears of a target's end apart from the others."""

    def __init__(self) -> None:
        self.contexts: dict[str, str] = {}

    def saw(self, info: object) -> None:
        if isinstance(info, dict) and isinstance(target := info.get("targetId"), str) and \
                isinstance(context := info.get("browserContextId"), str):
            self.contexts[target] = context

    def context_of(self, target: object) -> Optional[str]:
        return self.contexts.get(target) if isinstance(target, str) else None

    def forget(self) -> None:
        self.contexts.clear()


@dataclass(frozen=True)
class Ruling:
    """What a connection's scope makes of a command: ``params`` as sent, its scope added when it acts in a context and
    names none, else ``refused`` and why, with the ``error`` it's answered with in place of agentenv's own."""

    params: dict
    refused: Optional[str] = None
    error: Optional[dict] = None


@dataclass
class Scope:
    """What one connection of ``user``'s sees and acts in: the lease ``lease`` names, the browser's id for it, else
    the default context, and every context its user made. ``target_contexts`` is each target's context, and
    ``frame_context`` the context of the tab a frame is in, None for one it hasn't seen."""

    contexts: Contexts
    target_contexts: Targets
    user: str
    lease: Optional[str] = None
    frame_context: Callable[[object], Optional[str]] = lambda frame: None
    #: The windows of its tabs, which it learns asking for them, and the downloads begun where it doesn't see.
    windows: set[int] = field(default_factory=set)
    unseen_downloads: set[str] = field(default_factory=set)

    def sees(self, context: Optional[str]) -> bool:
        """Whether it sees the context ``context``: its scope, or one its user made. A target no context is known for
        is the default context's."""
        if context is not None and context == self.lease:
            return True
        if context is not None and (lease := self.contexts.of(context)) is not None and lease.made_by == self.user:
            return True
        return self.lease is None and (context is None or context not in self.contexts.control.other_contexts)

    def verdict(self, method: str, params: dict, *, on_browser_session: bool) -> Ruling:
        """Its verdict on a command, sent on its browser session or on a page's: refused when it would dispose a
        context its user didn't make, open a second browser session, touch a target it doesn't see or, scoped to a
        lease, act on every context or a window not its tabs'; else as sent, with its lease added where a command acts
        in a context and names none, refused where one names a context it doesn't see."""
        if method == ENDS_CONTEXT and (reason := self.contexts.may_dispose(self.user, params.get("browserContextId"))):
            return Ruling(params, reason)
        if method == "Target.attachToBrowserTarget":
            return Ruling(params, "each connection has its browser session; another would see past its scope")
        if on_browser_session and method in ON_TARGET and not self.sees(
                self.target_contexts.context_of(params["targetId"]) if params.get("targetId") else self.lease):
            return Ruling(params, "the target isn't one this address sees", NO_TARGET)
        if on_browser_session and self.lease is not None:
            if method.startswith(BROWSER_WIDE) or (method.startswith("Storage.") and method not in COOKIES):
                return Ruling(params, "on an address scoped to a lease it would act on every context; a page's own "
                                       "session takes it")
            if method in ON_WINDOW and params.get("windowId") not in self.windows:
                return Ruling(params, "the window isn't one of this address's tabs'")
        if method not in ACTS_IN_CONTEXT:
            return Ruling(params)
        named = params.get("browserContextId")
        if named is None:
            return Ruling({**params, "browserContextId": self.lease} if self.lease is not None else params)
        if self.sees(named):
            return Ruling(params)
        return Ruling(params, f"the browser context {named} isn't one this address sees")

    def answer(self, method: str, result: dict) -> dict:
        """A command's ``result`` as it hears it: the targets of contexts it doesn't see left out, the contexts listed
        beside the default only its user's own, the lease shown as the default; the window a tab of its is in kept."""
        if method == "Browser.getWindowForTarget" and isinstance(window := result.get("windowId"), int):
            self.windows.add(window)
        if method == "Target.getTargets" and isinstance(infos := result.get("targetInfos"), list):
            for info in infos:
                self.target_contexts.saw(info)
            return {**result, "targetInfos": [info for info in infos if isinstance(info, dict)
                                              and self.sees(info.get("browserContextId"))]}
        if method == "Target.getBrowserContexts" and isinstance(ids := result.get("browserContextIds"), list):
            own = [context for context in ids if context != self.lease and (lease := self.contexts.of(context))
                   is not None and lease.made_by == self.user]
            shown = {**result, "browserContextIds": own}
            if self.lease is not None and "defaultBrowserContextId" in result:
                shown["defaultBrowserContextId"] = self.lease
            return shown
        return result

    def hears(self, method: str, params: dict) -> tuple[bool, Optional[str]]:
        """Whether it hears an event on its browser session, a target's only for a context it sees, a download's only
        for a frame of one; and the session of a target it doesn't see that auto-attached to it, which the relay
        lets go."""
        if method in WITH_INFO or method in WITH_ID:
            target = (params.get("targetInfo") or {}).get("targetId") if method in WITH_INFO else params.get("targetId")
            if self.sees(self.target_contexts.context_of(target)):
                return True, None
            attached = params.get("sessionId") if method == "Target.attachedToTarget" else None
            return False, attached if isinstance(attached, str) else None
        if method == "Browser.downloadWillBegin":
            frame = params.get("frameId")
            if self.sees(self.frame_context(frame) or self.target_contexts.context_of(frame)):
                return True, None
            self.unseen_downloads.add(params.get("guid"))
            return False, None
        if method == "Browser.downloadProgress":
            return params.get("guid") not in self.unseen_downloads, None
        return True, None
