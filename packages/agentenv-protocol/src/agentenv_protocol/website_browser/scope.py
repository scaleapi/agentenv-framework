"""What a CDP connection sees and acts in: its scope, a lease or the browser's default context, and every context its
user made. The relay keeps each target's context as Chromium reports it, shows a connection only the targets of the
contexts it sees, pages, cross-origin iframes and workers alike, adds its scope to the commands that act in a context
and name none, and refuses the ones that name a context, or a target, it doesn't see. A guardrail that keeps
cooperating agents apart; the sandbox stays the boundary."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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


def sees(contexts: Contexts, scope: Optional[str], user: str, context: Optional[str]) -> bool:
    """Whether a connection of ``user``'s scoped to the lease ``scope``, the browser's id for it, else to the default
    context, sees the context ``context``: its scope, or one its user made. A target no context is known for is the
    default context's."""
    if context is not None and context == scope:
        return True
    if context is not None and (lease := contexts.of(context)) is not None and lease.made_by == user:
        return True
    return scope is None and (context is None or context not in contexts.control.other_contexts)


def browser_wide(method: str) -> Optional[str]:
    """Why a scoped connection may not send ``method`` on its browser session, which would act on every context, or
    None."""
    if method.startswith(BROWSER_WIDE) or (method.startswith("Storage.") and method not in COOKIES):
        return "on an address scoped to a lease it would act on every context; a page's own session takes it"
    return None


def in_scope(contexts: Contexts, scope: Optional[str], user: str, method: str,
             params: dict) -> tuple[dict, Optional[str]]:
    """``method``'s ``params`` as sent for a connection of ``user``'s scoped to ``scope``: a command that acts in a
    context and names none gets the lease's; one that names a context the connection doesn't see is refused, why."""
    if method not in ACTS_IN_CONTEXT:
        return params, None
    named = params.get("browserContextId")
    if named is None:
        return ({**params, "browserContextId": scope} if scope is not None else params), None
    if sees(contexts, scope, user, named):
        return params, None
    return params, f"the browser context {named} isn't one this address sees"


def answered(contexts: Contexts, targets: Targets, scope: Optional[str], user: str, method: str,
             result: dict) -> dict:
    """``method``'s ``result`` as a connection of ``user``'s scoped to ``scope`` hears it: the targets of contexts it
    doesn't see left out, and the contexts listed beside the default only its user's own, the lease shown as the
    default."""
    if method == "Target.getTargets" and isinstance(infos := result.get("targetInfos"), list):
        for info in infos:
            targets.saw(info)
        return {**result, "targetInfos": [info for info in infos if isinstance(info, dict)
                                          and sees(contexts, scope, user, info.get("browserContextId"))]}
    if method == "Target.getBrowserContexts" and isinstance(ids := result.get("browserContextIds"), list):
        own = [context for context in ids if context != scope and (lease := contexts.of(context)) is not None
               and lease.made_by == user]
        shown = {**result, "browserContextIds": own}
        if scope is not None and "defaultBrowserContextId" in result:
            shown["defaultBrowserContextId"] = scope
        return shown
    return result
