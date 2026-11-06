"""Live websites in a run: marking a run that uses one, and signing the browser in to them.

A website child env whose card's website/v1 says ``live`` is a real site. ``mark_live``, which deploy_env runs on every
env it deploys, records each one under ``context.metadata["live_websites"]``, sets ``context.metadata["reproducible"]``
to False, and marks the deploy record ``live``. It also checks the site's domains and ``allow_hosts`` against the network
allowlist of each computer deployed in the run. An allowlist is fixed when its sandbox starts, so a host one leaves out
is recorded under ``unreachable`` and warned about, not added; a run whose sandboxes restrict no egress records
``network: "unrestricted"``, since nothing there enforces ``allow_hosts`` either.

``sign_in_browser`` calls each live site's website/v1 ``session`` and hands its storage state straight to the
browser's browser/v1: the site's old cookies go first, through ``cookies/delete`` on a browser that serves it, then its
cookies through ``cookies/set`` and its local storage through ``local_storage/set``, each called on the child env's own
card as the env's card nests it, so a site sharing its name with an MCP server, or with another child, is still the one
called. The values never enter the context, a log or the step's record, which keeps each cookie without its value and
the origins whose local storage was set. reset_env signs the browser in again after a reset, as the last sign-in did.
With several deployments of the env in a run, ``env_step_id`` names the deploy_env step whose deployment it signs in; the
record keeps that step and the call timeout, so the sign-in after a reset reaches the same browser the same way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import httpx
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.website import WEBSITE_EXTENSION_URI, session_state

from agent_env.entity.refs import EntityRef
from agent_env.env.env import EnvCapabilityUnsupported
from agent_env.env.gateway.constants import EXT_BROWSER_URI as BROWSER_EXTENSION_URI
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

LIVE_KEY = "live_websites"
REPRODUCIBLE_KEY = "reproducible"
SIGN_INS_KEY = "browser_sign_ins"
_ERROR_CHARS = 500
SIGN_IN_TIMEOUT_SECONDS = 30
SET_COOKIES, DELETE_COOKIES, SET_LOCAL_STORAGE = "cookies/set", "cookies/delete", "local_storage/set"


@dataclass(frozen=True)
class LiveSite:
    """A live website child env, as its card declares it; ``child`` is that child env, which its session is called on,
    and None for a site read back from a record."""

    name: str
    domains: tuple[str, ...]
    allow_hosts: tuple[str, ...]
    signs_in: bool
    child: Optional[protocol_v1.ChildEnv] = field(default=None, compare=False, repr=False)

    def hosts(self) -> list[str]:
        return [*self.domains, *(h for h in self.allow_hosts if h not in self.domains)]

    def record(self) -> dict:
        return {"name": self.name, "domains": list(self.domains), "allow_hosts": list(self.allow_hosts),
                "signs_in": self.signs_in}


def _children(card: dict) -> list[protocol_v1.ChildEnv]:
    """``card``'s child envs, or ``card`` itself, at the env's own address, for a leaf card."""
    if isinstance(card.get("children_environments"), list):
        return protocol_v1.children(card)
    return [protocol_v1.ChildEnv(name=card.get("name"), url=None, card=card)]


def live_sites(card: Optional[dict]) -> list[LiveSite]:
    """The live websites among ``card``'s child envs, or ``card`` itself for a leaf card."""
    sites = []
    for child in _children(card or {}):
        params = (child.find_extension(WEBSITE_EXTENSION_URI) or {}).get("params") or {}
        if params.get("live") is True and isinstance(child.name, str):
            sites.append(LiveSite(child.name, tuple(params.get("domains") or ()),
                                  tuple(params.get("allow_hosts") or ()),
                                  params.get("signs_in") is not False
                                  and protocol_v1.find_extension_method(child.card, WEBSITE_EXTENSION_URI,
                                                                        "session") is not None, child))
    return sites


def browsers(card: Optional[dict]) -> list[protocol_v1.ChildEnv]:
    """``card``'s child envs that serve browser/v1's cookies/set."""
    return [child for child in _children(card or {}) if isinstance(child.name, str)
            and protocol_v1.find_extension_method(child.card, BROWSER_EXTENSION_URI, SET_COOKIES) is not None]


async def _invoke(deployed: Any, child: protocol_v1.ChildEnv, uri: str, method: str, params: dict,
                  timeout: int) -> Any:
    """Call ``method`` of the extension at ``uri`` as ``child``'s own card advertises it, at ``deployed``'s address
    with its control headers, which a lookup of the child by its name could confuse with another of that name."""
    if not deployed.environment_url or protocol_v1.find_extension_method(child.card, uri, method) is None:
        raise EnvCapabilityUnsupported(uri, method, deployed.env_id, deployed.instance_id, child.name)
    return await protocol_v1.invoke_extension(deployed.environment_url, child.card, uri, params, timeout,
                                              method=method, headers=deployed.control_headers)


def _allows(entries: list[str], host: str) -> bool:
    host = host.lower().rstrip(".")
    for entry in (e.lower().rstrip(".") for e in entries):
        if entry == host or (entry.startswith("*.") and host.endswith(entry[1:])):
            return True
    return False


def network_check(context: TaskStepContext, sites: list[LiveSite], deployed: Any = None) -> dict:
    """How the run's computers' network allowlists treat the live ``sites``' hosts: each restricting computer's
    hosts it leaves out, or ``unrestricted`` when no computer in the run restricts egress. ``deployed``, an env being
    deployed that the context doesn't hold yet, is one of the run's."""
    from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy

    hosts = list(dict.fromkeys(h for site in sites for h in site.hosts()))
    unreachable: dict[str, list[str]] = {}
    restricted = False
    envs = [*context.deployed_envs, *([deployed] if deployed is not None and deployed not in context.deployed_envs
                                       else [])]
    for env in envs:
        network = ((env.metadata or {}).get("computer") or {}).get("network")
        if not network:
            continue
        policy = NetworkPolicy.from_dict(network)
        if not policy.restricts_egress:
            continue
        restricted = True
        if missing := [h for h in hosts if not _allows(list(policy.allow_hosts), h)]:
            unreachable[env.env_id] = missing
    if not restricted:
        return {"network": "unrestricted"}
    return {"network": "allowlist", "unreachable": unreachable}


def mark_live(context: TaskStepContext, deployed: Any) -> dict:
    """Record ``deployed``'s live websites in ``context`` and return what its deploy record gains: ``live`` and
    ``reproducible`` when it has one, nothing when it has none. The network check of every live website recorded so
    far is made again with ``deployed`` among the run's computers, so a computer deployed after a site is checked
    too."""
    sites = live_sites(deployed.environment_card)
    if sites:
        context.metadata.setdefault(LIVE_KEY, {})[deployed.env_id] = {"sites": [s.record() for s in sites]}
        context.metadata[REPRODUCIBLE_KEY] = False
    for env_id, entry in (context.metadata.get(LIVE_KEY) or {}).items():
        before = entry.pop("unreachable", None) or {}
        entry_sites = sites if env_id == deployed.env_id else [
            LiveSite(r["name"], tuple(r["domains"]), tuple(r["allow_hosts"]), r["signs_in"]) for r in entry["sites"]]
        network = network_check(context, entry_sites, deployed)
        entry.update(network)
        for computer, missing in (network.get("unreachable") or {}).items():
            if before.get(computer) != missing:
                logger.warning("live websites of env %s: env %s's network allowlist leaves out %s, which it can't add "
                               "once its sandbox runs, so those pages won't load there", env_id, computer, missing)
        if env_id == deployed.env_id and network["network"] == "unrestricted":
            logger.info("live websites of env %s (%s): no sandbox in this run restricts egress, so allow_hosts is "
                        "recorded, not enforced", env_id, [s.name for s in sites])
    return {"live": True, REPRODUCIBLE_KEY: False} if sites else {}


async def sign_in(context: TaskStepContext, deployed: Any, *, users: Optional[dict[str, str]] = None,
                  sites: Optional[list[str]] = None, browser: Optional[str] = None, clear: bool = True,
                  timeout: int = SIGN_IN_TIMEOUT_SECONDS) -> dict:
    """Sign ``deployed``'s browser in to its live websites (those named in ``sites``, else every one with a session),
    each as ``users[name]`` or the site's own choice; with ``clear``, the old cookies of a site's domains and of the
    domains its cookies name go first, on a browser that serves cookies/delete. Returns the record of what was set,
    cookies by everything but their values."""
    card = deployed.environment_card or {}
    live = {site.name: site for site in live_sites(card)}
    unknown = sorted(set(sites or []) - set(live))
    if unknown:
        raise RuntimeError(f"env {deployed.env_id!r} has no live website named {unknown}; its live ones are "
                           f"{sorted(live) or 'none'}")
    chosen = [live[name] for name in sites] if sites else [s for s in live.values() if s.signs_in]
    offered = [child for child in browsers(card) if browser is None or child.name == browser]
    if chosen and not offered:
        named = f"named {browser!r} " if browser is not None else ""
        raise RuntimeError(f"env {deployed.env_id!r} has live websites with a session ({[s.name for s in chosen]}) "
                           f"but no child env {named}serves {BROWSER_EXTENSION_URI} {SET_COOKIES} to sign a browser in")
    if chosen and len(offered) > 1:
        which = f"named {browser!r} " if browser is not None else ""
        hint = "the name can't say" if browser is not None else "set browser to say"
        raise RuntimeError(f"env {deployed.env_id!r} has {len(offered)} browsers {which}serving {SET_COOKIES} "
                           f"({', '.join(str(c.name) for c in offered)}), so {hint} which to sign in")
    target = offered[0] if chosen else None
    browser = target.name if target is not None else browser
    record: dict = {"env_id": deployed.env_id, "env_step_id": deploy_step_of(deployed), "browser": browser,
                    "timeout_seconds": timeout, "sites": []}
    for site in chosen:
        user = (users or {}).get(site.name)
        try:
            session = await _invoke(deployed, site.child, WEBSITE_EXTENSION_URI, "session", {"user": user}, timeout)
        except Exception as e:
            raise RuntimeError(f"live website {site.name!r}: its session failed ({_why(e)})") from None
        try:
            state = session_state(session.get("cookies") if isinstance(session, dict) else None,
                                  session.get("local_storage") or [] if isinstance(session, dict) else [], site.domains)
        except ValueError as e:
            raise RuntimeError(f"live website {site.name!r}: its session returned {e}") from None
        given, origins = state["cookies"], state["local_storage"]
        if origins and protocol_v1.find_extension_method(target.card, BROWSER_EXTENSION_URI, SET_LOCAL_STORAGE) is None:
            raise RuntimeError(f"live website {site.name!r}: its session sets local storage, which browser {browser!r} "
                               f"doesn't serve {SET_LOCAL_STORAGE} to take")
        try:
            if clear and protocol_v1.find_extension_method(target.card, BROWSER_EXTENSION_URI,
                                                           DELETE_COOKIES) is not None:
                for domain in _cookie_domains(site, given):
                    await _invoke(deployed, target, BROWSER_EXTENSION_URI, DELETE_COOKIES, {"domain": domain}, timeout)
            await _invoke(deployed, target, BROWSER_EXTENSION_URI, SET_COOKIES, {"cookies": given}, timeout)
            if origins:
                await _invoke(deployed, target, BROWSER_EXTENSION_URI, SET_LOCAL_STORAGE, {"origins": origins},
                              timeout)
        except Exception as e:
            raise RuntimeError(f"browser {browser!r}: setting {site.name!r}'s storage state failed ({_why(e)})") from None
        record["sites"].append({"name": site.name, "user": user, "cookies": _redacted(given),
                                "local_storage": [o["origin"] for o in origins]})
        logger.info("sign_in_browser: %s signed in to %s as %s (%s)", browser, site.name, user or "its own choice",
                    [c["name"] for c in given])
    return record


def _cookie_domains(site: LiveSite, given: list[dict]) -> list[str]:
    """The domains whose cookies a sign-in replaces: the site's own, ``.example.com`` for ``*.example.com``, a
    domain both with and without its leading dot, and those its cookies name."""
    named = [d.removeprefix("*") for d in site.domains] + [c["domain"] for c in given if c.get("domain")]
    return list(dict.fromkeys(d for name in named for d in (name.lstrip("."), "." + name.lstrip("."))))


def _redacted(given: list[dict]) -> list[dict]:
    """What a record keeps of each cookie: everything but its value."""
    return [{k: v for k, v in c.items() if k != "value"} for c in given]


def _why(error: Exception) -> str:
    """An error as a record keeps it: a call's status and error code, or the error's kind, never a message that may
    repeat a cookie."""
    if isinstance(error, httpx.HTTPStatusError):
        code = getattr(error, "code", None)
        return f"HTTP {error.response.status_code}{f' {code}' if code else ''}"
    if isinstance(error, EnvCapabilityUnsupported):
        return str(error)
    return type(error).__name__


def deploy_step_of(deployed: Any) -> Optional[str]:
    """The id of the deploy_env step that deployed ``deployed``, or None."""
    return (getattr(deployed, "metadata", None) or {}).get("deploy_step_id")


def deployment(context: TaskStepContext, env_id: str, env_step_id: Optional[str], step_id: str) -> Any:
    """The deployment of ``env_id`` a sign-in reaches: the one deploy_env step ``env_step_id`` made, else the only one;
    raises when there is none, or several and no ``env_step_id`` says which."""
    matches = [d for d in context.deployed_envs if d.env_id == env_id]
    if env_step_id is not None:
        named = [d for d in matches if deploy_step_of(d) == env_step_id]
        if not named:
            raise RuntimeError(f"sign_in_browser {step_id!r}: env_step_id {env_step_id!r} names no deployment of env "
                               f"{env_id!r} (deployed by: {[deploy_step_of(d) for d in matches]})")
        matches = named
    if not matches:
        raise RuntimeError(f"Env '{env_id}' not found in context.deployed_envs")
    if len(matches) > 1:
        raise RuntimeError(f"sign_in_browser {step_id!r}: {len(matches)} deployments of env {env_id!r} (deployed by: "
                           f"{[deploy_step_of(d) for d in matches]}); set env_step_id to the deploy_env step whose "
                           "browser to sign in")
    return matches[0]


async def sign_in_again(context: TaskStepContext, deployed: Any) -> None:
    """After a reset of ``deployed``, sign its browser in again as each sign-in of this run that reached it did, with
    the same timeout; never raises."""
    for step_id, entry in list((context.metadata.get(SIGN_INS_KEY) or {}).items()):
        if entry.get("env_id") != deployed.env_id or not entry.get("sites"):
            continue
        if entry.get("env_step_id") is not None and entry["env_step_id"] != deploy_step_of(deployed):
            continue
        users = {s["name"]: s["user"] for s in entry["sites"] if s.get("user")}
        try:
            context.metadata[SIGN_INS_KEY][step_id] = await sign_in(
                context, deployed, users=users, sites=[s["name"] for s in entry["sites"]], browser=entry.get("browser"),
                timeout=entry.get("timeout_seconds") or SIGN_IN_TIMEOUT_SECONDS)
        except Exception as e:
            logger.warning("reset_env: signing env %s's browser in again failed: %s", deployed.env_id, e)
            entry["error"] = str(e)[:_ERROR_CHARS]


class SignInBrowserTaskStep(TaskStep):
    """Sign a deployed env's browser in to its live websites before the agent starts. ``users`` maps a site's name to
    the user to sign in as; ``sites`` limits it to some live sites; ``browser`` names the child env serving browser/v1
    when there are several; ``env_step_id`` names the deploy_env step whose deployment it signs in, when the env is
    deployed more than once."""

    type: ClassVar[str] = "sign_in_browser"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, users: Optional[dict[str, str]] = None,
                 sites: Optional[list[str]] = None, browser: Optional[str] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True,
                 timeout_seconds: int = SIGN_IN_TIMEOUT_SECONDS, env_step_id: Optional[str] = None):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not env_id:
            raise ValueError("sign_in_browser needs the env_id of the env whose browser it signs in")
        if users is not None and not (isinstance(users, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in users.items())):
            raise ValueError("sign_in_browser's users maps a site's name to a user's")
        self.env_id, self.users, self.sites, self.browser = env_id, users, sites, browser
        self.timeout_seconds, self.env_step_id = timeout_seconds, env_step_id

    def to_dict(self) -> dict:
        data = {**super().to_dict(), "env_id": self.env_id, "timeout_seconds": self.timeout_seconds}
        for key in ("users", "sites", "browser", "env_step_id"):
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SignInBrowserTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], users=data.get("users"),
                   sites=data.get("sites"), browser=data.get("browser"),
                   timeout_seconds=data.get("timeout_seconds", SIGN_IN_TIMEOUT_SECONDS), env_step_id=data.get("env_step_id"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = deployment(context, self.env_id, self.env_step_id, self.id)
        if live_sites(deployed.environment_card) and self.env_id not in (context.metadata.get(LIVE_KEY) or {}):
            mark_live(context, deployed)
        try:
            record = await sign_in(context, deployed, users=self.users, sites=self.sites, browser=self.browser,
                                   timeout=self.timeout_seconds)
        except Exception as e:
            context.metadata.setdefault(SIGN_INS_KEY, {})[self.id] = {"env_id": self.env_id, "error": str(e)[:_ERROR_CHARS]}
            logger.warning("sign_in_browser %s: %s", self.id, e)
            if self.fail_task_on_error:
                raise
            return context
        context.metadata.setdefault(SIGN_INS_KEY, {})[self.id] = record
        if not record["sites"]:
            logger.info("sign_in_browser %s: env %s has no live website with a session; nothing to sign in to",
                        self.id, self.env_id)
        return context
