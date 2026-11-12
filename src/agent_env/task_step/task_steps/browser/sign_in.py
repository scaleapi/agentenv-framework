"""browser_sign_in: an env's website browser signed in to a site as a user, with the cookies and local storage the
site's website/v1 session gives, scoped to the site's names and loaded through the browser's data/add."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agentenv_protocol import METHOD_ADD, DataPart, client as protocol_v1
from agentenv_protocol.website import WEBSITE_EXTENSION_URI
from agentenv_protocol.website_browser import BROWSER

from agent_env.entity.refs import EntityRef
from agent_env.env.env import DeployedEnv, Env, website_child
from agent_env.env.envs.computer.utils import sites
from agent_env.env.envs.website import WebsiteEnv
from agent_env.store.base import NotFoundError
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.preflight import stored_env
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.task_step.task_steps.browser.base import LEASES, BrowserStep

logger = logging.getLogger(__name__)

#: How long a site may take to answer for a session, and the browser to load it.
SESSION_TIMEOUT_S = 30
ADD_TIMEOUT_S = 120


class BrowserSignInTaskStep(BrowserStep):
    """Sign the browser in to the website env ``site``, deployed in the run on its own or in a multi env, as ``user``:
    the site's ``session {user}`` gives the cookies and local storage, each cookie is set for every name the site is
    reached by, and the browser's ``data/add`` loads them, which opens its start tabs again while no agent has acted;
    in the lease ``context`` names, which must be leased, they load there and no tabs open again. A task runs it after
    the load that seeds the site, and again after a reset, which starts the browser afresh and ends every lease."""

    type: ClassVar[str] = "browser_sign_in"
    entity_refs = (EntityRef.env("env_id"), EntityRef.env("site"))

    def __init__(self, id: str, version: Optional[int], env_id: str, site: str, user: str,
                 browser: Optional[str] = None, depends_on: Optional[list[TaskStepDependency]] = None,
                 fail_task_on_error: bool = True, context: Optional[str] = None):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         context=context)
        self.site = site
        self.user = user

    def config_problems(self) -> list[str]:
        problems = super().config_problems()
        if not isinstance(self.site, str) or not self.site:
            problems.append(f"step {self.id!r}: site must be a website env's id")
        if not isinstance(self.user, str) or not self.user:
            problems.append(f"step {self.id!r}: user must be a non-empty string; browser_delete_cookies signs out")
        return problems

    def preflight(self) -> list[str]:
        problems = super().preflight()
        if not isinstance(self.site, str) or not self.site:
            return problems
        site = stored_env(self.site)
        if site is not None and not isinstance(site, WebsiteEnv):
            problems.append(f"step {self.id!r}: site {self.site!r} is a {type(site).__name__}, not a website env")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "site": self.site, "user": self.user}

    @classmethod
    def from_dict(cls, data: dict) -> BrowserSignInTaskStep:
        return cls(**cls._in_context_from_dict(data), site=data.get("site"), user=data.get("user"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        record, child, names = await self._site(context)
        if protocol_v1.find_extension_method(child.card, WEBSITE_EXTENSION_URI, "session") is None:
            raise RuntimeError(f"site {self.site!r} serves no website/v1 session, so it can't sign a browser in")
        if self.context_label is not None:
            await self._leased(context)
        session = await record.invoke(WEBSITE_EXTENSION_URI, "session", {"user": self.user}, child=child,
                                      timeout=SESSION_TIMEOUT_S)
        cookies = scoped(session.get("cookies") or [], names)
        owner, browser, card = self._find(context)
        if protocol_v1.serves(card, METHOD_ADD) is False:
            raise RuntimeError(f"the website browser {browser or owner.env_id!r} serves no data/add: rebuild it on a "
                               "newer agentenv-framework-protocol")
        state = {"cookies": cookies, "local_storage": session.get("local_storage") or []}
        if self.context_label is not None:
            state = {"contexts": {self.context_label: state}}
        await protocol_v1.add_data(_data_plane(owner, browser), [DataPart(data=state)], timeout=ADD_TIMEOUT_S,
                                   headers=owner.control_headers)
        self._record(context, site=self.site, user=self.user,
                     cookies=sorted({cookie["name"] for cookie in cookies}), names=names)
        logger.info(f"{self.id}: signed {browser or owner.env_id!r} in to {self.site!r} as {self.user!r} "
                    f"({len(cookies)} cookie(s) on {', '.join(names)})")
        return context

    async def _leased(self, context: TaskStepContext) -> None:
        """RuntimeError unless the browser leases contexts and has the lease ``context`` names: a sign-in loads into a
        lease, never makes one, so a mistyped label fails rather than signing an empty context in."""
        owner, browser, card = self._find(context)
        if protocol_v1.find_extension_method(card, BROWSER, LEASES) is None:
            raise RuntimeError(f"the website browser {browser or owner.env_id!r} leases no contexts: rebuild it on a "
                               "newer agentenv-framework-protocol")
        leased = [lease["context"] for lease in (await self._call(context, "contexts/list", {}, in_context=False))[
            "contexts"]]
        if self.context_label not in leased:
            raise RuntimeError(f"no context {self.context_label!r} is leased in the browser of env {self.env_id!r} "
                               f"(leased: {leased}): lease it with browser_lease_context first")

    async def _site(self, context: TaskStepContext) -> tuple[DeployedEnv, protocol_v1.ChildEnv, list[str]]:
        """The record that serves the site, the site among its children, and the names a browser reaches it by: a
        website env deployed on its own, or one of a deployed multi env's websites."""
        for record in context.deployed_envs:
            try:
                env = Env.get(record.env_id, record.env_version)
            except NotFoundError:
                continue
            website = env if record.env_id == self.site else next(
                (site for site in getattr(env, "website_envs", None) or [] if site.id == self.site), None)
            if not isinstance(website, WebsiteEnv):
                continue
            child = _served(record, website.environment_name)
            if child is None:
                raise RuntimeError(f"env {record.env_id!r}'s card serves no website as {website.environment_name!r}")
            found = await sites.deployed_sites([record], services=[website.environment_name])
            names = list(dict.fromkeys(name for site in found for name in site["domains"])) or _own_names(website, child)
            if not names:
                raise RuntimeError(f"site {self.site!r} has no names a browser reaches it by")
            return record, child, names
        raise RuntimeError(f"site {self.site!r} isn't deployed in this run: no env in context.deployed_envs is it or "
                           "has it among its websites")


def _served(record: DeployedEnv, service: str) -> Optional[protocol_v1.ChildEnv]:
    """The website ``record``'s card serves as ``service``, the name of the route it's at, which its card's name
    needn't be (``sites.served_as``); a leaf card, a website's deployed alone, is its own."""
    card = record.environment_card or {}
    if not isinstance(card.get("children_environments"), list):
        return protocol_v1.ChildEnv(name=card.get("name"), url=card.get("url"), card=card) if card else None
    return next((child for child in protocol_v1.children(card)
                 if website_child(child) and sites.served_as(child) == service), None)


def _own_names(website: WebsiteEnv, child: protocol_v1.ChildEnv) -> list[str]:
    """The names ``website`` stores, else those its card's ``website/v1`` declares, for a site ``deployed_sites``
    doesn't route: one a plugin deploys alone, with a card of its own and no frontend."""
    declared = protocol_v1.extension_params(child.card, WEBSITE_EXTENSION_URI).get("domains")
    names = website.domains or (declared if isinstance(declared, list) else [])
    return list(dict.fromkeys(name for name in names if isinstance(name, str) and name))


def scoped(cookies: list[dict], names: list[str]) -> list[dict]:
    """Each of ``cookies``, which ``session`` leaves unplaced, set for each of the site's ``names``: an exact name by
    its URL, so the cookie is the name's alone, over HTTPS, a ``.localhost`` one over HTTP; a ``*.`` name as a domain
    cookie on the name it ends in."""
    placed = []
    for cookie in cookies:
        unplaced = {key: value for key, value in cookie.items() if key not in ("url", "domain")}
        for name in names:
            if name.startswith("*."):
                placed.append({**unplaced, "domain": f".{name[2:]}", "path": unplaced.get("path") or "/"})
            else:
                scheme = "http" if name == "localhost" or name.endswith(".localhost") else "https"
                placed.append({**{key: value for key, value in unplaced.items() if key != "path"},
                               "url": f"{scheme}://{name}{unplaced.get('path') or '/'}"})
    return placed


def _data_plane(owner: DeployedEnv, browser: Optional[str]) -> str:
    """Where the browser's data plane is: its child's in ``owner``'s card, or ``owner``'s own for a browser alone."""
    if browser is None:
        return owner.environment_url
    child = owner.child_env(browser)
    if child is None:
        raise RuntimeError(f"env {owner.env_id!r}'s card lists no {browser!r}")
    return child.base_url(owner.environment_url)
