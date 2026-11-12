"""A multi env's computers: each on a sandbox of its own, deployed as it would be alone, which the multi env's gateway
fronts by its name. ``Computers`` holds them across deploys; the rules a computer's name and child_envs entry follow are
here too. Nothing here imports the computer env, which imports the env providers that import this package."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlsplit

from agentenv_protocol.computer import tool_prefix_problem

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, Env
from agent_env.env.envs.computer.utils import sites
from agent_env.env.gateway import ExternalChild
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_env_sandbox_provider

if TYPE_CHECKING:
    from agent_env.artifact import EnvironmentArtifact
    from agent_env.attribution import Attribution
    from agent_env.env.envs.computer.env import ComputerEnv

logger = logging.getLogger(__name__)


def name_problems(computers: Iterable[str], *, mcp_servers: Iterable[str] = (), websites: Iterable[str] = ()) -> list[str]:
    """What's wrong with ``computers``, the environment names of a multi env's computers, beside the names of its
    ``mcp_servers`` and ``websites``: a computer serves its tools under its name and ``_``, which must make a tool prefix,
    and the gateway knows a computer by its name alone, which no other child env and no route of the gateway's to an
    MCP server (``mcp-<server>``) may have."""
    computers, mcp_servers = list(computers), list(dict.fromkeys(mcp_servers))
    problems = [f"computer {name!r} serves its tools under its name and '_': {problem}"
                for name in dict.fromkeys(computers) if (problem := tool_prefix_problem(f"{name}_"))]
    counts = Counter([*computers, *dict.fromkeys([*mcp_servers, *websites])])
    routes = {f"mcp-{server}": server for server in mcp_servers}
    problems += [f"{name!r} names {counts[name]} of its child envs, and the gateway knows a computer by its name alone"
                 for name in dict.fromkeys(computers) if counts[name] > 1]
    problems += [f"{name!r} is the gateway's route to MCP server {routes[name]!r}, and the gateway knows a computer by its "
                 "name alone" for name in dict.fromkeys(computers) if name in routes]
    return problems


def app_name_problems(apps: Mapping[str, Iterable[str]], others: Iterable[str]) -> list[str]:
    """What's wrong with the names of each computer's apps, ``apps`` by the computer's name, beside ``others``, every
    other name the multi env's child envs go by: loads and snapshots find an environment by its name, so an app's must
    be no other's."""
    apps = {computer: list(dict.fromkeys(names)) for computer, names in apps.items()}
    taken = Counter(set(others))
    for names in apps.values():
        taken.update(names)
    return [f"app {name!r} of computer {computer!r} has the name of another of its child envs, and loads and snapshots "
            "find an environment by its name" for computer, names in apps.items() for name in names if taken[name] > 1]


def start_tab_problems(computer: ComputerEnv, entry: Optional[dict], names: Iterable[str]) -> list[str]:
    """Why a computer routing the multi env's websites can't open its browsers' start tabs: each http or https start
    URL, its app's own setting under its child_envs entry's ``app_settings``, names a host one of ``names`` covers,
    the routed sites' and the live sites' (their domains and ``allow_hosts``). A computer that routes no sites isn't
    asked."""
    params = entry.get("env_params") if isinstance(entry, dict) else None
    if not isinstance(params, dict) or not params.get("site_routing"):
        return []
    given = params.get("app_settings") if isinstance(params.get("app_settings"), dict) else {}
    defaults, names, problems = computer.app_defaults(), list(names), []
    for app in computer.composed_apps():
        if not app.declaration.browser:
            continue
        own = given.get(app.environment_name)
        urls = {**defaults.get(app.environment_name, {}), **(own if isinstance(own, dict) else {})}.get("start_urls")
        for url in urls if isinstance(urls, list) else []:
            parts = urlsplit(url) if isinstance(url, str) else None
            if parts is None or parts.scheme not in ("http", "https") or not parts.hostname:
                continue
            if not any(sites.matches(name, parts.hostname) for name in names):
                problems.append(f"{app.environment_name}'s start tab {url} names {parts.hostname}, which none of the "
                                "multi env's websites claims; add a website env at that name, or a live one (live = "
                                "true) whose domains or allow_hosts name it")
    return problems


def deploy_options(entry: Optional[dict], sandbox_type: str | None) -> dict:
    """A computer's deploy options from its child_envs entry: on ``sandbox_type``, the multi env's, unless it names one."""
    return {"sandbox_type": sandbox_type, **(entry or {})}


def routes_later(entry: Optional[dict]) -> bool:
    """Whether a computer with ``entry``, its child_envs entry, routes the multi env's websites, which come up with its
    gateway, after it: ``site_routing`` with no ``sites`` of its own."""
    params = (entry or {}).get("env_params")  # one that isn't a table is the computer's own params check to refuse
    return isinstance(params, dict) and bool(params.get("site_routing")) and not params.get("sites")


def _clock_read_token(record: DeployedEnv) -> str | None:
    """The read token a computer's clock watch carries when its card says it's a clock follower (``clock/v1``'s
    ``sync_time``): for now its control token, which a token of its own can replace without the gateway knowing."""
    return record.control_token if record.supports(EXT_CLOCK_URI, "sync_time") else None


def refusal(computer: ComputerEnv, entry: dict, sandbox_type: str | None, websites: Mapping[str, str]) -> str | None:
    """Why ``computer`` wouldn't deploy in a multi env with ``entry``, its child_envs entry: why it wouldn't alone,
    ``site_routing`` naming none of the multi env's ``websites`` (their environment names by env id), or a sandbox
    provider whose URLs take credentials, which the gateway doesn't send to a child env."""
    options = deploy_options(entry, sandbox_type)
    params = options.get("env_params")
    if routes_later(entry):
        routing = params["site_routing"]
        if not websites:
            return "env_params.site_routing routes the multi env's websites, and it has none"
        # A list of other than ids is the computer's own params check to refuse.
        if isinstance(routing, list) and all(isinstance(env_id, str) for env_id in routing) and \
                (others := [env_id for env_id in routing if env_id not in websites]):
            return (f"env_params.site_routing names {', '.join(map(repr, others))}, which isn't one of the multi env's "
                    f"websites ({', '.join(map(repr, websites))})")
    links = (build_sandbox_provider(options["sandbox_type"]) if options["sandbox_type"] else get_env_sandbox_provider()).links
    if headed := [type(link).__name__ for link in links if link.REQUEST_HEADERS]:
        return (f"{', '.join(headed)} takes credentials on each request to a sandbox's URL, which the multi env's gateway "
                f"doesn't send to a child env; set child_envs.{computer.environment_name}.sandbox_type to a sandbox type "
                "whose URLs take none")
    return computer.deploy_refusal(**options, sites_later=routes_later(entry))


class Computers:
    """A multi env's computers, its website browser when that is a computer, and those of each deployment a later
    deploy replaced, which ``close`` ends too.

    A deploy's computers are its own, as its provider and sandbox are (``deploy_through_provider``): ``begin`` gives it
    fresh ones when these already have a deployment and sets the earlier ones aside, so its failure closes only what it
    made and ``rollback`` puts them back; ``commit`` keeps them for ``close``."""

    def __init__(self, envs: list[ComputerEnv], owner: str) -> None:
        self.envs = envs  # the multi env's own, as it stores them
        self.browser: Optional[ComputerEnv] = None  # the deploy's website browser, a computer of its own
        self._owner = owner  # the multi env's id, for what's logged
        self._routes: dict[str, Optional[list[str]]] = {}  # each computer routing sites: the names it routes, or all
        self._entries: dict = {}  # the deploy's child_envs entries, by computer
        self._live: tuple[str, ...] = ()  # the live hosts of the multi env's live websites
        self._replaced: list[ComputerEnv] = []
        self._earlier: Optional[tuple[list[ComputerEnv], Optional[ComputerEnv], list[ComputerEnv]]] = None

    @property
    def every(self) -> list[ComputerEnv]:
        """Each computer a deploy deploys: the multi env's own, then its website browser."""
        return [*self.envs, *([self.browser] if self.browser is not None else [])]

    def begin(self, browser: Optional[ComputerEnv] = None) -> None:
        """Start a deploy of the multi env's computers and ``browser``, its website browser when that is a computer."""
        self._earlier = (self.envs, self.browser, self._replaced)
        if any(env._deployed is not None for env in self.envs):
            self.envs = [Env.get(env.id, env.version) for env in self.envs]
        self.browser = Env.get(browser.id, browser.version) if browser is not None and browser._deployed else browser
        self._replaced = []

    def rollback(self) -> None:
        (self.envs, self.browser, self._replaced), self._earlier = self._earlier, None

    def commit(self) -> None:
        (envs, browser, replaced), self._earlier = self._earlier, None
        kept = {id(env) for env in self.every}
        self._replaced = [*replaced, *(env for env in [*envs, *([browser] if browser else [])]
                                       if env._deployed is not None and id(env) not in kept)]

    def holding(self, environment_name: str) -> Optional[ComputerEnv]:
        """The computer a load of ``environment_name`` reaches: the one of that name, or with an app of it."""
        return next((env for env in self.every if env.holds(environment_name)), None)

    async def load(self, environments: list[EnvironmentArtifact]) -> None:
        """Load each computer's of ``environments`` into it, in its own order, the computers at once (``each``)."""
        await each(self._owner, "load", [(f"computer {env.environment_name!r}", env.load_environments(held))
                                         for env in self.every
                                         if (held := [e for e in environments if env.holds(e.environment_name)])])

    def resets(self, deployed: DeployedEnv) -> list[tuple[str, Awaitable[None]]]:
        """Each computer ``deployed``, the multi env's record, holds, by name, with its reset from its own record, as it
        resets alone: the multi env's own, and its website browser, which a multi env read from the store finds in the
        record; none is awaited yet."""
        records = getattr(deployed, "child_envs", None) or {}
        browser = self.browser or self._recorded_browser(records)
        return [(env.environment_name, env.reset(DeployedEnv.from_dict(records[env.environment_name])))
                for env in [*self.envs, *([browser] if browser else [])] if env.environment_name in records]

    def _recorded_browser(self, records: Mapping[str, dict]) -> Optional[ComputerEnv]:
        """The website browser ``records``, a multi env's computers' records, hold beside its own computers: the record
        of a name none of them has, its env at the version it deployed."""
        own = {env.environment_name for env in self.envs}
        record = next((record for name, record in records.items() if name not in own), None)
        return Env.get(record["env_id"], record["env_version"]) if record else None

    def deployer(self, child_envs: Optional[dict], *, sandbox_type: str | None, ttl_seconds: int,
                 attribution: Optional[Attribution], websites: Mapping[str, str], live_hosts: Iterable[str] = ()
                 ) -> Callable[[Optional[str]], Awaitable[list[ExternalChild]]]:
        """The gateway's ``children``: every computer deployed as its child_envs entry says, its tools served under its
        name and ``_``, and its network allowing ``reach``, the gateway's address when the gateway has one yet, and the
        ``live_hosts`` of the multi env's live websites; then each as the gateway fronts it. A computer routing the multi env's ``websites`` (their environment names by env id)
        boots with its boot apps held, for ``finish`` to route them once the gateway is up. A gateway deploy tried again
        deploys them again, after closing the earlier try's."""
        tries = 0
        live = tuple(live_hosts)
        self._routes, self._entries, self._live = {}, dict(child_envs or {}), live
        for name, entry in (child_envs or {}).items():
            if routes_later(entry):
                routing = entry["env_params"]["site_routing"]
                self._routes[name] = None if routing is True else [websites[env_id] for env_id in routing]

        async def deploy(reach: Optional[str]) -> list[ExternalChild]:
            nonlocal tries
            if tries:
                await self.close()
            tries += 1
            results = await asyncio.gather(*(env.deploy(
                ttl_seconds=ttl_seconds, attribution=attribution, tool_prefix=f"{env.environment_name}_",
                reach=reach or "", sites_later=env.environment_name in self._routes,
                **({"live_hosts": live} if live else {}),
                **deploy_options((child_envs or {}).get(env.environment_name), sandbox_type))
                for env in self.every), return_exceptions=True)
            if failed := [result for result in results if isinstance(result, BaseException)]:
                for error in failed[1:]:
                    logger.warning(f"multi env '{self._owner}': another computer failed to deploy too: {error!r}")
                raise failed[0]
            return [ExternalChild(env.environment_name, record.mcp_url, record.environment_url, _clock_read_token(record))
                    for env, record in zip(self.every, results)]
        return deploy

    async def finish(self, deployed: DeployedEnv) -> None:
        """Once the gateway is up, before ``deployed``, the multi env's record, is registered: each computer that routes
        the multi env's websites, those its ``site_routing`` names or all, has its start tabs checked against their
        names as the cards declare them and the live hosts (``start_tab_problems``), then routes them once they're
        ready and launches its held apps; then each computer's own record is kept in ``deployed`` (``record``)."""
        if self._routes:
            # Selected by the name each website is served as, its environment_name, which its card's name needn't be.
            routed = {name: await sites.deployed_sites([deployed], services=only) for name, only in self._routes.items()}
            if problems := [f"computer {env.environment_name!r}: {problem}" for env in self.every
                            if env.environment_name in routed for problem in start_tab_problems(
                                env, self._entries.get(env.environment_name), [
                                    *(name for site in routed[env.environment_name]
                                      for name in site.get("domains") or [sites.fallback_name(site["name"])]),
                                    *self._live])]:
                raise ValueError(f"multi env '{self._owner}': " + "; ".join(problems))
            await sites.require_ready([deployed])
            await each(self._owner, "route its sites", [(f"computer {env.environment_name!r}",
                                                      env.route_sites(routed[env.environment_name]))
                                                     for env in self.every if env.environment_name in routed])
        self.record(deployed)

    def record(self, deployed: DeployedEnv) -> None:
        """Keep each computer's own record in ``deployed``, the multi env's, before it is registered: whole in its
        ``child_envs``, for ``reattach``, and its sandbox in its ``sandbox_ids`` under the backend that made it, for
        teardown."""
        if not isinstance(deployed, DeployedGatewayEnv):
            raise TypeError(f"multi env '{self._owner}': its computers are recorded in a gateway's record, not a "
                            f"{type(deployed).__name__}")
        records = {env.environment_name: env._deployed for env in self.every}
        deployed.child_envs = {name: dataclasses.asdict(record) for name, record in records.items()}
        for name, record in records.items():
            deployed.sandbox_ids.setdefault(record.sandbox_type, {})[name] = record.sandbox_id

    async def reattach(self, deployed: DeployedEnv) -> None:
        """Each computer ``deployed``, the multi env's record, holds, reattached from its own record: the multi env's
        own, and its website browser, the record of a name none of them has."""
        records = getattr(deployed, "child_envs", None) or {}
        reattached = await asyncio.gather(*(type(env).from_deployed_env(DeployedEnv.from_dict(records[env.environment_name]))
                                            for env in self.envs if env.environment_name in records))
        by_name = {env.environment_name: env for env in reattached}
        self.envs = [by_name.get(env.environment_name, env) for env in self.envs]
        if (browser := self._recorded_browser(records)) is not None:
            self.browser = await type(browser).from_deployed_env(DeployedEnv.from_dict(
                records[browser.environment_name]))

    async def close(self) -> None:
        """Close each computer, and those of the deployments a later deploy replaced, their recordings ended first; one
        that fails to close is logged."""
        envs = [*self._replaced, *self.every]
        results = await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)
        for env, result in zip(envs, results):
            if isinstance(result, Exception):
                logger.warning(f"multi env '{self._owner}': failed to close computer {env.environment_name!r}: {result!r}")
            elif isinstance(result, BaseException):
                raise result


async def each(owner: str, what: str, calls: list[tuple[str, Awaitable[Any]]]) -> list[Any]:
    """Await ``calls``, each a child env's under its label, at once, and once all have ended, what each returned; or,
    when any failed to ``what``, a RuntimeError of the multi env ``owner`` naming each that did."""
    results = await asyncio.gather(*(call for _, call in calls), return_exceptions=True)
    if interrupted := next((r for r in results if isinstance(r, BaseException) and not isinstance(r, Exception)), None):
        raise interrupted
    if failed := [(child, result) for (child, _), result in zip(calls, results) if isinstance(result, Exception)]:
        raise RuntimeError(f"multi env '{owner}': " + "; ".join(f"{child} failed to {what}: {error!r}"
                                                                for child, error in failed)) from failed[0][1]
    return results
