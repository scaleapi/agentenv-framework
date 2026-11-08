"""A multi env's computers: each on a sandbox of its own, deployed as it would be alone, which the multi env's gateway
fronts by its name. ``Computers`` holds them across deploys; the rules a computer's name and child_envs entry follow are
here too. Nothing here imports the computer env, which imports the env providers that import this package."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Optional

from agentenv_protocol.computer import tool_prefix_problem

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, Env
from agent_env.env.envs.computer.utils import sites
from agent_env.env.gateway import ExternalChild
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


def deploy_options(entry: Optional[dict], sandbox_type: str | None) -> dict:
    """A computer's deploy options from its child_envs entry: on ``sandbox_type``, the multi env's, unless it names one."""
    return {"sandbox_type": sandbox_type, **(entry or {})}


def routes_later(entry: Optional[dict]) -> bool:
    """Whether a computer with ``entry``, its child_envs entry, routes the multi env's websites, which come up with its
    gateway, after it: ``site_routing`` with no ``sites`` of its own."""
    params = (entry or {}).get("env_params") or {}
    return bool(params.get("site_routing")) and not params.get("sites")


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
        if isinstance(routing, list) and (others := [env_id for env_id in routing if env_id not in websites]):
            return (f"env_params.site_routing names {', '.join(map(repr, others))}, which isn't one of the multi env's "
                    f"websites ({', '.join(map(repr, websites))})")
    links = (build_sandbox_provider(options["sandbox_type"]) if options["sandbox_type"] else get_env_sandbox_provider()).links
    if headed := [type(link).__name__ for link in links if link.REQUEST_HEADERS]:
        return (f"{', '.join(headed)} takes credentials on each request to a sandbox's URL, which the multi env's gateway "
                f"doesn't send to a child env; set child_envs.{computer.environment_name}.sandbox_type to a sandbox type "
                "whose URLs take none")
    return computer.deploy_refusal(**options, sites_later=routes_later(entry))


class Computers:
    """A multi env's computers, and those of each deployment a later deploy replaced, which ``close`` ends too.

    A deploy's computers are its own, as its provider and sandbox are (``deploy_through_provider``): ``begin`` gives it
    fresh ones when these already have a deployment and sets the earlier ones aside, so its failure closes only what it
    made and ``rollback`` puts them back; ``commit`` keeps them for ``close``."""

    def __init__(self, envs: list[ComputerEnv], owner: str) -> None:
        self.envs = envs
        self._owner = owner  # the multi env's id, for what's logged
        self._routes: dict[str, Optional[list[str]]] = {}  # each computer routing sites: the names it routes, or all
        self._replaced: list[ComputerEnv] = []
        self._earlier: Optional[tuple[list[ComputerEnv], list[ComputerEnv]]] = None

    def begin(self) -> None:
        self._earlier = (self.envs, self._replaced)
        if any(env._deployed is not None for env in self.envs):
            self.envs = [Env.get(env.id, env.version) for env in self.envs]
        self._replaced = []

    def rollback(self) -> None:
        (self.envs, self._replaced), self._earlier = self._earlier, None

    def commit(self) -> None:
        (earlier, replaced), self._earlier = self._earlier, None
        self._replaced = [*replaced, *(earlier if self.envs is not earlier else [])]

    def holding(self, environment_name: str) -> Optional[ComputerEnv]:
        """The computer a load of ``environment_name`` reaches: the one of that name, or with an app of it."""
        return next((env for env in self.envs if env.holds(environment_name)), None)

    async def load(self, environments: list[EnvironmentArtifact]) -> None:
        """Load each computer's of ``environments`` into it, in its own order, the computers at once."""
        loads = [(env, held) for env in self.envs if (held := [e for e in environments if env.holds(e.environment_name)])]
        await _all(self._owner, "load into", [env for env, _ in loads],
                   [env.load_environments(held) for env, held in loads])

    async def reset(self, deployed: DeployedEnv) -> None:
        """Reset each computer ``deployed``, the multi env's record, holds, from its own record, as it resets alone."""
        records = getattr(deployed, "child_envs", None) or {}
        envs = [env for env in self.envs if env.environment_name in records]
        await _all(self._owner, "reset", envs,
                   [env.reset(DeployedEnv.from_dict(records[env.environment_name])) for env in envs])

    def deployer(self, child_envs: Optional[dict], *, sandbox_type: str | None, ttl_seconds: int,
                 attribution: Optional[Attribution], websites: Mapping[str, str]
                 ) -> Callable[[Optional[str]], Awaitable[list[ExternalChild]]]:
        """The gateway's ``children``: every computer deployed as its child_envs entry says, its tools served under its
        name and ``_``, and its network allowing ``reach``, the gateway's address when the gateway has one yet; then each
        as the gateway fronts it. A computer routing the multi env's ``websites`` (their environment names by env id)
        boots with its boot apps held, for ``finish`` to route them once the gateway is up. A gateway deploy tried again
        deploys them again, after closing the earlier try's."""
        tries = 0
        self._routes = {}
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
                **deploy_options((child_envs or {}).get(env.environment_name), sandbox_type))
                for env in self.envs), return_exceptions=True)
            if failed := [result for result in results if isinstance(result, BaseException)]:
                for error in failed[1:]:
                    logger.warning(f"multi env '{self._owner}': another computer failed to deploy too: {error!r}")
                raise failed[0]
            return [ExternalChild(env.environment_name, record.mcp_url, record.environment_url)
                    for env, record in zip(self.envs, results)]
        return deploy

    async def finish(self, deployed: DeployedEnv) -> None:
        """Once the gateway is up, before ``deployed``, the multi env's record, is registered: each computer that routes
        the multi env's websites routes them, those its ``site_routing`` names or all, once they're ready, and launches
        its held apps; then each computer's own record is kept in ``deployed`` (``record``)."""
        if self._routes:
            found = await sites.deployed_sites([deployed])
            await sites.require_ready([deployed])
            envs = [env for env in self.envs if env.environment_name in self._routes]

            def routed(name: str) -> list[dict]:
                only = self._routes[name]
                return [site for site in found if only is None or site["name"] in only]

            await _all(self._owner, "route the sites of", envs, [env.route_sites(routed(env.environment_name)) for env in envs])
        self.record(deployed)

    def record(self, deployed: DeployedEnv) -> None:
        """Keep each computer's own record in ``deployed``, the multi env's, before it is registered: whole in its
        ``child_envs``, for ``reattach``, and its sandbox in its ``sandbox_ids`` under the backend that made it, for
        teardown."""
        if not isinstance(deployed, DeployedGatewayEnv):
            raise TypeError(f"multi env '{self._owner}': its computers are recorded in a gateway's record, not a "
                            f"{type(deployed).__name__}")
        records = {env.environment_name: env._deployed for env in self.envs}
        deployed.child_envs = {name: dataclasses.asdict(record) for name, record in records.items()}
        for name, record in records.items():
            deployed.sandbox_ids.setdefault(record.sandbox_type, {})[name] = record.sandbox_id

    async def reattach(self, deployed: DeployedEnv) -> None:
        """Each computer ``deployed``, the multi env's record, holds, reattached from its own record."""
        records = getattr(deployed, "child_envs", None) or {}
        reattached = await asyncio.gather(*(type(env).from_deployed_env(DeployedEnv.from_dict(records[env.environment_name]))
                                            for env in self.envs if env.environment_name in records))
        by_name = {env.environment_name: env for env in reattached}
        self.envs = [by_name.get(env.environment_name, env) for env in self.envs]

    async def close(self) -> None:
        """Close each computer, and those of the deployments a later deploy replaced, their recordings ended first; one
        that fails to close is logged."""
        envs = [*self._replaced, *self.envs]
        results = await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)
        for env, result in zip(envs, results):
            if isinstance(result, Exception):
                logger.warning(f"multi env '{self._owner}': failed to close computer {env.environment_name!r}: {result!r}")
            elif isinstance(result, BaseException):
                raise result


async def _all(owner: str, what: str, envs: list[ComputerEnv], calls: list[Awaitable]) -> None:
    """Await ``calls``, one per computer of ``envs``, at once, then raise the first failure, each other one logged."""
    results = await asyncio.gather(*calls, return_exceptions=True)
    failed = [(env, result) for env, result in zip(envs, results) if isinstance(result, BaseException)]
    for env, error in failed[1:]:
        logger.warning(f"multi env '{owner}': failed to {what} computer {env.environment_name!r} too: {error!r}")
    if failed:
        raise failed[0][1]
