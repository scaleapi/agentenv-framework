"""Multi environment combining multiple MCP servers, websites and computers."""
from __future__ import annotations

import logging
import os
import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

import yaml

logger = logging.getLogger(__name__)

# Off by default: a bake builds and pushes a multi-GB servicedb image, which is the right
# trade once per universe and the wrong one on every deploy. Callers that know they are
# seeding a reusable universe pass snapshot_after_load=True explicitly; this flag exists
# so an operator can turn it on for a whole worker without a code change.
_SNAPSHOT_AFTER_LOAD_ENV_VAR = "AGENT_ENV_SNAPSHOT_AFTER_LOAD"

# Escape hatch for the derived per-load concurrency cap: pin the old load-everything-at-once
# behaviour, or hold it fixed while bisecting a contention problem.
_LOAD_CONCURRENCY_ENV_VAR = "AGENT_ENV_LOAD_CONCURRENCY"

# `nproc` is GNU coreutils; macOS has only `getconf`.
_CORE_COUNT_PROBE = "nproc 2>/dev/null || getconf _NPROCESSORS_ONLN"

def _snapshot_after_load_default() -> bool:
    return os.environ.get(_SNAPSHOT_AFTER_LOAD_ENV_VAR, "").strip().lower() in ("1", "true", "yes")

if TYPE_CHECKING:
    from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.entity.export import ExportContext
    from agent_env.env.envs.computer.env import ComputerEnv
    from agent_env.env.env import LoadEnvironmentUniverseArtifactResult, LoadFileArtifactUniverseResult
    from agent_env.providers.env_state import DatabaseStateProvider

from agent_env.config import get_config, website_browser_env_id
from agent_env.entity.refs import EntityRef
from agent_env.env.env import DeployedEnv, Env, gateway_url_of
from agent_env.env.gateway import GatewayMode
from agent_env.env.envs._deployment import (
    as_builtin, builtin_provider_for, close_deployed, close_replaced, deploy_refusal, deploy_through_provider, host_staging_refusal,
    plugin_provider_like_a_builtin, provider_or_class, reset_server_data,
)
from agent_env.env import store as env_store
from agent_env.env.envs import _multi_computers
from agent_env.env.envs.computer import COMPUTER_ENV_TYPE
from agent_env.env.envs.computer.utils import sites
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.envs.service_db import DATABASE_SERVICE_NAME
from agent_env.providers.env_providers.constants import DOCKER_COMPOSE_PATH
from agent_env.providers.env_state.local_postgres import INIT_SCRIPT_MOUNT_PATH
from agent_env.store.base import NotFoundError
from agent_env.store.ids import derive_id
from agent_env.store.image_store.oci_registry_credentials import image_reference_problem
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER
from agent_env.env.registry import get_env_registry
from agent_env.attribution import Attribution


# A child_envs entry's options: a deploy_env step's for that env alone, less the multi env's (lifetime, gateway, state).
CHILD_ENV_OPTIONS = ("sandbox_type", "cpu", "memory_mb", "disk_size_gb", "env_params")



@dataclass(frozen=True)
class WebsiteBrowser:
    """The website browser a multi env's deploy adds, and where it runs: ``computer``, a browser computer, on a sandbox
    of its own beside its computers; ``server``, an MCP server env, among its gateway's MCP servers; or neither, with
    ``runs`` when it is the run's (``website_browser_env_id``) and the store doesn't hold it as either yet, an infra
    env a run may set up later, which the gateway looks up as it deploys."""

    computer: Optional[ComputerEnv] = None
    server: Optional[MCPServerEnv] = None
    runs: bool = False

    @property
    def env(self) -> Optional[Env]:
        return self.computer or self.server


def placed(browser: Optional[Env]) -> WebsiteBrowser:
    """``browser`` where a multi env runs it: a computer on a sandbox of its own, an MCP server env among its gateway's
    MCP servers, and an env of another kind left for the gateway to look up as it deploys."""
    if browser is None:
        return WebsiteBrowser()
    if isinstance(browser, get_env_registry()[COMPUTER_ENV_TYPE]):
        return WebsiteBrowser(computer=browser)
    if isinstance(browser, MCPServerEnv):
        return WebsiteBrowser(server=browser)
    return WebsiteBrowser(runs=True)


def browser_problems(envs: list[Env]) -> list[str]:
    """Why ``envs``, a multi env's ``browser_envs``, can't be its website browsers: more than one, as a multi env runs
    one, or an env that isn't a website browser env, an MCP server env serving one as ``playwright-mcp`` does or a
    computer env with a website browser app, as ``chromium`` is."""
    problems = []
    if len(envs) > 1:
        problems.append(f"browser_envs names {len(envs)} envs, and a multi env runs one website browser")
    computer = get_env_registry()[COMPUTER_ENV_TYPE]
    for i, env in enumerate(envs):
        if isinstance(env, computer) and not env.serves_website_browser():
            problems.append(f"browser_envs[{i}]: computer env {env.id!r} has no website browser among its apps")
        elif not isinstance(env, (MCPServerEnv, computer)):
            problems.append(f"browser_envs[{i}]: {env.id!r} is a {env.type} env, not a website browser env: an MCP "
                            "server env serving one, as playwright-mcp is, or a computer env with a website browser "
                            "app, as chromium is")
    return problems


# The servicedb's image line in a deploy's compose file, as the local Postgres state renders it.
_SERVICE_DB_IMAGE = re.compile(rf"^(  {re.escape(DATABASE_SERVICE_NAME)}:\n    image: ).*$", re.MULTILINE)


def snapshot_compose(compose: str, db_image: str) -> str:
    """``compose``, the file a deploy wrote, with its servicedb started from ``db_image`` and without the schema init
    script, whose work a snapshot already holds. Every other service stays as the deploy wrote it."""
    if problem := image_reference_problem(db_image):
        raise ValueError(problem)
    swapped, count = _SERVICE_DB_IMAGE.subn(lambda match: match.group(1) + db_image, compose)
    if count != 1:
        raise RuntimeError(f"{DOCKER_COMPOSE_PATH} names a {DATABASE_SERVICE_NAME} image {count} times, not once, so "
                           "the snapshot can't replace it")
    return swapped.replace(f"    volumes:\n      - ./init-schemas.sql:{INIT_SCRIPT_MOUNT_PATH}\n", "")


class MultiEnv(Env):
    type: ClassVar[str] = "multi"
    description = ("Multi environment combining multiple MCP servers, websites and computers, deployed through the environment "
                   "provider its env_provider_type names")
    toml_refs: ClassVar[tuple[EntityRef, ...]] = (EntityRef.env("mcp_server_envs[]", env_type=MCPServerEnv.type),
                                                  EntityRef.env("website_envs[]", env_type=WebsiteEnv.type),
                                                  EntityRef.env("computer_envs[]", env_type=COMPUTER_ENV_TYPE),
                                                  EntityRef.env("browser_envs[]"))
    toml_keys: ClassVar[dict[str, type]] = {"mcp_server_envs": list, "website_envs": list, "computer_envs": list,
                                            "browser_envs": list, "name": str, "env_provider_type": str}

    def __init__(self, id: str, version: Optional[int], mcp_server_envs: list[MCPServerEnv], website_envs: list[WebsiteEnv] | None = None, metadata: Optional[dict[str, str]] = None, name: Optional[str] = None, env_provider_type: str = "gateway",
                 computer_envs: list[ComputerEnv] | None = None, browser_envs: list[Env] | None = None):
        super().__init__(id, version, metadata=metadata)
        if name is not None and (not name or any(ch.isspace() for ch in name)):
            raise ValueError("name must be non-empty with no whitespace")
        if not env_provider_type:
            raise ValueError("env_provider_type cannot be empty")
        self.name = name
        self.env_provider_type = env_provider_type  # the children's own types are ignored: this one deploys them all
        self.mcp_server_envs = mcp_server_envs
        self.website_envs = website_envs or []
        # A live site's names are the real site's, which a synthetic site claiming one of them wins.
        if conflicts := sites.name_conflicts([sites.Claim(name, sites.ALLOWLIST if env.live else sites.SYNTHETIC,
                                                          f"{env.environment_name} at website_envs[{i}]")
                                              for i, env in enumerate(self.website_envs)
                                              for name in dict.fromkeys(env.domains)]):
            raise sites.SiteConflict(f"multi env {id!r}: its websites claim names twice: {'; '.join(conflicts)}")
        self._computers = _multi_computers.Computers(computer_envs or [], owner=id)
        # None leaves the browser to the run, as for every env with websites; [] is none.
        self.browser_envs = list(browser_envs) if browser_envs is not None else None
        problems = browser_problems(self.browser_envs or [])
        if not problems:
            problems = self._name_problems(placed(self.browser_envs[0] if self.browser_envs else None))
        if problems:
            raise ValueError(f"multi env {id!r}: {'; '.join(problems)}")
        self._sandbox = None
        self._gateway_url = None
        self._gateway_mode = GatewayMode.PERFORMANCE
        self._env_provider = None
        self._mcp_server_name: Optional[str] = None
        self._deployed: Optional[DeployedEnv] = None
        self._replaced: list[tuple] = []  # the provider and sandbox of each deployment a later deploy replaced, for close()

    @property
    def computer_envs(self) -> list[ComputerEnv]:
        """Its computers, each on a sandbox of its own, which the gateway fronts by its name: after a deploy, that deploy's."""
        return self._computers.envs

    def _name_problems(self, browser: WebsiteBrowser) -> list[str]:
        """What's wrong with the names of its computers and ``browser``, its website browser, beside its other child
        envs' (``_multi_computers.name_problems``, ``app_name_problems``): a browser computer named as a computer is,
        and an MCP server browser as an MCP server, which no other MCP server may name either."""
        computer, server = browser.computer, browser.server
        computer_envs = [*self.computer_envs, *([computer] if computer is not None else [])]
        mcp_servers = [env.environment_name for env in (*self.mcp_server_envs, *([server] if server else []))]
        websites = [env.environment_name for env in self.website_envs]
        servers = [*mcp_servers, *websites]
        computers = [env.environment_name for env in computer_envs]
        problems = []
        if server is not None and mcp_servers.count(server.environment_name) > 1:
            problems.append(f"website browser {server.environment_name!r} has the name of another of its MCP servers, "
                            "and the gateway runs each as a container of that name")
        return [*problems, *_multi_computers.name_problems(computers, mcp_servers=mcp_servers, websites=websites),
                *_multi_computers.app_name_problems({env.environment_name: [app.environment_name
                                                                            for app in env.composed_apps()
                                                                            if app is not env]  # an app deployed alone
                                                     for env in computer_envs}, [*servers, *computers])]

    def website_browser(self) -> WebsiteBrowser:
        """The website browser a deploy adds, and where it runs: the one ``browser_envs`` names, none when it names
        none, and without it the run's (``website_browser_env_id``) when it has websites, as every env with websites
        gets."""
        if self.browser_envs is not None:
            return placed(self.browser_envs[0] if self.browser_envs else None)
        if not self.website_envs:
            return WebsiteBrowser()
        try:
            return placed(Env.get(website_browser_env_id(get_config())))
        except NotFoundError:
            return WebsiteBrowser(runs=True)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["mcp_server_envs"] = [
            {"id": env.id, "type": env.type, "version": env.version}
            for env in self.mcp_server_envs
        ]
        base["website_envs"] = [
            {"id": env.id, "type": env.type, "version": env.version}
            for env in self.website_envs
        ]
        if self.computer_envs:
            base["computer_envs"] = [{"id": env.id, "type": env.type, "version": env.version} for env in self.computer_envs]
        if self.browser_envs is not None:
            base["browser_envs"] = [{"id": env.id, "type": env.type, "version": env.version}
                                    for env in self.browser_envs]
        if self.name:
            base["name"] = self.name
        base["env_provider_type"] = self.env_provider_type
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "MultiEnv":
        mcp_server_envs = [
            Env.get(ref["id"], version=ref["version"])
            for ref in data.get("mcp_server_envs", [])
        ]
        website_envs = [
            Env.get(ref["id"], version=ref["version"])
            for ref in data.get("website_envs", [])
        ]
        computer_envs = [Env.get(ref["id"], version=ref["version"]) for ref in data.get("computer_envs", [])]
        browser_envs = ([Env.get(ref["id"], version=ref["version"]) for ref in data["browser_envs"]]
                        if data.get("browser_envs") is not None else None)
        return cls(id=data["id"], version=data.get("version"), mcp_server_envs=mcp_server_envs, website_envs=website_envs, metadata=data.get("metadata", {}),
                   name=data.get("name"), env_provider_type=data.get("env_provider_type", "gateway"),
                   computer_envs=computer_envs, browser_envs=browser_envs)

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> MultiEnv:
        """Write the env authored as ``data`` (its env.toml, with ``mcp_server_envs``, ``website_envs``,
        ``computer_envs`` and ``browser_envs`` resolved to env ids) under ``ctx.id`` and return it, unless saving it
        would be refused."""
        fields = cls.accept_toml(data, ctx)
        computer = get_env_registry()[COMPUTER_ENV_TYPE]
        env = dict(mcp_server_envs=[ctx.env(ref, MCPServerEnv) for ref in fields.get("mcp_server_envs", [])],
                   website_envs=[ctx.env(ref, WebsiteEnv) for ref in fields.get("website_envs", [])],
                   computer_envs=[ctx.env(ref, computer) for ref in fields.get("computer_envs", [])],
                   browser_envs=([ctx.env(ref) for ref in fields["browser_envs"]] if "browser_envs" in fields
                                 else None),
                   name=fields.get("name"), env_provider_type=fields["env_provider_type"])
        try:
            refusal = cls(id=ctx.id, version=None, **env).save_refusal()
        except ValueError as e:
            refusal = str(e)
        if refusal:
            ctx.refuse([refusal])
        return cls.put(id=ctx.id, **env)

    def to_toml(self, out: ExportContext) -> dict:
        """The env.toml of a multi env naming its envs, which ``from_toml`` reads back as it."""
        toml = {"type": self.type, "mcp_server_envs": [out.ref(env) for env in self.mcp_server_envs]}
        if self.website_envs:
            toml["website_envs"] = [out.ref(env) for env in self.website_envs]
        if self.computer_envs:
            toml["computer_envs"] = [out.ref(env) for env in self.computer_envs]
        if self.browser_envs is not None:
            toml["browser_envs"] = [out.ref(env) for env in self.browser_envs]
        if self.name:
            toml["name"] = self.name
        return {**toml, **self.provider_toml(out, self.env_provider_type)}

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an env.toml, a multi env takes. Raises BundleError listing every problem."""
        fields, problems = ctx.accepted_env(data, cls)
        if not (fields.get("mcp_server_envs") or fields.get("website_envs") or fields.get("computer_envs")):
            problems.append(ctx.config_problem("a multi env needs at least one env in mcp_server_envs, website_envs or "
                                               "computer_envs"))
        name = fields.get("name")
        if name is not None and (not name or any(ch.isspace() for ch in name)):
            problems.append(ctx.config_problem(f"name must be non-empty with no whitespace, not {name!r}"))
        if problems:
            ctx.refuse(problems)
        return fields

    async def deploy(self, ttl_seconds: int = 10800, disk_size_gb: float = 10, gateway_mode: GatewayMode = GatewayMode.PERFORMANCE, cpu: float | None = None, memory_mb: int | None = None, sandbox_type: str | None = None, env_state_type: str | None = None, env_state_instance_id: str | None = None, artifact_id: str | None = None, artifact_version: int | None = None, *, attribution: Optional[Attribution] = None,
                     child_envs: Optional[dict] = None) -> DeployedEnv:
        browser = self.website_browser()
        if refusal := self._child_envs_refusal(child_envs, sandbox_type, browser):
            raise ValueError(refusal)
        self._computers.begin(browser.computer)
        deployer = self._computers.deployer(self._entries(child_envs, browser.computer), sandbox_type=sandbox_type,
                                            ttl_seconds=ttl_seconds, attribution=attribution, websites=self._websites(),
                                            live_hosts=self.live_hosts())
        computers = dict(children=deployer, then=self._computers.finish) if self._computers.every else {}
        try:
            deployed_env = await deploy_through_provider(
                self, environment_name=None, ttl_seconds=ttl_seconds, sandbox_type=sandbox_type,
                disk_size_gb=disk_size_gb, gateway_mode=gateway_mode, cpu=cpu, memory_mb=memory_mb,
                env_state_type=env_state_type, env_state_instance_id=env_state_instance_id, attribution=attribution,
                artifact_id=artifact_id, artifact_version=artifact_version, **computers,
            )
        except BaseException:
            self._computers.rollback()
            raise
        self._computers.commit()
        self._gateway_mode = gateway_mode
        self._mcp_server_name = deployed_env.mcp_server_name
        _hand_provider_to_child_envs(self)
        _hand_record_to_child_envs(self, deployed_env)
        return deployed_env

    def _entries(self, child_envs: Optional[dict], browser: Optional[ComputerEnv]) -> Optional[dict]:
        """``child_envs``, with ``browser``, its website browser when that is a computer, routing all its synthetic
        websites unless its entry's ``env_params`` say whether it routes them (``site_routing``), as the browser an MCP
        server runs reaches them all."""
        synthetic = any(not env.live for env in self.website_envs)
        if browser is None or not synthetic or not isinstance(child_envs, (dict, type(None))):
            return child_envs
        entry = (child_envs or {}).get(browser.environment_name)
        params = entry.get("env_params") if isinstance(entry, dict) else None
        if not isinstance(entry, (dict, type(None))) or not isinstance(params, (dict, type(None))) or \
                "site_routing" in (params or {}):
            return child_envs
        routing = {**(entry or {}), "env_params": {**(params or {}), "site_routing": True}}
        return {**(child_envs or {}), browser.environment_name: routing}

    def _websites(self) -> dict[str, str]:
        """Its websites' environment names by env id, the ids a computer's ``site_routing`` may name."""
        return {env.id: env.environment_name for env in self.website_envs}

    def _start_tab_names(self, entry: Optional[dict]) -> Optional[list[str]]:
        """The names a computer with ``entry`` may open start tabs at: those of the websites its ``site_routing``
        routes, all of them or those it lists, and every live site's domains and ``allow_hosts``, which the computer
        reaches the real sites at; None while a routed site stores no names, since its card may declare any at
        deploy, which checks the start tabs then."""
        params = entry.get("env_params") if isinstance(entry, dict) else None
        routing = params.get("site_routing") if isinstance(params, dict) else None
        listed = routing if isinstance(routing, list) else None
        routed = [env for env in self.website_envs if not env.live and (listed is None or env.id in listed)]
        if any(not env.domains for env in routed):
            return None
        return [name for env in routed for name in env.domains] + [
            name for env in self.website_envs if env.live for name in [*env.domains, *env.allow_hosts]]

    def live_hosts(self) -> list[str]:
        """The hosts its live websites reach the real sites at, each one's domains and ``allow_hosts``, which a
        computer whose network is an allowlist is let reach; a name one of its synthetic websites' names covers is
        that site's, routed to it, and left out, while a wider one is kept, the synthetic name still routed within
        it."""
        synthetic = [name for env in self.website_envs if not env.live for name in env.domains]
        hosts = []
        for env in (env for env in self.website_envs if env.live):
            for host in dict.fromkeys([*env.domains, *env.allow_hosts]):
                if claimed := next((name for name in synthetic if sites.covers(name, host)), None):
                    logger.warning(f"multi env '{self.id}': live site {env.environment_name!r} names {host!r}, which "
                                   f"the synthetic site at {claimed!r} claims, so the computer reaches that site there")
                elif host not in hosts:
                    hosts.append(host)
        return hosts

    def live_websites(self) -> list[dict]:
        """Its live websites, as a run records them (``WebsiteEnv.live_websites``)."""
        return [site for env in self.website_envs for site in env.live_websites()]

    def save_refusal(self) -> str | None:
        """Why saving the env would be refused, or None: its env_provider_type can't deploy it. How each computer fits a
        sandbox is a deploy's question, as it is for a computer saved alone (``deploy_refusal``)."""
        return deploy_refusal(self, provider_or_class(self), {})

    def deploy_refusal(self, *, sandbox_type: str | None = None, child_envs: Optional[dict] = None, **options) -> str | None:
        """Why deploy() would refuse these options before building anything, or None; raises, as deploy() does, for a type this process can't find.
        ``sandbox_type`` and ``child_envs`` are deploy()'s own, which no provider takes."""
        return (deploy_refusal(self, provider_or_class(self), options)
                or self.child_envs_refusal(child_envs, sandbox_type=sandbox_type))

    def child_envs_refusal(self, child_envs: Optional[dict], *, sandbox_type: str | None = None) -> str | None:
        """Why a deploy would refuse ``child_envs``, each child env's options by its environment name, or None. A
        computer, and its website browser when that is one (``website_browser``), is checked as its entry, or no entry,
        would deploy it on a sandbox of its own, ``sandbox_type`` (the multi env's) unless the entry names one, the
        browser routing its synthetic websites unless its entry says whether it does; any other child env runs on the
        multi env's sandbox and takes no entry."""
        return self._child_envs_refusal(child_envs, sandbox_type, self.website_browser())

    def _child_envs_refusal(self, child_envs: Optional[dict], sandbox_type: str | None,
                            browser: WebsiteBrowser) -> str | None:
        if child_envs and not isinstance(child_envs, dict):
            return f"child_envs is a table of child env names, not {type(child_envs).__name__}"
        computer = browser.computer
        entries = self._entries(child_envs, computer) or {}
        child_envs = child_envs or {}
        computers = {env.environment_name: env for env in [*self.computer_envs, *([computer] if computer else [])]}
        names = list(dict.fromkeys(env.environment_name for env in (*self.mcp_server_envs, *self.website_envs,
                                                                    *computers.values())))
        # A browser browser_envs names had its names checked at put; the run's is checked here.
        problems = self._name_problems(browser) if browser.env is not None and self.browser_envs is None else []
        for name, entry in child_envs.items():
            extra = sorted(map(str, set(entry) - set(CHILD_ENV_OPTIONS))) if isinstance(entry, dict) else []
            if name not in names:
                problems.append(f"child_envs.{name}: env '{self.id}' has no child env named {name!r}; its child envs "
                                f"are {', '.join(names)}")
            elif not isinstance(entry, dict) or extra:
                problems.append(f"child_envs.{name}: an entry is a table of {', '.join(CHILD_ENV_OPTIONS)}, not " +
                                (', '.join(extra) if extra else f"a {type(entry).__name__}"))
            elif name not in computers:
                problems.append(f"child_envs.{name}: {name} runs on the multi env's own sandbox, which the step's own "
                                "fields set, so it takes no entry")
        for name, env in computers.items():
            entry = entries.get(name)
            if entry is None or (isinstance(entry, dict) and set(entry) <= set(CHILD_ENV_OPTIONS)):
                if refusal := _multi_computers.refusal(env, entry or {}, sandbox_type, self._websites()):
                    problems.append(f"child_envs.{name}: {refusal}" if name in child_envs else refusal)
                if (names := self._start_tab_names(entry)) is not None:
                    problems += [f"child_envs.{name}: {problem}"
                                 for problem in _multi_computers.start_tab_problems(env, entry, names)]
        return "; ".join(problems) or None

    @classmethod
    async def from_deployed_env(cls, deployed: DeployedEnv) -> MultiEnv:
        env = await cls._attached(deployed)
        if getattr(deployed, "child_envs", None):
            await env._computers.reattach(deployed)
        return env

    @classmethod
    async def _attached(cls, deployed: DeployedEnv) -> MultiEnv:
        """The multi env ``deployed`` records, attached to it as its loads are: its gateway reattached, and its MCP
        servers and websites handed the provider and the record; its computers aren't reattached."""
        env = Env.get(deployed.env_id, deployed.env_version)
        if not isinstance(env, MultiEnv):
            raise TypeError(f"Expected MultiEnv, got {type(env).__name__}")
        provider = builtin_provider_for(env.env_provider_type) or plugin_provider_like_a_builtin(env.env_provider_type)
        if builtin := as_builtin(provider):
            env._sandbox = await builtin._reattach(env, deployed)
        # Any other plugin's deployment holds its own sandboxes: its children load through its card, and nothing is reattached.
        env._env_provider = provider
        env._gateway_url = gateway_url_of(deployed)
        env._deployed = deployed
        _hand_provider_to_child_envs(env)
        _hand_record_to_child_envs(env, deployed)
        env._instance_id = deployed.instance_id
        env._mcp_server_name = deployed.mcp_server_name
        return env

    async def load_environment_artifact(self, environment_artifact: EnvironmentArtifact) -> None:
        # A built-in's deploy, or a caller wiring one up by hand, leaves a gateway; a plugin's leaves only its record.
        if self._gateway_url is None and self._deployed is None:
            raise RuntimeError("Environment not deployed - call deploy() first")
        # MCP servers are checked first; if a website shares the same environment_name, the MCP server wins.
        for env in self.mcp_server_envs:
            if env.environment_name == environment_artifact.environment_name:
                await env.load_environment_artifact(environment_artifact)
                return
        for env in self.website_envs:
            if env.environment_name == environment_artifact.environment_name:
                await env.load_environment_artifact(environment_artifact)
                return
        if (computer := self._computers.holding(environment_artifact.environment_name)) is not None:
            await computer.load_environment_artifact(environment_artifact)
            return
        logger.warning(f"No env with environment_name '{environment_artifact.environment_name}' — skipping")

    async def load_environment_universe_artifact(
        self,
        environment_universe_artifact: EnvironmentUniverseArtifact,
        snapshot_after_load: bool | None = None,
    ) -> LoadEnvironmentUniverseArtifactResult:
        """Seed this env from a universe artifact, preferring a pre-baked snapshot.

        A snapshot restore swaps a pre-ingested servicedb image; a miss re-ingests every
        service over HTTP under a fixed 600s per-service cap.
        Either way each computer's environments then load into it, in its own order.

        Restore is NOT reliably faster. Measured on a 13-service / ~8.7GB universe:
        restore ~4min (an image pull plus recreating every service container) versus
        ~3.5min to re-ingest on 8 vCPU. What restore buys is a bound — no per-service
        timeout to squeeze under — and far less CPU: the same re-ingest FAILS on the
        1-vCPU sandbox default, where a 409MB service blew its cap while a 3.5GB one
        succeeded. Re-ingest cost tracks record count, not bytes, so the busier the
        universe the better restore looks.

        ``snapshot_after_load`` bakes a reusable snapshot after a successful re-ingest so
        the next load of the same (images, universe) pair can restore instead. It defaults
        to the ``AGENT_ENV_SNAPSHOT_AFTER_LOAD`` config flag (off), because a bake builds
        and pushes a multi-GB servicedb image (~5min, ~2.3GB stored) — worth it to make a
        universe cheap to re-serve, not worth it on a one-off deploy. The bake is
        best-effort and never fails a load that already succeeded.
        """
        import uuid
        from agent_env.env.env import LoadEnvironmentUniverseArtifactResult
        from agent_env.env.snapshot_store import compute_env_fingerprint, get_env_snapshot_store

        # Refused before any service is touched, rather than after they all load.
        if (refusal := host_staging_refusal(self, "Staging a universe's metadata files onto the env's host")) \
                and environment_universe_artifact.get_metadata():
            raise RuntimeError(refusal)

        # Check for a clean snapshot before doing slow REST-based loading.

        # Shortcutting data-load by using a snapshot is only supported when the env_state_type is local_postgres -
        # a remote-backed deploy has no local servicedb container to swap, so it falls through to
        # the normal load. So does a plugin's deployment, which has no state provider of ours.
        state_provider = _gateway_state_provider(self)
        can_restore = state_provider is not None and state_provider.supports_restore_from_snapshot
        # Keyed on the service images rather than self.id, so re-registering the same
        # servers under a new env id still hits (and a rebuilt server correctly misses).
        env_fingerprint = compute_env_fingerprint(self) if can_restore else None
        snapshot = get_env_snapshot_store().get_clean(
            self.id,
            environment_universe_artifact.id,
            environment_universe_artifact.version,
            env_fingerprint=env_fingerprint,
        ) if can_restore else None
        restored_from_snapshot = False
        # The universe's environments, read at most once, and only for a load that needs them: a restore reads none
        # unless the multi env has computers to load.
        environments: Optional[list[EnvironmentArtifact]] = None
        snapshot_baked: bool | None = None
        snapshot_bake_error: str | None = None

        if snapshot:
            logger.info(f"Found clean snapshot for env={self.id} universe={environment_universe_artifact.id}: artifact={snapshot.db_image_artifact_id} v{snapshot.db_image_artifact_version} instance={snapshot.instance_id}")
            await self._load_from_snapshot(snapshot)
            restored_from_snapshot = True
            # A restore swaps the whole database at once, so per-service resume state from
            # any earlier partial re-ingest no longer describes anything real.
            _forget_loaded_environments(self._instance_id)
        else:
            import asyncio
            import time
            # The services' alone: the computers' load after them, in each computer's own order.
            environments = environment_universe_artifact.get_environment_artifacts()
            environment_artifacts = [artifact for artifact in environments
                                     if self._computers.holding(artifact.environment_name) is None]
            total = len(environment_artifacts)
            # WARNING, not info: this is the path that can FAIL. Every service is
            # re-ingested over HTTP under a fixed per-service timeout, and one service
            # exceeding it fails the whole load -- which on an under-provisioned VM is the
            # normal outcome, not the edge case. The fingerprint belongs in the log
            # because "no snapshot was ever baked" and "one exists but the service images
            # changed" look identical from the outside and call for different fixes.
            logger.warning(
                "No clean snapshot for env=%s (fingerprint=%s) universe=%s v%s — "
                "re-ingesting all %d services over HTTP, each under its own timeout.%s",
                self.id, env_fingerprint, environment_universe_artifact.id,
                environment_universe_artifact.version, total,
                " Bake a snapshot to make subsequent loads an image swap instead." if can_restore else "",
            )

            # Bound how many load at once. Unbounded, every service races every other for
            # the same cores, and since each one is separately capped at
            # DATA_PLANE_LOAD_TIMEOUT_S, contention doesn't just slow the load down -- it
            # fails it. Measured on this universe: 13-way on 1 vCPU gives each service
            # ~1/13th of a core and a 409MB service exceeds its 600s cap, while a 3.5GB one
            # finishes; the cost tracks record count, not bytes. Throttling trades total
            # wall-clock (which nothing caps) for per-service headroom (which is capped).
            # Resume an interrupted load instead of re-running all of it. A step retry
            # re-enters this step from scratch, and because the load is destructive
            # (reset-then-add per service) that meant re-wiping and re-ingesting every
            # service -- three attempts x the full universe is how one slow service became
            # a ~44 minute failure.
            already = await self._resumable_environments(environment_universe_artifact, total)
            if already:
                environment_artifacts = [
                    a for a in environment_artifacts if a.environment_name not in already
                ]
                logger.warning(
                    "Resuming an interrupted load of %s v%s: skipping %d already-loaded "
                    "service(s) %s, loading the remaining %d",
                    environment_universe_artifact.id, environment_universe_artifact.version,
                    len(already), sorted(already), len(environment_artifacts),
                )
                total = len(environment_artifacts)
                if not environment_artifacts:
                    logger.info("Every service was already loaded; nothing to re-ingest")

            limit = await self._load_concurrency_limit(total) if total else 1
            semaphore = asyncio.Semaphore(limit)

            async def _load_one(idx: int, artifact):
                name = artifact.environment_name
                async with semaphore:
                    # Timed inside the semaphore: the per-service timeout only starts once
                    # the work does, so queue time must not be charged against it. A
                    # duration here is comparable to DATA_PLANE_LOAD_TIMEOUT_S.
                    t0 = time.monotonic()
                    logger.info(f"[{idx}/{total}] loading {name} ...")
                    print(f"[{idx}/{total}] loading {name} ...", flush=True)
                    try:
                        await self.load_environment_artifact(artifact)
                    except BaseException as e:  # noqa: BLE001 — includes CancelledError
                        dt = time.monotonic() - t0
                        logger.error(f"[{idx}/{total}] FAIL {name} after {dt:.0f}s: {type(e).__name__}: {e}")
                        print(f"[{idx}/{total}] ✗ {name} FAILED after {dt:.0f}s: {type(e).__name__}: {e}", flush=True)
                        raise
                    dt = time.monotonic() - t0
                    # Recorded per service, immediately, rather than once at the end: the
                    # whole point is that a load which dies partway leaves usable progress.
                    self._record_loaded_environment(environment_universe_artifact, name)
                    logger.info(f"[{idx}/{total}] OK {name} in {dt:.0f}s")
                    print(f"[{idx}/{total}] ✓ {name} loaded in {dt:.0f}s", flush=True)
                    return dt

            # Key failures by position, not environment_name: a duplicate service
            # must not let one success mask another's failure.
            outcomes = await asyncio.gather(
                *(_load_one(i, a) for i, a in enumerate(environment_artifacts, 1)),
                return_exceptions=True,
            )
            failed = [
                (i, environment_artifacts[i - 1].environment_name, outcome)
                for i, outcome in enumerate(outcomes, 1)
                if isinstance(outcome, BaseException)
            ]
            if failed:
                raise RuntimeError(
                    f"Failed to load {len(failed)}/{total} services: "
                    + ", ".join(
                        f"[{i}/{total}] {name} ({type(e).__name__}: {e})"
                        for i, name, e in failed
                    )
                )
        if self._computers.every:
            # Every time, never skipped as a resumed service is: the changelog a resume reads doesn't see a computer's
            # files.
            await self._computers.load(environments if environments is not None
                                       else environment_universe_artifact.get_environment_artifacts())
        if self._instance_id:
            env_store.update_env_instance_environment_universe(self._instance_id, environment_universe_artifact.id, environment_universe_artifact.version)

        # Bake the snapshot that makes the NEXT load of this (images, universe) pair an
        # image swap. Deliberately placed here: after the universe stamp, which
        # EnvSnapshot.create reads back off the instance record, and before the metadata
        # download, so the captured PGDATA reflects the data load and nothing else.
        if not restored_from_snapshot and can_restore:
            if snapshot_after_load is None:
                snapshot_after_load = _snapshot_after_load_default()
            if snapshot_after_load:
                snapshot_baked, snapshot_bake_error = await self._bake_snapshot_after_load(
                    environment_universe_artifact
                )

        metadata_filepaths: dict[str, str] = {}
        metadata_artifacts = environment_universe_artifact.get_metadata()
        if metadata_artifacts:
            if self._sandbox is None:
                raise RuntimeError("Environment not deployed - call deploy() first")
            dest_dir = f"/tmp/metadata-{uuid.uuid4().hex[:8]}"
            if self._sandbox.mode != SANDBOX_MODE_CONTAINER:
                await self._sandbox.exec_script(f"mkdir -p {dest_dir}")
            for key, file_artifact in metadata_artifacts.items():
                dest_path = f"{dest_dir}/{file_artifact.filename}"
                logger.info(f"Downloading metadata '{key}' -> {dest_path}")
                if self._sandbox.mode == SANDBOX_MODE_CONTAINER:
                    await self._sandbox.write_file_from_object(file_artifact.object_url, dest_path)
                else:
                    await self._sandbox.load_object_file(file_artifact.object_url, dest_path)
                metadata_filepaths[key] = dest_path
                logger.info(f"Successfully downloaded metadata '{key}'")
        return LoadEnvironmentUniverseArtifactResult(
            metadata_filepaths=metadata_filepaths,
            restored_from_snapshot=restored_from_snapshot,
            snapshot_db_image_artifact_id=snapshot.db_image_artifact_id if snapshot else None,
            snapshot_baked=snapshot_baked,
            snapshot_bake_error=snapshot_bake_error,
        )

    async def _resumable_environments(self, environment_universe_artifact, total: int) -> set[str]:
        """Services safe to skip because a previous attempt already loaded them.

        Gated on the changelog being EMPTY, which is what keeps this from quietly breaking
        the documented contract that loading a universe is destructive. Two cases have to be
        told apart, and the changelog is what distinguishes them:

        * A retry of an interrupted load. Nothing has mutated the data (load triggers are
          installed only after each service's own load), so the changelog is empty and the
          recorded services are still exactly what the artifact says. Skipping is correct
          and saves re-wiping them.
        * A deliberate re-load to reset an env an agent has been working in. The changelog
          is non-empty, so NOTHING is skipped and every service is wiped and re-seeded --
          today's behaviour, which callers depend on.

        Coarse on purpose: the changelog is global, not per-service, so any mutation
        anywhere disables resume entirely. That errs toward doing redundant work rather than
        toward leaving stale data in place, which is the right way to be wrong here.
        """
        if not self._instance_id or self._sandbox is None:
            return set()
        if self._sandbox.mode == SANDBOX_MODE_CONTAINER:
            return set()
        try:
            from agent_env.env.snapshot_store import _check_changelog_empty

            recorded = set(env_store.get_env_instance_store().get_loaded_environments(
                self._instance_id, environment_universe_artifact.id, environment_universe_artifact.version,
            ))
            if not recorded:
                return set()
            if not await _check_changelog_empty(self._sandbox):
                logger.info(
                    "Not resuming: the changelog is non-empty, so this env's data has been "
                    "modified since it was loaded and every service must be re-seeded"
                )
                return set()
            if len(recorded) >= total:
                # A complete prior load with an untouched changelog. Re-seeding would be a
                # no-op, but skipping everything would silently turn a caller's explicit
                # load into nothing -- so do the work rather than guess at intent.
                logger.info(
                    "All %d services are recorded as loaded and the changelog is clean; "
                    "re-loading anyway rather than turning an explicit load into a no-op",
                    total,
                )
                return set()
            return recorded
        except Exception as e:  # noqa: BLE001 -- an optimisation, not a dependency
            logger.warning("Could not determine resumable services (%s); loading all of them", e)
            return set()

    def _record_loaded_environment(self, environment_universe_artifact, environment_name: str) -> None:
        """Persist one service's completion. Best-effort: losing a record costs redundant
        work on a retry, whereas raising here would fail a service that actually loaded."""
        if not self._instance_id:
            return
        try:
            env_store.get_env_instance_store().record_loaded_environment(
                self._instance_id, environment_universe_artifact.id,
                environment_universe_artifact.version, environment_name,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not record %s as loaded (%s); a retry will re-load it",
                environment_name, e,
            )

    async def _load_concurrency_limit(self, total: int) -> int:
        """How many services may load at once.

        Derived from the sandbox's actual core count rather than a constant, because the
        failure this prevents is contention-driven: the per-service data-plane timeout is
        fixed, so what matters is how much CPU each concurrent load can get. Two in flight
        per core keeps some I/O overlap while leaving each service a real share.

        At >= ceil(total/2) cores this is a no-op -- a well-provisioned VM keeps loading
        everything at once, which is what the measured 8 vCPU run did. It only bites on the
        small boxes where unbounded concurrency was turning a slow load into a failed one.

        ``AGENT_ENV_LOAD_CONCURRENCY`` overrides it outright, for pinning the old behaviour
        or bisecting a contention problem. Falls back to loading everything at once if the
        core count can't be read: that is today's behaviour, so a probe failure must not
        quietly change how a load runs.
        """
        override = os.environ.get(_LOAD_CONCURRENCY_ENV_VAR, "").strip()
        if override:
            try:
                value = int(override)
                if value < 1:
                    raise ValueError(f"must be >= 1, got {value}")
            except ValueError as e:
                # Loud, not silent: a typo'd override that fell back to the default would
                # make a contention experiment quietly measure the wrong thing.
                raise ValueError(
                    f"{_LOAD_CONCURRENCY_ENV_VAR}={override!r} is not a positive integer: {e}"
                ) from None
            limit = min(value, total)
            logger.info("Load concurrency %d/%d (from %s)", limit, total, _LOAD_CONCURRENCY_ENV_VAR)
            return limit

        cores = None
        if self._sandbox is not None:
            try:
                cores = int((await self._sandbox.exec_script(_CORE_COUNT_PROBE)).strip())
            except Exception as e:  # noqa: BLE001 — a probe, not a dependency
                logger.warning("Could not read sandbox core count (%s); loading all %d services at once", e, total)
        if not cores or cores < 1:
            return total

        limit = max(1, min(total, cores * 2))
        logger.info(
            "Load concurrency %d/%d services (sandbox has %d core(s))", limit, total, cores,
        )
        return limit

    async def _bake_snapshot_after_load(
        self, environment_universe_artifact: EnvironmentUniverseArtifact
    ) -> tuple[bool, str | None]:
        """Capture a reusable clean snapshot of what was just loaded.

        Best-effort by contract: the data load already succeeded, and the caller wants a
        seeded env more than it wants a cache entry. Any failure here is logged and
        reported in the result, never raised — otherwise adding a cache would turn
        working loads into failures.

        Returns ``(baked, error)``.
        """
        import asyncio

        from agent_env.env.snapshot_store import EnvSnapshot

        if not self._instance_id:
            # create() resolves the env and its loaded universe from the instance record,
            # so an unregistered env (ad-hoc/local deploy) has nothing to key a snapshot on.
            msg = "no instance_id on this env; snapshot capture needs a registered env instance"
            logger.warning("Skipping snapshot bake for env=%s: %s", self.id, msg)
            return False, msg
        try:
            logger.info(
                "Baking clean snapshot for env=%s universe=%s v%s (instance=%s) so the "
                "next load can skip the re-ingest...",
                self.id, environment_universe_artifact.id,
                environment_universe_artifact.version, self._instance_id,
            )
            snapshot = await EnvSnapshot.create(self._instance_id)
        except asyncio.CancelledError:
            # Cancellation is control flow, not a bake failure. Swallowing it here would
            # let a cancelled/timed-out activity carry on and report success.
            raise
        except Exception as e:  # noqa: BLE001 — a cache miss must not fail a good load
            logger.warning(
                "Snapshot bake failed for env=%s universe=%s: %s: %s (the load itself "
                "succeeded; the next load will take the slow path)",
                self.id, environment_universe_artifact.id, type(e).__name__, e,
                exc_info=True,
            )
            return False, f"{type(e).__name__}: {e}"
        if not snapshot.is_clean:
            # create() derives is_clean from an empty changelog. Landing here means
            # something mutated the data between load and capture, so the snapshot exists
            # but get_clean will never serve it — say so rather than reporting success.
            msg = (
                "captured snapshot is dirty (changelog non-empty), so it will not be "
                "reused; something wrote to the env between load and capture"
            )
            logger.warning("Snapshot bake for env=%s: %s", self.id, msg)
            return False, msg
        logger.info(
            "Baked clean snapshot for env=%s universe=%s v%s: image=%s v%s",
            self.id, environment_universe_artifact.id, environment_universe_artifact.version,
            snapshot.db_image_artifact_id, snapshot.db_image_artifact_version,
        )
        return True, None

    async def load_file_artifact_universe(self, file_artifact_universe: "Any", destination_path: Optional[str] = None,
                                          ) -> "LoadFileArtifactUniverseResult":
        if refusal := host_staging_refusal(self, "Staging files onto the env's host"):
            raise RuntimeError(refusal)
        return await super().load_file_artifact_universe(file_artifact_universe, destination_path)

    async def load_artifact(self, artifact):
        from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
        if isinstance(artifact, EnvironmentUniverseArtifact):
            return await self.load_environment_universe_artifact(artifact)
        if isinstance(artifact, EnvironmentArtifact):
            return await self.load_environment_artifact(artifact)
        raise ValueError(f"{type(self).__name__} '{self.id}' cannot load artifact '{getattr(artifact, 'id', '?')}' of type '{getattr(artifact, 'type', '?')}' — expected a EnvironmentArtifact or EnvironmentUniverseArtifact")

    async def _load_from_snapshot(self, snapshot) -> None:
        """Swap the servicedb container with a pre-loaded snapshot image. Only supported when both the current deploy and the snapshot has env_state_type set to local_postgres.

        The compose file the deploy wrote is edited in place, not rendered again, so every other service, the website
        browser included, keeps the image and name it was deployed with whatever config says now."""
        import asyncio
        from agent_env.artifact import Artifact, DockerImageArtifact
        from agent_env.env.env import Env
        from agent_env.env.envs.service_db import DB_USER
        from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
        from agent_env.providers import EnvironmentGatewayProvider
        from agent_env.providers.env_providers.constants import GATEWAY_APP_DIR
        from agent_env.providers.env_state import LocalPostgresStateProvider
        from agent_env.config import get_config, service_db_env_id

        if self._sandbox is None:
            raise RuntimeError("Environment not deployed - call deploy() first")
        if self._sandbox.mode == SANDBOX_MODE_CONTAINER:
            raise NotImplementedError("Snapshot loading not supported on container mode")
        db_image = Artifact.get(snapshot.db_image_artifact_id, snapshot.db_image_artifact_version)
        if not isinstance(db_image, DockerImageArtifact):
            raise RuntimeError(f"Snapshot artifact {snapshot.db_image_artifact_id} is not a DockerImageArtifact")
        logger.info(f"Loading snapshot image {db_image.image_name} into sandbox...")
        await self._sandbox.load_docker_images([db_image])

        logger.info("Pointing docker-compose.yml's servicedb at the snapshot image...")
        # The write adds back the newline cat returns, so the file doesn't grow a blank line each reload.
        deployed = (await self._sandbox.exec_script(f"cat {DOCKER_COMPOSE_PATH}")).removesuffix("\n")
        compose = snapshot_compose(deployed, db_image.image_name)
        gw = EnvironmentGatewayProvider()
        gw._sandbox = self._sandbox
        # The state store is the deploy's local Postgres; the changelog triggers below run through it.
        gw._state_provider = LocalPostgresStateProvider(service_db_config=Env.get(service_db_env_id(get_config())).to_config())
        gw._state_instance = LocalPostgresStateProvider.default_instance()
        await gw._write_file(self._sandbox, DOCKER_COMPOSE_PATH, compose)

        # Remove old servicedb container and its anonymous volume, then start fresh from snapshot image
        logger.info("Recreating servicedb container with snapshot image...")
        await self._sandbox.exec_script(
            f"cd {GATEWAY_APP_DIR} && docker compose rm -sf -v {DATABASE_SERVICE_NAME} && docker compose up -d {DATABASE_SERVICE_NAME} 2>&1"
        )

        # Wait for servicedb to be healthy
        logger.info("Waiting for servicedb to be healthy...")
        for i in range(60):
            try:
                await self._sandbox.exec_script(
                    f"docker compose -f {DOCKER_COMPOSE_PATH} exec -T {DATABASE_SERVICE_NAME} pg_isready -U {DB_USER}",
                )
                logger.info(f"servicedb healthy after {i + 1}s")
                break
            except Exception:
                if i % 10 == 0:
                    logger.info(f"Still waiting for servicedb... ({i}s)")
                await asyncio.sleep(1)
        else:
            raise RuntimeError("servicedb not healthy after 60s")

        # Recreate every other service the compose file runs so it reconnects to the new servicedb.
        # Uses `up -d --force-recreate` which respects depends_on ordering
        # (MCP servers start before gateway), unlike `restart` which is unordered.
        environment_names = list(dict.fromkeys(
            [env.environment_name for env in self.mcp_server_envs] + [env.environment_name for env in self.website_envs]
        ))
        # BaseLoader keeps every name a string: safe_load reads a service named `on` as True and `123` as an int.
        svc_list = " ".join(map(shlex.quote, (name for name in yaml.load(compose, Loader=yaml.BaseLoader)["services"]
                                              if name != DATABASE_SERVICE_NAME)))
        logger.info(f"Recreating services: {svc_list}")
        await gw._compose_up(self._sandbox, f"--force-recreate {svc_list}")
        await gw._wait_for_gateway(self._sandbox, self._sandbox.host_port(AGENT_ENV_GATEWAY_MCP_PORT))

        # Install changelog triggers for each service
        for name in environment_names:
            await gw.install_changelog_triggers(name)

    async def validate(self, on_progress: Callable[[str], None] | None = None) -> str:
        """Run the basic env-card validator task and return the task instance ID.

        Deploys the env, fetches + persists its composed EnvironmentCard, then tears down.
        """
        from agent_env.task import Task
        from agent_env.task_step import DeployEnvTaskStep, VerifyEnvironmentCardStep

        task_id = derive_id(self.id, f"validate-v{self.version}")
        task = Task.put(
            id=task_id,
            steps=[
                DeployEnvTaskStep(id=f"{task_id}-deploy", version=None, env_id=self.id, env_version=self.version),
                VerifyEnvironmentCardStep(id=f"{task_id}-envcard", version=None, env_id=self.id),
            ],
        )
        logger.info(f"Created validation task: {task.id} version={task.version}")

        log = on_progress or (lambda msg: None)
        on_start = lambda i, total, step, ctx: log(f"Running step [{i+1}/{total}]: {step.type}...")
        on_complete = lambda i, total, step, ctx, dur: log(f"Completed step [{i+1}/{total}]: {step.type} [{dur:.1f}s]")
        context = await task.run(on_step_start=on_start, on_step_complete=on_complete)

        for deployed_env in context.deployed_envs:
            try:
                await close_deployed(deployed_env, MultiEnv)
            except Exception as e:
                logger.warning(f"Failed to clean up env {deployed_env.env_id}: {e}")
        return context.instance_id

    async def validate_universe_compatibility(self, universe_artifact_id: str, universe_artifact_version: int | None = None, on_progress: Callable[[str], None] | None = None) -> str:
        """Run universe compatibility validation and return the task instance ID.

        Two verdicts are produced: the programmatic verifier (``compare_dicts``) and an LLM
        agent-judge that diffs the original/export universes for *semantic* (major) differences the
        programmatic comparison misses. The round-trip step bundles original/export1/export2 into a
        FileArtifactUniverse, which is staged onto a judge agent. The judge's flagged differences are
        merged into the ``UNIVERSE_COMPATIBILITY`` result as per-service critical issues — so they
        surface (and gate ``compatible``) the same way programmatic issues do.
        """
        from agent_env.task import Task
        from agent_env.task_step import (
            CombineUniverseVerdictsStep,
            DeployAgentTaskStep,
            DeployEnvTaskStep,
            LoadArtifactTaskStep,
            PromptAgentTaskStep,
            RubricsVerifierTaskStep,
            VerifyUniverseLoadExportRoundtripStep,
        )
        from agent_env.task_step.task_steps.multienv_validator.verify_universe_agent_judge import (
            JUDGE_SYSTEM_PROMPT,
            UNIVERSES_DIR,
            build_criteria,
            build_user_prompt,
        )
        from agent_env.artifact import EnvironmentUniverseArtifact

        universe = EnvironmentUniverseArtifact.get(universe_artifact_id, universe_artifact_version)
        task_id = VerifyUniverseLoadExportRoundtripStep.validation_id(
            "validate-universe-compat", self.id, self.version, universe.id, universe.version
        )
        environment_names = [sa.environment_name for sa in universe.get_environment_artifacts()]
        fau_id = VerifyUniverseLoadExportRoundtripStep.file_artifact_universe_id(self.id, self.version, universe.id, universe.version)
        judge_name = "universe-judge"
        judge_prompt_id = f"{task_id}-judge-prompt"
        judge_verifier_id = f"{task_id}-judge-rubric"
        # Steps run sequentially (depends_on=None ⇒ depends on all prior steps), so the FAU exists
        # before it is loaded, and the judge is prompted before it is graded. The round-trip step
        # computes the programmatic verdict but does NOT persist it (persist_result=False); the final
        # CombineUniverseVerdictsStep is the single authoritative writer, so a crash before the judge
        # merges can't leave an ungated half-written result.
        steps = [
            DeployEnvTaskStep(id=f"{task_id}-deploy", version=None, env_id=self.id, env_version=self.version),
            VerifyUniverseLoadExportRoundtripStep(id=f"{task_id}-roundtrip", version=None, env_id=self.id, universe_artifact_id=universe.id, universe_artifact_version=universe.version, emit_file_artifact_universe=True, persist_result=False),
            DeployAgentTaskStep(id=f"{task_id}-judge-deploy", version=None, agent_name=judge_name, system_prompt=JUDGE_SYSTEM_PROMPT, ttl_seconds=1800),
            LoadArtifactTaskStep(id=f"{task_id}-judge-load", version=None, agent_name=judge_name, artifact_id=fau_id, destination_path=UNIVERSES_DIR),
            PromptAgentTaskStep(id=f"{task_id}-judge-prompt", version=None, agent_name=judge_name, prompt=build_user_prompt(), prompt_id=judge_prompt_id),
            RubricsVerifierTaskStep(id=judge_verifier_id, version=None, criteria=build_criteria(environment_names), prompt_id=judge_prompt_id, agent_name=judge_name, verifier_id=judge_verifier_id),
            CombineUniverseVerdictsStep(id=f"{task_id}-combine", version=None, env_id=self.id, universe_artifact_id=universe.id, universe_artifact_version=universe.version, judge_verifier_id=judge_verifier_id),
        ]
        task = Task.put(id=task_id, steps=steps)
        logger.info(f"Created validation task: {task.id} version={task.version}")

        log = on_progress or (lambda msg: None)
        on_start = lambda i, total, step, ctx: log(f"Running step [{i+1}/{total}]: {step.type}...")
        on_complete = lambda i, total, step, ctx, dur: log(f"Completed step [{i+1}/{total}]: {step.type} [{dur:.1f}s]")
        context = await task.run(on_step_start=on_start, on_step_complete=on_complete)

        for deployed_env in context.deployed_envs:
            try:
                await close_deployed(deployed_env, MultiEnv)
            except Exception as e:
                logger.warning(f"Failed to clean up env {deployed_env.env_id}: {e}")
        # Tear down any judge agent (ttl is a backstop if this is skipped).
        for deployed_agent in context.deployed_agents:
            if deployed_agent.sandbox_id:
                try:
                    from agent_env.providers import build_sandbox_provider, get_agent_sandbox_provider
                    provider = build_sandbox_provider(deployed_agent.sandbox_type) if deployed_agent.sandbox_type else get_agent_sandbox_provider()
                    sandbox = await provider.get_sandbox(deployed_agent.sandbox_id)
                    await sandbox.terminate()
                except Exception as e:
                    logger.warning(f"Failed to clean up agent sandbox {deployed_agent.sandbox_id}: {e}")

        # Note: the merged UNIVERSE_COMPATIBILITY result is written by CombineUniverseVerdictsStep
        # (the final task step), so it persists wherever the Task runs — not only via this method.
        return context.instance_id

    async def reset(self, deployed: DeployedEnv) -> None:
        """Reset every child env at once, each as it resets alone: a computer reboots from its own record and gets its
        ``tools`` again, and an MCP server or website is emptied of what loads put in, so a task loads again after; one
        serving no reset takes no load, and stays as it is. A child that fails to reset fails the reset, each such one
        named, once the others have reset. Before any server is emptied, which services the instance's loads put in is
        forgotten, so the next load puts each in again rather than resuming past one; a reset that can't forget it
        empties nothing."""
        services = self
        if self.mcp_server_envs or self.website_envs:
            services = await type(self)._attached(deployed)
            if deployed.instance_id:
                env_store.get_env_instance_store().clear_loaded_environments(deployed.instance_id)
        resets = [*((f"computer {name!r}", call) for name, call in self._computers.resets(deployed)),
                  *((f"MCP server {env.environment_name!r}", reset_server_data(env, mcp=True))
                    for env in services.mcp_server_envs),
                  *((f"website {env.environment_name!r}", reset_server_data(env, mcp=False))
                    for env in services.website_envs)]
        results = await _multi_computers.each(self.id, "reset", resets)
        for (child, _), result in zip(resets, results):
            if result is False:
                logger.info(f"multi env '{self.id}': {child} serves no reset and takes no load, so it stays as it is")

    async def close(self) -> None:
        await self._computers.close()
        await close_replaced(self)
        if self._env_provider is not None:
            try:
                await self._env_provider.close()
                self._env_provider = None
            except Exception as e:
                logger.warning(f"Failed to close env provider, kept for the next close(): {e}")
        if self._sandbox is not None:
            try:
                await self._sandbox.terminate()
            except Exception as e:  # a cancellation propagates, and keeps the sandbox for the next close()
                logger.warning(f"Failed to terminate sandbox {self._sandbox.sandbox_id}: {e}")
            self._sandbox = None


def _forget_loaded_environments(instance_id: Optional[str]) -> None:
    """Forget which services the instance's loads put in, so its next load puts each in again rather than resuming;
    best-effort, since only a resumed load reads it."""
    if not instance_id:
        return
    try:
        env_store.get_env_instance_store().clear_loaded_environments(instance_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not forget which services instance {instance_id!r} loaded; a resumed load may skip "
                       f"one: {e}")


def _hand_provider_to_child_envs(env: MultiEnv) -> None:
    """Point each child env at the MultiEnv's deployment: the provider and URL its loads use, and behind our gateway the sandbox they
    stage into. A plugin's provider leaves them no sandbox, so they load through the record's card."""
    provider = env._env_provider
    builtin = as_builtin(provider)
    for child in env.mcp_server_envs:
        child._sandbox = (builtin.environment_sandbox(child.environment_name) if builtin else None) or env._sandbox
        child._gateway_url = env._gateway_url
        child._env_provider = provider
    for child in env.website_envs:
        child._sandbox = env._sandbox
        child._gateway_url = env._gateway_url
        child._env_provider = provider


def _gateway_state_provider(env: MultiEnv) -> DatabaseStateProvider | None:
    """The state provider behind our gateway, which a snapshot restores into or captures from; None for a deployment without it."""
    from agent_env.providers import EnvironmentGatewayProvider
    return env._env_provider._state_provider if isinstance(env._env_provider, EnvironmentGatewayProvider) else None


def _hand_record_to_child_envs(env: MultiEnv, deployed: DeployedEnv) -> None:
    """Give each child env the record its loads read the card from; a name an MCP server and a website share keeps the live probe, since the card can't tell them apart by name."""
    shared = {c.environment_name for c in env.mcp_server_envs} & {c.environment_name for c in env.website_envs}
    for child in [*env.mcp_server_envs, *env.website_envs]:
        child._deployed = None if child.environment_name in shared else deployed
