"""The computer env: a base computer and a driver, named by the env with what makes it itself, checked against each
other at put (gate 1) and composed into an image for each form the base has and each arch, each recorded in its
metadata as it builds (``composed``), and the arches built written as its computer (``computer``)."""

from __future__ import annotations

import asyncio
import logging
import posixpath
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self

from agentenv_protocol import METHOD_ADD, RPC_PATH
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer.tools import COMPUTER_TOOLS
from agentenv_protocol.tool_access import (DISABLE, DISABLE_TOOL_EXTENSION_URI, ENABLE, ENABLE_TOOL_EXTENSION_URI,
                                           WILDCARD)
from agentenv_protocol.tool_context import DEFAULT_ROLE

from agent_env.artifact.artifacts.computer import ComputerArtifact, SettingValue
from agent_env.attribution import Attribution
from agent_env.config import get_config
from agent_env.entity.refs import EntityRef
from agent_env.env.env import Env, LoadFileArtifactUniverseResult, gateway_url_of
from agent_env.env.envs._deployment import (
    as_builtin,
    builtin_provider_for,
    close_replaced,
    deploy_refusal,
    deploy_through_provider,
    provider_or_class,
)
from agent_env.env.envs.computer import COMPUTER_ENV_TYPE
from agent_env.env.envs.computer.utils import boot, composer, gate, inside, sites
from agent_env.env.envs.computer.utils.composed import stale
from agent_env.env.envs.computer.utils.install import for_os, platform_of
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.envs.computer.utils import params as computer_params
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
from agent_env.env.gateway import GatewayMode
from agent_env.env.store import get_env_store, update_env_instance_metadata
from agent_env.store.local_state import awaiting_lock
from agent_env.utils.progress import current_progress
from agent_env.providers.env_providers.env_computer_provider import end_recordings
from agent_env.utils.paths import validate_relative_filename

if TYPE_CHECKING:
    from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.env.env import DeployedEnv
    from agent_env.env.envs.computer.app_env import ComputerAppEnv
    from agent_env.providers.env_providers.env_provider import EnvironmentProvider
    from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

logger = logging.getLogger(__name__)

#: Where load_artifact puts files in a computer by default: one of the folders data/get returns by default.
DEFAULT_DESTINATION = "Downloads"
#: The computer app env's type: an app composed into a computer env, and a computer env of its own.
APP_TYPE = "computer_app"
#: The statuses of a build a deploy boots, by form: a disk may also boot unprobed, as KVM is where a disk is probed, and
#: so may a VM bundle, which a Mac probes.
_DEPLOYABLE = {"container": ("ready",), "disk": ("ready", "unverified"), "vm_bundle": ("ready", "unverified")}
#: Every ``computer_*`` tool a driver can serve.
#: What a computer env's ``tools`` gives its agents, as the tool-access changes a deploy makes for them: ``standard``,
#: AgentEnv's ``computer_*`` tools alone; ``driver``, the driver's own alone; ``both``, every tool it serves.
TOOLS = {"standard": ((DISABLE, [WILDCARD]), (ENABLE, list(COMPUTER_TOOLS))), "driver": ((DISABLE, list(COMPUTER_TOOLS)),),
         "both": ()}
_TOOL_ACCESS_URIS = {DISABLE: DISABLE_TOOL_EXTENSION_URI, ENABLE: ENABLE_TOOL_EXTENSION_URI}


class ComputerEnv(Env):
    type: ClassVar[str] = COMPUTER_ENV_TYPE
    description = "A computer: a base computer and a driver, composed at put into an image for each form and arch"
    env_provider_type: ClassVar[str] = "computer"
    toml_refs: ClassVar[tuple[EntityRef, ...]] = (EntityRef.artifact("computer", artifact_type="computer"),
                                                  EntityRef.env("driver", env_type=ComputerDriverEnv.type),
                                                  EntityRef.env("apps[]", env_type=APP_TYPE,
                                                                beside=("install", "settings")))

    def __init__(self, id: str, version: Optional[int], environment_name: str, computer: ComputerArtifact,
                 driver: ComputerDriverEnv, display: Optional[dict] = None, network: Optional[dict] = None,
                 computer_settings: Optional[dict[str, SettingValue]] = None, metadata: Optional[dict[str, Any]] = None,
                 apps: Optional[list[ComputerAppEnv]] = None, app_settings: Optional[dict[str, dict]] = None,
                 tools: Optional[str] = None):
        super().__init__(id, version, metadata=metadata)
        self.environment_name = environment_name
        self.computer, self.driver, self.apps = computer, driver, apps or []
        self.display, self.network, self.tools = display, network, tools
        self.computer_settings = computer_settings or {}
        self.app_settings = app_settings or {}
        self._sandbox = None
        self._env_provider: Optional[EnvironmentProvider] = builtin_provider_for(self.env_provider_type)
        self._gateway_url = None
        self._deployed: Optional[DeployedEnv] = None
        self._replaced: list[tuple] = []  # the provider and sandbox of each deployment a later deploy replaced, for close()

    @property
    def computer_name(self) -> str:
        """The name the computer's own card, and the data plane of its own files, go by: its environment_name."""
        return self.environment_name

    def to_dict(self) -> dict:
        return {**super().to_dict(), "service_name": self.environment_name,
                "environment_name": self.environment_name,  # dual-write with service_name during the rename
                "computer": {"id": self.computer.id, "version": self.computer.version},
                "driver": {"id": self.driver.id, "version": self.driver.version}, "display": self.display,
                "network": self.network, "computer_settings": self.computer_settings,
                **({"tools": self.tools} if self.tools else {}),
                **({"apps": [{"id": app.id, "version": app.version,
                              **({"settings": own} if (own := self.app_settings.get(app.environment_name)) else {})}
                             for app in self.apps]} if self.apps else {})}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerEnv:
        driver = Env.get(data["driver"]["id"], data["driver"]["version"])
        if not isinstance(driver, ComputerDriverEnv):
            raise TypeError(f"{data['driver']['id']} is a {driver.type} env, not a computer_driver env")
        refs = data.get("apps") or []
        apps = [_app(Env.get(ref["id"], ref["version"])) for ref in refs]
        return cls(id=data["id"], version=data.get("version"),
                   environment_name=data["environment_name"] if "environment_name" in data else data["service_name"],
                   computer=ComputerArtifact.get(data["computer"]["id"], data["computer"]["version"]), driver=driver,
                   display=data.get("display"), network=data.get("network"),
                   computer_settings=data.get("computer_settings"), metadata=data.get("metadata") or {}, apps=apps,
                   app_settings={app.environment_name: ref["settings"] for app, ref in zip(apps, refs)
                                 if ref.get("settings")}, tools=data.get("tools"))

    @classmethod
    def put(cls, *, id: str, computer: ComputerArtifact, driver: ComputerDriverEnv, environment_name: str,
            apps: Any = (), display: Any = None, network: Any = None, computer_settings: Any = None,
            app_settings: Optional[dict[str, dict]] = None, tools: Optional[str] = None, compose: bool = True,
            composing: Composing = Composing()) -> Self:
        """Write a computer env of ``computer``, ``driver`` and ``apps``, computer app envs built into it in their
        order, with ``app_settings``, defaults for some of their settings by app name, and ``tools``, which of them its
        agents get (``TOOLS``), once gate 1 passes, then compose it as ``composing`` says unless ``compose`` is False.
        Refused, naming every mismatch, before anything is stored."""
        apps, app_settings = list(apps), app_settings or {}
        values, problems = gate.check(computer, driver, environment_name=environment_name, apps=apps, display=display,
                                      network=network, computer_settings=computer_settings or {})
        problems += gate.check_screen(driver, apps) + _tools_problems(tools)
        by_name = {app.environment_name: app for app in apps}
        for name, defaults in app_settings.items():
            problems += ([f"apps.{name}.settings: {id} has no app {name!r}"] if name not in by_name else
                         gate.check_app_settings(by_name[name], defaults, f"apps.{name}.settings",
                                                 platform_of(computer.os), partial=True))
        if problems:
            raise ValueError(f"computer env {id!r}: " + "; ".join(problems))
        env = get_env_store().put_document(cls(
            id=id, version=None, environment_name=environment_name, computer=computer, driver=driver, apps=apps,
            display=display, network=network, computer_settings=values, app_settings=app_settings, tools=tools))
        if compose:
            composer.compose(env, composing=composing)
        return env

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> ComputerEnv:
        """Write the computer env an env.toml declares, its base, driver and apps as the bundle wrote them or the ones
        they name, its environment_name the folder's when it sets none. An app's ``install`` is ``"build"``, the
        default: it's built into the computer."""
        fields = cls.accept_toml(data, ctx)
        base, driver = ctx.artifact(fields.pop("computer"), ComputerArtifact), ctx.env(fields.pop("driver"),
                                                                                       ComputerDriverEnv)
        fields["apps"], fields["app_settings"] = _listed_apps(fields.get("apps", []), ctx)
        try:
            return cls.put(id=ctx.id, computer=base, driver=driver, composing=ctx.composing, **fields)
        except ValueError as e:
            ctx.refuse([str(e)])

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an env.toml, a computer env takes: the ``computer`` and ``driver`` it's composed of,
        both required, its ``apps``, ``display``, ``network``, ``computer_settings`` and ``tools``, and
        ``environment_name``, the folder's when it sets none. Raises BundleError listing every problem."""
        fields, refused = ctx.accepted(data, environment_name=str, computer=object, driver=object, apps=list,
                                       display=dict, network=dict, computer_settings=dict, tools=str)
        missing = [ctx.config_problem(f"{key} is required: the {what}, such as {example}") for key, what, example in (
            ("computer", "base computer", '"ubuntu-24.04"'), ("driver", "driver", '"reference"')) if key not in data]
        refused += [ctx.config_problem(problem) for problem in _tools_problems(fields.get("tools"))]
        if refused or missing:
            ctx.refuse([*refused, *missing])
        return {"environment_name": ctx.name, **fields}

    def serves_website_browser(self) -> bool:
        """Whether a website browser is among its apps, whose CDP address an agent reaches by WebSocket."""
        return any(app.declaration.browser for app in self.composed_apps())

    def composed_apps(self) -> list[ComputerAppEnv]:
        """The computer app envs composed into the env's computer, each installed under its prefix."""
        return self.apps

    def shm_mb(self) -> int | None:
        """The shared memory its computer's ``/dev/shm`` needs, in MiB: the most any of its apps' installs on its OS
        declares, or None when none declares any."""
        sizes = [size for app in self.composed_apps()
                 if (size := getattr(for_os(app.install, self.computer.os), "shm_mb", None))]
        return max(sizes, default=None)

    def app_defaults(self) -> dict[str, dict[str, Any]]:
        """Each composed app's default settings, by name: its own env's ``[settings]`` under this env's for it, field by
        field."""
        return {app.environment_name: {**app.settings, **self.app_settings.get(app.environment_name, {})}
                for app in self.composed_apps()}

    def computer_values(self) -> dict[str, Any]:
        """The env's values of its base's settings: its ``computer_settings``, and its display as the resolution."""
        display = self.display
        return {**self.computer_settings, **({"resolution": f"{display['width']}x{display['height']}"} if display else {})}

    def composed_arches(self) -> list[str]:
        """The arches the env composes: those its base, driver and apps all run on."""
        return gate.arches(self.computer, self.driver, self.composed_apps())

    def computer_artifact(self) -> ComputerArtifact:
        """The env's computer: its base composed with its driver, over the images built."""
        return ComputerArtifact.get(self.metadata["computer"]["id"], self.metadata["computer"]["version"])

    async def compose_unstarted(self, sandbox_provider: SandboxProvider, form: Optional[str] = None) -> None:
        """Compose first what a deploy on ``sandbox_provider`` boots that no compose started, or a container it runs
        natively that was built but not probed (``composer.unprobed``), once for deploys at once,
        on the env's default sandbox provider, this machine unless the config names another, never on the deploy's,
        unless the run's fence would refuse the record (rules in ``computers/README.md``). A cancelled deploy holds the
        lock until its compose's thread, which no cancel stops, ends, so the next deploy waits for what it builds."""
        if get_config().local_run_refuses_write(self.stored_id):
            return
        if not composer.unstarted_or_building(self, sandbox_provider, form):
            return
        say, progress = _telling(self.id, current_progress())
        awaited: dict[tuple[str, str], str | None] = {}

        async def waiting(seconds: float) -> None:
            self.metadata = (await asyncio.to_thread(Env.get, self.stored_id, self.version)).metadata
            composed = self.metadata.get("composed", {})
            awaited.update({(name, arch): composed[name][arch].get("started_at") for name, arch in
                            composer.unstarted_or_building(self, sandbox_provider, form) if arch in composed.get(name, {})})
            say(_waiting(awaited, seconds))

        async with awaiting_lock(f"computer-compose:{self.stored_id}:{self.version}", waiting, composer.HEARTBEAT_S):
            self.metadata = (await asyncio.to_thread(Env.get, self.stored_id, self.version)).metadata
            if pairs := composer.unstarted(self, sandbox_provider, form):
                them = "it" if len(pairs) == 1 else "them"
                on = Composing(on_progress=progress)
                composed = self.metadata.get("composed", {})
                booting, building = (" or ".join(type(link).__name__ for link in provider.links)
                                     for provider in (sandbox_provider, on.provider()))
                records = [composed.get(name, {}).get(arch) for name, arch in pairs]
                why = (f"an agent-env that composes differently built {them}" if any(map(stale, records))
                       else f"no compose probed {them} on {booting} yet" if all(records)
                       else f"no compose built {them} for {booting} yet")
                say(f"composing {_pairs(pairs)} before this deploy on {building} ({why}); later deploys reuse {them}")
                composing = asyncio.ensure_future(asyncio.to_thread(composer.compose, self, pairs, on))
                try:
                    await asyncio.shield(composing)
                except asyncio.CancelledError:
                    await asyncio.wait({composing})
                    raise
            elif awaited:
                composed = self.metadata["composed"]
                say("reusing what another deploy composed: " + ", ".join(
                    f"{name}/{arch} {composed[name][arch]['status']}" for name, arch in awaited))

    def deployable_builds(self, form: Optional[str] = None) -> list[tuple[str, str, dict]]:
        """Each build a deploy may boot, as (form, arch, record), in the base's order of forms and of arches; only
        ``form``'s when it's given: a ready one, or a disk no compose could probe (unverified), which its first boot on
        a provider that boots it natively probes before it's used; never one an agent-env that composes differently
        recorded."""
        composed, arches = self.metadata.get("composed", {}), self.computer.arch
        return [(name, arch, composed[name][arch]) for name in gate.forms(self.computer) if form in (None, name)
                for arch in arches if composed.get(name, {}).get(arch, {}).get("status") in _DEPLOYABLE[name]
                and not stale(composed[name][arch])]

    def disk_size_refusal(self, form: str, disk_size_gb: float) -> Optional[str]:
        """Why a deploy of ``form`` can't have ``disk_size_gb``: a disk is the size its base makes it, which a deploy
        doesn't enlarge."""
        if form in ("disk", "vm_bundle") and disk_size_gb > (size := getattr(self.computer.forms, form).min_disk_gb):
            return f"{self.id}'s disk is {size} GB, which a deploy doesn't enlarge, and it asks for {disk_size_gb:g} GB"
        return None

    def build_statuses(self) -> dict[str, str]:
        return {f"{form}/{arch}": f"{result.get('status')}, by another composer" if stale(result) else
                result.get("status") for form, by_arch in self.metadata.get("composed", {}).items()
                for arch, result in by_arch.items()}

    async def deploy(self, ttl_seconds: int = 10800, disk_size_gb: float = 10,
                     gateway_mode: GatewayMode = GatewayMode.PERFORMANCE, cpu: float | None = None,
                     memory_mb: int | None = None, sandbox_type: str | None = None, env_state_type: str | None = None,
                     env_state_instance_id: str | None = None, *, attribution: Optional[Attribution] = None,
                     env_params: Optional[dict] = None, tool_prefix: str = "", reach: str = "",
                     sites_later: bool = False, live_hosts: tuple[str, ...] = ()) -> DeployedEnv:
        """With ``tool_prefix``, its driver serves its tools under it, with ``reach``, a network that restricts its
        egress allows that URL's host too, and the ``live_hosts`` of the run's live websites, and with ``sites_later``
        it routes the sites ``env_params.site_routing`` reaches when ``route_sites`` gives them: what a multi env gives
        each of its computers."""
        return await deploy_through_provider(
            self, environment_name=self.environment_name, ttl_seconds=ttl_seconds, sandbox_type=sandbox_type,
            then=self._give_tools, disk_size_gb=disk_size_gb, gateway_mode=gateway_mode, cpu=cpu, memory_mb=memory_mb,
            env_state_type=env_state_type, env_state_instance_id=env_state_instance_id, attribution=attribution,
            env_params=env_params, tool_prefix=tool_prefix, reach=reach, sites_later=sites_later,
            live_hosts=tuple(live_hosts))

    async def route_sites(self, sites: list[dict]) -> None:
        """Route ``sites`` on this computer, deployed with ``sites_later``, launch its held apps and finish its boot;
        its record, and the instance store's, then keep them."""
        await as_builtin(self._env_provider).route_sites(self, self._deployed, sites)
        if self._deployed.instance_id:
            update_env_instance_metadata(self._deployed.instance_id, {"computer": self._deployed.metadata["computer"]})

    async def run_env_params(self, env_params: Optional[dict], deployed_envs: list[DeployedEnv]) -> Optional[dict]:
        """With ``site_routing`` and no ``sites``, ``env_params`` naming as its sites the websites the run deployed
        before the computer, all of them or those of the env ids ``site_routing`` lists, once each says it's ready."""
        routing = (env_params or {}).get("site_routing")
        if routing is not None and routing is not False and routing is not True and not (
                isinstance(routing, list) and all(isinstance(env_id, str) for env_id in routing)):
            raise ValueError(f"computer env {self.id!r}: env_params.site_routing is true or a list of env ids, not "
                             f"{routing!r}")
        if not routing or env_params.get("sites"):
            return env_params
        only = None if routing is True else list(routing)
        found = await sites.deployed_sites(deployed_envs, only)
        if not found:
            raise ValueError(f"computer env {self.id!r}: env_params.site_routing reaches the run's websites, and the "
                             "run deployed none before the computer")
        await sites.require_ready(deployed_envs, only)
        return {**env_params, "sites": found}

    def deploy_refusal(self, *, sandbox_type: str | None = None, env_params: Optional[dict] = None,
                       **options) -> str | None:
        """Why deploy() would refuse before booting anything, or None: an option the computer provider doesn't take, a
        problem with ``env_params``, or a sandbox provider with no one link that runs one of the env's arches natively,
        carries WebSockets when a website browser is among its apps, and can enforce its network, as a deploy needs of
        the link it boots on."""
        from agent_env.providers import build_sandbox_provider, get_env_sandbox_provider
        from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy

        if refusal := deploy_refusal(self, provider_or_class(self), {**options, "env_params": env_params}):
            return refusal
        params, _, problems = computer_params.checked(self, env_params)
        sandbox_provider = build_sandbox_provider(sandbox_type) if sandbox_type else get_env_sandbox_provider()
        links, arches = sandbox_provider.links, self.composed_arches()
        forms = [params.form] if params.form else gate.forms(self.computer)
        native = [link for link in links if any(boot.native_links(link, arch, form, self.stored_id, self.computer.os.type)
                                                for form in forms for arch in arches)]
        usable = [link for link in native if link.WEBSOCKETS] if self.serves_website_browser() else native
        if native and not usable:
            problems.append(f"{', '.join(type(link).__name__ for link in native)} carries no WebSockets, which its "
                            "website browser's CDP address needs")
        if not native:
            unpulled = [type(link).__name__ for link in links if not boot.pulls_containers(link, self.stored_id)]
            problems.append(f"{', '.join(type(link).__name__ for link in links)} runs none of its arches "
                            f"({', '.join(arches)}) natively as {' or '.join(forms)}" + (
                                f", and {', '.join(unpulled)} can't pull its containers from the image store's registry "
                                "on this machine" if "container" in forms and unpulled else ""))
        policy = NetworkPolicy.from_dict(self.network) if self.network else None
        if policy is not None and usable and not any(link.supports_network_policy(policy) for link in usable):
            problems.append(f"{', '.join(type(link).__name__ for link in usable)} can't enforce its network")
        return f"computer env {self.id!r}: " + "; ".join(problems) if problems else None

    @classmethod
    async def from_deployed_env(cls, deployed: DeployedEnv) -> ComputerEnv:
        env = Env.get(deployed.env_id, deployed.env_version)
        if not isinstance(env, ComputerEnv):
            raise TypeError(f"Expected ComputerEnv, got {type(env).__name__}")
        env._sandbox = await as_builtin(env._env_provider)._reattach(env, deployed)
        env._gateway_url = gateway_url_of(deployed)
        env._instance_id = deployed.instance_id
        env._deployed = deployed
        return env

    async def load_file_artifact_universe(self, file_artifact_universe: Any,
                                          destination_path: Optional[str] = None) -> LoadFileArtifactUniverseResult:
        """Write each file of ``file_artifact_universe`` under ``destination_path``, a folder relative to the session
        user's home (``Downloads`` by default), through the driver's ``data/add``, as that user; the result's paths are
        relative to the home too, since the home's path differs by OS."""
        if destination_path is not None and posixpath.isabs(destination_path):
            raise ValueError(f"{self.id}: a computer's destination_path is relative to its session user's home, not "
                             f"{destination_path!r}")
        destination = posixpath.normpath(destination_path or DEFAULT_DESTINATION)
        validate_relative_filename(destination)
        files = file_artifact_universe.get_file_artifacts()
        for filename in files:
            validate_relative_filename(filename)
        paths = {filename: posixpath.normpath(posixpath.join(destination, filename)) for filename in files}
        await inside.add_artifacts(self._deployed, {paths[filename]: artifact for filename, artifact in files.items()})
        logger.info(f"Loaded {len(files)} file(s) of '{file_artifact_universe.id}' into {self.id}'s ~/{destination}")
        return LoadFileArtifactUniverseResult(destination_path=destination, files=paths)

    async def load_environment_artifact(self, environment_artifact: EnvironmentArtifact) -> None:
        """Write the files of ``environment_artifact`` at their paths through the data plane its name reaches: the
        computer's own relative to the session user's home, through the driver's ``data/add``, as
        ``snapshot_computer_env`` kept them; an app's through the app's, which the driver fronts. Other files stay."""
        await self._load([environment_artifact])

    async def load_environment_universe_artifact(self, environment_universe_artifact: EnvironmentUniverseArtifact) -> None:
        """Load the universe's environment named after the computer, then each named after one of its apps, so what an
        app keeps ends as its own export has it; any other is left out, with a warning."""
        environments = environment_universe_artifact.get_environment_artifacts()
        names = self._environment_names()
        if not (known := [environment for environment in environments if environment.environment_name in names]):
            raise RuntimeError(f"computer env '{self.id}' can't load '{environment_universe_artifact.id}': it has no "
                               f"environment named any of {names} (it has "
                               f"{[environment.environment_name for environment in environments]})")
        if others := [environment.environment_name for environment in environments
                      if environment.environment_name not in names]:
            logger.warning(f"[{self.id}] loading only {[environment.environment_name for environment in known]} from "
                           f"'{environment_universe_artifact.id}'; leaving out {others}")
        await self.load_environments(known)

    def holds(self, environment_name: str) -> bool:
        """Whether a load reaches ``environment_name`` in this computer: its own, or one of its apps'."""
        return environment_name in self._environment_names()

    async def load_environments(self, environments: list[EnvironmentArtifact]) -> None:
        """Load ``environments``, each named after the computer or one of its apps: its own first, then its apps', so
        what an app keeps ends as its own export has it."""
        await self._load(sorted(environments, key=lambda environment: environment.environment_name != self.computer_name))

    async def reset(self, deployed: DeployedEnv) -> None:
        """Return the computer to boot state: its recordings ended, its container replaced with a fresh one of the same
        image, on the same sandbox and port, so its address stays. An agent's MCP session ends with the old driver; one
        held across the reset is answered 404, and MCP clients then start a new one. Its agents get its ``tools``
        again, as at deploy."""
        await as_builtin(self._env_provider).reboot(self, deployed)
        await self._give_tools(deployed)

    async def _give_tools(self, deployed: DeployedEnv) -> None:
        """Make the tool-access changes ``tools`` makes for the agents' role, which a task's own then follow, naming
        each tool as the deploy's tool prefix has the driver serve it."""
        prefix = (deployed.metadata or {}).get("computer", {}).get("tool_prefix", "")
        for action, tools in TOOLS[self.tools or "both"]:
            served = [tool if tool == WILDCARD else f"{prefix}{tool}" for tool in tools]
            await deployed.invoke(_TOOL_ACCESS_URIS[action], action, {"role": DEFAULT_ROLE, "tools": served})

    async def close(self) -> None:
        """End the computer's recordings, so what they recorded reaches the store, then tear it down."""
        if self._deployed is not None and self._sandbox is not None and self._deployed.sandbox_id == self._sandbox.sandbox_id:
            await end_recordings(self._deployed, "teardown")
        await close_replaced(self)
        if self._env_provider is not None:
            try:
                await self._env_provider.close()
            except Exception as e:
                logger.warning(f"Failed to close env provider: {e}")
        if self._sandbox is not None:
            try:
                await self._sandbox.terminate()
            except Exception as e:  # a cancellation propagates, and keeps the sandbox for the next close()
                logger.warning(f"Failed to terminate sandbox {self._sandbox.sandbox_id}: {e}")
            self._sandbox = None

    async def _load(self, environments: list[EnvironmentArtifact]) -> None:
        """Write each environment's files through the data plane its name reaches, every one checked before any is
        sent."""
        loads = [(environment, self._data_plane(environment.environment_name), environment.get_file_artifacts())
                 for environment in environments]
        for _, _, files in loads:
            for path in files:
                validate_relative_filename(path)
        for environment, url, files in loads:
            await inside.add_artifacts(self._deployed, files, url)
            logger.info(f"Loaded {len(files)} file(s) of '{environment.id}' into {self.id}'s "
                        f"'{environment.environment_name}'")

    def _data_plane(self, environment_name: str) -> str:
        """The address of the data plane ``environment_name`` reaches in the deployed computer, as its card says: the
        driver's for the computer's own, an app's route for an app's."""
        card = self._deployed.get_child_env_card(environment_name)
        if card is None:
            raise ValueError(f"computer env '{self.id}' has no environment named '{environment_name}': it has "
                             f"{self._environment_names()}")
        if METHOD_ADD not in (protocol_v1.operations(card) or []):
            raise ValueError(f"'{environment_name}' in computer env '{self.id}' serves no data/add, so nothing loads "
                             "into it")
        return self._deployed.environment_url + card.get("url", RPC_PATH).removesuffix(RPC_PATH)

    def _environment_names(self) -> list[str]:
        """The environments a load reaches by name: the computer's own, then its apps'."""
        return list(dict.fromkeys([self.computer_name, *(app.environment_name for app in self.composed_apps())]))


def _app(env: Env) -> ComputerAppEnv:
    if refusal := _not_an_app(env):
        raise TypeError(refusal)
    return env


def _listed_apps(refs: list, ctx: AuthoringContext) -> tuple[list[ComputerAppEnv], dict[str, dict]]:
    """The computer app envs an env.toml's ``apps`` names, each a ref, or a table of one with its ``install`` and its
    ``settings``, defaults for some of its settings, beside it, and those defaults by app name; refused, naming each
    problem, unless every one is a computer app env installed at build."""
    apps, app_settings, problems = [], {}, []
    for i, ref in enumerate(refs):
        install = ref.get("install", "build") if isinstance(ref, dict) else "build"
        settings = ref.get("settings", {}) if isinstance(ref, dict) else {}
        if install != "build":
            problems.append(ctx.config_problem(f"apps[{i}]: install = {install!r}: an app installs at \"build\", into "
                                               "the computer's image; \"boot\" isn't supported yet"))
            continue
        if not isinstance(settings, dict):
            problems.append(ctx.config_problem(f"apps[{i}]: settings must be a table of the app's setting values, not "
                                               f"{settings!r}"))
            continue
        app = ctx.env({key: value for key, value in ref.items() if key not in ("install", "settings")}
                      if isinstance(ref, dict) else ref)
        if refusal := _not_an_app(app):
            problems.append(ctx.config_problem(f"apps[{i}]: {refusal}"))
        elif settings:
            app_settings[app.environment_name] = settings
        apps.append(app)
    if problems:
        ctx.refuse(problems)
    return apps, app_settings


def _not_an_app(env: Env) -> str | None:
    return f"{env.id} is a {env.type} env, not a {APP_TYPE} env" if env.type != APP_TYPE else None


def _telling(env_id: str, progress: Optional[Callable[[str], None]]) -> tuple[
        Callable[[str], None], Optional[Callable[[str], None]]]:
    """What a deploy that composes says, to the log and the run's ``progress`` when there's one, after the env's id;
    and the progress alone, for the compose, which logs its own lines."""
    told = None if progress is None else (lambda line: progress(f"{env_id}: {line}"))

    def say(line: str) -> None:
        logger.info(f"{env_id}: {line}")
        if told is not None:
            told(line)

    return say, told


def _pairs(pairs: list[tuple[str, str]]) -> str:
    return ", ".join(f"{name}/{arch}" for name, arch in pairs)


def _waiting(awaited: dict[tuple[str, str], str | None], seconds: float) -> str:
    """What a deploy waiting for another's compose of ``awaited``, by when each started, says after ``seconds``."""
    since = min((at for at in awaited.values() if at), default=None)
    building = f"building since {datetime.fromisoformat(since).strftime('%H:%M')} UTC" if since else "building"
    so_far = f", {composer.took(seconds)} so far" if seconds >= 1 else ""
    return f"waiting for another deploy composing {_pairs(list(awaited)) or 'it'} ({building}{so_far})"


def _tools_problems(tools: Any) -> list[str]:
    return [] if tools is None or tools in TOOLS else [
        f"tools is {', '.join(map(repr, list(TOOLS)[:-1]))} or {list(TOOLS)[-1]!r}, not {tools!r}"]
