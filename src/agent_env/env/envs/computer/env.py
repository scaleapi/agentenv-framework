"""The computer env: a base computer and a driver, named by the env with what makes it itself, checked against each
other at put (gate 1) and composed into an image for each form the base has and each arch, each recorded in its
metadata as it builds (``composed``), and the arches built written as its computer (``computer``)."""

from __future__ import annotations

import logging
import posixpath
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self

from agent_env.artifact.artifacts.computer import ComputerArtifact, SettingValue
from agent_env.attribution import Attribution
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
from agent_env.env.envs.computer.utils import composer, gate, inside
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.envs.computer.utils import params as computer_params
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
from agent_env.env.gateway import GatewayMode
from agent_env.env.store import get_env_store
from agent_env.providers.env_providers.env_computer_provider import end_recordings
from agent_env.utils.paths import validate_relative_filename

if TYPE_CHECKING:
    from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.env.env import DeployedEnv
    from agent_env.env.envs.computer.app_env import ComputerAppEnv
    from agent_env.providers.env_providers.env_provider import EnvironmentProvider

logger = logging.getLogger(__name__)

#: Where load_artifact puts files in a computer by default: one of the folders data/get returns by default.
DEFAULT_DESTINATION = "Downloads"
#: The computer app env's type: an app composed into a computer env, and a computer env of its own.
APP_TYPE = "computer_app"


class ComputerEnv(Env):
    type: ClassVar[str] = "computer"
    description = "A computer: a base computer and a driver, composed at put into an image for each form and arch"
    env_provider_type: ClassVar[str] = "computer"
    toml_refs: ClassVar[tuple[EntityRef, ...]] = (EntityRef.artifact("computer", artifact_type="computer"),
                                                  EntityRef.env("driver", env_type=ComputerDriverEnv.type),
                                                  EntityRef.env("apps[]", env_type=APP_TYPE, beside=("install",)))

    def __init__(self, id: str, version: Optional[int], environment_name: str, computer: ComputerArtifact,
                 driver: ComputerDriverEnv, display: Optional[dict] = None, network: Optional[dict] = None,
                 computer_settings: Optional[dict[str, SettingValue]] = None, metadata: Optional[dict[str, Any]] = None,
                 apps: Optional[list[ComputerAppEnv]] = None):
        super().__init__(id, version, metadata=metadata)
        self.environment_name = environment_name
        self.computer, self.driver, self.apps = computer, driver, apps or []
        self.display, self.network = display, network
        self.computer_settings = computer_settings or {}
        self._sandbox = None
        self._env_provider: Optional[EnvironmentProvider] = builtin_provider_for(self.env_provider_type)
        self._gateway_url = None
        self._deployed: Optional[DeployedEnv] = None
        self._replaced: list[tuple] = []  # the provider and sandbox of each deployment a later deploy replaced, for close()

    def to_dict(self) -> dict:
        return {**super().to_dict(), "service_name": self.environment_name,
                "environment_name": self.environment_name,  # dual-write with service_name during the rename
                "computer": {"id": self.computer.id, "version": self.computer.version},
                "driver": {"id": self.driver.id, "version": self.driver.version}, "display": self.display,
                "network": self.network, "computer_settings": self.computer_settings,
                **({"apps": [{"id": app.id, "version": app.version} for app in self.apps]} if self.apps else {})}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerEnv:
        driver = Env.get(data["driver"]["id"], data["driver"]["version"])
        if not isinstance(driver, ComputerDriverEnv):
            raise TypeError(f"{data['driver']['id']} is a {driver.type} env, not a computer_driver env")
        return cls(id=data["id"], version=data.get("version"),
                   environment_name=data["environment_name"] if "environment_name" in data else data["service_name"],
                   computer=ComputerArtifact.get(data["computer"]["id"], data["computer"]["version"]), driver=driver,
                   display=data.get("display"), network=data.get("network"),
                   computer_settings=data.get("computer_settings"), metadata=data.get("metadata") or {},
                   apps=[_app(Env.get(ref["id"], ref["version"])) for ref in data.get("apps") or []])

    @classmethod
    def put(cls, *, id: str, computer: ComputerArtifact, driver: ComputerDriverEnv, environment_name: str,
            apps: Any = (), display: Any = None, network: Any = None, computer_settings: Any = None,
            compose: bool = True, composing: Composing = Composing()) -> Self:
        """Write a computer env of ``computer``, ``driver`` and ``apps``, computer app envs built into it in their
        order, once gate 1 passes, then compose it as ``composing`` says unless ``compose`` is False. Refused, naming
        every mismatch, before anything is stored."""
        apps = list(apps)
        values, problems = gate.check(computer, driver, apps=apps, display=display, network=network,
                                      computer_settings=computer_settings or {})
        if problems:
            raise ValueError(f"computer env {id!r}: " + "; ".join(problems))
        env = get_env_store().put_document(cls(
            id=id, version=None, environment_name=environment_name, computer=computer, driver=driver, apps=apps,
            display=display, network=network, computer_settings=values))
        if compose:
            composer.compose(env, composing=composing)
        return env

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> ComputerEnv:
        """Write the computer env an env.toml declares, its base, driver and apps as the bundle wrote them or the ones
        they name, its environment_name the folder's when it sets none. An app's ``install`` is ``"build"``, the
        default: it's built into the computer."""
        fields, refused = ctx.accepted(data, environment_name=str, computer=object, driver=object, apps=list,
                                       display=dict, network=dict, computer_settings=dict)
        missing = [ctx.config_problem(f"{key} is required: the {what}, such as {example}") for key, what, example in (
            ("computer", "base computer", '"ubuntu-24.04"'), ("driver", "driver", '"reference"')) if key not in data]
        if refused or missing:
            ctx.refuse([*refused, *missing])
        base, driver = ctx.artifact(fields.pop("computer"), ComputerArtifact), ctx.env(fields.pop("driver"),
                                                                                       ComputerDriverEnv)
        fields["apps"] = _listed_apps(fields.get("apps", []), ctx)
        try:
            return cls.put(id=ctx.id, computer=base, driver=driver,
                           environment_name=fields.pop("environment_name", ctx.name), composing=ctx.composing,
                           **fields)
        except ValueError as e:
            ctx.refuse([str(e)])

    def composed_apps(self) -> list[ComputerAppEnv]:
        """The computer app envs composed into the env's computer, each installed under its prefix."""
        return self.apps

    def computer_artifact(self) -> ComputerArtifact:
        """The env's computer: its base composed with its driver, over the images built."""
        return ComputerArtifact.get(self.metadata["computer"]["id"], self.metadata["computer"]["version"])

    def ready_builds(self, form: Optional[str] = None) -> list[tuple[str, str, dict]]:
        """Each ready build, as (form, arch, record), in the base's order of forms and of arches; only
        ``form``'s when it's given."""
        composed, arches = self.metadata.get("composed", {}), self.computer.arch
        return [(name, arch, composed[name][arch]) for name in gate.forms(self.computer) if form in (None, name)
                for arch in arches if composed.get(name, {}).get(arch, {}).get("status") == "ready"]

    def build_statuses(self) -> dict[str, str]:
        return {f"{form}/{arch}": result.get("status") for form, by_arch in self.metadata.get("composed", {}).items()
                for arch, result in by_arch.items()}

    async def deploy(self, ttl_seconds: int = 10800, disk_size_gb: float = 10,
                     gateway_mode: GatewayMode = GatewayMode.PERFORMANCE, cpu: float | None = None,
                     memory_mb: int | None = None, sandbox_type: str | None = None, env_state_type: str | None = None,
                     env_state_instance_id: str | None = None, *, attribution: Optional[Attribution] = None,
                     env_params: Optional[dict] = None) -> DeployedEnv:
        return await deploy_through_provider(
            self, environment_name=self.environment_name, ttl_seconds=ttl_seconds, sandbox_type=sandbox_type,
            disk_size_gb=disk_size_gb, gateway_mode=gateway_mode, cpu=cpu, memory_mb=memory_mb,
            env_state_type=env_state_type, env_state_instance_id=env_state_instance_id, attribution=attribution,
            env_params=env_params)

    def deploy_refusal(self, *, sandbox_type: str | None = None, env_params: Optional[dict] = None,
                       **options) -> str | None:
        """Why deploy() would refuse before booting anything, or None: an option the computer provider doesn't take, a
        problem with ``env_params``, or a sandbox provider that runs none of the env's arches natively or can't enforce
        its network."""
        from agent_env.providers import build_sandbox_provider, get_env_sandbox_provider
        from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy

        if refusal := deploy_refusal(self, provider_or_class(self), {**options, "env_params": env_params}):
            return refusal
        _, _, problems = computer_params.checked(self, env_params)
        sandbox_provider = build_sandbox_provider(sandbox_type) if sandbox_type else get_env_sandbox_provider()
        links, arches = sandbox_provider.links, gate.arches(self.computer, self.driver, self.composed_apps())
        if not any(f"linux/{arch}" in link.native_platforms() for link in links for arch in arches):
            problems.append(f"{', '.join(type(link).__name__ for link in links)} runs none of its arches "
                            f"({', '.join(arches)}) natively")
        policy = NetworkPolicy.from_dict(self.network) if self.network else None
        if policy is not None and not any(link.supports_network_policy(policy) for link in links):
            problems.append(f"{', '.join(type(link).__name__ for link in links)} can't enforce its network")
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
        """Write the files of ``environment_artifact``, the computer's own, at their paths relative to the session
        user's home, through the driver's ``data/add``, as ``snapshot_computer_env`` kept them. Other files there
        stay."""
        if environment_artifact.environment_name != self.environment_name:
            raise ValueError(f"EnvironmentArtifact environment_name '{environment_artifact.environment_name}' does not "
                             f"match env environment_name '{self.environment_name}'")
        files = environment_artifact.get_file_artifacts()
        for path in files:
            validate_relative_filename(path)
        await inside.add_artifacts(self._deployed, files)
        logger.info(f"Loaded {len(files)} file(s) of '{environment_artifact.id}' into {self.id}'s home")

    async def load_environment_universe_artifact(self, environment_universe_artifact: EnvironmentUniverseArtifact) -> None:
        """Load the universe's environment named after the computer, ``load_environment_artifact``; any other is left
        out, with a warning."""
        environments = environment_universe_artifact.get_environment_artifacts()
        own = [environment for environment in environments if environment.environment_name == self.environment_name]
        if not own:
            raise RuntimeError(f"computer env '{self.id}' can't load '{environment_universe_artifact.id}': it has no "
                               f"environment named '{self.environment_name}' (it has "
                               f"{[environment.environment_name for environment in environments]})")
        if others := [environment.environment_name for environment in environments
                      if environment.environment_name != self.environment_name]:
            logger.warning(f"[{self.id}] loading only '{self.environment_name}' from '{environment_universe_artifact.id}'; "
                           f"leaving out {others}")
        await self.load_environment_artifact(own[0])

    async def reset(self, deployed: DeployedEnv) -> None:
        """Return the computer to boot state: its recordings ended, its container replaced with a fresh one of the same
        image, on the same sandbox and port, so its address stays. An agent's MCP session ends with the old driver; one
        held across the reset is answered 404, and MCP clients then start a new one."""
        await as_builtin(self._env_provider).reboot(self, deployed)

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


def _app(env: Env) -> ComputerAppEnv:
    if refusal := _not_an_app(env):
        raise TypeError(refusal)
    return env


def _listed_apps(refs: list, ctx: AuthoringContext) -> list[ComputerAppEnv]:
    """The computer app envs an env.toml's ``apps`` names, each a ref, or a table of one with its ``install`` beside it;
    refused, naming each problem, unless every one is a computer app env installed at build."""
    apps, problems = [], []
    for i, ref in enumerate(refs):
        install = ref.get("install", "build") if isinstance(ref, dict) else "build"
        if install != "build":
            problems.append(ctx.config_problem(f"apps[{i}]: install = {install!r}: an app installs at \"build\", into "
                                               "the computer's image; \"boot\" isn't supported yet"))
            continue
        app = ctx.env({key: value for key, value in ref.items() if key != "install"} if isinstance(ref, dict) else ref)
        if refusal := _not_an_app(app):
            problems.append(ctx.config_problem(f"apps[{i}]: {refusal}"))
        apps.append(app)
    if problems:
        ctx.refuse(problems)
    return apps


def _not_an_app(env: Env) -> str | None:
    return f"{env.id} is a {env.type} env, not a {APP_TYPE} env" if env.type != APP_TYPE else None
