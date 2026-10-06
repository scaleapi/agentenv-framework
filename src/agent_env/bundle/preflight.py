"""What a bundle run checks about where its tasks deploy, before it writes anything, and the infra envs it builds.

Each deploy, an env's, an agent's, a sandbox's or a rubrics judge's, runs on its effective sandbox provider: the run's
``--sandbox``, else the step's own field, else the config default, resolved as the step resolves it when it runs. A
comma-separated provider is a chain whose deploys fall back from one provider to the next, so every provider in it must
pass. A run is refused when a deploy would fail on its provider: an image only this machine has (in its local registry,
or saved in its local object store) on a provider that isn't the local one, a VM asked of a provider that can't create
one, or containers on the local provider without a Docker daemon. A deploy_agent step that names no agent, and a judge
that names none, deploy the configured default agent, which must be in the store.

A gateway deploy on the local provider runs on infra envs the store may not hold yet: the gateway, the service-db its
local Postgres state runs from, and the website browser it adds for websites. The run builds those once its writes are
done (``agent_env.env.bootstrap``), so they're named here, and refused here when they can't be built.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import urlparse

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.config import get_config
from agent_env.entity_refs import EntityKind, parse_toml_ref
from agent_env.env.bootstrap import (
    GATEWAY,
    SERVICE_DB,
    WEBSITE_BROWSER,
    InfraBuild,
    InfraError,
    default_env_id,
    infra_to_build,
    put_command,
)
from agent_env.env.env import Env
from agent_env.env.envs._deployment import provider_or_class
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.service_db import ServiceDBEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider, _gateway_topology
from agent_env.providers.env_providers.env_server_provider import EnvironmentServerProvider
from agent_env.providers.env_state.env_state_provider import LOCAL_POSTGRES_STATE_TYPE
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxProvider,
    build_sandbox_provider,
    get_agent_sandbox_provider,
    get_env_sandbox_provider,
    get_sandbox_provider,
)
from agent_env.store.base import NotFoundError
from agent_env.store.image_store.oci_registry_credentials import registry_host_from_ref
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.utils.docker_build import docker_unreachable

from ._fs import relative
from .parse import BundleError, BundleKind
from .plan import Plan
from .resolve import BuiltImage, ResolvedEntry, build_step

_SHOWN_DOCKER_USERS = 3


@dataclass(frozen=True)
class Preflight:
    """What a run needs before its tasks start, beyond its writes: the infra envs it builds."""

    infra_kinds: frozenset[str]  # the infra its deploys on the local provider run on
    infra: tuple[InfraBuild, ...]  # those the store doesn't hold yet, or another agent-env release built


def preflight_run(plan: Plan, tasks: Iterable[ResolvedEntry], sandbox: str | None) -> Preflight:
    """Check where each of ``tasks`` deploys, with ``sandbox`` overriding every deploy's provider, and name the infra
    envs the run builds. Raises BundleError listing every problem found. Reads the store but writes nothing."""
    walk = _Walk(plan, sandbox)
    for task in tasks:
        walk.task(task)
    return walk.finish()


@dataclass(frozen=True)
class _Image:
    """An image a deploy runs, as its message names it."""

    what: str
    local_only: str | None  # why only this machine has it, or None


class _Walk:
    def __init__(self, plan: Plan, sandbox: str | None):
        self.plan, self.sandbox = plan, sandbox
        self.problems: list[str] = []
        self.infra: set[str] = set()  # the infra kinds a deploy on the local provider needs
        self.remote_infra: list[tuple[str, SandboxProvider, set[str]]] = []  # (where, provider, kinds) of a deploy elsewhere
        self.docker_users: list[str] = []
        self.default_agent_users: list[tuple[str, str]] = []  # (where, what names no agent)
        self.written = {write.id for write in plan.writes}
        self.agents = {write.id: write.source for write in plan.writes if write.kind is BundleKind.AGENT}
        self.built = {write.id: write.source for write in plan.writes if isinstance(write.source, BuiltImage)}

    def task(self, task: ResolvedEntry) -> None:
        steps = [build_step(config) for config in task.config]
        sandboxes = {step.sandbox_name: step for step in steps if isinstance(step, DeploySandboxTaskStep)}
        for step in steps:
            where = f"{relative(self.plan.bundle.bundle.root, task.entry.path)}: step {step.id!r}"
            try:
                if isinstance(step, DeployEnvTaskStep):
                    self._env(where, step)
                elif isinstance(step, DeployAgentTaskStep):
                    self._agent(where, step, sandboxes)
                elif isinstance(step, DeploySandboxTaskStep):
                    self._sandbox(where, step)
                elif isinstance(step, RubricsVerifierTaskStep):
                    self._judge(where, step)
            except ValueError as e:  # a provider that names no backend, or can't be built from its config
                self.problems.append(f"{where}: {e}")

    def finish(self) -> Preflight:
        infra: list[InfraBuild] = []
        try:
            infra = infra_to_build(self.infra)
        except InfraError as e:
            self.problems.extend(e.problems)
        for where, provider, kinds in self.remote_infra:
            self._remote_infra(where, provider, kinds)
        self._default_agent()
        if self.docker_users and (reason := docker_unreachable()):
            shown = ", ".join(self.docker_users[:_SHOWN_DOCKER_USERS])
            more = len(self.docker_users) - _SHOWN_DOCKER_USERS
            self.problems.append(f"the local sandbox provider runs containers for {shown}"
                                 f"{f' and {more} more' if more > 0 else ''}, and {reason}")
        if self.problems:
            raise BundleError(self.problems)
        return Preflight(frozenset(self.infra), tuple(infra))

    # Deploys, each resolving its provider as its step does when it runs

    def _env(self, where: str, step: DeployEnvTaskStep) -> None:
        provider = _provider(self.sandbox or step.sandbox_type, get_env_sandbox_provider)
        if step.env_id in self.written:
            return  # a bundle env, refused at materialize until envs can be written
        try:
            env = Env.get(step.env_id, step.env_version)
        except NotFoundError:
            return  # the plan reports a store env that isn't there
        if not isinstance(env, (MCPServerEnv, WebsiteEnv, MultiEnv)):
            return  # an env type that deploys itself, as deploy_env's own preflight leaves it
        try:
            provider_class = provider_or_class(env)
        except (ValueError, KeyError):
            return  # deploy_env's own preflight reports a provider type this process can't load
        if not isinstance(provider_class, type):
            provider_class = type(provider_class)
        if issubclass(provider_class, EnvironmentGatewayProvider):
            topology = _gateway_topology(env)
            images = [*topology.mcp_server_images, *(topology.website_images or [])]
            kinds = {GATEWAY}
            if step.env_state_instance_id is None and (step.env_state_type or LOCAL_POSTGRES_STATE_TYPE) == LOCAL_POSTGRES_STATE_TYPE:
                kinds.add(SERVICE_DB)
            if topology.website_configs:
                kinds.add(WEBSITE_BROWSER)
        elif issubclass(provider_class, EnvironmentServerProvider) and isinstance(env, MCPServerEnv):
            images, kinds = [env.docker_image_artifact], set()
        else:  # a plugin's provider, or one deploy_env's own preflight refuses for this env
            return
        if _local_link(provider):
            self.infra |= kinds
            self.docker_users.append(where)
        if remote := _remote_links(provider):
            self._reachable(where, remote, [_Image(f"env {env.id!r}'s image {image.id!r}", _local_only(image)) for image in images])
            # Modal's gateway runs each server in a container of its own, and can't serve websites.
            containers = [link for link in remote if isinstance(link, ModalSandboxProvider)]
            vms = [link for link in remote if link not in containers]
            if WEBSITE_BROWSER in kinds and containers:
                self.problems.append(f"{where}: deploys env {env.id!r}, which has websites, on the {_shown(containers[0])} "
                                     "sandbox provider, whose gateway runs in containers and can't serve websites; run it "
                                     "on a VM provider, such as --sandbox local")
            if containers:
                self.remote_infra.append((where, containers[0], kinds - {WEBSITE_BROWSER}))
            if vms:
                self.remote_infra.append((where, vms[0], kinds))

    def _agent(self, where: str, step: DeployAgentTaskStep, sandboxes: dict[str, DeploySandboxTaskStep]) -> None:
        if step.sandbox_name:
            linked = sandboxes.get(step.sandbox_name)
            if linked is None:
                return  # the step itself refuses an agent linked to a sandbox the task doesn't deploy
            provider = _provider(self.sandbox or linked.sandbox_type, get_sandbox_provider)
        else:
            provider = _provider(self.sandbox or step.sandbox_type, get_agent_sandbox_provider)
        self._agent_deploy(where, provider, step.a2a_agent_id, step.a2a_agent_version, "names no agent")

    def _judge(self, where: str, step: RubricsVerifierTaskStep) -> None:
        if not step.use_agent_judge or step.agent_name is not None:
            return  # the direct LLM judge, or a judge the task deployed itself
        provider = _provider(self.sandbox or step.judge_sandbox_type, get_agent_sandbox_provider)
        self._agent_deploy(where, provider, step.judge_a2a_agent_id, None, "names no judge agent")

    def _agent_deploy(self, where: str, provider: SandboxProvider, agent_id: str | None, version: int | None,
                      unnamed: str) -> None:
        if _local_link(provider):
            self.docker_users.append(where)
        if agent_id is None:
            self.default_agent_users.append((where, unnamed))
            agent_id = get_config().get_default_a2a_agent_id()
        if remote := _remote_links(provider):
            image = self._agent_image(agent_id, version)
            if image is not None:
                self._reachable(where, remote, [image])

    def _sandbox(self, where: str, step: DeploySandboxTaskStep) -> None:
        provider = _provider(self.sandbox or step.sandbox_type, get_sandbox_provider)
        if step.sandbox_mode == "vm":
            if not _creates_vms(provider):
                self.problems.append(f"{where}: deploys a VM sandbox, and the {_shown(provider)} sandbox provider can't "
                                     "create a VM; run it on one that can, such as --sandbox local")
            return
        if _local_link(provider):
            self.docker_users.append(where)
        if step.image and (remote := _remote_links(provider)):
            self._reachable(where, remote, [_Image(f"the image {step.image}", _local_only(step.image))])

    # What a deploy runs

    def _agent_image(self, agent_id: str, version: int | None) -> _Image | None:
        """The image the agent ``agent_id`` runs, or None when the store doesn't hold the agent, which the plan or
        ``_default_agent`` reports."""
        what = f"agent {agent_id!r}'s image"
        if (agent := self.agents.get(agent_id)) is not None:
            image_id, image_version = parse_toml_ref(EntityKind.ARTIFACT, agent.config.get("image"))
            if (built := self.built.get(image_id)) is not None:
                path = relative(self.plan.bundle.bundle.root, built.entry.path)
                return _Image(what, f"it's built on this machine from {path}/{built.dockerfile}")
            if image_id in self.written:
                return None  # another of the bundle's writes, which materialize refuses or writes first
            return _Image(what, _local_only(DockerImageArtifact.get(image_id, image_version)))
        try:
            return _Image(what, _local_only(A2AAgent.get(agent_id, version).docker_image_artifact))
        except NotFoundError:
            return None

    def _reachable(self, where: str, remote: list[SandboxProvider], images: list[_Image]) -> None:
        for image in images:
            if image.local_only:
                self.problems.append(f"{where}: deploys {image.what} on the {_shown(remote[0])} sandbox provider, which "
                                     f"can't reach it: {image.local_only}; run it with --sandbox local")

    def _remote_infra(self, where: str, provider: SandboxProvider, kinds: set[str]) -> None:
        for kind in sorted(kinds):
            env_id = default_env_id(kind)
            try:
                env = Env.get(env_id)
            except NotFoundError:
                self.problems.append(f"{where}: deploys on the {_shown(provider)} sandbox provider, which needs the {kind} "
                                     f"env {env_id!r}, and the store doesn't hold it; agent-env builds it only for the local "
                                     f"sandbox provider, so put it in a store that provider can reach (`{put_command(kind)}`)")
                continue
            images = ([env.db_docker_image_artifact, env.db_web_docker_image_artifact, env.db_mcp_docker_image_artifact]
                      if isinstance(env, ServiceDBEnv) else [env.docker_image_artifact])
            self._reachable(where, [provider], [_Image(f"the {kind} env {env_id!r}'s image {image.id!r}", _local_only(image))
                                                for image in images])

    def _default_agent(self) -> None:
        if not self.default_agent_users:
            return
        agent_id = get_config().get_default_a2a_agent_id()
        if agent_id in self.agents:
            return
        try:
            A2AAgent.get(agent_id)
            return
        except NotFoundError:
            problem = (f"there is no agent {agent_id!r} in the store; register one under that id, or point "
                       "[agents] default_a2a_agent_id at an agent that is")
        except (ValueError, KeyError, TypeError) as e:
            problem = f"agent {agent_id!r} can't be read ({type(e).__name__}: {e})"
        for where, unnamed in self.default_agent_users:
            self.problems.append(f"{where}: {unnamed}, so it deploys the default, {agent_id!r}, and {problem}")


def _provider(spec: str | None, default: Callable[[], SandboxProvider]) -> SandboxProvider:
    return build_sandbox_provider(spec) if spec else default()


def _links(provider: SandboxProvider) -> list[SandboxProvider]:
    return list(provider.providers) if isinstance(provider, ChainedSandboxProvider) else [provider]


def _local_link(provider: SandboxProvider) -> bool:
    return any(isinstance(link, LocalSandboxProvider) for link in _links(provider))


def _remote_links(provider: SandboxProvider) -> list[SandboxProvider]:
    return [link for link in _links(provider) if not isinstance(link, LocalSandboxProvider)]


def _creates_vms(provider: SandboxProvider) -> bool:
    """Whether ``provider`` implements create_vm; a chain doesn't, whatever its providers do."""
    return type(provider).create_vm is not SandboxProvider.create_vm


def _shown(provider: SandboxProvider) -> str:
    return repr(",".join(_name(link) for link in _links(provider)))


def _name(provider: SandboxProvider) -> str:
    """The name ``provider`` is registered under, as ``--sandbox`` names it."""
    cls = type(provider)
    for name, section in get_config().sandbox_registry().items():
        impl = section.get("impl")
        if impl is cls or impl == f"{cls.__module__}:{cls.__qualname__}":
            return name
    return cls.__name__


def _local_only(image: DockerImageArtifact | str) -> str | None:
    """Why only this machine has ``image``: its reference names a registry on this machine, or it's saved in this
    machine's object store. None when neither."""
    ref = image if isinstance(image, str) else image.image_name
    if _loopback(registry_host_from_ref(ref)):
        return f"{ref} is in a registry on this machine"
    if not isinstance(image, str) and urlparse(image.tar_gz_object_url).scheme == "file":
        return f"{image.id!r} is saved in this machine's object store"
    return None


def _loopback(host: str | None) -> bool:
    if not host:
        return False
    name = host[1:host.find("]")] if host.startswith("[") else host.rsplit(":", 1)[0]
    return name == "localhost" or name.startswith("127.") or name == "::1"

