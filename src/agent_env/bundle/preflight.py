"""What a bundle run checks about where its tasks deploy, before it writes anything, and the infra envs it builds.

Each deploy, an env's, an agent's, a sandbox's or a rubrics judge's, runs on its effective sandbox provider: the run's
``--sandbox``, else the step's own field, else the config default, resolved as the step resolves it when it runs. A
comma-separated provider is a chain whose deploys fall back from one provider to the next, so every provider in it must
pass. A run is refused when a deploy would fail on its provider: an image only this machine has (in its local registry,
or saved in its local object store) on a provider that isn't the local one, a VM asked of a provider that can't create
one, or containers on the local provider without a Docker daemon. A deploy_agent step that names no agent, and a judge
that names none, deploy the configured default agent, which must be in the store. An env the bundle writes is
checked from its planned env.toml.

An image the bundle builds is written in the form its deploys can run: built on this machine when they deploy on the
local provider, and as its build context alone when they deploy on providers that build it, in a VM or, as Modal does,
themselves. One version is one of the two, so a run deploying it both ways is refused, as is one deploying it on a
provider that runs an image by name.

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
from agent_env.env.registry import get_env_registry
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider, _gateway_topology
from agent_env.providers.env_providers.env_provider import _env_provider_class
from agent_env.providers.env_providers.env_server_provider import EnvironmentServerProvider
from agent_env.providers.env_state.env_state_provider import LOCAL_POSTGRES_STATE_TYPE
from agent_env.providers.env_state.store import get_env_state_instance_store
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
from agent_env.store.image_store.oci_registry_credentials import is_loopback_host, registry_host_from_ref
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.utils.docker_build import docker_unreachable

from ._fs import relative
from .parse import BundleError, BundleKind
from .plan import Plan
from .resolve import BuiltImage, Reference, ResolvedEntry, build_step

_SHOWN_DOCKER_USERS = 3


@dataclass(frozen=True)
class Preflight:
    """What a run needs before its tasks start, beyond its writes: the infra envs it builds."""

    infra_kinds: frozenset[str]  # the infra its deploys on the local provider run on
    infra: tuple[InfraBuild, ...]  # those the store doesn't hold yet, or holds built from other inputs
    contexts: frozenset[str] = frozenset()  # the ids of the images the bundle builds that it writes as build contexts


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
    unloadable: str | None = None  # why no sandbox can get it, or None
    by_name: str | None = None  # why a sandbox that runs it by pulling its name can't, or None
    built: str | None = None  # the id of the image when the bundle builds it, whose form the walk decides at the end


@dataclass(frozen=True)
class _Deployment:
    """How a deploy_env step's env deploys, as far as the walk checks it."""

    provider: type  # the provider class its env_provider_type names
    images: list[_Image]  # what it runs, its children's included
    websites: bool
    one_server: bool  # a single MCP server, which a provider that deploys one server can take


class _Walk:
    def __init__(self, plan: Plan, sandbox: str | None):
        self.plan, self.sandbox = plan, sandbox
        self.problems: dict[str, dict[str, None]] = {}  # each deploy's problem, and the deploys it's found at
        self.run_problems: list[str] = []  # the run's as a whole
        self.infra: set[str] = set()  # the infra kinds a deploy on the local provider needs
        self.remote_infra: list[tuple[str, SandboxProvider, set[str]]] = []  # (where, provider, kinds) of a deploy elsewhere
        self.docker_users: list[str] = []
        self.default_agent_users: list[tuple[str, str]] = []  # (where, what names no agent)
        self.written = {write.id for write in plan.writes}
        self.agents = {write.id: write.source for write in plan.writes if write.kind is BundleKind.AGENT}
        self.envs = {write.id: write.source for write in plan.writes if write.kind is BundleKind.ENV}
        self.built = {write.id: write.source for write in plan.writes if isinstance(write.source, BuiltImage)}
        # Each deploy of an image the bundle builds: (where, what, its provider's links, those that run it by name)
        self.uses: dict[str, list[tuple[str, str, list[SandboxProvider], list[SandboxProvider]]]] = {}

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
                self._problem(where, str(e))

    def finish(self) -> Preflight:
        infra: list[InfraBuild] = []
        try:
            infra = infra_to_build(self.infra)
        except InfraError as e:
            self.run_problems.extend(e.problems)
        for where, provider, kinds in self.remote_infra:
            self._remote_infra(where, provider, kinds)
        contexts = self._contexts()
        self._default_agent()
        if self.docker_users and (reason := docker_unreachable()):
            shown = ", ".join(self.docker_users[:_SHOWN_DOCKER_USERS])
            more = len(self.docker_users) - _SHOWN_DOCKER_USERS
            self.run_problems.append(f"the local sandbox provider runs containers for {shown}"
                                     f"{f' and {more} more' if more > 0 else ''}, and {reason}")
        problems = list(self.run_problems)
        for problem, wheres in self.problems.items():
            first, *others = wheres
            more = f" (and {len(others)} more deploy{'s' if len(others) > 1 else ''})" if others else ""
            problems.append(f"{first}{more}: {problem}")
        if problems:
            raise BundleError(problems)
        return Preflight(frozenset(self.infra), tuple(infra), contexts)

    # Deploys, each resolving its provider as its step does when it runs

    def _env(self, where: str, step: DeployEnvTaskStep) -> None:
        provider = _provider(self.sandbox or step.sandbox_type, get_env_sandbox_provider)
        deployment = self._planned(step.env_id) if step.env_id in self.envs else self._stored(step)
        if deployment is None:
            return
        if issubclass(deployment.provider, EnvironmentGatewayProvider):
            kinds = {GATEWAY}
            if _state_type(step) == LOCAL_POSTGRES_STATE_TYPE:
                kinds.add(SERVICE_DB)
            if deployment.websites:
                kinds.add(WEBSITE_BROWSER)
        elif issubclass(deployment.provider, EnvironmentServerProvider) and deployment.one_server:
            kinds = set()
        else:  # a plugin's provider, or one deploy_env's own preflight refuses for this env
            return
        env_id, images = step.env_id, deployment.images
        if GATEWAY in kinds and (loaders := [link for link in _links(provider)
                                             if not isinstance(link, ModalSandboxProvider)]):
            self._loadable(where, loaders, images)  # a gateway VM loads its images; Modal's runs each by name
        # A lone server runs in a container from its image's name, though Modal's builds one that's only a build context;
        # a gateway VM loads its images, and Modal's gateway builds each.
        by_name = [] if GATEWAY in kinds else [link for link in _links(provider)
                                               if not isinstance(link, ModalSandboxProvider)]
        self._by_name(where, by_name, images)
        self._use(where, _links(provider), by_name, images)
        if _local_link(provider):
            self.infra |= kinds
            self.docker_users.append(where)
        if remote := _remote_links(provider):
            self._reachable(where, remote, images)
            # Modal's gateway runs each server in a container of its own, and can't serve websites.
            containers = [link for link in remote if isinstance(link, ModalSandboxProvider)]
            vms = [link for link in remote if link not in containers]
            if WEBSITE_BROWSER in kinds and containers:
                self._problem(where, f"deploys env {env_id!r}, which has websites, on the {_shown(containers[0])} "
                                     "sandbox provider, whose gateway runs in containers and can't serve websites; run it "
                                     "on a VM provider, such as --sandbox local")
            if containers:
                self.remote_infra.append((where, containers[0], kinds - {WEBSITE_BROWSER}))
            if vms:
                self.remote_infra.append((where, vms[0], kinds))

    def _stored(self, step: DeployEnvTaskStep) -> _Deployment | None:
        """How a store env deploys, or None when the walk leaves it to deploy_env's own preflight."""
        try:
            env = Env.get(step.env_id, step.env_version)
        except NotFoundError:
            return None  # the plan reports a store env that isn't there
        if not isinstance(env, (MCPServerEnv, WebsiteEnv, MultiEnv)):
            return None  # an env type that deploys itself, as deploy_env's own preflight leaves it
        try:
            provider_class = provider_or_class(env)
        except (ValueError, KeyError):
            return None  # deploy_env's own preflight reports a provider type this process can't load
        if not isinstance(provider_class, type):
            provider_class = type(provider_class)
        images, websites = _stored_images(env)
        return _Deployment(provider_class, images, websites, isinstance(env, MCPServerEnv))

    def _planned(self, env_id: str) -> _Deployment | None:
        """How an env the bundle writes deploys, from its planned config: what deploys it, and its images and its
        children's. None for a type the walk doesn't model, or a provider type the plan has refused."""
        cls = get_env_registry().get(self.envs[env_id].entry.type)
        if cls is None or not issubclass(cls, (MCPServerEnv, WebsiteEnv, MultiEnv)):
            return None
        provider_type = self.envs[env_id].config.get("env_provider_type", EnvironmentGatewayProvider.type)
        try:
            provider_class = _env_provider_class(provider_type)
        except ValueError:
            return None
        images, websites = self._planned_images(env_id)
        return _Deployment(provider_class, images, websites, issubclass(cls, MCPServerEnv))

    def _planned_images(self, env_id: str) -> tuple[list[_Image], bool]:
        """The images an env the bundle writes runs, its children's included, and whether it has websites."""
        source = self.envs[env_id]
        cls = get_env_registry().get(source.entry.type)
        images, websites = [], cls is not None and issubclass(cls, WebsiteEnv)
        for ref in source.references:
            if ref.kind is EntityKind.ARTIFACT and (image := self._planned_image(env_id, ref)) is not None:
                images.append(image)
            elif ref.kind is EntityKind.ENV:
                if ref.id in self.envs:
                    child_images, child_websites = self._planned_images(ref.id)
                else:
                    try:
                        child = Env.get(ref.id, self._planned_version(ref.kind, ref.id, ref.version))
                        child_images, child_websites = _stored_images(child)
                    except NotFoundError:
                        continue  # the plan reports a store env that isn't there
                images += child_images
                websites |= child_websites
        return images, websites

    def _planned_image(self, env_id: str, ref: Reference) -> _Image | None:
        what = f"env {env_id!r}'s image {ref.id!r}"
        if ref.id in self.built:
            return _Image(what, None, built=ref.id)
        if ref.id in self.written:
            return None  # another of the bundle's writes, which materialize refuses or writes first
        try:
            return _image(what, DockerImageArtifact.get(ref.id, self._planned_version(ref.kind, ref.id, ref.version)))
        except NotFoundError:
            return None  # the plan reports a store image that isn't there

    def _planned_version(self, kind: EntityKind, entity_id: str, version: int | None) -> int | None:
        """The version a write names a store entity at: its pin, else the one the plan read, which the writer pins."""
        return version if version is not None else self.plan.store_latest.get((kind, entity_id))

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
        containers = [link for link in _links(provider)
                      if isinstance(link, (LocalSandboxProvider, ModalSandboxProvider))]
        loaders = [link for link in _links(provider) if link not in containers]  # containers run an agent's image
        remote = _remote_links(provider)
        if (image := self._agent_image(agent_id, version)) is None:
            return
        if loaders:
            self._loadable(where, loaders, [image])
        by_name = [link for link in containers if isinstance(link, LocalSandboxProvider)]  # Modal's builds it
        self._by_name(where, by_name, [image])
        self._use(where, _links(provider), by_name, [image])
        if remote:
            self._reachable(where, remote, [image])

    def _sandbox(self, where: str, step: DeploySandboxTaskStep) -> None:
        provider = _provider(self.sandbox or step.sandbox_type, get_sandbox_provider)
        if step.sandbox_mode == "vm":
            if not _creates_vms(provider):
                self._problem(where, f"deploys a VM sandbox, and the {_shown(provider)} sandbox provider can't create a "
                                     "VM; run it on one that can, such as --sandbox local")
            return
        if _local_link(provider):
            self.docker_users.append(where)
        if step.image and (remote := _remote_links(provider)):
            self._reachable(where, remote, [_Image(f"the image {step.image}", _local_only(step.image))])

    # What a deploy runs

    def _agent_image(self, agent_id: str, version: int | None) -> _Image | None:
        """The image the agent ``agent_id`` runs, or None when the store doesn't hold the agent or can't read it, which
        the plan or ``_default_agent`` reports."""
        what = f"agent {agent_id!r}'s image"
        if (agent := self.agents.get(agent_id)) is not None:
            image_id, image_version = parse_toml_ref(EntityKind.ARTIFACT, agent.config.get("image"))
            if image_id in self.built:
                return _Image(what, None, built=image_id)
            if image_id in self.written:
                return None  # another of the bundle's writes, which materialize refuses or writes first
            planned = self._planned_version(EntityKind.ARTIFACT, image_id, image_version)
            return _image(what, DockerImageArtifact.get(image_id, planned))
        try:
            return _image(what, A2AAgent.get(agent_id, version).docker_image_artifact)
        except (NotFoundError, ValueError, KeyError, TypeError):
            return None

    def _loadable(self, where: str, loaders: list[SandboxProvider], images: list[_Image]) -> None:
        """Refuse each of ``images`` a VM on ``loaders`` can't get: it loads an image's tar.gz, or pulls an image with
        none by name."""
        for image in images:
            if image.unloadable:
                self._problem(where, f"deploys {image.what} on the {_shown(loaders[0])} sandbox provider, which can't "
                                     f"load it: {image.unloadable}")

    def _by_name(self, where: str, links: list[SandboxProvider], images: list[_Image]) -> None:
        """Refuse each of ``images`` a sandbox on ``links`` can't run: it pulls an image by its name, and a context-only
        image's name is a tag only its build gives it."""
        on_modal = bool(links) and isinstance(links[0], ModalSandboxProvider)
        builds = "whose VMs build it, such as --sandbox modal_vm" if on_modal else "that builds it, such as --sandbox modal"
        for image in images:
            if links and image.by_name:
                self._problem(where, f"deploys {image.what} on the {_shown(links[0])} sandbox provider, which runs it "
                                     f"by name: {image.by_name}; run it on a provider {builds}")

    def _reachable(self, where: str, remote: list[SandboxProvider], images: list[_Image]) -> None:
        for image in images:
            if image.local_only:
                self._problem(where, f"deploys {image.what} on the {_shown(remote[0])} sandbox provider, which can't "
                                     f"reach it: {image.local_only}; run it with --sandbox local")

    def _remote_infra(self, where: str, provider: SandboxProvider, kinds: set[str]) -> None:
        for kind in sorted(kinds):
            env_id = default_env_id(kind)
            try:
                env = Env.get(env_id)
            except NotFoundError:
                self._problem(where, f"deploys on the {_shown(provider)} sandbox provider, which needs the {kind} env "
                                     f"{env_id!r}, and the store doesn't hold it; agent-env builds it only for the local "
                                     f"sandbox provider, so put it in a store that provider can reach (`{put_command(kind)}`)")
                continue
            artifacts = ([env.db_docker_image_artifact, env.db_web_docker_image_artifact,
                          env.db_mcp_docker_image_artifact] if isinstance(env, ServiceDBEnv) else [env.docker_image_artifact])
            images = [_image(f"the {kind} env {env_id!r}'s image {image.id!r}", image) for image in artifacts]
            if not isinstance(provider, ModalSandboxProvider):  # whose containers run by name, or are swapped out
                self._loadable(where, [provider], images)
            elif kind != GATEWAY:  # whose image Modal builds when it's only a build context
                self._by_name(where, [provider], images)
            self._reachable(where, [provider], images)

    def _use(self, where: str, links: list[SandboxProvider], by_name: list[SandboxProvider],
             images: list[_Image]) -> None:
        for image in images:
            if image.built is not None:
                self.uses.setdefault(image.built, []).append((where, image.what, links, by_name))

    def _contexts(self) -> frozenset[str]:
        """The images the bundle builds that it writes as build contexts: those every deploy runs on providers that
        build them. One its deploys run only on the local provider, or that none runs, is built on this machine."""
        contexts = set()
        for image_id, uses in self.uses.items():
            built = self.built[image_id]
            source = f"{relative(self.plan.bundle.bundle.root, built.entry.path)}/{built.dockerfile}"
            local, vms = [], []
            for where, what, links, by_name in uses:
                for link in links:
                    if isinstance(link, LocalSandboxProvider):
                        local.append((where, what, links, link))
                    elif link in by_name or not _builds(link):
                        how = "runs it by name" if link in by_name else "can't build it from its build context"
                        self._problem(where, f"deploys {what} on the {_shown(link)} sandbox provider, which {how}, and "
                                             f"the bundle builds it from {source}, which a sandbox runs only once it's "
                                             "built on this machine or by the provider it runs on; run it with "
                                             "--sandbox local, or on a provider that builds it, such as --sandbox modal")
                    else:
                        vms.append((where, what, links, link))
            if local and vms:
                (where, what, links, link), (elsewhere, _, _, vm) = local[0], vms[0]
                on = (f"{_named(links)} sandbox provider" if elsewhere == where else
                      f"{_shown(link)} sandbox provider, and {elsewhere} on the {_shown(vm)} one")
                self._problem(where, f"deploys {what} on the {on}: the bundle builds it from {source} on this machine "
                                     "for the local provider, and as a build context for one that builds it, and a "
                                     "run writes it one way; run its deploys on one kind, such as --sandbox local or "
                                     "--sandbox modal_vm")
            elif vms:
                contexts.add(image_id)
        return frozenset(contexts)

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
            self._problem(where, f"{unnamed}, so it deploys the default, {agent_id!r}, and {problem}")

    def _problem(self, where: str, problem: str) -> None:
        """Record ``problem`` at the deploy ``where``. One found at several deploys, such as an infra env's image every
        gateway deploy runs, is reported once, naming the first of them."""
        self.problems.setdefault(problem, {})[where] = None


def _stored_images(env: Env) -> tuple[list[_Image], bool]:
    """The images a store env runs through a gateway, and whether it has websites."""
    topology = _gateway_topology(env)
    images = [*topology.mcp_server_images, *(topology.website_images or [])]
    return [_image(f"env {env.id!r}'s image {image.id!r}", image) for image in images], bool(topology.website_configs)


def _state_type(step: DeployEnvTaskStep) -> str | None:
    """The type of the env state store the deploy runs on: an attached instance's own, else the one it creates."""
    if step.env_state_instance_id is None:
        return step.env_state_type or LOCAL_POSTGRES_STATE_TYPE
    try:
        return get_env_state_instance_store().get(step.env_state_instance_id).state_type
    except NotFoundError:
        return None  # the deploy refuses an instance that isn't there


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


def _builds(provider: SandboxProvider) -> bool:
    """Whether ``provider`` builds an image that's only a build context: in a VM it creates, or itself before running
    it (``prepare_image``), as Modal does."""
    return _creates_vms(provider) or type(provider).prepare_image is not SandboxProvider.prepare_image


def _shown(provider: SandboxProvider) -> str:
    return _named(_links(provider))


def _named(links: list[SandboxProvider]) -> str:
    return repr(",".join(_name(link) for link in links))


def _name(provider: SandboxProvider) -> str:
    """The name ``provider`` is registered under, as ``--sandbox`` names it."""
    cls = type(provider)
    for name, section in get_config().sandbox_registry().items():
        impl = section.get("impl")
        if impl is cls or impl == f"{cls.__module__}:{cls.__qualname__}":
            return name
    return cls.__name__


def _image(what: str, image: DockerImageArtifact) -> _Image:
    return _Image(what, _local_only(image), image.load_problem(), image.by_name_problem())


def _local_only(image: DockerImageArtifact | str) -> str | None:
    """Why only this machine has ``image``: its reference names a registry on this machine, or it's saved in this
    machine's object store. None when neither."""
    ref = image if isinstance(image, str) else image.image_name
    if is_loopback_host(registry_host_from_ref(ref)):
        return f"{ref} is in a registry on this machine"
    if not isinstance(image, str) and image.tar_gz_object_url and urlparse(image.tar_gz_object_url).scheme == "file":
        return f"{image.id!r} is saved in this machine's object store"
    return None
