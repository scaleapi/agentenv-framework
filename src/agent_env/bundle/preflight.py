"""What a bundle run checks about where its tasks deploy, before it writes anything, and the infra envs it builds.

Each deploy, an env's, an agent's, a sandbox's or a rubrics judge's, runs on its effective sandbox provider: the run's
``--sandbox``, else the step's own field, else the config default, resolved as the step resolves it when it runs. A
comma-separated provider is a chain whose deploys skip a provider that refuses them and fall back from one provider to
the next, so a deploy is refused only when every provider in it refuses it. A run is refused when a deploy would fail on
its provider: an image the provider can't run, by the check the deploy itself makes, such as one a sandbox elsewhere
would pull from this machine's registry; a VM asked of a provider that can't create one; or containers on the local
provider without a Docker daemon. An image's tar.gz or build context
that only this machine's object store holds isn't refused: agent-env pushes it to the sandbox. A deploy_agent step that
names no agent, and a judge that names none, deploy the configured default agent, which must be in the store. An env the
bundle writes is checked from its planned env.toml.

An image the bundle builds is written in the form its deploys can run: built on this machine when they deploy on the
local provider, and as its build context alone when they deploy on providers that build it, in a VM or, as Modal does,
themselves. One version is one of the two, so a run deploying it both ways is refused, as is one deploying it on a
provider that runs an image by name.

A gateway deploy runs on infra envs the store may not hold yet: the gateway, the service-db its local Postgres state
runs from, and the website browser it adds for websites. In local stores the run builds those once its writes are done
(``agent_env.env.bootstrap``), for this machine's platform and with their build contexts, which a VM or Modal running
another platform builds them from; so they're named here, and refused here when they can't be built.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import PUT_AGAIN, DockerImageArtifact, pull_problem
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
    needs_docker,
    put_command,
    stores_are_local,
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
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_VM,
    Accepts,
    SandboxProvider,
    build_sandbox_provider,
    get_agent_sandbox_provider,
    get_env_sandbox_provider,
    get_sandbox_provider,
)
from agent_env.store.base import NotFoundError
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep, linked_sandbox_problem
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.utils.docker_build import DEFAULT_BUILD_PLATFORM, docker_unreachable

from ._fs import relative
from .parse import BundleError, BundleKind
from .plan import Plan
from .resolve import BuiltImage, Reference, ReferencedImage, ResolvedEntry, build_step

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
    artifact: DockerImageArtifact | None = None  # the store's image, checked as each provider runs it
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
        self.docker_users: list[str] = []
        self.default_agent_users: list[tuple[str, str]] = []  # (where, what names no agent)
        self.written = {write.id for write in plan.writes}
        self.agents = {write.id: write.source for write in plan.writes if write.kind is BundleKind.AGENT}
        self.envs = {write.id: write.source for write in plan.writes if write.kind is BundleKind.ENV}
        self.built = {write.id: write.source for write in plan.writes if isinstance(write.source, BuiltImage)}
        # Each image a folder names in a registry, as the store will hold it, so it's checked as a stored one is.
        self.referenced = {write.id: DockerImageArtifact(id=write.id, description="", image_name=write.source.ref)
                           for write in plan.writes if isinstance(write.source, ReferencedImage)}
        # Each deploy of an image the bundle builds: (where, what, its provider's links, the images it accepts on each)
        self.uses: dict[str, list[tuple[str, str, list[SandboxProvider], Callable[[SandboxProvider], Accepts]]]] = {}

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
        contexts = self._contexts()
        self._default_agent()
        reason = docker_unreachable() if self.docker_users or infra else None
        if reason and self.docker_users:
            shown = ", ".join(self.docker_users[:_SHOWN_DOCKER_USERS])
            more = len(self.docker_users) - _SHOWN_DOCKER_USERS
            self.run_problems.append(f"the local sandbox provider runs containers for {shown}"
                                     f"{f' and {more} more' if more > 0 else ''}, and {reason}")
        elif reason:  # infra the run builds for deploys elsewhere
            self.run_problems.append(needs_docker(infra, reason))
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
        env_id, images, env_provider = step.env_id, deployment.images, deployment.provider
        elsewhere = functools.partial(self._gateway_elsewhere, env_id, env_provider, kinds) if GATEWAY in kinds else None
        taking = self._check(where, provider, env_provider.accepts, images, vms_load=GATEWAY in kinds, elsewhere=elsewhere)
        if any(link.ON_THIS_MACHINE for link in taking):
            self.infra |= kinds
            self.docker_users.append(where)
        elif taking and stores_are_local():
            self.infra |= kinds

    def _gateway_elsewhere(self, env_id: str, env_provider: type, kinds: set[str], link: SandboxProvider) -> list[str]:
        """Why ``link``, a provider not on this machine, can't run env ``env_id``'s gateway deploy: its sandboxes, or
        the infra envs of ``kinds`` it runs."""
        found = []
        if problem := env_provider.sandbox_problem(link, websites=WEBSITE_BROWSER in kinds):
            found.append(f"deploys env {env_id!r} on the {_shown(link)} sandbox provider, which {problem}; run it on a VM "
                         "provider, such as --sandbox local")
        return found + self._remote_infra(link, kinds if env_provider.in_vm(link) else kinds - {WEBSITE_BROWSER})

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
            return _Image(what, built=ref.id)
        if ref.id in self.referenced:
            return _Image(what, self.referenced[ref.id])
        if ref.id in self.written:
            return None  # another of the bundle's writes, which isn't an image: a plugin env's artifact
        try:
            return _Image(what, DockerImageArtifact.get(ref.id, self._planned_version(ref.kind, ref.id, ref.version)))
        except NotFoundError:
            return None  # the plan reports a store image that isn't there

    def _planned_version(self, kind: EntityKind, entity_id: str, version: int | None) -> int | None:
        """The version a write names a store entity at: its pin, else the one the plan read, which the writer pins."""
        return version if version is not None else self.plan.store_latest.get((kind, entity_id))

    def _agent(self, where: str, step: DeployAgentTaskStep, sandboxes: dict[str, DeploySandboxTaskStep]) -> None:
        accepts: Callable[[SandboxProvider], Accepts] = A2AAgent.accepts
        if step.sandbox_name:
            linked = sandboxes.get(step.sandbox_name)
            if linked is None:
                return  # the step itself refuses an agent linked to a sandbox the task doesn't deploy
            if problem := linked_sandbox_problem(step.sandbox_name, linked.sandbox_mode, linked.exposed_ports or ()):
                self._problem(where, problem)
                return
            provider = _provider(self.sandbox or linked.sandbox_type, get_sandbox_provider)
            accepts = functools.partial(A2AAgent.accepts, sandbox_mode=linked.sandbox_mode)  # the VM loads its image
        else:
            provider = _provider(self.sandbox or step.sandbox_type, get_agent_sandbox_provider)
        self._agent_deploy(where, provider, step.a2a_agent_id, step.a2a_agent_version, "names no agent", accepts)

    def _judge(self, where: str, step: RubricsVerifierTaskStep) -> None:
        if not step.use_agent_judge or step.agent_name is not None:
            return  # the direct LLM judge, or a judge the task deployed itself
        provider = _provider(self.sandbox or step.judge_sandbox_type, get_agent_sandbox_provider)
        self._agent_deploy(where, provider, step.judge_a2a_agent_id, None, "names no judge agent")

    def _agent_deploy(self, where: str, provider: SandboxProvider, agent_id: str | None, version: int | None,
                      unnamed: str, accepts: Callable[[SandboxProvider], Accepts] = A2AAgent.accepts) -> None:
        if agent_id is None:
            self.default_agent_users.append((where, unnamed))
            agent_id = get_config().get_default_a2a_agent_id()
        image = self._agent_image(agent_id, version)
        taking = self._check(where, provider, accepts, [image] if image else [], vms_load=True)
        if any(link.ON_THIS_MACHINE for link in taking):
            self.docker_users.append(where)

    def _sandbox(self, where: str, step: DeploySandboxTaskStep) -> None:
        provider = _provider(self.sandbox or step.sandbox_type, get_sandbox_provider)
        if step.sandbox_mode not in provider.SUPPORTED_SANDBOX_MODES:
            kind = "VM" if step.sandbox_mode == SANDBOX_MODE_VM else step.sandbox_mode
            self._problem(where, f"deploys a {kind} sandbox, and the {_shown(provider)} sandbox provider can't create a "
                                 f"{kind}; run it on one that can, such as --sandbox local")
            return
        if step.sandbox_mode == SANDBOX_MODE_VM:
            return
        if _local_link(provider):
            self.docker_users.append(where)
        if step.image and not _local_link(provider):  # one on this machine pulls from its registry, and a deploy skips the rest
            if problem := pull_problem(step.image, on_this_machine=False):
                self._problem(where, f"deploys the image {step.image} on the {_shown(provider)} sandbox provider: "
                                     f"{problem}; run it with --sandbox local")

    # What a deploy runs

    def _agent_image(self, agent_id: str, version: int | None) -> _Image | None:
        """The image the agent ``agent_id`` runs, or None when the store doesn't hold the agent or can't read it, which
        the plan or ``_default_agent`` reports."""
        what = f"agent {agent_id!r}'s image"
        if (agent := self.agents.get(agent_id)) is not None:
            image_id, image_version = parse_toml_ref(EntityKind.ARTIFACT, agent.config.get("image"))
            if image_id in self.built:
                return _Image(what, built=image_id)
            if image_id in self.referenced:
                return _Image(what, self.referenced[image_id])
            planned = self._planned_version(EntityKind.ARTIFACT, image_id, image_version)
            return _Image(what, DockerImageArtifact.get(image_id, planned))
        try:
            return _Image(what, A2AAgent.get(agent_id, version).docker_image_artifact)
        except (NotFoundError, ValueError, KeyError, TypeError):
            return None

    def _check(self, where: str, provider: SandboxProvider, accepts: Callable[[SandboxProvider], Accepts],
               images: list[_Image], *, vms_load: bool,
               elsewhere: Callable[[SandboxProvider], list[str]] | None = None) -> list[SandboxProvider]:
        """The providers the deploy ``where`` may run on that take it, refusing the deploy, naming why each refuses
        (``_link_refusals``), when none does: a deploy skips a provider that refuses it, so one that takes it is enough.
        One on this machine that takes it runs it on this machine's Docker; one it skips needs none. An image the
        bundle builds is noted with them, for ``_contexts`` to choose its form."""
        links = list(provider.links)
        refusals = [(link, self._link_refusals(link, accepts, images, vms_load=vms_load, elsewhere=elsewhere))
                    for link in links]
        if all(found for _, found in refusals):
            for _, found in refusals:
                for problem in found:
                    self._problem(where, problem)
        for image in images:
            if image.built is not None:
                self.uses.setdefault(image.built, []).append((where, image.what, links, accepts))
        return [link for link, found in refusals if not found]

    def _link_refusals(self, link: SandboxProvider, accepts: Callable[[SandboxProvider], Accepts], images: list[_Image],
                       *, vms_load: bool, elsewhere: Callable[[SandboxProvider], list[str]] | None) -> list[str]:
        """Why ``link`` can't take a deploy of ``images``: each it can't run as ``accepts`` says the deploy runs images
        there (``_refusal``), and, when it isn't on this machine, whatever ``elsewhere`` says of it. ``vms_load`` when
        the deploy loads images on a VM provider, as an agent's and a gateway's do."""
        found = [refusal for image in images if (refusal := self._refusal(link, accepts(link), image, vms_load=vms_load))]
        if elsewhere is not None and not link.ON_THIS_MACHINE:
            found += elsewhere(link)
        return found

    def _refusal(self, link: SandboxProvider, form: Accepts, image: _Image, *, vms_load: bool) -> str | None:
        """Why a deploy can't run ``image`` on ``link``, which takes images in ``form``, by the check the deploy makes
        (``Accepts.problem``), or None when it can: a VM loads an image's tar.gz, or pulls an image with none when its
        name names a registry it reaches, and a provider that runs an image by name can't run one that is only a build
        context, whose name is a tag only its build gives it, nor, from elsewhere, pull one from this machine's
        registry. A VM it would create checks the image's platform once it exists."""
        if image.artifact is None or (refusal := form.problem(image.artifact, on_this_machine=link.ON_THIS_MACHINE)) is None:
            return None
        if form.problem(image.artifact, on_this_machine=True) is None:  # refused only because the sandbox isn't here
            loads = (", or on a provider whose VMs load its tar.gz, such as --sandbox modal_vm"
                     if vms_load and image.artifact.tar_gz_object_url and form is not Accepts.LOADABLE
                     and image.artifact.platform_problem(DEFAULT_BUILD_PLATFORM) is None else "")
            if image.artifact.stored_under_build_tag:  # its remedy, with the CLI's flags
                return (f"deploys {image.what} on the {_shown(link)} sandbox provider: "
                        f"{image.artifact.build_tag_reason()}; run it with --sandbox local{loads}, or {PUT_AGAIN}")
            return (f"deploys {image.what} on the {_shown(link)} sandbox provider: {refusal}; run it with "
                    f"--sandbox local{loads}")
        if form is Accepts.LOADABLE:
            return f"deploys {image.what} on the {_shown(link)} sandbox provider, which can't load it: {refusal}"
        builds_itself = link.CONTAINER_ACCEPTS is Accepts.NAME_OR_CONTEXT
        builds = ("whose VMs build it, such as --sandbox modal_vm" if builds_itself
                  else "that builds it, such as --sandbox modal")
        return (f"deploys {image.what} on the {_shown(link)} sandbox provider, which runs it by name: {refusal}; run it "
                f"on a provider {builds}")

    def _remote_infra(self, link: SandboxProvider, kinds: set[str]) -> list[str]:
        """Why ``link``, a provider not on this machine, can't run the infra envs of ``kinds``. One the run builds, in
        local stores missing or built from other inputs, isn't checked: it's built anew, with its build context."""
        built = {build.kind for build in infra_to_build(kinds)} if stores_are_local() else set()
        found = []
        for kind in sorted(kinds - built):
            env_id = default_env_id(kind)
            try:
                env = Env.get(env_id)
            except NotFoundError:
                found.append(f"deploys on the {_shown(link)} sandbox provider, which needs the {kind} env {env_id!r}, "
                             f"and the store doesn't hold it; agent-env builds infra envs only into local stores, so put "
                             f"it with `{put_command(kind)}`")
                continue
            artifacts = ([env.db_docker_image_artifact, env.db_web_docker_image_artifact,
                          env.db_mcp_docker_image_artifact] if isinstance(env, ServiceDBEnv) else [env.docker_image_artifact])
            accepts = EnvironmentGatewayProvider.accepts(link)
            found += [refusal for image in artifacts
                      if (refusal := self._refusal(link, accepts, _Image(f"the {kind} env {env_id!r}'s image {image.id!r}",
                                                                         image), vms_load=True))]
        return found

    def _contexts(self) -> frozenset[str]:
        """The images the bundle builds that it writes as build contexts: those every deploy runs on providers that
        build them. One its deploys run only on the local provider, or that none runs, is built on this machine."""
        contexts = set()
        for image_id, uses in self.uses.items():
            built = self.built[image_id]
            source = f"{relative(self.plan.bundle.bundle.root, built.entry.path)}/{built.dockerfile}"
            local, vms = [], []
            for where, what, links, accepts in uses:
                for link in links:
                    if link.ON_THIS_MACHINE:
                        local.append((where, what, links, link))
                    elif accepts(link) is Accepts.NAME:
                        self._problem(where, f"deploys {what} on the {_shown(link)} sandbox provider, which runs it by "
                                             f"name, and the bundle builds it from {source}, which a sandbox runs only "
                                             "once it's built on this machine or by the provider it runs on; run it "
                                             "with --sandbox local, or on a provider that builds it, such as --sandbox "
                                             "modal")
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
    return [_Image(f"env {env.id!r}'s image {image.id!r}", image) for image in images], bool(topology.website_configs)


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


def _local_link(provider: SandboxProvider) -> bool:
    return any(link.ON_THIS_MACHINE for link in provider.links)


def _shown(provider: SandboxProvider) -> str:
    return _named(list(provider.links))


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

