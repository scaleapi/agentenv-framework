"""Website environment: one site, served by one server image that holds its pages and API. A site
built before the server image held its pages may still name a separate frontend image, which is deprecated.

A deploy reads whether the site is ready from its env card, the protocol layer: ``website/v1``'s ``ready``, as
``computer-app/v1``'s says of an app. ``health_path`` (env.toml's ``health``) is read only for a site whose card has no
``ready``, or that serves no card, and is deprecated.

A live website env (``live``) is a real site, at its ``domains``, its pages reaching the ``allow_hosts`` too: its server
serves none of its pages, only its card, data plane, readiness and session, so a computer reaches the real site over
its own network, and a gateway gives it no site route. The env keeps them, as it keeps ``domains``, since a computer's
network is fixed before any card is read. ``settings`` are what its server is handed as ``WEBSITE_SETTINGS_ENV``,
each of its ``secret_settings`` (its ``Settings``' ``SecretStr`` fields) a ``secret:`` or ``env:`` reference, resolved only
when its container is written."""

from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import shlex
import re
import uuid
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

from agentenv_protocol import client as protocol_v1
from agent_env.artifact import Artifact, DockerImageArtifact, EnvironmentArtifact
from agent_env.config import ENV_REF_PREFIX, SECRET_REF_PREFIX, is_bare_reference
from agent_env.artifact.artifacts.docker_image import ProgressCallback, refuse_local_github_build
from agent_env.entity.refs import EntityRef
from agent_env.env.env import Env, gateway_url_of
from agent_env.env.envs.computer.utils.sites import name_problem
from agent_env.store.ids import derive_id
from agent_env.utils.card_naming import WebsiteSource
from agent_env.env import legacy_protocol
from agent_env.env.envs._deployment import (
    as_builtin, builtin_provider_for, close_deployed, close_replaced, deploy_refusal, deploy_through_provider, host_staging_refusal,
    file_parts, legacy_load_file, load_by_signed_url, plugin_provider_like_a_builtin, provider_or_class, reset_server_data,
    staged_paths,
)
from agent_env.env.gateway import GatewayMode
from agent_env.attribution import Attribution
from agent_env.providers.env_providers.constants import AGENT_ENV_WEBSITE_HEALTH_PATH
from agent_env.utils.paths import validate_file_path
if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.entity.export import ExportContext
    from agent_env.env.env import DeployedEnv, LoadFileArtifactUniverseResult
    from agent_env.providers.env_providers.env_provider import EnvironmentProvider

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_PATH = AGENT_ENV_WEBSITE_HEALTH_PATH
_HEALTH_PATH = re.compile(r"/[A-Za-z0-9._~/-]*")


class WebsiteEnv(Env):
    type: ClassVar[str] = "website"
    description = "Website environment: one server image serving a site's pages and API, deployed through the environment provider its env_provider_type names"
    toml_refs: ClassVar[tuple[EntityRef, ...]] = (EntityRef.artifact("backend_image", artifact_type="docker_image"),
                                                  EntityRef.artifact("frontend_image", artifact_type="docker_image",
                                                                     optional=True))
    toml_keys: ClassVar[dict[str, type]] = {"backend_image": object, "frontend_image": object, "environment_name": str,
                                            "health": str, "domains": list, "env_provider_type": str, "live": bool,
                                            "allow_hosts": list, "settings": dict}

    def __init__(
        self,
        id: str,
        version: Optional[int],
        server_docker_image_artifact: Optional[DockerImageArtifact] = None,
        frontend_docker_image_artifact: Optional[DockerImageArtifact] = None,
        environment_name: Optional[str] = None,
        *,
        backend_docker_image_artifact: Optional[DockerImageArtifact] = None,
        health_path: str = DEFAULT_HEALTH_PATH,
        domains: Optional[list[str]] = None,
        metadata: Optional[dict[str, str]] = None,
        env_provider_type: str = "gateway",
        live: bool = False,
        allow_hosts: Optional[list[str]] = None,
        settings: Optional[dict[str, Any]] = None,
        secret_settings: Optional[list[str]] = None,
    ):
        super().__init__(id, version, metadata=metadata)
        if not environment_name:
            raise ValueError("environment_name cannot be empty")
        if not env_provider_type:
            raise ValueError("env_provider_type cannot be empty")
        if server_docker_image_artifact is not None and backend_docker_image_artifact is not None:
            raise ValueError("a website env names its server image once, as server_docker_image_artifact or as its "
                             "older name backend_docker_image_artifact")
        self.server_docker_image_artifact = server_docker_image_artifact or backend_docker_image_artifact
        self.frontend_docker_image_artifact = frontend_docker_image_artifact
        if problem := health_path_problem(health_path):
            raise ValueError(f"website env {id!r}: {problem}")
        self.health_path = health_path
        if problem := domains_problem(domains or []):
            raise ValueError(f"website env {id!r}: {problem}")
        self.domains = list(domains or [])
        if problems := live_problems(live, self.domains, allow_hosts or [], settings or {}, secret_settings or []):
            raise ValueError(f"website env {id!r}: {'; '.join(problems)}")
        self.live = live
        self.allow_hosts = list(allow_hosts or [])
        self.settings = dict(settings or {})
        self.secret_settings = list(secret_settings or [])
        self.environment_name = environment_name
        self.env_provider_type = env_provider_type
        self._sandbox = None
        # As for an MCPServerEnv: a built-in's provider is built with the env, a plugin's when it deploys.
        self._env_provider: Optional[EnvironmentProvider] = builtin_provider_for(env_provider_type)
        self._gateway_url = None
        self._deployed: Optional[DeployedEnv] = None
        self._replaced: list[tuple] = []  # the provider and sandbox of each deployment a later deploy replaced, for close()

    def live_websites(self) -> list[dict]:
        """The live site it is, as a run records one: its env id, environment name, domains and ``allow_hosts``;
        none for a synthetic site."""
        if not self.live:
            return []
        return [{"env_id": self.id, "environment_name": self.environment_name, "domains": list(self.domains),
                 "allow_hosts": list(self.allow_hosts)}]

    @property
    def backend_docker_image_artifact(self) -> DockerImageArtifact:
        """The server image, under the name it had while a site's pages came from a separate frontend image."""
        return self.server_docker_image_artifact

    def to_dict(self) -> dict:
        base = super().to_dict()
        server = {"id": self.server_docker_image_artifact.id, "version": self.server_docker_image_artifact.version,
                  "type": self.server_docker_image_artifact.type}
        base["server_docker_image_artifact"] = server
        base["backend_docker_image_artifact"] = server
        if self.frontend_docker_image_artifact is not None:
            base["frontend_docker_image_artifact"] = {
                "id": self.frontend_docker_image_artifact.id,
                "version": self.frontend_docker_image_artifact.version,
                "type": self.frontend_docker_image_artifact.type,
            }
        base["health_path"] = self.health_path
        if self.domains:
            base["domains"] = list(self.domains)
        if self.live:
            base["live"] = True
        if self.allow_hosts:
            base["allow_hosts"] = list(self.allow_hosts)
        if self.settings:
            base["settings"] = dict(self.settings)
        if self.secret_settings:
            base["secret_settings"] = list(self.secret_settings)
        base["service_name"] = self.environment_name
        base["environment_name"] = self.environment_name  # dual-write with service_name during the rename
        base["env_provider_type"] = self.env_provider_type
        return base

    @classmethod
    def from_dict(cls, data: dict) -> WebsiteEnv:
        server_ref = data.get("server_docker_image_artifact") or data["backend_docker_image_artifact"]
        frontend_ref = data.get("frontend_docker_image_artifact")
        return cls(
            id=data["id"],
            version=data.get("version"),
            server_docker_image_artifact=Artifact.get(server_ref["id"], version=server_ref["version"]),
            frontend_docker_image_artifact=(Artifact.get(frontend_ref["id"], version=frontend_ref["version"])
                                            if frontend_ref else None),
            environment_name=data["environment_name"] if "environment_name" in data else data["service_name"],
            health_path=data.get("health_path", DEFAULT_HEALTH_PATH),
            domains=data.get("domains") or [],
            metadata=data.get("metadata", {}),
            env_provider_type=data.get("env_provider_type", "gateway"),
            live=bool(data.get("live", False)),
            allow_hosts=data.get("allow_hosts") or [],
            settings=data.get("settings") or {},
            secret_settings=data.get("secret_settings") or [],
        )

    @classmethod
    def put(cls, **kwargs: Any) -> WebsiteEnv:
        """Write the website env, warning when it names a separate frontend image: a site's server image serves its
        pages, and a frontend image is deprecated, still deployed for the sites built on one."""
        if kwargs.get("frontend_docker_image_artifact") is not None:
            logger.warning(f"website env {kwargs.get('id')!r} names a separate frontend image, which is deprecated: "
                           "build the site's pages into its server image instead")
        return super().put(**kwargs)

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> WebsiteEnv:
        """Write the env authored as ``data`` (its env.toml, with ``backend_image``, the server image, and any
        ``frontend_image`` resolved to image artifacts' ids) under ``ctx.id`` and return it."""
        fields = cls.accept_toml(data, ctx)
        frontend = fields.get("frontend_image")
        return cls.put(id=ctx.id,
                       server_docker_image_artifact=ctx.artifact(fields["backend_image"], DockerImageArtifact),
                       frontend_docker_image_artifact=ctx.artifact(frontend, DockerImageArtifact) if frontend else None,
                       environment_name=fields["environment_name"],
                       health_path=fields.get("health", DEFAULT_HEALTH_PATH),
                       domains=fields.get("domains") or [],
                       env_provider_type=fields["env_provider_type"],
                       live=fields.get("live", False),
                       allow_hosts=fields.get("allow_hosts") or [],
                       settings=fields.get("settings") or {},
                       secret_settings=fields.get("secret_settings") or [])

    def to_toml(self, out: ExportContext) -> dict:
        """The env.toml of a folder this env's images are built from, which ``from_toml`` reads back as it."""
        toml = {"type": self.type, "environment_name": self.environment_name,
                "backend_image": out.image(self.server_docker_image_artifact, "backend_image")}
        if self.frontend_docker_image_artifact is not None:
            toml["frontend_image"] = out.image(self.frontend_docker_image_artifact, "frontend_image")
        if self.health_path != DEFAULT_HEALTH_PATH:
            toml["health"] = self.health_path
        if self.domains:
            toml["domains"] = list(self.domains)
        if self.live:
            toml["live"] = True
        if self.allow_hosts:
            toml["allow_hosts"] = list(self.allow_hosts)
        if self.settings:
            toml["settings"] = dict(self.settings)
        return {**toml, **self.provider_toml(out, self.env_provider_type)}

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an env.toml, a website env takes, ``backend_image`` naming its server image, its
        environment_name read from that image's @environment_card when it isn't set. Raises BundleError listing every
        problem."""
        health = data.get("health", DEFAULT_HEALTH_PATH)
        if isinstance(health, str) and (problem := health_path_problem(health)):
            ctx.refuse([ctx.config_problem(problem)])
        domains = data.get("domains", [])
        if isinstance(domains, list) and (problem := domains_problem(domains)):
            ctx.refuse([ctx.config_problem(problem)])
        fields = ctx.accept_env(data, cls, named_by="backend_image")
        declared, problems = declared_live(ctx.website_source("backend_image"), live=fields.get("live"),
                                           domains=fields.get("domains"), allow_hosts=fields.get("allow_hosts"))
        problems += live_problems(declared["live"], declared.get("domains", fields.get("domains") or []),
                                  declared["allow_hosts"], fields.get("settings") or {}, declared["secret_settings"])
        fields.update({key: value for key, value in declared.items() if value})
        if problems:
            ctx.refuse([ctx.config_problem(problem) for problem in problems])
        return fields

    async def deploy(self, ttl_seconds: int = 10800, disk_size_gb: float = 10, gateway_mode: GatewayMode = GatewayMode.PERFORMANCE, cpu: float | None = None, memory_mb: int | None = None, sandbox_type: str | None = None, env_state_type: str | None = None, env_state_instance_id: str | None = None, artifact_id: str | None = None, artifact_version: int | None = None, *, attribution: Optional[Attribution] = None) -> DeployedEnv:
        return await deploy_through_provider(
            self, environment_name=self.environment_name, ttl_seconds=ttl_seconds, sandbox_type=sandbox_type,
            disk_size_gb=disk_size_gb, gateway_mode=gateway_mode, cpu=cpu, memory_mb=memory_mb,
            env_state_type=env_state_type, env_state_instance_id=env_state_instance_id, attribution=attribution,
            artifact_id=artifact_id, artifact_version=artifact_version,
        )

    def deploy_refusal(self, *, sandbox_type: str | None = None, **options) -> str | None:
        """Why deploy() would refuse these options before building anything, or None; raises, as deploy() does, for a type this process can't find.
        ``sandbox_type`` is deploy()'s own, which no provider takes."""
        return deploy_refusal(self, provider_or_class(self), options)

    @classmethod
    async def from_deployed_env(cls, deployed: DeployedEnv) -> WebsiteEnv:
        env = Env.get(deployed.env_id, deployed.env_version)
        if not isinstance(env, WebsiteEnv):
            raise TypeError(f"Expected WebsiteEnv, got {type(env).__name__}")
        if env._env_provider is None:
            env._env_provider = plugin_provider_like_a_builtin(env.env_provider_type)
        if builtin := as_builtin(env._env_provider):
            env._sandbox = await builtin._reattach(env, deployed)
        env._gateway_url = gateway_url_of(deployed)
        env._instance_id = deployed.instance_id
        env._deployed = deployed
        return env

    async def load_environment_artifact(self, environment_artifact: EnvironmentArtifact) -> None:
        builtin = as_builtin(self._env_provider)
        # A built-in's deploy leaves a sandbox and gateway to stage through; a plugin's leaves only its record.
        if (self._sandbox is None or self._gateway_url is None) if builtin else self._deployed is None:
            raise RuntimeError("Environment not deployed - call deploy() first")
        if environment_artifact.environment_name != self.environment_name:
            raise ValueError(
                f"EnvironmentArtifact environment_name '{environment_artifact.environment_name}' "
                f"does not match env environment_name '{self.environment_name}'"
            )
        if builtin is None:
            await load_by_signed_url(self, environment_artifact)
            return
        base_url = await legacy_protocol.v1_base_url(self._deployed, self._gateway_url, self.environment_name, mcp=False)
        if base_url is not None:
            from agent_env.env.gateway.constants import DATA_PLANE_LOAD_TIMEOUT_S
            files = environment_artifact.get_file_artifacts()
            folder, paths = staged_paths(environment_artifact, files)
            try:
                staged = {path: f"file://{await self._copy_artifact_into_container(file, paths[path])}"
                          for path, file in files.items()}
                await protocol_v1.reset_data(base_url)
                await protocol_v1.add_data(base_url, file_parts(environment_artifact, files, staged),
                                           timeout=DATA_PLANE_LOAD_TIMEOUT_S)
            finally:
                if folder is not None:
                    await self._remove_from_container(folder)
        else:
            await self._load_environment_artifact_legacy(legacy_load_file(self, environment_artifact))
        await builtin.install_changelog_triggers(self.environment_name)

    async def load_file_artifact_universe(self, file_artifact_universe: "Any", destination_path: Optional[str] = None,
                                          ) -> "LoadFileArtifactUniverseResult":
        if refusal := host_staging_refusal(self, "Staging files onto the env's host"):
            raise RuntimeError(refusal)
        return await super().load_file_artifact_universe(file_artifact_universe, destination_path)

    async def _copy_artifact_into_container(self, file_artifact, path: str) -> str:
        from agent_env.providers.env_providers.constants import AGENT_ENV_WEBSITE_BACKEND_SUFFIX, DOCKER_COMPOSE_PATH
        validate_file_path(path)
        backend_service = f"{self.environment_name}-{AGENT_ENV_WEBSITE_BACKEND_SUFFIX}"
        # Random suffix so concurrent loads on a shared sandbox (parallel /load-universe,
        # MCP+website sharing a service_name, repeated loads) don't race on a shared path.
        vm_path = f"/tmp/_artifact_{self.environment_name}-{uuid.uuid4().hex[:8]}-{posixpath.basename(path)}"
        container_path = f"/tmp/data/{path}"
        compose = f"docker compose -f {shlex.quote(DOCKER_COMPOSE_PATH)}"

        await self._sandbox.load_object_file(file_artifact.object_url, vm_path)
        await self._sandbox.exec_script(
            f"{compose} exec -T {shlex.quote(backend_service)} mkdir -p {shlex.quote(posixpath.dirname(container_path))}")
        await self._sandbox.exec_script(f"{compose} cp {shlex.quote(vm_path)} {shlex.quote(f'{backend_service}:{container_path}')}")
        await self._sandbox.exec_script(f"rm -f {shlex.quote(vm_path)}")
        return container_path

    async def _remove_from_container(self, path: str) -> None:
        """Remove what a load staged at ``/tmp/data/<path>``, best-effort: a load that finished stands either way."""
        from agent_env.providers.env_providers.constants import AGENT_ENV_WEBSITE_BACKEND_SUFFIX, DOCKER_COMPOSE_PATH
        backend_service = f"{self.environment_name}-{AGENT_ENV_WEBSITE_BACKEND_SUFFIX}"
        container_path = f"/tmp/data/{path}"
        try:
            await self._sandbox.exec_script(f"docker compose -f {shlex.quote(DOCKER_COMPOSE_PATH)} exec -T "
                                            f"{shlex.quote(backend_service)} rm -rf {shlex.quote(container_path)}")
        except Exception as e:  # noqa: BLE001 -- cleanup, not the load
            logger.warning(f"[{self.environment_name}] could not remove {container_path} after a load (ignored): {e}")

    async def _load_environment_artifact_legacy(self, file_artifact) -> None:
        container_path = await self._copy_artifact_into_container(file_artifact, file_artifact.filename)
        base_url = legacy_protocol.environment_base_url(self._gateway_url, self.environment_name, mcp=False)
        from agent_env.env.gateway.constants import DATA_PLANE_LOAD_TIMEOUT_S
        await legacy_protocol.reset_via_rest(base_url, timeout=60)
        try:
            await legacy_protocol.add_via_rest(base_url, container_path, timeout=DATA_PLANE_LOAD_TIMEOUT_S)
        except Exception as e:
            logger.error(
                "load_environment_artifact: /api/add failed after /api/reset succeeded; "
                f"environment '{self.environment_name}' backend is in empty state: {e}"
            )
            raise

    async def reset(self, deployed: DeployedEnv) -> None:
        """Empty what loads put into the website, attached to ``deployed`` as a load is, so a task loads it again after;
        NotImplementedError for one serving no reset, which takes no load either."""
        if not await reset_server_data(await type(self).from_deployed_env(deployed), mcp=False):
            raise NotImplementedError(f"website '{self.environment_name}' serves no reset")

    async def close(self) -> None:
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

    async def validate(self, on_progress: Optional[Callable[[str], None]] = None) -> str:
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
                await close_deployed(deployed_env, WebsiteEnv)
            except Exception as e:
                logger.warning(f"Failed to clean up env {deployed_env.env_id}: {e}")
        return context.instance_id

    @classmethod
    async def put_from_github(
        cls,
        id: str,
        backend_dockerfile_github_url: str,
        backend_docker_context_github_url: str | None = None,
        frontend_dockerfile_github_url: str = "",
        frontend_docker_context_github_url: str | None = None,
        environment_name: str | None = None,
        *,
        metadata: dict[str, str] | None = None,
        on_backend_progress: ProgressCallback | None = None,
        on_frontend_progress: ProgressCallback | None = None,
        github_token: str | None = None,
        env_provider_type: str = "gateway",
        health_path: str = DEFAULT_HEALTH_PATH,
        domains: list[str] | None = None,
        live: bool = False,
        allow_hosts: list[str] | None = None,
        settings: dict[str, Any] | None = None,
        secret_settings: list[str] | None = None,
    ) -> WebsiteEnv:
        """Build the server image, from ``backend_dockerfile_github_url``, and any deprecated frontend image from
        GitHub in parallel, and create a WebsiteEnv."""
        refuse_local_github_build(id)
        builds = [DockerImageArtifact.put_from_github(
            id=derive_id(id, "backend_image"),
            dockerfile_github_url=backend_dockerfile_github_url,
            docker_context_github_url=backend_docker_context_github_url,
            on_progress=on_backend_progress,
            github_token=github_token,
        )]
        if frontend_dockerfile_github_url:
            builds.append(DockerImageArtifact.put_from_github(
                id=derive_id(id, "frontend_image"),
                dockerfile_github_url=frontend_dockerfile_github_url,
                docker_context_github_url=frontend_docker_context_github_url,
                on_progress=on_frontend_progress,
                github_token=github_token,
            ))
        backend, *frontend = await asyncio.gather(*builds)

        combined_metadata: dict[str, str] = {}
        user = os.getenv("USER")
        if user:
            combined_metadata["created_by"] = user
        try:
            combined_metadata["agent_env_version"] = pkg_version("agentenv-framework")
        except Exception:
            pass
        for prefix, build in [("backend", backend), *(("frontend", build) for build in frontend)]:
            combined_metadata[f"{prefix}_dockerfile_github_url"] = build.dockerfile_github_url
            combined_metadata[f"{prefix}_github_owner"] = build.github_owner
            combined_metadata[f"{prefix}_github_repo"] = build.github_repo
            combined_metadata[f"{prefix}_github_commit"] = build.github_commit
            if build.docker_context_github_url:
                combined_metadata[f"{prefix}_docker_context_github_url"] = build.docker_context_github_url
            if build.github_ref:
                combined_metadata[f"{prefix}_github_ref"] = build.github_ref
        if metadata:
            combined_metadata.update(metadata)

        env = cls.put(
            id=id,
            server_docker_image_artifact=backend.artifact,
            frontend_docker_image_artifact=frontend[0].artifact if frontend else None,
            environment_name=environment_name,
            metadata=combined_metadata,
            env_provider_type=env_provider_type,
            health_path=health_path,
            domains=domains,
            live=live,
            allow_hosts=allow_hosts,
            settings=settings,
            secret_settings=secret_settings,
        )
        return env


def health_path_problem(health_path: str) -> str | None:
    """Why ``health_path`` can't be where a site answers once it's usable, or None: it's a path of plain characters,
    since the gateway writes it into a health check."""
    if _HEALTH_PATH.fullmatch(health_path):
        return None
    return f"health: {health_path!r} is not a path of letters, digits, '.', '_', '~', '-' and '/' starting with '/'"


def declared_live(source: Optional[WebsiteSource], *, live: Optional[bool] = None, domains: Optional[list[str]] = None,
                  allow_hosts: Optional[list[str]] = None) -> tuple[dict, list[str]]:
    """A website env's ``live``, ``allow_hosts``, ``secret_settings`` and, for a live one, ``domains``: each one
    given, else what ``source``, its server's ``@website(...)``, declares; and the problems, one for each value given
    that the source declares otherwise, since the env and the site's card would then disagree. A synthetic site's
    domains stay as given: without them its card's are read at deploy."""
    source = source or WebsiteSource()
    problems = [f"{key} is {given!r} here, but the site's @website(...) declares {key}={declared!r}"
                for key, given, declared in (("live", live, source.live), ("allow_hosts", allow_hosts, source.allow_hosts))
                if given is not None and declared is not None and given != declared]
    resolved = live if live is not None else bool(source.live)
    fields = {"live": resolved, "allow_hosts": list(allow_hosts if allow_hosts is not None else source.allow_hosts or []),
              "secret_settings": list(source.secret_settings)}
    if resolved:
        if domains and source.domains is not None and list(domains) != source.domains:
            problems.append(f"domains are {list(domains)!r} here, but the site's @website(...) declares "
                            f"domains={source.domains!r}")
        fields["domains"] = list(domains or source.domains or [])
    return fields, problems


def live_problems(live: Any, domains: list[str], allow_hosts: list[str], settings: dict, secret_settings: list[str]
                  ) -> list[str]:
    """Why these can't be a website env's: a live site names the domains the real site is at, its ``allow_hosts`` are
    host names as ``domains`` are and only a live site's, each setting's value is a string, number or boolean, and each
    of the secret settings given is a ``secret:`` or ``env:`` reference, never its value."""
    problems = []
    if not isinstance(live, bool):
        problems.append(f"live is true or false, not {live!r}")
    elif live and not domains:
        problems.append("a live website env names the domains the real site is at")
    if not isinstance(allow_hosts, list):
        problems.append(f"allow_hosts is a list of host names, not {allow_hosts!r}")
    elif allow_hosts:
        if live is not True:
            problems.append("allow_hosts are the other hosts a live site's pages reach; set live = true")
        if problem := domains_problem(allow_hosts):
            problems.append(problem.replace("domains:", "allow_hosts:", 1))
    if not isinstance(settings, dict):
        return [*problems, f"settings is a table of a setting's value by its name, not {settings!r}"]
    for name, value in settings.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            problems.append(f"settings: {name!r} isn't a setting's name")
        elif not isinstance(value, (str, int, float, bool)):
            problems.append(f"settings.{name} is a string, number or boolean, not {value!r}")
        elif name in secret_settings and not is_bare_reference(value):
            problems.append(f"settings.{name} is a secret, so it's a {SECRET_REF_PREFIX}KEY or {ENV_REF_PREFIX}NAME "
                            "reference with no ?default, never its value")
    return problems


def domains_problem(domains: list[str]) -> str | None:
    """Why ``domains`` can't be the names a site is reached by, or None: each a DNS name, as a computer's site table
    takes it, a wildcard only as its whole first label."""
    problems = [name_problem(name) if isinstance(name, str) else f"{name!r} is not a host name" for name in domains]
    return next((f"domains: {problem}" for problem in problems if problem), None)
