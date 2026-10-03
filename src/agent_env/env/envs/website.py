"""Website environment with frontend and backend containers."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Callable, ClassVar, Optional

from agentenv_protocol import FilePart, client as protocol_v1
from agent_env.artifact import Artifact, DockerImageArtifact, EnvironmentArtifact
from agent_env.artifact.artifacts.docker_image import GitHubBuildResult, ProgressCallback, refuse_local_github_build
from agent_env.env.env import Env, gateway_url_of
from agent_env.store.ids import derive_id
from agent_env.env import legacy_protocol
from agent_env.env.envs._deployment import (
    as_builtin, builtin_provider_for, close_deployed, close_replaced, deploy_refusal, deploy_through_provider, host_staging_refusal,
    load_by_signed_url, plugin_provider_like_a_builtin, provider_or_class,
)
from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT, GatewayMode
from agent_env.attribution import Attribution
if TYPE_CHECKING:
    from agent_env.env.env import DeployedEnv
    from agent_env.providers.env_providers.env_provider import EnvironmentProvider

logger = logging.getLogger(__name__)




class WebsiteEnv(Env):
    type: ClassVar[str] = "website"
    description = "Website environment with frontend and backend containers, deployed through the environment provider its env_provider_type names"

    def __init__(
        self,
        id: str,
        version: Optional[int],
        backend_docker_image_artifact: DockerImageArtifact,
        frontend_docker_image_artifact: DockerImageArtifact,
        environment_name: Optional[str] = None,
        *,
        metadata: Optional[dict[str, str]] = None,
        env_provider_type: str = "gateway",
    ):
        super().__init__(id, version, metadata=metadata)
        if not environment_name:
            raise ValueError("environment_name cannot be empty")
        if not env_provider_type:
            raise ValueError("env_provider_type cannot be empty")
        self.backend_docker_image_artifact = backend_docker_image_artifact
        self.frontend_docker_image_artifact = frontend_docker_image_artifact
        self.environment_name = environment_name
        self.env_provider_type = env_provider_type
        self._sandbox = None
        # As for an MCPServerEnv: a built-in's provider is built with the env, a plugin's when it deploys.
        self._env_provider: Optional[EnvironmentProvider] = builtin_provider_for(env_provider_type)
        self._gateway_url = None
        self._deployed: Optional[DeployedEnv] = None
        self._replaced: list[tuple] = []  # the provider and sandbox of each deployment a later deploy replaced, for close()

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["backend_docker_image_artifact"] = {
            "id": self.backend_docker_image_artifact.id,
            "version": self.backend_docker_image_artifact.version,
            "type": self.backend_docker_image_artifact.type,
        }
        base["frontend_docker_image_artifact"] = {
            "id": self.frontend_docker_image_artifact.id,
            "version": self.frontend_docker_image_artifact.version,
            "type": self.frontend_docker_image_artifact.type,
        }
        base["service_name"] = self.environment_name
        base["environment_name"] = self.environment_name  # dual-write with service_name during the rename
        base["env_provider_type"] = self.env_provider_type
        return base

    @classmethod
    def from_dict(cls, data: dict) -> WebsiteEnv:
        backend_ref = data["backend_docker_image_artifact"]
        backend_artifact = Artifact.get(backend_ref["id"], version=backend_ref["version"])
        frontend_ref = data["frontend_docker_image_artifact"]
        frontend_artifact = Artifact.get(frontend_ref["id"], version=frontend_ref["version"])
        return cls(
            id=data["id"],
            version=data.get("version"),
            backend_docker_image_artifact=backend_artifact,
            frontend_docker_image_artifact=frontend_artifact,
            environment_name=data["environment_name"] if "environment_name" in data else data["service_name"],
            metadata=data.get("metadata", {}),
            env_provider_type=data.get("env_provider_type", "gateway"),
        )

    async def deploy(self, ttl_seconds: int = 10800, disk_size_gb: float = 10, gateway_mode: GatewayMode = GatewayMode.PERFORMANCE, cpu: float | None = None, memory_mb: int | None = None, sandbox_type: str | None = None, priority: Optional[int] = None, env_state_type: str | None = None, env_state_instance_id: str | None = None, *, attribution: Optional[Attribution] = None) -> DeployedEnv:
        return await deploy_through_provider(
            self, environment_name=self.environment_name, ttl_seconds=ttl_seconds, sandbox_type=sandbox_type,
            disk_size_gb=disk_size_gb, gateway_mode=gateway_mode, cpu=cpu, memory_mb=memory_mb, priority=priority,
            env_state_type=env_state_type, env_state_instance_id=env_state_instance_id, attribution=attribution,
        )

    def deploy_refusal(self, **options) -> str | None:
        """Why deploy() would refuse these options before building anything, or None; raises, as deploy() does, for a type this process can't find."""
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
        file_artifact = environment_artifact.get_file_artifact()
        if builtin is None:
            await load_by_signed_url(self, file_artifact)
            return
        base_url = await legacy_protocol.v1_base_url(self._deployed, self._gateway_url, self.environment_name, mcp=False)
        if base_url is not None:
            from agent_env.env.gateway.constants import DATA_PLANE_LOAD_TIMEOUT_S
            container_path = await self._copy_artifact_into_container(file_artifact)
            await protocol_v1.reset_data(base_url)
            await protocol_v1.add_data(base_url, [FilePart(file={
                "uri": f"file://{container_path}",
                "mimeType": file_artifact.content_type,
                "name": file_artifact.filename,
            })], timeout=DATA_PLANE_LOAD_TIMEOUT_S)
        else:
            await self._load_environment_artifact_legacy(file_artifact)
        await builtin.install_changelog_triggers(self.environment_name)

    async def load_file_artifact_universe(self, file_artifact_universe: "Any", destination_path: Optional[str] = None,
                                          ) -> "LoadFileArtifactUniverseResult":
        if refusal := host_staging_refusal(self, "Staging files onto the env's host"):
            raise RuntimeError(refusal)
        return await super().load_file_artifact_universe(file_artifact_universe, destination_path)

    async def _copy_artifact_into_container(self, file_artifact) -> str:
        from agent_env.providers.env_providers.constants import AGENT_ENV_WEBSITE_BACKEND_SUFFIX, DOCKER_COMPOSE_PATH
        filename = file_artifact.filename
        backend_service = f"{self.environment_name}-{AGENT_ENV_WEBSITE_BACKEND_SUFFIX}"
        # Random suffix so concurrent loads on a shared sandbox (parallel /load-universe,
        # MCP+website sharing a service_name, repeated loads) don't race on a shared path.
        vm_path = f"/tmp/_artifact_{self.environment_name}-{uuid.uuid4().hex[:8]}-{filename}"
        container_path = f"/tmp/data/{filename}"
        compose = f"docker compose -f {DOCKER_COMPOSE_PATH}"

        await self._sandbox.load_s3_file(file_artifact.object_url, vm_path)
        await self._sandbox.exec_script(f"{compose} exec -T {backend_service} mkdir -p /tmp/data")
        await self._sandbox.exec_script(f"{compose} cp {vm_path} {backend_service}:{container_path}")
        await self._sandbox.exec_script(f"rm -f {vm_path}")
        return container_path

    async def _load_environment_artifact_legacy(self, file_artifact) -> None:
        container_path = await self._copy_artifact_into_container(file_artifact)
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

    async def close(self) -> None:
        await close_replaced(self)
        if self._env_provider is not None:
            try:
                await self._env_provider.close()
            except BaseException as e:
                logger.warning(f"Failed to close env provider: {e}")
        if self._sandbox is not None:
            try:
                await self._sandbox.terminate()
            except BaseException as e:
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
    ) -> WebsiteEnv:
        """Build backend and frontend Docker images from GitHub in parallel and create a WebsiteEnv."""
        refuse_local_github_build(id)
        backend, frontend = await asyncio.gather(
            DockerImageArtifact.put_from_github(
                id=f"website-backend-{id}",
                dockerfile_github_url=backend_dockerfile_github_url,
                docker_context_github_url=backend_docker_context_github_url,
                on_progress=on_backend_progress,
                github_token=github_token,
            ),
            DockerImageArtifact.put_from_github(
                id=f"website-frontend-{id}",
                dockerfile_github_url=frontend_dockerfile_github_url,
                docker_context_github_url=frontend_docker_context_github_url,
                on_progress=on_frontend_progress,
                github_token=github_token,
            ),
        )

        combined_metadata: dict[str, str] = {}
        user = os.getenv("USER")
        if user:
            combined_metadata["created_by"] = user
        try:
            combined_metadata["agent_env_version"] = pkg_version("agentenv-framework")
        except Exception:
            pass
        for prefix, build in [("backend", backend), ("frontend", frontend)]:
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
            backend_docker_image_artifact=backend.artifact,
            frontend_docker_image_artifact=frontend.artifact,
            environment_name=environment_name,
            metadata=combined_metadata,
            env_provider_type=env_provider_type,
        )
        return env
