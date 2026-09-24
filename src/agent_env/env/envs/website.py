"""Website environment with frontend and backend containers."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Callable, ClassVar, Optional

from agentenv_protocol import FilePart, WELL_KNOWN_PATH, client as protocol_v1
from agent_env.artifact import Artifact, DockerImageArtifact, EnvironmentArtifact
from agent_env.artifact.artifacts.docker_image import GitHubBuildResult, ProgressCallback
from agent_env.env.env import Env
from agent_env.env import legacy_protocol
from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT, GatewayMode
from agent_env.env.store import register_env_instance
from agent_env.attribution import Attribution
if TYPE_CHECKING:
    from agent_env.providers.gateway_provider import GatewayProvider, WebsiteConfig

logger = logging.getLogger(__name__)




class WebsiteEnv(Env):
    type: ClassVar[str] = "website"
    description = "Website environment with frontend and backend containers"

    def __init__(
        self,
        id: str,
        version: Optional[int],
        backend_docker_image_artifact: DockerImageArtifact,
        frontend_docker_image_artifact: DockerImageArtifact,
        environment_name: Optional[str] = None,
        service_version: Optional[int] = None,
        metadata: Optional[dict[str, str]] = None,
    ):
        super().__init__(id, version, metadata=metadata)
        if not environment_name:
            raise ValueError("environment_name cannot be empty")
        self.backend_docker_image_artifact = backend_docker_image_artifact
        self.frontend_docker_image_artifact = frontend_docker_image_artifact
        self.environment_name = environment_name
        # Coerced, not just tolerated: to_dict writes this key unconditionally.
        self.service_version = 1 if service_version is None else service_version
        self._sandbox = None
        from agent_env.providers.gateway_provider import GatewayProvider
        self._gateway_provider = GatewayProvider()
        self._gateway_url = None

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
        base["service_version"] = self.service_version  # no environment_version: deprecated, not renamed
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
            service_version=data.get("service_version", 1),
            metadata=data.get("metadata", {}),
        )

    async def deploy(self, ttl_seconds: int = 10800, disk_size_gb: float = 10, gateway_mode: GatewayMode = GatewayMode.PERFORMANCE, cpu: float | None = None, memory_mb: int | None = None, sandbox_type: str | None = None, priority: Optional[int] = None, env_state_type: str | None = None, env_state_instance_id: str | None = None, *, attribution: Optional[Attribution] = None) -> DeployedEnv:
        attribution = dict(attribution or {})
        from agent_env.env.env import DeployedEnv
        from agent_env.providers import DB_WEB_PORT, DB_MCP_PORT
        from agent_env.config import get_config

        config = get_config()
        service_db = Env.get(config.default_service_db_env_id)

        try:
            from agent_env.providers import build_sandbox_provider, get_env_sandbox_provider
            from agent_env.providers.gateway_provider import WebsiteConfig
            from agent_env.providers.state import acquire_state_for_deploy
            sandbox_provider = build_sandbox_provider(sandbox_type) if sandbox_type else get_env_sandbox_provider()
            website_config = WebsiteConfig(
                backend_image=self.backend_docker_image_artifact.image_name,
                frontend_image=self.frontend_docker_image_artifact.image_name,
                environment_name=self.environment_name,
            )

            state_instance = await acquire_state_for_deploy(
                env_state_type=env_state_type, ttl_seconds=ttl_seconds, name_hint=self.id, env_state_instance_id=env_state_instance_id,
            )
            result = await self._gateway_provider.create_gateway(
                sandbox_provider=sandbox_provider,
                mcp_servers=[],
                mcp_server_images=[],
                website_configs=[website_config],
                website_images=[self.backend_docker_image_artifact, self.frontend_docker_image_artifact],
                gateway_mode=gateway_mode,
                ttl_seconds=ttl_seconds,
                disk_size_gb=disk_size_gb,
                cpu=cpu, memory_mb=memory_mb,
                attribution=attribution,
                priority=priority,
                env_id=self.id,
                state_instance=state_instance,
            )
            self._sandbox = self._gateway_provider.sandbox
            self._gateway_url = result.gateway_url
            deployed_env = DeployedEnv(
                env_id=self.id,
                env_version=self.version,
                gateway_url=result.gateway_url,
                mcp_url=result.mcp_url,
                db_web_url=result.db_web_url,
                sandbox_id=self._sandbox.sandbox_id, sandbox_type=self._sandbox.type,
                db_mcp_url=result.db_mcp_url,
                environment_card_url=f"{result.gateway_url}{WELL_KNOWN_PATH}",
                environment_card=result.environment_card,
                environment_card_read_at_utc=result.environment_card_read_at_utc,
                website_frontend_urls=result.website_frontend_urls,
                gateway_mode=gateway_mode.value,
            )
            deployed_env = register_env_instance(deployed_env, ttl_seconds)
            self._instance_id = deployed_env.instance_id
            return deployed_env
        except BaseException:
            await self.close()
            raise

    @classmethod
    async def from_deployed_env(cls, deployed: DeployedEnv) -> WebsiteEnv:
        from agent_env.env.env import DeployedEnv
        env = Env.get(deployed.env_id, deployed.env_version)
        if not isinstance(env, WebsiteEnv):
            raise TypeError(f"Expected WebsiteEnv, got {type(env).__name__}")
        from agent_env.providers import build_sandbox_provider, get_env_sandbox_provider
        provider = build_sandbox_provider(deployed.sandbox_type) if deployed.sandbox_type else get_env_sandbox_provider()
        env._sandbox = await provider.get_sandbox(deployed.sandbox_id)
        env._gateway_url = deployed.gateway_url
        env._instance_id = deployed.instance_id
        return env

    async def load_environment_artifact(self, environment_artifact: EnvironmentArtifact) -> None:
        if self._sandbox is None or self._gateway_url is None:
            raise RuntimeError("Environment not deployed - call deploy() first")
        if environment_artifact.environment_name != self.environment_name:
            raise ValueError(
                f"EnvironmentArtifact environment_name '{environment_artifact.environment_name}' "
                f"does not match env environment_name '{self.environment_name}'"
            )
        file_artifact = environment_artifact.get_file_artifact()
        base_url = legacy_protocol.environment_base_url(self._gateway_url, self.environment_name, mcp=False)
        if await protocol_v1.supports_v1(base_url):
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
        await self._gateway_provider.install_changelog_triggers(self.environment_name)

    async def _copy_artifact_into_container(self, file_artifact) -> str:
        from agent_env.providers.gateway_provider import AGENT_ENV_WEBSITE_BACKEND_SUFFIX, DOCKER_COMPOSE_PATH
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
        if self._gateway_provider is not None:
            try:
                await self._gateway_provider.close()
            except BaseException as e:
                logger.warning(f"Failed to close gateway provider: {e}")
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

        task_id = f"validate-{self.id}-v{self.version}"
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
                env = await WebsiteEnv.from_deployed_env(deployed_env)
                await env.close()
            except Exception as e:
                logger.warning(f"Failed to clean up env sandbox {deployed_env.sandbox_id}: {e}")
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
        service_version: int = 0,
        metadata: dict[str, str] | None = None,
        on_backend_progress: ProgressCallback | None = None,
        on_frontend_progress: ProgressCallback | None = None,
        github_token: str | None = None,
    ) -> WebsiteEnv:
        """Build backend and frontend Docker images from GitHub in parallel and create a WebsiteEnv."""
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
            service_version=service_version,
            metadata=combined_metadata,
        )
        return env
