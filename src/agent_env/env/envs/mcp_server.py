"""MCP Server environment."""

from __future__ import annotations

import logging
import os
import shlex
import uuid
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Callable, ClassVar, Optional

from agentenv_protocol import FilePart, WELL_KNOWN_PATH, client as protocol_v1
from agent_env.artifact import Artifact, DockerImageArtifact
from agent_env.artifact.artifacts.docker_image import GitHubBuildResult, ProgressCallback
from agent_env.env.env import Env
from agent_env.env import legacy_protocol
from agent_env.env.gateway import GatewayMode
from agent_env.env.store import register_env_instance
from agent_env.providers.sandbox_provider import SANDBOX_MODE_CONTAINER
from agent_env.attribution import Attribution

if TYPE_CHECKING:
    from agent_env.artifact import CliArtifact, EnvironmentArtifact, EnvironmentUniverseArtifact

logger = logging.getLogger(__name__)


class MCPServerEnv(Env):
    type: ClassVar[str] = "mcp_server"
    description = "Access MCP servers through a unified gateway"
    _MCP_MAX_RETRIES: ClassVar[int] = 5

    def __init__(self, id: str, version: Optional[int], docker_image_artifact: DockerImageArtifact, environment_name: Optional[str] = None, service_version: Optional[int] = None, metadata: Optional[dict[str, str]] = None):
        from agent_env.providers import GatewayProvider
        super().__init__(id, version, metadata=metadata)
        if not environment_name:
            raise ValueError("environment_name cannot be empty")
        self.docker_image_artifact = docker_image_artifact
        self.environment_name = environment_name
        # Coerced, not just tolerated: to_dict writes this key unconditionally.
        self.service_version = 1 if service_version is None else service_version
        self._sandbox = None
        self._gateway_provider = GatewayProvider()
        self._gateway_url = None

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["docker_image_artifact"] = {
            "id": self.docker_image_artifact.id,
            "version": self.docker_image_artifact.version,
            "type": self.docker_image_artifact.type,
        }
        base["service_name"] = self.environment_name
        base["environment_name"] = self.environment_name  # dual-write with service_name during the rename
        base["service_version"] = self.service_version  # no environment_version: deprecated, not renamed
        return base

    @classmethod
    def from_dict(cls, data: dict) -> MCPServerEnv:
        artifact_ref = data["docker_image_artifact"]
        docker_image_artifact = Artifact.get(artifact_ref["id"], version=artifact_ref["version"])
        return cls(
            id=data["id"],
            version=data.get("version"),
            docker_image_artifact=docker_image_artifact,
            environment_name=data["environment_name"] if "environment_name" in data else data["service_name"],
            service_version=data.get("service_version", 1),
            metadata=data.get("metadata", {}),
        )

    async def deploy(self, ttl_seconds: int = 10800, disk_size_gb: float = 10, gateway_mode: GatewayMode = GatewayMode.PERFORMANCE, cpu: float | None = None, memory_mb: int | None = None, sandbox_type: str | None = None, priority: Optional[int] = None, env_state_type: str | None = None, env_state_instance_id: str | None = None, *, attribution: Optional[Attribution] = None) -> DeployedEnv:
        attribution = dict(attribution or {})
        from agent_env.env.env import DeployedEnv
        from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
        from agent_env.providers import MCPServerConfig, DB_WEB_PORT, DB_MCP_PORT, build_sandbox_provider, get_env_sandbox_provider
        from agent_env.providers.state import acquire_state_for_deploy
        from agent_env.config import get_config

        config = get_config()
        service_db = Env.get(config.default_service_db_env_id)

        try:
            sandbox_provider = build_sandbox_provider(sandbox_type) if sandbox_type else get_env_sandbox_provider()
            mcp_servers = [MCPServerConfig(image=self.docker_image_artifact.image_name, environment_name=self.environment_name)]
            state_instance = await acquire_state_for_deploy(
                env_state_type=env_state_type, ttl_seconds=ttl_seconds, name_hint=self.id, env_state_instance_id=env_state_instance_id,
            )
            result = await self._gateway_provider.create_gateway(
                sandbox_provider=sandbox_provider,
                mcp_servers=mcp_servers,
                mcp_server_images=[self.docker_image_artifact],
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
                gateway_mode=gateway_mode.value,
                env_state_instance_ids=result.env_state_instance_ids,
            )
            deployed_env = register_env_instance(deployed_env, ttl_seconds)
            self._instance_id = deployed_env.instance_id
            return deployed_env
        except BaseException:
            await self.close()
            raise

    @classmethod
    async def from_deployed_env(cls, deployed: DeployedEnv) -> MCPServerEnv:
        from agent_env.env.env import DeployedEnv
        env = Env.get(deployed.env_id, deployed.env_version)
        if not isinstance(env, MCPServerEnv):
            raise TypeError(f"Expected MCPServerEnv, got {type(env).__name__}")
        from agent_env.providers import build_sandbox_provider, get_env_sandbox_provider
        provider = build_sandbox_provider(deployed.sandbox_type) if deployed.sandbox_type else get_env_sandbox_provider()
        env._sandbox = await provider.get_sandbox(deployed.sandbox_id)
        env._gateway_url = deployed.gateway_url
        env._instance_id = deployed.instance_id
        return env

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

    async def load_environment_artifact(self, environment_artifact: EnvironmentArtifact) -> None:
        if self._sandbox is None or self._gateway_url is None:
            raise RuntimeError("Environment not deployed - call deploy() first")
        if environment_artifact.environment_name != self.environment_name:
            raise ValueError(
                f"EnvironmentArtifact environment_name '{environment_artifact.environment_name}' "
                f"does not match env environment_name '{self.environment_name}'"
            )
        file_artifact = environment_artifact.get_file_artifact()
        base_url = legacy_protocol.environment_base_url(self._gateway_url, self.environment_name, mcp=True)
        if await protocol_v1.supports_v1(base_url):
            from agent_env.env.gateway.constants import data_plane_load_timeout_s
            container_path = await self._copy_artifact_into_container(file_artifact)
            # Sized from the payload actually staged on the VM rather than a flat constant:
            # a 3.5GB service and a 30KB one were previously given the same 600s, which
            # killed healthy large loads. Measured after staging so the number reflects real
            # bytes rather than whatever the artifact document claims.
            timeout = data_plane_load_timeout_s(await self._staged_artifact_size(container_path))
            # reset_data previously took the client default (30s), tighter than the load it
            # precedes -- dropping and recreating a large service's schemas can plausibly
            # exceed that. Give it the same budget as the load it belongs to.
            await protocol_v1.reset_data(base_url, timeout=timeout)
            await protocol_v1.add_data(base_url, [FilePart(file={
                "uri": f"file://{container_path}",
                "mimeType": file_artifact.content_type,
                "name": file_artifact.filename,
            })], timeout=timeout)
        else:
            await self._load_environment_artifact_legacy(file_artifact)
        await self._gateway_provider.install_changelog_triggers(self.environment_name)

    async def load_environment_universe_artifact(self, environment_universe_artifact: EnvironmentUniverseArtifact) -> None:
        environment_artifacts = environment_universe_artifact.get_environment_artifacts()
        matching = [sa for sa in environment_artifacts if sa.environment_name == self.environment_name]
        if not matching:
            raise RuntimeError(
                f"MCPServerEnv '{self.id}' (environment '{self.environment_name}') cannot load "
                f"EnvironmentUniverseArtifact '{environment_universe_artifact.id}': it has no EnvironmentArtifact for "
                f"environment '{self.environment_name}' (universe environments: {[sa.environment_name for sa in environment_artifacts]})."
            )
        skipped = [sa.environment_name for sa in environment_artifacts if sa.environment_name != self.environment_name]
        if skipped:
            logger.warning(f"[{self.id}] loading only service '{self.environment_name}' from universe '{environment_universe_artifact.id}'; skipping non-hostable services {skipped}")
        await self.load_environment_artifact(matching[0])
        if self._instance_id:
            from agent_env.env.store import update_env_instance_environment_universe
            update_env_instance_environment_universe(self._instance_id, environment_universe_artifact.id, environment_universe_artifact.version)

    async def load_artifact(self, artifact):
        from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
        if isinstance(artifact, EnvironmentUniverseArtifact):
            return await self.load_environment_universe_artifact(artifact)
        if isinstance(artifact, EnvironmentArtifact):
            return await self.load_environment_artifact(artifact)
        raise ValueError(f"{type(self).__name__} '{self.id}' cannot load artifact '{getattr(artifact, 'id', '?')}' of type '{getattr(artifact, 'type', '?')}' — expected a EnvironmentArtifact or EnvironmentUniverseArtifact")

    async def _staged_artifact_size(self, container_path: str) -> int | None:
        """Bytes of the staged payload, or None if it can't be measured.

        Best-effort on purpose: the size only picks a timeout, so a failed stat should fall
        back to the floor (today's behaviour) rather than fail a load that is fine.
        """
        if self._sandbox is None:
            return None
        try:
            container_id = await self._gateway_provider._get_container_id(self._sandbox, self.environment_name)
            out = await self._sandbox.exec_script(
                f"docker exec {container_id} stat -c %s {shlex.quote(container_path)}"
            )
            return int(out.strip())
        except Exception as e:  # noqa: BLE001 -- a measurement, not a dependency
            logger.warning(
                "[%s] could not stat staged payload %s (%s); using the floor timeout",
                self.environment_name, container_path, e,
            )
            return None

    async def _copy_artifact_into_container(self, file_artifact) -> str:
        container_path = f"/data/{file_artifact.filename}"
        if self._sandbox.mode == SANDBOX_MODE_CONTAINER:
            await self._sandbox.write_file_from_s3(file_artifact.object_url, container_path)
        else:
            # Stage on the disk-backed app dir, not /tmp (tmpfs/RAM): a large
            # artifact (github ~8GB) OOM-kills the copy under concurrent loads.
            # Random suffix avoids collisions between concurrent loads.
            from agent_env.providers.gateway_provider import GATEWAY_APP_DIR
            stage_dir = f"{GATEWAY_APP_DIR}/_artifact_staging"
            vm_temp_path = f"{stage_dir}/{self.environment_name}-{uuid.uuid4().hex[:8]}-{file_artifact.filename}"
            await self._sandbox.exec_script(f"mkdir -p {stage_dir}")
            await self._sandbox.load_s3_file(file_artifact.object_url, vm_temp_path)
            container_id = await self._gateway_provider._get_container_id(self._sandbox, self.environment_name)
            await self._sandbox.exec_script(f"docker exec {container_id} mkdir -p /data")
            await self._sandbox.exec_script(f"docker cp {vm_temp_path} {container_id}:{container_path}")
            await self._sandbox.exec_script(f"rm -f {vm_temp_path}")
        return container_path

    async def _load_environment_artifact_legacy(self, file_artifact) -> None:
        container_path = await self._copy_artifact_into_container(file_artifact)
        await legacy_protocol.reset(
            self._gateway_url, self.environment_name, container_path, max_retries=self._MCP_MAX_RETRIES
        )

    async def validate(self, on_progress: Optional[Callable[[str], None]] = None) -> str:
        """Run the env validator task and return the task instance ID.

        Creates a new versioned Task with ID ``validate-{env_id}-v{version}`` each time.
        """
        from agent_env.task import Task
        from agent_env.task_step import DeployAgentTaskStep, DeployEnvTaskStep, PromptAgentTaskStep, ValidationGateAggregatorStep, VerifyCoreEnvironmentProtocolStep, VerifyEnvironmentCardStep, VerifyMCPToolSchemaTaskStep, VerifySpecConformanceTaskStep, VerifyMCPEnvAssessmentStep
        from agent_env.task_step.task_steps.mcp_env_validator import TOOL_CORRECTNESS_PROMPT, TOOL_CORRECTNESS_OUTPUT_FORMAT

        task_id = f"validate-{self.id}-v{self.version}"
        prompt_id = f"{task_id}-prompt"

        # Env metadata can narrow the gate, e.g. {"required_gates": ["mcp_tool_schema"]}.
        # Unset = DEFAULT_REQUIRED_GATES.
        required_gates = self.metadata.get("required_gates")
        if required_gates is not None and not isinstance(required_gates, list):
            logger.warning(f"Ignoring non-list required_gates in env metadata: {required_gates!r}")
            required_gates = None

        task = Task.put(
            id=task_id,
            steps=[
                DeployEnvTaskStep(id=f"{task_id}-deploy", version=None, env_id=self.id, env_version=self.version),
                VerifyEnvironmentCardStep(id=f"{task_id}-envcard", version=None, env_id=self.id, depends_on=[{"task_step_id": f"{task_id}-deploy"}]),
                VerifyCoreEnvironmentProtocolStep(id=f"{task_id}-protocol", version=None, env_id=self.id, depends_on=[{"task_step_id": f"{task_id}-deploy"}]),
                VerifyMCPToolSchemaTaskStep(id=f"{task_id}-schema", version=None, env_id=self.id, depends_on=[{"task_step_id": f"{task_id}-deploy"}]),
                # Tolerant: an infra failure here must still let the gate aggregate,
                # so `put` reports "gate did not run" rather than a traceback.
                VerifySpecConformanceTaskStep(id=f"{task_id}-spec", version=None, env_id=self.id, depends_on=[{"task_step_id": f"{task_id}-deploy"}], fail_task_on_error=False),
                DeployAgentTaskStep(id=f"{task_id}-agent", version=None, env_ids=[self.id], depends_on=[{"task_step_id": f"{task_id}-schema"}]),
                PromptAgentTaskStep(id=f"{task_id}-prompt", version=None, prompt=TOOL_CORRECTNESS_PROMPT, prompt_id=prompt_id, output_format=TOOL_CORRECTNESS_OUTPUT_FORMAT, timeout_seconds=300, depends_on=[{"task_step_id": f"{task_id}-agent"}]),
                VerifyMCPEnvAssessmentStep(id=f"{task_id}-assess", version=None, env_id=self.id, prompt_id=prompt_id, depends_on=[{"task_step_id": f"{task_id}-prompt"}]),
                ValidationGateAggregatorStep(id=f"{task_id}-gate", version=None, env_id=self.id, required_gates=required_gates, depends_on=[{"task_step_id": f"{task_id}-schema"}, {"task_step_id": f"{task_id}-spec"}, {"task_step_id": f"{task_id}-assess"}]),
            ],
        )
        logger.info(f"Created validation task: {task.id} version={task.version}")

        log = on_progress or (lambda msg: None)
        on_start = lambda i, total, step, ctx: log(f"Running step [{i+1}/{total}]: {step.type}...")
        on_complete = lambda i, total, step, ctx, dur: log(f"Completed step [{i+1}/{total}]: {step.type} [{dur:.1f}s]")
        context = await task.run(on_step_start=on_start, on_step_complete=on_complete)
        for deployed_env in context.deployed_envs:
            try:
                env = await MCPServerEnv.from_deployed_env(deployed_env)
                await env.close()
            except Exception as e:
                logger.warning(f"Failed to clean up env sandbox {deployed_env.sandbox_id}: {e}")
        for deployed_agent in context.deployed_agents:
            if deployed_agent.sandbox_id:
                try:
                    from agent_env.providers import build_sandbox_provider, get_agent_sandbox_provider
                    provider = build_sandbox_provider(deployed_agent.sandbox_type) if deployed_agent.sandbox_type else get_agent_sandbox_provider()
                    sandbox = await provider.get_sandbox(deployed_agent.sandbox_id)
                    await sandbox.terminate()
                except Exception as e:
                    logger.warning(f"Failed to clean up agent sandbox {deployed_agent.sandbox_id}: {e}")
        return context.instance_id

    async def create_cli(self, command_name: Optional[str] = None, on_progress: Optional[Callable[[str], None]] = None, force: bool = False) -> "CliArtifact":
        from agent_env.store.base import ConcurrentModificationError, NotFoundError
        from agent_env.task import Task
        from agent_env.task_step import BuildMcpCliTaskStep, DeployEnvTaskStep

        if not force:
            cached = self.metadata.get("cli_artifact")
            if cached and cached.get("id") and cached.get("version") is not None:
                try:
                    artifact = Artifact.get(cached["id"], version=cached["version"])
                    logger.info(f"Reusing CliArtifact from env metadata: id={artifact.id} version={artifact.version} (pass force=True to rebuild)")
                    return artifact
                except NotFoundError:
                    logger.warning(f"Cached CliArtifact {cached} not found in store; rebuilding")

        cmd = command_name or self.environment_name
        task_id = f"create-cli-{self.id}-v{self.version}"
        cli_artifact_id = f"cli-{self.id}"

        task = Task.put(id=task_id, steps=[
            DeployEnvTaskStep(id=f"{task_id}-deploy", version=None, env_id=self.id, env_version=self.version),
            BuildMcpCliTaskStep(id=f"{task_id}-build", version=None, env_id=self.id, command_name=cmd, cli_artifact_id=cli_artifact_id),
        ])
        logger.info(f"Created create-cli task: {task.id} version={task.version}")

        log = on_progress or (lambda msg: None)
        on_start = lambda i, total, step, ctx: log(f"Running step [{i+1}/{total}]: {step.type}...")
        on_complete = lambda i, total, step, ctx, dur: log(f"Completed step [{i+1}/{total}]: {step.type} [{dur:.1f}s]")
        context = await task.run(on_step_start=on_start, on_step_complete=on_complete)
        for deployed_env in context.deployed_envs:
            try:
                env = await MCPServerEnv.from_deployed_env(deployed_env)
                await env.close()
            except Exception as e:
                logger.warning(f"Failed to clean up env sandbox {deployed_env.sandbox_id}: {e}")

        ref = context.metadata["cli_artifact"]
        artifact = Artifact.get(ref["id"], version=ref["version"])
        try:
            self.update_metadata({**self.metadata, "cli_artifact": {"id": artifact.id, "version": artifact.version}})
        except ConcurrentModificationError as e:
            logger.warning(f"Concurrent metadata update lost CAS; cli_artifact built but env metadata not updated: {e}")
        return artifact

    @classmethod
    async def put_from_github(
        cls,
        id: str,
        dockerfile_github_url: str,
        docker_context_github_url: str | None = None,
        environment_name: str | None = None,
        service_version: int = 0,
        metadata: dict[str, str] | None = None,
        on_progress: ProgressCallback | None = None,
        github_token: str | None = None,
    ) -> MCPServerEnv:
        """Build a Docker image from a GitHub repo on a temporary VM and create an MCPServerEnv.

        The resulting DockerImageArtifact and MCPServerEnv are identical to the local-build path.
        """
        docker_image_artifact = await DockerImageArtifact.put_from_github(
            id=f"mcp-server-{id}",
            dockerfile_github_url=dockerfile_github_url,
            docker_context_github_url=docker_context_github_url,
            on_progress=on_progress,
            github_token=github_token,
        )

        combined_metadata: dict[str, str] = {}
        user = os.getenv("USER")
        if user:
            combined_metadata["created_by"] = user
        try:
            combined_metadata["agent_env_version"] = pkg_version("agentenv-framework")
        except Exception:
            pass
        combined_metadata["dockerfile_github_url"] = docker_image_artifact.dockerfile_github_url
        combined_metadata["github_owner"] = docker_image_artifact.github_owner
        combined_metadata["github_repo"] = docker_image_artifact.github_repo
        combined_metadata["github_commit"] = docker_image_artifact.github_commit
        if docker_image_artifact.docker_context_github_url:
            combined_metadata["docker_context_github_url"] = docker_image_artifact.docker_context_github_url
        if docker_image_artifact.github_ref:
            combined_metadata["github_ref"] = docker_image_artifact.github_ref
        if metadata:
            combined_metadata.update(metadata)

        return cls.put(
            id=id,
            docker_image_artifact=docker_image_artifact.artifact,
            environment_name=environment_name,
            service_version=service_version,
            metadata=combined_metadata,
        )
