"""MCP Server environment."""

from __future__ import annotations

import logging
import os
import shlex
import uuid
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Callable, ClassVar, Optional

from agentenv_protocol import FilePart, client as protocol_v1
from agent_env.artifact import Artifact, DockerImageArtifact
from agent_env.artifact.artifacts.docker_image import GitHubBuildResult, ProgressCallback, refuse_local_github_build
from agent_env.env.env import Env, gateway_url_of
from agent_env.store.ids import derive_id
from agent_env.env import legacy_protocol
from agent_env.env.envs._deployment import (
    as_builtin, builtin_provider_for, close_deployed, close_replaced, deploy_refusal, deploy_through_provider, host_staging_refusal,
    load_by_signed_url, plugin_provider_like_a_builtin, provider_or_class,
)
from agent_env.env.gateway import GatewayMode
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER
from agent_env.attribution import Attribution

if TYPE_CHECKING:
    from agent_env.artifact import CliArtifact, EnvironmentArtifact, EnvironmentUniverseArtifact
    from agent_env.env.env import DeployedEnv
    from agent_env.providers.env_providers.env_provider import EnvironmentProvider

logger = logging.getLogger(__name__)


class MCPServerEnv(Env):
    type: ClassVar[str] = "mcp_server"
    description = "An MCP server, deployed through the environment provider its env_provider_type names"
    _MCP_MAX_RETRIES: ClassVar[int] = 5

    def __init__(self, id: str, version: Optional[int], docker_image_artifact: DockerImageArtifact, environment_name: Optional[str] = None, *, metadata: Optional[dict[str, str]] = None, env_provider_type: str = "gateway"):
        super().__init__(id, version, metadata=metadata)
        if not environment_name:
            raise ValueError("environment_name cannot be empty")
        if not env_provider_type:
            raise ValueError("env_provider_type cannot be empty")
        self.docker_image_artifact = docker_image_artifact
        self.environment_name = environment_name
        self.env_provider_type = env_provider_type
        self._sandbox = None
        # A built-in's provider is built with the env, as callers that wire an env up by hand rely on; a plugin's is built
        # when the env deploys, so reading a stored env needs no plugin installed. A MultiEnv hands its children its own.
        self._env_provider: Optional[EnvironmentProvider] = builtin_provider_for(env_provider_type)
        self._gateway_url = None
        self._deployed: Optional[DeployedEnv] = None
        self._replaced: list[tuple] = []  # the provider and sandbox of each deployment a later deploy replaced, for close()

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["docker_image_artifact"] = {
            "id": self.docker_image_artifact.id,
            "version": self.docker_image_artifact.version,
            "type": self.docker_image_artifact.type,
        }
        base["service_name"] = self.environment_name
        base["environment_name"] = self.environment_name  # dual-write with service_name during the rename
        base["env_provider_type"] = self.env_provider_type
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
            metadata=data.get("metadata", {}),
            env_provider_type=data.get("env_provider_type", "gateway"),
        )

    async def deploy(self, ttl_seconds: int = 10800, disk_size_gb: float = 10, gateway_mode: GatewayMode = GatewayMode.PERFORMANCE, cpu: float | None = None, memory_mb: int | None = None, sandbox_type: str | None = None, env_state_type: str | None = None, env_state_instance_id: str | None = None, *, attribution: Optional[Attribution] = None) -> DeployedEnv:
        # In container mode the server runs in its own container, so loads stage there, as for a MultiEnv child.
        return await deploy_through_provider(
            self, environment_name=self.environment_name, ttl_seconds=ttl_seconds, sandbox_type=sandbox_type,
            disk_size_gb=disk_size_gb, gateway_mode=gateway_mode, cpu=cpu, memory_mb=memory_mb,
            env_state_type=env_state_type, env_state_instance_id=env_state_instance_id, attribution=attribution,
        )

    def deploy_refusal(self, **options) -> str | None:
        """Why deploy() would refuse these options before building anything, or None; raises, as deploy() does, for a type this process can't find."""
        return deploy_refusal(self, provider_or_class(self), options)

    @classmethod
    async def from_deployed_env(cls, deployed: DeployedEnv) -> MCPServerEnv:
        env = Env.get(deployed.env_id, deployed.env_version)
        if not isinstance(env, MCPServerEnv):
            raise TypeError(f"Expected MCPServerEnv, got {type(env).__name__}")
        if env._env_provider is None:
            env._env_provider = plugin_provider_like_a_builtin(env.env_provider_type)
        if builtin := as_builtin(env._env_provider):
            env._sandbox = await builtin._reattach(env, deployed)
        # Any other plugin's deployment holds its own sandboxes: its env loads through its card and reattaches none.
        env._gateway_url = gateway_url_of(deployed)
        env._instance_id = deployed.instance_id
        env._deployed = deployed
        return env

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

    async def load_environment_artifact(self, environment_artifact: EnvironmentArtifact) -> None:
        builtin = as_builtin(self._env_provider)
        # A built-in's deploy leaves a sandbox to stage into; a plugin's leaves only its record.
        if (self._sandbox if builtin else self._deployed) is None:
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
        base_url = await legacy_protocol.v1_base_url(self._deployed, self._gateway_url, self.environment_name, mcp=True)
        if base_url is not None:
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
        await builtin.install_changelog_triggers(self.environment_name)

    async def load_file_artifact_universe(self, file_artifact_universe: "Any", destination_path: Optional[str] = None,
                                          ) -> "LoadFileArtifactUniverseResult":
        if refusal := host_staging_refusal(self, "Staging files onto the env's host"):
            raise RuntimeError(refusal)
        return await super().load_file_artifact_universe(file_artifact_universe, destination_path)

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
            container_id = await self._env_provider._get_container_id(self._sandbox, self.environment_name)
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
            from agent_env.providers.env_providers.constants import GATEWAY_APP_DIR
            stage_dir = f"{GATEWAY_APP_DIR}/_artifact_staging"
            vm_temp_path = f"{stage_dir}/{self.environment_name}-{uuid.uuid4().hex[:8]}-{file_artifact.filename}"
            await self._sandbox.exec_script(f"mkdir -p {stage_dir}")
            await self._sandbox.load_s3_file(file_artifact.object_url, vm_temp_path)
            container_id = await self._env_provider._get_container_id(self._sandbox, self.environment_name)
            await self._sandbox.exec_script(f"docker exec {container_id} mkdir -p /data")
            await self._sandbox.docker_cp(vm_temp_path, f"{container_id}:{container_path}")
            await self._sandbox.exec_script(f"rm -f {vm_temp_path}")
        return container_path

    async def _load_environment_artifact_legacy(self, file_artifact) -> None:
        container_path = await self._copy_artifact_into_container(file_artifact)
        await legacy_protocol.reset(
            self._gateway_url, self.environment_name, container_path, max_retries=self._MCP_MAX_RETRIES
        )

    async def validate(self, on_progress: Optional[Callable[[str], None]] = None) -> str:
        """Run the env validator task and return the task instance ID.

        Creates a new versioned Task with ID ``{env_id}__validate-v{version}`` each time.
        """
        from agent_env.task import Task
        from agent_env.task_step import DeployAgentTaskStep, DeployEnvTaskStep, PromptAgentTaskStep, ValidationGateAggregatorStep, VerifyCoreEnvironmentProtocolStep, VerifyEnvironmentCardStep, VerifyMCPToolSchemaTaskStep, VerifySpecConformanceTaskStep, VerifyMCPEnvAssessmentStep
        from agent_env.task_step.task_steps.mcp_env_validator import TOOL_CORRECTNESS_PROMPT, TOOL_CORRECTNESS_OUTPUT_FORMAT

        task_id = derive_id(self.id, f"validate-v{self.version}")
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
                await close_deployed(deployed_env, MCPServerEnv)
            except Exception as e:
                logger.warning(f"Failed to clean up env {deployed_env.env_id}: {e}")
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
        task_id = derive_id(self.id, f"create-cli-v{self.version}")

        task = Task.put(id=task_id, steps=[
            DeployEnvTaskStep(id=f"{task_id}-deploy", version=None, env_id=self.id, env_version=self.version),
            BuildMcpCliTaskStep(id=f"{task_id}-build", version=None, env_id=self.id, command_name=cmd),
        ])
        logger.info(f"Created create-cli task: {task.id} version={task.version}")

        log = on_progress or (lambda msg: None)
        on_start = lambda i, total, step, ctx: log(f"Running step [{i+1}/{total}]: {step.type}...")
        on_complete = lambda i, total, step, ctx, dur: log(f"Completed step [{i+1}/{total}]: {step.type} [{dur:.1f}s]")
        context = await task.run(on_step_start=on_start, on_step_complete=on_complete)
        for deployed_env in context.deployed_envs:
            try:
                await close_deployed(deployed_env, MCPServerEnv)
            except Exception as e:
                logger.warning(f"Failed to clean up env {deployed_env.env_id}: {e}")

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
        *,
        metadata: dict[str, str] | None = None,
        on_progress: ProgressCallback | None = None,
        github_token: str | None = None,
        env_provider_type: str = "gateway",
    ) -> MCPServerEnv:
        """Build a Docker image from a GitHub repo on a temporary VM and create an MCPServerEnv.

        The resulting DockerImageArtifact and MCPServerEnv are identical to the local-build path.
        """
        refuse_local_github_build(id)
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
            metadata=combined_metadata,
            env_provider_type=env_provider_type,
        )
