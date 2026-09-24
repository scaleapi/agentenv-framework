"""Deploy environment task step."""

from __future__ import annotations

import inspect
import logging
from typing import ClassVar, Optional

from agent_env.env.gateway import GatewayMode
from agent_env.env.store import update_env_instance_metadata
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.attribution import attribution_of, metadata_from_legacy_document

logger = logging.getLogger(__name__)


class DeployEnvTaskStep(TaskStep):
    type: ClassVar[str] = "deploy_env"
    entity_refs = (EntityRef.env("env_id", version_field="env_version"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        env_version: Optional[int] = None,
        ttl_seconds: int = TaskStep.DEFAULT_TTL_SECONDS,
        disk_size_gb: float = 10,
        gateway_mode: str = GatewayMode.PERFORMANCE.value,
        cpu: Optional[float] = None,
        memory_mb: Optional[int] = None,
        sandbox_type: Optional[str] = None,
        priority: Optional[int] = None,
        env_state_type: Optional[str] = None,
        env_state_instance_id: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        metadata: Optional[dict] = None,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.env_version = env_version
        self.ttl_seconds = ttl_seconds
        self.disk_size_gb = disk_size_gb
        self.gateway_mode = gateway_mode
        self.cpu = cpu
        self.memory_mb = memory_mb
        self.sandbox_type = sandbox_type
        self.env_state_type = env_state_type
        # Existing env state instance to attach to; the provider interprets it.
        self.env_state_instance_id = env_state_instance_id
        self.priority = priority
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["env_version"] = self.env_version
        base["ttl_seconds"] = self.ttl_seconds
        base["disk_size_gb"] = self.disk_size_gb
        base["gateway_mode"] = self.gateway_mode
        base["cpu"] = self.cpu
        base["memory_mb"] = self.memory_mb
        base["sandbox_type"] = self.sandbox_type
        base["env_state_type"] = self.env_state_type
        base["env_state_instance_id"] = self.env_state_instance_id
        base["priority"] = self.priority
        base["metadata"] = self.metadata
        return base

    @classmethod
    def from_dict(cls, data: dict) -> DeployEnvTaskStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            env_version=data.get("env_version"),
            ttl_seconds=data.get("ttl_seconds", TaskStep.DEFAULT_TTL_SECONDS),
            disk_size_gb=data.get("disk_size_gb", 10),
            gateway_mode=data.get("gateway_mode", GatewayMode.PERFORMANCE.value),
            cpu=data.get("cpu"),
            memory_mb=data.get("memory_mb"),
            sandbox_type=data.get("sandbox_type"),
            env_state_type=data.get("env_state_type"),
            env_state_instance_id=data.get("env_state_instance_id"),
            priority=data.get("priority"),
            metadata=metadata_from_legacy_document(data),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.env.env import Env

        existing = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if existing is not None:
            raise RuntimeError(f"Env '{self.env_id}' is already deployed")

        env = Env.get(self.env_id, self.env_version)
        user_overrides = context.metadata.get("user_overrides", {})
        resolved_sandbox_type = user_overrides.get("env_sandbox") or self.sandbox_type
        resolved_env_state_type = user_overrides.get("env_state_type") or self.env_state_type
        resolved_env_state_instance_id = (
            user_overrides.get("env_state_instance_id") or self.env_state_instance_id
        )
        _priority_override = user_overrides.get("priority")
        resolved_priority = _priority_override if _priority_override is not None else self.priority
        _ttl_override = user_overrides.get("ttl_seconds")
        resolved_ttl = _ttl_override if _ttl_override is not None else self.ttl_seconds
        attribution = attribution_of(self)
        logger.info(
            f"Deploying env '{self.env_id}' (ttl={resolved_ttl}s, "
            f"disk_size_gb={self.disk_size_gb}, gateway_mode={self.gateway_mode}, "
            f"cpu={self.cpu}, memory_mb={self.memory_mb}, "
            f"sandbox_type={resolved_sandbox_type}, env_state_type={resolved_env_state_type}, "
            f"env_state_instance_id={resolved_env_state_instance_id}, "
            f"project_id={attribution.get('project_id')}, priority={resolved_priority})"
        )
        deploy_kwargs = {
            "ttl_seconds": resolved_ttl,
            "disk_size_gb": self.disk_size_gb,
            "gateway_mode": GatewayMode(self.gateway_mode),
            "cpu": self.cpu,
            "memory_mb": self.memory_mb,
            "priority": resolved_priority,
            "attribution": attribution,
        }
        if resolved_sandbox_type is not None:
            deploy_kwargs["sandbox_type"] = resolved_sandbox_type
        if resolved_env_state_type is not None:
            deploy_kwargs["env_state_type"] = resolved_env_state_type
        if resolved_env_state_instance_id is not None:
            deploy_kwargs["env_state_instance_id"] = resolved_env_state_instance_id
        # Forward the per-run LiteLLM key so envs that spin up their own LLM-calling
        # sidecars attribute spend to the same key as the agent, and the desktop sandbox
        # override to envs that take one. Guard on the signature — most envs' deploy()
        # takes a fixed param list and would TypeError on an unexpected kwarg.
        params = inspect.signature(env.deploy).parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        model_api_key = user_overrides.get("litellm_api_key")
        if model_api_key and ("litellm_api_key" in params or accepts_kwargs):
            deploy_kwargs["litellm_api_key"] = model_api_key
        cua_sandbox_type = user_overrides.get("cua_sandbox")
        if cua_sandbox_type is not None:
            if "cua_sandbox_type" in params or accepts_kwargs:
                deploy_kwargs["cua_sandbox_type"] = cua_sandbox_type
            else:
                logger.warning(
                    f"Ignoring cua_sandbox={cua_sandbox_type!r}: env '{self.env_id}' "
                    f"({type(env).__name__}) does not accept cua_sandbox_type"
                )
        deployed_env = await env.deploy(**deploy_kwargs)
        # Annotations for the instance record. `deploy_step_id` was previously set
        # on the in-memory object only, which never reached Mongo: create_instance
        # runs INSIDE env.deploy() above, so anything assigned here lands after the
        # insert and is dropped. Records showed metadata=null as a result.
        #
        # `deployed_by_oauth_subject` is the identity of whoever asked for this env,
        # carried from the hub in context.metadata because the record is written
        # here — in the worker, minutes after the request whose bearer identified
        # them is gone. It is what env-scoped destructive operations (a universe
        # load) authorize against, so it has to be on the record itself rather than
        # reconstructed later from run documents.
        annotations = {"deploy_step_id": self.id}
        hub_metadata = context.metadata.get("agent_env_hub") or {}
        deploying_subject = hub_metadata.get("oauth_subject")
        if isinstance(deploying_subject, str) and deploying_subject.strip():
            annotations["deployed_by_oauth_subject"] = deploying_subject.strip()
        deployed_env.metadata = {**(deployed_env.metadata or {}), **annotations}
        if deployed_env.instance_id:
            # Best-effort: annotating the record must not fail a good deploy.
            update_env_instance_metadata(deployed_env.instance_id, annotations)
        context.deployed_envs.append(deployed_env)

        # Some envs (e.g. harbor coding tasks) build + run the agent as part of
        # deploy() and have no separate DeployAgent step. They advertise the
        # running agent via DeployedEnv.metadata["deployed_agent"]; register it
        # so PromptAgent can resolve it by name from context.deployed_agents.
        agent_info = (deployed_env.metadata or {}).get("deployed_agent")
        if agent_info:
            from agent_env.task_step.context import DeployedAgent

            if any(a.agent_name == agent_info["agent_name"] for a in context.deployed_agents):
                raise RuntimeError(
                    f"Agent '{agent_info['agent_name']}' already in context.deployed_agents"
                )
            context.deployed_agents.append(DeployedAgent(
                agent_name=agent_info["agent_name"],
                api_url=agent_info["api_url"],
                sandbox_id=agent_info.get("sandbox_id"),
                sandbox_type=agent_info.get("sandbox_type"),
                a2a_url=agent_info.get("a2a_url"),
                a2a_card=agent_info.get("a2a_card"),
                instance_id=agent_info.get("instance_id") or deployed_env.instance_id,
            ))
        return context
