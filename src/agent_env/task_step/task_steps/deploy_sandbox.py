"""Deploy bare sandbox task step."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.attribution import deploy_attribution

logger = logging.getLogger(__name__)

_SUPPORTED_MODES = {"vm", "container"}


class DeploySandboxTaskStep(TaskStep):
    type: ClassVar[str] = "deploy_sandbox"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: Optional[int],
        sandbox_name: str,
        sandbox_mode: str,
        image: Optional[str] = None,
        cpu: float = 2.0,
        memory_mb: int = 8192,
        disk_size_gb: float = 10,
        ttl_seconds: int = TaskStep.DEFAULT_TTL_SECONDS,
        exposed_ports: Optional[list[int]] = None,
        boot_mode: Optional[str] = None,
        port: Optional[int] = None,
        env_vars: Optional[dict[str, str]] = None,
        sandbox_type: Optional[str] = None,
        network_policy: Optional[dict] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        metadata: Optional[dict] = None,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if sandbox_mode not in _SUPPORTED_MODES:
            raise ValueError(f"sandbox_mode must be one of {sorted(_SUPPORTED_MODES)}, got {sandbox_mode!r}")
        self.sandbox_name = sandbox_name
        self.sandbox_mode = sandbox_mode
        self.image = image
        self.cpu = cpu
        self.memory_mb = memory_mb
        self.disk_size_gb = disk_size_gb
        self.ttl_seconds = ttl_seconds
        self.exposed_ports = exposed_ports
        self.boot_mode = boot_mode
        self.port = port
        self.env_vars = env_vars
        self.sandbox_type = sandbox_type
        self.metadata = metadata or {}
        self.network_policy = NetworkPolicy.from_dict(network_policy).to_dict() if network_policy else None

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["sandbox_name"] = self.sandbox_name
        base["sandbox_mode"] = self.sandbox_mode
        base["image"] = self.image
        base["cpu"] = self.cpu
        base["memory_mb"] = self.memory_mb
        base["disk_size_gb"] = self.disk_size_gb
        base["ttl_seconds"] = self.ttl_seconds
        base["exposed_ports"] = self.exposed_ports
        base["boot_mode"] = self.boot_mode
        base["port"] = self.port
        base["env_vars"] = self.env_vars
        base["sandbox_type"] = self.sandbox_type
        base["metadata"] = self.metadata
        base["network_policy"] = self.network_policy
        return base

    @classmethod
    def from_dict(cls, data: dict) -> DeploySandboxTaskStep:
        return cls(
            **cls._base_from_dict(data),
            sandbox_name=data["sandbox_name"],
            sandbox_mode=data["sandbox_mode"],
            image=data.get("image"),
            cpu=data.get("cpu", 2.0),
            memory_mb=data.get("memory_mb", 8192),
            disk_size_gb=data.get("disk_size_gb", 10),
            ttl_seconds=data.get("ttl_seconds", TaskStep.DEFAULT_TTL_SECONDS),
            exposed_ports=data.get("exposed_ports"),
            boot_mode=data.get("boot_mode"),
            port=data.get("port"),
            env_vars=data.get("env_vars"),
            sandbox_type=data.get("sandbox_type"),
            metadata=dict(data.get("metadata") or {}),
            network_policy=data.get("network_policy"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_sandbox_provider

        if any(s.sandbox_name == self.sandbox_name for s in context.deployed_sandboxes):
            raise RuntimeError(f"Sandbox '{self.sandbox_name}' is already deployed")

        user_overrides = context.metadata.get("user_overrides") or {}
        # Moves only deploy_sandbox steps: deploy_env and deploy_agent read env_sandbox and agent_sandbox.
        resolved_sandbox_type = user_overrides.get("sandbox") or self.sandbox_type
        provider = (
            build_sandbox_provider(resolved_sandbox_type)
            if resolved_sandbox_type
            else get_sandbox_provider()
        )

        _ttl_override = user_overrides.get("ttl_seconds")
        resolved_ttl = _ttl_override if _ttl_override is not None else self.ttl_seconds
        attribution = deploy_attribution(self, context)
        logger.info(
            f"Deploying sandbox '{self.sandbox_name}' (mode={self.sandbox_mode}, "
            f"image={self.image}, cpu={self.cpu}, memory_mb={self.memory_mb}, "
            f"disk_size_gb={self.disk_size_gb}, ttl_seconds={resolved_ttl}, "
            f"sandbox_type={resolved_sandbox_type}, attribution={attribution})"
        )

        _override = user_overrides.get("network_policy")
        _resolved = _override if _override is not None else self.network_policy
        policy = NetworkPolicy.from_dict(_resolved) if _resolved is not None else None

        if self.sandbox_mode == "vm":
            sandbox = await provider.create_vm(
                image=self.image,
                boot_mode=self.boot_mode,
                cpu=self.cpu,
                memory=self.memory_mb,
                disk_size_gb=self.disk_size_gb,
                timeout=resolved_ttl,
                exposed_ports=self.exposed_ports or [],
                attribution=attribution,
                network_policy=policy,
            )
        else:
            if self.image is None or self.port is None:
                raise ValueError("container-mode sandbox requires `image` and `port`")
            sandbox = await provider.create_sandbox(
                image_name=self.image,
                port=self.port,
                env=self.env_vars or {},
                cpu=self.cpu,
                memory=self.memory_mb,
                disk_size_gb=self.disk_size_gb,
                timeout=resolved_ttl,
                attribution=attribution,
                network_policy=policy,
            )

        context.deployed_sandboxes.append(DeployedSandbox(
            sandbox_name=self.sandbox_name,
            sandbox_id=sandbox.sandbox_id,
            sandbox_mode=sandbox.mode,
            sandbox_type=sandbox.type,
            tunnel_urls={str(p): u for p, u in sandbox.tunnel_urls.items()} if sandbox.tunnel_urls else None,
            vnc_url=sandbox.vnc_url,
            network_policy=sandbox.network_policy.to_dict() if sandbox.network_policy else None,
        ))
        return context
