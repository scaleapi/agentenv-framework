"""``osworld_deploy_controller``: the OSWorld controller beside a computer env, in a container sandbox of its own.

The controller holds the grader, a task's checks and the judge's key, and a computer's apps run as its session user,
the agent's, so it runs apart from the computer, in remote mode against the guest server at the endpoint the
computer's card advertises (``agent_env.osworld.guest``), with the env's control token on every call. Once it answers,
the step records the controller's and the guest server's addresses on the env for ``osworld_initialize`` and
``osworld_evaluate``, and the sandbox in ``context.deployed_sandboxes`` for teardown."""
from __future__ import annotations

import json
import logging
from typing import ClassVar, Optional

from agent_env.artifact import Artifact, DockerImageArtifact
from agent_env.artifact.artifacts.docker_image import pull_problem
from agent_env.attribution import deploy_attribution
from agent_env.config import ConfigError, get_config
from agent_env.entity.refs import EntityRef
from agent_env.osworld import controller, guest
from agent_env.providers.sandbox_providers.chained_sandbox_provider import skip_refusing
from agent_env.providers.sandbox_providers.sandbox_provider import get_sandbox_provider
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

#: The controller image's artifact id when a step names none.
DEFAULT_CONTROLLER_IMAGE_ARTIFACT_ID = "cua-controller"
CONTROLLER_PORT = 8000
#: A controller's sandbox name, by its computer env's id.
SANDBOX_NAME_PREFIX = "osworld-controller-"


class OSWorldDeployControllerTaskStep(TaskStep):
    type: ClassVar[str] = "osworld_deploy_controller"
    entity_refs = (EntityRef.env("env_id"),
                   EntityRef.artifact("controller_image_artifact_id", artifact_type="docker_image"))

    def __init__(self, id: str, version: Optional[int], env_id: str, controller_image_artifact_id: Optional[str] = None,
                 ttl_seconds: int = TaskStep.DEFAULT_TTL_SECONDS, cpu: float = 2.0, memory_mb: int = 4096,
                 env_vars: Optional[dict[str, str]] = None, depends_on: Optional[list[TaskStepDependency]] = None,
                 fail_task_on_error: bool = True):
        """``env_vars`` go to the controller beside the ones the step sets, which win: a deployment's own, the
        judge's model (``JUDGE_MODEL``) say."""
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.controller_image_artifact_id = controller_image_artifact_id
        self.ttl_seconds = ttl_seconds
        self.cpu = cpu
        self.memory_mb = memory_mb
        self.env_vars = env_vars

    def to_dict(self) -> dict:
        return {**super().to_dict(), "env_id": self.env_id,
                "controller_image_artifact_id": self.controller_image_artifact_id, "ttl_seconds": self.ttl_seconds,
                "cpu": self.cpu, "memory_mb": self.memory_mb, "env_vars": self.env_vars}

    @classmethod
    def from_dict(cls, data: dict) -> OSWorldDeployControllerTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"],
                   controller_image_artifact_id=data.get("controller_image_artifact_id"),
                   ttl_seconds=data.get("ttl_seconds", TaskStep.DEFAULT_TTL_SECONDS), cpu=data.get("cpu", 2.0),
                   memory_mb=data.get("memory_mb", 4096), env_vars=data.get("env_vars"))

    def controller_env(self, deployed, guest_url: str) -> dict[str, str]:
        """Remote mode against the guest server, with the env's control token on every call and the model endpoint's
        key for the judge."""
        env = {**(self.env_vars or {}), "PROVIDER": "remote", "OS_TYPE": "Ubuntu", "REMOTE_VM_IP": guest_url,
               "REMOTE_SERVER_HEADERS": json.dumps(deployed.control_headers)}
        config = get_config()
        try:
            env["LITELLM_API_KEY"] = config.get_litellm_api_key()
        except (KeyError, ConfigError):
            logger.warning("no model endpoint key configured; the OSWorld controller's judge will have none")
        try:
            env["LITELLM_BASE_URL"] = config.get_litellm_base_url()
        except ConfigError:
            pass
        return env

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        sandbox_name = f"{SANDBOX_NAME_PREFIX}{self.env_id}"
        if any(s.sandbox_name == sandbox_name for s in context.deployed_sandboxes):
            raise RuntimeError(f"Sandbox '{sandbox_name}' is already deployed")
        guest_url = await guest.resolve_guest_url(deployed)
        image = Artifact.get(self.controller_image_artifact_id or DEFAULT_CONTROLLER_IMAGE_ARTIFACT_ID)
        if not isinstance(image, DockerImageArtifact):
            raise TypeError(f"the controller image {image.id!r} is a {image.type} artifact, not a docker image")
        env = self.controller_env(deployed, guest_url)
        logger.info(f"Deploying the OSWorld controller for env '{self.env_id}' (image={image.id} v{image.version}, "
                    f"env={sorted(env)})")
        provider = skip_refusing(get_sandbox_provider(),
                                 lambda link: pull_problem(image.pinned_name, on_this_machine=link.ON_THIS_MACHINE),
                                 what=f"sandbox '{sandbox_name}'")
        sandbox = await provider.create_container(image_name=image.pinned_name, port=CONTROLLER_PORT, env=env,
                                                  cpu=self.cpu, memory=self.memory_mb, disk_size_gb=10,
                                                  timeout=self.ttl_seconds, attribution=deploy_attribution(self, context))
        context.deployed_sandboxes.append(DeployedSandbox(
            sandbox_name=sandbox_name, sandbox_id=sandbox.sandbox_id, sandbox_mode=sandbox.mode,
            sandbox_type=sandbox.type,
            tunnel_urls={str(p): u for p, u in sandbox.tunnel_urls.items()} if sandbox.tunnel_urls else None))
        controller_url = sandbox.tunnel_urls[CONTROLLER_PORT]
        await controller.wait_ready(controller_url)
        deployed.metadata = {**(deployed.metadata or {}), guest.CONTROLLER_URL_KEY: controller_url,
                             guest.CONTROLLER_SANDBOX_ID_KEY: sandbox.sandbox_id, guest.GUEST_URL_KEY: guest_url}
        logger.info(f"OSWorld controller for env '{self.env_id}' at {controller_url} (sandbox {sandbox.sandbox_id})")
        return context
