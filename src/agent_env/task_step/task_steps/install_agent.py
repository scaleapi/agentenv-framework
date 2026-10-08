"""Install an A2A agent onto a running task container by executing the agent's install/v1 extension commands.

install_commands use `{name}` placeholders (Python str.format) for everything that
varies per install — container name, work dir, agent build context, A2A port, and
config-sourced creds. The runner resolves each declared `required_params` entry
via the registry below, `shlex.quote`s the value, and `cmd.format(**resolved)`s
each command before exec. No host env vars are exported.

To add a new param: register it in `_build_param_resolvers` (and document what
it returns); the agent's extension declares it in `required_params` to opt in.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from datetime import datetime, timezone
from typing import Any, Callable, ClassVar, Optional

import httpx

from agent_env.providers.sandbox_providers.local_sandbox import host_url_for
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

INSTALL_COMMANDS_KEY = "install_commands"
INSTALL_COMMANDS_HOST_KEY = "install_commands_host"
REQUIRED_PARAMS_KEY = "required_params"
REQUIRED_PARAMS_HOST_KEY = "required_params_host"
A2A_PORT_KEY = "a2a_port"

def _build_param_resolvers(
    *,
    container_name: Optional[str],
    work_dir: str,
    agent_ctx_tar: str,
    a2a_port: int,
    config,
    agent_name: str,
    workspace_dir: Optional[str] = None,
    litellm_api_key_override: Optional[str] = None,
    sandbox_type: Optional[str] = None,
) -> dict[str, Callable[[], str]]:
    """Per-install registry: required_params name -> zero-arg resolver returning the raw value.

    Anything not in this registry is rejected at install time with a clear error,
    so an extension can't ask for a param the runner doesn't know how to source.
    """

    def _resolve_workspace_dir() -> str:
        # The agent's container working directory (e.g. a Harbor task's
        # `workdir`). Opt-in: an agent declares `workspace_dir` in
        # required_params and uses `{workspace_dir}` in its install_commands
        # (e.g. `docker exec -w {workspace_dir} ...`). Inert for agents that
        # don't, so existing agents are unaffected.
        if workspace_dir is None:
            raise KeyError(
                "agent declares required_param 'workspace_dir' but the InstallAgentTaskStep "
                "was constructed without one; pass workspace_dir=<container working directory>"
            )
        return workspace_dir

    def _resolve_container() -> str:
        if container_name is None:
            raise KeyError(
                "agent declares required_param 'container' but this is a host-mode install "
                "(InstallAgentTaskStep has no container_name)"
            )
        return container_name

    return {
        # Runner-supplied (from this step's own state)
        "container":      _resolve_container,
        "work_dir":       lambda: work_dir,
        "agent_ctx_tar":  lambda: agent_ctx_tar,
        "a2a_port":       lambda: str(a2a_port),
        "agent_name":     lambda: agent_name,
        "workspace_dir":  _resolve_workspace_dir,
        # Config-sourced (from agent-env config)
        "litellm_api_key":   lambda: litellm_api_key_override or config.get_litellm_api_key(),
        "litellm_base_url":  lambda: host_url_for(config.get_litellm_base_url(), sandbox_type),
    }


class InstallAgentTaskStep(TaskStep):
    type: ClassVar[str] = "install_agent"
    entity_refs = (EntityRef.agent("a2a_agent_id", version_field="a2a_agent_version"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        sandbox_name: str,
        container_name: Optional[str] = None,
        a2a_agent_id: str = "",
        a2a_agent_version: Optional[int] = None,
        agent_name: str = TaskStep.DEFAULT_AGENT_NAME,
        a2a_port: Optional[int] = None,
        workspace_dir: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not a2a_agent_id:
            raise ValueError("a2a_agent_id is required")
        self.sandbox_name = sandbox_name
        self.container_name = container_name
        self.a2a_agent_id = a2a_agent_id
        self.a2a_agent_version = a2a_agent_version
        self.agent_name = agent_name
        self.a2a_port = a2a_port
        self.workspace_dir = workspace_dir

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["sandbox_name"] = self.sandbox_name
        base["container_name"] = self.container_name
        base["a2a_agent_id"] = self.a2a_agent_id
        base["a2a_agent_version"] = self.a2a_agent_version
        base["agent_name"] = self.agent_name
        base["a2a_port"] = self.a2a_port
        base["workspace_dir"] = self.workspace_dir
        return base

    @classmethod
    def from_dict(cls, data: dict) -> InstallAgentTaskStep:
        return cls(
            **cls._base_from_dict(data),
            sandbox_name=data["sandbox_name"],
            container_name=data.get("container_name"),
            a2a_agent_id=data["a2a_agent_id"],
            a2a_agent_version=data.get("a2a_agent_version"),
            agent_name=data.get("agent_name", TaskStep.DEFAULT_AGENT_NAME),
            a2a_port=data.get("a2a_port"),
            workspace_dir=data.get("workspace_dir"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent
        from agent_env.providers.sandbox_providers.sandbox_provider import (
            SANDBOX_MODE_VM,
            build_sandbox_provider,
            get_sandbox_provider,
        )
        from agent_env.config import ConfigError, get_config

        existing = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
        if existing is not None:
            raise RuntimeError(f"Agent with name '{self.agent_name}' is already deployed")

        if self.container_name is not None:
            containers = context.metadata.get("deployed_docker_containers", [])
            container = next(
                (
                    c for c in containers
                    if c.get("container_name") == self.container_name
                    and c.get("sandbox_name") == self.sandbox_name
                ),
                None,
            )
            if container is None:
                raise RuntimeError(
                    f"Container '{self.container_name}' not found on sandbox '{self.sandbox_name}' in "
                    f"context.metadata['deployed_docker_containers']. Run RunDockerContainerTaskStep first."
                )

        ds = next((s for s in context.deployed_sandboxes if s.sandbox_name == self.sandbox_name), None)
        if ds is None:
            raise RuntimeError(f"Sandbox '{self.sandbox_name}' not found in context.deployed_sandboxes")
        if ds.sandbox_mode != SANDBOX_MODE_VM:
            raise RuntimeError(
                f"InstallAgent requires a VM-mode sandbox; '{self.sandbox_name}' is mode={ds.sandbox_mode!r}"
            )

        provider = (
            build_sandbox_provider(ds.sandbox_type) if ds.sandbox_type else get_sandbox_provider()
        )
        sandbox = await provider.get_sandbox(ds.sandbox_id)

        # Honor `--a2a-agent-id` override (same pattern as DeployAgentTaskStep) — lets the
        # caller swap which agent gets installed without rewriting the task definition.
        user_overrides = context.metadata.get("user_overrides", {})
        a2a_agent_id = user_overrides.get("a2a_agent_id") or self.a2a_agent_id
        agent = A2AAgent.get(a2a_agent_id, self.a2a_agent_version)
        if not agent.docker_image_artifact.build_context_object_url:
            raise RuntimeError(
                f"Agent {agent.id} v{agent.version} has no build_context_object_url on its docker_image_artifact; "
                f"install/v1 needs the build context (agent source files) to install into the task container"
            )

        logger.info(f"Loading agent image {agent.docker_image_artifact.image_name} onto VM (needed to read agent card)...")
        await sandbox.load_docker_images([agent.docker_image_artifact])

        install_ext = await self._extract_install_extension(sandbox, agent.docker_image_artifact.image_name)
        params = install_ext.get("params") or {}
        host_mode = self.container_name is None
        commands_key = INSTALL_COMMANDS_HOST_KEY if host_mode else INSTALL_COMMANDS_KEY
        install_commands = params.get(commands_key)
        a2a_port = self.a2a_port or params.get(A2A_PORT_KEY)
        required_params = params.get(REQUIRED_PARAMS_HOST_KEY if host_mode else REQUIRED_PARAMS_KEY) or []
        if host_mode and not install_commands:
            raise RuntimeError(
                f"Agent {agent.id} v{agent.version} install/v1 params have no '{INSTALL_COMMANDS_HOST_KEY}'; "
                f"this agent does not support host-mode install (pass container_name for container mode)"
            )
        if not isinstance(install_commands, list) or not install_commands:
            raise RuntimeError(f"Agent {agent.id} v{agent.version} install/v1 params missing '{commands_key}' (list)")
        if not isinstance(a2a_port, int):
            raise RuntimeError(f"Agent {agent.id} v{agent.version} install/v1 params missing '{A2A_PORT_KEY}' (int) and the step has no a2a_port override")
        if not isinstance(required_params, list):
            raise RuntimeError(f"Agent {agent.id} v{agent.version} install/v1 params 'required_params' must be a list")

        if not ds.tunnel_urls or str(a2a_port) not in ds.tunnel_urls:
            raise RuntimeError(
                f"Sandbox '{self.sandbox_name}' does not tunnel port {a2a_port} required by agent {agent.id}; "
                f"include exposed_ports=[{a2a_port}] on DeploySandboxTaskStep"
                + ("" if host_mode else f" and ports=[{a2a_port}] on RunDockerContainerTaskStep")
            )
        a2a_url = ds.tunnel_urls[str(a2a_port)]
        self._check_tunnel_collision(context, ds.sandbox_id, a2a_url, a2a_port)

        # Stage the agent build context onto the VM. The install_commands docker-cp from this dir.
        work_dir = f"/tmp/install-agent-{sandbox.scoped_name(self.agent_name)}"
        agent_ctx_tar = f"{work_dir}/agent-ctx.tar.gz"
        await sandbox.exec_script(f"rm -rf {shlex.quote(work_dir)} && mkdir -p {shlex.quote(work_dir)}")
        logger.info(f"Downloading agent build context {agent.docker_image_artifact.build_context_object_url} -> {agent_ctx_tar}")
        await sandbox.load_object_file(agent.docker_image_artifact.build_context_object_url, agent_ctx_tar)

        # Resolve each required_param via the registry, shlex.quote so str.format
        # substitution produces shell-safe tokens.
        resolvers = _build_param_resolvers(
            container_name=sandbox.scoped_name(self.container_name) if self.container_name else None,
            work_dir=work_dir,
            agent_ctx_tar=agent_ctx_tar,
            # On the host, the agent listens where the sandbox publishes its port, which is where its URL points.
            a2a_port=sandbox.host_port(a2a_port) if host_mode else a2a_port,
            config=get_config(),
            workspace_dir=self.workspace_dir,
            agent_name=self.agent_name,
            litellm_api_key_override=user_overrides.get("litellm_api_key"),
            # A host install on a local sandbox runs on this machine itself, not in a container.
            sandbox_type=None if host_mode else sandbox.type,
        )
        resolved: dict[str, str] = {}
        for name in required_params:
            if name not in resolvers:
                raise RuntimeError(
                    f"Agent {agent.id} declares required_params={name!r} but no resolver is registered "
                    f"in InstallAgentTaskStep._build_param_resolvers. Known params: {sorted(resolvers)}"
                )
            try:
                raw = resolvers[name]()
            except (KeyError, ConfigError) as e:
                raise RuntimeError(f"Failed to resolve required_param {name!r}: {e}") from e
            resolved[name] = shlex.quote(raw)

        target = "the VM host" if host_mode else f"container '{self.container_name}'"
        logger.info(
            f"Running {len(install_commands)} install command(s) against {target} "
            f"on sandbox {ds.sandbox_id} (params resolved: {sorted(resolved)})"
        )
        for i, cmd in enumerate(install_commands, 1):
            try:
                formatted = cmd.format(**resolved)
            except KeyError as e:
                raise RuntimeError(
                    f"{commands_key}[{i-1}] references undeclared placeholder {{{e.args[0]}}}; "
                    f"add it to the extension's required_params"
                )
            logger.info(f"  [{i}/{len(install_commands)}] {formatted.splitlines()[0][:140]}")
            await sandbox.exec_script(formatted)
        if not host_mode:  # copied into the container; a host install may run from it
            await sandbox.exec_script(f"rm -rf {shlex.quote(work_dir)}")

        agent_card = await self._wait_for_agent_card(a2a_url)
        logger.info(f"Agent '{self.agent_name}' reachable at {a2a_url} (card name={agent_card.get('name')})")

        deployed = self._register_instance(agent, a2a_url, ds, agent_card)

        context.deployed_agents.append(DeployedAgent(
            agent_name=self.agent_name,
            api_url=a2a_url,
            sandbox_id=ds.sandbox_id,
            sandbox_type=ds.sandbox_type,
            a2a_url=a2a_url,
            a2a_card=agent_card,
            instance_id=deployed.instance_id,
            on_host=host_mode,
        ))

        default_model = agent.metadata.get("default_model") or get_config().get_model_for_role("agent")
        if default_model and not context.default_agent_model:
            context.default_agent_model = default_model
            logger.info(f"Set context.default_agent_model to '{default_model}'")

        return context

    def _register_instance(self, agent, a2a_url: str, ds, agent_card: dict):
        """Register the installed agent in the instance store, same as A2AAgent.deploy(),
        so store-resolving consumers (AddSkillsTaskStep, the a2a-agent CLIs) can find it."""
        from agent_env.a2a_agent.a2a_agent import DeployedA2AAgent
        from agent_env.a2a_agent.store import get_a2a_agent_instance_store

        deployed = get_a2a_agent_instance_store().create_instance(
            DeployedA2AAgent(
                agent_id=agent.id,
                agent_version=agent.version,
                a2a_url=a2a_url,
                sandbox_id=ds.sandbox_id,
                agent_card=agent_card,
                sandbox_type=ds.sandbox_type,
                on_host=self.container_name is None,
            ),
            ttl_seconds=self._instance_ttl_seconds(ds),
        )
        logger.info(f"Registered installed agent '{self.agent_name}' as instance {deployed.instance_id}")
        return deployed

    @staticmethod
    def _instance_ttl_seconds(ds) -> int:
        # The installed agent lives exactly as long as its host sandbox, so mirror the
        # sandbox expiry when the context record carries one.
        if ds.expires_at_utc:
            try:
                expires = datetime.strptime(ds.expires_at_utc, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
                return max(60, int((expires - datetime.now(timezone.utc)).total_seconds()))
            except ValueError:
                pass
        return TaskStep.DEFAULT_TTL_SECONDS

    async def _extract_install_extension(self, sandbox, agent_image_name: str) -> dict:
        """Run the agent image briefly with --entrypoint python3 to dump its AGENT_CARD and find the install/v1 extension."""
        from agent_env.a2a_agent import A2AAgent

        script = (
            "import sys, json; sys.path.insert(0, '/app'); "
            "from a2a_server import AGENT_CARD; "
            "print(json.dumps(AGENT_CARD.model_dump()))"
        )
        exit_code, stdout, stderr = await sandbox.exec_with_output(
            "sudo", "docker", "run", "--rm", "--entrypoint", "python3",
            agent_image_name, "-c", script,
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to extract AGENT_CARD from {agent_image_name}: {stderr or stdout}")
        try:
            card = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Could not parse AGENT_CARD JSON from {agent_image_name}: {e}; stdout={stdout[:500]!r}")
        install_ext = A2AAgent.find_extension(card, A2AAgent.EXT_INSTALL)
        if install_ext is None:
            raise RuntimeError(
                f"Agent image {agent_image_name} does not declare {A2AAgent.EXT_INSTALL} on its card; "
                f"cannot use InstallAgentTaskStep with this agent"
            )
        return install_ext

    def _check_tunnel_collision(self, context: TaskStepContext, sandbox_id: Optional[str], a2a_url: str, a2a_port: int) -> None:
        # same sandbox + same port => identical tunnel URL
        for other in context.deployed_agents:
            if other.sandbox_id == sandbox_id and other.api_url == a2a_url:
                raise RuntimeError(f"Tunnel for port {a2a_port} ({a2a_url}) already serves deployed agent '{other.agent_name}' on sandbox {sandbox_id}; pass a distinct a2a_port on this step")

    async def _wait_for_agent_card(self, a2a_url: str, timeout_s: int = 300, poll_s: int = 5) -> dict:
        deadline = asyncio.get_event_loop().time() + timeout_s
        last_err: Exception | None = None
        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(f"{a2a_url}/.well-known/agent.json", timeout=10)
                    if resp.status_code == 200:
                        return resp.json()
                except Exception as e:
                    last_err = e
                await asyncio.sleep(poll_s)
        raise RuntimeError(
            f"Agent card at {a2a_url}/.well-known/agent.json did not respond within {timeout_s}s "
            f"(last error: {type(last_err).__name__}: {last_err})"
        )
