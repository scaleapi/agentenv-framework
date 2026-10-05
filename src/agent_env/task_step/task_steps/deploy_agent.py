"""Deploy A2A agent task step."""
from __future__ import annotations

import asyncio
import logging
from typing import ClassVar, Optional

import httpx
from agentenv_protocol.a2a_agent import (
    NamespaceChangelogEnableResponse,
    ObjectChangelogApplyResponse,
    ObjectSnapshotLoadResponse,
)

from agent_env.a2a_agent import protocol
from agent_env.a2a_agent.object_transfer import (
    REPLY_TIMEOUT_SECONDS,
    TRANSFER_TIMEOUT_SECONDS,
    bounded_echo,
    changelog_apply_call,
    changelog_enable_call,
    check_changelog_applied,
    invoke_transfer,
    snapshot_load_call,
)
from agent_env.config import get_config
from agent_env.env.env import DeployedEnv, DeployedSandboxEnv, Env
from agent_env.providers.sandbox_providers.sandbox_provider import reachable_url, sandbox_request_headers_for_url
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.attribution import deploy_attribution

logger = logging.getLogger(__name__)


def _mcp_add_body(mcp_ext: dict, url: str, headers: dict | None, card_name: str | None) -> dict:
    """Body for the agent's mcp-config `add`; the card name is relayed verbatim when the agent's card lists `name` as optional."""
    body: dict = {"url": url}
    if headers:
        body["headers"] = headers
    add_request = mcp_ext.get("params", {}).get("methods", {}).get("add", {}).get("request", {})
    if card_name and "name" in (add_request.get("optional") or []):
        body["name"] = card_name
    return body


def _choose_mcp_url(agent, env) -> str:
    """The env's MCP URL, as reachable from wherever the agent runs; an env outside our sandboxes gives its own."""
    if not isinstance(env, DeployedSandboxEnv):
        return env.mcp_url
    return reachable_url(env.mcp_url, from_sandbox_type=env.sandbox_type, to_sandbox_type=agent.sandbox_type)


def _live_mcp_url(env_id: str) -> str:
    """The MCP URL of an env that is referenced by id but never deployed here.

    Such an env is a pointer at a live endpoint: it exposes ``mcp_url`` and has no
    deployment to bind to.
    """
    url = getattr(Env.get(env_id), "mcp_url", None)
    if not isinstance(url, str) or not url.strip().startswith(("http://", "https://")):
        raise RuntimeError(
            f"Env '{env_id}' is not in context.deployed_envs; add a DeployEnvTaskStep "
            "for it, or reference an env that exposes a live http(s) 'mcp_url'."
        )
    return url.strip()


def _skill_fields(skill: dict) -> dict:
    """A ``skills`` entry as ``A2AAgent.register_skill`` arguments. ``content`` and ``s3_uri`` are
    older names for ``skill_md`` and ``skill_s3_url``; ``skill_md`` is sent verbatim."""
    skill_md = skill.get("skill_md", skill.get("content"))
    return {
        "name": skill.get("name", ""),
        "description": skill.get("description", ""),
        "skill_md": skill_md,
        "object_url": (
            skill.get("skill_s3_url", skill.get("s3_uri")) if skill_md is None else None
        ),
    }


class DeployAgentTaskStep(TaskStep):
    type: ClassVar[str] = "deploy_agent"
    entity_refs = (
        EntityRef.env("env_ids[]"),
        EntityRef.agent("a2a_agent_id", version_field="a2a_agent_version"),
        EntityRef.artifact(
            "agent_snapshot_files_artifact_id",
            version_field="agent_snapshot_files_artifact_version",
            artifact_type="file_artifact_universe",
        ),
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_ids: Optional[list[str]] = None,
        env_step_id: Optional[str] = None,
        a2a_agent_id: Optional[str] = None,
        a2a_agent_version: Optional[int] = None,
        agent_name: Optional[str] = None,
        agent_description: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
        skills: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
        disk_size_gb: Optional[float] = None,
        litellm_base_url: Optional[str] = None,
        sandbox_type: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        cpu: Optional[float] = None,
        memory_mb: Optional[int] = None,
        agent_snapshot_files_artifact_id: Optional[str] = None,
        agent_snapshot_files_artifact_version: Optional[int] = None,
        agent_snapshot_target_context_id: Optional[str] = None,
        network_policy: Optional[dict] = None,
        role: Optional[str] = None,
        enable_agent_changelog: bool = False,
        agent_changelog_s3_prefix: Optional[str] = None,
        agent_changelog_toolcall_position_exclusive: Optional[int] = None,
        sandbox_name: Optional[str] = None,
        enable_docker: bool = False,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        metadata: Optional[dict] = None,
        agent_changelog_object_url: Optional[str] = None,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_ids = env_ids or []
        self.env_step_id = env_step_id
        self.a2a_agent_id = a2a_agent_id
        self.a2a_agent_version = a2a_agent_version
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
        self.agent_description = agent_description
        self.env_vars = env_vars or {}
        self.skills = skills or []
        self.system_prompt = system_prompt
        self.disk_size_gb = disk_size_gb
        self.litellm_base_url = litellm_base_url
        self.sandbox_type = sandbox_type
        self.ttl_seconds = ttl_seconds
        self.cpu = cpu
        self.memory_mb = memory_mb
        self.agent_snapshot_files_artifact_id = agent_snapshot_files_artifact_id
        self.agent_snapshot_files_artifact_version = agent_snapshot_files_artifact_version
        self.agent_snapshot_target_context_id = agent_snapshot_target_context_id
        self.metadata = metadata or {}
        self.network_policy = NetworkPolicy.from_dict(network_policy).to_dict() if network_policy else None
        self.role = role
        self.enable_agent_changelog = enable_agent_changelog
        self.agent_changelog_object_url = agent_changelog_object_url
        self.agent_changelog_s3_prefix = agent_changelog_s3_prefix
        self.agent_changelog_toolcall_position_exclusive = agent_changelog_toolcall_position_exclusive
        if self.agent_changelog_object_url and self.agent_changelog_s3_prefix:
            raise ValueError(
                "set only one of agent_changelog_object_url and "
                "agent_changelog_s3_prefix"
            )
        if self.agent_changelog_toolcall_position_exclusive is not None and not (
            self.agent_changelog_object_url or self.agent_changelog_s3_prefix
        ):
            raise ValueError(
                "agent_changelog_toolcall_position_exclusive requires a changelog source"
            )
        self.sandbox_name = sandbox_name
        # Rootless Docker-in-Docker for the agent (not the VM socket); VM mode only.
        self.enable_docker = enable_docker

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_ids"] = self.env_ids
        base["env_step_id"] = self.env_step_id
        base["a2a_agent_id"] = self.a2a_agent_id
        base["a2a_agent_version"] = self.a2a_agent_version
        base["agent_name"] = self.agent_name
        base["agent_description"] = self.agent_description
        base["env_vars"] = self.env_vars
        base["skills"] = self.skills
        base["system_prompt"] = self.system_prompt
        base["disk_size_gb"] = self.disk_size_gb
        base["litellm_base_url"] = self.litellm_base_url
        base["sandbox_type"] = self.sandbox_type
        base["ttl_seconds"] = self.ttl_seconds
        base["cpu"] = self.cpu
        base["memory_mb"] = self.memory_mb
        base["agent_snapshot_files_artifact_id"] = self.agent_snapshot_files_artifact_id
        base["agent_snapshot_files_artifact_version"] = self.agent_snapshot_files_artifact_version
        base["agent_snapshot_target_context_id"] = self.agent_snapshot_target_context_id
        base["metadata"] = self.metadata
        base["network_policy"] = self.network_policy
        base["role"] = self.role
        base["enable_agent_changelog"] = self.enable_agent_changelog
        base["agent_changelog_object_url"] = self.agent_changelog_object_url
        base["agent_changelog_s3_prefix"] = self.agent_changelog_s3_prefix
        base["agent_changelog_toolcall_position_exclusive"] = self.agent_changelog_toolcall_position_exclusive
        base["sandbox_name"] = self.sandbox_name
        base["enable_docker"] = self.enable_docker
        return base

    @classmethod
    def from_dict(cls, data: dict) -> DeployAgentTaskStep:
        return cls(
            **cls._base_from_dict(data),
            env_ids=data.get("env_ids"),
            env_step_id=data.get("env_step_id"),
            a2a_agent_id=data.get("a2a_agent_id"),
            a2a_agent_version=data.get("a2a_agent_version"),
            agent_name=data.get("agent_name"),
            agent_description=data.get("agent_description"),
            env_vars=data.get("env_vars"),
            skills=data.get("skills"),
            system_prompt=data.get("system_prompt"),
            disk_size_gb=data.get("disk_size_gb"),
            litellm_base_url=data.get("litellm_base_url"),
            sandbox_type=data.get("sandbox_type"),
            ttl_seconds=data.get("ttl_seconds"),
            cpu=data.get("cpu"),
            memory_mb=data.get("memory_mb"),
            agent_snapshot_files_artifact_id=data.get("agent_snapshot_files_artifact_id"),
            agent_snapshot_files_artifact_version=data.get("agent_snapshot_files_artifact_version"),
            agent_snapshot_target_context_id=data.get("agent_snapshot_target_context_id"),
            metadata=dict(data.get("metadata") or {}),
            network_policy=data.get("network_policy"),
            role=data.get("role"),
            enable_agent_changelog=data.get("enable_agent_changelog", False),
            agent_changelog_object_url=data.get("agent_changelog_object_url"),
            agent_changelog_s3_prefix=data.get("agent_changelog_s3_prefix"),
            agent_changelog_toolcall_position_exclusive=data.get("agent_changelog_toolcall_position_exclusive"),
            sandbox_name=data.get("sandbox_name"),
            enable_docker=data.get("enable_docker", False),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent
        from agent_env.a2a_agent.a2a_agent import DEFAULT_A2A_PORT
        from agent_env.providers.sandbox_providers.sandbox_provider import (
            SANDBOX_MODE_VM,
            build_sandbox_provider,
            get_sandbox_provider,
        )

        # Resolved before anything is provisioned so an unbound env fails before a sandbox exists.
        env_deployments = {env_id: self._deployed_env(context, env_id) for env_id in self.env_ids}

        existing = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
        if existing is not None:
            raise RuntimeError(f"Agent with name '{self.agent_name}' is already deployed")

        linked_sandbox = None
        if self.sandbox_name:
            ds = next((s for s in context.deployed_sandboxes if s.sandbox_name == self.sandbox_name), None)
            if ds is None:
                raise RuntimeError(
                    f"Sandbox '{self.sandbox_name}' not found in context.deployed_sandboxes; "
                    f"deploy it with DeploySandboxTaskStep before this step"
                )
            if ds.sandbox_mode != SANDBOX_MODE_VM:
                raise RuntimeError(
                    f"Cannot link agent to sandbox '{self.sandbox_name}' (mode={ds.sandbox_mode!r}); "
                    f"only VM-mode sandboxes can host an additional agent container"
                )
            if not ds.tunnel_urls or str(DEFAULT_A2A_PORT) not in ds.tunnel_urls:
                raise RuntimeError(
                    f"Sandbox '{self.sandbox_name}' does not expose port {DEFAULT_A2A_PORT}; "
                    f"re-deploy DeploySandboxTaskStep with exposed_ports=[{DEFAULT_A2A_PORT}]"
                )
            provider = (
                build_sandbox_provider(ds.sandbox_type) if ds.sandbox_type else get_sandbox_provider()
            )
            linked_sandbox = await provider.get_sandbox(ds.sandbox_id)
            logger.info(
                f"Linking agent '{self.agent_name}' to pre-existing sandbox "
                f"'{self.sandbox_name}' (sandbox_id={ds.sandbox_id}, type={ds.sandbox_type})"
            )

        config = get_config()
        user_overrides = context.metadata.get("user_overrides", {})
        a2a_agent_id = user_overrides.get("a2a_agent_id") or self.a2a_agent_id or config.get_default_a2a_agent_id()
        a2a_agent_version = (
            user_overrides["a2a_agent_version"]
            if "a2a_agent_version" in user_overrides
            else self.a2a_agent_version
        )
        agent = A2AAgent.get(a2a_agent_id, a2a_agent_version)
        requested_disk = user_overrides.get("agent_disk_size_gb") or self.disk_size_gb
        min_disk = agent.metadata.get("min_disk_size_gb")
        agent_disk_size_gb = max(requested_disk or 0, min_disk or 0) or None
        env_vars = dict(self.env_vars) if self.env_vars else {}
        override_litellm_key = user_overrides.get("litellm_api_key")
        if override_litellm_key:
            env_vars["LITELLM_API_KEY"] = override_litellm_key
        resolved_litellm_base_url = user_overrides.get("litellm_base_url") or self.litellm_base_url
        if resolved_litellm_base_url:
            env_vars["LITELLM_BASE_URL"] = resolved_litellm_base_url

        resolved_sandbox_type = user_overrides.get("agent_sandbox") or self.sandbox_type
        deploy_kwargs = {"env_vars": env_vars if env_vars else None}
        if agent_disk_size_gb is not None:
            deploy_kwargs["disk_size_gb"] = agent_disk_size_gb
        if resolved_sandbox_type is not None:
            deploy_kwargs["sandbox_type"] = resolved_sandbox_type
        resolved_ttl = self._resolved_ttl_seconds(context)
        deploy_kwargs["ttl_seconds"] = resolved_ttl
        if self.cpu is not None:
            deploy_kwargs["cpu"] = self.cpu
        if self.memory_mb is not None:
            deploy_kwargs["memory"] = self.memory_mb
        deploy_kwargs["attribution"] = deploy_attribution(self, context)
        _policy_override = user_overrides.get("network_policy")
        resolved_policy = _policy_override if _policy_override is not None else self.network_policy
        if resolved_policy is not None:
            deploy_kwargs["network_policy"] = NetworkPolicy.from_dict(resolved_policy)
        if self.enable_docker:
            deploy_kwargs["enable_docker"] = True
        if linked_sandbox is not None:
            deploy_kwargs["sandbox"] = linked_sandbox
        deployed = await agent.deploy(**deploy_kwargs)
        a2a_url = deployed.a2a_url
        card = deployed.agent_card
        logger.info(f"A2A agent '{a2a_agent_id}' deployed at {a2a_url}")

        env_mcp_urls: list[tuple[str, str, dict | None, str | None]] = []  # (env_id, url, headers, card_name)
        for env_id in self.env_ids:
            deployed_env = env_deployments[env_id]
            if deployed_env is not None:
                url = _choose_mcp_url(deployed, deployed_env)
                card_name = (deployed_env.environment_card or {}).get("name")
                if not card_name:
                    logger.warning(f"Env {env_id} has no env card name; the agent will pick its own MCP alias")
            else:
                url = _live_mcp_url(env_id)
                card_name = None
            env_mcp_urls.append((env_id, url, sandbox_request_headers_for_url(url) or None, card_name))

        mcp_ext = A2AAgent.find_extension(card, A2AAgent.EXT_MCP_CONFIG)
        if mcp_ext and env_mcp_urls:
            endpoint = a2a_url + mcp_ext["params"]["endpoint"]
            async with httpx.AsyncClient() as client:
                for env_id, mcp_url, headers, card_name in env_mcp_urls:
                    body = _mcp_add_body(mcp_ext, mcp_url, headers, card_name)
                    logger.info(f"Registering MCP for env {env_id} via {mcp_url}")
                    resp = await client.post(endpoint, json=body, timeout=180)
                    if resp.status_code >= 400:
                        logger.error(f"Failed to register MCP {env_id}: {resp.status_code} {resp.text}")
                    resp.raise_for_status()
                    logger.info(f"Registered MCP {env_id}: {resp.json()}")

        skill_ext = A2AAgent.find_extension(card, A2AAgent.EXT_SKILL_CONFIG)
        if skill_ext and self.skills:
            for skill in self.skills:
                result = await A2AAgent.register_skill(deployed, **_skill_fields(skill))
                logger.info(f"Registered skill: {result}")

        if self.system_prompt is not None:
            config_ext = A2AAgent.find_extension(card, A2AAgent.EXT_AGENT_CONFIG)
            if config_ext is None:
                logger.warning(f"Agent '{self.agent_name}' does not advertise {A2AAgent.EXT_AGENT_CONFIG} — skipping system_prompt")
            else:
                system_prompt = self.system_prompt
                seed = context.metadata.get("seed", {})
                if seed:
                    for key, value in seed.items():
                        system_prompt = system_prompt.replace(f"<{key}>", str(value))
                supported = (config_ext.get("params") or {}).get("methods", {}).get("set", {}).get("request", {}).get("supported", [])
                if "system_prompt" not in supported:
                    logger.warning(f"Agent '{self.agent_name}' does not list 'system_prompt' in supported agent-config fields — skipping")
                else:
                    endpoint = a2a_url + (config_ext.get("params") or {}).get("endpoint", "/ext/agent-config")
                    await protocol.post_agent_config(endpoint, {"system_prompt": system_prompt})
                    logger.info(f"Set system_prompt on agent '{self.agent_name}'")

        identity_payload: dict[str, str] = {"name": self.agent_name}
        if self.agent_description is not None:
            identity_payload["description"] = self.agent_description
        if self.role is not None:
            identity_payload["role"] = self.role
        config_ext = A2AAgent.find_extension(card, A2AAgent.EXT_AGENT_CONFIG)
        if config_ext is not None:
            supported = (config_ext.get("params") or {}).get("methods", {}).get("set", {}).get("request", {}).get("supported", [])
            identity_payload = {k: v for k, v in identity_payload.items() if k in supported}
            if identity_payload:
                endpoint = a2a_url + (config_ext.get("params") or {}).get("endpoint", "/ext/agent-config")
                await protocol.post_agent_config(endpoint, identity_payload)
                card.update(identity_payload)
                logger.info(f"Stamped card identity on '{self.agent_name}': {sorted(identity_payload)}")

        default_model = config.get_model_for_role("agent")
        if default_model and not context.default_agent_model:
            context.default_agent_model = default_model
            logger.info(f"Set context.default_agent_model to '{default_model}'")

        if self.agent_snapshot_files_artifact_id:
            await self._load_snapshot(a2a_url, card, a2a_agent_id, context)

        if self.agent_changelog_object_url or self.agent_changelog_s3_prefix:
            await self._apply_agent_changelog(a2a_url, card, context)

        if self.enable_agent_changelog:
            await self._configure_agent_changelog(
                a2a_url, card, context, expires_in=resolved_ttl
            )

        context.deployed_agents.append(DeployedAgent(
            agent_name=self.agent_name,
            api_url=a2a_url,
            sandbox_id=deployed.sandbox_id,
            sandbox_type=deployed.sandbox_type,
            a2a_url=a2a_url,
            a2a_card=card,
            instance_id=deployed.instance_id,
            role=self.role,
            network_policy=deployed.network_policy,
        ))
        return context

    @staticmethod
    def _deploy_step_of(deployed_env: DeployedEnv) -> Optional[str]:
        return (deployed_env.metadata or {}).get("deploy_step_id")

    def _deployed_env(self, context: TaskStepContext, env_id: str) -> Optional[DeployedEnv]:
        """The deployment of `env_id` this step is bound to, or None if it isn't deployed here."""
        matches = [d for d in context.deployed_envs if d.env_id == env_id]
        if self.env_step_id is not None:
            # A filter, not a hint: an unmatched id is an error, never a substitute.
            # Envs with no deployments fall through to the caller's live-endpoint path.
            named = [d for d in matches if self._deploy_step_of(d) == self.env_step_id]
            if matches and not named:
                raise RuntimeError(
                    f"deploy_agent '{self.id}': env_step_id '{self.env_step_id}' names no deployment of "
                    f"env '{env_id}' (deployed by: {[self._deploy_step_of(d) for d in matches]})"
                )
            matches = named
        if len(matches) > 1:
            raise RuntimeError(
                f"deploy_agent '{self.id}': {len(matches)} deployments of env '{env_id}' "
                f"(deployed by: {[self._deploy_step_of(d) for d in matches]}). Set env_step_id "
                "to the deploy_env step this agent should bind to."
            )
        return matches[0] if matches else None

    def _resolved_ttl_seconds(self, context: TaskStepContext) -> int:
        override = context.metadata.get("user_overrides", {}).get("ttl_seconds")
        if override is not None:
            return override
        if self.ttl_seconds is not None:
            return self.ttl_seconds
        return TaskStep.DEFAULT_TTL_SECONDS

    async def _configure_agent_changelog(
        self,
        a2a_url: str,
        card: dict,
        context: TaskStepContext,
        *,
        expires_in: int,
    ) -> None:
        """Enable whole-writable-layer changelog capture via the snapshot enable-changelog method; records the derived prefix in context.metadata['agent_changelog']."""
        from agent_env.a2a_agent import A2AAgent

        snapshot_ext = A2AAgent.find_extension(card, A2AAgent.EXT_SNAPSHOT)
        method, path = A2AAgent.operation(snapshot_ext, A2AAgent.SNAPSHOT_METHOD_ENABLE_CHANGELOG)
        if method is None:
            raise RuntimeError(
                f"Agent '{self.agent_name}' does not support the snapshot "
                f"'{A2AAgent.SNAPSHOT_METHOD_ENABLE_CHANGELOG}' method; cannot enable_agent_changelog"
            )
        instance_id = (context.instance_id or "noinstance").replace("/", "_")
        agent_name = self.agent_name.replace("/", "_")
        config = get_config()
        store = config.get_object_store()
        namespace_url = store.object_url(f"{config.get_artifact_key_prefix()}agent_changelog/{instance_id}/{agent_name}")
        call = await asyncio.to_thread(
            changelog_enable_call,
            method,
            store,
            agent_name=self.agent_name,
            namespace_url=namespace_url,
            expires_in=expires_in,
        )
        answer = await invoke_transfer(
            a2a_url + path,
            call,
            verb="POST",
            operation="changelog enable",
            timeout=REPLY_TIMEOUT_SECONDS,
            response_model=NamespaceChangelogEnableResponse,
        )
        if call.mode == "objects":
            roots, object_url = answer.roots, namespace_url
        else:
            roots = answer.get("roots")
            object_url = bounded_echo(namespace_url, answer.get("s3_prefix") or namespace_url)
        context.metadata.setdefault("agent_changelog", []).append({
            "agent_name": self.agent_name,
            "roots": roots,
            "transfer_mode": call.mode,
            "object_url": object_url,
        })
        logger.info(f"agent-changelog capture enabled on '{self.agent_name}': {object_url}")

    async def _apply_agent_changelog(self, a2a_url: str, card: dict, context: TaskStepContext) -> None:
        """Rewind this fresh agent to a point in a source changelog (fs + conversation) via the snapshot apply-changelog method and resume."""
        from agent_env.a2a_agent import A2AAgent

        snapshot_ext = A2AAgent.find_extension(card, A2AAgent.EXT_SNAPSHOT)
        method, path = A2AAgent.operation(snapshot_ext, A2AAgent.SNAPSHOT_METHOD_APPLY_CHANGELOG)
        if method is None:
            raise RuntimeError(
                f"Agent '{self.agent_name}' does not support the snapshot "
                f"'{A2AAgent.SNAPSHOT_METHOD_APPLY_CHANGELOG}' method; cannot apply agent changelog"
            )
        store = get_config().get_object_store()
        source_url = self.agent_changelog_object_url or self.agent_changelog_s3_prefix
        call = await asyncio.to_thread(
            changelog_apply_call,
            method,
            store,
            agent_name=self.agent_name,
            source_url=source_url,
            portable=bool(self.agent_changelog_object_url),
            up_to_tool_call_exclusive=self.agent_changelog_toolcall_position_exclusive,
            resume_conversation=True,
            target_context_id=self.agent_snapshot_target_context_id,
        )
        answer = await invoke_transfer(
            a2a_url + path,
            call,
            verb="PUT",
            operation="changelog apply",
            timeout=TRANSFER_TIMEOUT_SECONDS,
            response_model=ObjectChangelogApplyResponse,
        )
        if call.mode == "objects":
            check_changelog_applied(answer, call, agent_name=self.agent_name)
            context_id = answer.context_id
        else:
            context_id = answer.get("context_id")
        context.metadata.setdefault("agent_changelog_rewinds", []).append({
            "agent_name": self.agent_name,
            "context_id": context_id,
            "up_to_tool_call_exclusive": self.agent_changelog_toolcall_position_exclusive,
            "transfer_mode": call.mode,
            "object_url": source_url,
        })
        logger.info(
            f"agent-changelog applied onto '{self.agent_name}' from {source_url} "
            f"(position_exclusive={self.agent_changelog_toolcall_position_exclusive}, context_id={context_id})"
        )

    async def _load_snapshot(
        self,
        a2a_url: str,
        card: dict,
        a2a_agent_id: str,
        context: TaskStepContext,
    ) -> None:
        """Restore a snapshot into the freshly-deployed agent via /ext/snapshot."""
        from agent_env.a2a_agent import A2AAgent
        from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse

        snapshot_ext = A2AAgent.find_extension(card, A2AAgent.EXT_SNAPSHOT)
        if not snapshot_ext:
            raise RuntimeError(
                f"Cannot load snapshot: agent '{a2a_agent_id}' does not advertise "
                f"the snapshot extension ({A2AAgent.EXT_SNAPSHOT})"
            )
        universe = FileArtifactUniverse.get(
            self.agent_snapshot_files_artifact_id,
            self.agent_snapshot_files_artifact_version,
        )
        if not universe.bundle_object_url:
            raise RuntimeError(
                f"FileArtifactUniverse '{universe.id}' v{universe.version} has no bundle_s3_url; "
                "snapshot universes must be created via put_existing or put_bundled"
            )
        load_method, load_path = A2AAgent.operation(snapshot_ext, "load")
        call = await asyncio.to_thread(
            snapshot_load_call,
            load_method,
            get_config().get_object_store(),
            agent_name=self.agent_name,
            bundle_url=universe.bundle_object_url,
            file_names=set((universe.file_artifact_refs or universe.file_artifact_ids or {}).keys()),
            target_context_id=self.agent_snapshot_target_context_id,
        )
        answer = await invoke_transfer(
            a2a_url + load_path,
            call,
            verb="PUT",
            operation="snapshot load",
            timeout=TRANSFER_TIMEOUT_SECONDS,
            response_model=ObjectSnapshotLoadResponse,
        )
        loaded_context_id = (
            answer.context_id if call.mode == "objects" else answer.get("context_id")
        )
        logger.info(
            f"Loaded snapshot universe={universe.id} v{universe.version} "
            f"into agent '{self.agent_name}' as context_id={loaded_context_id}"
        )
        if loaded_context_id:
            loaded = context.metadata.setdefault("agent_loaded_snapshots", [])
            loaded.append({
                "agent_name": self.agent_name,
                "context_id": loaded_context_id,
                "source_artifact_id": universe.id,
                "source_artifact_version": universe.version,
            })
