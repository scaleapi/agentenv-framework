"""Deploy A2A agent task step."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import ClassVar, Optional
from urllib.parse import parse_qs, urlparse

import httpx

from agent_env.a2a_agent import protocol
from agent_env.env.env import DeployedEnv, Env
from agent_env.providers.sandbox_provider import reachable_url, sandbox_request_headers_for_url
from agent_env.providers.sandbox import NetworkPolicy
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.attribution import attribution_of, metadata_from_legacy_document

logger = logging.getLogger(__name__)


# Re-sign with this much buffer left on the original URL. 2h covers a worst-case
# task: the workspace tarball can be fetched again later in the run (snapshot save
# round-trip, mid-task /ext/snapshot reloads, retried activities), and any of
# those need the URL to still be valid an hour or two after deploy_agent ran.
_PRESIGN_REFRESH_BUFFER_SECONDS = 2 * 3600
# 7 days is the SigV4 max for IAM-credentialed presigning.
_PRESIGN_REFRESH_TTL_SECONDS = 7 * 24 * 3600


async def _card_name(client: httpx.AsyncClient, card_url: str | None) -> str | None:
    """The MCP server name from the env's card; None without a card."""
    if not card_url:
        return None
    try:
        resp = await client.get(card_url, timeout=30)
        resp.raise_for_status()
        name = resp.json().get("name")
        return name if isinstance(name, str) and name else None
    except Exception as e:
        logger.warning(f"Could not read the environment card at {card_url}: {e!r}")
        return None


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
    """The env's MCP URL, as reachable from wherever the agent runs."""
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


def _maybe_resign_presigned_url(url: str) -> str:
    """If ``url`` is a presigned S3 GET URL near expiry, re-sign it.

    Why: sandbox pods can't reach the agent-env artifact bucket (their IAM,
    `sandbox-pods-sa`, only covers the sandbox user-artifacts bucket), so
    the hub upserts a long-lived presigned URL as `bundle_s3_url`. If the
    URL was minted >7 days ago it's expired and the sidecar gets 403.
    The agent-env worker has the cross-account perms to re-sign at deploy
    time, which is the right boundary.

    Pass-through for non-presigned URLs (raw `s3://`, plain HTTPS, other
    schemes).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return url
    qs = parse_qs(parsed.query)
    amz_date = (qs.get("X-Amz-Date") or [None])[0]
    amz_expires_str = (qs.get("X-Amz-Expires") or [None])[0]
    if not amz_date or not amz_expires_str:
        # Not a SigV4 presigned URL — leave alone.
        return url
    try:
        signed_at = datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        expires_at = signed_at.timestamp() + int(amz_expires_str)
    except (ValueError, TypeError):
        logger.warning("Could not parse presigned URL expiry; passing through")
        return url

    now_ts = datetime.now(timezone.utc).timestamp()
    if expires_at - now_ts > _PRESIGN_REFRESH_BUFFER_SECONDS:
        return url

    # Recover bucket + key from the URL host/path. SigV4 presigned URLs use
    # virtual-hosted-style: `https://<bucket>.s3.<region>.amazonaws.com/<key>`.
    host_parts = parsed.netloc.split(".")
    if len(host_parts) < 4 or host_parts[1] != "s3":
        logger.warning(
            "Presigned URL near expiry but host shape unexpected (%s); passing through", parsed.netloc,
        )
        return url
    bucket = host_parts[0]
    # Region sits at index 2 for virtual-hosted URLs like
    # <bucket>.s3.<region>.amazonaws.com; fall back to None so boto3 uses
    # the environment default if the URL shape is unusual.
    region = host_parts[2] if len(host_parts) >= 5 else None
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        logger.warning("Could not extract bucket/key from presigned URL; passing through")
        return url

    import boto3
    fresh = boto3.client("s3", region_name=region).generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=_PRESIGN_REFRESH_TTL_SECONDS,
    )
    logger.info(
        "Re-signed presigned bundle URL (was expiring at %s, now valid for %ds) bucket=%s key=%s",
        datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        _PRESIGN_REFRESH_TTL_SECONDS,
        bucket,
        key,
    )
    return fresh


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
        priority: Optional[int] = None,
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
        self.priority = priority
        self.metadata = metadata or {}
        self.network_policy = NetworkPolicy.from_dict(network_policy).to_dict() if network_policy else None
        self.role = role
        self.enable_agent_changelog = enable_agent_changelog
        self.agent_changelog_s3_prefix = agent_changelog_s3_prefix
        self.agent_changelog_toolcall_position_exclusive = agent_changelog_toolcall_position_exclusive
        if self.agent_changelog_toolcall_position_exclusive is not None and not self.agent_changelog_s3_prefix:
            raise ValueError("agent_changelog_toolcall_position_exclusive requires agent_changelog_s3_prefix")
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
        base["priority"] = self.priority
        base["metadata"] = self.metadata
        base["network_policy"] = self.network_policy
        base["role"] = self.role
        base["enable_agent_changelog"] = self.enable_agent_changelog
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
            priority=data.get("priority"),
            metadata=metadata_from_legacy_document(data),
            network_policy=data.get("network_policy"),
            role=data.get("role"),
            enable_agent_changelog=data.get("enable_agent_changelog", False),
            agent_changelog_s3_prefix=data.get("agent_changelog_s3_prefix"),
            agent_changelog_toolcall_position_exclusive=data.get("agent_changelog_toolcall_position_exclusive"),
            sandbox_name=data.get("sandbox_name"),
            enable_docker=data.get("enable_docker", False),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent
        from agent_env.a2a_agent.a2a_agent import DEFAULT_A2A_PORT
        from agent_env.providers.sandbox_provider import (
            SANDBOX_MODE_VM,
            build_sandbox_provider,
            get_sandbox_provider,
        )
        from agent_env.config import get_config

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

        # Cost-attribution (project_id / task_id) is delivered per-prompt
        # via the A2A `/ext/agent-config` extension from prompt_agent.py —
        # not baked as container env vars at deploy time. See
        # the openai_agents_sdk A2A agent in the universe-generation pipeline
        # for the reference implementation that reads agent-config and threads
        # `user` / `metadata.tags` into each LiteLLM call.

        resolved_sandbox_type = user_overrides.get("agent_sandbox") or self.sandbox_type
        deploy_kwargs = {"env_vars": env_vars if env_vars else None}
        if agent_disk_size_gb is not None:
            deploy_kwargs["disk_size_gb"] = agent_disk_size_gb
        if resolved_sandbox_type is not None:
            deploy_kwargs["sandbox_type"] = resolved_sandbox_type
        _ttl_override = user_overrides.get("ttl_seconds")
        resolved_ttl = _ttl_override if _ttl_override is not None else self.ttl_seconds
        if resolved_ttl is not None:
            deploy_kwargs["ttl_seconds"] = resolved_ttl
        if self.cpu is not None:
            deploy_kwargs["cpu"] = self.cpu
        if self.memory_mb is not None:
            deploy_kwargs["memory"] = self.memory_mb
        deploy_kwargs["attribution"] = attribution_of(self)
        _priority_override = user_overrides.get("priority")
        resolved_priority = _priority_override if _priority_override is not None else self.priority
        if resolved_priority is not None:
            deploy_kwargs["priority"] = resolved_priority
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

        # Per-server headers the agent sends on MCP requests; Bearer tokens are per-run, never persisted.
        remote_tokens = user_overrides.get("remote_tokens") or {}

        def _headers_for(url: str, env_id: str) -> dict[str, str] | None:
            headers: dict[str, str] = dict(sandbox_request_headers_for_url(url))
            token = remote_tokens.get(env_id)
            if token:
                headers["Authorization"] = f"Bearer {token}"
            return headers or None

        env_mcp_urls: list[tuple[str, str, dict | None, str | None]] = []  # (env_id, url, headers, card_url)
        for env_id in self.env_ids:
            deployed_env = env_deployments[env_id]
            if deployed_env is not None:
                url = _choose_mcp_url(deployed, deployed_env)
                card_url = deployed_env.environment_card_url
            else:
                url = _live_mcp_url(env_id)
                card_url = None
            env_mcp_urls.append((env_id, url, _headers_for(url, env_id), card_url))

        mcp_ext = A2AAgent.find_extension(card, A2AAgent.EXT_MCP_CONFIG)
        if mcp_ext and env_mcp_urls:
            endpoint = a2a_url + mcp_ext["params"]["endpoint"]
            async with httpx.AsyncClient() as client:
                for env_id, mcp_url, headers, card_url in env_mcp_urls:
                    body = _mcp_add_body(mcp_ext, mcp_url, headers, await _card_name(client, card_url))
                    logger.info(f"Registering MCP for env {env_id} via {mcp_url}")
                    resp = await client.post(endpoint, json=body, timeout=180)
                    if resp.status_code >= 400:
                        logger.error(f"Failed to register MCP {env_id}: {resp.status_code} {resp.text}")
                    resp.raise_for_status()
                    logger.info(f"Registered MCP {env_id}: {resp.json()}")

        skill_ext = A2AAgent.find_extension(card, A2AAgent.EXT_SKILL_CONFIG)
        if skill_ext and self.skills:
            endpoint = a2a_url + skill_ext["params"]["endpoint"]
            async with httpx.AsyncClient() as client:
                for skill in self.skills:
                    payload = {"name": skill.get("name", ""), "description": skill.get("description", "")}
                    if "skill_md" in skill:
                        payload["skill_md"] = skill["skill_md"]
                    elif "content" in skill:
                        payload["skill_md"] = skill["content"]
                    elif "skill_s3_url" in skill:
                        payload["skill_s3_url"] = skill["skill_s3_url"]
                    elif "s3_uri" in skill:
                        payload["skill_s3_url"] = skill["s3_uri"]
                    resp = await client.post(endpoint, json=payload, timeout=180)
                    if resp.status_code >= 400:
                        logger.error(f"Failed to register skill: {resp.status_code} {resp.text}")
                    resp.raise_for_status()
                    logger.info(f"Registered skill: {resp.json()}")

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

        if self.agent_changelog_s3_prefix:
            await self._apply_agent_changelog(a2a_url, card, context)

        if self.enable_agent_changelog:
            await self._configure_agent_changelog(a2a_url, card, context)

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

    async def _configure_agent_changelog(self, a2a_url: str, card: dict, context: TaskStepContext) -> None:
        """Enable whole-writable-layer changelog capture via the snapshot enable-changelog method; records the derived prefix in context.metadata['agent_changelog']."""
        from agent_env.a2a_agent import A2AAgent
        from agent_env.config import get_config

        method = A2AAgent.extension_method(card, A2AAgent.EXT_SNAPSHOT, A2AAgent.SNAPSHOT_METHOD_ENABLE_CHANGELOG)
        if method is None:
            raise RuntimeError(
                f"Agent '{self.agent_name}' does not support the snapshot "
                f"'{A2AAgent.SNAPSHOT_METHOD_ENABLE_CHANGELOG}' method; cannot enable_agent_changelog"
            )
        instance_id = (context.instance_id or "noinstance").replace("/", "_")
        agent_name = self.agent_name.replace("/", "_")
        bucket = get_config().get_s3_bucket()
        prefix = f"s3://{bucket}/agent_changelog/{instance_id}/{agent_name}"
        endpoint = a2a_url + method.get("endpoint", "/ext/snapshot/changelog")
        async with httpx.AsyncClient() as client:
            resp = await client.post(endpoint, json={"s3_prefix": prefix}, timeout=120)
            resp.raise_for_status()
            body = resp.json()
        entry = {
            "agent_name": self.agent_name,
            "s3_prefix": body.get("s3_prefix", prefix),
            "roots": body.get("roots"),
        }
        context.metadata.setdefault("agent_changelog", []).append(entry)
        logger.info(f"agent-changelog capture enabled on '{self.agent_name}': {entry['s3_prefix']}")

    async def _apply_agent_changelog(self, a2a_url: str, card: dict, context: TaskStepContext) -> None:
        """Rewind this fresh agent to a point in a source changelog (fs + conversation) via the snapshot apply-changelog method and resume."""
        from agent_env.a2a_agent import A2AAgent

        method = A2AAgent.extension_method(card, A2AAgent.EXT_SNAPSHOT, A2AAgent.SNAPSHOT_METHOD_APPLY_CHANGELOG)
        if method is None:
            raise RuntimeError(
                f"Agent '{self.agent_name}' does not support the snapshot "
                f"'{A2AAgent.SNAPSHOT_METHOD_APPLY_CHANGELOG}' method; cannot apply agent changelog"
            )
        payload: dict = {"s3_prefix": self.agent_changelog_s3_prefix, "resume_conversation": True}
        if self.agent_changelog_toolcall_position_exclusive is not None:
            payload["up_to_tool_call_exclusive"] = self.agent_changelog_toolcall_position_exclusive
        if self.agent_snapshot_target_context_id:
            payload["target_context_id"] = self.agent_snapshot_target_context_id
        endpoint = a2a_url + method.get("endpoint", "/ext/snapshot/changelog")
        async with httpx.AsyncClient() as client:
            resp = await client.put(endpoint, json=payload, timeout=600)
            resp.raise_for_status()
            body = resp.json()
        context.metadata.setdefault("agent_changelog_rewinds", []).append({
            "agent_name": self.agent_name,
            "context_id": body.get("context_id"),
            "s3_prefix": self.agent_changelog_s3_prefix,
            "up_to_tool_call_exclusive": self.agent_changelog_toolcall_position_exclusive,
        })
        logger.info(
            f"agent-changelog applied onto '{self.agent_name}' from {self.agent_changelog_s3_prefix} "
            f"(position_exclusive={self.agent_changelog_toolcall_position_exclusive}, context_id={body.get('context_id')})"
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
        ext_params = snapshot_ext.get("params") or {}
        load_url = a2a_url + ext_params.get("endpoint", "/ext/snapshot")
        # Re-sign if the stored URL is a presigned URL near expiry (the hub's
        # openclaw-tasks/upsert mints these because sandbox pods lack S3 IAM
        # for the artifact bucket). No-op for raw s3:// or non-S3 URLs.
        bundle_url = _maybe_resign_presigned_url(universe.bundle_object_url)
        load_payload: dict[str, str] = {"s3_prefix": bundle_url}
        if self.agent_snapshot_target_context_id:
            load_payload["target_context_id"] = self.agent_snapshot_target_context_id
        async with httpx.AsyncClient() as client:
            resp = await client.put(load_url, json=load_payload, timeout=180)
            if resp.status_code >= 400:
                raise RuntimeError(f"snapshot load failed: {resp.status_code} {resp.text}")
            load_body = resp.json()
        loaded_context_id = load_body.get("context_id")
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
                "source_bundle_s3_url": universe.bundle_object_url,
            })
