"""A2A Agent model for AgentEnv."""
from __future__ import annotations

import asyncio
import json
import logging
import posixpath
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

from agentenv_protocol.a2a_agent import STANDARD_EXTENSIONS

from agent_env.config import get_config
from agent_env.entity_refs import EntityRef
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, NetworkPolicyUnsupportedError, port_bindings
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.a2a_agent.object_transfer import (
    TRANSFER_TIMEOUT_SECONDS,
    invoke_transfer,
    skill_add_call,
)
from agent_env.a2a_agent.staging import transfer_store
from agent_env.providers.sandbox_providers.local_sandbox import (
    LOCAL_TRUST_ENV,
    LocalSandbox,
    LocalSandboxProvider,
    local_grant_trust,
    start_trusting,
)
from agent_env.providers.sandbox_providers.sandbox_provider import all_sandbox_container_env, all_sandbox_url_rewrites
from agent_env.attribution import Attribution

if TYPE_CHECKING:
    from agent_env.a2a_agent.store import A2AAgentQuery
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.artifact import CliArtifact
    from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
    from agent_env.providers.sandbox_providers.sandbox import Sandbox
    from agent_env.task_step.task_steps.add_skills import Skill

logger = logging.getLogger(__name__)

DEFAULT_A2A_PORT = 8000
AGENT_CARD_TIMEOUT = 300

# Rootless (not the VM socket), so a container escape is an unprivileged VM user,
# not VM root. Digest-pinned because the sidecar runs --privileged.
_DIND_IMAGE = (
    "public.ecr.aws/docker/library/docker:27-dind-rootless"
    "@sha256:e2ac8e8f66ae21a060b0a8e3005c70f6ed9441aabf409434463d1f6eecd38026"
)
_DIND_CONTAINER = "agent-dind"
_DIND_NETWORK = "agent-docker-net"
_DIND_PORT = 2375


@dataclass
class DeployedA2AAgent:
    agent_id: str
    agent_version: int
    a2a_url: str
    sandbox_id: str
    agent_card: dict
    sandbox_type: str | None = None
    instance_id: str | None = None
    created_at_utc: str | None = None
    expires_at_utc: str | None = None
    network_policy: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "DeployedA2AAgent":
        return cls(
            agent_id=data["agent_id"],
            agent_version=data["agent_version"],
            a2a_url=data["a2a_url"],
            sandbox_id=data["sandbox_id"],
            agent_card=data.get("agent_card") or {},
            sandbox_type=data.get("sandbox_type"),
            instance_id=data.get("instance_id"),
            created_at_utc=data.get("created_at_utc"),
            expires_at_utc=data.get("expires_at_utc"),
            network_policy=data.get("network_policy"),
        )


@dataclass(frozen=True)
class NegotiatedAgentConfig:
    endpoint: str
    fields: dict


class A2AAgent:
    type: ClassVar[str] = "a2a_agent"
    toml_refs: ClassVar[tuple[EntityRef, ...]] = (EntityRef.artifact("image", artifact_type="docker_image"),)
    toml_keys: ClassVar[dict[str, type]] = {"image": object, "default_env_vars": dict, "metadata": dict}
    # What an agent.toml's [metadata] may set: the types each key takes, and how a problem names them.
    toml_metadata: ClassVar[dict[str, tuple[tuple[type, ...], str]]] = {
        "default_model": ((str,), "a string"), "min_disk_size_gb": ((int, float), "a number"),
    }

    EXT_MCP_CONFIG: ClassVar[str] = "urn:agentenv:mcp-config/v1"
    EXT_SKILL_CONFIG: ClassVar[str] = "urn:agentenv:skill-config/v1"
    EXT_TRAJECTORY: ClassVar[str] = "urn:agentenv:trajectory/v1"
    EXT_AGENT_CONFIG: ClassVar[str] = "urn:agentenv:agent-config/v1"
    EXT_SNAPSHOT: ClassVar[str] = "urn:agentenv:snapshot/v1"
    EXT_PEER_AGENTS: ClassVar[str] = "urn:agentenv:peer-agents/v1"
    EXT_INSTALL: ClassVar[str] = "urn:agentenv:install/v1"
    EXT_TOOLS: ClassVar[str] = "urn:agentenv:tools/v1"
    EXT_TRIGGERS: ClassVar[str] = "urn:agentenv:triggers/v1"

    # Whole-writable-layer changelog (capture + reconstruct) is exposed as two
    # methods on the snapshot extension, so capability is per-method via
    # extension_method() rather than a separate extension.
    SNAPSHOT_METHOD_ENABLE_CHANGELOG: ClassVar[str] = "enable-changelog"
    SNAPSHOT_METHOD_APPLY_CHANGELOG: ClassVar[str] = "apply-changelog"

    @staticmethod
    def find_extension(card: dict, uri: str) -> dict | None:
        caps = card.get("capabilities") or {}
        for ext in caps.get("extensions", []):
            if ext.get("uri") == uri:
                return ext
        return None

    @staticmethod
    def extension_method(card: dict, uri: str, method: str) -> dict | None:
        """Return an advertised extension method, or ``None`` if absent."""
        return A2AAgent.operation(A2AAgent.find_extension(card, uri), method)[0]

    @staticmethod
    def operation(extension: dict | None, name: str) -> tuple[dict | None, str | None]:
        """An advertised extension's ``name`` method (None when unlisted) and its endpoint path.

        As the protocol renders cards: a method's own endpoint wins; otherwise one the protocol
        serves at the extension's path uses the card's extension endpoint, and one it serves
        elsewhere uses the protocol's path for it."""
        if extension is None:
            return None, None
        config = extension.get("config") or extension.get("params") or {}
        method = (config.get("methods") or {}).get(name)
        if method and method.get("endpoint"):
            return method, method["endpoint"]
        definition = STANDARD_EXTENSIONS.get(extension.get("uri"))
        spec = next(
            (op for op in (definition.operations.values() if definition else ()) if op.name == name),
            None,
        )
        if spec is not None and spec.path != definition.endpoint:
            return method, spec.path
        return method, config.get("endpoint") or (spec.path if spec else None)

    @staticmethod
    def negotiate_agent_config(card: dict, desired: dict) -> Optional[NegotiatedAgentConfig]:
        """Filter ``desired`` to the fields the card's agent-config extension accepts (capability negotiation)."""
        ext = A2AAgent.find_extension(card, A2AAgent.EXT_AGENT_CONFIG)
        if not ext:
            return None
        params = ext.get("params") or {}
        supported = params.get("methods", {}).get("set", {}).get("request", {}).get("supported", [])
        fields = {k: v for k, v in desired.items() if k in supported}
        if not fields:
            return None
        return NegotiatedAgentConfig(params.get("endpoint", "/ext/agent-config"), fields)

    @staticmethod
    async def add_skill(deployed: "DeployedA2AAgent", skill: "Skill") -> dict:
        """Validate ``skill`` and register it on the deployed agent."""
        A2AAgent._skill_extension(deployed)
        await asyncio.to_thread(skill.validate)
        if skill.skill_artifact_id is not None:
            from agent_env.artifact import SkillArtifact
            artifact = SkillArtifact.get(skill.skill_artifact_id, skill.skill_artifact_version)
            return await A2AAgent.register_skill(
                deployed,
                name=artifact.skill_name,
                description=artifact.description,
                object_url=artifact.skill_object_url,
            )
        if skill.object_url is not None:
            return await A2AAgent.register_skill(
                deployed, name=skill.name, description=skill.description, object_url=skill.object_url
            )
        return await A2AAgent.register_skill(
            deployed, name=skill.name, description=skill.description, skill_md=skill.to_skill_md()
        )

    @staticmethod
    async def register_skill(
        deployed: "DeployedA2AAgent",
        *,
        name: str,
        description: str,
        skill_md: str | None = None,
        object_url: str | None = None,
    ) -> dict:
        """Send one skill, as SKILL.md text inline or the objects under ``object_url`` as a
        bundle of read grants; returns the agent's answer."""
        add_method, add_path = A2AAgent.operation(A2AAgent._skill_extension(deployed), "add")
        store = transfer_store(
            get_config().get_object_store(), deployed.a2a_url, deployed.agent_card, sandbox_type=deployed.sandbox_type
        )
        call = await asyncio.to_thread(
            skill_add_call,
            add_method,
            store,
            name=name,
            description=description,
            skill_md=skill_md,
            object_url=object_url,
            sandbox_type=deployed.sandbox_type,
        )
        return await invoke_transfer(
            deployed.a2a_url + add_path,
            call,
            verb="POST",
            operation=f"skill add ({name})",
            timeout=TRANSFER_TIMEOUT_SECONDS,
            store=store,
        )

    @staticmethod
    def _skill_extension(deployed: "DeployedA2AAgent") -> dict:
        skill_ext = A2AAgent.find_extension(deployed.agent_card, A2AAgent.EXT_SKILL_CONFIG)
        if not skill_ext:
            raise RuntimeError(
                f"Deployed agent {deployed.instance_id or deployed.agent_id} does not advertise {A2AAgent.EXT_SKILL_CONFIG}"
            )
        return skill_ext

    @staticmethod
    async def install_cli(deployed: "DeployedA2AAgent", cli_artifact: "CliArtifact", gateway_url: str) -> str:
        """Install a CliArtifact bundle into the agent's filesystem at /opt/cli/<command>/.
        Writes a sidecar bin/.env with the gateway URL and chmod +x's the entrypoint.
        Idempotent. Returns the in-process path to the executable entrypoint."""
        from agent_env.providers.sandbox_providers.sandbox import VmSandbox
        from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_agent_sandbox_provider
        provider = build_sandbox_provider(deployed.sandbox_type) if deployed.sandbox_type else get_agent_sandbox_provider()
        sandbox = await provider.get_sandbox(deployed.sandbox_id)
        target_root = f"/opt/cli/{cli_artifact.command_name}"
        entrypoint_path = f"{target_root}/{cli_artifact.entrypoint}"
        env_dir = posixpath.dirname(entrypoint_path)

        # write .env first — write_file_from_text creates the parent dir; write_file_from_object does not.
        await sandbox.write_file_from_text(f"AGENT_ENV_GATEWAY_URL={gateway_url}\n", f"{env_dir}/.env")

        for rel_path, file_artifact in cli_artifact.get_cli_files().get_file_artifacts().items():
            await sandbox.write_file_from_object(file_artifact.object_url, f"{target_root}/{rel_path}")

        if isinstance(sandbox, VmSandbox):
            await sandbox.exec_script(f"docker exec -u 0 {shlex.quote(sandbox.container_name)} chmod +x {entrypoint_path}")
        else:
            await sandbox.exec("chmod", "+x", entrypoint_path)

        logger.info(f"Installed CliArtifact '{cli_artifact.id}' v{cli_artifact.version} at {entrypoint_path}")
        return entrypoint_path

    @staticmethod
    async def load_file_artifact_universe(
        deployed: "DeployedA2AAgent",
        universe: "FileArtifactUniverse",
        destination_path: str,
    ) -> dict[str, str]:
        """Stage every FileArtifact in `universe` into the agent sandbox under destination_path."""
        from agent_env.providers.sandbox_providers.sandbox import stage_files_into_container
        from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_agent_sandbox_provider

        destination = destination_path.rstrip("/") or "/"
        provider = build_sandbox_provider(deployed.sandbox_type) if deployed.sandbox_type else get_agent_sandbox_provider()
        sandbox = await provider.get_sandbox(deployed.sandbox_id)

        file_artifacts = universe.get_file_artifacts()
        if not file_artifacts:
            logger.warning(
                f"FileArtifactUniverse '{universe.id}' v{universe.version} has no files; nothing to load into agent"
            )
            return {}

        logger.info(
            f"Loading FileArtifactUniverse '{universe.id}' v{universe.version} "
            f"({len(file_artifacts)} file(s)) into agent at {destination}"
        )
        return await stage_files_into_container(sandbox, file_artifacts, destination)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        docker_image_artifact: DockerImageArtifact,
        default_env_vars: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.id = id
        self.version = version
        self.docker_image_artifact = docker_image_artifact
        self.default_env_vars = default_env_vars or {}
        self.metadata = metadata or {}
        self._sandbox: Sandbox | None = None
        self._owns_sandbox: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "version": self.version,
            "docker_image_artifact": {
                "id": self.docker_image_artifact.id,
                "version": self.docker_image_artifact.version,
                "type": self.docker_image_artifact.type,
            },
            "default_env_vars": self.default_env_vars,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> A2AAgent:
        from agent_env.artifact import Artifact
        artifact_ref = data["docker_image_artifact"]
        docker_image_artifact = Artifact.get(artifact_ref["id"], version=artifact_ref["version"])
        metadata = data.get("metadata", {})
        # Migrate top-level fields from older documents into metadata
        for key in ("min_disk_size_gb", "default_model"):
            if key in data and data[key] is not None and key not in metadata:
                metadata[key] = data[key]
        return cls(
            id=data["id"], version=data.get("version"),
            docker_image_artifact=docker_image_artifact,
            default_env_vars=data.get("default_env_vars", {}),
            metadata=metadata,
        )

    @classmethod
    def from_toml(cls, data: dict[str, Any], ctx: AuthoringContext) -> A2AAgent:
        """Write the agent authored as ``data`` (its agent.toml, with ``image`` resolved to an image
        artifact's id) under ``ctx.id`` and return it. The card isn't authored: the image serves it."""
        fields = cls.accept_toml(data, ctx)
        return cls.put(id=ctx.id, docker_image_artifact=ctx.artifact(fields["image"], DockerImageArtifact),
                       default_env_vars=fields.get("default_env_vars"), metadata=fields.get("metadata"))

    @classmethod
    def accept_toml(cls, data: dict[str, Any], ctx: AuthoringContext) -> dict[str, Any]:
        """The keys of ``data``, an agent.toml, that an agent takes, each checked the way ``from_toml``
        needs it. Raises BundleError listing every problem."""
        fields, problems = ctx.accepted(data, **cls.toml_keys)
        values = [f"default_env_vars.{name} must be a string, not {value!r}"
                  for name, value in fields.get("default_env_vars", {}).items() if not isinstance(value, str)]
        for name, value in fields.get("metadata", {}).items():
            if name not in cls.toml_metadata:
                values.append(f"unknown [metadata] key {name!r}; an agent's [metadata] takes "
                              f"{' and '.join(cls.toml_metadata)}")
                continue
            kinds, described = cls.toml_metadata[name]
            if isinstance(value, bool) or not isinstance(value, kinds):
                values.append(f"metadata.{name} must be {described}, not {value!r}")
        problems += [ctx.config_problem(problem) for problem in values]
        if problems:
            ctx.refuse(problems)
        return fields

    @classmethod
    def get(cls, id: str, version: Optional[int] = None) -> A2AAgent:
        from .store import get_a2a_agent_store
        return get_a2a_agent_store().get(id, version)

    @classmethod
    def put(cls, **kwargs) -> A2AAgent:
        from .store import get_a2a_agent_store
        kwargs.setdefault("version", None)
        instance = cls(**kwargs)
        return get_a2a_agent_store().put_document(instance)

    @classmethod
    def query(cls) -> A2AAgentQuery:
        from .store import A2AAgentQuery, get_a2a_agent_store
        return A2AAgentQuery(get_a2a_agent_store())

    VALIDATION_ENV_ID: ClassVar[str] = "multi-caf64d"

    def update_metadata(self, new_metadata: dict[str, Any]) -> None:
        """Replace this agent's metadata in the database using compare-and-swap."""
        from .store import get_a2a_agent_store
        if self.version is None:
            raise ValueError("Cannot update metadata on an unsaved agent (version is None)")
        self.metadata = get_a2a_agent_store().update_metadata(self.id, self.version, self.metadata, new_metadata)

    async def validate(self, on_progress: Callable[[str], None] | None = None, litellm_api_key: str | None = None) -> dict:
        from agent_env.a2a_agent.validator import A2AAgentValidator
        return await A2AAgentValidator.validate(self, on_progress=on_progress, litellm_api_key=litellm_api_key)

    async def deploy(
        self,
        ttl_seconds: int = 7200,
        env_vars: dict[str, str] | None = None,
        a2a_port: int = DEFAULT_A2A_PORT,
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float | None = None,
        sandbox_type: str | None = None,
        sandbox: Sandbox | None = None,
        enable_docker: bool = False,
        network_policy: NetworkPolicy | None = None,
        *,
        attribution: Optional[Attribution] = None,
    ) -> DeployedA2AAgent:
        from agent_env.providers.sandbox_providers.sandbox_provider import (
            SANDBOX_MODE_VM,
            build_sandbox_provider,
            get_agent_sandbox_provider,
        )

        attribution = dict(attribution or {})
        if disk_size_gb is None:
            disk_size_gb = float(self.metadata.get("min_disk_size_gb") or 10)

        config = get_config()
        resolved_env = dict(env_vars) if env_vars else {}
        if "LITELLM_API_KEY" not in resolved_env and "LITELLM_API_KEY" not in self.default_env_vars:
            resolved_env["LITELLM_API_KEY"] = config.get_litellm_api_key()
        if "LITELLM_BASE_URL" not in resolved_env and "LITELLM_BASE_URL" not in self.default_env_vars:
            resolved_env["LITELLM_BASE_URL"] = config.get_litellm_base_url()

        merged_env = await asyncio.to_thread(self._build_merged_env, resolved_env, a2a_port)
        image_name = self.docker_image_artifact.image_name

        try:
            if sandbox is None:
                provider = build_sandbox_provider(sandbox_type) if sandbox_type else get_agent_sandbox_provider()
                logger.info(f"Provisioning sandbox for A2A agent '{self.id}' via {type(provider).__name__}...")
                self._sandbox = await provider.create_sandbox(
                    image_name=image_name, port=a2a_port, env=merged_env,
                    cpu=cpu, memory=memory, disk_size_gb=disk_size_gb, timeout=ttl_seconds,
                    attribution=attribution,
                    network_policy=network_policy,
                )
                self._owns_sandbox = True
                logger.info(f"Sandbox created: {self._sandbox.sandbox_id} (mode={self._sandbox.mode})")
            else:
                if network_policy is not None and network_policy.restricts_egress:
                    raise NetworkPolicyUnsupportedError(
                        f"network_policy={network_policy.mode.value!r} cannot be applied to a pre-existing "
                        f"sandbox {sandbox.sandbox_id!r}; deploy the agent on its own sandbox instead"
                    )
                self._sandbox = sandbox
                self._owns_sandbox = False
                logger.info(f"Using pre-existing sandbox {self._sandbox.sandbox_id} (mode={self._sandbox.mode}) for A2A agent '{self.id}'")

            if enable_docker and self._sandbox.mode != SANDBOX_MODE_VM:
                raise ValueError(
                    f"enable_docker=True requires a VM sandbox, but the agent sandbox mode "
                    f"is {self._sandbox.mode!r} (Docker-in-Docker is VM-only)"
                )

            if self._sandbox.mode == SANDBOX_MODE_VM:
                logger.info("Loading Docker image into VM...")
                await self._sandbox.load_docker_images([self.docker_image_artifact])
                logger.info("Starting A2A agent container...")
                await self._run_container(image_name, a2a_port, merged_env, enable_docker=enable_docker)

            a2a_url = self._sandbox.tunnel_urls.get(a2a_port)
            if not a2a_url:
                raise RuntimeError(f"No tunnel URL for port {a2a_port}")

            logger.info("Waiting for agent card...")
            agent_card = await self._wait_for_agent_card(a2a_url)

            deployed = DeployedA2AAgent(
                agent_id=self.id, agent_version=self.version,
                a2a_url=a2a_url, sandbox_id=self._sandbox.sandbox_id,
                agent_card=agent_card, sandbox_type=self._sandbox.type,
                network_policy=self._sandbox.network_policy.to_dict() if self._sandbox.network_policy else None,
            )
            from agent_env.a2a_agent.store import get_a2a_agent_instance_store
            deployed = get_a2a_agent_instance_store().create_instance(deployed, ttl_seconds)
            logger.info(f"A2A agent '{self.id}' deployed at {a2a_url} (instance={deployed.instance_id})")
            return deployed
        except BaseException:  # a cancelled deploy closes its sandbox too
            await self.close()
            raise

    def _build_merged_env(self, resolved_env: dict[str, str], a2a_port: int) -> dict[str, str]:
        merged_env = dict(self.default_env_vars)
        merged_env.update(resolved_env)
        merged_env["A2A_PORT"] = str(a2a_port)
        shared = get_config().get_object_store().shared_credentials_env()
        if shared.keys().isdisjoint(merged_env):
            merged_env.update(shared)

        merged_env.update(all_sandbox_container_env())
        rewrites = all_sandbox_url_rewrites()
        if rewrites:
            merged_env["SANDBOX_URL_REWRITES"] = json.dumps(rewrites, sort_keys=True)
        return merged_env

    async def close(self) -> None:
        if self._sandbox is not None and self._owns_sandbox:
            try:
                await self._sandbox.terminate()
            except Exception as e:
                logger.warning(f"Error terminating sandbox: {e}")
        self._sandbox = None

    async def _run_container(self, image_name: str, a2a_port: int, merged_env: dict[str, str], enable_docker: bool = False) -> None:
        # On a local VM sandbox the container runs on this host, so it is given the local CA as the provider's are.
        trust_dir = await asyncio.to_thread(local_grant_trust) if isinstance(self._sandbox, LocalSandbox) else None
        agent_env = dict(merged_env) if trust_dir is None else {**LOCAL_TRUST_ENV, **merged_env}
        setup_script = ""
        network_flag = "" if trust_dir is None or not LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS else (
            f"{LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS} \\\n    "
        )
        if enable_docker:
            logger.info("enable_docker: starting rootless Docker-in-Docker sidecar for the agent (no host socket)")
            setup_script = self._dind_setup_script()
            network_flag += f"--network {_DIND_NETWORK} \\\n    "
            agent_env["DOCKER_HOST"] = f"tcp://{_DIND_CONTAINER}:{_DIND_PORT}"

        env_flags = []
        for key, value in agent_env.items():
            escaped_value = value.replace("'", "'\\''")
            env_flags.append(f"-e {key}='{escaped_value}'")
        env_str = " \\\n    ".join(env_flags)
        publish = " ".join(f"-p {spec}" for spec in port_bindings(self._sandbox.host_ips, a2a_port, a2a_port))

        run_script = f"""#!/bin/bash
set -e
{setup_script}docker {"run -d" if trust_dir is None else "create"} \\
    --name {self._sandbox.container_name} \\
    {publish} \\
    {network_flag}{env_str} \\
    {image_name} > /dev/null
{"sleep 2" if trust_dir is None else ""}
"""
        await self._sandbox.exec_script(run_script)
        if trust_dir is not None:
            await start_trusting(self._sandbox, self._sandbox.container_name, trust_dir)
            await asyncio.sleep(2)

    def _dind_setup_script(self) -> str:
        # Constants become shell vars so the body stays a raw string (no f-string
        # brace escaping). TLS off is safe: the tcp endpoint is only reachable on
        # the VM-local bridge.
        env = (
            f"DIND_IMAGE={_DIND_IMAGE}\n"
            f"DIND_CONTAINER={_DIND_CONTAINER}\n"
            f"DIND_NETWORK={_DIND_NETWORK}\n"
            f"DIND_PORT={_DIND_PORT}\n"
        )
        return env + r"""docker network create "$DIND_NETWORK" >/dev/null 2>&1 || true
docker rm -f "$DIND_CONTAINER" >/dev/null 2>&1 || true
docker run -d \
    --name "$DIND_CONTAINER" \
    --privileged \
    --network "$DIND_NETWORK" \
    -e DOCKER_TLS_CERTDIR="" \
    "$DIND_IMAGE" > /dev/null
echo "waiting for rootless docker daemon..."
for i in $(seq 1 60); do
    if docker exec "$DIND_CONTAINER" docker -H "tcp://localhost:$DIND_PORT" info >/dev/null 2>&1; then
        echo "rootless docker daemon ready"
        break
    fi
    if [ "$i" = "60" ]; then
        echo "rootless docker daemon did not become ready" >&2
        docker logs --tail 60 "$DIND_CONTAINER" >&2 2>&1 || true
        exit 1
    fi
    sleep 2
done
"""

    async def _wait_for_agent_card(self, base_url: str) -> dict:
        import httpx
        card_url = f"{base_url}/.well-known/agent.json"
        async with httpx.AsyncClient(timeout=2.0) as client:
            for i in range(AGENT_CARD_TIMEOUT):
                try:
                    resp = await client.get(card_url)
                    if resp.status_code == 200 and resp.text.strip():
                        card = resp.json()
                        logger.info(f"Agent card received: name={card.get('name')}")
                        return card
                except Exception:
                    pass
                if i > 0 and i % 10 == 0:
                    logger.info(f"Waiting for agent card... ({i}s)")
                await asyncio.sleep(1)
        raise RuntimeError(
            f"A2A agent '{self.id}' did not serve /.well-known/agent.json "
            f"within {AGENT_CARD_TIMEOUT}s"
        )
