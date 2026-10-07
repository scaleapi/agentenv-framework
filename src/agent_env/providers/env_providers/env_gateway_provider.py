"""The gateway environment provider: MCP servers and websites behind one agent-env gateway, on a VM or on Modal containers."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from agent_env.attribution import Attribution
from agent_env.env.env import DeployedGatewayEnv, _mcp_url
from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT, GatewayMode
from agent_env.env.gateway.constants import DEFAULT_MCP_SERVER_NAME, WELL_KNOWN_PATH, random_mcp_server_name
from agent_env.env.envs.service_db import (
    DATABASE_SERVICE_NAME,
    DB_MCP_CONTAINER_PORT,
    DB_MCP_PORT,
    DB_MCP_SERVICE_NAME,
    DB_WEB_PORT,
    PGWEB_SERVICE_NAME,
)
from agent_env.providers.env_providers.constants import (
    AGENT_ENV_WEBSITE_BACKEND_PORT,
    AGENT_ENV_WEBSITE_BACKEND_SUFFIX,
    AGENT_ENV_WEBSITE_FRONTEND_SUFFIX,
    DOCKER_COMPOSE_PATH,
    GATEWAY_APP_DIR,
    GATEWAY_SERVICE_NAME,
)
from agent_env.providers.env_providers.env_provider import _SandboxEnvironmentProvider, _size_kwargs, _tool_names
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import Sandbox, VmSandbox, port_bindings
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER, SandboxProvider

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent_env.artifact import DockerImageArtifact
    from agent_env.env.env import DeployedEnv, Env
    from agent_env.providers.env_state import DatabaseStateProvider, EnvStateInstance, LocalPostgresStoreSpec

logger = logging.getLogger(__name__)

_TRAJECTORY_READ_ATTEMPTS = 3

# MCP server health-check budget. Some servers wait on an upstream before they bind; a server
# that is up passes its first probe, so the long start period only delays failing a broken one.
MCP_HC_RETRIES = 30

MCP_HC_START_PERIOD = "60s"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server in a gateway deployment."""

    image: str
    environment_name: str
    extra_env_vars: dict[str, str] | None = None


@dataclass
class DeployedGateway:
    """Result of a gateway deployment."""

    gateway_url: str
    mcp_url: str
    db_web_url: str
    db_mcp_url: str | None = None
    website_frontend_urls: dict[str, str] | None = None
    mcp_server_name: str | None = None
    # Ids of the EnvStateInstance record(s) this deploy acquired; the caller records them
    # on DeployedGatewayEnv.env_state_instance_ids. One entry today (one backend per deploy).
    env_state_instance_ids: list[str] = field(default_factory=list)
    # The env card the readiness probe read, and when (ISO 8601, UTC).
    environment_card: dict | None = None
    environment_card_read_at_utc: str | None = None


@dataclass
class WebsiteConfig:
    """Configuration for a website (frontend + backend) in a gateway deployment."""

    backend_image: str
    frontend_image: str
    environment_name: str


@dataclass
class SidecarConfig:
    """A plain co-deployed container on the gateway that is *not* an MCP server.

    Rendered into the gateway's docker-compose as its own service, with its env
    (secrets included) injected via the compose ``environment:`` block — so nothing
    lands on a ``docker run`` command line / the host process table — and its port
    published on the VM host so callers can reach it over the sandbox tunnel.

    ``image_artifact`` is side-loaded onto the VM from the object store exactly like the
    MCP server images (``load_docker_images``): no external-registry pull happens at
    ``docker compose up`` time, so the deploy has no registry dependency or ``docker login``.
    """

    environment_name: str
    image_artifact: "DockerImageArtifact"
    container_port: int
    host_port: int
    env: dict[str, str] | None = None
    restart: str = "unless-stopped"

    @property
    def image(self) -> str:
        """The local image tag compose references — the same tag ``load_docker_images``
        restores on the VM (the artifact's agent-env image name)."""
        return self.image_artifact.image_name


class EnvironmentGatewayProvider(_SandboxEnvironmentProvider):
    """Provider for creating gateway deployments."""

    type = "gateway"
    record_class = DeployedGatewayEnv

    def __init__(self):
        super().__init__()
        self._sandbox: Sandbox | None = None
        self._db_sandbox: Sandbox | None = None
        self._pgweb_sandbox: Sandbox | None = None
        self._db_mcp_sandbox: Sandbox | None = None
        # Set by _deploy_gateway (external instance) or _build_local_store (local) on deploy, and by _reattach
        self._state_provider: "DatabaseStateProvider | None" = None
        self._state_instance: "EnvStateInstance | None" = None

    @property
    def sandbox(self) -> Sandbox | None:
        return self._sandbox

    async def close(self) -> None:
        if self._state_provider is not None and self._state_instance is not None:
            try:
                await self._state_provider.teardown(self._state_instance)
            except Exception as e:
                logger.warning(f"Failed to tear down state backend {type(self._state_provider).__name__}: {e}")
            finally:
                self._state_provider = None
                self._state_instance = None
        await super().close()
        self._db_sandbox = None
        self._pgweb_sandbox = None
        self._db_mcp_sandbox = None

    async def install_changelog_triggers(self, environment_name: str) -> None:
        """Install audit triggers on the given service schema.

        Thin shim over the state provider (which owns the changelog SQL). The provider +
        instance are set by ``_build_local_store``/``_deploy_gateway`` (deploy) or ``_reattach``
        (reattach); when neither ran (no state store wired — e.g. a standalone env reattached without one) this is
        a no-op, skipping rather than guessing a local backend.
        """
        from agent_env.providers.env_state import DatabaseStateProvider

        state_provider, state_instance = self._state_provider, self._state_instance
        if state_provider is None or state_instance is None:
            logger.debug("install_changelog_triggers: no state provider/instance set; skipping")
            return
        if not isinstance(state_provider, DatabaseStateProvider):
            return

        if self._db_sandbox is not None:
            async def store_exec(argv: list[str]) -> tuple[int, str, str]:
                return await self._db_sandbox.exec_with_output(*argv)
        else:
            store_name = state_provider.env_state_docker_service_name()

            async def store_exec(argv: list[str]) -> tuple[int, str, str]:
                return await self._sandbox.exec_with_output(
                    "sudo", "docker", "compose", "-f", DOCKER_COMPOSE_PATH,
                    "exec", "-T", store_name, *argv,
                )

        try:
            await state_provider.install_changelog_triggers(
                environment_name, instance=state_instance, store_exec=store_exec,
            )
            logger.info(f"Changelog triggers installed for schema '{environment_name}'")
        except Exception as e:
            logger.warning(f"Failed to install changelog triggers for schema '{environment_name}': {type(e).__name__}: {e}")

    def _all_environments(self, mcp_servers, website_configs) -> list[str]:
        """The DB-client services (one schema each) for a deploy: MCP servers + website backends,
        deduped. This is the authoritative list — it reflects any servers the gateway auto-adds (e.g.
        website_browser) — and drives both the rendered compose and the state provider's ``prepare``."""
        return list(dict.fromkeys(
            [s.environment_name for s in mcp_servers]
            + [wc.environment_name for wc in (website_configs or [])]
        ))

    def create_docker_compose(
        self,
        mcp_servers: list[MCPServerConfig],
        gateway_image: str = "env-gateway",
        gateway_port: int = AGENT_ENV_GATEWAY_MCP_PORT,
        website_configs: list[WebsiteConfig] | None = None,
        expose_gateway_port: bool = True,
        gateway_mode: GatewayMode = GatewayMode.PERFORMANCE,
        sidecars: list[SidecarConfig] | None = None,
        *,
        state_provider: "DatabaseStateProvider",
        state_instance: "EnvStateInstance",
        host_port: Optional[Callable[[int], int]] = None,
        host_ips: tuple[str, ...] = (),
        mcp_server_name: str | None = None,
    ) -> str:
        """Generate docker-compose.yml content for a gateway deployment onto a VM/laptop/arbitrary machine.

        Supports MCP servers, websites, or both. Website containers (backend + frontend) are added
        to the same docker network. The gateway receives WEBSITE_URLS env var for discovery.

        Args:
            mcp_servers: List of MCP server configurations for internal servers.
            gateway_image: Docker image for the gateway.
            gateway_port: Port to expose for the gateway.
            website_configs: Optional list of website configurations (frontend + backend pairs).

        Returns:
            docker-compose.yml content as a string.

        Example output for mcp_servers=[MCPServerConfig("mcp-slack", "slack")]
        and website_configs=[WebsiteConfig("slack-web-be", "slack-web-fe", "slack")]
        (using AGENT_ENV_GATEWAY_MCP_PORT=18765):

            services:
              servicedb:
                image: public.ecr.aws/docker/library/postgres:16-alpine
                environment:
                  - POSTGRES_USER=agentenv
                  - POSTGRES_PASSWORD=agentenv
                  - POSTGRES_DB=agentenv
                healthcheck:
                  test: ["CMD-SHELL", "pg_isready -U agentenv"]
                  interval: 5s
                  timeout: 5s
                  retries: 5
                volumes:
                  - ./init-schemas.sql:/docker-entrypoint-initdb.d/init-schemas.sql
                networks:
                  - env-network

              slack:
                image: mcp-slack
                depends_on:
                  servicedb:
                    condition: service_healthy
                environment:
                  - MCP_HOST=0.0.0.0
                  - MCP_PORT=<gateway_port>
                  - SERVICE_NAME=slack
                  - ENVIRONMENT_NAME=slack
                  - DATABASE_URL=postgresql://agentenv:agentenv@servicedb:5432/agentenv?options=-c%20search_path%3D%22slack%22,public
                networks:
                  - env-network

              gateway:
                image: env-gateway
                depends_on:
                  - slack
                environment:
                  - MCP_HOST=0.0.0.0
                  - MCP_PORT=<gateway_port>
                  - MCP_SERVER_NAME=crm
                  - INTERNAL_MCP_SERVERS=slack
                  - INTERNAL_MCP_PORT=<gateway_port>
                  - WEBSITE_URLS=http://slack-website-frontend:80
                  - REST_PROXY_URLS=slack=http://slack-website-backend:8000
                ports:
                  - "<gateway_port>:<gateway_port>"
                networks:
                  - env-network

              slack-website-backend:
                image: slack-web-be
                depends_on:
                  servicedb:
                    condition: service_healthy
                environment:
                  - DATABASE_URL=postgresql://agentenv:agentenv@servicedb:5432/agentenv?options=-c%20search_path%3D%22slack%22,public
                  - SERVICE_NAME=slack
                  - ENVIRONMENT_NAME=slack
                  - PYTHONUNBUFFERED=1
                healthcheck:
                  test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
                  interval: 10s
                  timeout: 5s
                  start_period: 30s
                  retries: 3
                networks:
                  - env-network

              slack-website-frontend:
                image: slack-web-fe
                depends_on:
                  slack-website-backend:
                    condition: service_healthy
                networks:
                  - env-network

            networks:
              env-network:
                driver: bridge
        """
        lines = ["services:"]

        # MCP servers and website backends are both DB clients (one schema each).
        environment_names = self._all_environments(mcp_servers, website_configs)

        lines.extend(
            state_provider.render_compose_containers(
                environment_names, instance=state_instance, host_port=host_port, host_ips=host_ips
            )
        )
        store_dep_lines = state_provider.docker_service_dependency()

        # Add internal server services
        website_configs = website_configs or []
        website_backend_servers = {
            wc.environment_name: f"{wc.environment_name}-{AGENT_ENV_WEBSITE_BACKEND_SUFFIX}"
            for wc in website_configs
        }
        for server in mcp_servers:
            db_url = state_provider.url_for_environment(server.environment_name, instance=state_instance)
            lines.extend([
                f"  {server.environment_name}:",
                f"    image: {server.image}",
            ])
            dep_lines = list(store_dep_lines)
            if server.environment_name in website_backend_servers:
                dep_lines += [
                    f"      {website_backend_servers[server.environment_name]}:",
                    "        condition: service_healthy",
                ]
            if dep_lines:
                lines.append("    depends_on:")
                lines.extend(dep_lines)
            lines.extend([
                "    environment:",
                "      - MCP_HOST=0.0.0.0",
                f"      - MCP_PORT={gateway_port}",
                # Both names are injected during the service_name -> environment_name
                # migration. The SDK resolves identity from ENVIRONMENT_NAME
                # (then the card's own name) and no longer consults SERVICE_NAME; it stays
                # only for images that read it themselves, and a later pass drops it.
                f"      - SERVICE_NAME={server.environment_name}",
                f"      - ENVIRONMENT_NAME={server.environment_name}",
                f"      - DATABASE_URL={db_url}",
            ])
            if server.extra_env_vars:
                for key, value in server.extra_env_vars.items():
                    lines.append(f"      - {key}={value}")
            lines.extend([
                "    healthcheck:",
                f'      test: ["CMD-SHELL", "python3 -c \\"import socket; s=socket.create_connection((\'localhost\',{gateway_port}),2); s.close()\\""]',
                "      interval: 5s",
                "      timeout: 5s",
                f"      retries: {MCP_HC_RETRIES}",
                f"      start_period: {MCP_HC_START_PERIOD}",
                "    networks:",
                "      - env-network",
                "",
            ])

        # Add gateway service
        mcp_server_names = [s.environment_name for s in mcp_servers]
        lines.extend([
            f"  {GATEWAY_SERVICE_NAME}:",
            f"    image: {gateway_image}",
        ])
        dep_lines = list(store_dep_lines)
        for name in mcp_server_names:
            dep_lines += [f"      {name}:", "        condition: service_healthy"]
        for backend_name in website_backend_servers.values():
            dep_lines += [f"      {backend_name}:", "        condition: service_healthy"]
        if dep_lines:
            lines.append("    depends_on:")
            lines.extend(dep_lines)
        server_name = (mcp_server_name or DEFAULT_MCP_SERVER_NAME).replace("$", "$$")  # compose would interpolate a bare $
        gateway_env = [
            "      - MCP_HOST=0.0.0.0",
            f"      - MCP_PORT={gateway_port}",
            f"      - MCP_SERVER_NAME={server_name}",
            f"      - INTERNAL_MCP_SERVERS={','.join(mcp_server_names)}",
            f"      - INTERNAL_MCP_PORT={gateway_port}",
            "      - GATEWAY_TRAJECTORY_FILE=/var/log/agentenv/trajectory.jsonl",
        ]
        # Build REST_PROXY_URLS: mcp-{name} for MCP servers + {name} for website backends
        rest_proxy_urls = []
        for server in mcp_servers:
            rest_proxy_urls.append(
                f"mcp-{server.environment_name}=http://{server.environment_name}:{gateway_port}"
            )
        if website_configs:
            website_urls = ",".join(
                f"{wc.environment_name}=http://{wc.environment_name}-{AGENT_ENV_WEBSITE_FRONTEND_SUFFIX}:80"
                for wc in website_configs
            )
            gateway_env.append(f"      - WEBSITE_URLS={website_urls}")
            for wc in website_configs:
                rest_proxy_urls.append(
                    f"{wc.environment_name}=http://{wc.environment_name}-{AGENT_ENV_WEBSITE_BACKEND_SUFFIX}:{AGENT_ENV_WEBSITE_BACKEND_PORT}"
                )
        if rest_proxy_urls:
            gateway_env.append(f"      - REST_PROXY_URLS={','.join(rest_proxy_urls)}")
        gateway_env.append(f"      - GATEWAY_MODE={gateway_mode.value}")
        gateway_env.append(f"      - SERVICE_DB_URL={state_provider.base_url(state_instance)}")
        publish = host_port or (lambda port: port)
        exposed_gateway_ports = []
        if expose_gateway_port:
            exposed_gateway_ports.extend([
                "    ports:",
                *(f'      - "{spec}"' for spec in port_bindings(host_ips, publish(gateway_port), gateway_port)),
            ])
        lines.extend([
            "    environment:",
            *gateway_env,
            *exposed_gateway_ports,
            "    restart: on-failure",
            "    healthcheck:",
            f'      test: ["CMD-SHELL", "python3 -c \\"import socket; s=socket.create_connection((\'localhost\',{gateway_port}),2); s.close()\\""]',
            "      interval: 15s",
            "      timeout: 5s",
            "      retries: 10",
            "      start_period: 15s",
            "    networks:",
            "      - env-network",
            "",
        ])

        # Add website services
        for wc in website_configs:
            backend_name = f"{wc.environment_name}-{AGENT_ENV_WEBSITE_BACKEND_SUFFIX}"
            frontend_name = f"{wc.environment_name}-{AGENT_ENV_WEBSITE_FRONTEND_SUFFIX}"
            db_url = state_provider.url_for_environment(wc.environment_name, instance=state_instance)
            lines.extend([
                f"  {backend_name}:",
                f"    image: {wc.backend_image}",
            ])
            # depends_on the backend's readiness service.
            lines.append("    depends_on:")
            lines.extend(store_dep_lines)
            lines.extend([
                "    environment:",
                f"      - DATABASE_URL={db_url}",
                f"      - SERVICE_NAME={wc.environment_name}",
                f"      - ENVIRONMENT_NAME={wc.environment_name}",
                f"      - PORT={AGENT_ENV_WEBSITE_BACKEND_PORT}",
                "      - PYTHONUNBUFFERED=1",
                "    healthcheck:",
                f'      test: ["CMD", "curl", "-f", "http://localhost:{AGENT_ENV_WEBSITE_BACKEND_PORT}/api/health"]',
                "      interval: 10s",
                "      timeout: 5s",
                "      start_period: 30s",
                "      retries: 3",
                "    networks:",
                "      - env-network",
                "",
                f"  {frontend_name}:",
                f"    image: {wc.frontend_image}",
                "    depends_on:",
                f"      {backend_name}:",
                "        condition: service_healthy",
                "    networks:",
                "      - env-network",
                "",
            ])

        # Add sidecar services.
        # Secrets go through the compose environment block, never a docker-run command line.
        for sc in sidecars or []:
            lines.extend([
                f"  {sc.environment_name}:",
                f"    image: {sc.image}",
            ])
            if sc.env:
                lines.append("    environment:")
                for key, value in sc.env.items():
                    lines.append(f"      - {key}={value}")
            lines.extend([
                "    ports:",
                *(f'      - "{spec}"' for spec in port_bindings(host_ips, publish(sc.host_port), sc.container_port)),
                f"    restart: {sc.restart}",
                "    networks:",
                "      - env-network",
                "",
            ])

        # Add networks
        lines.extend([
            "networks:",
            "  env-network:",
            "    driver: bridge",
        ])
        return "\n".join(lines)

    async def deploy(
        self,
        env: Env,
        sandbox_provider: SandboxProvider,
        *,
        gateway_mode: GatewayMode = GatewayMode.PERFORMANCE,
        env_state_type: str | None = None,
        env_state_instance_id: str | None = None,
        ttl_seconds: int = 10800,
        disk_size_gb: float = 10,
        cpu: float | None = None,
        memory_mb: int | None = None,
        attribution: Optional[Attribution] = None,
    ) -> DeployedGatewayEnv:
        """A gateway in front of the env, with the state store it acquires (local Postgres by default); returns the record.

        A MultiEnv's gateway fronts its MCP servers and websites, a WebsiteEnv's its website, and any other env's its one MCP server.
        """
        from agent_env.config import get_config
        from agent_env.env.env import Env
        from agent_env.providers.env_state import LOCAL_POSTGRES_STATE_TYPE, acquire_state_for_deploy

        topology = _gateway_topology(env)
        state_instance = await acquire_state_for_deploy(
            env_state_type=env_state_type, ttl_seconds=ttl_seconds, name_hint=env.id, env_state_instance_id=env_state_instance_id,
        )
        if state_instance is None or state_instance.state_type == LOCAL_POSTGRES_STATE_TYPE:
            Env.get(get_config().default_service_db_env_id)  # the local store runs from it: check it before any sandbox
        deployed = await self._launch(
            sandbox_provider, env.id, state_instance=state_instance, mcp_servers=topology.mcp_servers,
            mcp_server_images=topology.mcp_server_images, website_configs=topology.website_configs,
            website_images=topology.website_images, mcp_server_name=topology.mcp_server_name, gateway_mode=gateway_mode,
            ttl_seconds=ttl_seconds, disk_size_gb=disk_size_gb, cpu=cpu, memory_mb=memory_mb, attribution=attribution,
        )
        return DeployedGatewayEnv(
            env_id=env.id,
            env_version=env.version,
            env_provider_type=self.type,
            gateway_url=deployed.gateway_url,
            db_web_url=deployed.db_web_url,
            sandbox_id=self._sandbox.sandbox_id, sandbox_type=self._sandbox.type,
            sandbox_ids=self._sandbox_ids(always_name_gateway=topology.names_gateway_on_vm),
            db_mcp_url=deployed.db_mcp_url,
            environment_card_url=f"{deployed.gateway_url}{WELL_KNOWN_PATH}",
            environment_card=deployed.environment_card,
            environment_card_read_at_utc=deployed.environment_card_read_at_utc,
            website_frontend_urls=deployed.website_frontend_urls,
            gateway_mode=gateway_mode.value,
            env_state_instance_ids=deployed.env_state_instance_ids,
        )

    async def create_gateway(
        self,
        sandbox_provider: SandboxProvider,
        mcp_servers: list[MCPServerConfig],
        mcp_server_images: list[DockerImageArtifact],
        gateway_port: int = AGENT_ENV_GATEWAY_MCP_PORT,
        website_configs: list[WebsiteConfig] | None = None,
        website_images: list[DockerImageArtifact] | None = None,
        gateway_mode: GatewayMode = GatewayMode.PERFORMANCE,
        ttl_seconds: int = 10800,
        disk_size_gb: float = 10,
        cpu: float | None = None,
        memory_mb: int | None = None,
        existing_sandbox: Sandbox | None = None,
        env_id: str | None = None,
        state_instance: "EnvStateInstance | None" = None,
        sidecars: list[SidecarConfig] | None = None,
        mcp_server_name: str | None = None,
        *,
        attribution: Optional[Attribution] = None,
    ) -> DeployedGateway:
        """Create and start a gateway from explicit topology, for a caller that builds its own record rather than calling ``deploy()``.

        ``state_instance`` is a pre-acquired *external* store handle (e.g. remote Postgres),
        provisioned upstream by the caller. ``None`` => local Postgres, which the gateway self-provisions
        (its store container's host doesn't exist until deploy). See ``_build_local_store``.

        ``existing_sandbox`` lets callers pre-allocate the gateway VM in parallel
        with other work and pass the handle in. VM-mode only — Modal container
        deploys manage their own sandbox lifecycle and reject this kwarg.

        ``mcp_server_name`` is the name the gateway presents on its card and as its MCP server; drawn as ``env`` +
        4 random digits when the env declares none; the record takes it from the gateway's card.
        """
        return await self._launch(
            sandbox_provider, env_id, state_instance=state_instance, mcp_servers=mcp_servers, mcp_server_images=mcp_server_images,
            gateway_port=gateway_port, website_configs=website_configs, website_images=website_images,
            gateway_mode=gateway_mode, ttl_seconds=ttl_seconds, disk_size_gb=disk_size_gb, cpu=cpu, memory_mb=memory_mb,
            existing_sandbox=existing_sandbox, sidecars=sidecars, mcp_server_name=mcp_server_name,
            attribution=attribution,
        )

    async def _launch(self, sandbox_provider: SandboxProvider, env_id: str | None, *, mcp_server_name: str | None = None, **gateway) -> DeployedGateway:
        """One gateway, deployed on the sandbox provider or each chained member in turn, then its tools probed."""
        attempt = functools.partial(
            self._deploy_gateway, **gateway,
            mcp_server_name=mcp_server_name or random_mcp_server_name(),  # drawn once, so every chained attempt presents the same name
        )
        result = await self._run(sandbox_provider, env_id, attempt)
        await _probe_tools(env_id, result)
        return result

    async def _deploy_via_vm(
        self,
        sandbox_provider: SandboxProvider,
        mcp_servers: list[MCPServerConfig],
        mcp_server_images: list[DockerImageArtifact],
        gateway_port: int,
        website_configs: list[WebsiteConfig] | None,
        website_images: list[DockerImageArtifact] | None,
        gateway_mode: GatewayMode,
        ttl_seconds: int,
        disk_size_gb: float,
        cpu: float | None = None,
        memory_mb: int | None = None,
        attribution: Optional[Attribution] = None,
        existing_sandbox: Sandbox | None = None,
        sidecars: list[SidecarConfig] | None = None,
        mcp_server_name: str | None = None,
    ) -> DeployedGateway:
        from agent_env.env.env import Env
        from agent_env.config import get_config

        website_configs = website_configs or []
        website_images = website_images or []
        sidecars = sidecars or []

        if existing_sandbox is not None:
            self._sandbox = existing_sandbox
        else:
            vm_kwargs: dict = {}
            if cpu is not None:
                vm_kwargs["cpu"] = cpu
            if memory_mb is not None:
                vm_kwargs["memory"] = memory_mb
            self._sandbox = await sandbox_provider.create_vm(
                exposed_ports=[gateway_port, DB_WEB_PORT, DB_MCP_PORT] + [sc.host_port for sc in sidecars],
                timeout=ttl_seconds,
                disk_size_gb=disk_size_gb,
                attribution=attribution,
                **vm_kwargs,
            )
        sandbox = self._sandbox

        config = get_config()
        gateway_env = Env.get(config.default_gateway_env_id)

        # Auto-include website_browser MCP server when websites are present
        if website_configs:
            website_browser_env = Env.get(config.default_website_browser_env_id)
            browser_server = MCPServerConfig(
                image=website_browser_env.docker_image_artifact.image_name,
                environment_name=website_browser_env.environment_name,
            )
            mcp_servers = list(mcp_servers) + [browser_server]
            mcp_server_images = list(mcp_server_images) + [website_browser_env.docker_image_artifact]

        # Local Postgres also preloads its servicedb image set (servicedb + pgweb + db-mcp); an
        # external store has no co-deployed containers, and image pull dominates VM deploy latency.
        all_images = [
            gateway_env.docker_image_artifact,
            *mcp_server_images,
            *website_images,
            *[sc.image_artifact for sc in sidecars],  # sidecars side-load like MCP images (no registry pull)
        ]
        if self._needs_local_postgres:
            all_images += self._state_provider.store_images_to_load()
        await sandbox.load_docker_images(all_images)

        # Compose declares the local_state service by name, so "standing up" is just naming the host;
        # the container is rendered into the compose and started on deploy.
        async def _stand_up(_spec):
            return DATABASE_SERVICE_NAME

        local_store_spec = (
            await self._build_local_store(
                mcp_servers, _stand_up, website_configs=website_configs, ttl_seconds=ttl_seconds
            )
            if self._needs_local_postgres else None
        )
        state_instance = self._state_instance

        # Stage 2 (prepare): create per-service schemas now that the final service list is known —
        # it includes the auto-added website_browser above. Local no-ops (its servicedb container
        # runs the init script declaratively); remote/external stores create schemas here.
        if self._state_provider is not None:
            await self._state_provider.prepare(
                self._all_environments(mcp_servers, website_configs), instance=state_instance
            )

        compose_content = self.create_docker_compose(
            mcp_servers=mcp_servers,
            gateway_image=gateway_env.docker_image_artifact.image_name,
            gateway_port=gateway_port,
            website_configs=website_configs,
            gateway_mode=gateway_mode,
            sidecars=sidecars,
            state_provider=self._state_provider,
            state_instance=state_instance,
            host_port=sandbox.host_port,
            host_ips=sandbox.host_ips,
            mcp_server_name=mcp_server_name,
        )
        logger.info(f"Generated docker-compose.yml:\n{_redact_compose_secrets(compose_content)}")

        # Write files to VM
        logger.info("Writing files to VM...")
        await sandbox.exec_script(f"mkdir -p {GATEWAY_APP_DIR}", max_retries=2)

        # Only local postgres needs its init script to be directly run by the gateway's servicedb
        # container; remote/external stores are already initialized upstream.
        if self._needs_local_postgres:
            init_script = local_store_spec.init_sql
            logger.info(f"Generated init-schemas.sql:\n{init_script}")
            init_script_path = f"{GATEWAY_APP_DIR}/init-schemas.sql"
            write_init_script = f'''cat > {init_script_path} << 'INIT_EOF'
{init_script}
INIT_EOF'''
            await sandbox.exec_script(write_init_script, max_retries=2)

        # Write docker-compose.yml
        write_compose_script = f'''cat > {DOCKER_COMPOSE_PATH} << 'COMPOSE_EOF'
{compose_content}
COMPOSE_EOF'''
        await sandbox.exec_script(write_compose_script, max_retries=2)

        # Start MCP services with docker compose
        logger.info("Starting MCP services...")
        await sandbox.exec_script(f"cd {GATEWAY_APP_DIR} && docker compose up -d", max_retries=2)

        # Wait for containers to start
        await asyncio.sleep(10)

        # Verify containers are running
        logger.info("Verifying containers...")
        exit_code, stdout, stderr = await sandbox.exec_with_output( "sudo", "docker", "ps", "-a")
        logger.info(f"  docker ps -a:\n{stdout}")
        if exit_code != 0:
            raise RuntimeError(f"docker ps failed: {stderr}")

        # Check gateway container is running
        exit_code, stdout, stderr = await sandbox.exec_with_output( "sudo", "docker", "ps")
        if GATEWAY_SERVICE_NAME not in stdout:
            raise RuntimeError(f"Gateway container not running. docker ps: {stdout}")

        # Wait for gateway to be fully ready (needs time to discover internal servers)
        # Host-side port: exec_with_output runs inside the VM on per-VM backends but on
        # the host for LocalSandbox, where the gateway is published elsewhere.
        gateway_ready = await self._wait_for_gateway(sandbox, sandbox.host_port(gateway_port))

        # Print gateway logs for debugging
        logger.info("Gateway logs:")
        gateway_container_id = await self._get_container_id(sandbox, GATEWAY_SERVICE_NAME)
        if gateway_container_id:
            exit_code, logs, stderr = await sandbox.exec_with_output( "sudo", "docker", "logs", gateway_container_id)
            logger.info(f"  stdout:\n{logs}")
            if stderr:
                logger.info(f"  stderr:\n{stderr}")
        if not gateway_ready:
            raise RuntimeError("Gateway did not become ready in time")

        # Check container status only on the sidecars the provider actually rendered
        sidecar_names = (
            self._state_provider.rendered_sidecar_service_names()
            if self._needs_local_postgres else []
        )
        for name in sidecar_names:
            container_id = await self._get_container_id(sandbox, name)
            if container_id:
                exit_code, logs, stderr = await sandbox.exec_with_output("sudo", "docker", "logs", container_id)
                logger.info(f"{name} logs:\n  stdout:\n{logs}")
                if stderr:
                    logger.info(f"  stderr:\n{stderr}")
            else:
                logger.warning(f"{name} container not found")

        gateway_url = sandbox.tunnel_urls.get(gateway_port)
        db_web_base_url = sandbox.tunnel_urls.get(DB_WEB_PORT) if PGWEB_SERVICE_NAME in sidecar_names else None
        db_web_url = f"{db_web_base_url}/" if db_web_base_url else None
        db_mcp_base_url = sandbox.tunnel_urls.get(DB_MCP_PORT) if DB_MCP_SERVICE_NAME in sidecar_names else None
        db_mcp_url = f"{db_mcp_base_url}/mcp" if db_mcp_base_url else None
        logger.info(f"pgweb URL: {db_web_url}")
        if db_mcp_url:
            logger.info(f"DB MCP endpoint: {db_mcp_url}")

        # Wait for the tunnel to serve the env card, which the deploy records
        card = await self._wait_for_tunnel(gateway_url, timeout=120)
        if card is None:
            raise RuntimeError("Tunnel did not become ready in time")
        card_read_at_utc = datetime.now(timezone.utc).isoformat()
        mcp_url = _mcp_url(gateway_url, card)
        logger.info(f"MCP endpoint: {mcp_url}")

        # Build frontend URLs proxied through the gateway
        website_frontend_urls: dict[str, str] | None = None
        if website_configs:
            website_frontend_urls = {
                wc.environment_name: f"{gateway_url}/website/{wc.environment_name}/"
                for wc in website_configs
            }
            for svc_name, url in website_frontend_urls.items():
                logger.info(f"Website frontend ({svc_name}): {url}")

        logger.info(f"Gateway URL: {gateway_url}")
        return DeployedGateway(
            gateway_url=gateway_url, mcp_url=mcp_url, db_web_url=db_web_url, db_mcp_url=db_mcp_url,
            website_frontend_urls=website_frontend_urls, mcp_server_name=mcp_server_name,
            env_state_instance_ids=[state_instance.instance_id],
            environment_card=card, environment_card_read_at_utc=card_read_at_utc,
        )

    async def _deploy_via_containers(
        self,
        sandbox_provider: SandboxProvider,
        mcp_servers: list[MCPServerConfig],
        mcp_server_images: list[DockerImageArtifact],
        gateway_port: int,
        website_configs: list[WebsiteConfig] | None,
        gateway_mode: GatewayMode,
        ttl_seconds: int,
        disk_size_gb: float,
        cpu: float | None = None,
        memory_mb: int | None = None,
        attribution: Optional[Attribution] = None,
        mcp_server_name: str | None = None,
    ) -> DeployedGateway:
        from agent_env.env.env import Env
        from agent_env.config import get_config

        if website_configs:
            raise NotImplementedError(
                "Container-mode gateway deploy does not support websites yet; use a VM-mode sandbox provider."
            )

        config = get_config()
        gateway_env = Env.get(config.default_gateway_env_id)

        i6pn_kwargs = {"i6pn": True, "region": config.modal_default_region} if isinstance(sandbox_provider, ModalSandboxProvider) else {}
        deploy = _ContainerDeploy(
            sandbox_provider=sandbox_provider, cpu=cpu, disk_size_gb=disk_size_gb, ttl_seconds=ttl_seconds,
            attribution=attribution, i6pn_kwargs=i6pn_kwargs,
        )

        try:
            if self._needs_local_postgres:
                await self._build_local_store(
                    mcp_servers, functools.partial(self._stand_up_servicedb, deploy), ttl_seconds=ttl_seconds
                )
            state_instance = self._state_instance

            # Stage 2 (prepare): create per-service schemas before the MCP servers connect. Local
            # no-ops (its servicedb init runs above); remote/external stores create schemas here.
            if self._state_provider is not None:
                await self._state_provider.prepare(
                    self._all_environments(mcp_servers, None), instance=state_instance
                )

            sidecar_specs = (
                self._state_provider.sidecar_specs(instance=state_instance)
                if self._needs_local_postgres else []
            )
            # Step 2: MCP server sandboxes (HTTPS), provisioned in parallel.
            mcp_coros = [
                self._provision_mcp(deploy, cfg, art, state_instance)
                for cfg, art in zip(mcp_servers, mcp_server_images)
            ]
            aux_coros = [self._provision_sidecar(deploy, spec) for spec in sidecar_specs]
            all_results = await asyncio.gather(*mcp_coros, *aux_coros, return_exceptions=True)

            mcp_results: list[tuple[str, "Sandbox"]] = []
            pgweb_sb = None
            db_mcp_sb = None
            errors: list[BaseException] = []
            for i, r in enumerate(all_results):
                if isinstance(r, BaseException):
                    logger.warning(f"sandbox batch child failed [{type(r).__name__}]: {(str(r) or repr(r))[-500:]}")
                    errors.append(r)
                    continue
                if i < len(mcp_coros):
                    environment_name, mcp_sb = r
                    self._container_sandboxes.append(mcp_sb)
                    self._environment_sandboxes[environment_name] = mcp_sb
                    mcp_results.append((environment_name, mcp_sb))
                else:
                    spec = sidecar_specs[i - len(mcp_coros)]
                    self._container_sandboxes.append(r)
                    if spec.name == PGWEB_SERVICE_NAME:
                        pgweb_sb = r
                        self._pgweb_sandbox = r
                    elif spec.name == DB_MCP_SERVICE_NAME:
                        db_mcp_sb = r
                        self._db_mcp_sandbox = r

            if errors:
                raise RuntimeError(f"{len(errors)} of {len(all_results)} sandbox provisioning tasks failed") from errors[0]

            mcp_url_public_by_name: dict[str, str] = {}
            mcp_url_i6pn_by_name: dict[str, str] = {}
            for environment_name, mcp_sb in mcp_results:
                mcp_url_public_by_name[environment_name] = f"{mcp_sb.tunnel_urls[AGENT_ENV_GATEWAY_MCP_PORT]}/mcp"
                mcp_i6pn = getattr(mcp_sb, "i6pn_address", None)
                if mcp_i6pn:
                    mcp_url_i6pn_by_name[environment_name] = f"http://[{mcp_i6pn}]:{AGENT_ENV_GATEWAY_MCP_PORT}/mcp"

            if i6pn_kwargs:
                missing_i6pn = [n for n in mcp_url_public_by_name if n not in mcp_url_i6pn_by_name]
                if missing_i6pn:
                    raise RuntimeError(f"i6pn requested but resolution failed for MCPs: {missing_i6pn}")
                internal_servers_by_name = mcp_url_i6pn_by_name
            else:
                internal_servers_by_name = mcp_url_public_by_name

            # Step 3: gateway sandbox (HTTPS) with full MCP URLs in env
            logger.info(f"Provisioning gateway from {gateway_env.docker_image_artifact.image_name}...")
            gateway_env_vars = {
                "INTERNAL_MCP_SERVERS": ",".join(f"{name}={url}" for name, url in internal_servers_by_name.items()),
                "REST_PROXY_URLS": ",".join(f"mcp-{name}={url.removesuffix('/mcp')}" for name, url in internal_servers_by_name.items()),
                "GATEWAY_MODE": gateway_mode.value,
                "SERVICE_DB_URL": self._state_provider.base_url(state_instance),
                "MCP_SERVER_NAME": mcp_server_name or DEFAULT_MCP_SERVER_NAME,
            }
            gateway_sb = await sandbox_provider.create_container(
                image_name=gateway_env.docker_image_artifact.image_name,
                port=gateway_port,
                env=gateway_env_vars,
                **_size_kwargs(cpu, memory_mb),
                disk_size_gb=disk_size_gb, timeout=ttl_seconds,
                attribution=attribution,
                **i6pn_kwargs,
            )
            self._container_sandboxes.append(gateway_sb)
            self._sandbox = gateway_sb
            gateway_url = gateway_sb.tunnel_urls[gateway_port]
            logger.info(f"Gateway URL: {gateway_url}")

            card = await self._wait_for_tunnel(gateway_url, timeout=120)
            if card is None:
                raise RuntimeError("Gateway tunnel did not become ready in time")
            card_read_at_utc = datetime.now(timezone.utc).isoformat()

            db_web_url = pgweb_sb.tunnel_urls[DB_WEB_PORT] + "/" if pgweb_sb else None
            db_mcp_url = db_mcp_sb.tunnel_urls[DB_MCP_CONTAINER_PORT] + "/mcp" if db_mcp_sb else None
            return DeployedGateway(
                gateway_url=gateway_url,
                mcp_url=_mcp_url(gateway_url, card),
                db_web_url=db_web_url,
                db_mcp_url=db_mcp_url,
                website_frontend_urls=None,
                mcp_server_name=mcp_server_name,
                env_state_instance_ids=[state_instance.instance_id],
                environment_card=card,
                environment_card_read_at_utc=card_read_at_utc,
            )
        except Exception:
            await self.close()
            raise

    async def _wait_for_gateway(self, sandbox: VmSandbox, port: int, timeout: int = 300) -> bool:
        """Wait for the gateway to answer on ``port``.

        Probes the well-known document, not ``/mcp``: a healthy MCP endpoint replies with
        an open event-stream, which a readiness check has no business holding. Any status
        below 500 means the app is up and routing.

        Each attempt is bounded so one hung request cannot consume the whole budget --
        ``exec_with_output`` runs on the host for some backends, where a stream can stall.
        """
        logger.info(f"Waiting for gateway to be ready (timeout={timeout}s)...")
        probe_url = f"http://localhost:{port}{WELL_KNOWN_PATH}"
        for i in range(timeout):
            exit_code, stdout, stderr = await sandbox.exec_with_output(
                "curl", "-s", "--max-time", "5", "-o", "/dev/null", "-w", "%{http_code}", probe_url
            )
            http_code = stdout.strip()
            if http_code.isdigit() and int(http_code) > 0 and int(http_code) < 500:
                logger.info(f"  Gateway ready (HTTP {http_code}) after {i+1}s")
                return True
            if i % 10 == 0:
                logger.info(f"  Still waiting... (HTTP {http_code})")
            await asyncio.sleep(1)
        logger.info(f"  Gateway not ready after {timeout}s")
        return False

    async def read_trajectory(self, sandbox: VmSandbox) -> list[dict]:
        """Read trajectory JSONL from gateway container. A read whose exec transport failed (exit -1) may
        hold only part of the history, so it is retried and then raised, never parsed."""
        container_id = await self._get_container_id(sandbox, GATEWAY_SERVICE_NAME)
        if not container_id:
            return []

        for attempt in range(_TRAJECTORY_READ_ATTEMPTS):
            exit_code, stdout, stderr = await sandbox.exec_with_output(
                "sudo", "docker", "exec", container_id, "cat", "/var/log/agentenv/trajectory.jsonl"
            )
            if exit_code != -1:
                break
            logger.warning(f"Trajectory read lost its exec transport (attempt {attempt + 1}): {stderr[-200:]}")
        else:
            raise RuntimeError(f"Could not read the gateway trajectory: exec transport failed {_TRAJECTORY_READ_ATTEMPTS} times")
        events = []
        for line in stdout.strip().split("\n"):
            if line:
                events.append(json.loads(line))
        return events

    async def _get_container_id(self, sandbox: VmSandbox, compose_service: str) -> str | None:
        """Get container ID for a docker-compose service (including exited containers). A lookup whose
        exec transport failed (exit -1) is retried, then raised: it can't tell "no container" apart."""
        for attempt in range(_TRAJECTORY_READ_ATTEMPTS):
            exit_code, stdout, stderr = await sandbox.exec_with_output(
                "sudo", "docker", "compose", "-f", DOCKER_COMPOSE_PATH, "ps", "-a", "-q", compose_service
            )
            if exit_code != -1:
                break
            logger.warning(f"Container lookup for {compose_service} lost its exec transport (attempt {attempt + 1}): {stderr[-200:]}")
        else:
            raise RuntimeError(f"Could not look up the {compose_service} container: exec transport failed {_TRAJECTORY_READ_ATTEMPTS} times")
        container_id = stdout.strip()
        return container_id if container_id else None

    async def _provision_sidecar(self, deploy: _ContainerDeploy, spec) -> Sandbox:
        logger.info(f"Provisioning {spec.name} from {spec.image}...")
        return await deploy.sandbox_provider.create_container(
            image_name=spec.image,
            port=spec.port,
            env=spec.env,
            command=spec.command,
            **_size_kwargs(deploy.cpu, None),
            disk_size_gb=deploy.disk_size_gb, timeout=deploy.ttl_seconds,
            attribution=deploy.attribution,
            **deploy.i6pn_kwargs,
        )

    @property
    def _needs_local_postgres(self) -> bool:
        from agent_env.providers.env_state import LocalPostgresStateProvider

        return isinstance(self._state_provider, LocalPostgresStateProvider)

    async def _build_local_store(
        self,
        mcp_servers: list[MCPServerConfig],
        stand_up: Callable[[LocalPostgresStoreSpec], Awaitable[str]],
        website_configs: list[WebsiteConfig] | None = None,
        ttl_seconds: int = 10800,
    ) -> LocalPostgresStoreSpec:
        """A **local** Postgres store, which can only be built within the gateway."""
        from agent_env.providers.env_state import (
            LocalPostgresStateContext,
            LocalPostgresStateProvider,
        )

        # MCP servers and websites are both DB clients (one schema each); the container path
        # passes no websites, so normalize to a list once.
        configs = [*mcp_servers, *(website_configs or [])]
        environment_names = list(dict.fromkeys(c.environment_name for c in configs))
        provider = self._state_provider  # set by _deploy_gateway
        assert isinstance(provider, LocalPostgresStateProvider)
        spec = provider.store_spec(environment_names)
        host = await stand_up(spec)
        self._state_instance = await provider.acquire(
            LocalPostgresStateContext(
                environment_names=environment_names, host=host, ttl_seconds=ttl_seconds
            )
        )
        return spec

    async def _init_service_db_via_exec(self, db_sb: Sandbox, init_sql: str) -> None:
        from agent_env.env.envs.service_db import DB_NAME, DB_USER

        logger.info("Initializing service-db schemas + changelog functions via sandbox exec...")
        for attempt in range(60):
            exit_code, _, stderr = await db_sb.exec_with_output("pg_isready", "-U", DB_USER, "-d", DB_NAME)
            if exit_code == 0:
                break
            if attempt % 10 == 0:
                logger.info(f"  Waiting for postgres to be ready... (last stderr: {stderr.strip()[-200:]})")
            await asyncio.sleep(1)
        else:
            raise RuntimeError("Postgres not ready after 60s")

        await db_sb.write_file_from_text(init_sql, "/tmp/init-schemas.sql")
        exit_code, stdout, stderr = await db_sb.exec_with_output(
            "psql", "-U", DB_USER, "-d", DB_NAME, "-v", "ON_ERROR_STOP=1", "-f", "/tmp/init-schemas.sql",
        )
        if exit_code != 0:
            raise RuntimeError(f"Service-db init failed: {stderr[-1000:]}")
        logger.info("Service-db schemas initialized.")

    async def _stand_up_servicedb(self, deploy: _ContainerDeploy, spec: "LocalPostgresStoreSpec") -> str:
        """Container mode stands the servicedb up as its own i6pn-only sandbox; returns its host."""
        logger.info("Provisioning service-db sandbox (Modal, i6pn-only)...")
        # Local backend only (external skips standup); the provider is set by _deploy_gateway.
        db_sb = await deploy.sandbox_provider.create_container(
            image_name=self._state_provider.container_store_image(),
            port=spec.port,
            env=spec.env,
            **_size_kwargs(deploy.cpu, None),
            disk_size_gb=deploy.disk_size_gb, timeout=deploy.ttl_seconds,
            expose_externally=False,
            attribution=deploy.attribution,
            **deploy.i6pn_kwargs,
        )
        self._container_sandboxes.append(db_sb)
        self._db_sandbox = db_sb
        db_i6pn_address = getattr(db_sb, "i6pn_address", None)
        if not db_i6pn_address:
            raise RuntimeError("service-db has no i6pn address; cannot route postgres traffic")
        host = f"[{db_i6pn_address}]"
        await self._init_service_db_via_exec(db_sb, spec.init_sql)
        logger.info(f"Service-db ready at {host}:{spec.port} (i6pn-only)")
        return host

    async def _provision_mcp(
        self,
        deploy: _ContainerDeploy,
        cfg: MCPServerConfig,
        image_artifact: "DockerImageArtifact",
        state_instance: "EnvStateInstance | None",
    ) -> tuple[str, Sandbox]:
        logger.info(f"Provisioning MCP server '{cfg.environment_name}' from {image_artifact.image_name}...")
        env = dict(cfg.extra_env_vars or {})
        env["DATABASE_URL"] = self._state_provider.url_for_environment(cfg.environment_name, instance=state_instance)
        env["ENVIRONMENT_NAME"] = cfg.environment_name
        if deploy.i6pn_kwargs:
            env.setdefault("MCP_HOST", "::")
        sb = await deploy.sandbox_provider.create_container(
            image_name=image_artifact.image_name,
            port=AGENT_ENV_GATEWAY_MCP_PORT,
            env=env,
            **_size_kwargs(deploy.cpu, None),
            disk_size_gb=deploy.disk_size_gb, timeout=deploy.ttl_seconds,
            attribution=deploy.attribution,
            **deploy.i6pn_kwargs,
        )
        return cfg.environment_name, sb

    async def _deploy_gateway(
        self,
        sandbox_provider: SandboxProvider,
        *,
        state_instance: "EnvStateInstance | None",
        mcp_servers: list[MCPServerConfig],
        mcp_server_images: list[DockerImageArtifact],
        gateway_port: int = AGENT_ENV_GATEWAY_MCP_PORT,
        website_configs: list[WebsiteConfig] | None = None,
        website_images: list[DockerImageArtifact] | None = None,
        gateway_mode: GatewayMode = GatewayMode.PERFORMANCE,
        ttl_seconds: int = 10800,
        disk_size_gb: float = 10,
        cpu: float | None = None,
        memory_mb: int | None = None,
        existing_sandbox: Sandbox | None = None,
        sidecars: list[SidecarConfig] | None = None,
        mcp_server_name: str | None = None,
        attribution: Optional[Attribution] = None,
    ) -> DeployedGateway:
        """One gateway deploy on one (unchained) sandbox provider: Modal containers for a ``ModalSandboxProvider``, else a VM."""
        from agent_env.providers.env_state import LocalPostgresStateProvider, build_state_provider

        # Every attempt starts from the caller's store: a failed attempt's close() cleared it.
        self._state_instance = state_instance
        self._state_provider = (
            build_state_provider(state_instance.state_type)
            if state_instance is not None
            else LocalPostgresStateProvider()
        )
        deploy_kwargs = {"attribution": dict(attribution or {})}
        if isinstance(sandbox_provider, ModalSandboxProvider):
            if existing_sandbox is not None:
                raise ValueError("existing_sandbox is only supported for VM-mode providers")
            if sidecars:
                raise NotImplementedError("sidecars are only supported for VM-mode gateway deploys")
            return await self._deploy_via_containers(
                sandbox_provider, mcp_servers, mcp_server_images,
                gateway_port=gateway_port, website_configs=website_configs, gateway_mode=gateway_mode,
                ttl_seconds=ttl_seconds, disk_size_gb=disk_size_gb,
                cpu=cpu, memory_mb=memory_mb, mcp_server_name=mcp_server_name, **deploy_kwargs,
            )
        return await self._deploy_via_vm(
            sandbox_provider, mcp_servers, mcp_server_images,
            gateway_port=gateway_port, website_configs=website_configs, website_images=website_images,
            gateway_mode=gateway_mode, ttl_seconds=ttl_seconds, disk_size_gb=disk_size_gb,
            cpu=cpu, memory_mb=memory_mb, existing_sandbox=existing_sandbox, sidecars=sidecars,
            mcp_server_name=mcp_server_name, **deploy_kwargs,
        )

    def _sandbox_ids(self, always_name_gateway: bool = False) -> dict[str, str | dict[str, str]]:
        """The containers a container-mode deploy created, in the record's ``sandbox_ids`` shape; ``{}`` on a VM, or only the
        gateway where ``always_name_gateway``, as a MultiEnv's records have always named it."""
        from agent_env.env.envs.gateway_server import GatewayEnv
        from agent_env.env.envs.mcp_server import MCPServerEnv
        from agent_env.env.envs.service_db import ServiceDBEnv

        if not self._environment_sandboxes and not always_name_gateway:
            return {}
        sandbox_ids: dict[str, str | dict[str, str]] = {GatewayEnv.type: self._sandbox.sandbox_id}
        if self._environment_sandboxes:
            sandbox_ids[MCPServerEnv.type] = {name: sb.sandbox_id for name, sb in self._environment_sandboxes.items()}
        service_db = {role: sb.sandbox_id for role, sb in [
            (DATABASE_SERVICE_NAME, self._db_sandbox), (PGWEB_SERVICE_NAME, self._pgweb_sandbox), (DB_MCP_SERVICE_NAME, self._db_mcp_sandbox),
        ] if sb is not None}
        if service_db:
            sandbox_ids[ServiceDBEnv.type] = service_db
        return sandbox_ids

    async def _reattach(self, env: Env, deployed: DeployedEnv) -> Sandbox:
        """A record this provider wrote, reattached: a MultiEnv's through its own rebuild, any other env's as the base reattaches it."""
        from agent_env.env.envs.multi_env import MultiEnv

        if isinstance(env, MultiEnv):
            return await self._reattach_multi_env(deployed)
        return await super()._reattach(env, deployed)

    async def _reattach_multi_env(self, deployed: DeployedEnv) -> Sandbox:
        """A MultiEnv's record reattached: every sandbox it names and its state store, for its children's loads and close().

        Unlike a single server's reattach, which skips a dead database or sidecar, any dead sandbox the record names raises.
        """
        from agent_env.env.envs.mcp_server import MCPServerEnv
        from agent_env.env.envs.service_db import ServiceDBEnv
        from agent_env.providers import build_sandbox_provider, get_env_sandbox_provider
        from agent_env.providers.env_state import LocalPostgresStateProvider, build_state_provider
        from agent_env.providers.env_state.store import get_env_state_instance_store
        from agent_env.store.base import NotFoundError

        provider = build_sandbox_provider(deployed.sandbox_type) if deployed.sandbox_type else get_env_sandbox_provider()
        sandbox_ids = deployed.sandbox_ids or {}

        self._sandbox = await provider.get_sandbox(deployed.sandbox_id)

        for environment_name, sb_id in (sandbox_ids.get(MCPServerEnv.type) or {}).items():
            sb = await provider.get_sandbox(sb_id)
            self._environment_sandboxes[environment_name] = sb
            self._container_sandboxes.append(sb)

        db_ids = sandbox_ids.get(ServiceDBEnv.type) or {}
        for role, slot_name in [
            (DATABASE_SERVICE_NAME, "_db_sandbox"),
            (PGWEB_SERVICE_NAME, "_pgweb_sandbox"),
            (DB_MCP_SERVICE_NAME, "_db_mcp_sandbox"),
        ]:
            sb_id = db_ids.get(role)
            if sb_id:
                sb = await provider.get_sandbox(sb_id)
                setattr(self, slot_name, sb)
                self._container_sandboxes.append(sb)

        if self._sandbox.mode == SANDBOX_MODE_CONTAINER:
            self._container_sandboxes.append(self._sandbox)

        state_instance = None
        ids = deployed.env_state_instance_ids if isinstance(deployed, DeployedGatewayEnv) else None
        if ids:
            try:
                state_instance = get_env_state_instance_store().get(ids[0])
            except NotFoundError:
                logger.warning(f"env_state_instance {ids[0]} not found on reattach; defaulting to local")
        # No record (env predates state instances, or it expired) → default to the local store.
        state_instance = state_instance or LocalPostgresStateProvider.default_instance()
        self._state_instance = state_instance
        self._state_provider = build_state_provider(state_instance.state_type)
        return self._sandbox


@dataclass(frozen=True)
class _GatewayTopology:
    """What a gateway fronts for one env, and the one way its record differs by env type."""

    mcp_servers: list[MCPServerConfig]
    mcp_server_images: list[DockerImageArtifact]
    website_configs: list[WebsiteConfig] | None
    website_images: list[DockerImageArtifact] | None
    mcp_server_name: str | None  # a MultiEnv's name; None draws one
    names_gateway_on_vm: bool  # a MultiEnv's record names its gateway sandbox even on a VM


def _gateway_topology(env: Env) -> _GatewayTopology:
    """What a gateway fronts for env: a MultiEnv's MCP servers and websites (whatever their own env_provider_type) under its
    name, a WebsiteEnv's website, or else the env's one MCP server."""
    from agent_env.env.envs.multi_env import MultiEnv
    from agent_env.env.envs.website import WebsiteEnv

    is_multi = isinstance(env, MultiEnv)
    if is_multi:
        servers, websites, name = env.mcp_server_envs, env.website_envs, env.name
    elif isinstance(env, WebsiteEnv):
        servers, websites, name = [], [env], None
    else:
        servers, websites, name = [env], [], None
    return _GatewayTopology(
        mcp_servers=[MCPServerConfig(image=s.docker_image_artifact.image_name, environment_name=s.environment_name) for s in servers],
        mcp_server_images=[s.docker_image_artifact for s in servers],
        website_configs=[
            WebsiteConfig(backend_image=w.backend_docker_image_artifact.image_name, frontend_image=w.frontend_docker_image_artifact.image_name,
                          environment_name=w.environment_name)
            for w in websites
        ] or None,
        website_images=[image for w in websites for image in (w.backend_docker_image_artifact, w.frontend_docker_image_artifact)] or None,
        mcp_server_name=name,
        names_gateway_on_vm=is_multi,
    )


# Compose `environment:` keys whose values are secrets and must be masked before the
# rendered compose is logged. Matches the key to the left of the first '=' on an
# `      - KEY=VALUE` env line.
_SECRET_ENV_KEY_RE = re.compile(r"^(\s*-\s*)([A-Za-z0-9_]*(?:SECRET|PASSWORD|TOKEN|API_KEY)[A-Za-z0-9_]*)=.*$")


def _redact_compose_secrets(compose_content: str) -> str:
    """Mask secret-valued env lines so the logged compose never carries a live secret."""
    return "\n".join(
        _SECRET_ENV_KEY_RE.sub(r"\1\2=***", line) for line in compose_content.splitlines()
    )


# Cap on the post-readiness tools/list probe; it only logs, so a slow gateway costs deploy time, never the deploy.
_TOOLS_PROBE_TIMEOUT_S = 10.0


async def _probe_tools(env_id: str | None, gateway: DeployedGateway) -> None:
    """Log whether the gateway's MCP tools cover every tool its card's children advertise; never raises."""
    start = time.monotonic()
    head = f"env_deploy_tools_probe env_id={env_id or 'unknown'}"
    children = ""
    try:
        cards = (gateway.environment_card or {}).get("children_environments") or []
        children = ",".join(c.get("name") or "" for c in cards)
        async with asyncio.timeout(_TOOLS_PROBE_TIMEOUT_S):
            tools = set(await _tool_names(gateway.mcp_url))
        missing = [c.get("name") or "" for c in cards if not _card_tool_names(c) <= tools]
    except Exception as e:
        logger.warning(f"{head} status=failed duration_s={time.monotonic() - start:.1f} children={children} error_type={type(e).__name__} error={e!r}")
        return
    status = "ok" if tools and not missing else "missing"
    log = logger.info if status == "ok" else logger.warning
    log(f"{head} status={status} tools={len(tools)} duration_s={time.monotonic() - start:.1f} children={children} missing={','.join(missing)}")


def _card_tool_names(card: dict) -> set[str]:
    return {tool.get("name") for tool in (card.get("capabilities") or {}).get("tools") or []}


@dataclass(frozen=True)
class _ContainerDeploy:
    """What every container of one container-mode deploy is created with."""

    sandbox_provider: SandboxProvider
    cpu: float | None
    disk_size_gb: float
    ttl_seconds: int
    attribution: Optional[Attribution]
    i6pn_kwargs: dict
