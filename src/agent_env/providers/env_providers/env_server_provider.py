"""The server environment provider: one MCP server in its own container on the configured sandbox provider, with no gateway in front and no state store."""

from __future__ import annotations

import asyncio
import functools
from datetime import datetime, timezone
from typing import TYPE_CHECKING, ClassVar, Optional

import anyio

from agent_env.attribution import Attribution
from agent_env.env.env import DeployedSandboxEnv
from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.providers.env_providers.env_provider import _SandboxEnvironmentProvider, _size_kwargs, _tool_names
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

if TYPE_CHECKING:
    from agent_env.env.env import Env


class EnvironmentServerProvider(_SandboxEnvironmentProvider):
    """Deploys one MCP server with no gateway in front; the server's own card is the env card."""

    type: ClassVar[str] = "server"
    record_class = DeployedSandboxEnv

    async def deploy(
        self,
        env: Env,
        sandbox_provider: SandboxProvider,
        *,
        ttl_seconds: int = 10800,
        disk_size_gb: float = 10,
        cpu: float | None = None,
        memory_mb: int | None = None,
        attribution: Optional[Attribution] = None,
        artifact_id: str | None = None,
        artifact_version: int | None = None,
    ) -> DeployedSandboxEnv:
        """The env's MCP server in its own container; returns the record once it serves its card and its tools."""
        from agent_env.env.envs.multi_env import MultiEnv
        from agent_env.env.envs.website import WebsiteEnv

        if isinstance(env, (MultiEnv, WebsiteEnv)):
            raise TypeError(f"env_provider_type '{self.type}' deploys one MCP server, not a {env.type} env")
        attempt = functools.partial(
            self._deploy_server, env, ttl_seconds=ttl_seconds, disk_size_gb=disk_size_gb,
            cpu=cpu, memory_mb=memory_mb, attribution=attribution,
        )
        return await self._run(sandbox_provider, env.id, attempt)

    async def install_changelog_triggers(self, environment_name: str) -> None:
        """No state store, so no changelog to install."""

    async def _deploy_server(
        self,
        env: Env,
        sandbox_provider: SandboxProvider,
        *,
        ttl_seconds: int,
        disk_size_gb: float,
        cpu: float | None,
        memory_mb: int | None,
        attribution: Optional[Attribution],
    ) -> DeployedSandboxEnv:
        """One attempt: the server's container, then its card and tools; returns its record."""
        from agent_env.env.envs.mcp_server import MCPServerEnv

        name = env.environment_name
        server = await sandbox_provider.create_container(
            image_name=env.docker_image_artifact.image_name,
            port=AGENT_ENV_GATEWAY_MCP_PORT,
            env={
                "ENVIRONMENT_NAME": name,
                "MCP_HOST": "0.0.0.0",  # IPv4 too: an IPv6-only "::" bind starves tunnels and published ports
                "DATABASE_URL": f"sqlite:////tmp/{name}.sqlite",
            },
            **_size_kwargs(cpu, memory_mb),
            disk_size_gb=disk_size_gb, timeout=ttl_seconds,
            attribution=dict(attribution or {}),
        )
        self._container_sandboxes.append(server)
        self._environment_sandboxes[name] = server
        server_url = server.tunnel_urls[AGENT_ENV_GATEWAY_MCP_PORT]
        try:
            async with asyncio.timeout(_READINESS_TIMEOUT_S):
                card = await self._wait_for_tunnel(server_url, timeout=int(_READINESS_TIMEOUT_S))
        except TimeoutError:
            card = None
        if card is None:
            raise RuntimeError(f"'{name}' did not serve its env card at {server_url}{WELL_KNOWN_PATH} in time; "
                               "an env deployed without a gateway must serve its own card")
        card_read_at_utc = datetime.now(timezone.utc).isoformat()
        if card["name"] != name:
            raise RuntimeError(f"'{name}' serves a card named '{card['name']}'; "
                               "an env deployed without a gateway must serve its card under its environment_name")
        record = DeployedSandboxEnv(
            env_id=env.id,
            env_version=env.version,
            env_provider_type=self.type,
            environment_card_url=f"{server_url}{WELL_KNOWN_PATH}",
            environment_card=card,
            environment_card_read_at_utc=card_read_at_utc,
            sandbox_id=server.sandbox_id,
            sandbox_type=server.type,
            sandbox_ids={MCPServerEnv.type: {name: server.sandbox_id}},
        )
        await _require_tools(name, record.mcp_url)
        return record


# Wall-clock cap on the server's readiness: attempts alone don't bound a tunnel that hangs each request.
_READINESS_TIMEOUT_S = 180.0

_TOOLS_GATE_TIMEOUT_S = 60.0


async def _require_tools(environment_name: str, mcp_url: str) -> None:
    """Fail unless the server at ``mcp_url`` lists at least one MCP tool and no tool name twice."""
    try:
        # anyio re-cancels in cleanup; asyncio.timeout would wait out the MCP client's session-closing DELETE (up to 300 s).
        with anyio.fail_after(_TOOLS_GATE_TIMEOUT_S):
            tools = await _tool_names(mcp_url)
    except Exception as e:
        raise RuntimeError(f"'{environment_name}' did not list its MCP tools at {mcp_url}: {e!r}") from e
    if not tools:
        raise RuntimeError(f"'{environment_name}' lists no MCP tools at {mcp_url}; an env deployed without a gateway must serve at least one")
    duplicates = sorted({t for t in tools if tools.count(t) > 1})
    if duplicates:
        raise RuntimeError(f"'{environment_name}' lists duplicate MCP tool names at {mcp_url}: {', '.join(duplicates)}")
