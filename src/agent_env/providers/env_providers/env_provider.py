"""Environment providers: the contract a provider implements, the built-ins' shared base, and finding a provider by type."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from agent_env.env.env import DeployedEnv
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.plugins import _registration
from agent_env.providers.sandbox_providers.sandbox import Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER

if TYPE_CHECKING:
    from agent_env.config.runtime import Config
    from agent_env.env.env import Env
    from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

logger = logging.getLogger(__name__)


class EnvironmentProvider(ABC):
    """Deploys an env and hands back its record; tears down what it created."""

    type: ClassVar[str]  # the name an env declares it by, and that its records carry as env_provider_type

    @abstractmethod
    async def deploy(self, env: Env, sandbox_provider: SandboxProvider) -> DeployedEnv:
        """Deploy the env and return its record, unregistered: the env registers it. Take ``**options``, which receives every option
        the env's deploy was given; a deploy that names its options instead is refused any other one set away from its default."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down everything deploy() created."""


def record_class_for(env_provider_type: str | None) -> type[DeployedEnv] | None:
    """The record class a built-in provider type writes; None for any other type, whose records load by shape."""
    provider = _builtin_env_providers().get(env_provider_type)
    return provider.record_class if provider is not None else None


def build_env_provider(env_provider_type: str) -> EnvironmentProvider:
    """A fresh provider of the type an env declares, built in or from a plugin; raises for a type this process can't find."""
    return _env_provider_class(env_provider_type)()


def _env_provider_class(env_provider_type: str) -> type[EnvironmentProvider]:
    """The registered provider class of that type; raises, naming the registered types, for one this process can't find."""
    from agent_env.config import runtime

    providers = runtime.get_config().env_provider_registry()
    if env_provider_type not in providers:
        raise ValueError(f"Unknown env_provider_type: {env_provider_type!r} (expected one of {sorted(providers)})"
                         f"{_registration.failure_note(_registration.ENV_PROVIDERS, env_provider_type)}")
    return providers[env_provider_type]


class _SandboxEnvironmentProvider(EnvironmentProvider):
    """The built-in providers' base: tracks the containers it creates in our sandboxes, closes them, reattaches to the records it wrote,
    and retries across a chained spec."""

    record_class: ClassVar[type[DeployedEnv]] = DeployedEnv  # the class its records load back as

    def __init__(self):
        self._container_sandboxes: list[Sandbox] = []
        self._environment_sandboxes: dict[str, Sandbox] = {}

    def environment_sandbox(self, environment_name: str) -> Sandbox | None:
        return self._environment_sandboxes.get(environment_name)

    async def close(self) -> None:
        for sb in self._container_sandboxes:
            try:
                await sb.terminate()
            except Exception as e:
                logger.warning(f"Failed to terminate {type(sb).__name__} {sb.sandbox_id}: {e}")
        self._container_sandboxes.clear()
        self._environment_sandboxes.clear()

    async def _reattach(self, env: Env, deployed: DeployedEnv) -> Sandbox:
        """A record this provider wrote, reattached: its primary sandbox, or in container mode the server's container, with every recorded one for close()."""
        from agent_env.env.envs.mcp_server import MCPServerEnv
        from agent_env.providers import build_sandbox_provider, get_env_sandbox_provider
        provider = build_sandbox_provider(deployed.sandbox_type) if deployed.sandbox_type else get_env_sandbox_provider()
        sandbox = await provider.get_sandbox(deployed.sandbox_id)
        if env.environment_name in deployed.sandbox_ids.get(MCPServerEnv.type, {}):
            # Container mode.
            sandbox = await self._reattach_containers(provider, sandbox, deployed.sandbox_ids, env.environment_name)
        return sandbox

    async def _reattach_containers(self, provider: SandboxProvider, gateway: Sandbox, sandbox_ids: dict, environment_name: str) -> Sandbox:
        """Reattach the recorded containers for close(), skipping a dead servicedb or sidecar; returns the server's container, raising if it's dead."""
        from agent_env.env.envs.mcp_server import MCPServerEnv
        from agent_env.env.envs.service_db import ServiceDBEnv
        server_id = sandbox_ids[MCPServerEnv.type][environment_name]
        db_ids = list(sandbox_ids.get(ServiceDBEnv.type, {}).values())
        server, *dbs = await asyncio.gather(*(provider.get_sandbox(i) for i in [server_id, *db_ids]), return_exceptions=True)
        if isinstance(server, BaseException):
            raise RuntimeError(f"[{environment_name}] can't reattach the server's container {server_id}: {type(server).__name__}: {server}") from server
        server.mode = SANDBOX_MODE_CONTAINER  # recorded under mcp_server, so create_container made it
        for db_id, db in zip(db_ids, dbs):
            if isinstance(db, BaseException):
                logger.warning(f"[{environment_name}] skipping recorded container {db_id}: {type(db).__name__}: {db}")
        self._container_sandboxes = [server, *(db for db in dbs if not isinstance(db, BaseException)), gateway]
        return server

    async def _wait_for_tunnel(self, url: str, timeout: int = 60) -> dict | None:
        """Wait until the deployed env serves its card to this process; returns the card.

        Probes the well-known document rather than ``/mcp``, whose healthy endpoint holds the
        connection open. Only a 200 carrying a card counts, and that card is what the deploy
        records. Returns None if none arrives within ``timeout`` attempts.
        """
        probe_url = f"{url}{WELL_KNOWN_PATH}"
        logger.info(f"Waiting for tunnel to be ready (timeout={timeout}s) via {probe_url}...")
        last = "no response"
        async with httpx.AsyncClient() as client:
            for i in range(timeout):
                try:
                    response = await client.get(probe_url, timeout=5)
                    card = _card_from(response)
                    if card is not None:
                        logger.info(f"  Tunnel ready (HTTP {response.status_code}) after {i+1}s")
                        return card
                    last = f"HTTP {response.status_code} {response.headers.get('content-type', '')}".rstrip()
                except Exception as e:
                    last = f"error: {type(e).__name__}"
                if i % 10 == 0:
                    logger.info(f"  Still waiting... ({last})")
                await asyncio.sleep(1)
        logger.info(f"  Tunnel not ready after {timeout}s (last: {last})")
        return None

    def _sandbox_providers(self, sandbox_provider: SandboxProvider, env_id: str | None) -> list[SandboxProvider]:
        """The sandbox providers to try, in order: a chained one's members, or itself."""
        from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider

        if isinstance(sandbox_provider, ChainedSandboxProvider):
            return list(sandbox_provider._providers)
        return [sandbox_provider]

    async def _run(self, sandbox_provider: SandboxProvider, env_id: str | None, attempt: Callable[[SandboxProvider], Awaitable[Any]]) -> Any:
        """Run ``attempt`` on the sandbox provider, or on each chained member until one succeeds; a deploy that fails or is
        cancelled is closed."""
        try:
            sandbox_providers = self._sandbox_providers(sandbox_provider, env_id)
            if len(sandbox_providers) == 1:
                return await self._attempt(sandbox_providers[0], env_id, attempt)
            errors: list[tuple[str, Exception]] = []
            for p in sandbox_providers:
                try:
                    return await self._attempt(p, env_id, attempt)
                except Exception as e:
                    errors.append((type(p).__name__, e))
                    await self.close()
            detail = "; ".join(f"{n}: {e!r}" for n, e in errors)
            raise RuntimeError(f"All {len(sandbox_providers)} chained providers failed to deploy: {detail}")
        except BaseException:
            await self.close()
            raise

    async def _attempt(self, sandbox_provider: SandboxProvider, env_id: str | None, attempt: Callable[[SandboxProvider], Awaitable[Any]]) -> Any:
        """One deploy on one sandbox provider, logged as an ``env_deploy_provider_attempt``."""
        start = time.monotonic()
        provider_name = type(sandbox_provider).__name__
        try:
            result = await attempt(sandbox_provider)
            logger.info(f"env_deploy_provider_attempt env_id={env_id or 'unknown'} provider={provider_name} status=success duration_s={time.monotonic() - start:.1f}")
        except Exception as e:
            logger.warning(f"env_deploy_provider_attempt env_id={env_id or 'unknown'} provider={provider_name} status=failure duration_s={time.monotonic() - start:.1f} error_type={type(e).__name__} error={e!r}")
            raise
        return result


def _size_kwargs(cpu: float | None, memory_mb: int | None) -> dict:
    """Forward only what the caller set, so the provider's own default stands otherwise.

    Passing None through would not reach that default: the providers hand cpu/memory
    straight to the backend, so an explicit None means "no reservation at all" rather than
    "your floor".
    """
    kwargs: dict = {}
    if cpu is not None:
        kwargs["cpu"] = cpu
    if memory_mb is not None:
        kwargs["memory"] = memory_mb
    return kwargs


def _card_from(response: httpx.Response) -> dict | None:
    """The env card in a readiness response: a 200 whose JSON body is an object with a name."""
    if response.status_code != 200:
        return None
    try:
        card = response.json()
    except ValueError:
        return None
    return card if isinstance(card, dict) and isinstance(card.get("name"), str) and card["name"] else None


async def _tool_names(mcp_url: str) -> list[str]:
    """The server's tool names as it lists them, duplicates included."""
    async with streamable_http_client(mcp_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return [tool.name for tool in (await session.list_tools()).tools]


def _builtin_env_providers() -> dict[str, type[_SandboxEnvironmentProvider]]:
    """The built-in providers, by type."""
    from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
    from agent_env.providers.env_providers.env_server_provider import EnvironmentServerProvider

    return {p.type: p for p in (EnvironmentGatewayProvider, EnvironmentServerProvider)}


def _build_registry(source: Config | None = None) -> dict[str, type[EnvironmentProvider]]:
    """The built-in providers, then ``agent_env.env_providers`` plugins."""
    registry: dict[str, type[EnvironmentProvider]] = dict(_builtin_env_providers())
    _registration.merge(registry, _registration.ENV_PROVIDERS, _validate_plugin, source=source)
    return registry


def _validate_plugin(name: str, loaded: Any) -> type[EnvironmentProvider]:
    cls = _registration.require_subclass(loaded, EnvironmentProvider)
    if getattr(cls, "type", None) != name:
        raise TypeError(f"{cls.__qualname__} has type {getattr(cls, 'type', None)!r}; it must equal the entry-point "
                        "name, which its records carry as env_provider_type")
    if problem := _registration.unimplemented(cls, EnvironmentProvider):
        raise TypeError(problem)
    return cls
