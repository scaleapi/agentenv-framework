"""Vercel Sandbox provider."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from typing import Any, Callable, ClassVar, Self

from agent_env.attribution import Attribution
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.sandbox import (
    NetworkPolicy,
    NetworkPolicyUnsupportedError,
)
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxProvider,
    apply_default_attribution,
)
from agent_env.providers.sandbox_providers.vercel.sandbox import (
    VercelSandbox,
    network_policy_from_vercel,
    vercel_network_policy,
)

logger = logging.getLogger(__name__)

_MAX_VCPUS = 32
_MAX_MEMORY_MB = 64 * 1024
_MAX_DISK_GB = 64
_MAX_PORTS = 15
_MAX_TAGS = 5
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
_MEMORY_PER_VCPU_MB = 2048
_CLEANUP_ATTEMPTS = 3


def resource_shape(cpu: float, memory: int) -> tuple[int, int]:
    if isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or not math.isfinite(cpu) or cpu <= 0:
        raise ValueError("cpu must be a finite number greater than zero")
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0 or memory > _MAX_MEMORY_MB:
        raise ValueError(f"memory must be an integer from 1 to {_MAX_MEMORY_MB} MB")
    vcpus = max(math.ceil(cpu), math.ceil(memory / _MEMORY_PER_VCPU_MB), 1)
    if vcpus != 1 and vcpus % 2:
        vcpus += 1
    if vcpus > _MAX_VCPUS:
        raise ValueError(
            f"no Vercel resource shape fits cpu={cpu}, memory={memory}MB "
            f"(largest supported provider shape is {_MAX_VCPUS} vCPUs)"
        )
    return vcpus, vcpus * _MEMORY_PER_VCPU_MB


def _sdk_resources(vcpus: int, memory: int) -> Any:
    try:
        from vercel.sandbox import SandboxResources
    except ImportError as error:
        raise RuntimeError(
            "The vercel sandbox provider requires vercel-sandbox; "
            "install agentenv-framework[vercel]"
        ) from error
    return SandboxResources(vcpus=vcpus, memory=memory)


def _is_missing_resource(error: BaseException) -> bool:
    return getattr(error, "status_code", None) == 404


async def _destroy_named(client: Any, name: str) -> None:
    try:
        sandbox = await client.get_sandbox(name=name)
        await sandbox.destroy(delete_orphan_snapshots=True)
    except BaseException as error:
        if not _is_missing_resource(error):
            raise


async def _reap(client: Any, name: str) -> None:
    for attempt in range(_CLEANUP_ATTEMPTS):
        try:
            await _destroy_named(client, name)
            logger.info("Destroyed Vercel sandbox %s left by an interrupted create", name)
            return
        except Exception as error:
            logger.warning(
                "Destroying Vercel sandbox %s failed (attempt %s): %s",
                name,
                attempt + 1,
                error,
            )
            await asyncio.sleep(2**attempt)
    logger.error("Vercel sandbox %s may remain after %s destroy attempts", name, _CLEANUP_ATTEMPTS)


async def _create_or_reclaim(
    client: Any, create: Callable[[], Any], name: str, pending: set[asyncio.Task]
) -> Any:
    task = asyncio.ensure_future(create())
    pending.add(task)
    task.add_done_callback(pending.discard)

    async def reclaim() -> None:
        try:
            await task
        except BaseException:
            pass
        await _reap(client, name)

    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        reaper = asyncio.create_task(reclaim())
        pending.add(reaper)
        reaper.add_done_callback(pending.discard)
        raise


class VercelSandboxProvider(SandboxProvider):
    """Ephemeral Docker-capable Vercel Sandboxes."""

    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.vercel.run",)

    def __init__(
        self,
        *,
        image: str = "vercel/sandbox/universal",
        region: str | None = None,
        failover_regions: list[str] | None = None,
        network_id: str | None = None,
        token: str | None = None,
        team_id: str | None = None,
        project_id: str | None = None,
        client_factory: Callable[[], Any] | None = None,
    ):
        self._image = image
        self._region = region
        self._failover_regions = failover_regions
        self._network_id = network_id
        self._token = token
        self._team_id = team_id
        self._project_id = project_id
        self._client_factory = client_factory or self._default_client_factory
        self._client: Any | None = None
        self._pending: set[asyncio.Task] = set()

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        section = "[sandbox.providers.vercel.config]"
        for key in ("image", "region", "network_id"):
            value = config.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{section} '{key}' must be a non-empty string when set")
        credentials = {key: config.get(key) for key in ("token", "team_id", "project_id")}
        if any(value is not None for value in credentials.values()) and not all(
            isinstance(value, str) and value.strip() for value in credentials.values()
        ):
            raise ConfigError(
                f"{section} requires token, team_id and project_id together; omit all three for SDK OIDC credentials"
            )
        failover = config.get("failover_regions")
        if failover is not None:
            if not isinstance(failover, list) or not all(
                isinstance(value, str) and value.strip() for value in failover
            ):
                raise ConfigError(f"{section} 'failover_regions' must be a list of non-empty strings")
            if len(failover) != len(set(failover)):
                raise ConfigError(f"{section} 'failover_regions' cannot contain duplicates")
            if config.get("region") in failover:
                raise ConfigError(f"{section} 'failover_regions' cannot include region")
        unknown = set(config) - {
            "image",
            "region",
            "failover_regions",
            "network_id",
            "token",
            "team_id",
            "project_id",
        }
        if unknown:
            raise ConfigError(f"{section} has unknown key(s): {sorted(unknown)}")
        return cls(**config)

    def _default_client_factory(self) -> Any:
        try:
            from vercel.sandbox import SandboxClient, SandboxServiceOptions
        except ImportError as error:
            raise RuntimeError(
                "The vercel sandbox provider requires vercel-sandbox; "
                "install agentenv-framework[vercel]"
            ) from error

        options_kwargs: dict[str, Any] = {"region": self._region}
        if self._token is None:
            return SandboxClient.create(options=SandboxServiceOptions(**options_kwargs))
        from vercel.sandbox import SandboxCredentials

        async def credentials() -> Any:
            return SandboxCredentials(
                token=self._token,
                team_id=self._team_id,
                project_id=self._project_id,
            )

        options_kwargs["credentials_factory"] = credentials
        return SandboxClient.create(options=SandboxServiceOptions(**options_kwargs))

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        return True

    @classmethod
    def shares_network_with(cls, sandbox_type: str | None) -> bool:
        return False

    @staticmethod
    def _validated_ports(exposed_ports: list[int] | None) -> list[int]:
        ports = list(dict.fromkeys(exposed_ports or []))
        if len(ports) > _MAX_PORTS:
            raise ValueError(f"Vercel exposes at most {_MAX_PORTS} distinct ports")
        if any(isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 for port in ports):
            raise ValueError("ports must be integers from 1 to 65535")
        return ports

    @staticmethod
    def _validated_timeout(timeout: int) -> int:
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 0 < timeout <= _MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout must be an integer from 1 to {_MAX_TIMEOUT_SECONDS} seconds")
        return timeout

    @staticmethod
    def _validated_disk(disk_size_gb: float) -> None:
        if (
            isinstance(disk_size_gb, bool)
            or not isinstance(disk_size_gb, (int, float))
            or not math.isfinite(disk_size_gb)
            or not 0 < disk_size_gb <= _MAX_DISK_GB
        ):
            raise ValueError(f"disk_size_gb must be a finite number greater than zero and at most {_MAX_DISK_GB}")

    @staticmethod
    def _tags(attribution: Attribution | None) -> dict[str, str]:
        tags = {
            key: str(value)
            for key, value in apply_default_attribution(dict(attribution or {})).items()
            if value is not None
        }
        if len(tags) > _MAX_TAGS:
            raise ValueError(f"Vercel accepts at most {_MAX_TAGS} attribution tags")
        if any(not key or not value for key, value in tags.items()):
            raise ValueError("attribution keys and values must be non-empty")
        return tags

    async def create_vm(
        self,
        *,
        image: str | None = None,
        boot_mode: str | None = None,
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        exposed_ports: list[int] | None = None,
        setup_for_gateway: bool = True,
        attribution: Attribution | None = None,
        network_policy: NetworkPolicy | None = None,
    ) -> VercelSandbox:
        if image is not None:
            raise ValueError("the Vercel outer image is configured by the provider; image overrides are unsupported")
        if boot_mode is not None:
            raise ValueError("Vercel custom kernel boot modes are unsupported")
        vcpus, memory_mb = resource_shape(cpu, memory)
        self._validated_disk(disk_size_gb)
        if vcpus != cpu or memory_mb != memory:
            logger.info(
                "Vercel resource shape adjusted: requested_cpu=%s requested_memory_mb=%s "
                "allocated_vcpus=%s allocated_memory_mb=%s",
                cpu,
                memory,
                vcpus,
                memory_mb,
            )
        if disk_size_gb != _MAX_DISK_GB:
            logger.warning(
                "Vercel sandbox disk size is fixed: requested_disk_gb=%s fixed_disk_gb=%s",
                disk_size_gb,
                _MAX_DISK_GB,
            )
        timeout = self._validated_timeout(timeout)
        ports = self._validated_ports(exposed_ports)
        tags = self._tags(attribution)
        effective_policy = self.effective_network_policy(network_policy)
        if not self.supports_network_policy(effective_policy):
            raise NetworkPolicyUnsupportedError(
                f"Vercel cannot enforce network policy {effective_policy.to_dict()}"
            )

        client = self._get_client()
        resources = _sdk_resources(vcpus, memory_mb)
        name = f"agentenv-{uuid.uuid4().hex[:12]}"

        async def create() -> Any:
            return await client.create_sandbox(
                name=name,
                image=self._image,
                ports=ports,
                execution_time_limit=timeout,
                resources=resources,
                persistent=False,
                network_policy=vercel_network_policy(effective_policy),
                network_id=self._network_id,
                region=self._region,
                tags=tags,
                failover_regions=self._failover_regions,
            )

        try:
            raw = await _create_or_reclaim(client, create, name, self._pending)
        except asyncio.CancelledError:
            raise
        except BaseException:
            await _reap(client, name)
            raise

        applied = network_policy_from_vercel(raw.network_policy)
        if applied is None:
            await _destroy_named(client, name)
            raise RuntimeError(
                f"Vercel sandbox {name} returned a network policy agent-env cannot represent"
            )
        sandbox = VercelSandbox(
            raw,
            client=client,
            tunnel_urls={route.port: route.url for route in raw.routes if route.port in ports},
            network_policy=applied,
        )
        missing = [port for port in ports if port not in sandbox.tunnel_urls]
        if missing:
            await sandbox.terminate()
            raise RuntimeError(f"Vercel sandbox {name} has no public route for port(s) {missing}")
        try:
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            logger.info(
                "Vercel sandbox started: name=%s image=%s vcpus=%s memory=%sMB ports=%s attribution=%s",
                name,
                self._image,
                vcpus,
                memory_mb,
                ports,
                tags,
            )
            return sandbox
        except BaseException:
            try:
                await sandbox.terminate()
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to destroy Vercel sandbox %s after setup failure: %s",
                    name,
                    cleanup_error,
                )
            raise

    async def create_sandbox(
        self,
        *,
        image_name: str,
        port: int,
        env: dict[str, str],
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        attribution: Attribution | None = None,
        network_policy: NetworkPolicy | None = None,
    ) -> VercelSandbox:
        del image_name, env
        return await self.create_vm(
            cpu=cpu,
            memory=memory,
            disk_size_gb=disk_size_gb,
            timeout=timeout,
            exposed_ports=[port],
            attribution=attribution,
            network_policy=network_policy,
        )

    async def get_sandbox(self, sandbox_id: str) -> VercelSandbox:
        raw = await self._get_client().get_sandbox(name=sandbox_id)
        applied = network_policy_from_vercel(raw.network_policy)
        if applied is None:
            logger.warning(
                "Vercel sandbox %s has a network policy agent-env can't represent; image loading will fail closed",
                sandbox_id,
            )
        return VercelSandbox(
            raw,
            client=self._get_client(),
            tunnel_urls={route.port: route.url for route in raw.routes},
            network_policy=applied,
        )

    async def close(self) -> None:
        while self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()
