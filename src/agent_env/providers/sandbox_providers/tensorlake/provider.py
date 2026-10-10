"""Tensorlake VM sandbox provider.

Each sandbox is an ephemeral Tensorlake MicroVM booted from a Docker-capable host image, by default
the public ``agentenv-dind-host-v1`` built from ``host_image.Dockerfile``.
The gateway and agent containers run inside it through the inherited ``VmSandbox`` helpers.
"""

from __future__ import annotations

import asyncio
import logging
import math
from ipaddress import ip_network
from pathlib import Path
from typing import Any, ClassVar, Self

from tensorlake.sandbox import AsyncSandbox, AsyncSandboxClient, RemoteAPIError, SandboxNotFoundError

from agent_env.attribution import PIPELINE_STEP_KEY, RUN_ID_KEY, Attribution
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import (
    Accepts,
    SandboxProvider,
    apply_default_attribution,
    refuse_unenforceable_policy,
)
from agent_env.providers.sandbox_providers.tensorlake.sandbox import (
    TensorlakeSandbox,
    network_config_fields,
    network_policy_from_config,
)

logger = logging.getLogger(__name__)

# Tensorlake hosts run without KVM, so cores are slower than the request suggests.
_MIN_CPUS = 2
_MIN_MEMORY_MB = 4 * 1024
_MIN_MEMORY_MB_PER_CPU = 1024
_MAX_MEMORY_MB_PER_CPU = 8 * 1024
# The disk the host image is published with; a smaller sandbox disk cannot hold it.
_MIN_DISK_MB = 30 * 1024
_MAX_DISK_MB = 100 * 1024
# The disk the image builder needs to build the host image.
_BUILDER_DISK_MB = 24 * 1024
# Passed to every SDK call: the SDK otherwise falls back to TENSORLAKE_API_URL in some calls
# and to the public endpoint in others, so creation and cleanup could hit different services.
DEFAULT_API_URL = "https://api.tensorlake.ai"
DEFAULT_IMAGE = "agentenv-dind-host-v1"
HOST_IMAGE_DOCKERFILE = Path(__file__).with_name("host_image.Dockerfile")
SANDBOX_STARTED_EVENT = "agent_env.tensorlake_sandbox_started"
_CONFIG_KEYS = ("api_key", "image", "api_url")

_reapers: set[asyncio.Task] = set()


class TensorlakeSandboxProvider(SandboxProvider):
    """Provision Docker-capable Tensorlake sandboxes.

    ``api_key`` comes from resolved provider config, never from the process environment:
    configure it with a ``secret:`` reference in ``[sandbox.providers.tensorlake.config]``.
    ``image`` names a registered Docker-capable image and defaults to the public host image.
    ``api_url`` defaults to the public Tensorlake API.
    """

    CREATES_VMS = True
    SANDBOX_ACCEPTS = Accepts.LOADABLE  # a sandbox is a VM, which loads the image; a container pulls it by name

    # Exposed ports are served at ``https://<port>-<sandbox-id>.sandbox.tensorlake.ai``.
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.sandbox.tensorlake.ai",)

    def __init__(
        self,
        *,
        api_key: str,
        image: str = DEFAULT_IMAGE,
        api_url: str = DEFAULT_API_URL,
        sandbox_cls: Any = AsyncSandbox,
        client_cls: Any = AsyncSandboxClient,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Tensorlake api_key must be a non-empty resolved secret")
        if not isinstance(image, str) or not image.strip():
            raise ValueError("Tensorlake image must be a non-empty registered image name")
        if not isinstance(api_url, str) or not api_url.strip():
            raise ValueError("Tensorlake api_url must be a non-empty URL")
        self._api_key = api_key
        self._image = image.strip()
        self._api_url = api_url.strip()
        # Injectable so unit tests never build an SDK client.
        self._sandbox_cls = sandbox_cls
        self._client_cls = client_cls

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        section = "[sandbox.providers.tensorlake.config]"
        unknown = set(config) - set(_CONFIG_KEYS)
        if unknown:
            raise ConfigError(f"{section} has unknown key(s): {sorted(unknown)}")
        if not config.get("api_key"):
            raise ConfigError(f"{section} requires a non-empty 'api_key'")
        for key in _CONFIG_KEYS:
            value = config.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{section} '{key}' must be a non-empty string")
        return cls(**config)

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        """Tensorlake filters IPv4 addresses, IPv4 CIDRs and hostnames; it has no IPv6 rules."""
        for entry in (*policy.allow_hosts, *policy.allow_cidrs):
            try:
                if ip_network(entry, strict=False).version == 6:
                    return False
            except ValueError:
                continue
        return True

    def _client_kwargs(self) -> dict[str, Any]:
        return {"api_key": self._api_key, "api_url": self._api_url}

    @staticmethod
    def _resources(cpu: float, memory: int, disk_size_gb: float) -> tuple[float, int, int]:
        """Raise a request to Tensorlake's floors, and reject memory it cannot give that many CPUs."""
        cpus = max(float(cpu), _MIN_CPUS)
        memory_mb = max(int(memory), _MIN_MEMORY_MB, math.ceil(cpus * _MIN_MEMORY_MB_PER_CPU))
        if memory_mb > cpus * _MAX_MEMORY_MB_PER_CPU:
            raise ValueError(
                f"Tensorlake allows at most {_MAX_MEMORY_MB_PER_CPU} MB of memory per CPU; "
                f"requested {memory_mb} MB with {cpus:g} CPUs"
            )
        disk_mb = max(math.ceil(float(disk_size_gb) * 1024), _MIN_DISK_MB)
        if disk_mb > _MAX_DISK_MB:
            raise ValueError(f"Tensorlake allows at most {_MAX_DISK_MB // 1024} GB of disk; requested {disk_size_gb} GB")
        if (cpus, memory_mb, disk_mb) != (cpu, memory, float(disk_size_gb) * 1024):
            logger.info(
                "Raised the Tensorlake sandbox request from cpu=%s, memory=%sMB, disk=%sGB "
                "to cpu=%s, memory=%sMB, disk=%sMB",
                cpu, memory, disk_size_gb, cpus, memory_mb, disk_mb,
            )
        return cpus, memory_mb, disk_mb

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
    ) -> TensorlakeSandbox:
        """Create an ephemeral Tensorlake sandbox and expose ``exposed_ports`` publicly.

        ``image`` overrides the configured image and must be a registered Docker-capable
        Tensorlake image. ``timeout`` becomes Tensorlake's idle timeout, after which the
        sandbox terminates. ``boot_mode`` has no Tensorlake equivalent and is ignored. Tensorlake
        sandboxes have no labels, so ``attribution`` goes into the started event, keyed by sandbox id.
        """
        refuse_unenforceable_policy(self, network_policy)
        if boot_mode is not None:
            logger.warning("Ignoring boot_mode=%s: Tensorlake sandboxes have no boot modes", boot_mode)
        cpus, memory_mb, disk_mb = self._resources(cpu, memory, disk_size_gb)
        effective_policy = self.effective_network_policy(network_policy)
        image_name = image or self._image
        ports = list(dict.fromkeys(exposed_ports or []))
        logger.info(
            "Creating Tensorlake sandbox (image=%s, ports=%s, cpus=%s, memory=%sMB, disk=%sMB, timeout=%ss)",
            image_name, ports, cpus, memory_mb, disk_mb, timeout,
        )
        # Without wait=False the SDK leaves a sandbox queued on timeout and its id is
        # unknown if the caller cancels, so a failed or abandoned start could leak it.
        resolved_attribution = {
            key: str(value) for key, value in apply_default_attribution(dict(attribution or {})).items() if value is not None
        }
        try:
            pending = await self._create_or_reclaim(self._sandbox_cls.create(
                image=image_name,
                cpus=cpus,
                memory_mb=memory_mb,
                disk_mb=disk_mb,
                timeout_secs=timeout,
                **network_config_fields(effective_policy),
                **self._client_kwargs(),
                wait=False,
            ))
        except RemoteAPIError as exc:
            if exc.status_code == 400 and "not registered" in str(exc):
                raise ValueError(self._unregistered_image_message(image_name)) from exc
            raise
        try:
            raw_sandbox = await pending.ready()
        except BaseException:
            await self._delete_sandbox(pending.sandbox_id)
            raise
        try:
            tunnel_urls: dict[int, str] = {}
            if ports:
                # Public ports match the other remote backends; a Bearer header would hand
                # the Tensorlake API key to every workload that calls the URL.
                info = await raw_sandbox.update(exposed_ports=ports, allow_unauthenticated_access=True)
                tunnel_urls = self._tunnel_urls(info, ports)
            sandbox = TensorlakeSandbox(raw_sandbox, tunnel_urls=tunnel_urls, network_policy=effective_policy)
            logger.info(
                "Tensorlake sandbox started: sandbox_id=%s image=%s attribution=%s",
                raw_sandbox.sandbox_id, image_name, resolved_attribution,
                extra={
                    "event": SANDBOX_STARTED_EVENT,
                    "tensorlake_sandbox_id": raw_sandbox.sandbox_id,
                    "tensorlake_attribution": resolved_attribution,
                    PIPELINE_STEP_KEY: resolved_attribution.get(PIPELINE_STEP_KEY),
                    RUN_ID_KEY: resolved_attribution.get(RUN_ID_KEY),
                    "cpus": cpus,
                    "memory_mb": memory_mb,
                    "disk_mb": disk_mb,
                },
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            return sandbox
        except BaseException:
            try:
                await raw_sandbox.terminate()
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup must not mask the create failure
                logger.warning(
                    "Failed to terminate Tensorlake sandbox %s after setup failure: %s",
                    raw_sandbox.sandbox_id, cleanup_error,
                )
            raise

    async def _create_or_reclaim(self, create: Any) -> Any:
        """Await a create. If the caller is cancelled first, wait for the create to settle and delete the
        sandbox it yields before the cancellation goes on: nothing else holds its id, and ``asyncio.run``
        cancels tasks still pending when it ends. Only a second cancellation leaves that to a callback."""
        task = asyncio.ensure_future(create)

        def delete_orphan(done: asyncio.Future) -> None:
            if done.cancelled() or done.exception() is not None:
                return
            reaper = asyncio.ensure_future(self._delete_sandbox(done.result().sandbox_id))
            _reapers.add(reaper)
            reaper.add_done_callback(_reapers.discard)

        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                created = await asyncio.shield(task)
            except asyncio.CancelledError:
                task.add_done_callback(delete_orphan)
                raise
            except Exception as exc:  # noqa: BLE001 - a create that failed left nothing to delete
                logger.debug("Tensorlake create cancelled by the caller then failed: %s", exc)
            else:
                await self._delete_sandbox(created.sandbox_id)
            raise

    def _unregistered_image_message(self, image_name: str) -> str:
        return (
            f"Tensorlake image {image_name!r} is not registered at {self._api_url}. Publish it once with:\n"
            f"  tl sbx image create {HOST_IMAGE_DOCKERFILE} -n {image_name} "
            f"--disk_mb {_MIN_DISK_MB} --builder_disk_mb {_BUILDER_DISK_MB} --docker_compat\n"
            "or set `image` in [sandbox.providers.tensorlake.config] to a registered Docker-capable image."
        )

    async def _delete_sandbox(self, sandbox_id: str) -> None:
        try:
            async with self._client_cls.for_cloud(**self._client_kwargs()) as client:
                await client.delete(sandbox_id)
        except SandboxNotFoundError:
            pass
        except Exception as cleanup_error:  # noqa: BLE001 - cleanup must not mask the create failure
            logger.warning("Failed to delete Tensorlake sandbox %s after start failure: %s", sandbox_id, cleanup_error)

    @staticmethod
    def _tunnel_urls(info: Any, ports: list[int]) -> dict[int, str]:
        urls: dict[int, str] = {}
        for port in ports:
            url = info.url_for_port(port)
            if url is None:
                raise RuntimeError(f"Tensorlake returned no ingress endpoint for port {port}")
            urls[port] = url
        return urls

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
    ) -> TensorlakeSandbox:
        # A VM backend: the gateway and agent paths start image_name inside the VM afterwards.
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

    async def get_sandbox(self, sandbox_id: str) -> TensorlakeSandbox:
        raw_sandbox = await self._sandbox_cls.connect(sandbox_id, **self._client_kwargs())
        info = await raw_sandbox.info()
        # A sandbox that isn't running has no ingress endpoint; teardown still has to reach it to terminate it.
        tunnel_urls = {
            port: url for port in info.exposed_ports or [] if (url := info.url_for_port(port)) is not None
        }
        sandbox = TensorlakeSandbox(raw_sandbox, tunnel_urls=tunnel_urls)
        try:
            sandbox.network_policy = network_policy_from_config(info.network_policy, sandbox_id=sandbox_id)
        except RuntimeError as exc:
            logger.warning(
                "Reconnected to Tensorlake sandbox %s, but image loading will fail closed: %s", sandbox_id, exc,
            )
        return sandbox
