"""Agent37 sandbox provider.

Each sandbox is one Agent37 VM instance (``type: "vm"`` on the ``agent37-ubuntu`` template): a Linux machine with its
own kernel and Docker running inside. The gateway and agent containers run under that Docker through the inherited
``VmSandbox`` helpers.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, ClassVar, Self

import httpx

from agent_env.attribution import Attribution
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.agent37.sandbox import (
    Agent37Api,
    Agent37Sandbox,
    applied_policy,
    public_urls,
)
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxProvider,
    apply_default_attribution,
    refuse_unenforceable_policy,
)

logger = logging.getLogger(__name__)

TEMPLATE = "agent37-ubuntu"
REGIONS = ("us", "eu", "ap-south")
NAME_PREFIX = "agentenv"
NAME_SUFFIX_LENGTH = 16
_SECTION = "[sandbox.providers.agent37.config]"
_CONFIG_KEYS = frozenset({"api_key", "region"})

# Agent37's VM shapes, (vCPU, GiB of memory, largest disk in GB), smallest first.
SHAPES = ((2, 4, 12), (4, 8, 20), (4, 16, 40), (4, 32, 80))
_cleanups: set[asyncio.Task] = set()


def instance_resources(cpu: float, memory_mb: int, disk_size_gb: float) -> dict[str, int]:
    """The smallest shape that covers the request, with that shape's largest disk. A request beyond the largest shape is
    refused, not shrunk."""
    for vcpu, memory_gib, disk_gb in SHAPES:
        if cpu <= vcpu and memory_mb <= memory_gib * 1024 and disk_size_gb <= disk_gb:
            return {"cpu": vcpu, "memory": memory_gib, "disk": disk_gb}
    vcpu, memory_gib, disk_gb = SHAPES[-1]
    raise ValueError(
        f"Agent37 cannot size a VM for cpu={cpu}, memory={memory_mb}MB, disk={disk_size_gb}GB "
        f"(at most {vcpu} vCPU, {memory_gib} GiB memory, {disk_gb} GB disk)"
    )


def instance_name() -> str:
    """``agentenv-<random>``, unique, so the instance a create lost its answer for can be found again."""
    return f"{NAME_PREFIX}-{uuid.uuid4().hex[:NAME_SUFFIX_LENGTH]}"


async def _settle(cleanup: Any) -> None:
    """Run ``cleanup`` to its end, also while the caller is being cancelled: ``asyncio.run`` waits for the task it
    cancels, so it waits for the cleanup too, which no Agent37 call holds past ``API_TIMEOUT_SECONDS``. A second
    cancellation leaves the cleanup running on its own."""
    task = asyncio.ensure_future(cleanup)
    _cleanups.add(task)
    task.add_done_callback(_cleanups.discard)
    await asyncio.shield(task)


class Agent37SandboxProvider(SandboxProvider):
    """Provision Docker-capable Agent37 VMs.

    ``api_key`` comes from resolved provider config, never from the process environment: configure it with an ``env:``
    or ``secret:`` reference in ``[sandbox.providers.agent37.config]``. Workloads never see it.
    """

    # Exposed ports get public URLs under this domain.
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.agent37.app",)
    WEBSOCKETS = True

    def __init__(self, *, api_key: str, region: str = "us", transport: httpx.AsyncBaseTransport | None = None):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Agent37 api_key must be a non-empty resolved secret")
        if region not in REGIONS:
            raise ValueError(f"Agent37 region must be one of {list(REGIONS)}, not {region!r}")
        self._region = region
        self._api = Agent37Api(api_key, transport=transport)

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            raise ConfigError(f"{_SECTION} has unknown key(s): {sorted(unknown)}")
        api_key = config.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigError(f"{_SECTION} requires a non-empty 'api_key'")
        if config.get("region", "us") not in REGIONS:
            raise ConfigError(f"{_SECTION} 'region' must be one of {list(REGIONS)}")
        return cls(**config)

    async def close(self) -> None:
        """Close this event loop's client; a sandbox this provider made reopens its connection if it is used again."""
        await self._api.close()

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
    ) -> Agent37Sandbox:
        """Create an Agent37 VM with a public URL for each of ``exposed_ports``. Agent37 has no delete-after timer, so
        ``timeout`` is ignored and the VM runs until it is terminated. VMs boot Agent37's Docker-capable Ubuntu image,
        so a per-call ``image`` is refused rather than ignored; ``boot_mode`` has no Agent37 equivalent and is
        ignored."""
        refuse_unenforceable_policy(self, network_policy)
        if image is not None:
            raise ValueError(f"Agent37 VMs boot the {TEMPLATE} image, which runs Docker; a per-call image is unsupported")
        if boot_mode is not None:
            logger.warning("Ignoring boot_mode=%s: Agent37 VMs have no boot modes", boot_mode)
        resources = instance_resources(cpu, memory, disk_size_gb)
        ports = list(dict.fromkeys(exposed_ports or []))
        metadata = {
            key: str(value)
            for key, value in apply_default_attribution(dict(attribution or {})).items()
            if value is not None
        }
        body = {
            "template": TEMPLATE,
            "type": "vm",
            "region": self._region,
            "resources": resources,
            "name": instance_name(),
            "metadata": metadata,
            "public_ports": [{"port": port} for port in ports],
        }
        instance = await self._create(body)
        sandbox = Agent37Sandbox(self._api, instance["id"], tunnel_urls=public_urls(instance),
                                 network_policy=NetworkPolicy())
        try:
            logger.info(
                "Agent37 instance started: id=%s name=%s region=%s resources=%s ports=%s attribution=%s "
                "(timeout=%ss not enforced)",
                sandbox.sandbox_id, body["name"], self._region, resources, ports, metadata, timeout,
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            return sandbox
        except BaseException:
            await _settle(self._delete_quietly(sandbox.sandbox_id))
            raise

    async def _create(self, body: dict[str, Any]) -> dict[str, Any]:
        """Await the create. When it fails or its caller is cancelled, the instance it may have made anyway is deleted
        once it settles, found by its unique name if its answer was lost."""
        task = asyncio.ensure_future(self._api.create_instance(body))
        try:
            return await asyncio.shield(task)
        except BaseException:
            await _settle(self._reclaim(task, body["name"]))
            raise

    async def _reclaim(self, create: asyncio.Future, name: str) -> None:
        try:
            instance_id = (await create)["id"]
        except BaseException:
            try:
                instance_id = next((i["id"] for i in await self._api.list_instances() if i.get("name") == name), None)
            except Exception as exc:  # noqa: BLE001 - logged; the instance, if any, is named in the message
                logger.error("Could not look for an Agent37 instance %s left by a failed create: %s", name, exc)
                return
        if instance_id is not None:
            await self._delete_quietly(instance_id)

    async def _delete_quietly(self, instance_id: str) -> None:
        try:
            await self._api.delete_instance(instance_id)
            logger.info("Deleted Agent37 instance %s after its create failed or was cancelled", instance_id)
        except Exception as exc:  # noqa: BLE001 - logged; the caller's own error is the one raised
            logger.error("Agent37 instance %s from a failed create is still running; delete it: %s", instance_id, exc)

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
    ) -> Agent37Sandbox:
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

    async def get_sandbox(self, sandbox_id: str) -> Agent37Sandbox:
        """Reconnect, restoring the public URLs and the applied egress policy."""
        instance = await self._api.get_instance(sandbox_id)
        return Agent37Sandbox(self._api, instance["id"], tunnel_urls=public_urls(instance),
                              network_policy=applied_policy(instance.get("network")))

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        await self._api.delete_instance(sandbox_id)
