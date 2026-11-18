"""kosa8 Cloud sandbox provider.

Each sandbox is one kosa8 microVM, booted from an image that runs Docker (``docker: true``). The gateway and agent
containers run under that Docker through the inherited ``VmSandbox`` helpers.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
import re
import time
import uuid
from typing import Any, ClassVar, Self

import httpx

from agent_env.attribution import Attribution
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.kosa8.sandbox import (
    DEFAULT_BASE_URL,
    Kosa8Api,
    Kosa8Error,
    Kosa8Sandbox,
    policy_from_enforced,
)
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider, apply_default_attribution

logger = logging.getLogger(__name__)

# kosa8's universal Ubuntu 24.04 image: dockerd, containerd, the Docker CLI, buildx and the Compose v2 plugin.
DEFAULT_IMAGE = "kosa8-template/universal-24.04-20261115.2"
NAME_PREFIX = "agentenv"
_SECTION = "[sandbox.providers.kosa8.config]"
_CONFIG_KEYS = frozenset({"api_key", "base_url", "region", "image"})

# kosa8's sizes, (vCPU, GiB of memory), smallest first.
SIZES = ((2, 4), (4, 8), (8, 16), (16, 32))
DEFAULT_DISK_GB = 32
MAX_DISK_GB = 200
MIN_TTL_SEC = 60
MAX_TTL_SEC = 90 * 86400
MAX_EGRESS_ENTRIES = 256
READY_TIMEOUT_SEC = 600
POLL_SEC = (1.0, 5.0)
_FAILED_STATES = frozenset({"error", "deleted", "stopped", "paused", "host_removed", "host_lost"})
_NAME = re.compile(r"(?=.{1,253}$)[a-z0-9_-]{1,63}(\.[a-z0-9_-]{1,63})*")
_cleanups: set[asyncio.Task] = set()


def sandbox_size(cpu: float, memory_mb: int) -> str:
    """The smallest kosa8 size, ``<vcpu>x<GiB>``, that covers the request. A request beyond the largest is refused, not
    shrunk."""
    for vcpu, memory_gib in SIZES:
        if cpu <= vcpu and memory_mb <= memory_gib * 1024:
            return f"{vcpu}x{memory_gib}"
    vcpu, memory_gib = SIZES[-1]
    raise ValueError(f"kosa8 cannot size a sandbox for cpu={cpu}, memory={memory_mb}MB "
                     f"(at most {vcpu} vCPU, {memory_gib} GiB memory)")


def disk_gb(disk_size_gb: float) -> int:
    """The data disk for ``disk_size_gb``: at least kosa8's default, which holds the images Docker pulls."""
    wanted = max(DEFAULT_DISK_GB, math.ceil(disk_size_gb))
    if wanted > MAX_DISK_GB:
        raise ValueError(f"kosa8 cannot give a sandbox a {disk_size_gb} GB disk (at most {MAX_DISK_GB} GB)")
    return wanted


def lifetime(timeout: int) -> int:
    """``ttl_s`` for ``timeout``: kosa8 deletes the sandbox then. It takes 60 s to 90 days."""
    if timeout > MAX_TTL_SEC:
        raise ValueError(f"kosa8 cannot keep a sandbox for {timeout}s (at most {MAX_TTL_SEC}s)")
    return max(MIN_TTL_SEC, int(timeout))


def sandbox_name() -> str:
    """``agentenv-<random>``, unique, so the sandbox a create lost its answer for can be found again."""
    return f"{NAME_PREFIX}-{uuid.uuid4().hex[:16]}"


def egress_entries(policy: NetworkPolicy) -> list[str]:
    """The sandbox's egress allow list for ``policy``: its hosts, then its addresses and ranges."""
    return list(dict.fromkeys([*policy.allow_hosts, *policy.allow_cidrs]))


def _allowed_entry(entry: str) -> bool:
    """A hostname, a ``*.`` wildcard over at least two labels, or an IPv4 address or range, as kosa8 takes them."""
    entry = entry.strip().lower().rstrip(".")
    try:
        network = ipaddress.ip_network(entry, strict=False)
    except ValueError:
        if entry.startswith("*."):
            name = entry[2:]
            return "." in name and _NAME.fullmatch(name) is not None
        return _NAME.fullmatch(entry) is not None and not re.fullmatch(r"[0-9.]+", entry)
    return network.version == 4


async def _settle(cleanup: Any) -> None:
    """Run ``cleanup`` to its end, also while the caller is being cancelled: ``asyncio.run`` waits for the task it
    cancels, so it waits for the cleanup too. A second cancellation leaves the cleanup running on its own."""
    task = asyncio.ensure_future(cleanup)
    _cleanups.add(task)
    task.add_done_callback(_cleanups.discard)
    await asyncio.shield(task)


class Kosa8SandboxProvider(SandboxProvider):
    """Provision Docker-capable kosa8 microVMs.

    ``api_key`` comes from resolved provider config, never from the process environment: configure it with an ``env:``
    or ``secret:`` reference in ``[sandbox.providers.kosa8.config]``. Workloads never see it.
    """

    # Exposed ports get public preview URLs under this domain.
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.kosa8labs.com",)
    # kosa8's preview proxy passes a WebSocket upgrade through to the sandbox.
    WEBSOCKETS = True

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        region: str | None = None,
        image: str = DEFAULT_IMAGE,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("kosa8 api_key must be a non-empty resolved secret")
        self._region = region
        self._image = image
        self._api = Kosa8Api(api_key, base_url=base_url, transport=transport)

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            raise ConfigError(f"{_SECTION} has unknown key(s): {sorted(unknown)}")
        if not config.get("api_key"):
            raise ConfigError(f"{_SECTION} requires a non-empty 'api_key'")
        for key in _CONFIG_KEYS:
            value = config.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{_SECTION} '{key}' must be a non-empty string")
        return cls(**config)

    @classmethod
    def effective_network_policy(cls, policy: NetworkPolicy | None) -> NetworkPolicy:
        """Every platform's hosts, as on the base, and always kosa8's own, so a restricted sandbox still reaches the
        environments it talks to."""
        return super().effective_network_policy(policy).with_hosts(cls.EGRESS_HOSTS)

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        """Allow-all, or an allow list of at most 256 hostnames, ``*.`` wildcards and IPv4 addresses or ranges."""
        if not policy.restricts_egress:
            return True
        entries = (*policy.allow_hosts, *policy.allow_cidrs)
        return len(entries) <= MAX_EGRESS_ENTRIES and all(_allowed_entry(entry) for entry in entries)

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
        env: dict[str, str] | None = None,
    ) -> Kosa8Sandbox:
        """Create a kosa8 microVM that runs Docker, with a public URL for each of ``exposed_ports``. ``timeout`` is the
        sandbox's ``ttl_s``, its hard maximum lifetime. An allow list goes with the create, so it is in force before the
        VM answers. Sandboxes boot the configured image, so a per-call ``image`` is refused rather than ignored;
        ``boot_mode`` and ``env`` have no kosa8 equivalent here and are ignored."""
        if image is not None:
            raise ValueError("kosa8 sandboxes boot the configured Docker-capable image; a per-call image is unsupported")
        if boot_mode is not None:
            logger.warning("Ignoring boot_mode=%s: kosa8 sandboxes have no boot modes", boot_mode)
        if env:
            logger.warning("Ignoring env for a kosa8 VM: it boots no disk image to seed")
        size, disk, ttl = sandbox_size(cpu, memory), disk_gb(disk_size_gb), lifetime(timeout)
        effective_policy = self.effective_network_policy(network_policy)
        if not self.supports_network_policy(effective_policy):
            raise NetworkPolicyUnsupportedError(
                "kosa8 enforces allow-all or an allow list of at most 256 hostnames, wildcards and IPv4 addresses or "
                f"ranges, not {effective_policy.to_dict()}"
            )
        resolved_attribution = {
            key: str(value)
            for key, value in apply_default_attribution(dict(attribution or {})).items()
            if value is not None
        }
        ports = list(dict.fromkeys(exposed_ports or []))
        body: dict[str, Any] = {
            "image": self._image,
            "size": size,
            "disk_gb": disk,
            "docker": True,
            "sleep_after_s": 0,
            "ttl_s": ttl,
            "name": sandbox_name(),
        }
        if self._region:
            body["region"] = self._region
        if effective_policy.restricts_egress:
            body["egress"] = {"allow": egress_entries(effective_policy)}
        created = await self._create_with_policy(body)
        sandbox_id = created["id"]
        sandbox = Kosa8Sandbox(self._api, sandbox_id, tunnel_urls={}, network_policy=effective_policy)
        try:
            await self._wait_until_running(created)
            sandbox.network_policy = policy_from_enforced(await self._api.enforced_egress(sandbox_id))
            sandbox.tunnel_urls = {port: await self._api.open_port(sandbox_id, port) for port in ports}
            logger.info(
                "kosa8 sandbox started: id=%s name=%s size=%s disk_gb=%s ttl_s=%s ports=%s attribution=%s",
                sandbox_id, body["name"], size, disk, ttl, ports, resolved_attribution,
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            return sandbox
        except BaseException:
            # The sandbox is deleted before the create's own error or cancellation is raised.
            await _settle(self._delete_quietly(sandbox_id))
            raise

    async def _create_with_policy(self, body: dict[str, Any]) -> dict[str, Any]:
        """The create, with an allow list kosa8 refuses (an entry it can't parse, a host without egress controls, an
        org policy that doesn't allow it) raised as a policy it can't enforce, so a fallback chain moves on."""
        try:
            return await self._create(body)
        except Kosa8Error as exc:
            if "egress" not in body or ("egress" not in exc.code and exc.status_code != 501):
                raise
            raise NetworkPolicyUnsupportedError(f"kosa8 refused the egress allow list {body['egress']['allow']}: "
                                                f"{exc}") from exc

    async def _create(self, body: dict[str, Any]) -> dict[str, Any]:
        """Await the create. When it fails or its caller is cancelled, the sandbox it may have made anyway is deleted
        once it settles, found by its unique name if its answer was lost."""
        task = asyncio.ensure_future(self._api.create_sandbox(body))
        try:
            return await asyncio.shield(task)
        except BaseException:
            await _settle(self._reclaim(task, body["name"]))
            raise

    async def _reclaim(self, create: asyncio.Future, name: str) -> None:
        try:
            ids = [(await create)["id"]]
        except BaseException:
            try:
                ids = [sandbox["id"] for sandbox in await self._api.sandboxes_named(name)]
            except Exception as exc:  # noqa: BLE001 - logged; its ttl_s still removes the sandbox
                logger.error("Could not look for a kosa8 sandbox %s left by a failed create: %s", name, exc)
                return
        for sandbox_id in ids:
            await self._delete_quietly(sandbox_id)

    async def _delete_quietly(self, sandbox_id: str) -> None:
        try:
            await self._api.delete_sandbox(sandbox_id)
            logger.info("Deleted kosa8 sandbox %s after its create failed or was cancelled", sandbox_id)
        except Exception as exc:  # noqa: BLE001 - logged; the caller's own error is the one raised
            logger.error("kosa8 sandbox %s from a failed create is still running until its ttl_s: %s", sandbox_id, exc)

    async def _wait_until_running(self, sandbox: dict[str, Any]) -> None:
        """A sized create answers while the VM boots (state ``creating``); poll until it runs."""
        deadline = time.monotonic() + READY_TIMEOUT_SEC
        interval, longest = POLL_SEC
        while sandbox.get("state") != "running":
            if sandbox.get("state") in _FAILED_STATES:
                raise RuntimeError(f"kosa8 sandbox {sandbox['id']} did not start ({sandbox.get('state')}): "
                                   f"{sandbox.get('last_error') or 'no reason given'}")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"kosa8 sandbox {sandbox['id']} still {sandbox.get('state')} "
                                   f"after {READY_TIMEOUT_SEC}s")
            await asyncio.sleep(interval)
            interval = min(interval * 1.5, longest)
            sandbox = await self._api.get_sandbox(sandbox["id"])

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
    ) -> Kosa8Sandbox:
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

    async def get_sandbox(self, sandbox_id: str) -> Kosa8Sandbox:
        """Reconnect, restoring the public URLs and the applied egress policy."""
        sandbox = await self._api.get_sandbox(sandbox_id)
        return Kosa8Sandbox(
            self._api,
            sandbox["id"],
            tunnel_urls=await self._api.open_ports(sandbox["id"]),
            network_policy=policy_from_enforced(await self._api.enforced_egress(sandbox["id"])),
        )

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        await self._api.delete_sandbox(sandbox_id)
