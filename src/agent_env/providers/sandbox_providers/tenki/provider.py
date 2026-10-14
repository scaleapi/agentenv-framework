"""Tenki Sandbox VM sandbox provider.

Each sandbox is one Tenki session, a full Linux VM. The gateway and agent containers run under Docker inside it
through the inherited ``VmSandbox`` helpers; a VM booted from an image without Docker installs it first.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import math
import re
import uuid
import weakref
from datetime import UTC, datetime
from typing import Any, ClassVar, Self
from urllib.parse import urlparse

from tenki import AsyncClient, SessionNotFoundError

from agent_env.attribution import Attribution
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider, apply_default_attribution
from agent_env.providers.sandbox_providers.tenki.sandbox import ENDED_STATES, TenkiSandbox, close_session, policy_from_info

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.tenki.cloud"
SESSION_TAG = "agentenv"
_SECTION = "[sandbox.providers.tenki.config]"
_STRING_KEYS = ("api_key", "base_url", "image", "snapshot_id")
_CONFIG_KEYS = frozenset({*_STRING_KEYS, "min_disk_gb"})

# Tenki's API bounds; a workspace's own limit may be lower and is enforced at create.
# Tenki restores VMs of at least this size from a warm snapshot; a smaller one cold boots, which is slower.
MIN_CPUS = 2
MIN_MEMORY_MB = 4096
MAX_CPUS = 128
MAX_MEMORY_MB = 512 * 1024
MIN_DISK_GB = 5
MAX_DISK_GB = 100
# Docker keeps both the compressed and the extracted layers, so 10 GB runs out once a gateway and a few MCP server
# images are on the VM.
DEFAULT_MIN_DISK_GB = 20
MAX_ALLOW_DOMAINS = 256
MAX_ALLOW_CIDRS = 64

READY_TIMEOUT_SEC = 300
# Clock skew and rounding between this machine and Tenki, below which a shorter lifetime isn't a cap.
_LIFETIME_SLACK_SEC = 60
_REAP_ATTEMPTS = 3
_reapers: set[asyncio.Task] = set()

# A workspace can serve its ports under its own domain, two labels under the regional suffix
# (<token>.<workspace>.us.sb.tenki.sh). The parents seen on exposed ports are added to later allowlists, so a
# restricted agent reaches environments served there.
_preview_parents: set[str] = set()
# A process sees a handful of workspaces; the cap keeps the learned entries well inside MAX_ALLOW_DOMAINS.
_MAX_PREVIEW_PARENTS = 16


def vm_shape(cpu: float, memory_mb: int, disk_size_gb: float, *, min_disk_gb: int) -> tuple[int, int, int]:
    """``(cpu_cores, memory_mb, disk_size_gb)``: whole cores and memory rounded up to Tenki's 2 MiB alignment, raised to
    the warm-start floor, and a disk of at least ``min_disk_gb``. A request beyond Tenki's bounds is refused, not
    shrunk."""
    cores = max(MIN_CPUS, math.ceil(cpu))
    memory = max(MIN_MEMORY_MB, math.ceil(memory_mb / 2) * 2)
    disk = max(MIN_DISK_GB, min_disk_gb, math.ceil(disk_size_gb))
    if cores > MAX_CPUS or memory > MAX_MEMORY_MB or disk > MAX_DISK_GB:
        raise ValueError(
            f"Tenki cannot size a VM for cpu={cpu}, memory={memory_mb}MB, disk={disk_size_gb}GB "
            f"(at most {MAX_CPUS} vCPU, {MAX_MEMORY_MB} MB memory, {MAX_DISK_GB} GB disk)"
        )
    return cores, memory, disk


def _split_entries(policy: NetworkPolicy) -> tuple[list[str], list[str]]:
    """Tenki's ``allow_domains`` and ``allow_cidrs``, CIDRs normalized; an IP address listed as a host becomes a /32.
    A malformed CIDR raises ValueError."""
    domains: list[str] = []
    cidrs = [str(ipaddress.ip_network(cidr, strict=False)) for cidr in policy.allow_cidrs]
    for host in policy.allow_hosts:
        try:
            cidrs.append(str(ipaddress.ip_network(host, strict=False)))
        except ValueError:
            domains.append(host)
    return domains, cidrs


def egress_kwargs(policy: NetworkPolicy) -> dict[str, Any]:
    """``AsyncClient.create`` egress arguments for ``policy``. Tenki reads empty allowlists as unrestricted, so an
    allowlist with no entries turns outbound off instead."""
    if not policy.restricts_egress:
        return {}
    domains, cidrs = _split_entries(policy)
    if not domains and not cidrs:
        return {"allow_outbound": False}
    return {"allow_domains": domains or None, "allow_cidrs": cidrs or None}


def session_name(attribution: Attribution) -> str:
    """``agentenv-<random>`` and the attribution values, slugged into Tenki's 63-character name."""
    slugs = [re.sub(r"[^a-z0-9]+", "-", str(attribution[key]).lower()).strip("-") for key in sorted(attribution)]
    return "-".join(["agentenv", uuid.uuid4().hex[:8], *filter(None, slugs)])[:63].rstrip("-")


def _tunnel_url(url: str) -> str:
    url = url if url.startswith(("http://", "https://")) else f"https://{url}"
    parent = (urlparse(url).hostname or "").partition(".")[2]
    entry = f"*.{parent}"
    if (parent.endswith(".sb.tenki.sh") and entry not in TenkiSandboxProvider.EGRESS_HOSTS
            and len(_preview_parents) < _MAX_PREVIEW_PARENTS):
        _preview_parents.add(entry)
    return url


def _granted_lifetime(timeout_at: datetime | None, requested_at: datetime) -> float:
    """The lifetime Tenki gave a VM requested at ``requested_at``, boot included; infinite when it reports none."""
    if timeout_at is None:
        return math.inf
    if timeout_at.tzinfo is None:
        timeout_at = timeout_at.replace(tzinfo=UTC)
    return (timeout_at - requested_at).total_seconds()


def _remaining_lifetime(timeout_at: datetime | None) -> int:
    if timeout_at is None:
        return 2 * 3600
    if timeout_at.tzinfo is None:
        timeout_at = timeout_at.replace(tzinfo=UTC)
    return max(60, math.ceil((timeout_at - datetime.now(UTC)).total_seconds()))


async def _reap(sandbox: TenkiSandbox) -> None:
    """Terminate ``sandbox``, retrying. A cancellation, as ``asyncio.run`` sends every task left when it ends, is
    honored only after the terminate settles, so Ctrl-C during a create doesn't leave its VM running."""
    cancelled = False
    for attempt in range(_REAP_ATTEMPTS):
        try:
            await sandbox.terminate()
            logger.info("Terminated Tenki VM %s after its create failed or was cancelled", sandbox.sandbox_id)
            break
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:  # noqa: BLE001 - retried, then logged
            logger.warning("Terminating Tenki VM %s failed (attempt %s): %s", sandbox.sandbox_id, attempt + 1, exc)
            if attempt + 1 < _REAP_ATTEMPTS and not cancelled:
                try:
                    await asyncio.sleep(2**attempt)
                except asyncio.CancelledError:
                    cancelled = True
    else:
        logger.error("Tenki VM %s from a failed create is still running; it stops at its max lifetime",
                     sandbox.sandbox_id)
        with contextlib.suppress(Exception):
            await sandbox.release()
    if cancelled:
        raise asyncio.CancelledError


async def _reclaim_by_name(client: Any, name: str, policy: NetworkPolicy, timeout: int) -> None:
    """Terminate the VM a failed create may have made anyway, its response lost: the active session tagged
    ``agentenv`` with the create's unique name."""
    try:
        sessions = await client.list(tags=[SESSION_TAG])
    except Exception as exc:  # noqa: BLE001 - best effort; the VM still stops at its max lifetime
        logger.warning("Could not look for a Tenki VM %s left by a failed create: %s", name, exc)
        return
    for session in sessions:
        if session.info.name == name and session.info.state not in ENDED_STATES:
            await _reap(TenkiSandbox(session, tunnel_urls={}, network_policy=policy, command_timeout=timeout))


def _spawn(cleanup: Any) -> asyncio.Task:
    """Run ``cleanup`` as a task no caller's cancellation reaches, kept referenced until it ends."""
    task = asyncio.ensure_future(cleanup)
    _reapers.add(task)
    task.add_done_callback(_reapers.discard)
    return task


class TenkiSandboxProvider(SandboxProvider):
    """Provision Docker-capable Tenki VMs.

    ``api_key`` comes from resolved provider config, never from the process environment: configure it with an
    ``env:`` or ``secret:`` reference in ``[sandbox.providers.tenki.config]``. Workloads never see it.
    """

    # Exposed ports get public URLs under these domains. Tenki wildcards cover one label, hence the regional entry.
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.sb.tenki.sh", "*.us.sb.tenki.sh")

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        image: str | None = None,
        snapshot_id: str | None = None,
        min_disk_gb: int = DEFAULT_MIN_DISK_GB,
        client_cls: Any = AsyncClient,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Tenki api_key must be a non-empty resolved secret")
        if image and snapshot_id:
            raise ValueError("Pass at most one of image and snapshot_id")
        self._api_key = api_key
        self._base_url = base_url
        self._image = image
        self._snapshot_id = snapshot_id
        self._min_disk_gb = min_disk_gb
        # Injectable so unit tests never build an SDK client.
        self._client_cls = client_cls
        # One client per event loop, shared by every sandbox this provider creates or reconnects to: an SDK client
        # holds a gRPC channel bound to the loop it was made in, and the CLI runs several loops in one process.
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = weakref.WeakKeyDictionary()

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            raise ConfigError(f"{_SECTION} has unknown key(s): {sorted(unknown)}")
        if not config.get("api_key"):
            raise ConfigError(f"{_SECTION} requires a non-empty 'api_key'")
        for key in _STRING_KEYS:
            value = config.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{_SECTION} '{key}' must be a non-empty string")
        if config.get("image") and config.get("snapshot_id"):
            raise ConfigError(f"{_SECTION} takes at most one of 'image' and 'snapshot_id'")
        min_disk_gb = config.get("min_disk_gb", DEFAULT_MIN_DISK_GB)
        if isinstance(min_disk_gb, bool) or not isinstance(min_disk_gb, int) or not (
            MIN_DISK_GB <= min_disk_gb <= MAX_DISK_GB
        ):
            raise ConfigError(f"{_SECTION} 'min_disk_gb' must be an integer from {MIN_DISK_GB} to {MAX_DISK_GB}")
        return cls(**config)

    @classmethod
    def effective_network_policy(cls, policy: NetworkPolicy | None) -> NetworkPolicy:
        """Every platform's hosts, as on the base, and the workspace preview domains seen so far."""
        return super().effective_network_policy(policy).with_hosts(sorted(_preview_parents))

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        """Allow-all, or an allowlist of hostnames and IPv4 addresses or CIDRs within Tenki's limits."""
        if not policy.restricts_egress:
            return True
        try:
            domains, cidrs = _split_entries(policy)
        except ValueError:
            return False
        return (
            len(domains) <= MAX_ALLOW_DOMAINS
            and len(cidrs) <= MAX_ALLOW_CIDRS
            and all(ipaddress.ip_network(cidr, strict=False).version == 4 for cidr in cidrs)
        )

    def _client(self) -> Any:
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            client = self._clients[loop] = self._client_cls(api_key=self._api_key, base_url=self._base_url)
        return client

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
    ) -> TenkiSandbox:
        """Create a Tenki VM and expose ``exposed_ports`` publicly. ``timeout`` is the VM's hard maximum lifetime. The
        VM boots the configured image or snapshot, so a per-call ``image`` is refused rather than ignored;
        ``boot_mode`` has no Tenki equivalent and is ignored."""
        if image is not None:
            raise ValueError(
                "Tenki VMs boot the image or snapshot_id in [sandbox.providers.tenki.config]; "
                "a per-call image is unsupported"
            )
        if boot_mode is not None:
            logger.warning("Ignoring boot_mode=%s: Tenki VMs have no boot modes", boot_mode)
        cores, memory_mb, disk_gb = vm_shape(cpu, memory, disk_size_gb, min_disk_gb=self._min_disk_gb)
        effective_policy = self.effective_network_policy(network_policy)
        if not self.supports_network_policy(effective_policy):
            raise NetworkPolicyUnsupportedError(
                f"Tenki enforces allow-all or an allowlist of up to {MAX_ALLOW_DOMAINS} hostnames and "
                f"{MAX_ALLOW_CIDRS} IPv4 CIDRs, not {effective_policy.to_dict()}"
            )
        metadata = {
            key: str(value)
            for key, value in apply_default_attribution(dict(attribution or {})).items()
            if value is not None
        }
        ports = list(dict.fromkeys(exposed_ports or []))
        create_kwargs: dict[str, Any] = {
            **egress_kwargs(effective_policy),
            "name": session_name(metadata),
            "cpu_cores": cores,
            "memory_mb": memory_mb,
            "disk_size_gb": disk_gb,
            "max_duration": timeout,
            "metadata": metadata,
            "tags": [SESSION_TAG],
            "wait": False,
        }
        if self._image:
            create_kwargs["image"] = self._image
        if self._snapshot_id:
            create_kwargs["snapshot_id"] = self._snapshot_id

        client = self._client()
        requested_at = datetime.now(UTC)
        session = await self._create_or_reclaim(client, create_kwargs, effective_policy, timeout)
        sandbox = TenkiSandbox(session, tunnel_urls={}, network_policy=effective_policy, command_timeout=timeout)
        try:
            await session.wait_ready(timeout=READY_TIMEOUT_SEC)
            if (granted := _granted_lifetime(session.info.timeout_at, requested_at)) < timeout - _LIFETIME_SLACK_SEC:
                logger.warning(
                    "Tenki capped VM %s's lifetime to %ss of the %ss requested: the workspace's maximum session "
                    "duration is lower", session.id, granted, timeout,
                )
                sandbox.command_timeout = _remaining_lifetime(session.info.timeout_at)
            sandbox.tunnel_urls = await self._expose(session, ports)
            logger.info(
                "Tenki VM started: session_id=%s cpu=%s memory=%sMB disk=%sGB ports=%s attribution=%s",
                session.id, cores, memory_mb, disk_gb, ports, metadata,
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            return sandbox
        except BaseException as error:
            # The VM is terminated, with retries, whatever happens to this task; a failure waits for it and a
            # cancellation leaves it running. The create's own error is the one raised.
            reaper = _spawn(_reap(sandbox))
            if not isinstance(error, asyncio.CancelledError):
                # A cancellation while waiting propagates; the reaper keeps running on its own.
                await asyncio.shield(reaper)
            raise

    @staticmethod
    async def _create_or_reclaim(
        client: Any, create_kwargs: dict[str, Any], policy: NetworkPolicy, timeout: int
    ) -> Any:
        """Await the create. A VM that would otherwise run with no handle until its max lifetime is terminated: the
        one a create yields after its caller was cancelled, and the one a create that failed may have made anyway,
        its response lost, which is found by its unique name."""
        task = asyncio.ensure_future(client.create(**create_kwargs))

        def terminate_orphan(done: asyncio.Future) -> None:
            # A create cancelled itself, as the loop's shutdown cancels it, may have made its VM all the same.
            if done.cancelled() or done.exception() is not None:
                _spawn(_reclaim_by_name(client, create_kwargs["name"], policy, timeout))
            else:
                _spawn(_reap(TenkiSandbox(done.result(), tunnel_urls={}, network_policy=policy,
                                          command_timeout=timeout)))

        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(terminate_orphan)
            raise
        except Exception:
            await asyncio.shield(_spawn(_reclaim_by_name(client, create_kwargs["name"], policy, timeout)))
            raise

    @staticmethod
    async def _expose(session: Any, ports: list[int]) -> dict[int, str]:
        exposed = await asyncio.gather(*(session.expose_port(port) for port in ports))
        return {port: _tunnel_url(item.url) for port, item in zip(ports, exposed, strict=True)}

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
    ) -> TenkiSandbox:
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

    async def get_sandbox(self, sandbox_id: str) -> TenkiSandbox:
        """Reconnect, restoring the exposed ports and the applied egress policy."""
        session = await self._client().get(sandbox_id)
        exposed = await session.list_exposed_ports()
        info = session.info
        return TenkiSandbox(
            session,
            tunnel_urls={item.port: _tunnel_url(item.url) for item in exposed},
            network_policy=policy_from_info(info),
            command_timeout=_remaining_lifetime(info.timeout_at),
        )

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        try:
            session = await self._client().get(sandbox_id)
        except SessionNotFoundError:
            logger.info("Tenki session %s was already gone at terminate", sandbox_id)
            return
        await close_session(session)
