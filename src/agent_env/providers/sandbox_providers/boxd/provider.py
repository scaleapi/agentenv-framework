"""boxd machine sandbox provider.

Each sandbox is one boxd machine, a hardware-isolated Linux VM. The gateway and agent containers run under Docker
inside it through the inherited ``VmSandbox`` helpers.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import uuid
import weakref
from typing import Any, ClassVar, Self

from boxd import AsyncBoxd, BoxdError, ConflictError, NotFoundError, ProxyEntry

from agent_env.attribution import Attribution
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.boxd.sandbox import (
    READY_TIMEOUT_SEC,
    BoxdSandbox,
    delete_machine,
    policy_from_entries,
)
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxNotFoundError,
    SandboxProvider,
    apply_default_attribution,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://boxd.sh:9443"
NAME_PREFIX = "agentenv"
_SECTION = "[sandbox.providers.boxd.config]"
_CONFIG_KEYS = frozenset({"api_key", "org", "base_url", "snapshot"})

# boxd's machine sizes, (vCPU, GiB of memory), smallest first; every machine has a fixed 100 GB disk.
SIZES = ((1, 4), (2, 8), (4, 16), (8, 32))
DISK_GB = 100
# boxd refuses allowlist addresses in or overlapping these ranges.
_REFUSED_RANGES = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("10.0.0.0/8", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12", "192.168.0.0/16", "224.0.0.0/4")
)
_PROXY_NAME = re.compile(r"p(\d+)")
_REAP_ATTEMPTS = 3
_reapers: set[asyncio.Task] = set()


def machine_size(cpu: float, memory_mb: int, disk_size_gb: float) -> tuple[int, str]:
    """``(vcpu, memory)`` of the smallest boxd size that covers the request. A request beyond the largest size or the
    fixed disk is refused, not shrunk."""
    if disk_size_gb <= DISK_GB:
        for vcpu, memory_gib in SIZES:
            if cpu <= vcpu and memory_mb <= memory_gib * 1024:
                return vcpu, f"{memory_gib}G"
    vcpu, memory_gib = SIZES[-1]
    raise ValueError(
        f"boxd cannot size a machine for cpu={cpu}, memory={memory_mb}MB, disk={disk_size_gb}GB "
        f"(at most {vcpu} vCPU, {memory_gib} GiB memory, {DISK_GB} GB disk)"
    )


def egress_entries(policy: NetworkPolicy) -> list[str]:
    """The machine's egress allowlist for ``policy``: its hosts, then its addresses and ranges."""
    return list(dict.fromkeys([*policy.allow_hosts, *policy.allow_cidrs]))


def machine_name() -> str:
    """``agentenv-<random>``. A machine's name is its public DNS name, so attribution stays out of it."""
    return f"{NAME_PREFIX}-{uuid.uuid4().hex[:16]}"


def _allowed_entry(entry: str) -> bool:
    """A hostname, a ``*.`` wildcard or a public IPv4 address or range, each bare, as boxd's allowlist takes them."""
    try:
        network = ipaddress.ip_network(entry, strict=False)
    except ValueError:
        name = entry.removeprefix("*.")
        return "." in name and not re.search(r"[^A-Za-z0-9.-]", name)
    return network.version == 4 and not any(network.overlaps(refused) for refused in _REFUSED_RANGES)


async def _reap(client: Any, machine_id: str) -> None:
    """Delete ``machine_id``, retrying. A cancellation, as ``asyncio.run`` sends every task left when it ends, is
    honored only after the delete settles, so Ctrl-C during a create doesn't leave its machine running."""
    cancelled = False
    for attempt in range(_REAP_ATTEMPTS):
        try:
            await delete_machine(client, machine_id)
            logger.info("Deleted boxd machine %s after its create failed or was cancelled", machine_id)
            break
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:  # noqa: BLE001 - retried, then logged
            logger.warning("Deleting boxd machine %s failed (attempt %s): %s", machine_id, attempt + 1, exc)
            if attempt + 1 < _REAP_ATTEMPTS and not cancelled:
                try:
                    await asyncio.sleep(2**attempt)
                except asyncio.CancelledError:
                    cancelled = True
    else:
        logger.error("boxd machine %s from a failed create is still running; its delete-after timer removes it",
                     machine_id)
    if cancelled:
        raise asyncio.CancelledError


async def _find_by_name(client: Any, org: str | None, name: str) -> Any:
    for machine in await client.machines.list(org=org):
        if machine.name == name:
            return machine
    return None


async def _reclaim_by_name(client: Any, org: str | None, name: str) -> None:
    """Delete the machine a failed create may have made anyway, its response lost, found by its unique name."""
    try:
        machine = await _find_by_name(client, org, name)
    except Exception as exc:  # noqa: BLE001 - best effort; the delete-after timer still removes it
        logger.warning("Could not look for a boxd machine %s left by a failed create: %s", name, exc)
        return
    if machine is not None:
        await _reap(client, machine.id)


async def _delete_outcome(create: asyncio.Future, client: Any, org: str | None, name: str) -> None:
    """Delete what a create its caller gave up on made, once it answers: the machine it returns, or the one it may
    have made though it failed or was cancelled."""
    try:
        machine = await create
    except (Exception, asyncio.CancelledError):
        await _reclaim_by_name(client, org, name)
        return
    await _reap(client, machine.id)


def _spawn(cleanup: Any) -> asyncio.Task:
    """Run ``cleanup`` as a task no caller's cancellation reaches, kept referenced until it ends."""
    task = asyncio.ensure_future(cleanup)
    _reapers.add(task)
    task.add_done_callback(_reapers.discard)
    return task


async def _settle(cleanup: Any) -> None:
    """Run ``cleanup`` to its end, also while the caller is being cancelled: ``asyncio.run`` waits for the task it
    cancels, so it waits for the cleanup too. A second cancellation leaves the cleanup running on its own."""
    await asyncio.shield(_spawn(cleanup))


async def _tunnel_urls(client: Any, machine_id: str) -> dict[int, str]:
    """The public URL of each port exposed through a ``p<port>`` proxy."""
    urls: dict[int, str] = {}
    for proxy in await client.machines.proxies.list(machine_id):
        if proxy.name and (match := _PROXY_NAME.fullmatch(proxy.name)):
            urls[int(match.group(1))] = f"https://{proxy.domain}"
    return urls


async def _expose(client: Any, machine_id: str, ports: list[int]) -> dict[int, str]:
    """Each port's public URL. A machine restored from a snapshot starts with the snapshot's proxies instead of the
    ones asked for, so those for other ports are removed, and the missing ones added."""
    urls = await _tunnel_urls(client, machine_id)
    extra = [port for port in urls if port not in ports]
    absent = [port for port in ports if port not in urls]
    for port in extra:
        await client.machines.proxies.delete(machine_id, f"p{port}")
    for port in absent:
        await client.machines.proxies.create(machine_id, f"p{port}", port)
    if extra or absent:
        urls = await _tunnel_urls(client, machine_id)
    if missing := [port for port in ports if port not in urls]:
        raise RuntimeError(f"boxd machine {machine_id} has no proxy for port(s) {missing}")
    return {port: urls[port] for port in ports}


class BoxdSandboxProvider(SandboxProvider):
    """Provision Docker-capable boxd machines, isolated, so a workload can't reach the account's other machines.

    ``api_key`` comes from resolved provider config, never from the process environment: configure it with an
    ``env:`` or ``secret:`` reference in ``[sandbox.providers.boxd.config]``. Workloads never see it.
    """

    # Exposed ports get public URLs under this domain.
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.boxd.sh",)

    def __init__(
        self,
        *,
        api_key: str,
        org: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        snapshot: str | None = None,
        client_cls: Any = AsyncBoxd,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("boxd api_key must be a non-empty resolved secret")
        self._api_key = api_key
        self._org = org
        self._base_url = base_url
        self._snapshot = snapshot
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
        for key in _CONFIG_KEYS:
            value = config.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{_SECTION} '{key}' must be a non-empty string")
        return cls(**config)

    @classmethod
    def effective_network_policy(cls, policy: NetworkPolicy | None) -> NetworkPolicy:
        """Every platform's hosts, as on the base, and always boxd's own, so an allowlist is never empty: boxd reads an
        empty one as unrestricted."""
        return super().effective_network_policy(policy).with_hosts(cls.EGRESS_HOSTS)

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        """Allow-all, or an allowlist of hostnames, ``*.`` wildcards and public IPv4 addresses or ranges."""
        if not policy.restricts_egress:
            return True
        return all(_allowed_entry(entry) for entry in (*policy.allow_hosts, *policy.allow_cidrs))

    def _client(self) -> Any:
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            client = self._clients[loop] = self._client_cls(api_key=self._api_key, base_url=self._base_url)
        return client

    async def close(self) -> None:
        """Close this event loop's client; a sandbox this provider made reopens its connection if it is used again."""
        client = self._clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.close()

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
    ) -> BoxdSandbox:
        """Create an isolated boxd machine and expose ``exposed_ports`` publicly. ``timeout`` is the machine's
        delete-after timer, its hard maximum lifetime. Machines boot boxd's image, which runs Docker, or the configured
        ``snapshot``, so a per-call ``image`` is refused rather than ignored; ``boot_mode`` has no boxd equivalent and is
        ignored."""
        if image is not None:
            raise ValueError("boxd machines boot boxd's Docker-capable image; a per-call image is unsupported")
        if boot_mode is not None:
            logger.warning("Ignoring boot_mode=%s: boxd machines have no boot modes", boot_mode)
        vcpu, memory_size = machine_size(cpu, memory, disk_size_gb)
        effective_policy = self.effective_network_policy(network_policy)
        allowlist = egress_entries(effective_policy)
        if not self.supports_network_policy(effective_policy):
            raise NetworkPolicyUnsupportedError(
                "boxd enforces allow-all or an allowlist of hostnames, wildcards and public IPv4 addresses or ranges, "
                f"not {effective_policy.to_dict()}"
            )
        resolved_attribution = {
            key: str(value)
            for key, value in apply_default_attribution(dict(attribution or {})).items()
            if value is not None
        }
        ports = list(dict.fromkeys(exposed_ports or []))
        create_kwargs: dict[str, Any] = {
            "name": machine_name(),
            "org": self._org,
            "isolated": True,
            "delete_after": timeout,
            "auto_suspend_timeout": 0,
        }
        client = self._client()
        if self._snapshot:
            vcpu, memory_size = await self._snapshot_size(client, cpu, memory)
            create_kwargs["from_snapshot"] = self._snapshot
        else:
            create_kwargs.update(vcpu=vcpu, memory=memory_size,
                                 proxies=[ProxyEntry(name=f"p{port}", port=port) for port in ports])

        machine = await self._create_or_reclaim(client, create_kwargs, self._org)
        sandbox = BoxdSandbox(client, machine.id, tunnel_urls={}, network_policy=effective_policy)
        try:
            if effective_policy.restricts_egress:
                await self._set_allowlist(client, machine.id, allowlist)
            sandbox.tunnel_urls = await _expose(client, machine.id, ports)
            logger.info(
                "boxd machine started: machine_id=%s name=%s vcpu=%s memory=%s ports=%s attribution=%s",
                machine.id, machine.name, vcpu, memory_size, ports, resolved_attribution,
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            else:
                await client.machines.wait_until_ready(machine.id, timeout=READY_TIMEOUT_SEC)
            return sandbox
        except BaseException:
            # The machine is deleted, with retries, before the create's own error or cancellation is raised.
            await _settle(_reap(client, machine.id))
            raise

    async def _snapshot_size(self, client: Any, cpu: float, memory_mb: int) -> tuple[int, str]:
        """``(vcpu, memory)`` of the machines the snapshot restores, the size it was saved at. A request bigger than
        that is refused, since a restored machine can't be resized as it starts."""
        snapshot = await client.snapshots.get(self._snapshot, org=self._org)
        memory_gib = snapshot.memory_bytes >> 30
        if cpu > snapshot.vcpu or memory_mb > memory_gib * 1024:
            raise ValueError(
                f"boxd snapshot {self._snapshot!r} restores machines of {snapshot.vcpu} vCPU / {memory_gib} GiB, less "
                f"than the cpu={cpu}, memory={memory_mb}MB requested; save it from a machine at least that size"
            )
        return snapshot.vcpu, f"{memory_gib}G"

    @staticmethod
    async def _set_allowlist(client: Any, machine_id: str, allowlist: list[str]) -> None:
        """Apply the allowlist from outside the machine, before anything runs in it; one boxd refuses (a wildcard on a
        shared-hosting domain, say) is a policy it can't enforce, so a fallback chain moves on."""
        try:
            await client.machines.set_egress_allow(machine_id, allowlist)
        except BoxdError as exc:
            if exc.code != "invalid_argument":
                raise
            raise NetworkPolicyUnsupportedError(f"boxd refused the egress allowlist {allowlist}: {exc}") from exc

    @staticmethod
    async def _create_or_reclaim(client: Any, create_kwargs: dict[str, Any], org: str | None) -> Any:
        """Await the create. A machine that would otherwise run with no handle until its delete-after timer is
        deleted: the one a create yields after its caller was cancelled, and the one a create that failed may have made
        anyway, its response lost, which is found by its unique name. The SDK retries a create that lost its
        connection, so one refused because its name is taken can be this create's own machine, which is kept."""
        name = create_kwargs["name"]
        task = asyncio.ensure_future(client.machines.create(**create_kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await _settle(_delete_outcome(task, client, org, name))
            raise
        except ConflictError:
            if (machine := await _find_by_name(client, org, name)) is not None:
                return machine
            raise
        except Exception:
            await _settle(_reclaim_by_name(client, org, name))
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
    ) -> BoxdSandbox:
        # A VM backend: the gateway and agent paths start image_name inside the machine afterwards.
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

    async def get_sandbox(self, sandbox_id: str) -> BoxdSandbox:
        """Reconnect, restoring the exposed ports and the applied egress allowlist."""
        client = self._client()
        try:
            machine = await client.machines.get(sandbox_id)
        except NotFoundError as exc:
            raise SandboxNotFoundError(f"boxd machine {sandbox_id!r} was not found") from exc
        return BoxdSandbox(
            client,
            machine.id,
            tunnel_urls=await _tunnel_urls(client, machine.id),
            network_policy=policy_from_entries(machine.egress_allow),
        )

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        await delete_machine(self._client(), sandbox_id)
