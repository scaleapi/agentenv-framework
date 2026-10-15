"""Run AgentEnv workloads in local SmolVM machines."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
import shlex
import socket
import uuid
from urllib.parse import urlsplit
from typing import ClassVar

try:
    from smol import ConnectOptions, Machine, MachineConfig, PortSpec, ResourceSpec
except ModuleNotFoundError as exc:
    if exc.name != "smol":
        raise
    raise ImportError("SmolVM requires pip install 'agentenv-framework[smol]'") from exc

from agent_env.attribution import Attribution
from agent_env.config import get_config
from agent_env.store.image_store.local_registry_image_store import LocalRegistryImageStore
from agent_env.store.routing import LocalRunImageStore
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.local_sandbox import LOCAL_TRUST_ENV, LocalSandboxProvider, local_grant_trust, start_trusting
from agent_env.providers.sandbox_providers.sandbox_provider import (
    Accepts, SandboxProvider, refuse_unenforceable_policy,
)
from agent_env.providers.sandbox_providers.smol_vm.sandbox import SmolVmSandbox

_DUMMY_PORT = 60832
logger = logging.getLogger(__name__)
_reapers: set[asyncio.Task] = set()


async def _delete_machine(machine: Machine) -> None:
    try:
        await asyncio.to_thread(machine.delete)
    except Exception:
        logger.exception("Could not delete SmolVM machine %s after failed creation", machine.id)


async def _delete_when_created(creation: asyncio.Task) -> None:
    try:
        machine = await asyncio.shield(creation)
    except Exception:  # native creation failed and left no machine to delete
        return
    await _delete_machine(machine)


def _start_reaper(coroutine) -> asyncio.Task:
    # Retain the task until cleanup ends, even if a caller is cancelled again.
    task = asyncio.create_task(coroutine)
    _reapers.add(task)
    task.add_done_callback(_reapers.discard)
    return task


def _host_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class SmolVmSandboxProvider(SandboxProvider):
    _DOCKER_READY_ATTEMPTS: ClassVar[int] = 40
    _LOCAL_REGISTRY_DEFAULT_PORT: ClassVar[int] = 5000
    ON_THIS_MACHINE: ClassVar[bool] = True
    REACHES_HOST_AT: ClassVar[str] = "127.0.0.1"
    CREATES_VMS: ClassVar[bool] = True
    SANDBOX_ACCEPTS: ClassVar[Accepts] = Accepts.LOADABLE
    CONTAINER_ACCEPTS: ClassVar[Accepts] = Accepts.NAME

    @classmethod
    def shares_network_with(cls, sandbox_type: str | None) -> bool:
        return False

    @classmethod
    def get_external_url(cls, url: str) -> str:
        return LocalSandboxProvider.get_external_url(url)

    def url_from_sandbox(self, url: str) -> str:
        return LocalSandboxProvider.get_external_url(url)

    async def create_sandbox(self, *, image_name: str, port: int, env: dict[str, str],
                             cpu: float = 1.0, memory: int = 8192, disk_size_gb: float = 10,
                             timeout: int = 7200, attribution: Attribution | None = None,
                             network_policy: NetworkPolicy | None = None) -> SmolVmSandbox:
        return await self.create_vm(
            exposed_ports=[port], cpu=cpu, memory=memory, disk_size_gb=disk_size_gb,
            timeout=timeout, attribution=attribution, network_policy=network_policy,
        )

    async def create_vm(self, *, image: str | None = None, boot_mode: str | None = None,
                        cpu: float = 1.0, memory: int = 8192, disk_size_gb: float = 10,
                        timeout: int = 7200, exposed_ports: list[int] | None = None,
                        setup_for_gateway: bool = True, attribution: Attribution | None = None,
                        network_policy: NetworkPolicy | None = None) -> SmolVmSandbox:
        refuse_unenforceable_policy(self, network_policy)
        if image is not None or boot_mode is not None:
            raise ValueError("smol_vm provisions its own Docker-capable guest; image and boot_mode overrides are unsupported")
        if cpu < 1 or memory <= 0 or disk_size_gb <= 0:
            raise ValueError("CPU, memory, and storage must be positive")
        name = f"agentenv-{uuid.uuid4().hex[:12]}"
        ports = {port: _host_port() for port in dict.fromkeys(exposed_ports or [])}
        # Even a portless VM needs a published mapping to select routed virtio-net for nested Docker.
        published = ports or {_DUMMY_PORT: _host_port()}
        config = MachineConfig(
            name=name, persistent=True, wait_for_ports=False, ttl_seconds=timeout,
            resources=ResourceSpec(cpus=math.ceil(cpu), memory_mb=memory, storage_gb=math.ceil(disk_size_gb), network=True),
            ports=[PortSpec(host=host, guest=guest) for guest, host in published.items()],
        )
        creation = asyncio.create_task(asyncio.to_thread(Machine.create, config, ConnectOptions(target="local")))
        try:
            machine = await asyncio.shield(creation)
        except asyncio.CancelledError:
            # Native creation continues after cancellation. The independent reaper also
            # survives a second cancellation while waiting for the machine to appear.
            await asyncio.shield(_start_reaper(_delete_when_created(creation)))
            raise
        try:
            sandbox = SmolVmSandbox(machine, ports)
            await self._setup_docker(sandbox)
            await sandbox.detect_platform()
            await self._configure_host_access(sandbox)
            await sandbox.persist_ports()
            sandbox.network_policy = self.effective_network_policy(network_policy)
            return sandbox
        except BaseException:
            await asyncio.shield(_start_reaper(_delete_machine(machine)))
            raise

    async def _setup_docker(self, sandbox: SmolVmSandbox) -> None:
        code, stdout, stderr = await sandbox.exec_with_output(
            "sh", "-c", "command -v bash >/dev/null && command -v docker >/dev/null && "
            "command -v curl >/dev/null && command -v python3 >/dev/null && command -v unzip >/dev/null && command -v socat >/dev/null "
            "|| apk add --no-cache docker docker-cli-compose bash curl python3 unzip socat",
        )
        if code:
            raise RuntimeError(f"Could not install guest Docker dependencies ({code}): {stdout[-500:]} {stderr[-500:]}")
        await sandbox.exec_script(f"""
set -e
mkdir -p /storage/docker /storage/containerd /var/lib/docker /var/lib/containerd
mountpoint -q /var/lib/docker || mount --bind /storage/docker /var/lib/docker
mountpoint -q /var/lib/containerd || mount --bind /storage/containerd /var/lib/containerd
if ! docker info >/dev/null 2>&1; then
    rm -f /var/run/docker.pid /run/containerd/containerd.pid
    dockerd --storage-driver=overlay2 >/tmp/agentenv-dockerd.log 2>&1 &
    for i in $(seq 1 {self._DOCKER_READY_ATTEMPTS}); do docker info >/dev/null 2>&1 && exit 0; sleep 1; done
    tail -80 /tmp/agentenv-dockerd.log
    exit 1
fi
""")
        await self._ensure_local_registry_proxy(sandbox)

    async def _ensure_local_registry_proxy(self, sandbox: SmolVmSandbox) -> None:
        store = get_config().get_image_store()
        stores = (store.configured, store.local) if isinstance(store, LocalRunImageStore) else (store,)
        ports = set()
        for candidate in stores:
            if isinstance(candidate, LocalRegistryImageStore):
                registry = urlsplit(f"//{candidate.registry_host}")
                if registry.hostname in ("localhost", "127.0.0.1"):
                    ports.add(registry.port or self._LOCAL_REGISTRY_DEFAULT_PORT)
        for port in sorted(ports):
            pid_file = f"/run/agentenv-registry-proxy-{port}.pid"
            await sandbox.exec_script(f"""
if ! test -f {pid_file} || ! kill -0 "$(cat {pid_file})" 2>/dev/null; then
    socat TCP-LISTEN:{port},bind=127.0.0.1,reuseaddr,fork TCP:host.smolvm.internal:{port} \
        >/tmp/agentenv-registry-proxy-{port}.log 2>&1 &
    echo $! > {pid_file}
fi
sleep 0.1
kill -0 "$(cat {pid_file})"
""")

    async def get_sandbox(self, sandbox_id: str) -> SmolVmSandbox:
        sandbox = await SmolVmSandbox.reconnect(sandbox_id)
        await self._setup_docker(sandbox)
        await self._configure_host_access(sandbox)
        sandbox.network_policy = self.effective_network_policy(None)
        return sandbox

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        """Connecting starts a stopped machine, and the SDK deletes only a connected one, so a listed machine is
        connected first; one not listed is gone."""
        conn = ConnectOptions(target="local", wait_for_ports=False)
        if sandbox_id not in {machine.name for machine in await asyncio.to_thread(Machine.list, conn)}:
            logger.info("SmolVM machine %s was already gone at terminate", sandbox_id)
            return
        machine = await asyncio.to_thread(Machine.connect, sandbox_id, conn)
        await asyncio.to_thread(machine.delete)

    async def _configure_host_access(self, sandbox: SmolVmSandbox) -> None:
        code, stdout, stderr = await sandbox.exec_with_output("getent", "ahostsv4", "host.smolvm.internal")
        if code or not stdout.strip():
            raise RuntimeError(f"Cannot resolve SmolVM host gateway: {stderr[-500:]}")
        gateway = str(ipaddress.IPv4Address(stdout.split()[0]))
        # VM-level downloads use the same host name as the containers and local grants.
        # Keep the entry current across reconnects without changing the host's trust CA.
        await sandbox.exec_script(
            "sed -i -e 's/[[:space:]]host\\.docker\\.internal\\([[:space:]]\\|$\\)/\\1/g' "
            "-e '/^[[:space:]]*[[:xdigit:].:]\\+[[:space:]]*$/d' /etc/hosts && "
            f"printf '%s\\n' {shlex.quote(gateway + ' host.docker.internal')} >> /etc/hosts"
        )
        sandbox.extra_hosts = (f"host.docker.internal:{gateway}",)

    def _container_args(self, sandbox: SmolVmSandbox, *, image_name: str, port: int, env: dict[str, str]) -> str:
        args = super()._container_args(sandbox, image_name=image_name, port=port, env=env)
        return " ".join(f"--add-host {shlex.quote(entry)}" for entry in sandbox.extra_hosts) + " " + args

    async def _start_container(self, sandbox: SmolVmSandbox, *, image_name: str, port: int, env: dict[str, str]) -> None:
        trust_dir = await asyncio.to_thread(local_grant_trust)
        if trust_dir is None:
            await super()._start_container(sandbox, image_name=image_name, port=port, env=env)
            return
        args = self._container_args(sandbox, image_name=image_name, port=port, env={**LOCAL_TRUST_ENV, **env})
        await sandbox.exec_script(f"docker create {args} > /dev/null")
        await start_trusting(sandbox, sandbox.container_name, trust_dir)
