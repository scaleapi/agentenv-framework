"""Run AgentEnv workloads in local SmolVM machines."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import shlex
import socket
import uuid
from urllib.parse import urlsplit
from typing import ClassVar

from smol import ConnectOptions, Machine, MachineConfig, PortSpec, ResourceSpec

from agent_env.attribution import Attribution
from agent_env.config import get_config
from agent_env.store.image_store.local_registry_image_store import LocalRegistryImageStore
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.local_sandbox import LOCAL_TRUST_ENV, local_grant_trust, start_trusting
from agent_env.providers.sandbox_providers.sandbox_provider import (
    Accepts, SandboxProvider, refuse_unenforceable_policy,
)
from agent_env.providers.sandbox_providers.smol_vm.sandbox import SmolVmSandbox, _host_url

_DUMMY_PORT = 60832


def _host_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class SmolVmSandboxProvider(SandboxProvider):
    ON_THIS_MACHINE: ClassVar[bool] = True
    CREATES_VMS: ClassVar[bool] = True
    SANDBOX_ACCEPTS: ClassVar[Accepts] = Accepts.LOADABLE
    CONTAINER_ACCEPTS: ClassVar[Accepts] = Accepts.NAME

    @classmethod
    def shares_network_with(cls, sandbox_type: str | None) -> bool:
        return False

    @classmethod
    def get_external_url(cls, url: str) -> str:
        return _host_url(url)

    def url_from_sandbox(self, url: str) -> str:
        return _host_url(url)

    async def create_sandbox(self, *, image_name: str, port: int, env: dict[str, str],
                             cpu: float = 1.0, memory: int = 8192, disk_size_gb: float = 10,
                             timeout: int = 7200, attribution: Attribution | None = None,
                             network_policy: NetworkPolicy | None = None) -> SmolVmSandbox:
        return await self.create_container(
            image_name=image_name, port=port, env=env, cpu=cpu, memory=memory,
            disk_size_gb=disk_size_gb, timeout=timeout, attribution=attribution,
            network_policy=network_policy,
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
            name=name, persistent=True, wait_for_ports=False,
            resources=ResourceSpec(cpus=math.ceil(cpu), memory_mb=memory, storage_gb=math.ceil(disk_size_gb), network=True),
            ports=[PortSpec(host=host, guest=guest) for guest, host in published.items()],
        )
        creation = asyncio.create_task(asyncio.to_thread(Machine.create, config, ConnectOptions(target="local")))
        try:
            machine = await asyncio.shield(creation)
        except asyncio.CancelledError:
            # The native create call keeps running after its awaiting task is cancelled.
            # Wait for it and delete the result so cancellation cannot orphan a VM.
            try:
                created = await creation
            except Exception:
                pass
            else:
                await asyncio.to_thread(created.delete)
            raise
        sandbox = SmolVmSandbox(machine, ports)
        try:
            await self._setup_docker(sandbox)
            await self._configure_host_access(sandbox)
            await sandbox.persist_ports()
            sandbox.network_policy = self.effective_network_policy(network_policy)
            return sandbox
        except BaseException:
            await sandbox.terminate()
            raise

    async def _setup_docker(self, sandbox: SmolVmSandbox) -> None:
        code, stdout, stderr = await sandbox.exec_with_output(
            "sh", "-c", "command -v bash >/dev/null && command -v docker >/dev/null && "
            "command -v curl >/dev/null && command -v python3 >/dev/null && command -v unzip >/dev/null && command -v socat >/dev/null "
            "|| apk add --no-cache docker docker-cli-compose bash curl python3 unzip socat",
        )
        if code:
            raise RuntimeError(f"Could not install guest Docker dependencies ({code}): {stdout[-500:]} {stderr[-500:]}")
        await sandbox.exec_script("""
set -e
mkdir -p /storage/docker /storage/containerd /var/lib/docker /var/lib/containerd
mountpoint -q /var/lib/docker || mount --bind /storage/docker /var/lib/docker
mountpoint -q /var/lib/containerd || mount --bind /storage/containerd /var/lib/containerd
if ! docker info >/dev/null 2>&1; then
    rm -f /var/run/docker.pid /run/containerd/containerd.pid
    dockerd --storage-driver=overlay2 >/tmp/agentenv-dockerd.log 2>&1 &
    for i in $(seq 1 40); do docker info >/dev/null 2>&1 && exit 0; sleep 1; done
    tail -80 /tmp/agentenv-dockerd.log
    exit 1
fi
""")
        await self._ensure_local_registry_proxy(sandbox)

    async def _ensure_local_registry_proxy(self, sandbox: SmolVmSandbox) -> None:
        store = get_config().get_image_store()
        if not isinstance(store, LocalRegistryImageStore):
            return
        registry = urlsplit(f"//{store.registry_host}")
        if registry.hostname not in ("localhost", "127.0.0.1"):
            return
        port = registry.port or 5000
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

    async def _configure_host_access(self, sandbox: SmolVmSandbox) -> None:
        code, stdout, stderr = await sandbox.exec_with_output("getent", "ahostsv4", "host.smolvm.internal")
        if code or not stdout.strip():
            raise RuntimeError(f"Cannot resolve SmolVM host gateway: {stderr[-500:]}")
        gateway = str(ipaddress.IPv4Address(stdout.split()[0]))
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
