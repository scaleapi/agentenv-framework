"""A local SmolVM machine with a Docker daemon, exposed through the VM sandbox contract."""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import ClassVar
from urllib.parse import urlsplit

from smol import ConnectOptions, Machine

from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import CURL_RETRY_FLAGS, VmSandbox
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from agent_env.store.object_store.local.tls import local_ca
from agent_env.utils.docker_build import normalize_platform
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM

_PORTS_FILE = "/storage/agentenv-ports.json"


class SmolVmSandbox(VmSandbox):
    _READ_GRANT_LIFETIME_SECONDS: ClassVar[int] = 3600
    type = "smol_vm"
    ON_THIS_MACHINE = True

    def __init__(self, machine: Machine, port_map: dict[int, int]):
        self._machine = machine
        self._port_map = port_map
        self.sandbox_id = machine.id
        self.tunnel_urls = {port: machine.endpoint(port).http_url for port in port_map}
        self.vnc_url = None
        self.mode = SANDBOX_MODE_VM
        self.network_policy = None
        self.extra_hosts = ()

    def url_from_sandbox(self, url: str) -> str:
        return LocalSandboxProvider.get_external_url(url)

    async def detect_platform(self) -> None:
        code, stdout, stderr = await self.exec_with_output("uname", "-m")
        if code or not stdout.strip():
            raise RuntimeError(f"Cannot determine SmolVM guest architecture: {stderr[-500:]}")
        self.platform = normalize_platform(f"linux/{stdout.strip()}")

    async def terminate(self) -> None:
        await asyncio.to_thread(self._machine.delete)

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        command = list(args[1:] if args and args[0] == "sudo" else args)
        result = await asyncio.to_thread(self._machine.exec, command)
        return result.exit_code, result.stdout, result.stderr

    async def write_host_file(self, data: bytes, vm_path: str) -> None:
        await asyncio.to_thread(self._machine.write_file, vm_path, data)

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        store = get_config().get_object_store_at(object_url)
        signed = await asyncio.to_thread(store.signed_get_url, object_url)
        local_grant = isinstance(store, LocalFilesystemObjectStore) and store.supports_transfer_grants and store.grants_reach(self.type)
        if signed is None and local_grant:
            signed = (await asyncio.to_thread(
                store.issue_read_grant, object_url, expires_in=self._READ_GRANT_LIFETIME_SECONDS,
            )).url
        if signed is None:
            await self._write_unsigned_object(store, object_url, vm_path)
            return
        rewritten = LocalSandboxProvider.get_external_url(signed)
        local_https = local_grant or (urlsplit(rewritten).hostname == "host.docker.internal" and urlsplit(rewritten).scheme == "https")
        if local_https:
            await self.write_host_file(local_ca().bundle_path.read_bytes(), "/etc/agentenv-ca-bundle.pem")
        ca = "SSL_CERT_FILE=/etc/agentenv-ca-bundle.pem " if local_https else ""
        await self.exec_script(f"{ca}curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(rewritten)} -o {shlex.quote(vm_path)}")

    async def write_file_from_url(self, url: str, destination_path: str) -> None:
        vm_path = self._staging_path("url", destination_path)
        rewritten = LocalSandboxProvider.get_external_url(url)
        local_https = urlsplit(rewritten).hostname == "host.docker.internal" and urlsplit(rewritten).scheme == "https"
        try:
            if local_https:
                await self.write_host_file(local_ca().bundle_path.read_bytes(), "/etc/agentenv-ca-bundle.pem")
            ca = "SSL_CERT_FILE=/etc/agentenv-ca-bundle.pem " if local_https else ""
            await self.exec_script(f"{ca}curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(rewritten)} -o {shlex.quote(vm_path)}")
            await self._copy_into_container(vm_path, destination_path)
        finally:
            await self._remove_vm_temp_file(vm_path)

    async def persist_ports(self) -> None:
        await self.write_host_file(json.dumps(self._port_map).encode(), _PORTS_FILE)

    @classmethod
    async def reconnect(cls, name: str) -> SmolVmSandbox:
        machine = await asyncio.to_thread(Machine.connect, name, ConnectOptions(target="local", wait_for_ports=False))
        data = await asyncio.to_thread(machine.read_file, _PORTS_FILE)
        sandbox = cls(machine, {int(port): int(host) for port, host in json.loads(data).items()})
        await sandbox.detect_platform()
        return sandbox
