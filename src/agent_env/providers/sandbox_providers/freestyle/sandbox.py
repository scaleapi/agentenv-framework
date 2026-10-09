"""A Docker-capable Freestyle VM implementing the shared sandbox contract."""

from __future__ import annotations

import shlex
from collections.abc import AsyncIterator

import httpx

from agent_env.providers.sandbox_providers.freestyle.client import (
    FreestyleClient,
    vm_path,
)
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM


class _BytesReader:
    def __init__(self, value: str | None):
        self._value = (value or "").encode()

    async def read(self) -> bytes:
        return self._value

    async def _chunks(self) -> AsyncIterator[bytes]:
        if self._value:
            yield self._value

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._chunks()


class _CompletedProcess:
    def __init__(self, result: dict):
        self.stdout = _BytesReader(result.get("stdout"))
        self.stderr = _BytesReader(result.get("stderr"))
        status = result.get("statusCode")
        self._exit_code = 124 if status is None else int(status)

    async def wait(self) -> int:
        return self._exit_code


class FreestyleSandbox(VmSandbox):
    type = "freestyle"

    def __init__(
        self,
        client: FreestyleClient,
        sandbox_id: str,
        *,
        tunnel_urls: dict[int, str],
        exec_timeout_seconds: int = 300,
        network_policy: NetworkPolicy | None = None,
    ):
        self._client = client
        self.sandbox_id = sandbox_id
        self.tunnel_urls = tunnel_urls
        self._exec_timeout_seconds = exec_timeout_seconds
        self.network_policy = network_policy
        self.mode = SANDBOX_MODE_VM
        self.vnc_url = None

    async def exec(self, *command: str) -> _CompletedProcess:
        """Run argv as root; Freestyle buffers output and caps each command at five minutes."""
        argv = command[1:] if command[:1] == ("sudo",) else command
        result = await self._client.request(
            "POST",
            f"{vm_path(self.sandbox_id)}/exec-await",
            json={
                "command": shlex.join(argv),
                "linuxUser": "root",
                "timeoutMs": self._exec_timeout_seconds * 1000,
            },
        )
        return _CompletedProcess(result)

    async def setup_vm_for_gateway(
        self, exposed_ports: list[int] | None = None
    ) -> None:
        await self.wait_for_vm()
        await self.exec_script(
            "docker info > /dev/null && docker compose version > /dev/null"
        )

    async def _write_bytes_to_vm_path(self, data: bytes, vm_path_: str) -> None:
        await self._client.request(
            "PUT",
            f"{vm_path(self.sandbox_id)}/fs/write",
            params={"path": vm_path_},
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )

    async def terminate(self) -> None:
        try:
            await self._client.request("DELETE", vm_path(self.sandbox_id), timeout=30)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
