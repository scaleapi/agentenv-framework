"""Freestyle VM provisioning, resource sizing and reconnection."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from typing import Any, ClassVar, Self

import httpx

from agent_env.attribution import Attribution
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.freestyle.client import (
    FreestyleClient,
    vm_path,
)
from agent_env.providers.sandbox_providers.freestyle.sandbox import FreestyleSandbox
from agent_env.providers.sandbox_providers.sandbox import (
    NetworkPolicy,
    NetworkPolicyUnsupportedError,
)
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxProvider,
    apply_default_attribution,
)

logger = logging.getLogger(__name__)


class FreestyleSandboxProvider(SandboxProvider):
    """Boot Docker-capable snapshots and publish the requested ports over HTTPS."""

    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.style.dev",)

    def __init__(
        self,
        *,
        api_key: str,
        snapshot_id: str = "freestyle/ubuntu",
        api_url: str = "https://beta-api.freestyle.sh",
        exec_timeout_seconds: int = 300,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigError(
                "[sandbox.providers.freestyle.config] requires a non-empty 'api_key'"
            )
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ConfigError(
                "Freestyle snapshot_id must be a non-empty Docker-capable snapshot"
            )
        if (
            type(exec_timeout_seconds) is not int
            or not 1 <= exec_timeout_seconds <= 300
        ):
            raise ConfigError(
                "Freestyle exec_timeout_seconds must be an integer from 1 to 300"
            )
        self._client = FreestyleClient(api_key=api_key, api_url=api_url)
        self._snapshot_id = snapshot_id
        self._exec_timeout_seconds = exec_timeout_seconds
        self._cleanups: set[asyncio.Task] = set()

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        if not config.get("api_key"):
            raise ConfigError(
                "[sandbox.providers.freestyle.config] requires a non-empty 'api_key'"
            )
        return cls(**config)

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
    ) -> FreestyleSandbox:
        policy = network_policy or NetworkPolicy()
        if not self.supports_network_policy(policy):
            raise NetworkPolicyUnsupportedError(
                f"Freestyle cannot enforce network policy mode {policy.mode.value}; "
                "hostname allowlists require a different sandbox backend"
            )
        if boot_mode is not None:
            raise ValueError(
                "Freestyle boot_mode overrides are unsupported; select a snapshot instead"
            )
        if (
            not math.isfinite(cpu)
            or cpu <= 0
            or not math.isfinite(disk_size_gb)
            or disk_size_gb <= 0
        ):
            raise ValueError(
                "Freestyle CPU and disk requests must be finite and positive"
            )
        if (
            type(memory) is not int
            or memory <= 0
            or type(timeout) is not int
            or timeout <= 0
        ):
            raise ValueError("Freestyle memory and timeout must be positive integers")
        ports = list(dict.fromkeys(exposed_ports or []))
        if any(type(port) is not int or not 1 <= port <= 65535 for port in ports):
            raise ValueError("Freestyle exposed ports must be integers from 1 to 65535")
        metadata = {
            key: value
            for key, value in apply_default_attribution(dict(attribution or {})).items()
            if value is not None
        }
        if len(metadata) > 64 or any(
            not isinstance(key, str)
            or not key
            or len(key) > 63
            or key.startswith("freestyle.sh/")
            or not isinstance(value, str)
            or len(value) > 63
            for key, value in metadata.items()
        ):
            raise ValueError(
                "Freestyle attribution supports 64 entries, keys and values up to 63 characters, and no freestyle.sh/ keys"
            )
        slug = f"agentenv-{uuid.uuid4().hex}"
        tunnels = {port: f"https://{slug}-{port}.style.dev" for port in ports}
        request = {
            "snapshotId": image or self._snapshot_id,
            "slug": slug,
            "ttlSeconds": timeout,
            "metadata": metadata,
            "firewall": {
                "rules": [
                    {"action": "allow", "source": {}, "destination": {"public": True}}
                ]
            },
            "tls": {
                "rules": [
                    {
                        "action": "allow",
                        "domain": url.removeprefix("https://"),
                        "source": {"public": True},
                        "destination": {"port": port},
                    }
                    for port, url in tunnels.items()
                ]
            },
        }
        creation = asyncio.create_task(
            self._client.request("POST", "/v5/vms", json=request, timeout=60)
        )
        try:
            data = await asyncio.shield(creation)
            vm_id = data["id"]
            resources = data["resources"]
            requested = {
                "cpu": math.ceil(cpu),
                "memory": memory,
                "storage": math.ceil(disk_size_gb * 1024),
            }
            resize = {
                key: value for key, value in requested.items() if value > resources[key]
            }
            if resize:
                await self._client.request(
                    "POST", f"{vm_path(vm_id)}/resize", json=resize, timeout=60
                )
            sandbox = FreestyleSandbox(
                self._client,
                vm_id,
                tunnel_urls=tunnels,
                exec_timeout_seconds=self._exec_timeout_seconds,
                network_policy=policy,
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            return sandbox
        except BaseException:
            cleanup = asyncio.create_task(self._reap_creation(creation, slug))
            self._cleanups.add(cleanup)
            cleanup.add_done_callback(self._cleanups.discard)
            await asyncio.shield(cleanup)
            raise

    async def _reap_creation(self, creation: asyncio.Task, slug: str) -> None:
        """Wait for a cancelled create's response, or find its VM by the unique slug after a lost response."""
        try:
            data = await creation
            vm_id = data["id"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            vm_id = slug
        for attempt in range(3):
            try:
                await self._client.request("DELETE", vm_path(vm_id), timeout=30)
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    return
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    break
            except httpx.TransportError:
                pass
            if attempt < 2:
                await asyncio.sleep(2**attempt)
        logger.error(
            "Could not delete Freestyle VM %s after provisioning failure; its TTL bounds its lifetime",
            vm_id,
        )

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
    ) -> FreestyleSandbox:
        return await self.create_vm(
            cpu=cpu,
            memory=memory,
            disk_size_gb=disk_size_gb,
            timeout=timeout,
            exposed_ports=[port],
            attribution=attribution,
            network_policy=network_policy,
        )

    async def get_sandbox(self, sandbox_id: str) -> FreestyleSandbox:
        data = await self._client.request("GET", vm_path(sandbox_id))
        vm_id = data["id"]
        tunnels: dict[int, str] = {}
        offset = 0
        while True:
            page = await self._client.request(
                "GET", "/v5/tls", params={"vmId": vm_id, "limit": 100, "offset": offset}
            )
            for rule in page["rules"]:
                destination = rule["destination"]
                port = destination.get("port")
                if (
                    rule["protocol"] == "http"
                    and rule["source"] == {"public": True}
                    and destination.get("vmId") == vm_id
                    and port is not None
                    and rule["domain"] == f"{data['slug']}-{port}.style.dev"
                ):
                    tunnels[port] = f"https://{rule['domain']}"
            offset += len(page["rules"])
            if not page["rules"] or offset >= page["totalCount"]:
                break
        return FreestyleSandbox(
            self._client,
            vm_id,
            tunnel_urls=tunnels,
            exec_timeout_seconds=self._exec_timeout_seconds,
        )

    async def close(self) -> None:
        if self._cleanups:
            await asyncio.gather(*self._cleanups)
        await self._client.close()
