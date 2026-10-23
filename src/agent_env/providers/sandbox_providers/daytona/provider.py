"""Daytona sandbox provider.

Every sandbox is an ephemeral, public Daytona sandbox booted from the Docker host image (built and
cached by Daytona) or a configured snapshot, with agent-env's ``timeout`` as its hard lifetime. The
gateway and agent containers run under Docker inside it through the inherited ``VmSandbox`` helpers.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Callable
from typing import Any, ClassVar, Self

from daytona import AsyncDaytona, DaytonaBadRequestError, DaytonaConfig, DaytonaForbiddenError, DaytonaNotFoundError

from agent_env.attribution import PIPELINE_STEP_KEY, RUN_ID_KEY, Attribution
from agent_env.providers.sandbox_providers.daytona import lifecycle
from agent_env.providers.sandbox_providers.daytona.config import (
    PORTS_LABEL,
    Settings,
    create_params,
    parse_settings,
    sandbox_labels,
    sandbox_name,
    sandbox_resources,
    ttl_minutes,
)
from agent_env.providers.sandbox_providers.daytona.network import create_fields, policy_from_sandbox, refusal
from agent_env.providers.sandbox_providers.daytona.sandbox import DaytonaSandbox
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxNotFoundError,
    SandboxProvider,
    apply_default_attribution,
    refuse_unenforceable_policy,
)

logger = logging.getLogger(__name__)

SANDBOX_STARTED_EVENT = "agent_env.daytona_sandbox_started"
# Phrases of Daytona's refusals of a sandbox's egress lists (a Tier 1-2 organization, a list too long or
# malformed), as opposed to any other reason a create is refused.
_POLICY_REFUSALS = ("network access is restricted", "allow list", "allowlist", "invalid domain")


def _client(settings: Settings) -> Any:
    return AsyncDaytona(DaytonaConfig(api_key=settings.api_key, api_url=settings.api_url, target=settings.target))


class DaytonaSandboxProvider(SandboxProvider):
    """Docker-capable Daytona sandboxes as agent-env VMs. ``api_key`` comes from the resolved config
    (an ``env:`` or ``secret:`` reference), never from the process environment."""

    # Daytona Cloud serves exposed ports at https://<port>-<sandbox-id>.<proxy domain>, by region.
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.daytonaproxy01.net", "*.daytonaproxy01.eu")
    _RATE_LIMIT_ATTEMPTS = 5
    _RATE_LIMIT_BASE_DELAY = 2.0
    _RECLAIM_ATTEMPTS = 3
    _RECLAIM_DELAY = 1.0

    def __init__(self, settings: Settings, *, client_factory: Callable[[Settings], Any] = _client):
        self._settings = settings
        # Injectable so unit tests never build an SDK client.
        self._client_factory = client_factory
        # One client per event loop, shared by every sandbox this provider creates or reconnects to: an SDK client
        # holds an HTTP session bound to the loop it was made in, and the CLI runs several loops in one process.
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = weakref.WeakKeyDictionary()

    def __repr__(self) -> str:
        return f"DaytonaSandboxProvider(api_url={self._settings.api_url!r}, target={self._settings.target!r})"

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        return cls(parse_settings(config))

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        """Allow-all, or an allowlist of up to 100 hostnames (``*.`` wildcards allowed)."""
        return refusal(policy) is None

    def _client(self) -> Any:
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            client = self._clients[loop] = self._client_factory(self._settings)
        return client

    async def close(self) -> None:
        """Close this event loop's client; a sandbox this provider made reads its handle again on a new one."""
        client = self._clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.close()

    async def _wrap(self, raw: Any, ports: list[int], policy: NetworkPolicy | None) -> DaytonaSandbox:
        """``raw`` as a DaytonaSandbox, with its ports' preview URLs."""
        urls = {port: (await raw.get_preview_link(port)).url for port in ports}
        return DaytonaSandbox(self._client, raw, tunnel_urls=urls, network_policy=policy)

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
    ) -> DaytonaSandbox:
        """Create a sandbox; ``timeout`` is its hard lifetime."""
        refuse_unenforceable_policy(self, network_policy)
        if image is not None:
            raise ValueError("the Daytona provider boots its host image or snapshot; image overrides are unsupported")
        if boot_mode is not None:
            logger.warning("Ignoring boot_mode=%s: Daytona sandboxes have no boot modes", boot_mode)
        policy = self.effective_network_policy(network_policy).with_hosts(self._settings.egress_hosts)
        if reason := refusal(policy):
            raise NetworkPolicyUnsupportedError(reason)
        client = self._client()
        if self._settings.snapshot:
            await self._check_snapshot_size(client, cpu=cpu, memory_mb=memory, disk_gb=disk_size_gb)
        ports = list(dict.fromkeys(exposed_ports or []))
        defaults = apply_default_attribution(dict(attribution or {}))
        resolved = {key: str(value) for key, value in defaults.items() if value is not None}
        resources = sandbox_resources(cpu=cpu, memory_mb=memory, disk_gb=disk_size_gb)
        common = {
            "name": sandbox_name(resolved),
            "labels": sandbox_labels(resolved, ports, host_image=self._settings.snapshot is None),
            "public": True,
            "ephemeral": True,
            "auto_stop_interval": 0,
            "ttl_minutes": ttl_minutes(timeout),
            **create_fields(policy),
        }
        params, wait = create_params(self._settings, common, resources)
        retry = lifecycle.Retry(
            self._RATE_LIMIT_ATTEMPTS, self._RATE_LIMIT_BASE_DELAY, self._RECLAIM_ATTEMPTS, self._RECLAIM_DELAY
        )
        try:
            raw = await lifecycle.create(client, params, timeout=wait, retry=retry)
        except (DaytonaForbiddenError, DaytonaBadRequestError) as exc:
            refusal_text = str(exc).lower()
            if policy.restricts_egress and any(phrase in refusal_text for phrase in _POLICY_REFUSALS):
                raise NetworkPolicyUnsupportedError(f"Daytona refused the egress policy: {exc}") from exc
            raise
        try:
            sandbox = await self._wrap(raw, ports, policy)
            logger.info(
                "Daytona sandbox started: sandbox_id=%s cpu=%s memory=%sGiB disk=%sGiB attribution=%s",
                raw.id,
                resources.cpu,
                resources.memory,
                resources.disk,
                resolved,
                extra={
                    "event": SANDBOX_STARTED_EVENT,
                    "daytona_sandbox_id": raw.id,
                    "daytona_attribution": resolved,
                    PIPELINE_STEP_KEY: resolved.get(PIPELINE_STEP_KEY),
                    RUN_ID_KEY: resolved.get(RUN_ID_KEY),
                },
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            return sandbox
        except BaseException:
            # The sandbox is deleted, with retries, before the create's own error or cancellation is raised.
            await lifecycle.settle(lifecycle.delete_sandbox(raw, params.name or raw.id, retry))
            raise

    async def _check_snapshot_size(self, client: Any, *, cpu: float, memory_mb: int, disk_gb: float) -> None:
        """Refuse a request bigger than the configured snapshot, which fixes the size of what it boots."""
        snapshot = await client.snapshot.get(self._settings.snapshot)
        if cpu > snapshot.cpu or memory_mb > snapshot.mem * 1024 or disk_gb > snapshot.disk:
            raise ValueError(
                f"Daytona snapshot {self._settings.snapshot!r} boots sandboxes of {snapshot.cpu} vCPU, {snapshot.mem} "
                f"GiB memory and {snapshot.disk} GiB disk, less than the cpu={cpu}, memory={memory_mb}MB, "
                f"disk={disk_gb}GB requested; create it with at least that size"
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
    ) -> DaytonaSandbox:
        """A bare VM the caller loads and starts ``image_name`` in, as on the other VM providers."""
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

    async def _get_raw(self, sandbox_id: str) -> Any:
        try:
            return await self._client().get(sandbox_id)
        except DaytonaNotFoundError as exc:
            raise SandboxNotFoundError(f"Daytona sandbox {sandbox_id!r} was not found") from exc

    async def get_sandbox(self, sandbox_id: str) -> DaytonaSandbox:
        """Reconnect, restoring ports, preview URLs and the applied egress policy; it changes nothing."""
        raw = await self._get_raw(sandbox_id)
        labels = getattr(raw, "labels", None) or {}
        ports = [int(port) for port in str(labels.get(PORTS_LABEL, "")).split(",") if port.strip().isdigit()]
        if (policy := policy_from_sandbox(raw)) is None:
            logger.warning("Daytona sandbox %s applies an egress policy agent-env can't express", sandbox_id)
        return await self._wrap(raw, ports, policy)

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        """Delete by id, with no preview links or policy read; a sandbox already gone is done."""
        try:
            raw = await self._get_raw(sandbox_id)
        except SandboxNotFoundError:
            logger.info("Daytona sandbox %s was already gone at terminate", sandbox_id)
            return
        await DaytonaSandbox(self._client, raw, tunnel_urls={}, network_policy=None).terminate()
