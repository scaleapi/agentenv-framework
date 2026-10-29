"""boat sandbox provider.

Every sandbox is a boat.dev sandbox, an Ubuntu machine with Docker, created with none of the boat account's secrets
(``noEnv``) and no snapshots, since agent-env never resumes one; boat stops it at agent-env's ``timeout``. Exposed ports
are hosted on public HTTPS URLs, recorded in the sandbox's labels so a provider built fresh reconnects to them. boat has
no egress controls, so a policy that restricts egress is refused.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Self

from boat_sdk import ApiClient, Configuration, run_command
from boat_sdk.api.boat_api import BoatApi
from boat_sdk.exceptions import ApiException
from boat_sdk.models import CreateSandboxRequest, HostPortRequest

from agent_env.attribution import PIPELINE_STEP_KEY, RUN_ID_KEY, Attribution
from agent_env.providers.sandbox_providers.boat import lifecycle
from agent_env.providers.sandbox_providers.boat.config import (
    Settings,
    parse_settings,
    ports_from_labels,
    sandbox_labels,
    sandbox_name,
    sandbox_type,
)
from agent_env.providers.sandbox_providers.boat.sandbox import BoatSandbox, RunCommand
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxNotFoundError,
    SandboxProvider,
    apply_default_attribution,
    refuse_unenforceable_policy,
)

logger = logging.getLogger(__name__)

SANDBOX_STARTED_EVENT = "agent_env.boat_sandbox_started"
# States of a sandbox that accepts commands.
_READY_STATES = frozenset({"ready", "idle", "running"})
# States a starting sandbox does not come back from.
_FAILED_STATES = frozenset({"archiving", "archived", "error"})
# A stopped sandbox: with snapshots off, its disk went with it.
_GONE_STATES = frozenset({"archiving", "archived"})


def _api(settings: Settings) -> BoatApi:
    configuration = Configuration(host=settings.api_url, access_token=settings.api_key)
    # A connection for each thread SDK calls run on, so none is opened and dropped per call.
    configuration.connection_pool_maxsize = lifecycle.THREADS
    return BoatApi(ApiClient(configuration))


class BoatSandboxProvider(SandboxProvider):
    """Docker-capable boat sandboxes as agent-env VMs. ``api_key`` comes from the resolved config (an ``env:`` or
    ``secret:`` reference), never from the process environment."""

    # boat serves hosted ports at https://<sandbox subdomain>-<port>.on.boat.dev.
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.on.boat.dev",)
    _CREATE_ATTEMPTS = 6
    _RETRY_BASE_DELAY = 2.0
    _RETRY_MAX_DELAY = 60.0
    _DELETE_ATTEMPTS = 4
    _DELETE_DELAY = 1.0
    _CANCEL_WAIT = 120.0
    _READY_TIMEOUT = 300.0
    _READY_POLL_INTERVAL = 1.0

    def __init__(
        self,
        settings: Settings,
        *,
        api_factory: Callable[[Settings], Any] = _api,
        run_command: RunCommand = run_command,
    ):
        self._settings = settings
        # Injectable so unit tests never build an SDK client or stream from boat.
        self._api = api_factory(settings)
        self._run_command = run_command
        self._retry = lifecycle.Retry(
            attempts=self._CREATE_ATTEMPTS,
            base_delay=self._RETRY_BASE_DELAY,
            max_delay=self._RETRY_MAX_DELAY,
            delete_attempts=self._DELETE_ATTEMPTS,
            delete_delay=self._DELETE_DELAY,
            cancel_wait=self._CANCEL_WAIT,
        )

    def __repr__(self) -> str:
        return f"BoatSandboxProvider(api_url={self._settings.api_url!r})"

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        return cls(parse_settings(config))

    async def close(self) -> None:
        """Close the SDK client's connections, which its urllib3 pool keeps; the SDK's own exit leaves them open."""
        pool = getattr(getattr(getattr(self._api, "api_client", None), "rest_client", None), "pool_manager", None)
        if pool is not None:
            await lifecycle.call(pool.clear)

    def _handle(self, sandbox_id: str, tunnel_urls: Mapping[int, str]) -> BoatSandbox:
        # boat applies no egress policy, so every sandbox's is unrestricted.
        return BoatSandbox(self._api, sandbox_id, tunnel_urls=tunnel_urls, network_policy=NetworkPolicy(),
                           retry=self._retry, run_command=self._run_command)

    async def _host(self, sandbox_id: str, port: int) -> str:
        """The public URL of ``port``; hosting a hosted port again answers its URL."""
        hosted = await lifecycle.call(self._api.host_port, sandbox_id, HostPortRequest(port=port, public=True))
        return hosted.url

    async def _hosted_urls(self, sandbox_id: str, ports: Sequence[int], hosted: Sequence[Any] | None) -> dict[int, str]:
        """The URL of each port: as the create hosted it, or hosted now where the create could not."""
        urls = {
            entry.port: entry.url
            for entry in hosted or []
            if entry.port in ports and entry.url and getattr(entry, "success", True) is not False
        }
        for port in ports:
            if port not in urls:
                urls[port] = await self._host(sandbox_id, port)
        return urls

    async def _wait_ready(self, sandbox_id: str) -> None:
        """Poll until the sandbox accepts commands, within ``_READY_TIMEOUT``."""
        deadline = asyncio.get_running_loop().time() + self._READY_TIMEOUT
        while True:
            state = (await lifecycle.call(self._api.get, sandbox_id)).sandbox.state
            if state in _READY_STATES:
                return
            if state in _FAILED_STATES:
                raise RuntimeError(f"boat sandbox {sandbox_id} went to state {state!r} while it started")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"boat sandbox {sandbox_id} was not ready after {self._READY_TIMEOUT}s ({state!r})")
            await asyncio.sleep(self._READY_POLL_INTERVAL)

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
    ) -> BoatSandbox:
        """Create a sandbox of the smallest type that covers the request; ``timeout`` is when boat stops it."""
        refuse_unenforceable_policy(self, network_policy)
        if image is not None:
            raise ValueError("the boat provider boots boat's Ubuntu image; image overrides are unsupported")
        if boot_mode is not None:
            logger.warning("Ignoring boot_mode=%s: boat sandboxes have no boot modes", boot_mode)
        size = sandbox_type(cpu=cpu, memory_mb=memory, disk_gb=disk_size_gb)
        ports = list(dict.fromkeys(exposed_ports or []))
        defaults = apply_default_attribution(dict(attribution or {}))
        resolved = {key: str(value) for key, value in defaults.items() if value is not None}
        token = uuid.uuid4().hex
        request = CreateSandboxRequest(
            type=size,
            ttl_seconds=int(timeout),
            name=sandbox_name(resolved),
            labels=sandbox_labels(resolved, ports, token),
            no_env=True,
            snapshots=False,
            hosted_ports=[{"port": port, "public": True} for port in ports] or None,
        )
        created = await lifecycle.create(self._api, request, token, self._retry)
        sandbox_id = created.sandbox.id
        try:
            await self._wait_ready(sandbox_id)
            sandbox = self._handle(sandbox_id, await self._hosted_urls(sandbox_id, ports, created.hosted_ports))
            logger.info(
                "boat sandbox started: sandbox_id=%s type=%s ports=%s attribution=%s",
                sandbox_id,
                size,
                ports,
                resolved,
                extra={
                    "event": SANDBOX_STARTED_EVENT,
                    "boat_sandbox_id": sandbox_id,
                    "boat_attribution": resolved,
                    PIPELINE_STEP_KEY: resolved.get(PIPELINE_STEP_KEY),
                    RUN_ID_KEY: resolved.get(RUN_ID_KEY),
                },
            )
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            return sandbox
        except BaseException:
            # The sandbox is deleted, with retries, before the create's own error or cancellation is raised.
            await lifecycle.settle(lifecycle.delete_sandbox(self._api, sandbox_id, self._retry))
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
    ) -> BoatSandbox:
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

    async def get_sandbox(self, sandbox_id: str) -> BoatSandbox:
        """Reconnect, restoring the ports and URLs the sandbox's labels record; it changes nothing."""
        try:
            raw = (await lifecycle.call(self._api.get, sandbox_id)).sandbox
        except ApiException as exc:
            if lifecycle.is_not_found(exc):
                raise SandboxNotFoundError(f"boat sandbox {sandbox_id!r} was not found") from exc
            raise
        if raw.state in _GONE_STATES:
            raise SandboxNotFoundError(f"boat sandbox {sandbox_id!r} is stopped, and without snapshots its disk is gone")
        ports = ports_from_labels(raw.labels)
        return self._handle(sandbox_id, {port: await self._host(sandbox_id, port) for port in ports})

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        """Delete by id, with nothing read first; a sandbox already gone is done."""
        if not await lifecycle.delete_sandbox(self._api, sandbox_id, self._retry):
            raise RuntimeError(f"boat sandbox {sandbox_id} could not be deleted")
