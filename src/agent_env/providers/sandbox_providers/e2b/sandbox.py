"""E2B's async sandbox presented through the :class:`VmSandbox` contract.

The E2B SDK intentionally returns a completed ``CommandResult`` from
``commands.run``.  ``VmSandbox``, on the other hand, is built on a small
process interface whose stdout and stderr are async byte streams.  This module
bridges those two shapes without initializing E2B SDK clients at module import
time, keeping provider registration and offline unit tests isolated.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM

if TYPE_CHECKING:
    from e2b import AsyncSandbox

logger = logging.getLogger(__name__)

E2B_ALL_TRAFFIC = "0.0.0.0/0"


# One lock per E2B sandbox in each event loop, held to widen its egress policy, so concurrent downloads don't each widen
# a stale copy and drop the other's host.
_policy_locks: weakref.WeakValueDictionary[tuple[asyncio.AbstractEventLoop, str], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)


class _BytesReader:
    """The small ``StreamReader`` subset consumed by ``Sandbox.exec_with_output``."""

    def __init__(self, value: str | bytes | None):
        self._value = value.encode() if isinstance(value, str) else (value or b"")

    async def read(self) -> bytes:
        return self._value


class _E2BCompletedProcess:
    """Adapt E2B's completed command result to the common process protocol."""

    def __init__(self, result: Any):
        self.stdout = _BytesReader(getattr(result, "stdout", ""))
        self.stderr = _BytesReader(getattr(result, "stderr", ""))
        self._exit_code = int(getattr(result, "exit_code", -1))

    async def wait(self) -> int:
        return self._exit_code


class _E2BTunnelURLs(dict[int, str]):
    """A port map that asks E2B for an external host on first use.

    E2B doesn't allocate persistent tunnel objects.  Its ``get_host(port)``
    derives the public endpoint for any port, so laziness both avoids a stale
    static port list after reconnect and preserves the normal ``tunnel_urls``
    indexing used throughout agent-env.
    """

    def __init__(self, sandbox: E2BSandbox):
        super().__init__()
        self._sandbox = sandbox

    def __missing__(self, port: int) -> str:
        return self._sandbox.get_host(port)

    def get(self, port: int, default: str | None = None) -> str | None:
        # ``dict.get`` is used as a capability query by deployment callers.
        # Unlike indexing, it must not imply that every arbitrary port has a
        # reachable E2B endpoint (nor should it mutate this cache).
        return super().get(port, default)


class E2BSandbox(VmSandbox):
    """An E2B Linux sandbox with Docker, exposed as an agent-env VM sandbox.

    ``sandbox`` is an already-created ``e2b.AsyncSandbox``.  Keeping creation
    in the provider lets this adapter stay focused on VM process, lifecycle,
    reconnect, and public-port behavior.
    """

    type = "e2b"
    # ``docker info`` should normally return in well under one second.  Keep
    # each readiness probe short so a wedged daemon cannot consume the entire
    # VM-ready budget in one await.
    _DOCKER_READINESS_COMMAND_TIMEOUT = 10
    _MIN_COMMAND_TIMEOUT_SECONDS = 0.001
    _UNBOUNDED_COMMAND_TIMEOUT_SECONDS = 0
    _DOCKER_LOG_TAIL_LINES = 40
    _UNKNOWN_EXIT_CODE = -1

    def __init__(
        self,
        sandbox: AsyncSandbox | Any,
        *,
        exposed_ports: Iterable[int] = (),
        network_policy: NetworkPolicy | None = None,
    ):
        self._sandbox = sandbox
        self.sandbox_id = sandbox.sandbox_id
        self.vnc_url = None
        self.mode = SANDBOX_MODE_VM
        self.network_policy = network_policy
        self.tunnel_urls = _E2BTunnelURLs(self)
        for port in exposed_ports:
            self.get_host(port)

    async def terminate(self) -> None:
        """Irreversibly stop the E2B sandbox (``kill`` is idempotent)."""
        killed = await self._sandbox.kill()
        if not killed:
            logger.info("E2B sandbox %s was already absent during terminate", self.sandbox_id)

    async def exec(self, *command: str) -> _E2BCompletedProcess:
        """Run argv safely through E2B's shell-command API.

        E2B accepts one command string and itself invokes Bash.  ``shlex.join``
        keeps arguments distinct, notably the scripts supplied to inherited
        ``exec_script``.  Do not strip ``sudo``: E2B commands run as the normal
        sandbox user and Docker administration requires the inherited sudo
        prefixes (``sudo bash`` / ``sudo docker``).
        """
        return await self._run_command(
            *command,
            timeout=self._UNBOUNDED_COMMAND_TIMEOUT_SECONDS,
        )

    async def _run_command(
        self,
        *command: str,
        timeout: float,
    ) -> _E2BCompletedProcess:
        """Run an E2B command with an explicit SDK command timeout.

        Normal VM operations remain unbounded (``timeout=0``) because image
        loads and agent commands can legitimately run for a long time.
        Readiness uses this primitive with a finite timeout so its wall-clock
        budget remains meaningful even when Docker is wedged.
        """
        try:
            result = await self._sandbox.commands.run(shlex.join(command), timeout=timeout)
        except Exception as exc:
            # E2B raises CommandExitException for a non-zero foreground command;
            # it is also a CommandResult, so expose that failure through the
            # common process result rather than treating it as an SDK transport
            # error.  Other exceptions (network/auth/etc.) must still surface.
            if not all(hasattr(exc, attribute) for attribute in ("exit_code", "stdout", "stderr")):
                raise
            result = exc
        return _E2BCompletedProcess(result)

    async def _exec_with_output_bounded(
        self,
        *command: str,
        timeout: float,
    ) -> tuple[int, str, str]:
        """Execute one short readiness/diagnostic command with an E2B timeout."""
        process = await self._run_command(*command, timeout=timeout)
        stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
        return await process.wait(), stdout.decode(), stderr.decode()

    async def setup_vm_for_gateway(self, exposed_ports: list[int] | None = None) -> None:
        """Start Docker and verify the VM has the Compose v2 gateway runtime.

        E2B publishes every sandbox port through its dynamic ``get_host``
        endpoint, so unlike the KubeVirt implementation there is no guest
        firewall to configure. ``EnvironmentGatewayProvider`` starts all co-located
        services with ``docker compose up``, so accepting a Docker-only base
        template here would defer a deterministic configuration error until
        deploy-env is already staging images and files.
        """
        await self.wait_for_vm()
        exit_code, version, stderr = await self._exec_with_output_bounded(
            "sudo",
            "docker",
            "compose",
            "version",
            timeout=self._DOCKER_READINESS_COMMAND_TIMEOUT,
        )
        if exit_code != 0:
            detail = (stderr or version).strip()
            raise RuntimeError(
                "E2B base template must include the Docker Compose v2 plugin "
                "required by deploy-env"
                + (f": {detail}" if detail else "")
            )
        logger.info(
            "Docker Compose ready in E2B sandbox %s: %s",
            self.sandbox_id,
            version.strip(),
        )

    async def wait_for_vm(self) -> None:
        """Ensure E2B's Docker daemon is available to inherited VM helpers."""
        await self._ensure_docker_running()

    async def _ensure_docker_running(self) -> None:
        # The base template normally has systemd, but retain service and direct
        # dockerd fallbacks for custom templates.  `nohup ... &` detaches the
        # last fallback from E2B's completed-command stream.
        deadline = time.monotonic() + self._VM_READY_TIMEOUT
        start_command = (
            "sudo systemctl start docker || sudo service docker start || "
            "(sudo nohup dockerd > /var/log/dockerd.log 2>&1 &)"
        )
        try:
            start_timeout = max(
                self._MIN_COMMAND_TIMEOUT_SECONDS,
                min(self._DOCKER_READINESS_COMMAND_TIMEOUT, self._VM_READY_TIMEOUT),
            )
            start_result = await self._run_command("bash", "-c", start_command, timeout=start_timeout)
            if await start_result.wait() != 0:
                logger.debug("E2B Docker startup command exited nonzero in %s", self.sandbox_id)
        except Exception as exc:
            logger.debug("E2B Docker startup command did not complete: %s", exc)

        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            remaining = deadline - time.monotonic()
            command_timeout = min(self._DOCKER_READINESS_COMMAND_TIMEOUT, remaining)
            try:
                exit_code, _, _ = await self._exec_with_output_bounded(
                    "sudo",
                    "docker",
                    "info",
                    timeout=command_timeout,
                )
            except Exception as exc:
                # A timed-out or temporarily unavailable command is simply a
                # failed readiness probe.  The enclosing deadline, rather than
                # a single SDK call, owns the readiness budget.
                logger.debug(
                    "Docker readiness probe failed in E2B sandbox %s: %s",
                    self.sandbox_id,
                    exc,
                )
                exit_code = self._UNKNOWN_EXIT_CODE
            if exit_code == 0:
                logger.info("Docker ready in E2B sandbox %s after %s polls", self.sandbox_id, attempts)
                return
            sleep_seconds = min(self._VM_READY_POLL_INTERVAL, max(0, deadline - time.monotonic()))
            if sleep_seconds:
                await asyncio.sleep(sleep_seconds)

        try:
            _, log_tail, _ = await self._exec_with_output_bounded(
                "sudo",
                "tail",
                "-n",
                str(self._DOCKER_LOG_TAIL_LINES),
                "/var/log/dockerd.log",
                timeout=self._DOCKER_READINESS_COMMAND_TIMEOUT,
            )
        except Exception as exc:
            log_tail = f"unavailable ({type(exc).__name__}: {exc})"
        raise RuntimeError(
            f"Docker not ready in E2B sandbox {self.sandbox_id} after "
            f"{self._VM_READY_TIMEOUT}s ({attempts} polls); "
            f"/var/log/dockerd.log tail:\n{log_tail}"
        )

    def get_host(self, port: int) -> str:
        """Return and cache E2B's current HTTPS URL for ``port``.

        ``AsyncSandbox.get_host`` returns a hostname, not a URL.  Normalizing it
        here keeps ``tunnel_urls`` compatible with the other providers.
        """
        normalized_port = int(port)
        host = self._sandbox.get_host(normalized_port)
        url = host if host.startswith(("http://", "https://")) else f"https://{host}"
        self.tunnel_urls[normalized_port] = url
        return url

    async def apply_network_policy(self, policy: NetworkPolicy) -> None:
        """Atomically replace E2B egress policy with ``policy``.

        E2B's ``update_network`` clears omitted fields.  ALLOW_ALL explicitly
        enables internet access.  ALLOWLIST combines E2B's canonical
        deny-all sentinel with explicit hostname/CIDR exceptions.
        """
        # ``allow_public_traffic`` is creation-only and is silently discarded
        # by update_network.  Do not confuse ingress configuration with the
        # workload's outbound policy here.
        network: dict[str, Any] = {"allow_internet_access": True}
        if policy.mode is NetworkMode.ALLOWLIST:
            network["allow_out"] = [*policy.allow_hosts, *policy.allow_cidrs]
            network["deny_out"] = [E2B_ALL_TRAFFIC]
        await self._sandbox.update_network(network)
        self.network_policy = policy

    async def _load_tarballs(self, artifacts: list) -> None:
        """Load image tarballs after adding their signed-download hosts to the policy.

        With a restrictive policy, the object store that supplies image tarballs
        is infrastructure rather than a workload destination.  E2B replaces an
        egress policy atomically, so union the exact signed-URL hostnames into the
        sandbox's applied policy. CIDRs stay untouched through
        :meth:`NetworkPolicy.with_hosts`.
        """
        policy = self.network_policy
        if policy is None:
            raise RuntimeError(
                f"Cannot load Docker images in reconnected E2B sandbox {self.sandbox_id}: "
                "its applied network policy is unknown, so signed download hosts cannot "
                "be added safely"
            )
        if not policy.restricts_egress:
            await super()._load_tarballs(artifacts)
            return

        signed_urls = await self._signed_image_urls(artifacts)
        await self._allow_download_hosts(signed_urls)
        await self._load_docker_images(artifacts, signed_urls)

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        """Download an object after adding its signed-download host to a restrictive policy, as image tarballs are: a
        build context is fetched this way before a build in the VM. A reconnected sandbox, whose policy is unknown,
        downloads as before."""
        policy = self.network_policy
        if policy is not None and policy.restricts_egress:
            signed = await asyncio.to_thread(get_config().get_object_store_at(object_url).signed_get_url, object_url)
            await self._allow_download_hosts([signed])
        await super()._download_object_to_vm(object_url, vm_path)

    async def _allow_download_hosts(self, signed_urls: list[str | None]) -> None:
        """Add the hosts of ``signed_urls`` to a restrictive applied policy, reading the policy under the sandbox's
        lock, so a concurrent download's host isn't dropped."""
        hosts = {parsed.hostname for url in signed_urls if url and (parsed := urlparse(url)).hostname}
        if not hosts:
            return
        async with _policy_locks.setdefault((asyncio.get_running_loop(), self.sandbox_id), asyncio.Lock()):
            policy = self.network_policy
            if policy is not None and policy.restricts_egress and not hosts <= set(policy.allow_hosts):
                await self.apply_network_policy(policy.with_hosts(sorted(hosts)))

    @classmethod
    async def reconnect(
        cls,
        sandbox_id: str,
        *,
        api_key: str | None = None,
        sandbox_cls: type[AsyncSandbox] | Any | None = None,
        exposed_ports: Iterable[int] = (),
        network_policy: NetworkPolicy | None = None,
    ) -> E2BSandbox:
        """Reconnect to an existing sandbox without relying on ambient credentials."""
        if sandbox_cls is None:
            # Optional dependency: importing this adapter must not require E2B.
            from e2b import AsyncSandbox as sandbox_cls

        sandbox = await sandbox_cls.connect(sandbox_id, api_key=api_key)
        return cls(sandbox, exposed_ports=exposed_ports, network_policy=network_policy)
