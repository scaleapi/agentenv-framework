"""Tensorlake's async sandbox presented through the :class:`VmSandbox` contract.

Tensorlake applies the egress policy outside the VM, so one policy covers ``dockerd`` and every
container it runs. Public ports go through the Tensorlake ingress proxy, so the guest needs no
firewall rules.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import logging
import shlex
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Mapping
from ipaddress import ip_network
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from tensorlake.sandbox import NetworkConfig, SandboxConnectionError, SandboxNotFoundError

from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM

if TYPE_CHECKING:
    from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)


def network_config_fields(policy: NetworkPolicy) -> dict[str, Any]:
    """``policy`` as Tensorlake ``NetworkConfig`` fields, which ``create`` also takes as keywords.

    A non-empty ``allow_out`` is default-deny; an empty one allows everything, so an empty
    allowlist has to turn internet access off instead.
    """
    entries = [*policy.allow_hosts, *policy.allow_cidrs]
    return {
        "allow_internet_access": not policy.restricts_egress or bool(entries),
        "allow_out": entries if policy.restricts_egress else [],
        "deny_out": [],
    }


def network_policy_from_config(network: Any, *, sandbox_id: str) -> NetworkPolicy:
    """Translate Tensorlake's applied ``NetworkConfig`` back, failing on a policy agent-env did not set.

    Image loading extends the applied policy with download hosts, and extending a misread one
    is a security bug, so anything unrecognized raises.
    """
    if network is None:
        return NetworkPolicy()
    allow_out = list(getattr(network, "allow_out", None) or [])
    deny_out = list(getattr(network, "deny_out", None) or [])
    allow_internet = getattr(network, "allow_internet_access", True)
    if deny_out or (allow_out and not allow_internet):
        raise RuntimeError(
            f"Cannot recover the network policy of Tensorlake sandbox {sandbox_id}: "
            f"allow_internet_access={allow_internet}, allow_out={allow_out}, deny_out={deny_out} "
            "is not a policy agent-env applies"
        )
    if not allow_out:
        return NetworkPolicy() if allow_internet else NetworkPolicy(mode=NetworkMode.ALLOWLIST)
    hosts: list[str] = []
    cidrs: list[str] = []
    for entry in allow_out:
        try:
            ip_network(entry, strict=False)
        except ValueError:
            hosts.append(entry)
        else:
            cidrs.append(entry)
    return NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=tuple(hosts), allow_cidrs=tuple(cidrs))


# One lock per sandbox in each event loop, held to widen its egress policy, so concurrent downloads don't each widen
# a stale copy and drop the other's host.
_policy_locks: weakref.WeakValueDictionary[tuple[asyncio.AbstractEventLoop, str], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)

# What exec returns when the transport drops without an exit status; exec_script retries only this.
_UNKNOWN_EXIT_CODE = -1


class _BytesReader:
    """The small ``StreamReader`` subset consumed by ``Sandbox.exec_with_output``."""

    def __init__(self, value: str | bytes | None):
        self._value = value.encode() if isinstance(value, str) else (value or b"")

    async def read(self) -> bytes:
        return self._value


class _CompletedProcess:
    """Adapt Tensorlake's completed ``CommandResult`` to the common process protocol."""

    def __init__(self, result: Any):
        self.stdout = _BytesReader(getattr(result, "stdout", ""))
        self.stderr = _BytesReader(getattr(result, "stderr", ""))
        self._exit_code = int(getattr(result, "exit_code", _UNKNOWN_EXIT_CODE))

    async def wait(self) -> int:
        return self._exit_code


class TensorlakeSandbox(VmSandbox):
    """A Tensorlake MicroVM with a systemd-managed Docker daemon.

    ``sandbox`` is an already-created ``tensorlake.sandbox.AsyncSandbox``; the provider owns
    creation, ports and reconnect.
    """

    type = "tensorlake"
    _DOCKER_READINESS_COMMAND_TIMEOUT = 10
    _DOCKER_LOG_TAIL_LINES = 40
    # The ingress proxy rejects request bodies above about 4 MB.
    _UPLOAD_CHUNK_BYTES = 512 * 1024
    _UPLOADS_IN_FLIGHT = 8
    # The join re-runs only when the transport drops; it rewrites the whole file, so a retry is safe.
    _JOIN_RETRIES = 2
    # A policy change reaches the host firewall in about 5-10 s.
    _EGRESS_PROPAGATION_TIMEOUT = 60
    _EGRESS_PROBE_INTERVAL = 2
    _EGRESS_PROBE_TIMEOUT = 5

    def __init__(
        self,
        sandbox: Any,
        *,
        tunnel_urls: Mapping[int, str] | None = None,
        network_policy: NetworkPolicy | None = None,
    ):
        self._sandbox = sandbox
        self.sandbox_id = sandbox.sandbox_id
        self.mode = SANDBOX_MODE_VM
        self.vnc_url = None
        self.tunnel_urls = dict(tunnel_urls or {})
        self.network_policy = network_policy

    async def terminate(self) -> None:
        """Delete the sandbox; one that is already gone is not an error."""
        try:
            await self._sandbox.terminate()
        except SandboxNotFoundError:
            logger.info("Tensorlake sandbox %s was already absent during terminate", self.sandbox_id)

    async def exec(self, *command: str) -> _CompletedProcess:
        """Run argv directly; commands run as ``tl-user``, so the inherited ``sudo`` prefixes stay."""
        return await self._run_command(*command, timeout=None)

    async def _run_command(self, *command: str, timeout: float | None) -> _CompletedProcess:
        try:
            result = await self._sandbox.run(command[0], args=list(command[1:]), timeout=timeout)
        except SandboxConnectionError as exc:
            return _CompletedProcess(SimpleNamespace(exit_code=_UNKNOWN_EXIT_CODE, stdout="", stderr=str(exc)))
        return _CompletedProcess(result)

    async def _exec_with_output_bounded(self, *command: str, timeout: float) -> tuple[int, str, str]:
        process = await self._run_command(*command, timeout=timeout)
        stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
        return await process.wait(), stdout.decode(), stderr.decode()

    async def setup_vm_for_gateway(self, exposed_ports: list[int] | None = None) -> None:
        """Start Docker and check for Compose v2, which ``deploy_env`` needs; ports need no firewall."""
        await self.wait_for_vm()
        exit_code, version, stderr = await self._exec_with_output_bounded(
            "sudo", "docker", "compose", "version", timeout=self._DOCKER_READINESS_COMMAND_TIMEOUT,
        )
        if exit_code != 0:
            detail = (stderr or version).strip()
            raise RuntimeError(
                "Tensorlake sandbox image must include the Docker Compose v2 plugin required by deploy-env"
                + (f": {detail}" if detail else "")
            )
        logger.info("Docker Compose ready in Tensorlake sandbox %s: %s", self.sandbox_id, version.strip())

    async def wait_for_vm(self) -> None:
        """Start the systemd ``docker.service`` and poll ``docker info`` until it answers.

        The platform can stop a ``dockerd`` started by hand, so there is no manual fallback.
        """
        deadline = time.monotonic() + self._VM_READY_TIMEOUT
        try:
            exit_code, _, stderr = await self._exec_with_output_bounded(
                "sudo", "systemctl", "start", "docker", timeout=self._DOCKER_READINESS_COMMAND_TIMEOUT * 6,
            )
            if exit_code != 0:
                logger.debug("systemctl start docker exited %s in %s: %s", exit_code, self.sandbox_id, stderr)
        except Exception as exc:
            logger.debug("systemctl start docker did not complete in %s: %s", self.sandbox_id, exc)

        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            timeout = max(0.001, min(self._DOCKER_READINESS_COMMAND_TIMEOUT, deadline - time.monotonic()))
            try:
                exit_code, _, _ = await self._exec_with_output_bounded("sudo", "docker", "info", timeout=timeout)
            except Exception as exc:
                logger.debug("Docker readiness probe failed in Tensorlake sandbox %s: %s", self.sandbox_id, exc)
                exit_code = -1
            if exit_code == 0:
                logger.info("Docker ready in Tensorlake sandbox %s after %s polls", self.sandbox_id, attempts)
                return
            sleep_seconds = min(self._VM_READY_POLL_INTERVAL, max(0.0, deadline - time.monotonic()))
            if sleep_seconds:
                await asyncio.sleep(sleep_seconds)

        try:
            _, log_tail, _ = await self._exec_with_output_bounded(
                "sudo", "journalctl", "-u", "docker", "--no-pager", "-n", str(self._DOCKER_LOG_TAIL_LINES),
                timeout=self._DOCKER_READINESS_COMMAND_TIMEOUT,
            )
        except Exception as exc:
            log_tail = f"unavailable ({type(exc).__name__}: {exc})"
        raise RuntimeError(
            f"Docker not ready in Tensorlake sandbox {self.sandbox_id} after {self._VM_READY_TIMEOUT}s "
            f"({attempts} polls); docker.service journal tail:\n{log_tail}"
        )

    async def _write_bytes_to_vm_path(self, data: bytes, vm_path: str) -> None:
        await self._upload_parts(_read_parts(io.BytesIO(data), self._UPLOAD_CHUNK_BYTES), vm_path)

    async def _write_unsigned_object(self, object_store: ObjectStore, object_url: str, vm_path: str) -> None:
        """Stream an object the store can't sign through the file API, not as base64 over exec."""
        with contextlib.closing(await asyncio.to_thread(object_store.open, object_url)) as source:
            await self._upload_parts(_read_parts(source, self._UPLOAD_CHUNK_BYTES), vm_path)

    async def _upload_parts(self, parts: AsyncIterator[bytes], vm_path: str) -> None:
        """Upload ``parts`` through the file API, several at once, then join them as root at ``vm_path``
        and check its sha256.

        The parts go to /tmp because ``vm_path`` may be in a root-owned directory. The join and
        cleanup loop over the part indexes, so the command stays short for any payload size.
        """
        stem = f"/tmp/_tl_upload_{uuid.uuid4().hex[:12]}"
        digest = hashlib.sha256()
        gate = asyncio.Semaphore(self._UPLOADS_IN_FLIGHT)
        count = 0

        async def upload(path: str, part: bytes) -> None:
            try:
                await self._sandbox.write_file(path, part)
            finally:
                gate.release()

        try:
            try:
                async with asyncio.TaskGroup() as group:
                    async for part in parts:
                        await gate.acquire()
                        digest.update(part)
                        group.create_task(upload(f"{stem}.{count}", part))
                        count += 1
            except ExceptionGroup as failed:
                raise failed.exceptions[0]
            quoted = shlex.quote(vm_path)
            written = await self.exec_script(
                f'for i in $(seq 0 {count - 1}); do cat "{stem}.$i" || exit 1; done > {quoted} && sha256sum {quoted}',
                max_retries=self._JOIN_RETRIES,
            )
        finally:
            if count:
                try:
                    await self.exec_script(f'for i in $(seq 0 {count - 1}); do rm -f "{stem}.$i"; done')
                except Exception as e:
                    logger.warning("Best-effort cleanup of %s.* failed (ignored): %s", stem, e)
        actual = written.split()[0] if written.split() else ""
        if actual != digest.hexdigest():
            raise RuntimeError(
                f"{vm_path} in Tensorlake sandbox {self.sandbox_id} does not match the upload: its sha256 is "
                f"{actual or 'missing'}, the upload's {digest.hexdigest()}"
            )

    async def apply_network_policy(self, policy: NetworkPolicy) -> None:
        """Replace the sandbox's egress policy; Tensorlake rejects the update if a host does not resolve."""
        await self._sandbox.update(network=NetworkConfig(**network_config_fields(policy)))
        self.network_policy = policy

    async def _load_tarballs(self, artifacts: list) -> None:
        """Load image tarballs after adding their signed-download hosts to a restrictive policy.

        The object store that serves image tarballs is infrastructure, not a workload
        destination. A reconnected sandbox whose policy is unknown fails closed.
        """
        policy = self.network_policy
        if policy is None:
            raise RuntimeError(
                f"Cannot load Docker images in reconnected Tensorlake sandbox {self.sandbox_id}: "
                "its applied network policy is unknown, so signed download hosts cannot be added safely"
            )
        if not policy.restricts_egress:
            await super()._load_tarballs(artifacts)
            return
        signed_urls = await self._signed_image_urls(artifacts)
        await self._allow_download_hosts(signed_urls)
        await self._load_docker_images(artifacts, signed_urls)

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        """Download an object after adding its signed-download host to a restrictive policy, as image tarballs are.
        A reconnected sandbox, whose policy is unknown, downloads as before."""
        policy = self.network_policy
        if policy is not None and policy.restricts_egress:
            signed = await asyncio.to_thread(get_config().get_object_store_at(object_url).signed_get_url, object_url)
            await self._allow_download_hosts([signed])
        await super()._download_object_to_vm(object_url, vm_path)

    async def _allow_download_hosts(self, signed_urls: list[str | None]) -> None:
        """Add the hosts of ``signed_urls`` to a restrictive applied policy and wait for them, under the sandbox's
        lock, so a concurrent download's host isn't dropped and no download starts before its host is reachable."""
        hosts = {parsed.hostname for url in signed_urls if url and (parsed := urlparse(url)).hostname}
        if not hosts:
            return
        async with _policy_locks.setdefault((asyncio.get_running_loop(), self.sandbox_id), asyncio.Lock()):
            # Another wrapper of this sandbox may have widened the policy since this one last read it.
            info = await self._sandbox.info()
            policy = network_policy_from_config(info.network_policy, sandbox_id=self.sandbox_id)
            new_hosts = sorted(hosts - set(policy.allow_hosts))
            if new_hosts:
                await self.apply_network_policy(policy.with_hosts(new_hosts))
                await self._wait_for_egress(new_hosts)

    async def _wait_for_egress(self, hosts: list[str]) -> None:
        """Wait until each host accepts a connection, since a policy change is not instant."""
        deadline = time.monotonic() + self._EGRESS_PROPAGATION_TIMEOUT
        probes = " && ".join(
            f"curl -s -o /dev/null --max-time {self._EGRESS_PROBE_TIMEOUT} {shlex.quote(f'https://{host}/')}"
            for host in hosts
        )
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                exit_code, _, _ = await self._exec_with_output_bounded("sh", "-c", probes, timeout=remaining)
            except Exception as exc:
                logger.debug("Egress probe failed in Tensorlake sandbox %s: %s", self.sandbox_id, exc)
                exit_code = -1
            if exit_code == 0:
                return
            await asyncio.sleep(min(self._EGRESS_PROBE_INTERVAL, max(0.0, deadline - time.monotonic())))
        logger.warning(
            "Tensorlake sandbox %s still cannot reach %s after %ss; loading images anyway",
            self.sandbox_id, hosts, self._EGRESS_PROPAGATION_TIMEOUT,
        )


async def _read_parts(source: Any, size: int) -> AsyncIterator[bytes]:
    while part := await asyncio.to_thread(source.read, size):
        yield part
