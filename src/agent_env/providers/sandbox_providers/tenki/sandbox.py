"""A Tenki Sandbox VM presented through the :class:`VmSandbox` contract.

Tenki VMs boot Tenki images, not OCI images, so the gateway's docker-compose, an agent's ``docker run`` and image
loading run under Docker inside the VM. Every command runs as root through Tenki's privileged exec. Tenki applies the
egress policy outside the VM, so one policy covers ``dockerd`` and every container it runs.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import shlex
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from tenki import InvalidStateError, SandboxError, SessionNotFoundError, SessionTerminatedError

from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import (
    CURL_RETRY_FLAGS,
    NetworkMode,
    NetworkPolicy,
    VmSandbox,
    push_object_over_stdin,
)
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM

if TYPE_CHECKING:
    from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)

# Installs Docker and Compose v2 only when the image lacks them, then starts the daemon. Every fd of the fallback
# dockerd is redirected, since a child holding the exec's output stream keeps the exec from returning.
DOCKER_BOOTSTRAP = r"""
set -e
sysctl -qw vm.overcommit_memory=1 || true
if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get -o DPkg::Lock::Timeout=180 update -qq
  apt-get -o DPkg::Lock::Timeout=180 install -y -qq --no-install-recommends \
    docker.io docker-compose-v2 docker-buildx curl ca-certificates >/dev/null
fi
if ! iptables -t nat -L >/dev/null 2>&1; then
  echo "this VM's kernel has no netfilter, which Docker networking needs; boot from a snapshot taken on a VM" \
    "where Docker runs" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1 && ! systemctl start docker >/dev/null 2>&1; then
  setsid nohup dockerd >/var/log/dockerd.log 2>&1 </dev/null &
fi
for _ in $(seq 90); do
  docker info >/dev/null 2>&1 && exit 0
  sleep 1
done
echo "docker daemon did not become ready" >&2
tail -n 40 /var/log/dockerd.log >&2 2>/dev/null || true
exit 1
"""

BOOTSTRAP_TIMEOUT_SEC = 900
ENDED_STATES = frozenset({"TERMINATING", "TERMINATED", "GONE"})
# What exec returns when the stream drops without an exit status; exec_script retries only this.
_UNKNOWN_EXIT_CODE = -1
_TIMED_OUT_EXIT_CODE = 124
_STDIN_QUEUE_LIMIT = 16
# How the SDK reports a run stream that ended before the command did.
_STREAM_CLOSED = "stream closed before exit"
# Tenki reports a command killed by a signal, or one that never started, as exit -1; agent-env reads -1 as a lost
# stream, so these get the shell's codes instead: 128 + the signal's number, or 127.
_SIGNAL_NUMBERS = {"hangup": 1, "interrupt": 2, "quit": 3, "aborted": 6, "killed": 9, "segmentation fault": 11,
                   "broken pipe": 13, "terminated": 15}
_NOT_STARTED_EXIT_CODE = 127
_REAP_TIMEOUT_SEC = 10


def allows_host(policy: NetworkPolicy, host: str) -> bool:
    """Whether ``policy`` lets the VM reach ``host`` by Tenki's rules: an exact name, or a ``*.domain`` entry, which
    covers exactly one label under ``domain``."""
    if not policy.restricts_egress:
        return True
    host = host.lower().rstrip(".")
    for entry in policy.allow_hosts:
        entry = entry.lower().rstrip(".")
        if entry == host:
            return True
        if entry.startswith("*."):
            suffix = entry[1:]
            label = host.removesuffix(suffix)
            if host.endswith(suffix) and label and "." not in label:
                return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in ipaddress.ip_network(cidr, strict=False) for cidr in policy.allow_cidrs)


def policy_from_info(info: Any) -> NetworkPolicy:
    """The policy a Tenki session applies. Outbound off is an allowlist with no entries; empty allowlists with outbound
    on leave egress open."""
    if not info.outbound_enabled:
        return NetworkPolicy(mode=NetworkMode.ALLOWLIST)
    egress = info.egress
    if not egress.allow_domains and not egress.allow_cidrs:
        return NetworkPolicy()
    return NetworkPolicy(
        mode=NetworkMode.ALLOWLIST, allow_hosts=tuple(egress.allow_domains), allow_cidrs=tuple(egress.allow_cidrs),
    )


def _is_transient(exc: BaseException) -> bool:
    """A dropped stream or an unavailable data plane, which a rerun can get past; a terminated session is not."""
    if isinstance(exc, SessionTerminatedError):
        return False
    return bool(getattr(exc, "retryable", False)) or _STREAM_CLOSED in str(exc)


def _exit_status(result: Any) -> tuple[int, bytes]:
    """``result``'s exit code and a note for its stderr, with Tenki's -1 for a signal or a failed start mapped."""
    if result.exit_code != _UNKNOWN_EXIT_CODE:
        return result.exit_code, b""
    if result.signal:
        return 128 + _SIGNAL_NUMBERS.get(str(result.signal).lower(), 0), f"\nkilled by signal: {result.signal}".encode()
    return _NOT_STARTED_EXIT_CODE, f"\ncommand did not run: {result.reason or 'unknown reason'}".encode()


class _BytesReader:
    """The small ``StreamReader`` subset consumed by ``Sandbox.exec_with_output``."""

    def __init__(self, value: bytes):
        self._value = value

    async def read(self) -> bytes:
        return self._value


class _CompletedProcess:
    """A finished Tenki exec in the common process protocol."""

    def __init__(self, stdout: bytes, stderr: bytes, exit_code: int):
        self.stdout = _BytesReader(stdout)
        self.stderr = _BytesReader(stderr)
        self._exit_code = exit_code

    async def wait(self) -> int:
        return self._exit_code


class TenkiSandbox(VmSandbox):
    """A Tenki VM with Docker; its client is its provider's, shared with the provider's other sandboxes.
    ``command_timeout`` is every command's execution budget, normally the VM's remaining lifetime: Tenki bounds a
    command that declares none to one hour."""

    type = "tenki"
    _DOCKER_PROBE_TIMEOUT = 15

    def __init__(
        self,
        session: Any,
        *,
        tunnel_urls: dict[int, str],
        network_policy: NetworkPolicy | None,
        command_timeout: int,
    ):
        self._session = session
        self.command_timeout = command_timeout
        self.sandbox_id = session.id
        self.mode = SANDBOX_MODE_VM
        self.vnc_url = None
        self.tunnel_urls = tunnel_urls
        self.network_policy = network_policy

    async def terminate(self) -> None:
        await close_session(self._session)

    async def release(self) -> None:
        await release_session(self._session)

    async def _current_state(self) -> str:
        return await _session_state(self._session)

    async def exec(self, *command: str) -> _CompletedProcess:
        """Run argv to completion as root, so a leading ``sudo`` is dropped. A lost stream is exit -1, which
        ``exec_script`` retries."""
        return await self._run(*command, timeout=self.command_timeout)

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        """An exec whose stream was cut (exit -1) returns no stdout, so a caller that skips the exit code can't mistake
        partial output for the whole."""
        exit_code, stdout, stderr = await self._exec_with_output_bounded(*args, timeout=self.command_timeout)
        return exit_code, "" if exit_code == _UNKNOWN_EXIT_CODE else stdout, stderr

    async def _run(self, *command: str, timeout: int) -> _CompletedProcess:
        argv = list(command[1:] if command[:1] == ("sudo",) else command)
        try:
            result = await self._session.exec(*argv, timeout=timeout, privileged=True)
        except SandboxError as exc:
            if not _is_transient(exc):
                raise
            return _CompletedProcess(b"", f"{type(exc).__name__}: {exc}".encode(), _UNKNOWN_EXIT_CODE)
        if result.timed_out:
            note = f"\ncommand timed out after {timeout}s".encode()
            return _CompletedProcess(result.stdout, result.stderr + note, _TIMED_OUT_EXIT_CODE)
        exit_code, note = _exit_status(result)
        return _CompletedProcess(result.stdout, result.stderr + note, exit_code)

    async def _exec_with_output_bounded(self, *command: str, timeout: int) -> tuple[int, str, str]:
        process = await self._run(*command, timeout=timeout)
        stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
        return await process.wait(), stdout.decode(errors="replace"), stderr.decode(errors="replace")

    async def _exec_with_stdin(self, script: str, stdin: AsyncIterator[bytes]) -> tuple[int, str, str]:
        """Run ``script`` with bash as root, its standard input the bytes ``stdin`` yields."""
        process = await self._session.start("bash", "-c", script, timeout=self.command_timeout, privileged=True)
        try:
            async for piece in stdin:
                await process.write_stdin(piece)
                if await self._wait_for_stdin_drain(process):
                    break
            await process.close_stdin()
            result = await process.wait()
        except BaseException:
            await self._reap(process)
            raise
        exit_code, note = _exit_status(result)
        return exit_code, result.stdout_text, result.stderr_text + note.decode()

    @staticmethod
    async def _reap(process: Any) -> None:
        """Close stdin, kill and wait for a process whose input failed, so the remote command and the SDK stream end
        now rather than at the command's timeout. Its own failures are logged, never raised over the original one; a
        cancellation is honored once every step has run."""
        cancelled = False
        for step in (process.close_stdin, process.kill, lambda: process.wait(timeout=_REAP_TIMEOUT_SEC)):
            try:
                await step()
            except asyncio.CancelledError:
                cancelled = True
            except Exception as exc:  # noqa: BLE001 - the input's failure is the one to report
                logger.debug("Reaping a failed stdin exec: %s: %s", type(exc).__name__, exc)
        if cancelled:
            raise asyncio.CancelledError

    @staticmethod
    async def _wait_for_stdin_drain(process: Any) -> bool:
        """Stdin is queued as it is written, so a fast reader would buffer a whole object in memory; wait for the
        stream to catch up. True once the command has ended, so the caller stops reading input nobody will consume;
        a stream that ended in error raises it."""
        queue = getattr(process, "_outbound", None)
        done = getattr(process, "_done", None)
        if not isinstance(queue, asyncio.Queue) or not isinstance(done, asyncio.Event):
            return False
        while not done.is_set() and queue.qsize() > _STDIN_QUEUE_LIMIT:
            await asyncio.sleep(0.005)
        if done.is_set() and (error := getattr(process, "_error", None)) is not None:
            raise error
        return done.is_set()

    async def wait_for_vm(self) -> None:
        """Install Docker and Compose v2 if the image lacks them, start the daemon and wait for it."""
        exit_code, stdout, stderr = await self._exec_with_output_bounded(
            "bash", "-c", DOCKER_BOOTSTRAP, timeout=BOOTSTRAP_TIMEOUT_SEC,
        )
        if exit_code != 0:
            raise RuntimeError(f"Docker not ready in Tenki VM {self.sandbox_id}: {(stderr or stdout).strip()[-1500:]}")
        logger.info("Docker ready in Tenki VM %s", self.sandbox_id)

    async def setup_vm_for_gateway(self, exposed_ports: list[int] | None = None) -> None:
        """Ready Docker and check for Compose v2, which ``deploy_env`` needs; Tenki routes exposed ports, so no
        firewall."""
        await self.wait_for_vm()
        exit_code, stdout, stderr = await self._exec_with_output_bounded(
            "docker", "compose", "version", timeout=self._DOCKER_PROBE_TIMEOUT,
        )
        if exit_code != 0:
            raise RuntimeError(f"Tenki VM {self.sandbox_id} has no Docker Compose v2: {(stderr or stdout).strip()}")

    # Bytes go over exec stdin as root, which reaches any path on the VM and needs no egress, so they arrive whatever
    # the VM's egress policy, which is set when the VM is created.

    async def _write_bytes_to_vm_path(self, data: bytes, vm_path: str) -> None:
        async def pieces() -> AsyncIterator[bytes]:
            yield data

        exit_code, stdout, stderr = await self._exec_with_stdin(f"cat > {shlex.quote(vm_path)}", pieces())
        if exit_code != 0:
            raise RuntimeError(f"Writing {vm_path} failed (exit {exit_code}): {(stderr or stdout)[-1500:]}")

    async def _write_unsigned_object(self, object_store: ObjectStore, object_url: str, vm_path: str) -> None:
        await push_object_over_stdin(self, object_store, object_url, vm_path)

    def _can_download(self, url: str | None) -> bool:
        """Whether the VM can fetch ``url`` itself; an unknown policy counts as no."""
        if url is None or self.network_policy is None:
            return False
        host = urlparse(url).hostname
        return host is not None and allows_host(self.network_policy, host)

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        object_store = get_config().get_object_store_at(object_url)
        signed = await asyncio.to_thread(object_store.signed_get_url, object_url)
        if self._can_download(signed):
            await self.exec_script(f"curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(signed)} -o {shlex.quote(vm_path)}")
            return
        await push_object_over_stdin(self, object_store, object_url, vm_path)

    async def _load_tarballs(self, artifacts: list) -> None:
        """Load image tarballs, pushing over stdin those the VM's egress policy can't download."""
        signed_urls = await self._signed_image_urls(artifacts)
        reachable = [url if self._can_download(url) else None for url in signed_urls]
        await self._load_docker_images(artifacts, reachable)


async def close_session(session: Any) -> None:
    """Terminate a session's VM. One another handle already terminated, while this one's cached state still says
    running, is not an error."""
    try:
        await session.close_if_open()
    except SessionNotFoundError:
        logger.info("Tenki session %s was already gone at terminate", session.id)
    except (InvalidStateError, SessionTerminatedError):
        if await _session_state(session) not in ENDED_STATES:
            raise
        logger.info("Tenki session %s was already terminating at terminate", session.id)
    await release_session(session)


async def release_session(session: Any) -> None:
    """Stop the session's data-plane credential renewal, which close_if_open leaves to the caller for a session
    that is already ending."""
    reset = getattr(session, "_reset_data_plane", None)
    if reset is not None:
        await reset()


async def _session_state(session: Any) -> str:
    try:
        return (await session.refresh()).state
    except SessionNotFoundError:
        return "GONE"
