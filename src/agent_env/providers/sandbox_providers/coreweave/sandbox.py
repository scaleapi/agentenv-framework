"""CoreWeave Sandboxes presented through the :class:`Sandbox` contracts, in two modes.

A container sandbox runs a workload's image itself, with its own entrypoint and command; exec and file writes reach
that container. A VM sandbox is one privileged container in a Kata VM, booted from a Docker host image: its start
command readies the host (installs the tools it lacks, mounts Docker's storage, starts dockerd), and the gateway's
docker-compose, an agent's ``docker run`` and image loading run under that Docker. Every command on a VM runs as root,
so a leading ``sudo`` is dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
from cwsandbox import (
    EgressRule,
    NetworkOptions,
    SandboxError,
    SandboxExecutionError,
    SandboxRequestTimeoutError,
    SandboxStreamTruncatedError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)

from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import (
    CURL_RETRY_FLAGS,
    NetworkMode,
    NetworkPolicy,
    Sandbox,
    VmSandbox,
    push_object_over_stdin,
)
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER, SANDBOX_MODE_VM

if TYPE_CHECKING:
    from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)

DOCKER_STORAGE_LOOP = "loop"
DOCKER_STORAGE_MEMORY = "memory"
_HOST_DIR = "/run/agentenv"
_HOST_LOG = "/var/log/agentenv-host.log"
HOST_SETUP_TIMEOUT_SEC = 600
_DOCKERD_READY_TIMEOUT_SEC = 120
# How long the host-wait exec outlives the wait it runs, so the wait's own timeout message comes back.
_HOST_WAIT_GRACE_SEC = 60
_URL_DOWNLOAD_TIMEOUT_SEC = 120
# What exec returns when the stream drops without an exit status; exec_script retries only this.
_UNKNOWN_EXIT_CODE = -1
_TIMED_OUT_EXIT_CODE = 124
_WAIT_ATTEMPTS = 3


def host_setup_script(docker_storage: str, disk_gb: int) -> str:
    """The sandbox's start command: install bash, curl and (for a loop-mounted disk) e2fsprogs when the image lacks
    them, mount Docker's storage, start dockerd, and mark the host ready or failed. overlayfs can't sit on the
    sandbox's virtiofs root, so ``/var/lib/docker`` is an ext4 file loop-mounted there or, for ``memory``, a tmpfs."""
    if docker_storage == DOCKER_STORAGE_LOOP:
        tools, packages = "bash curl mkfs.ext4", "bash curl e2fsprogs"
        mount = (f"truncate -s {disk_gb}G /agentenv-docker.img && mkfs.ext4 -q -F /agentenv-docker.img && "
                 "mount -o loop /agentenv-docker.img /var/lib/docker")
    else:
        tools, packages = "bash curl", "bash curl"
        mount = f"mount -t tmpfs -o size={disk_gb}g tmpfs /var/lib/docker"
    return f"""
# The image's entrypoint, when it runs, points the client at a docker:2375 service when it starts before dockerd.
export DOCKER_HOST=unix:///var/run/docker.sock
setup() {{
  for tool in {tools}; do
    command -v "$tool" >/dev/null 2>&1 && continue
    if ! command -v apk >/dev/null 2>&1 || ! apk add --no-cache {packages}; then
      echo "the image lacks {packages} and they could not be installed; set image to a prepared host image" \\
        "(a network policy restricting egress blocks the package mirror)"
      return 1
    fi
    break
  done
  mkdir -p /var/lib/docker && {mount} || return 1
  dockerd --host=unix:///var/run/docker.sock &
  for _ in $(seq {_DOCKERD_READY_TIMEOUT_SEC}); do
    docker info >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "dockerd did not become ready"
  return 1
}}
mkdir -p {_HOST_DIR}
if setup >{_HOST_LOG} 2>&1; then : >{_HOST_DIR}/ready; else : >{_HOST_DIR}/failed; fi
trap "exit 0" TERM INT
sleep infinity &
wait
"""


_WAIT_FOR_HOST = f"""
for _ in $(seq {HOST_SETUP_TIMEOUT_SEC}); do
  [ -e {_HOST_DIR}/ready ] && exit 0
  [ -e {_HOST_DIR}/failed ] && break
  sleep 1
done
[ -e {_HOST_DIR}/failed ] || echo "host setup did not finish within {HOST_SETUP_TIMEOUT_SEC}s" >&2
tail -n 40 {_HOST_LOG} >&2
exit 1
"""


def runs_docker_host(raw: Any) -> bool | None:
    """Whether the sandbox ``raw`` describes was started as a Docker host, by ``host_setup_script``; None when the
    Sandbox API echoes no container spec to tell by, as while it prepares the image."""
    containers = list(raw.containers or ())
    if not containers:
        return None
    primary = next((container for container in containers if container.primary), containers[0])
    return any(_HOST_LOG in arg for arg in primary.args or ())


def _cidr_or_none(entry: str) -> str | None:
    try:
        return str(ip_network(entry, strict=False))
    except ValueError:
        return None


def network_options(policy: NetworkPolicy) -> NetworkOptions | None:
    """``Sandbox.run`` network options for ``policy``: none for allow-all, which leaves the runner's default egress (the
    public internet on CoreWeave's runners), and deny-all for an allowlist with no entries. An IP address listed as a
    host becomes a CIDR rule. A malformed entry raises ValueError."""
    if not policy.restricts_egress:
        return None
    rules = [EgressRule(cidr=cidr) if (cidr := _cidr_or_none(host)) else EgressRule(dns_name=host)
             for host in policy.allow_hosts]
    for entry in policy.allow_cidrs:
        if (cidr := _cidr_or_none(entry)) is None:
            raise ValueError(f"{entry!r} is not a CIDR")
        rules.append(EgressRule(cidr=cidr))
    return NetworkOptions(egress=rules) if rules else NetworkOptions(deny_egress=True)


def policy_from_egress(rules: Iterable[Any]) -> NetworkPolicy | None:
    """The policy a sandbox's effective egress rules apply, or None for rules agent-env can't represent. A rule for
    every destination, or for all of IPv4 with private ranges carved out (a runner's default), is allow-all: no
    sandbox off this machine reaches those ranges anyway."""
    hosts: list[str] = []
    cidrs: list[str] = []
    for rule in rules:
        if rule.any:
            return NetworkPolicy()
        if rule.dns_name:
            hosts.append(rule.dns_name)
            continue
        if rule.cidr is None or rule.ports:
            return None
        if rule.cidr.cidr == "0.0.0.0/0":
            return NetworkPolicy()
        if rule.cidr.except_cidrs:
            return None
        cidrs.append(rule.cidr.cidr)
    return NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=tuple(hosts), allow_cidrs=tuple(cidrs))


def allows_download(policy: NetworkPolicy | None, url: str | None) -> bool:
    """Whether a sandbox under ``policy`` can fetch ``url`` itself; an unknown policy counts as no. CoreWeave grants a
    hostname for HTTPS on port 443 only, and a ``*.domain`` entry covers exactly one label under ``domain``."""
    if url is None or policy is None:
        return False
    if not policy.restricts_egress:
        return True
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        address = ip_address(host)
    except ValueError:
        if parsed.scheme != "https" or parsed.port not in (None, 443):
            return False
        for entry in policy.allow_hosts:
            entry = entry.lower().rstrip(".")
            if entry == host:
                return True
            if entry.startswith("*.") and host.endswith(entry[1:]):
                label = host.removesuffix(entry[1:])
                if label and "." not in label:
                    return True
        return False
    networks = [cidr for entry in (*policy.allow_hosts, *policy.allow_cidrs) if (cidr := _cidr_or_none(entry))]
    return any(address in ip_network(cidr) for cidr in networks)


@dataclass(frozen=True)
class Client:
    """The SDK's ``Sandbox`` class, with the credentials and API endpoint every call a provider makes uses."""

    sandbox_cls: Any
    auth: Any
    base_url: str | None

    async def get(self, sandbox_id: str) -> Any:
        return await self.sandbox_cls.from_id(sandbox_id, auth=self.auth, base_url=self.base_url)

    async def list(self, tags: list[str]) -> list[Any]:
        return await self.sandbox_cls.list(tags=tags, auth=self.auth, base_url=self.base_url)

    async def delete(self, sandbox_id: str) -> None:
        await self.sandbox_cls.delete(sandbox_id, auth=self.auth, base_url=self.base_url, missing_ok=True)


async def _settle(process: Any, timeout: float) -> tuple[int, bytes, bytes]:
    """``process``'s exit code, stdout and stderr. A stream that dropped or was cut short is exit -1 with no stdout, so
    a caller that skips the exit code can't mistake partial output for the whole; a timed-out command is exit 124."""
    try:
        result = await process
    except SandboxError as exc:
        if isinstance(exc, (SandboxUnavailableError, SandboxStreamTruncatedError, SandboxRequestTimeoutError)):
            return _UNKNOWN_EXIT_CODE, b"", f"{type(exc).__name__}: {exc}".encode()
        if isinstance(exc, SandboxTimeoutError):
            return _TIMED_OUT_EXIT_CODE, b"", f"command timed out after {timeout}s".encode()
        raise
    return result.returncode, result.stdout_bytes, result.stderr_bytes


async def _sent(send: Callable[[], Any], finished: asyncio.Future) -> bool:
    """Send one stdin write or close; True when the command ended first, so nothing more is worth sending. The SDK
    queues stdin with a bound, and a command that has ended stops reading it, so a send could otherwise wait forever."""
    try:
        sending = asyncio.ensure_future(send())
    except SandboxExecutionError:
        return True
    await asyncio.wait({sending, finished}, return_when=asyncio.FIRST_COMPLETED)
    if not sending.done():
        sending.cancel()
        return True
    if isinstance(sending.exception(), SandboxExecutionError):
        return True
    sending.result()
    return False


class _BytesReader:
    """The small ``StreamReader`` subset consumed by ``Sandbox.exec_with_output``."""

    def __init__(self, value: bytes):
        self._value = value

    async def read(self) -> bytes:
        return self._value


class _CompletedProcess:
    """A finished CoreWeave exec in the common process protocol."""

    def __init__(self, exit_code: int, stdout: bytes, stderr: bytes):
        self.stdout = _BytesReader(stdout)
        self.stderr = _BytesReader(stderr)
        self._exit_code = exit_code

    async def wait(self) -> int:
        return self._exit_code


class _CoreWeaveSandbox(Sandbox):
    """A CoreWeave Sandbox. ``command_timeout`` is every command's execution budget, normally the sandbox's lifetime:
    the SDK otherwise bounds a command to five minutes."""

    type = "coreweave"

    def __init__(
        self,
        sandbox: Any,
        *,
        client: Client,
        tunnel_urls: dict[int, str],
        network_policy: NetworkPolicy | None,
        command_timeout: int,
    ):
        self._sandbox = sandbox
        self._client = client
        self.command_timeout = command_timeout
        self.sandbox_id = sandbox.sandbox_id
        self.vnc_url = None
        self.tunnel_urls = tunnel_urls
        self.network_policy = network_policy

    async def terminate(self) -> None:
        await self._client.delete(self.sandbox_id)

    async def _run(self, *command: str, timeout: float) -> tuple[int, bytes, bytes]:
        return await _settle(self._sandbox.exec(list(command), timeout_seconds=timeout), timeout)

    async def exec(self, *command: str) -> _CompletedProcess:
        """Run argv to completion. A lost stream is exit -1, which a VM's ``exec_script`` retries."""
        return _CompletedProcess(*await self._run(*command, timeout=self.command_timeout))

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        exit_code, stdout, stderr = await self._run(*args, timeout=self.command_timeout)
        return exit_code, stdout.decode(errors="replace"), stderr.decode(errors="replace")


class CoreWeaveContainerSandbox(_CoreWeaveSandbox):
    """A CoreWeave Sandbox running a workload's image itself, with no Docker. Commands run as the image's user. Files
    go through the Sandbox API's file write, which needs ``sh``, ``mkdir``, ``head`` and ``tar`` in the image and takes
    at most about 256 MiB."""

    mode = SANDBOX_MODE_CONTAINER

    async def write_file_from_text(self, content: str, destination_path: str) -> None:
        await self._sandbox.write_file(destination_path, content.encode())

    async def write_file_from_object(self, object_url: str, destination_path: str) -> None:
        store = get_config().get_object_store_at(object_url)
        await self._sandbox.write_file(destination_path, await asyncio.to_thread(_read_object, store, object_url))

    async def write_file_from_url(self, url: str, destination_path: str) -> None:
        """Fetched here and written over the Sandbox API, since the image need not have curl."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=_URL_DOWNLOAD_TIMEOUT_SEC) as http:
            response = await http.get(url)
            response.raise_for_status()
        await self._sandbox.write_file(destination_path, response.content)


def _read_object(store: ObjectStore, object_url: str) -> bytes:
    with contextlib.closing(store.open(object_url)) as body:
        return body.read()


class CoreWeaveSandbox(_CoreWeaveSandbox, VmSandbox):
    """A CoreWeave Sandbox that is a Docker host."""

    mode = SANDBOX_MODE_VM

    async def _run(self, *command: str, timeout: float) -> tuple[int, bytes, bytes]:
        return await super()._run(*(command[1:] if command[:1] == ("sudo",) else command), timeout=timeout)

    async def _exec_with_stdin(self, script: str, stdin: AsyncIterator[bytes]) -> tuple[int, str, str]:
        """Run ``script`` with bash as root, its standard input the bytes ``stdin`` yields."""
        process = self._sandbox.exec(["bash", "-c", script], stdin=True, timeout_seconds=self.command_timeout)
        finished = asyncio.ensure_future(_settle(process, self.command_timeout))
        try:
            async for piece in stdin:
                if await _sent(lambda: process.stdin.write(piece), finished):
                    break
            else:
                await _sent(process.stdin.close, finished)
            exit_code, stdout, stderr = await finished
        except BaseException:
            # The command reads to the end of its input, so closing it ends the command rather than its lifetime.
            process.stdin.close()
            finished.cancel()
            raise
        return exit_code, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    async def wait_for_vm(self) -> None:
        """Wait for the start command to ready Docker, and raise what it reported if it couldn't."""
        for _ in range(_WAIT_ATTEMPTS):
            exit_code, stdout, stderr = await self._run("sh", "-c", _WAIT_FOR_HOST,
                                                        timeout=HOST_SETUP_TIMEOUT_SEC + _HOST_WAIT_GRACE_SEC)
            if exit_code != _UNKNOWN_EXIT_CODE:
                break
        if exit_code != 0:
            output = (stderr or stdout).decode(errors="replace").strip()
            raise RuntimeError(f"Docker not ready in CoreWeave sandbox {self.sandbox_id}: {output[-1500:]}")
        logger.info("Docker ready in CoreWeave sandbox %s", self.sandbox_id)

    async def setup_vm_for_gateway(self, exposed_ports: list[int] | None = None) -> None:
        """Ready Docker; CoreWeave routes exposed ports to the sandbox, so no firewall."""
        await self.wait_for_vm()

    # Bytes go over exec stdin as root, which reaches any path and needs no egress, so they arrive whatever the
    # sandbox's egress policy, which is set when the sandbox is created.

    async def _write_bytes_to_vm_path(self, data: bytes, vm_path: str) -> None:
        async def pieces() -> AsyncIterator[bytes]:
            yield data

        exit_code, stdout, stderr = await self._exec_with_stdin(f"cat > {shlex.quote(vm_path)}", pieces())
        if exit_code != 0:
            raise RuntimeError(f"Writing {vm_path} failed (exit {exit_code}): {(stderr or stdout)[-1500:]}")

    async def _write_unsigned_object(self, object_store: ObjectStore, object_url: str, vm_path: str) -> None:
        await push_object_over_stdin(self, object_store, object_url, vm_path)

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        object_store = get_config().get_object_store_at(object_url)
        signed = await asyncio.to_thread(object_store.signed_get_url, object_url)
        if allows_download(self.network_policy, signed):
            await self.exec_script(f"curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(signed)} -o {shlex.quote(vm_path)}")
            return
        await push_object_over_stdin(self, object_store, object_url, vm_path)

    async def _load_tarballs(self, artifacts: list) -> None:
        """Load image tarballs, pushing over stdin those the sandbox's egress policy can't download."""
        signed_urls = await self._signed_image_urls(artifacts)
        reachable = [url if allows_download(self.network_policy, url) else None for url in signed_urls]
        await self._load_docker_images(artifacts, reachable)
