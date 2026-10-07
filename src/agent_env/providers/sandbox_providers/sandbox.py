"""Sandbox interfaces for agent-env compute backends."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import posixpath
import shlex
import tempfile
import time
import uuid
import warnings
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Optional

from agent_env.config import get_config
from agent_env.utils.paths import validate_relative_filename

if TYPE_CHECKING:
    from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)

# curl flags that make in-VM downloads resilient to transient DNS / network
# blips inside the sandbox VM — most notably `curl: (6) Could not resolve host`
# (exit 6), which we've seen cascade into thousands of step failures when
# per-node DNS throughput is throttled on busy worker nodes.
#
# curl does NOT retry exit 6 by default, and `--retry-connrefused` only covers
# connection-refused — neither retries a resolution failure. `--retry-all-errors`
# is required to retry on exit 6 (curl >= 7.71; the Ubuntu 22.04 containerdisk
# ships 7.81). Without it, a single transient DNS hiccup escalates straight to
# runner-level step retries instead of being absorbed locally in ~seconds.
#
# IMPORTANT: only safe for downloads written to a file via `-o`. curl cannot
# rewind data it has already streamed to stdout, so retrying a
# `curl ... | gunzip | docker load` pipeline would append a second response to
# the partial bytes already consumed, corrupting the stream. Pipe consumers must
# download to a temp file first, then read the file (see load_docker_images).
CURL_RETRY_FLAGS = "--retry 5 --retry-all-errors --retry-delay 1"
# Docker label on what a step starts on a sandbox's Docker host (containers, images, networks), valued with the
# sandbox id, so a sandbox that shares its host (the local one) can remove its own when it terminates.
SANDBOX_LABEL = "agentenv.sandbox"

# At most this many image signs in flight per event loop: each holds a worker thread, for up to a
# remote signer's whole retry window, and a connection from its session's pool.
_CONCURRENT_SIGNS = 8
_sign_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()

# A VM file is read off its host one range per exec, whose base64 every provider's exec returns in one response. A
# provider serves each exec a fraction of its bandwidth, so many ranges are in flight.
_READ_RANGE_BYTES = 4 * 1024 * 1024
_READS_IN_FLIGHT = 16


class NetworkPolicyUnsupportedError(NotImplementedError):
    """A backend was asked to enforce a policy it cannot."""


class NetworkMode(str, Enum):
    ALLOW_ALL = "allow_all"
    ALLOWLIST = "allowlist"


@dataclass(frozen=True)
class NetworkPolicy:
    """Outbound egress intent for a sandbox; hostnames are the primary form."""

    mode: NetworkMode = NetworkMode.ALLOW_ALL
    allow_hosts: tuple[str, ...] = ()
    allow_cidrs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # NetworkMode subclasses str, so an uncoerced mode misreads every `is` comparison.
        object.__setattr__(self, "mode", NetworkMode(self.mode))
        for field in ("allow_hosts", "allow_cidrs"):
            value = getattr(self, field)
            if isinstance(value, (str, bytes)):
                raise ValueError(f"{field} must be a sequence of entries, not the string {value!r}")
            object.__setattr__(self, field, tuple(value))
        if "*" in self.allow_hosts:
            raise ValueError("allow_hosts cannot contain a bare '*'; use mode=allow_all instead")
        if self.mode is NetworkMode.ALLOW_ALL and (self.allow_hosts or self.allow_cidrs):
            raise ValueError("allow_all permits everything; entries would be silently ignored")

    @property
    def restricts_egress(self) -> bool:
        return self.mode is not NetworkMode.ALLOW_ALL

    def with_hosts(self, hosts: Iterable[str]) -> "NetworkPolicy":
        """Copy with ``hosts`` unioned in; a no-op for ALLOW_ALL, which needs no list."""
        if self.mode is not NetworkMode.ALLOWLIST:
            return self
        merged = list(self.allow_hosts) + [h for h in hosts if h not in self.allow_hosts]
        return replace(self, allow_hosts=tuple(merged))

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "allow_hosts": list(self.allow_hosts),
            "allow_cidrs": list(self.allow_cidrs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NetworkPolicy":
        if not isinstance(data, dict):
            raise ValueError(f"network policy must be a mapping, got {type(data).__name__}")
        raw = data.get("mode", NetworkMode.ALLOW_ALL.value)
        try:
            mode = NetworkMode(raw)
        except ValueError:
            raise ValueError(f"Unknown network policy mode {raw!r}; expected one of {[m.value for m in NetworkMode]}")
        # Raw, not tuple()'d: __post_init__ must see a string to reject it.
        return cls(
            mode=mode,
            allow_hosts=data.get("allow_hosts") or (),
            allow_cidrs=data.get("allow_cidrs") or (),
        )


class Sandbox(ABC):
    """Universal sandbox contract — anything that can host a process and expose ports."""

    type: str
    mode: str  # "vm" or "container"
    sandbox_id: str
    tunnel_urls: dict[int, str]
    vnc_url: str | None
    # Effective, as applied by the backend; None when it cannot tell us.
    network_policy: NetworkPolicy | None = None
    # Host IPs published ports bind to; empty binds every interface.
    host_ips: tuple[str, ...] = ()

    _VM_READY_TIMEOUT = 1200      # wait_for_vm wall-clock budget (s)
    _VM_READY_POLL_INTERVAL = 30  # sparse polling (s)

    def host_port(self, port: int) -> int:
        """The host-side port a published container port is reachable on.

        Identity for any backend that gives a deployment its own network namespace.
        Backends that share a host override this to avoid collisions between concurrent
        deployments; container ports are unaffected either way.
        """
        return port

    @abstractmethod
    async def terminate(self) -> None:
        """Terminate the sandbox."""

    async def exec(self, *command: str) -> Any:
        """Execute a command in the sandbox. Returns a ContainerProcess-like object with stdout, stderr, wait()."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support exec")

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        """Execute command and return (exit_code, stdout, stderr)."""
        process = await self.exec(*args)
        stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
        exit_code = await process.wait()
        return exit_code, stdout.decode(), stderr.decode()

    async def write_file_from_object(self, object_url: str, destination_path: str) -> None:
        """Write the object store's object at ``object_url`` into the agent process's filesystem at
        ``destination_path``."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support write_file_from_object")

    async def write_file_from_s3(self, s3_url: str, destination_path: str) -> None:
        """Deprecated: ``write_file_from_object``."""
        warnings.warn(
            "Sandbox.write_file_from_s3 is deprecated; use write_file_from_object", DeprecationWarning, stacklevel=2
        )
        await self.write_file_from_object(s3_url, destination_path)

    async def write_file_from_url(self, url: str, destination_path: str) -> None:
        """Download an HTTP(S) URL into the agent process's filesystem at destination_path."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support write_file_from_url")

    async def write_file_from_text(self, content: str, destination_path: str) -> None:
        """Write inline text content into the agent process's filesystem at destination_path."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support write_file_from_text")


class VmSandbox(Sandbox):
    """VM-style sandbox with a Docker daemon and shell access inside."""

    _VM_READY_TIMEOUT = 180
    _VM_READY_POLL_INTERVAL = 5

    @property
    def container_name(self) -> str:
        """Docker name of the agent container this sandbox runs. Fixed ``agent-api`` by default —
        one agent per VM, so callers can find it by name. A backend that packs multiple agents onto
        one Docker host (the local sandbox) overrides this per-sandbox so they don't collide and
        teardown removes only its own container."""
        return "agent-api"

    async def exec_script(self, script: str, *, max_retries: int = 0) -> str:
        """Execute a bash script in the sandbox.

        Set ``max_retries`` > 0 only for idempotent scripts. Retries are gated
        on exit code -1, which a provider's exec client returns when the server
        closes the websocket without sending an exit frame (e.g. a control
        plane wrapping a transient port-forward 500 as a generic error). Real
        script failures (positive exit codes) raise immediately.
        """
        for attempt in range(max_retries + 1):
            exit_code, stdout, stderr = await self.exec_with_output("sudo", "bash", "-c", script)
            if exit_code == 0:
                return stdout
            if exit_code != -1 or attempt == max_retries:
                raise RuntimeError(f"Script failed (exit {exit_code}):\nstdout: {stdout[-1500:]}\nstderr: {stderr[-1500:]}")
            backoff = 2 ** attempt
            logger.warning(
                f"exec_script exit -1 (transient server error), retrying in {backoff}s "
                f"(attempt {attempt + 1}/{max_retries + 1}); stderr tail: {stderr[-200:]!r}"
            )
            await asyncio.sleep(backoff)
        raise AssertionError("unreachable")

    async def setup_vm_for_gateway(self, exposed_ports: Optional[list[int]] = None) -> None:
        """Wait for VM and open firewall ports. Docker is pre-installed in the containerdisk image."""
        await self.wait_for_vm()
        if exposed_ports:
            logger.info("Configuring firewall for exposed ports...")
            for port in exposed_ports:
                await self.exec_script(f"iptables -I INPUT -p tcp --dport {port} -j ACCEPT || true")

    async def wait_for_vm(self) -> None:
        """Poll until the VM's /exec endpoint is reachable, bounded by
        ``_VM_READY_TIMEOUT`` wall-clock seconds."""
        logger.info(
            f"Waiting for VM /exec to become reachable "
            f"(budget {self._VM_READY_TIMEOUT}s wall-clock)..."
        )
        deadline = time.monotonic() + self._VM_READY_TIMEOUT
        attempts = 0
        last_err: BaseException | None = None
        while time.monotonic() < deadline:
            attempts += 1
            try:
                # Probe both bash and the Docker data dir — confirms /exec is
                # routing, the rootfs is mounted, and Docker is installed
                # before the deploy starts loading images into the daemon.
                await self.exec_script("ls /var/lib/docker > /dev/null && echo ready")
                elapsed = self._VM_READY_TIMEOUT - max(0.0, deadline - time.monotonic())
                logger.info(f"VM /exec reachable after {elapsed:.0f}s ({attempts} attempts)")
                return
            except Exception as e:
                last_err = e
            # Sleep is unconditional (every iteration); the log just throttles
            # to one line per 5 attempts so the deploy log isn't flooded.
            if attempts % 5 == 0:
                elapsed = self._VM_READY_TIMEOUT - max(0.0, deadline - time.monotonic())
                logger.info(
                    f"  Still waiting for VM /exec... ({elapsed:.0f}s elapsed, "
                    f"{attempts} attempts)"
                )
            await asyncio.sleep(self._VM_READY_POLL_INTERVAL)
        raise RuntimeError(
            f"VM {self.sandbox_id} /exec not reachable after "
            f"{self._VM_READY_TIMEOUT}s wall-clock ({attempts} attempts; "
            f"last error: {type(last_err).__name__}: {last_err}) — "
            f"platform reports Running but /exec proxy not routing to the VM"
        )

    async def load_docker_images(self, artifacts: list) -> None:
        """Load Docker images from DockerImageArtifacts into the sandbox in parallel."""
        if not artifacts:
            return
        await self._load_docker_images(artifacts, await self._signed_image_urls(artifacts))

    @staticmethod
    async def _signed_image_urls(artifacts: list) -> list[str | None]:
        """Each image tarball's signed URL, or None where the store cannot sign one. Signed
        concurrently and off the event loop, since a remote signer is a network round trip."""
        logger.info(f"Loading {len(artifacts)} Docker image(s) into the sandbox...")
        config = get_config()
        slots = _sign_slots.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(_CONCURRENT_SIGNS))
        failed = False

        async def sign(artifact) -> str | None:
            nonlocal failed
            async with slots:
                if failed:
                    return None
                try:
                    store = config.get_object_store_at(artifact.tar_gz_object_url)
                    return await asyncio.to_thread(store.signed_get_url, artifact.tar_gz_object_url)
                except BaseException:
                    failed = True
                    raise

        return list(await asyncio.gather(*(sign(artifact) for artifact in artifacts)))

    async def _load_docker_images(self, artifacts: list, signed_urls: list[str | None]) -> None:
        load_commands = []
        for idx, (artifact, signed) in enumerate(zip(artifacts, signed_urls, strict=True)):
            tmp_tar = f"/tmp/_docker_image_{self.sandbox_id}_{idx}.tar.gz"
            if signed is not None:
                # Download to a file first (retry-safe with -o); a `curl | ... docker load`
                # pipe can't be retried without corrupting the stream (curl won't rewind).
                load_commands.append(
                    f'(curl -fsSL {CURL_RETRY_FLAGS} "{signed}" -o {shlex.quote(tmp_tar)} '
                    f"&& gunzip -c {shlex.quote(tmp_tar)} | docker load && rm -f {shlex.quote(tmp_tar)})"
                )
            else:
                await self._download_object_to_vm(artifact.tar_gz_object_url, tmp_tar)
                load_commands.append(
                    f"(gunzip -c {shlex.quote(tmp_tar)} | docker load && rm -f {shlex.quote(tmp_tar)})"
                )
            logger.info(f"  Queued: {artifact.image_name}")
        await self.exec_script(" & ".join(load_commands) + " & wait", max_retries=2)

        logger.info("Verifying Docker images...")
        exit_code, stdout, stderr = await self.exec_with_output("sudo", "docker", "images")
        if exit_code != 0:
            raise RuntimeError(f"docker images failed: {stderr}")
        for artifact in artifacts:
            base_name = artifact.image_name.split(":")[0]
            if base_name not in stdout:
                raise RuntimeError(f"{artifact.image_name} image not found. stdout: {stdout}")
        logger.info("  All images loaded successfully")

    async def load_object_file(self, object_url: str, destination_path: str) -> None:
        """Download the object store's object at ``object_url`` onto the VM host at ``destination_path``."""
        await self._download_object_to_vm(object_url, destination_path)

    async def load_s3_file(self, s3_url: str, destination_path: str) -> None:
        """Deprecated: ``load_object_file``."""
        warnings.warn("VmSandbox.load_s3_file is deprecated; use load_object_file", DeprecationWarning, stacklevel=2)
        await self.load_object_file(s3_url, destination_path)

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        """Place object_url onto the VM host at vm_path, backend-agnostically."""
        object_store = get_config().get_object_store_at(object_url)
        signed = await asyncio.to_thread(object_store.signed_get_url, object_url)
        if signed is not None:
            await self.exec_script(f"curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(signed)} -o {shlex.quote(vm_path)}")
        else:
            await self._write_unsigned_object(object_store, object_url, vm_path)

    async def _write_unsigned_object(self, object_store: ObjectStore, object_url: str, vm_path: str) -> None:
        """Place an object the store cannot sign a URL for: its bytes, streamed over exec."""
        await self._write_bytes_to_vm_path(await asyncio.to_thread(object_store.get, object_url), vm_path)

    async def _remove_vm_temp_file(self, *vm_paths: str) -> None:
        try:
            await self.exec_script(f"rm -f {' '.join(shlex.quote(p) for p in vm_paths)}")
        except Exception as e:
            logger.warning(f"Best-effort cleanup of {', '.join(vm_paths)} failed (ignored): {e}")

    async def docker_cp(self, source: str, destination: str, *, remove_source: bool = False) -> None:
        """``docker cp source destination``, one side ``container:path``. The paths go as arguments, not
        script text, so a sandbox that maps its paths (the local one maps /app) maps only the host side.
        ``remove_source`` deletes the copied host file in the same exec."""
        script = 'docker cp "$1" "$2"' + (' && rm -f "$1"' if remove_source else "")
        exit_code, stdout, stderr = await self.exec_with_output("sudo", "bash", "-c", script, "docker-cp", source, destination)
        if exit_code != 0:
            raise RuntimeError(
                f"docker cp {source} {destination} failed (exit {exit_code}):\nstdout: {stdout[-1500:]}\nstderr: {stderr[-1500:]}"
            )

    async def _copy_into_container(self, vm_path: str, destination_path: str) -> None:
        parent = os.path.dirname(destination_path)
        if parent:
            await self.exec_script(f"docker exec -u 0 {shlex.quote(self.container_name)} mkdir -p {shlex.quote(parent)}")
        await self.docker_cp(vm_path, f"{self.container_name}:{destination_path}")

    @staticmethod
    def _staging_path(kind: str, destination_path: str) -> str:
        """Where one write is staged on the VM host: unique per call, since sandboxes can share a
        host's /tmp, as local ones do."""
        return f"/tmp/_{kind}_{uuid.uuid4().hex[:12]}_{destination_path.replace('/', '_').lstrip('_')}"

    async def write_file_from_object(self, object_url: str, destination_path: str) -> None:
        vm_path = self._staging_path("obj", destination_path)
        try:
            await self.load_object_file(object_url, vm_path)
            await self._copy_into_container(vm_path, destination_path)
        finally:
            await self._remove_vm_temp_file(vm_path)

    async def write_file_from_url(self, url: str, destination_path: str) -> None:
        vm_path = self._staging_path("url", destination_path)
        try:
            await self.exec_script(f"curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(url)} -o {shlex.quote(vm_path)}")
            await self._copy_into_container(vm_path, destination_path)
        finally:
            await self._remove_vm_temp_file(vm_path)

    # One exec_script is a single `bash -c <script>` arg, capped by Linux MAX_ARG_STRLEN
    # (128 KiB); 96 KiB leaves room for the printf wrapper.
    _WFT_CHUNK_BYTES = 96 * 1024

    async def _write_bytes_to_vm_path(self, data: bytes, vm_path: str) -> None:
        """Stream bytes from agent-env onto the VM host at vm_path (base64 over exec)."""
        encoded = base64.b64encode(data).decode()
        if len(encoded) <= self._WFT_CHUNK_BYTES:
            await self.exec_script(f"base64 -d <<'ENDB64' > {shlex.quote(vm_path)}\n{encoded}\nENDB64")
            return
        # Too big for one heredoc arg: append the (shell-safe) base64 in bounded chunks.
        vm_b64 = f"{vm_path}.b64"
        await self.exec_script(f": > {shlex.quote(vm_b64)}")
        for i in range(0, len(encoded), self._WFT_CHUNK_BYTES):
            await self.exec_script(f"printf '%s' {shlex.quote(encoded[i:i + self._WFT_CHUNK_BYTES])} >> {shlex.quote(vm_b64)}")
        await self.exec_script(f"base64 -d {shlex.quote(vm_b64)} > {shlex.quote(vm_path)} && rm -f {shlex.quote(vm_b64)}")

    async def write_host_file(self, data: bytes, vm_path: str) -> None:
        """Write bytes to vm_path on the VM host itself, not into the agent container."""
        parent = os.path.dirname(vm_path)
        if parent:
            await self.exec_script(f"mkdir -p {shlex.quote(parent)}")
        await self._write_bytes_to_vm_path(data, vm_path)

    async def write_file_from_text(self, content: str, destination_path: str) -> None:
        vm_path = self._staging_path("wft", destination_path)
        try:
            await self._write_bytes_to_vm_path(content.encode(), vm_path)
            await self._copy_into_container(vm_path, destination_path)
        finally:
            await self._remove_vm_temp_file(vm_path, f"{vm_path}.b64")


def port_bindings(host_ips: Iterable[str], host_port: int, container_port: int) -> list[str]:
    """Docker publish specs for one port: one per host IP, or a bare one (every interface) when there are none."""
    return [f"{ip}:{host_port}:{container_port}" for ip in host_ips] or [f"{host_port}:{container_port}"]


async def stage_files_into_container(sandbox: Sandbox, file_artifacts: dict[str, Any], destination: str) -> dict[str, str]:
    """Write each object-store file into the sandbox's container under ``destination``; returns ``{name: path}``.
    A VM-backed sandbox stages through its host into ``container_name``; a container sandbox is written directly."""
    loaded: dict[str, str] = {}
    dirs_to_make: set[str] = {destination}
    for filename in file_artifacts:
        validate_relative_filename(filename)
        dest_path = posixpath.join(destination, filename)
        dirs_to_make.add(posixpath.dirname(dest_path))
        loaded[filename] = dest_path

    for d in sorted(dirs_to_make):
        if isinstance(sandbox, VmSandbox):
            await sandbox.exec_script(f"docker exec -u 0 {shlex.quote(sandbox.container_name)} mkdir -p {shlex.quote(d)}")
        else:
            await sandbox.exec("mkdir", "-p", d)

    # Bounded so a universe with many files doesn't serialize per-file presign + copy latency.
    sem = asyncio.Semaphore(8)

    async def _stage(filename: str, file_artifact: Any) -> None:
        async with sem:
            logger.info(f"  {file_artifact.object_url} -> {loaded[filename]}")
            await sandbox.write_file_from_object(file_artifact.object_url, loaded[filename])

    await asyncio.gather(*(_stage(fn, fa) for fn, fa in file_artifacts.items()))
    return loaded


async def read_vm_file(sandbox: VmSandbox, vm_path: str, local_path: str) -> None:
    """Copy ``vm_path`` off the VM host into ``local_path``, a range at a time, base64 over exec, so memory holds only
    the ranges in flight. A range shorter than it should be fails the copy rather than leave a truncated file."""
    quoted = shlex.quote(vm_path)
    reported = (await sandbox.exec_script(f"wc -c < {quoted}")).strip()
    if not reported.isdigit():
        raise RuntimeError(f"the VM reported {reported!r} as the size of {vm_path}; refusing to copy it")
    size = int(reported)
    gate = asyncio.Semaphore(_READS_IN_FLIGHT)
    with open(local_path, "wb") as out:
        out.truncate(size)

        async def copy(offset: int) -> None:
            length = min(_READ_RANGE_BYTES, size - offset)
            async with gate:
                encoded = await sandbox.exec_script(f"tail -c +{offset + 1} {quoted} | head -c {length} | base64")
            data = base64.b64decode(encoded)
            if len(data) != length:
                raise RuntimeError(f"read {len(data)} of {length} bytes at offset {offset} of {vm_path}; "
                                   "refusing a truncated copy")
            out.seek(offset)
            out.write(data)

        copies = [asyncio.ensure_future(copy(offset)) for offset in range(0, size, _READ_RANGE_BYTES)]
        try:
            await asyncio.gather(*copies)
        finally:
            for pending in copies:
                pending.cancel()
            await asyncio.gather(*copies, return_exceptions=True)


async def upload_vm_file(sandbox: VmSandbox, vm_path: str, store: ObjectStore, object_url: str) -> None:
    """Put ``vm_path`` from the VM host into ``store`` at ``object_url``. The VM uploads it to a presigned URL when the
    store signs one; otherwise agent-env copies it off the VM and puts it, since a sandbox can't reach a store on this
    machine."""
    put_url = await asyncio.to_thread(store.signed_put_url, object_url)
    if put_url is not None:
        await sandbox.exec_script(f'curl -fsSL -X PUT --upload-file {shlex.quote(vm_path)} "{put_url}"')
        return
    with tempfile.TemporaryDirectory(prefix="agentenv-vm-file-") as directory:
        local_path = os.path.join(directory, posixpath.basename(vm_path))
        await read_vm_file(sandbox, vm_path, local_path)
        await asyncio.to_thread(store.put_file_at, object_url, local_path)
