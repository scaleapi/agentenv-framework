"""Sandbox interfaces for agent-env compute backends."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import logging
import os
import posixpath
import shlex
import stat
import tempfile
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import IO, TYPE_CHECKING, Any, AsyncIterator, Callable, Iterable, Optional

from agent_env.config import get_config
from agent_env.utils.deprecation import warn_deprecated
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
# An object pushed onto a VM host over exec arguments goes a chunk per exec, the chunk's base64 in the script; over stdin
# it goes as segments of at least _MIN_SEGMENT_BYTES, each in pieces. An object of one chunk or one segment goes in a
# single exec. Chunks and segments start at multiples of _PUSH_BLOCK, a multiple of the 4096-byte block dd seeks in and
# of 3, so a piece's base64 is unpadded.
_PUSH_BLOCK = 3 * 4096
_STDIN_PIECE_BYTES = 16 * _PUSH_BLOCK
_MIN_SEGMENT_BYTES = 1024 * 1024 // _PUSH_BLOCK * _PUSH_BLOCK


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
    # ``name:address`` entries the containers started on this sandbox add to their hosts file.
    extra_hosts: tuple[str, ...] = ()

    _VM_READY_TIMEOUT = 1200      # wait_for_vm wall-clock budget (s)
    _VM_READY_POLL_INTERVAL = 30  # sparse polling (s)

    def host_port(self, port: int) -> int:
        """The host-side port a published container port is reachable on.

        Identity for any backend that gives a deployment its own network namespace.
        Backends that share a host override this to avoid collisions between concurrent
        deployments; container ports are unaffected either way.
        """
        return port

    def scoped_name(self, name: str) -> str:
        """The name a container, network, image or temp path a step calls ``name`` takes on this sandbox's
        host: ``name`` itself on a host of its own. A backend whose sandboxes share a host overrides this."""
        return name

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
        warn_deprecated("Sandbox.write_file_from_s3", "write_file_from_object", kind="method")
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
        warn_deprecated("VmSandbox.load_s3_file", "load_object_file", kind="method")
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
        """Place an object the store cannot sign a URL for: pushed over exec, a chunk per exec."""
        await push_object_over_exec(self, object_store, object_url, vm_path)

    async def _exec_with_stdin(self, script: str, stdin: AsyncIterator[bytes]) -> tuple[int, str, str]:
        """Run ``script`` with bash on the VM host, its standard input the bytes ``stdin`` yields, and return (exit_code,
        stdout, stderr). A sandbox whose exec takes stdin implements it, and pushes objects with
        ``push_object_over_stdin``."""
        raise NotImplementedError(f"{self.__class__.__name__} does not take stdin on exec")

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
    # How many execs carry one object pushed onto the host at once, chunks or segments. A provider whose execs scale
    # differently sets its own.
    _PUSHES_IN_FLIGHT = 8

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


async def push_object_over_exec(sandbox: VmSandbox, store: ObjectStore, object_url: str, vm_path: str) -> None:
    """Write the object at ``object_url`` to ``vm_path`` on the VM host a chunk per exec, the chunk's base64 in the
    script and written at its offset, ``sandbox._PUSHES_IN_FLIGHT`` execs at once; an object of one chunk goes in a
    single exec. The file is checked against the object's sha256."""
    chunk = _exec_chunk_bytes(sandbox)
    quoted = shlex.quote(vm_path)
    digest = hashlib.sha256()
    with contextlib.closing(await asyncio.to_thread(store.open, object_url)) as source:
        data = await asyncio.to_thread(_read_exactly, source, chunk)
        digest.update(data)
        if len(data) < chunk:
            written = await _write_in_one_exec(sandbox, quoted, data)
        else:
            await sandbox.exec_script(f": > {quoted}")
            await _push_chunks(sandbox, quoted, source, chunk, data, digest)
            written = (await sandbox.exec_script(f"sha256sum {quoted}")).split()[:1]
    _check_written(vm_path, object_url, written, digest.hexdigest())


def _exec_chunk_bytes(sandbox: VmSandbox) -> int:
    """How much of an object one exec's script carries, as base64, within the sandbox's command limit."""
    return max(_PUSH_BLOCK, sandbox._WFT_CHUNK_BYTES // 4 * 3 // _PUSH_BLOCK * _PUSH_BLOCK)


async def _write_in_one_exec(sandbox: VmSandbox, quoted: str, data: bytes) -> list[str]:
    """Write ``data`` as the whole VM file in one exec whose script carries it, and return the sha256 it reports."""
    encoded = shlex.quote(base64.b64encode(data).decode())
    reported = await sandbox.exec_script(f"printf %s {encoded} | base64 -d > {quoted} && sha256sum {quoted}",
                                         max_retries=2)
    return reported.split()[:1]


def _check_written(vm_path: str, object_url: str, written: list[str], expected: str) -> None:
    if written != [expected]:
        raise RuntimeError(f"{vm_path} on the VM doesn't match {object_url}: its sha256 is {written}, the object's "
                           f"{expected}")


async def _push_chunks(
    sandbox: VmSandbox, quoted: str, source: IO[bytes], chunk: int, data: bytes, digest: Any,
) -> None:
    """Write ``data``, the object's first chunk, and every chunk ``source`` has after it, each at its offset."""
    gate = asyncio.Semaphore(sandbox._PUSHES_IN_FLIGHT)
    failed = False

    async def write(offset: int, data: bytes) -> None:
        nonlocal failed
        try:
            encoded = shlex.quote(base64.b64encode(data).decode())
            await sandbox.exec_script(
                f"printf %s {encoded} | base64 -d | dd of={quoted} bs=4096 seek={offset // 4096} conv=notrunc",
                max_retries=2,
            )
        except BaseException:
            failed = True
            raise
        finally:
            gate.release()

    writes: list[asyncio.Future] = []
    try:
        offset = 0
        while not failed and data:
            await gate.acquire()
            if failed:
                gate.release()
                break
            writes.append(asyncio.ensure_future(write(offset, data)))
            offset += len(data)
            data = await asyncio.to_thread(_read_exactly, source, chunk)
            digest.update(data)
        await asyncio.gather(*writes)
    finally:
        for pending in writes:
            pending.cancel()
        await asyncio.gather(*writes, return_exceptions=True)


async def push_object_over_stdin(sandbox: VmSandbox, store: ObjectStore, object_url: str, vm_path: str) -> None:
    """Write the object at ``object_url`` to ``vm_path`` on the VM host over exec stdin. When the store opens it as a
    file, it goes as up to ``sandbox._PUSHES_IN_FLIGHT`` segments at once, all read from that one open file, so one
    version of it; one that fits an exec's script goes in it, since a stdin exec takes more round trips. An object of
    one segment, or any other reader, goes in a single stdin exec. Each segment is written at its offset and checked
    against its sha256 by the exec that writes it."""
    quoted = shlex.quote(vm_path)
    with contextlib.closing(await asyncio.to_thread(store.open, object_url)) as source:
        descriptor = _file_descriptor(source)
        if descriptor is None:
            await _push_segment(sandbox, quoted, lambda size: _read_exactly(source, size), 0, None, whole_file=True)
            return
        size = os.fstat(descriptor).st_size
        if size < _exec_chunk_bytes(sandbox):
            data = await asyncio.to_thread(_reader_at(descriptor, 0), size)
            if len(data) != size:
                raise RuntimeError(f"{object_url} ended {size - len(data)} bytes short; it changed during the push")
            written = await _write_in_one_exec(sandbox, quoted, data)
            _check_written(vm_path, object_url, written, hashlib.sha256(data).hexdigest())
            return
        segment = max(_MIN_SEGMENT_BYTES, -(-size // (sandbox._PUSHES_IN_FLIGHT * _PUSH_BLOCK)) * _PUSH_BLOCK)
        if size <= segment:
            await _push_segment(sandbox, quoted, _reader_at(descriptor, 0), 0, size, whole_file=True)
            return
        await sandbox.exec_script(f": > {quoted}")
        pushes = [
            asyncio.ensure_future(_push_segment(
                sandbox, quoted, _reader_at(descriptor, offset), offset, min(segment, size - offset), whole_file=False))
            for offset in range(0, size, segment)
        ]
        try:
            await asyncio.gather(*pushes)
        finally:
            for pending in pushes:
                pending.cancel()
            await asyncio.gather(*pushes, return_exceptions=True)


async def _push_segment(
    sandbox: VmSandbox, quoted: str, read: Callable[[int], bytes], offset: int, length: int | None, *, whole_file: bool,
) -> None:
    """Stream ``length`` bytes from ``read`` (all it gives when None) into the VM file at ``offset``, and check them.
    A ``whole_file`` segment creates the file, which holds nothing else."""
    digest = hashlib.sha256()

    async def pieces() -> AsyncIterator[bytes]:
        left = length
        while left is None or left > 0:
            data = await asyncio.to_thread(read, _STDIN_PIECE_BYTES if left is None else min(_STDIN_PIECE_BYTES, left))
            if not data:
                if left is not None:
                    raise RuntimeError(f"The object ended {left} bytes short of the segment at offset {offset} of "
                                       f"{quoted}; it changed during the push")
                return
            digest.update(data)
            if left is not None:
                left -= len(data)
            yield base64.b64encode(data)

    if whole_file:
        script = f"base64 -d > {quoted} && sha256sum {quoted}"
    else:
        script = (f"base64 -d | dd of={quoted} bs=4096 seek={offset // 4096} conv=notrunc && "
                  f"tail -c +{offset + 1} {quoted} | head -c {length} | sha256sum")
    code, stdout, stderr = await sandbox._exec_with_stdin(script, pieces())
    if code != 0:
        raise RuntimeError(f"Writing {quoted} at offset {offset} on the VM failed (exit {code}): {stderr[-1500:]}")
    if stdout.split()[:1] != [digest.hexdigest()]:
        raise RuntimeError(f"{quoted} on the VM doesn't match at offset {offset}: its sha256 is {stdout.split()[:1]}, "
                           f"the object's {digest.hexdigest()}")


def _file_descriptor(reader: IO[bytes]) -> int | None:
    """The descriptor of the regular file ``reader`` reads as it is, or None when it reads anything else. A reader that
    transforms what it reads, such as a decompressing one, can still name its file's descriptor, so only a plain file
    reader counts."""
    if not isinstance(reader, (io.BufferedReader, io.FileIO)):
        return None
    try:
        descriptor = reader.fileno()
    except OSError:
        return None
    return descriptor if stat.S_ISREG(os.fstat(descriptor).st_mode) else None


def _reader_at(descriptor: int, offset: int) -> Callable[[int], bytes]:
    """Reads the file ``descriptor`` names from ``offset`` on, by position, so readers of one file don't share a
    file offset."""
    position = offset

    def read(size: int) -> bytes:
        nonlocal position
        parts, got = [], 0
        while got < size and (part := os.pread(descriptor, size - got, position + got)):
            parts.append(part)
            got += len(part)
        position += got
        return b"".join(parts)

    return read


def _read_exactly(reader: IO[bytes], size: int) -> bytes:
    """``size`` bytes of ``reader``, fewer only at its end: a stream may return less than asked for."""
    parts, got = [], 0
    while got < size and (part := reader.read(size - got)):
        parts.append(part)
        got += len(part)
    return b"".join(parts)
