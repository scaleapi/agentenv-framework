"""Local sandbox for running agent-env environments on the local machine.

A drop-in replacement for a remote VM sandbox that executes all commands locally
via subprocess instead of on a remote VM.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import functools
import glob
import json
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
from ipaddress import ip_address
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from agent_env import config
from agent_env.attribution import Attribution
from agent_env.providers.sandbox_providers.sandbox import SANDBOX_LABEL, NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_CONTAINER,
    SANDBOX_MODE_VM,
    SandboxProvider,
    refuse_unenforceable_policy,
)
from agent_env.store.object_store.local.tls import local_ca
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from agent_env.store.routing import LocalRunObjectStore

if TYPE_CHECKING:
    from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)

# The name a container on this machine reaches it by. Only Linux lacks it natively; on Rancher Desktop an explicit
# mapping would point it at the VM.
_HOST_ALIAS = "host.docker.internal"
_EXTRA_HOSTS = (f"{_HOST_ALIAS}:host-gateway",) if platform.system() == "Linux" else ()

_APP_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_./~})$-])/app(?=(?:/|:|[\s'\";)&|]|$))")
_IN_CONTAINER_SCRIPT = re.compile(r"\s*(?:sudo\s+)?docker\s+exec\b")


def _runs_in_container(cmd: list[str]) -> bool:
    """A ``docker exec``, as arguments or as a ``bash -c`` script: its /app is the container's."""
    return cmd[:2] == ["docker", "exec"] or (
        cmd[:2] == ["bash", "-c"] and len(cmd) > 2 and bool(_IN_CONTAINER_SCRIPT.match(cmd[2]))
    )


_REAP_SECONDS = 5
# How long a finished command's output is still read, for a job it left in the background that holds it open; and
# how much of each stream is kept, the end of it: above the largest output core reads whole, run_code's 50 MiB result.
_OUTPUT_GRACE_SECONDS = 1
_OUTPUT_TAIL_BYTES = 64 * 1024 * 1024

# Every command a local sandbox runs carries this, set to the sandbox's id, and so does whatever the command starts, so
# teardown can find a process left running on this machine, such as the agent a host-mode install started.
_SANDBOX_ENV = "AGENTENV_SANDBOX"

# Where a container finds the local transfer CA's trust files, and the variables that point TLS clients at them:
# SSL_CERT_FILE replaces a client's roots, so it gets the public roots plus the CA; NODE_EXTRA_CA_CERTS adds.
LOCAL_TRUST_DIR = "/etc/agentenv"
LOCAL_TRUST_ENV = {
    "SSL_CERT_FILE": f"{LOCAL_TRUST_DIR}/ca-bundle.pem",
    "REQUESTS_CA_BUNDLE": f"{LOCAL_TRUST_DIR}/ca-bundle.pem",
    "NODE_EXTRA_CA_CERTS": f"{LOCAL_TRUST_DIR}/ca.pem",
}

# Marker dropped in the work dir when this sandbox runs a container, so a later get_sandbox()
# (post-run teardown reconstructs the sandbox from disk) knows it owns that container.
_CONTAINER_MODE_MARKER = ".agent-container-mode"
# The sandbox's container port -> host port map, so a sandbox rebuilt from its id publishes and finds the same ports.
_PORT_MAP_FILE = ".port-map.json"


def _local_sandbox_root_path() -> Path:
    """Base dir for local-sandbox work dirs (pure — no side effects, safe for read-only lookups).

    It MUST live on a path Docker Desktop shares with its VM: the gateway compose bind-mounts files
    from here (e.g. init-schemas.sql), and on macOS the default temp dirs ($TMPDIR under /var/folders,
    and /tmp) are NOT shared — a bind mount from there silently becomes an empty directory (Postgres
    then reads a dir as SQL and the init crashes). The user's home (/Users/…) is shared by default.
    Override with AGENT_ENV_LOCAL_SANDBOX_DIR."""
    base = os.environ.get("AGENT_ENV_LOCAL_SANDBOX_DIR")
    return Path(base) if base else Path.home() / ".agent-env-sandboxes"


def _local_sandbox_root() -> Path:
    """The shared root, created if missing — for placing a *new* work dir. Lookups must not use this
    (its mkdir can raise); scan via _local_sandbox_root_path instead so an uncreatable new root never
    blocks reconstructing an existing sandbox."""
    root = _local_sandbox_root_path()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _free_host_port() -> int:
    """A currently-free host port.

    Binds all interfaces, as docker does when publishing a port: a port free on loopback
    alone can still be taken on another interface.

    Racy: the port is released before docker binds it. The window is small and a loser
    fails loudly at bind time.
    """
    with socket.socket() as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


class LocalSandbox(VmSandbox):
    """Sandbox implementation that runs everything on the local machine.

    Implements the same VmSandbox interface as the remote providers but executes commands
    locally via asyncio.create_subprocess_exec. This gives 1:1 behavior
    parity with prod for docker compose generation, data loading, etc.

    The /app path used by EnvironmentGatewayProvider is redirected to a local temp
    directory since /app is not writable on macOS.
    """

    type = "local"
    extra_hosts = _EXTRA_HOSTS

    def __init__(self, exposed_ports: list[int] | None = None, work_dir: Path | None = None, sandbox_id: str | None = None,
                 port_map: dict[int, int] | None = None):
        self.sandbox_id = sandbox_id or f"local-{uuid4().hex[:8]}"
        # Container port -> host port. Deployments share one host, so the reserved ports
        # can only be published once; the provider supplies a map to spread them. Callers
        # that just want a handle pass `exposed_ports` and get identity.
        self._port_map = dict(port_map) if port_map is not None else {
            port: port for port in (exposed_ports or [])
        }
        # Keyed by container port: callers index this with the well-known constants.
        # 127.0.0.1, not localhost: ports are published on IPv4 loopback, and a client may try ::1 first.
        self.tunnel_urls = {
            container: f"http://127.0.0.1:{host}" for container, host in self._port_map.items()
        }
        self.vnc_url = None
        self.mode = SANDBOX_MODE_VM
        self._work_dir = work_dir or Path(
            tempfile.mkdtemp(prefix=f"agent-env-{self.sandbox_id}-", dir=_local_sandbox_root())
        )

    def host_port(self, port: int) -> int:
        """The allocated host port for a published container port (identity if unmapped)."""
        return self._port_map.get(port, port)

    @property
    def host_ips(self) -> tuple[str, ...]:
        """Loopback, so a local deploy is not reachable from the network (see ``_host_ips``)."""
        return _host_ips()

    @classmethod
    def find_work_dir(cls, sandbox_id: str) -> Path | None:
        """Find the work directory for a previously created local sandbox.

        Scans the current shared root first, then the legacy temp-dir root: sandboxes created before
        the work-dir relocation landed under ``tempfile.gettempdir()``, and must still be locatable
        (e.g. for teardown) after an upgrade."""
        seen: set[Path] = set()
        for root in (_local_sandbox_root_path(), Path(tempfile.gettempdir())):
            if root in seen:
                continue
            seen.add(root)
            matches = sorted(glob.glob(f"{root}/agent-env-{glob.escape(sandbox_id)}-*"))
            if matches:
                return Path(matches[-1])
        return None

    @property
    def container_name(self) -> str:
        """Per-sandbox agent container name. The local backend packs every agent onto one Docker
        host, so a fixed name (the VmSandbox default) would collide across concurrent deploys and
        make teardown ownership-blind. Deriving it from the sandbox id gives each deploy its own
        container and lets teardown remove only the one this sandbox created."""
        return self.scoped_name("agent")

    def scoped_name(self, name: str) -> str:
        """``name`` made this sandbox's own: every local sandbox shares this machine's Docker and /tmp."""
        return f"{name}-{self.sandbox_id}"

    @property
    def owns_container(self) -> bool:
        """Whether create_container started ``container_name`` for this sandbox, not an agent placed on it."""
        return (self._work_dir / _CONTAINER_MODE_MARKER).exists()

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    def _rewrite_app_arg(self, arg: str) -> str:
        if arg == "/app" or arg.startswith(("/app/", "/app:")):
            return str(self._work_dir) + arg[4:]
        return arg

    def _rewrite_app_script(self, script: str) -> str:
        return _APP_PATH_PATTERN.sub(lambda _: str(self._work_dir), script)

    async def terminate(self) -> None:
        """Tear down whatever this sandbox is running.

        A sandbox create_container made (the A2A agent; VM mode once reattached) owns the container
        ``self.container_name`` it started — remove it. A VM-mode sandbox (env/gateway) runs a docker compose
        stack out of its work dir — ``docker compose down`` it — and may carry an agent placed on it,
        which runs as the same ``self.container_name`` container. Each path only touches resources
        this sandbox created: the container name is per-sandbox (LocalSandbox.container_name), so a
        stale marker from an old run can only target that run's own (already-gone) container, never a
        newer one; and the compose-down is scoped to this sandbox's project via its work dir. Without
        this, local runs leak their containers/compose stacks, which squat host ports and block the
        next deploy. Prod backends override terminate() to tear the whole VM down.
        Processes its commands left running on this machine are stopped first (on Linux; see ``_marked_pids``),
        then, once none is left to run from it, what its steps staged in /tmp is removed.
        """
        try:
            if await _stop_marked(self.sandbox_id):
                await asyncio.to_thread(_remove_staged, self.sandbox_id)
            if self.mode == SANDBOX_MODE_VM and not self.owns_container:
                await self._remove_labeled()
        finally:  # a container that wouldn't go must not keep the compose stack up
            await self.exec_script(f"docker rm -f {shlex.quote(self.container_name)} >/dev/null 2>&1 || true")
            if self.mode == SANDBOX_MODE_VM and (self._work_dir / "docker-compose.yml").exists():
                # No `|| true`: `docker compose down` is idempotent (an already-down stack exits 0), so a
                # non-zero exit is a real failure — surface it (exec_script raises, with the compose output)
                # instead of silently leaving the stack running and its host ports held. Callers wrap
                # terminate() in try/except, so a raised teardown failure is caught, not fatal.
                await self.exec_script(
                    f"cd {shlex.quote(str(self._work_dir))} && docker compose down -v --remove-orphans"
                )

    async def _remove_labeled(self) -> None:
        """Remove what steps started on this host for this sandbox (``run_docker_container``'s containers, their
        images and networks), which a real VM would take down with it. Images and networks go best-effort: another
        local sandbox's container can still be using one."""
        label = f"label={SANDBOX_LABEL}={self.sandbox_id}"
        try:
            exit_code, stdout, stderr = await self.exec_with_output("docker", "ps", "-aq", "--filter", label)
        except FileNotFoundError:  # no Docker installed: nothing could have been started
            return
        if exit_code != 0:
            # No Docker to reach (a run that never needed it) or Docker is down; either way nothing can be removed now.
            logger.warning(
                "Could not list %s's containers (%s); anything it started is still there. Remove it with: "
                "docker rm -f $(docker ps -aq --filter %s)", self.sandbox_id, stderr.strip()[-300:], label,
            )
            return
        q = shlex.quote(label)
        if stdout.split():
            await self.exec_script(f"docker rm -f {' '.join(stdout.split())} >/dev/null")
        await self.exec_script(
            f"ids=$(docker images -q --filter {q} | sort -u); [ -z \"$ids\" ] || docker rmi -f $ids >/dev/null 2>&1; "
            f"ids=$(docker network ls -q --filter {q}); [ -z \"$ids\" ] || docker network rm $ids >/dev/null 2>&1; "
            f"exit 0"
        )

    async def exec(self, *command: str) -> Any:
        """Execute a command locally via subprocess.

        Strips 'sudo' and points /app at the local work directory, both as a path argument and
        inside the script of a top-level ``bash -c``, unless the command runs in a container. The
        command carries ``_SANDBOX_ENV``, so teardown can stop whatever it leaves running.
        Returns an object with .stdout, .stderr streams and .wait() method, matching the
        interface expected by exec_with_output().
        """
        cmd = [c for c in command if c != "sudo"]
        if not _runs_in_container(cmd):
            is_script = cmd[:2] == ["bash", "-c"]
            cmd = [
                self._rewrite_app_script(c) if is_script and i == 2 else self._rewrite_app_arg(c)
                for i, c in enumerate(cmd)
            ]
        logger.debug(f"LocalSandbox exec: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, _SANDBOX_ENV: self.sandbox_id},
        )
        return process

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        """Like the base, but as a remote exec session behaves. Cancelling it kills the command and every process it
        started; this host is the sandbox's VM, so nothing else would. The call ends with the command, even if a job
        it left in the background still holds its output, which is then closed. Each stream keeps its last
        ``_OUTPUT_TAIL_BYTES``, and bytes that aren't UTF-8 become replacement characters."""
        spawning = asyncio.ensure_future(self.exec(*args))
        try:
            process = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await _stop(await spawning)
            raise
        stdout, stderr = _Tail(), _Tail()
        reading = [asyncio.ensure_future(tail.drain(stream))
                   for tail, stream in ((stdout, process.stdout), (stderr, process.stderr))]
        try:
            exit_code = await _exit_code(process)
            await asyncio.wait(reading, timeout=_OUTPUT_GRACE_SECONDS)
        except asyncio.CancelledError:
            await _stop(process)
            raise
        finally:
            for reader in reading:
                reader.cancel()
            # Our ends of its pipes, which a background job may hold. asyncio has no public way to close them.
            process._transport.close()
        output = stdout.text(), stderr.text()
        for name, tail in (("stdout", stdout), ("stderr", stderr)):
            if tail.dropped:  # not the command itself: it can carry secrets
                logger.warning("Kept the last %d bytes of a command's %s on sandbox %s; %d before them were dropped",
                               _OUTPUT_TAIL_BYTES, name, self.sandbox_id, tail.dropped)
        return exit_code, *output

    async def _write_unsigned_object(self, object_store: ObjectStore, object_url: str, vm_path: str) -> None:
        """This host is the VM, so the store writes the object in place, with no exec transport."""
        await asyncio.to_thread(object_store.download_to_file, object_url, self._rewrite_app_arg(vm_path))

    async def setup_vm_for_gateway(self, exposed_ports=None) -> None:
        """No-op — Docker is already installed locally."""
        logger.info("Local sandbox: skipping VM setup (Docker assumed installed)")

    async def wait_for_vm(self) -> None:
        """No-op — local machine is always ready."""
        pass


class _Tail:
    """The end of a stream as it is read: its last ``_OUTPUT_TAIL_BYTES``, and how many bytes before them were
    dropped. Chunks, because cutting the front off one growing buffer keeps the memory it grew to."""

    def __init__(self) -> None:
        self.chunks: collections.deque[bytes] = collections.deque()
        self.size = self.dropped = 0

    async def drain(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(1 << 16):
            self.chunks.append(chunk)
            self.size += len(chunk)
            while self.size - len(self.chunks[0]) >= _OUTPUT_TAIL_BYTES:
                first = self.chunks.popleft()
                self.size -= len(first)
                self.dropped += len(first)

    def text(self) -> str:
        data = b"".join(self.chunks)
        if (excess := len(data) - _OUTPUT_TAIL_BYTES) > 0:
            data = data[excess:]
            self.dropped += excess
        return data.decode(errors="replace")


async def _exit_code(process: asyncio.subprocess.Process) -> int:
    """``process``'s exit code, as soon as it exits. ``Process.wait()`` also waits for its pipes to close, which a job
    it left in the background can hold open, so the exit is checked for meanwhile."""
    waiting = asyncio.ensure_future(process.wait())
    try:
        while process.returncode is None:
            await asyncio.wait([waiting], timeout=0.05)
        return process.returncode
    finally:
        waiting.cancel()


async def _stop(process: asyncio.subprocess.Process) -> None:
    """SIGKILL ``process`` and every process it started, wait a few seconds at most for it to exit, then close our
    ends of its pipes, which a job that escaped the kill may still hold."""
    _kill_tree(process.pid)
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(_exit_code(process), _REAP_SECONDS)
    process._transport.close()


def _kill_tree(root: int) -> None:
    """SIGKILL ``root`` and its descendants, as one ``ps`` lists them."""
    try:
        listing = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid="], capture_output=True, text=True,
                                 timeout=_REAP_SECONDS).stdout
    except (OSError, subprocess.SubprocessError):
        listing = ""
    children: dict[int, list[int]] = {}
    for line in listing.splitlines():
        pid, _, parent = line.strip().partition(" ")
        if pid.isdigit() and parent.strip().isdigit():
            children.setdefault(int(parent), []).append(int(pid))
    tree, pending = [], [root]
    while pending:
        tree.append(pid := pending.pop())
        pending.extend(children.get(pid, ()))
    for pid in tree:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def _marked_pids(sandbox_id: str) -> list[int]:
    """The processes carrying ``sandbox_id``'s ``_SANDBOX_ENV``. Read from /proc, so Linux only: macOS doesn't let
    one process read another's environment, and finds none."""
    marker = f"{_SANDBOX_ENV}={sandbox_id}".encode()
    pids = []
    for environ in glob.glob("/proc/[0-9]*/environ"):
        with contextlib.suppress(OSError):  # gone, or another user's
            if marker in Path(environ).read_bytes().split(b"\0"):
                pids.append(int(environ.split("/")[2]))
    return pids


async def _stop_marked(sandbox_id: str) -> bool:
    """Stop the processes ``sandbox_id``'s commands left running, which a real VM would take down with it: SIGTERM,
    then SIGKILL for any still there a few seconds later. False if some are still running."""
    if not (pids := await asyncio.to_thread(_marked_pids, sandbox_id)):
        return True
    logger.info("Stopping %d process(es) sandbox %s left running on this machine", len(pids), sandbox_id)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)
        for _ in range(_REAP_SECONDS * 10):
            await asyncio.sleep(0.1)
            if not (pids := await asyncio.to_thread(_marked_pids, sandbox_id)):
                return True
    logger.warning("Processes %s that sandbox %s started are still running; keeping what it staged in /tmp",
                   pids, sandbox_id)
    return False


def _remove_staged(sandbox_id: str) -> None:
    """Remove what steps staged in /tmp for ``sandbox_id``, such as a host install's work dir, which a host agent may
    run from until it is stopped. Steps name these with ``scoped_name``, so they end in the sandbox's id."""
    for path in glob.glob(f"/tmp/*-{glob.escape(sandbox_id)}"):
        shutil.rmtree(path, ignore_errors=True)


def local_grant_trust() -> Path | None:
    """The trust files a container on this host needs to use the configured object store's grants: the local
    transfer CA's, when the store is a local one that hands out grants; None otherwise."""
    store = config.get_config().get_object_store()
    if isinstance(store, LocalRunObjectStore):
        store = store.local
    if isinstance(store, LocalFilesystemObjectStore) and store.supports_transfer_grants:
        return local_ca().trust_dir
    return None


async def start_trusting(sandbox: VmSandbox, container: str, trust_dir: Path) -> None:
    """Copy ``trust_dir`` into the created ``container`` where ``LOCAL_TRUST_ENV`` points, then start it. A container
    that cannot be given the files or started is removed, so its name is free for the next attempt."""
    try:
        await asyncio.to_thread(_copy_into_container, trust_dir, container, LOCAL_TRUST_DIR)
        await sandbox.exec_script(f"docker start {shlex.quote(container)} > /dev/null")
    except Exception:
        await sandbox.exec_script(f"docker rm -f {shlex.quote(container)} >/dev/null 2>&1 || true")
        raise


def _copy_into_container(source: Path, container: str, destination: str) -> None:
    """Copy what the host directory ``source`` holds to ``destination`` in ``container``. Run directly, not
    through a sandbox's shell, which would rewrite a host path under /app."""
    copied = subprocess.run(["docker", "cp", f"{source}/.", f"{container}:{destination}"], capture_output=True, text=True)
    if copied.returncode != 0:
        raise RuntimeError(f"Could not copy {source} into container {container}: {copied.stderr.strip()}")


def remove_local_work_dir(sandbox_id: str) -> Path | None:
    """Delete a local sandbox's work folder, and return it, when it lies directly under the sandbox root and
    isn't a symlink; anything else, a legacy folder under the temp dir included, is left alone. Raises OSError
    when the folder can't be removed."""
    work_dir = LocalSandbox.find_work_dir(sandbox_id)
    root = _local_sandbox_root_path().resolve()
    if work_dir is None or work_dir.is_symlink() or work_dir.parent.resolve() != root:
        return None
    shutil.rmtree(work_dir)
    return work_dir


class LocalSandboxProvider(SandboxProvider):
    """SandboxProvider that runs VM-style gateway deployments on local Docker."""

    EXTRA_CONTAINER_RUN_ARGS = " ".join(f"--add-host {entry}" for entry in _EXTRA_HOSTS)

    async def create_vm(
        self,
        *,
        image: Optional[str] = None,
        boot_mode: Optional[str] = None,
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        exposed_ports: Optional[list[int]] = None,
        setup_for_gateway: bool = True,
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> LocalSandbox:
        refuse_unenforceable_policy(self, network_policy)
        # Allocated here rather than in the sandbox constructor: probing for a free port
        # is I/O, and callers that only need a handle must not pay for it.
        sandbox = LocalSandbox(
            port_map={port: _free_host_port() for port in (exposed_ports or [])},
        )
        (sandbox.work_dir / _PORT_MAP_FILE).write_text(json.dumps(sandbox._port_map))
        sandbox.network_policy = self.effective_network_policy(network_policy)
        return sandbox

    async def create_container(
        self,
        *,
        image_name: str,
        port: int,
        env: dict[str, str],
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> LocalSandbox:
        sandbox = await super().create_container(
            image_name=image_name,
            port=port,
            env=env,
            cpu=cpu,
            memory=memory,
            disk_size_gb=disk_size_gb,
            timeout=timeout,
            attribution=attribution,
            network_policy=network_policy,
        )
        # The marker is how a later get_sandbox() (post-run teardown rebuilds the sandbox from disk)
        # learns this is a container and removes it. If we can't persist it, that reconstructed
        # teardown would take the VM path and leak the running container — so we can't just swallow
        # the error. Tear the container down now (the live handle is still container-mode) and fail
        # loudly, rather than return a container nothing can reliably reclaim.
        if sandbox.mode == SANDBOX_MODE_CONTAINER:
            try:
                (sandbox.work_dir / _CONTAINER_MODE_MARKER).write_text(sandbox.container_name)
            except OSError as e:
                # Can't persist the marker → a reconstructed teardown couldn't find/remove this
                # container. Remove it now, but do NOT go through terminate() (its `docker rm … || true`
                # would mask a failed removal): run the removal so a genuinely stuck container is
                # escalated (ERROR) rather than silently leaked with its host port held.
                logger.warning(
                    "Could not persist container-mode marker for %s: %s; removing the container",
                    sandbox.sandbox_id, e,
                )
                try:
                    await sandbox.exec_script(f"docker rm -f {shlex.quote(sandbox.container_name)}")
                except Exception:
                    logger.error(
                        "Container %s could not be removed after marker-persist failure; it may still "
                        "be running and holding its host port — remove it manually.",
                        sandbox.container_name, exc_info=True,
                    )
                raise
        return sandbox

    async def _start_container(self, sandbox: VmSandbox, *, image_name: str, port: int, env: dict[str, str]) -> None:
        """Where the configured store hands out local grants, create the container, copy the local transfer CA's
        trust files in, then start it, so its TLS clients trust the grant server. Variables the caller sets win."""
        trust_dir = await asyncio.to_thread(local_grant_trust)
        if trust_dir is None:
            await super()._start_container(sandbox, image_name=image_name, port=port, env=env)
            return
        args = self._container_args(sandbox, image_name=image_name, port=port, env={**LOCAL_TRUST_ENV, **env})
        await sandbox.exec_script(f"docker create {args} > /dev/null")
        await start_trusting(sandbox, sandbox.container_name, trust_dir)

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
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> LocalSandbox:
        return await self.create_container(
            image_name=image_name, port=port, env=env, cpu=cpu, memory=memory, disk_size_gb=disk_size_gb,
            timeout=timeout, attribution=attribution, network_policy=network_policy,
        )

    async def get_sandbox(self, sandbox_id: str) -> LocalSandbox:
        work_dir = LocalSandbox.find_work_dir(sandbox_id)
        if work_dir is None:
            raise RuntimeError(f"Local sandbox work directory not found for sandbox_id={sandbox_id!r}")
        port_map = work_dir / _PORT_MAP_FILE
        ports = {int(c): h for c, h in json.loads(port_map.read_text()).items()} if port_map.exists() else None
        # VM mode even for a container it owns: exec runs on this host, so steps must `docker exec` into it.
        return LocalSandbox(sandbox_id=sandbox_id, work_dir=work_dir, port_map=ports)

    @classmethod
    def shares_network_with(cls, sandbox_type: Optional[str]) -> bool:
        """Local sandboxes never share a Docker network. The agent runs via ``docker run`` on the
        default bridge; the env/gateway runs in its own compose network. They reach each other only
        through published host ports — so cross-sandbox URLs must be externalized (below), never kept
        as-is. (The base default assumes same-provider ⇒ shared network, which is wrong here.)"""
        return False

    @classmethod
    def get_external_url(cls, url: str) -> str:
        """``url`` as a container on this machine reaches it: a loopback host (a published host port, or a
        service on this machine) becomes ``host.docker.internal``, and anything else is left alone. Docker
        Desktop and Rancher Desktop resolve that name themselves; on Linux ``extra_hosts`` maps it to the
        bridge gateway, so a service there must listen on more than loopback."""
        parts = urlsplit(url)
        if not _is_loopback(parts.hostname):
            return url
        userinfo, at, _ = parts.netloc.rpartition("@")
        port = f":{parts.port}" if parts.port is not None else ""
        return urlunsplit(parts._replace(netloc=f"{userinfo}{at}{_HOST_ALIAS}{port}"))


def host_url_for(url: str, sandbox_type: Optional[str]) -> str:
    """``url``, which this machine reaches, as a container on ``sandbox_type`` reaches it: only a local
    sandbox's containers run on this machine, and they reach its loopback by another name."""
    return LocalSandboxProvider.get_external_url(url) if sandbox_type == LocalSandbox.type else url


def _is_loopback(host: Optional[str]) -> bool:
    """Whether ``host`` names this machine from itself: ``localhost``, a loopback address, or the unspecified
    address a server listening everywhere prints."""
    if host == "localhost":
        return True
    try:
        address = ip_address(host or "")
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


@functools.cache
def _host_ips() -> tuple[str, ...]:
    """Loopback, plus on Linux the bridge gateway ``host-gateway`` resolves to, where containers reach the host."""
    # Docker Desktop publishes through a host-side proxy, which can't bind the bridge address inside its VM.
    if platform.system() != "Linux" or _docker("info", "--format", "{{.OperatingSystem}}") == "Docker Desktop":
        return ("127.0.0.1",)
    # Listing, unlike inspecting, answers a missing network with nothing rather than an error.
    if not _docker("network", "ls", "--quiet", "--filter", "name=^bridge$"):
        return ("127.0.0.1",)
    return ("127.0.0.1", _docker("network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}"))


def _docker(*args: str) -> str:
    """A docker CLI query's output. A failure raises, so the cache above never keeps it."""
    run = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=_DOCKER_QUERY_SECONDS)
    if run.returncode:
        raise RuntimeError(f"docker {' '.join(args)} failed: {run.stderr.strip()}")
    return run.stdout.strip()


_DOCKER_QUERY_SECONDS = 30
