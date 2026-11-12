"""Local sandbox for running agent-env environments on the local machine.

A drop-in replacement for a remote VM sandbox that executes all commands locally
via subprocess instead of on a remote VM.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import errno
import functools
import glob
import json
import logging
import os
import platform
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
from ipaddress import ip_address
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

try:
    import fcntl
except ImportError:  # Windows, which agent-env does not target.
    fcntl = None

from agent_env import config
from agent_env.attribution import Attribution
from agent_env.config.paths import state_root
from agent_env.providers.sandbox_providers.host_images import build_here, built_here, host_platform, present, pull_here
from agent_env.providers.sandbox_providers.disk_launcher import RESET, UNRESETTABLE, forward, reset_input, unresettable
from agent_env.providers.sandbox_providers.local_docker.cli import DockerFailed, docker
from agent_env.providers.sandbox_providers.local_docker.disk import kvm_platforms, launcher_args, machine_for
from agent_env.providers.sandbox_providers.local_docker.relay import relay_image
from agent_env.providers.sandbox_providers.sandbox import (
    SANDBOX_LABEL,
    NetworkPolicy,
    NetworkPolicyUnsupportedError,
    Sandbox,
    VmSandbox,
    port_bindings,
)
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_CONTAINER,
    SANDBOX_MODE_VM,
    Accepts,
    SandboxProvider,
    on_this_machine_provider,
    refuse_unenforceable_policy,
)
from agent_env.store.object_store.local.tls import local_ca
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from agent_env.store.local_state import ensure_state_dir
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
# How much of a failed Docker command's stderr a teardown error carries.
_ERROR_TAIL_CHARS = 300

# Every command a local sandbox runs carries this, set to the sandbox's id, and so does whatever the command starts, so
# teardown can find a process left running on this machine, such as the agent a host-mode install started.
_SANDBOX_ENV = "AGENTENV_SANDBOX"

# Where a container finds the local transfer CA's trust files, and the variables that point TLS clients at them:
# SSL_CERT_FILE replaces a client's roots, so it gets the public roots plus the CA; NODE_EXTRA_CA_CERTS adds.
LOCAL_TRUST_DIR = "/etc/agentenv"


def trust_env(directory: str) -> dict[str, str]:
    """The variables that point TLS clients at the local transfer CA's trust files in ``directory``."""
    return {
        "SSL_CERT_FILE": f"{directory}/ca-bundle.pem",
        "REQUESTS_CA_BUNDLE": f"{directory}/ca-bundle.pem",
        "NODE_EXTRA_CA_CERTS": f"{directory}/ca.pem",
    }


LOCAL_TRUST_ENV = trust_env(LOCAL_TRUST_DIR)

# Marker dropped in the work dir when this sandbox runs a container, so a later get_sandbox()
# (post-run teardown reconstructs the sandbox from disk) knows it owns that container.
_CONTAINER_MODE_MARKER = ".agent-container-mode"
# The sandbox's container port -> host port map, so a sandbox rebuilt from its id publishes and finds the same ports.
_PORT_MAP_FILE = ".port-map.json"
# The internal network a container with no network runs on, so a sandbox rebuilt from its id removes it and its relay.
_CLOSED_NETWORK_FILE = ".closed-network"
# The size of the VM a disk booted on, so a reset boots it at the same size.
_VM_SIZE_FILE = ".vm-size.json"
# The host ports sandboxes publish on: below the default ranges outgoing connections take their local ports from (32768
# up on Linux, the VM Docker Desktop or Rancher Desktop runs Docker in included; 49152 up on macOS and Windows) and
# below Kubernetes' node ports (30000-32767).
_HOST_PORTS = range(20000, 30000)
# Holds the last host port handed out, in the per-user state root's locks, so processes sharing that root hand them
# out in turn.
_HOST_PORT_CURSOR = "host-ports"


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
    """The next port in ``_HOST_PORTS`` after the last one handed out, skipping those in use.

    Docker binds it later, so nothing may take it meanwhile: a deploy sharing this state root draws it again only after
    the whole range has gone round, and outgoing connections don't use the range by default. The probe alone could not
    keep it: it releases the port, and can't see the sockets of a VM Docker runs in.
    """
    locks = state_root() / "locks"
    ensure_state_dir(locks)
    fd = os.open(locks / _HOST_PORT_CURSOR, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        last = os.read(fd, 16).strip()
        start = int(last) + 1 if last.isdigit() else secrets.choice(_HOST_PORTS)
        for i in range(len(_HOST_PORTS)):
            port = _HOST_PORTS[(start + i - _HOST_PORTS.start) % len(_HOST_PORTS)]
            if _bindable(port):
                os.ftruncate(fd, 0)
                os.pwrite(fd, str(port).encode(), 0)
                return port
    finally:
        os.close(fd)
    raise RuntimeError(f"Every host port from {_HOST_PORTS.start} to {_HOST_PORTS.stop - 1} is in use")


def _bindable(port: int) -> bool:
    """Whether ``port`` is free on every interface, as docker binds it when publishing: a port free on loopback alone
    can still be taken on another interface. A bind that fails for another reason raises."""
    with socket.socket() as s:
        try:
            s.bind(("", port))
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return False
            raise
        return True


class LocalSandbox(VmSandbox):
    """Sandbox implementation that runs everything on the local machine.

    Implements the same VmSandbox interface as the remote providers but executes commands
    locally via asyncio.create_subprocess_exec. This gives 1:1 behavior
    parity with prod for docker compose generation, data loading, etc.

    The /app path used by EnvironmentGatewayProvider is redirected to a local temp
    directory since /app is not writable on macOS.
    """

    ON_THIS_MACHINE = True
    platform = None  # this machine's Docker runs its own platform, and emulates others where it's set up
    # The images its provider's prepare_image had this machine's Docker hold, which its containers start from as they are.
    prepared: frozenset[str] = frozenset()

    def url_from_sandbox(self, url: str) -> str:
        return LocalSandboxProvider.get_external_url(url)

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

    async def build_images(self, artifacts: list) -> None:
        """Build each on this machine's Docker, which runs this sandbox's containers: natively, once per sources. One
        given a platform of its own, as a remote VM records, builds as a VM does, for that platform."""
        if self.platform is not None:
            await super().build_images(artifacts)
            return
        for artifact in artifacts:
            await build_here(artifact)

    async def pull_images(self, image_names: list[str]) -> None:
        """Pull each image but one this machine built from a build context, which no registry holds, and one its Docker
        holds that its provider prepared (``prepared``) or under a name pinned to a digest, so a tag isn't pulled again
        over the image a digest names."""
        held = await asyncio.gather(*(asyncio.to_thread(self._held, name) for name in image_names))
        if pulled := [name for name, here in zip(image_names, held) if not here]:
            await super().pull_images(pulled)

    def _held(self, image_name: str) -> bool:
        return built_here(image_name) or ((image_name in self.prepared or "@sha256:" in image_name)
                                          and present(image_name))

    async def pull_published(self, artifacts: list) -> list:
        """Pull each published image but one this machine's Docker already holds under its name, so a run after the
        first needs no registry."""
        held = await asyncio.gather(*(asyncio.to_thread(present, artifact.image_name) for artifact in artifacts))
        return await super().pull_published([artifact for artifact, here in zip(artifacts, held) if not here])

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
    def closed_network(self) -> str | None:
        """The internal network create_container ran this sandbox's container on, with no route out, if it did."""
        marker = self._work_dir / _CLOSED_NETWORK_FILE
        return marker.read_text() if marker.exists() else None

    @property
    def owns_container(self) -> bool:
        """Whether create_container started ``container_name`` for this sandbox, not an agent placed on it."""
        return (self._work_dir / _CONTAINER_MODE_MARKER).exists()

    async def exit_status(self) -> Optional[int]:
        """The exit status of the container it was made to run, a computer's or the launcher booting a disk, once it
        has exited; None while it runs, when it runs none, or when Docker doesn't say."""
        if not self.owns_container:
            return None
        try:
            state = await asyncio.to_thread(docker, "container", "inspect", "--format",
                                            "{{.State.Status}} {{.State.ExitCode}}", self.container_name,
                                            timeout=_DOCKER_QUERY_SECONDS)
        except (DockerFailed, subprocess.TimeoutExpired):
            return None
        status, _, code = state.partition(" ")
        return int(code) if status in ("exited", "dead") else None

    async def output_tail(self, lines: int) -> list[str]:
        """The last ``lines`` lines that container wrote, its output's and its errors' together as Docker logged them,
        with no order kept between the two; none when it runs none, or when Docker doesn't say."""
        if not self.owns_container:
            return []
        try:
            done = await asyncio.to_thread(subprocess.run, ["docker", "logs", "--tail", str(lines), self.container_name],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           timeout=_DOCKER_QUERY_SECONDS)
        except subprocess.TimeoutExpired:
            return []
        return done.stdout.decode(errors="replace").splitlines() if done.returncode == 0 else []

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    def _rewrite_app_arg(self, arg: str) -> str:
        if arg == "/app" or arg.startswith(("/app/", "/app:")):
            return str(self._work_dir) + arg[4:]
        return arg

    def _rewrite_app_script(self, script: str) -> str:
        return _APP_PATH_PATTERN.sub(lambda _: str(self._work_dir), script)

    async def copy_trust_dir_into_container(self, trust_dir: Path, container: str, destination: str) -> None:
        """Directly: the container is on this machine's Docker, and this sandbox's shell would rewrite the path."""
        await asyncio.to_thread(_copy_into_container, trust_dir, container, destination)

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
        then, once none is left to run from it, what its steps staged in /tmp is removed. With no Docker installed,
        nothing of it can have been started; with Docker unreachable, the teardown fails only for a sandbox whose folder
        records Docker objects of its own.
        """
        failures: list[Exception] = []
        try:
            if not await _stop_marked(self.sandbox_id):
                raise RuntimeError(f"Sandbox {self.sandbox_id} still has marked processes")
            await asyncio.to_thread(_remove_staged, self.sandbox_id)
        except Exception as error:
            failures.append(error)
        try:
            code, _, stderr = await self.exec_with_output("docker", "version", "--format", "{{.Server.Version}}")
        except FileNotFoundError:
            code, stderr = None, ""
        if code == 0:
            await self._remove_docker_objects(failures)
        elif code is not None:
            reason = stderr.strip()[-_ERROR_TAIL_CHARS:]
            if self._made_docker_objects:
                failures.append(RuntimeError(f"Docker can't be reached to remove sandbox {self.sandbox_id}'s container, "
                                             f"compose stack or network: {reason}"))
            else:
                logger.warning("Could not reach Docker (%s); anything sandbox %s's steps started with it is still there. "
                               "Remove it with: docker rm -f $(docker ps -aq --filter label=%s=%s)",
                               reason, self.sandbox_id, SANDBOX_LABEL, self.sandbox_id)
        if failures:
            raise RuntimeError("Local sandbox teardown incomplete: " + "; ".join(str(error) for error in failures)) from failures[0]

    @property
    def _made_docker_objects(self) -> bool:
        """Whether its folder records Docker objects of its own: its container, a compose stack or a closed network."""
        return self.owns_container or self.closed_network is not None or (self._work_dir / "docker-compose.yml").exists()

    async def _remove_docker_objects(self, failures: list[Exception]) -> None:
        """Remove its container, compose stack and labeled objects, each attempted whatever the others did, adding
        what failed to ``failures``."""
        try:
            await self._remove_own_container()
        except Exception as error:
            failures.append(error)
        if self.mode == SANDBOX_MODE_VM and (self._work_dir / "docker-compose.yml").exists():
            try:
                await self.exec_script(
                    f"cd {shlex.quote(str(self._work_dir))} && docker compose down -v --remove-orphans"
                )
            except Exception as error:
                failures.append(error)
        if (self.mode == SANDBOX_MODE_VM and not self.owns_container) or self.closed_network is not None:
            try:
                await self._remove_labeled()
            except Exception as error:
                failures.append(error)

    async def _remove_own_container(self) -> None:
        """Remove this sandbox's named container, or verify that it is absent."""
        name = f"name=^{re.escape(self.container_name)}$"
        code, stdout, stderr = await self.exec_with_output("docker", "ps", "-aq", "--filter", name)
        if code != 0:
            raise RuntimeError(f"Could not list sandbox {self.sandbox_id}'s container: "
                               f"{stderr.strip()[-_ERROR_TAIL_CHARS:]}")
        if containers := stdout.split():
            await self.exec_script(f"docker rm -f {shlex.join(containers)} >/dev/null")

    async def _remove_labeled(self) -> None:
        """Remove labeled containers and networks; images can still be shared and are best-effort."""
        label = f"label={SANDBOX_LABEL}={self.sandbox_id}"
        failures: list[Exception] = []
        for resource, args in (
            ("containers", ("docker", "ps", "-aq", "--filter", label)),
            ("networks", ("docker", "network", "ls", "-q", "--filter", label)),
        ):
            try:
                code, stdout, stderr = await self.exec_with_output(*args)
                if code != 0:
                    raise RuntimeError(f"Could not list sandbox {self.sandbox_id}'s {resource}: "
                                       f"{stderr.strip()[-_ERROR_TAIL_CHARS:]}")
                if identifiers := stdout.split():
                    command = "docker rm -f" if resource == "containers" else "docker network rm"
                    await self.exec_script(f"{command} {shlex.join(identifiers)} >/dev/null")
            except Exception as error:
                failures.append(error)
        q = shlex.quote(label)
        await self.exec_script(
            f"ids=$(docker images -q --filter {q} | sort -u); [ -z \"$ids\" ] || docker rmi -f $ids >/dev/null 2>&1; "
            f"exit 0"
        )
        if failures:
            raise RuntimeError("; ".join(str(error) for error in failures)) from failures[0]

    async def exec(self, *command: str) -> Any:
        """Execute a command locally via subprocess.

        Strips 'sudo' and points /app at the local work directory, both as a path argument and
        inside the script of a top-level ``bash -c``, unless the command runs in a container. The
        command carries ``_SANDBOX_ENV``, so teardown can stop whatever it leaves running.
        Returns an object with .stdout, .stderr streams and .wait() method, matching the
        interface expected by exec_with_output().
        """
        return await self._spawn(command, asyncio.subprocess.DEVNULL)

    async def _exec_with_stdin(self, script: str, stdin: AsyncIterator[bytes]) -> tuple[int, str, str]:
        """``bash -c script`` as ``exec`` runs it, its standard input the bytes ``stdin`` yields."""
        process = await self._spawn(("bash", "-c", script), asyncio.subprocess.PIPE)

        async def feed() -> None:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                async for piece in stdin:
                    process.stdin.write(piece)
                    await process.stdin.drain()
            process.stdin.close()

        try:
            _, stdout, stderr = await asyncio.gather(feed(), process.stdout.read(), process.stderr.read())
            return await process.wait(), stdout.decode(errors="replace"), stderr.decode(errors="replace")
        except BaseException:
            await _stop(process)
            raise

    async def _spawn(self, command: tuple[str, ...], stdin: int) -> asyncio.subprocess.Process:
        cmd = [c for c in command if c != "sudo"]
        if not _runs_in_container(cmd):
            is_script = cmd[:2] == ["bash", "-c"]
            cmd = [
                self._rewrite_app_script(c) if is_script and i == 2 else self._rewrite_app_arg(c)
                for i, c in enumerate(cmd)
            ]
        logger.debug(f"LocalSandbox exec: {' '.join(cmd)}")
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, _SANDBOX_ENV: self.sandbox_id},
        )

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
    if not (found := await asyncio.to_thread(_running_marked, sandbox_id, {})):
        return True
    logger.info("Stopping %d process(es) sandbox %s left running on this machine", len(found), sandbox_id)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in found:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)
        for _ in range(_REAP_SECONDS * 10):
            await asyncio.sleep(0.1)
            if not (found := await asyncio.to_thread(_running_marked, sandbox_id, found)):
                return True
    logger.warning("Processes %s that sandbox %s started are still running; keeping what it staged in /tmp",
                   sorted(found), sandbox_id)
    return False


def _running_marked(sandbox_id: str, found: dict[int, int]) -> dict[int, int]:
    """The running processes carrying ``sandbox_id``'s marker, and those of ``found`` that haven't exited, each with its
    start time. One found before counts until it exits: its environment reads empty while it execs or is exiting."""
    running = {pid: start for pid, start in found.items() if _started(pid) == start}
    for pid in _marked_pids(sandbox_id):
        if pid not in running and (start := _started(pid)) is not None:
            running[pid] = start
    return running


def _started(pid: int) -> int | None:
    """When ``pid`` started, in clock ticks since boot, or None once it has exited (a zombie, or reaped)."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):  # a process that exits mid-read gives ESRCH
        return None
    return None if fields[0] in "ZX" else int(fields[19])


def _remove_staged(sandbox_id: str) -> None:
    """Remove what steps staged in /tmp for ``sandbox_id``, such as a host install's work dir, which a host agent may
    run from until it is stopped. Steps name these with ``scoped_name``, so they end in the sandbox's id."""
    for path in glob.glob(f"/tmp/*-{glob.escape(sandbox_id)}"):
        shutil.rmtree(path, ignore_errors=True)


def local_grant_trust() -> Path | None:
    """The trust files a container on this host needs to use the configured object store's grants: the local
    transfer CA's, when the store is a local one that hands out grants; None otherwise."""
    return local_ca().trust_dir if _granting_store() is not None else None


def local_grants() -> Optional[tuple[int, Path]]:
    """The port of the configured object store's grant server, started, and the local transfer CA's trust files, when
    the store is a local one handing out grants; None otherwise."""
    store = _granting_store()
    return (store.grant_port(), local_ca().trust_dir) if store is not None else None


def _granting_store() -> Optional[LocalFilesystemObjectStore]:
    """The configured object store, or the local one a run routes to, when it's a local one that hands out grants."""
    store = config.get_config().get_object_store()
    if isinstance(store, LocalRunObjectStore):
        store = store.local
    return store if isinstance(store, LocalFilesystemObjectStore) and store.supports_transfer_grants else None


async def start_trusting(sandbox: VmSandbox, container: str, trust_dir: Path) -> None:
    """Copy ``trust_dir`` into the created ``container`` where ``LOCAL_TRUST_ENV`` points, as the sandbox's host can
    reach it, then start it. A container that cannot be given the files or started is removed, so its name is free
    for the next attempt."""
    try:
        await sandbox.copy_trust_dir_into_container(trust_dir, container, LOCAL_TRUST_DIR)
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


def local_sandbox_ids() -> list[str]:
    """The ids of the local sandboxes with a work folder under the sandbox root, made before anything its sandbox runs:
    a supervisor that gave a run its own ``AGENT_ENV_LOCAL_SANDBOX_DIR`` terminates each after killing it, removing its
    folder, and keeps one whose ``LocalSandboxProvider().terminate_sandbox`` raises, not fully gone, to retry."""
    root = _local_sandbox_root_path()
    if not root.is_dir():
        return []
    prefix, found = "agent-env-", set()
    for entry in root.iterdir():
        sandbox_id, _, suffix = entry.name.removeprefix(prefix).rpartition("-")
        if entry.name.startswith(prefix) and sandbox_id and suffix and entry.is_dir() and not entry.is_symlink():
            found.add(sandbox_id)
    return sorted(found)


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
    ON_THIS_MACHINE = True
    # Its containers start from an image's name, or one it builds first from the image's build context (prepare_image);
    # a VM it creates (create_vm) loads images.
    SANDBOX_ACCEPTS = Accepts.NAME_OR_CONTEXT
    CONTAINER_ACCEPTS = Accepts.NAME_OR_CONTEXT
    WEBSOCKETS = True
    SHARED_MEMORY = True

    @property
    def _prepared(self) -> set[str]:
        """The images prepare_image had this machine's Docker hold, by name; made on first use, so a subclass with an
        ``__init__`` of its own needn't call this one's."""
        return self.__dict__.setdefault("_prepared_images", set())

    def url_from_sandbox(self, url: str) -> str:
        return self.get_external_url(url)

    @classmethod
    def native_platforms(cls) -> frozenset[str]:
        return frozenset({host_platform()})

    @classmethod
    def disk_image_platforms(cls) -> frozenset[str]:
        return kvm_platforms()

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        """Unrestricted, or a container with no network at all (``create_container``)."""
        return not policy.restricts_egress or policy.closed

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
        env: Optional[dict[str, str]] = None,
    ) -> LocalSandbox:
        """This machine as a VM, or with ``image``, a computer's disk form booted under KVM in the QEMU launcher's
        container, reached on each of ``exposed_ports``, at the disk's own size: ``disk_size_gb`` is this machine's."""
        refuse_unenforceable_policy(self, network_policy)
        if network_policy is not None and network_policy.restricts_egress:
            raise NetworkPolicyUnsupportedError(f"{type(self).__name__} takes a container's network away, not a VM's")
        if image is not None and host_platform() not in self.disk_image_platforms():
            raise ValueError(f"{type(self).__name__} boots a disk only where this machine's Docker can pass it /dev/kvm")
        # Allocated here rather than in the sandbox constructor: probing for a free port
        # is I/O, and callers that only need a handle must not pay for it. Off the event
        # loop, since a draw waits for any other process's draw to finish.
        port_map = await asyncio.to_thread(lambda: {port: _free_host_port() for port in exposed_ports or []})
        sandbox = LocalSandbox(port_map=port_map)
        (sandbox.work_dir / _PORT_MAP_FILE).write_text(json.dumps(sandbox._port_map))
        sandbox.network_policy = self.effective_network_policy(network_policy)
        sandbox.prepared = frozenset(self._prepared)
        if image is not None:
            (sandbox.work_dir / _VM_SIZE_FILE).write_text(json.dumps({"cpu": cpu, "memory": memory}))
            try:
                await self._boot_disk(sandbox, image=image, env=env or {})
                (sandbox.work_dir / _CONTAINER_MODE_MARKER).write_text(sandbox.container_name)
            except BaseException:
                with contextlib.suppress(Exception):
                    await sandbox.terminate()
                    await asyncio.to_thread(remove_local_work_dir, sandbox.sandbox_id)
                raise
        return sandbox

    async def reset_vm(self, sandbox: Sandbox, *, image: str, env: dict[str, str]) -> None:
        """The launcher in the sandbox's container booting its disk afresh with ``env``; refused for a container an
        agent-env older than reset in place started, before an upgrade."""
        try:
            await asyncio.to_thread(docker, "exec", "-i", sandbox.container_name, *RESET, input=reset_input(env),
                                    timeout=_RESET_SECONDS)
        except DockerFailed as e:
            if e.returncode == UNRESETTABLE:
                raise RuntimeError(unresettable(f"the computer in container {sandbox.container_name}")) from None
            raise

    async def _boot_disk(self, sandbox: LocalSandbox, *, image: str, env: dict[str, str]) -> None:
        """The launcher booting ``image`` as the sandbox's container, its ports published as a container's are."""
        size = json.loads((sandbox.work_dir / _VM_SIZE_FILE).read_text())
        await sandbox.pull_images([image])
        machine = await machine_for(image)
        ports = list(sandbox._port_map)
        publish = [arg for port in ports for spec in port_bindings(sandbox.host_ips, sandbox.host_port(port), port)
                   for arg in ("-p", spec)]
        grants = await asyncio.to_thread(local_grants)
        running = asyncio.ensure_future(asyncio.to_thread(
            docker, "run", "-d", "--name", sandbox.container_name, "--label", f"{SANDBOX_LABEL}={sandbox.sandbox_id}",
            *publish, *machine.args, *launcher_args(ports, env, size["cpu"], size["memory"]),
            *(_grant_args(*grants) if grants else ()), machine.image, timeout=_DOCKER_QUERY_SECONDS))
        try:
            await asyncio.shield(running)
        except asyncio.CancelledError:  # the container docker is still starting must exist before teardown removes it
            with contextlib.suppress(Exception):
                await running
            raise

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
        shm_mb: Optional[int] = None,
    ) -> LocalSandbox:
        refuse_unenforceable_policy(self, network_policy)
        if network_policy is not None and network_policy.closed:
            sandbox = await self._create_closed(image_name=image_name, port=port, env=env, timeout=timeout,
                                                attribution=attribution, shm_mb=shm_mb)
        else:
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
                shm_mb=shm_mb,
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
                if sandbox.closed_network is not None:
                    await sandbox._remove_labeled()
                raise
        return sandbox

    async def _create_closed(self, *, image_name: str, port: int, env: dict[str, str], timeout: int,
                             attribution: Optional[Attribution], shm_mb: Optional[int] = None) -> LocalSandbox:
        """The container on an internal network of its own, with no route out, and a relay on that network and the
        default bridge publishing its ``port``: Docker publishes no port of a container on an internal network alone."""
        relay = await relay_image()
        sandbox = await self.create_vm(exposed_ports=[port], timeout=timeout, attribution=attribution)
        try:
            await sandbox.pull_images([image_name])
            network = sandbox.scoped_name("closed")
            label = shlex.quote(f"{SANDBOX_LABEL}={sandbox.sandbox_id}")
            (sandbox.work_dir / _CLOSED_NETWORK_FILE).write_text(network)
            await sandbox.exec_script(f"docker network create --internal --label {label} {network} >/dev/null")
            await self._run_closed(sandbox, relay, image_name=image_name, port=port, env=env, shm_mb=shm_mb)
            sandbox.mode = SANDBOX_MODE_CONTAINER
        except BaseException:
            try:
                await sandbox.terminate()
                await asyncio.to_thread(remove_local_work_dir, sandbox.sandbox_id)
            except Exception:
                pass
            raise
        sandbox.network_policy = NetworkPolicy(mode="allowlist")
        return sandbox

    async def reset_container(self, sandbox: Sandbox, *, image_name: str, port: int, env: dict[str, str],
                              shm_mb: Optional[int] = None) -> None:
        """Reset the sandbox's container to a fresh one, and on a closed network its relay too, since the fresh
        container's address on that network may differ."""
        if not (isinstance(sandbox, LocalSandbox) and sandbox.closed_network):
            await super().reset_container(sandbox, image_name=image_name, port=port, env=env, shm_mb=shm_mb)
            return
        await sandbox.exec_script(f"docker rm -f {shlex.quote(sandbox.container_name)} "
                                  f"{shlex.quote(sandbox.scoped_name('relay'))} > /dev/null")
        await self._run_closed(sandbox, await relay_image(), image_name=image_name, port=port, env=env, shm_mb=shm_mb)

    async def _run_closed(self, sandbox: LocalSandbox, relay: str, *, image_name: str, port: int,
                          env: dict[str, str], shm_mb: Optional[int] = None) -> None:
        """The container on the sandbox's closed network, then the relay publishing its ``port`` at its address there."""
        network, relay_name = sandbox.closed_network, sandbox.scoped_name("relay")
        label = shlex.quote(f"{SANDBOX_LABEL}={sandbox.sandbox_id}")
        await self._start_container(sandbox, image_name=image_name, port=port, env=env, shm_mb=shm_mb)
        address = await sandbox.exec_script(
            f"docker inspect -f '{{{{(index .NetworkSettings.Networks \"{network}\").IPAddress}}}}' "
            f"{shlex.quote(sandbox.container_name)}")
        publish = " ".join(f"-p {spec}" for spec in port_bindings(sandbox.host_ips, sandbox.host_port(port), port))
        await sandbox.exec_script(
            f"docker run -d --name {relay_name} --label {label} {publish} {relay} {port}={address.strip()}:{port} "
            f">/dev/null && docker network connect {network} {relay_name}")

    def _publish_args(self, sandbox: VmSandbox, port: int) -> str:
        """A container with no network is reached through its relay; any other is published as everywhere."""
        network = sandbox.closed_network if isinstance(sandbox, LocalSandbox) else None
        return f"--network {shlex.quote(network)}" if network is not None else super()._publish_args(sandbox, port)

    async def _start_container(self, sandbox: VmSandbox, *, image_name: str, port: int, env: dict[str, str],
                               shm_mb: Optional[int] = None) -> None:
        """Where the configured store hands out local grants, create the container, copy the local transfer CA's
        trust files in, then start it, so its TLS clients trust the grant server. Variables the caller sets win."""
        trust_dir = await asyncio.to_thread(local_grant_trust)
        if trust_dir is None:
            await super()._start_container(sandbox, image_name=image_name, port=port, env=env, shm_mb=shm_mb)
            return
        args = self._container_args(sandbox, image_name=image_name, port=port, env={**LOCAL_TRUST_ENV, **env},
                                    shm_mb=shm_mb)
        await sandbox.exec_script(f"docker create {args} > /dev/null")
        await start_trusting(sandbox, sandbox.container_name, trust_dir)

    async def prepare_image(self, image: Any, *, attribution: Optional[Attribution] = None) -> None:
        """Have this machine's Docker hold the image the local provider starts containers from by name: one that is only
        a build context built here, and one with a digest pulled by it, or built from its build context when it can't
        be."""
        if image.context_only:
            await build_here(image)
        elif image.image_digest:
            await pull_here(image)
            self._prepared.add(image.image_name)

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
        """The sandbox ``sandbox_id``; a disk's, with this process's grant server forwarded into its launcher, as
        lume's VMs get theirs as another process reconnects: each process that hands it grants serves them on a port
        of its own."""
        sandbox = _found(sandbox_id)
        if _is_disk(sandbox.work_dir) and (grants := await asyncio.to_thread(local_grants)) is not None:
            await asyncio.to_thread(_forward_grants, sandbox.container_name, grants[0])
        return sandbox

    async def terminate_sandbox(self, sandbox_id: str) -> None:
        """Terminate the sandbox ``sandbox_id``, forwarding no grant port into a disk's launcher it is removing."""
        await _found(sandbox_id).terminate()

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
    """``url``, which this machine reaches, as a container on ``sandbox_type`` reaches it: one on this machine (its
    provider's ``ON_THIS_MACHINE``) reaches this machine's loopback as ``host.docker.internal``; one elsewhere reaches
    the URL as it is, or not at all."""
    return LocalSandboxProvider.get_external_url(url) if on_this_machine_provider(sandbox_type) else url


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
    if platform.system() != "Linux" or docker("info", "--format", "{{.OperatingSystem}}",
                                              timeout=_DOCKER_QUERY_SECONDS) == "Docker Desktop":
        return ("127.0.0.1",)
    # Listing, unlike inspecting, answers a missing network with nothing rather than an error.
    if not docker("network", "ls", "--quiet", "--filter", "name=^bridge$", timeout=_DOCKER_QUERY_SECONDS):
        return ("127.0.0.1",)
    return ("127.0.0.1", docker("network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}",
                                timeout=_DOCKER_QUERY_SECONDS))


def _grant_args(port: int, trust: Path) -> list[str]:
    """The ``docker run`` arguments that let a disk's computer use this machine's local grants, as lume's VMs do: it
    finds ``host.docker.internal`` at QEMU's address for the launcher, where the launcher forwards the grant server's
    ``port`` to this machine, and trusts the local transfer CA, whose ``trust`` files the launcher puts in its seed."""
    return [*(arg for entry in _EXTRA_HOSTS for arg in ("--add-host", entry)),
            "--mount", f"type=bind,src={trust},dst={_LAUNCHER_TRUST},readonly",
            "-e", f"VM_HOST_PORTS={port}", "-e", f"VM_TRUST={_LAUNCHER_TRUST}"]


def _found(sandbox_id: str) -> LocalSandbox:
    """The sandbox ``sandbox_id`` from its work directory, in VM mode even for a container it owns: exec runs on this
    host, so steps must ``docker exec`` into it."""
    work_dir = LocalSandbox.find_work_dir(sandbox_id)
    if work_dir is None:
        raise RuntimeError(f"Local sandbox work directory not found for sandbox_id={sandbox_id!r}")
    try:
        ports = {int(c): h for c, h in json.loads((work_dir / _PORT_MAP_FILE).read_text()).items()}
    except (OSError, ValueError):  # none written, or cut short by a run killed while writing it
        ports = None
    return LocalSandbox(sandbox_id=sandbox_id, work_dir=work_dir, port_map=ports)


def _is_disk(work_dir: Path) -> bool:
    """Whether the sandbox in ``work_dir`` is a disk's computer: ``create_vm`` records the size of the VM it boots."""
    return (work_dir / _VM_SIZE_FILE).exists()


def _forward_grants(container: str, port: int) -> None:
    """``port`` forwarded in a disk's launcher ``container`` (``launch forward``), unless it is already, as a second
    listener on it exits; one that can't be is logged, the computer then reaching no grant of this process's."""
    try:
        docker("exec", "-d", container, *forward(port), timeout=_DOCKER_QUERY_SECONDS)
    except DockerFailed as e:
        logger.warning("Couldn't forward grant port %s into %s, so grants from this process won't reach its computer: "
                       "%s", port, container, e)


_DOCKER_QUERY_SECONDS = 30
#: How long a reset may take: longer than the minute the launcher's reset waits for the computer to start again.
_RESET_SECONDS = 90
#: Where a disk's launcher finds the local transfer CA's trust files.
_LAUNCHER_TRUST = "/agentenv-trust"
