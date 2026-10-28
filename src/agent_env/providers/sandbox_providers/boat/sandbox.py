"""A boat sandbox presented through agent-env's ``VmSandbox`` contract.

boat sandboxes boot Ubuntu with Docker and Compose running, as the user ``user``, who has passwordless sudo, so the
``sudo`` agent-env prefixes is kept. Every command runs detached and is followed to its end: a detached command has no
time limit and its output stream reconnects without a gap, where a synchronous one stops at 600 s. Hosted ports reach
the sandbox through boat's tunnel, so the sandbox opens no firewall ports.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import shlex
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import IO, TYPE_CHECKING, Any

import httpx
from boat_sdk import run_command
from boat_sdk.exceptions import ApiException
from boat_sdk.models import FileUrlRequest

from agent_env.providers.sandbox_providers.boat import lifecycle
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM

if TYPE_CHECKING:
    from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)

# Bytes past what one exec carries go through a signed upload URL: one PUT of any size, streamed.
_UPLOAD_URL_TTL = 3600
_UPLOAD_TIMEOUT = httpx.Timeout(600.0, connect=30.0)
_UPLOAD_PART_BYTES = 4 * 1024 * 1024

# What exec reports when the transport drops without an exit status; exec_script retries only this.
UNKNOWN_EXIT_CODE = -1
# How long a cancelled exec waits for its command to be stopped before the cancellation is raised.
_STOP_WAIT = 30.0

RunCommand = Callable[..., Iterator[dict[str, Any]]]


class _BytesReader:
    """The ``StreamReader`` subset ``Sandbox.exec_with_output`` reads."""

    def __init__(self, value: str):
        self._value = value.encode()

    async def read(self) -> bytes:
        return self._value


class _CompletedProcess:
    """A finished command as the process ``Sandbox.exec`` returns."""

    def __init__(self, exit_code: int | None, stdout: str, stderr: str):
        self.stdout = _BytesReader(stdout)
        self.stderr = _BytesReader(stderr)
        self._exit_code = UNKNOWN_EXIT_CODE if exit_code is None else int(exit_code)

    async def wait(self) -> int:
        return self._exit_code


class _Run:
    """One command, shared by the event loop and the thread that follows it, so a cancel stops it whether or not it
    has started yet."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process_id: int | None = None
        self.cancelled = False


def _follow(api: Any, sandbox_id: str, script: str, run: _Run, run_command: RunCommand) -> list[dict[str, Any]]:
    """Every frame of ``script`` after its start, run detached in the sandbox. A command cancelled before it started is
    stopped as soon as it does."""
    frames = []
    for frame in run_command(api, sandbox_id, script):
        if frame.get("type") == "started":
            with run.lock:
                run.process_id = frame.get("processId")
                stop = run.cancelled
            if stop:
                try:
                    api.stop_command(sandbox_id, run.process_id)
                except (ApiException, *lifecycle.NETWORK_ERRORS) as exc:
                    logger.warning("Stopping command %s in boat sandbox %s failed: %s", run.process_id, sandbox_id, exc)
                return frames
            continue
        frames.append(frame)
    return frames


def _completed(frames: list[dict[str, Any]]) -> _CompletedProcess:
    stdout = "".join(str(frame.get("data", "")) for frame in frames if frame.get("type") == "stdout")
    stderr = "".join(str(frame.get("data", "")) for frame in frames if frame.get("type") == "stderr")
    for frame in reversed(frames):
        if frame.get("type") == "exit":
            return _CompletedProcess(frame.get("exitCode"), stdout, stderr)
        if frame.get("type") == "error":
            message = frame.get("message") or frame.get("error") or "the command's output stream failed"
            return _CompletedProcess(UNKNOWN_EXIT_CODE, stdout, f"{stderr}{message}")
    return _CompletedProcess(UNKNOWN_EXIT_CODE, stdout, f"{stderr}the command's output stream ended without an exit")


class BoatSandbox(VmSandbox):
    """A boat sandbox with a Docker daemon. ``api`` is its provider's ``BoatApi``."""

    type = "boat"
    # The image starts dockerd at boot, and the first command can be ahead of it: polled this often, this many times.
    _DOCKER_READY_ATTEMPTS = 90
    _DOCKER_READY_INTERVAL_SECONDS = 1

    def __init__(
        self,
        api: Any,
        sandbox_id: str,
        *,
        tunnel_urls: Mapping[int, str],
        network_policy: NetworkPolicy | None,
        retry: lifecycle.Retry,
        run_command: RunCommand = run_command,
    ):
        self._api = api
        self._retry = retry
        # Injectable so unit tests never stream from boat.
        self._run_command = run_command
        self.sandbox_id = sandbox_id
        self.mode = SANDBOX_MODE_VM
        self.vnc_url = None
        self.tunnel_urls = dict(tunnel_urls)
        self.network_policy = network_policy

    async def terminate(self) -> None:
        """Delete the sandbox and its data; one already gone is not an error."""
        if not await lifecycle.delete_sandbox(self._api, self.sandbox_id, self._retry):
            raise RuntimeError(f"boat sandbox {self.sandbox_id} could not be deleted")

    async def exec(self, *command: str) -> _CompletedProcess:
        """Run argv as one shell command, detached and followed to its end. A cancelled exec stops the command."""
        run = _Run()
        script = shlex.join(command)
        try:
            frames = await lifecycle.stream(_follow, self._api, self.sandbox_id, script, run, self._run_command)
        except asyncio.CancelledError:
            with run.lock:
                run.cancelled = True
                process_id = run.process_id
            if process_id is not None:
                await lifecycle.settle(self._stop(process_id), within=_STOP_WAIT)
            raise
        except lifecycle.NETWORK_ERRORS as exc:
            return _CompletedProcess(UNKNOWN_EXIT_CODE, "", f"{type(exc).__name__}: {exc}")
        except ApiException as exc:
            # boat failed, or limited this client: the command may or may not have run.
            if exc.status is None or exc.status >= 500 or exc.status == 429:
                return _CompletedProcess(UNKNOWN_EXIT_CODE, "", f"boat {exc.status} {lifecycle.error_code(exc)}")
            raise
        return _completed(frames)

    async def _stop(self, process_id: int) -> None:
        try:
            await lifecycle.call(self._api.stop_command, self.sandbox_id, process_id)
        except (ApiException, *lifecycle.NETWORK_ERRORS) as exc:
            logger.warning("Stopping command %s in boat sandbox %s failed: %s", process_id, self.sandbox_id, exc)

    @classmethod
    def docker_ready_script(cls) -> str:
        """A script that waits for the Docker daemon, then prints the Compose version, failing when either is missing."""
        return (
            f"for _ in $(seq {cls._DOCKER_READY_ATTEMPTS}); do\n"
            "  docker info >/dev/null 2>&1 && exec docker compose version\n"
            f"  sleep {cls._DOCKER_READY_INTERVAL_SECONDS}\n"
            "done\n"
            'echo "docker daemon did not become ready" >&2\n'
            "exit 1\n"
        )

    async def wait_for_vm(self) -> None:
        """Wait for Docker, which boat starts at boot, and require Compose v2, which ``deploy_env`` uses."""
        code, version, err = await self.exec_with_output("bash", "-c", self.docker_ready_script())
        if code != 0:
            raise RuntimeError(
                f"Docker with Compose v2 is not ready in boat sandbox {self.sandbox_id}: {(err or version).strip()}"
            )
        logger.info("Docker Compose ready in boat sandbox %s: %s", self.sandbox_id, version.strip())

    async def setup_vm_for_gateway(self, exposed_ports: list[int] | None = None) -> None:
        """Wait for Docker and Compose; hosted ports need no firewall rule."""
        await self.wait_for_vm()

    async def _write_bytes_to_vm_path(self, data: bytes, vm_path: str) -> None:
        """Bytes one exec carries go as base64 in it; more go through an upload URL."""
        if len(base64.b64encode(data)) <= self._WFT_CHUNK_BYTES:
            await super()._write_bytes_to_vm_path(data, vm_path)
            return
        await self._upload(data, vm_path)

    async def _write_unsigned_object(self, object_store: ObjectStore, object_url: str, vm_path: str) -> None:
        """Stream an object through an upload URL, not as base64 over exec."""
        source = await asyncio.to_thread(object_store.open, object_url)
        try:
            await self._upload(_read_parts(source), vm_path)
        finally:
            source.close()

    async def _upload(self, body: bytes | AsyncIterator[bytes], vm_path: str) -> None:
        """PUT ``body`` to a file in ``/tmp``, where boat writes uploads, then move it to ``vm_path`` as root, as an
        exec would write it, and check its sha256."""
        request = FileUrlRequest(path=f"/tmp/.agentenv-upload-{uuid.uuid4().hex}", ttl_seconds=_UPLOAD_URL_TTL)
        grant = await lifecycle.call(self._api.create_upload_url, self.sandbox_id, request)
        digest = hashlib.sha256()

        async def parts() -> AsyncIterator[bytes]:
            async for part in body:  # type: ignore[union-attr]
                digest.update(part)
                yield part

        if isinstance(body, bytes):
            digest.update(body)
        staged, target = shlex.quote(grant.path), shlex.quote(vm_path)
        try:
            async with httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT) as client:
                response = await client.request(grant.method, grant.url, content=body if isinstance(body, bytes) else parts())
                response.raise_for_status()
            written = await self.exec_script(f"mv -f {staged} {target} && chown root:root {target} && sha256sum {target}")
        except BaseException:
            await self._remove_vm_temp_file(grant.path)
            raise
        actual = written.split()[0] if written.split() else ""
        if actual != digest.hexdigest():
            raise RuntimeError(f"{vm_path} in boat sandbox {self.sandbox_id} does not match the upload: its sha256 is "
                               f"{actual or 'missing'}, the upload's {digest.hexdigest()}")


async def _read_parts(source: IO[bytes]) -> AsyncIterator[bytes]:
    """``source`` in upload-sized parts, read off the event loop."""
    while part := await asyncio.to_thread(source.read, _UPLOAD_PART_BYTES):
        yield part
