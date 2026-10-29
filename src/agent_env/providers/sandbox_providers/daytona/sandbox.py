"""A Daytona sandbox presented through agent-env's ``VmSandbox`` contract.

Commands run as one-shot Daytona sessions: a session keeps stdout and stderr apart, which agent-env
relies on when it reads base64 and checksums off stdout, and ``bash -c <quoted argv>`` keeps each
argument intact whatever the sandbox's login shell is. Public ports are served by Daytona's preview
proxy, so the sandbox opens no firewall ports.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from daytona import (
    DaytonaBadGatewayError,
    DaytonaConflictError,
    DaytonaConnectionError,
    DaytonaConnectionTimeoutError,
    DaytonaNotFoundError,
    DaytonaServiceUnavailableError,
    DaytonaSessionEndedError,
    DaytonaTimeoutError,
    SessionExecuteRequest,
)

from agent_env.config import get_config
from agent_env.providers.sandbox_providers.daytona import lifecycle, transfer
from agent_env.providers.sandbox_providers.daytona.image import DOCKER_START_SCRIPT
from agent_env.providers.sandbox_providers.daytona.network import allows_host
from agent_env.providers.sandbox_providers.sandbox import CURL_RETRY_FLAGS, NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM

if TYPE_CHECKING:
    from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)

# What exec reports when the transport drops without an exit status; exec_script retries only this.
UNKNOWN_EXIT_CODE = -1
# Failures of the transport, not of the command: they read as UNKNOWN_EXIT_CODE.
_TRANSPORT_ERRORS = (
    DaytonaConnectionError,
    DaytonaConnectionTimeoutError,
    DaytonaBadGatewayError,
    DaytonaServiceUnavailableError,
    DaytonaSessionEndedError,
    DaytonaTimeoutError,
)


class _BytesReader:
    """The ``StreamReader`` subset ``Sandbox.exec_with_output`` reads."""

    def __init__(self, value: str):
        self._value = value.encode()

    async def read(self) -> bytes:
        return self._value


class _CompletedProcess:
    """A finished session command as the process ``Sandbox.exec`` returns."""

    def __init__(self, exit_code: int | None, stdout: str, stderr: str):
        self.stdout = _BytesReader(stdout)
        self.stderr = _BytesReader(stderr)
        self._exit_code = UNKNOWN_EXIT_CODE if exit_code is None else int(exit_code)

    async def wait(self) -> int:
        return self._exit_code


class DaytonaSandbox(VmSandbox):
    """A Daytona sandbox with a Docker daemon. ``client`` returns its provider's SDK client for the running event
    loop; the SDK handle is read again through it when the client it came from was closed or belongs to another
    loop, since a handle is bound to its client's HTTP session."""

    type = "daytona"
    _COMMAND_TIMEOUT = 10
    _LOG_TAIL_LINES = 40
    _DELETE_ATTEMPTS = 5
    _DELETE_RETRY_DELAY = 1.0

    def __init__(
        self,
        client: Callable[[], Any],
        raw: Any,
        *,
        tunnel_urls: Mapping[int, str],
        network_policy: NetworkPolicy | None,
    ):
        self._client = client
        self._raw = raw
        self._raw_client = client()
        self.sandbox_id = raw.id
        self.mode = SANDBOX_MODE_VM
        self.vnc_url = None
        self.tunnel_urls = dict(tunnel_urls)
        self.network_policy = network_policy

    async def handle(self) -> Any:
        """The SDK's ``AsyncSandbox`` for this sandbox, on the provider's current client."""
        client = self._client()
        if client is not self._raw_client:
            self._raw, self._raw_client = await client.get(self.sandbox_id), client
        return self._raw

    async def terminate(self) -> None:
        """Delete the sandbox; one already gone, or already being deleted, is not an error. A 409 for any
        other state change in progress is retried."""
        try:
            raw = await self.handle()
        except DaytonaNotFoundError:
            logger.info("Daytona sandbox %s was already gone", self.sandbox_id)
            return
        for attempt in range(1, self._DELETE_ATTEMPTS + 1):
            try:
                await raw.delete()
                return
            except DaytonaNotFoundError:
                logger.info("Daytona sandbox %s was already gone", self.sandbox_id)
                return
            except DaytonaConflictError:
                if await lifecycle.being_destroyed(raw):
                    logger.info("Daytona sandbox %s is already being deleted", self.sandbox_id)
                    return
                if attempt == self._DELETE_ATTEMPTS:
                    raise
                await asyncio.sleep(self._DELETE_RETRY_DELAY)

    async def exec(self, *command: str) -> _CompletedProcess:
        """Run argv in a session of its own; the inherited ``sudo`` prefixes stay, as commands run
        as the sandbox's sudo-capable user."""
        return await self._run(command, timeout=None)

    async def _run(self, command: tuple[str, ...], *, timeout: float | None) -> _CompletedProcess:
        session = f"ae-{uuid.uuid4().hex}"
        script = "bash -c " + shlex.quote(shlex.join(command))
        request = SessionExecuteRequest(command=script, run_async=False)
        try:
            process = (await self.handle()).process
        except _TRANSPORT_ERRORS as exc:
            return _CompletedProcess(UNKNOWN_EXIT_CODE, "", f"{type(exc).__name__}: {exc}")
        try:
            await process.create_session(session)
            seconds = None if timeout is None else max(1, int(timeout))
            response = await process.execute_session_command(session, request, timeout=seconds)
        except _TRANSPORT_ERRORS as exc:
            return _CompletedProcess(UNKNOWN_EXIT_CODE, "", f"{type(exc).__name__}: {exc}")
        finally:
            try:
                await process.delete_session(session)
            except Exception as exc:  # noqa: BLE001 - the command's own outcome matters more
                logger.debug("Deleting session %s in %s failed: %s", session, self.sandbox_id, exc)
        return _CompletedProcess(response.exit_code, response.stdout or "", response.stderr or "")

    async def _bounded(self, *command: str, timeout: float) -> tuple[int, str, str]:
        process = await self._run(command, timeout=timeout)
        out, err = await asyncio.gather(process.stdout.read(), process.stderr.read())
        return await process.wait(), out.decode(), err.decode()

    async def wait_for_vm(self) -> None:
        """Start dockerd, which Daytona leaves to the sandbox, and poll ``docker info`` until it
        answers, within ``_VM_READY_TIMEOUT``."""
        deadline = time.monotonic() + self._VM_READY_TIMEOUT
        start = ("sudo", "bash", "-c", DOCKER_START_SCRIPT)
        code, _, err = await self._bounded(*start, timeout=self._COMMAND_TIMEOUT * 3)
        if code != 0:
            logger.debug("Starting dockerd in %s exited %s: %s", self.sandbox_id, code, err)
        polls = 0
        while time.monotonic() < deadline:
            polls += 1
            remaining = max(0.001, min(self._COMMAND_TIMEOUT, deadline - time.monotonic()))
            if (await self._bounded("sudo", "docker", "info", timeout=remaining))[0] == 0:
                logger.info("Docker ready in Daytona sandbox %s after %s polls", self.sandbox_id, polls)
                return
            pause = min(self._VM_READY_POLL_INTERVAL, max(0.0, deadline - time.monotonic()))
            await asyncio.sleep(pause)
        log_tail = ("sudo", "tail", "-n", str(self._LOG_TAIL_LINES), "/var/log/dockerd.log")
        _, tail, _ = await self._bounded(*log_tail, timeout=self._COMMAND_TIMEOUT)
        raise RuntimeError(
            f"Docker not ready in Daytona sandbox {self.sandbox_id} after {self._VM_READY_TIMEOUT}s "
            f"({polls} polls); /var/log/dockerd.log tail:\n{tail}"
        )

    async def setup_vm_for_gateway(self, exposed_ports: list[int] | None = None) -> None:
        """Start Docker and require Compose v2, which ``deploy_env`` uses; ports need no firewall."""
        await self.wait_for_vm()
        compose = ("sudo", "docker", "compose", "version")
        code, version, err = await self._bounded(*compose, timeout=self._COMMAND_TIMEOUT)
        if code != 0:
            raise RuntimeError(
                f"Daytona sandbox {self.sandbox_id} has no Docker Compose v2 plugin, which deploy-env "
                f"needs: {(err or version).strip()}"
            )
        logger.info("Docker Compose ready in Daytona sandbox %s: %s", self.sandbox_id, version.strip())

    async def _write_bytes_to_vm_path(self, data: bytes, vm_path: str) -> None:
        async def parts() -> AsyncIterator[bytes]:
            for offset in range(0, max(len(data), 1), transfer.UPLOAD_CHUNK_BYTES):
                yield data[offset : offset + transfer.UPLOAD_CHUNK_BYTES]

        await transfer.upload(self, parts(), vm_path)

    async def _write_unsigned_object(self, object_store: ObjectStore, object_url: str, vm_path: str) -> None:
        """Stream an object through the file API, not as base64 over exec."""
        source = await asyncio.to_thread(object_store.open, object_url)
        try:
            await transfer.upload(self, transfer.read_parts(source), vm_path)
        finally:
            source.close()

    def _can_download(self, url: str | None) -> bool:
        """Whether the sandbox can fetch ``url`` itself; an unknown policy counts as no."""
        if url is None:
            return False
        host = urlparse(url).hostname
        return host is not None and allows_host(self.network_policy, host)

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        """Curl a signed URL the sandbox's egress policy admits; upload anything else through the file API, which
        needs no egress, so a restricted policy is never widened."""
        object_store = get_config().get_object_store_at(object_url)
        signed = await asyncio.to_thread(object_store.signed_get_url, object_url)
        if self._can_download(signed):
            await self.exec_script(f"curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(signed)} -o {shlex.quote(vm_path)}")
            return
        await self._write_unsigned_object(object_store, object_url, vm_path)

    async def _load_tarballs(self, artifacts: list) -> None:
        """Load image tarballs, uploading through the file API those the egress policy can't download."""
        signed_urls = await self._signed_image_urls(artifacts)
        reachable = [url if self._can_download(url) else None for url in signed_urls]
        await self._load_docker_images(artifacts, reachable)
