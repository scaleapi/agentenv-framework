"""A boxd machine presented through the :class:`VmSandbox` contract.

boxd machines boot Ubuntu with Docker and Compose v2 running, so the gateway's docker-compose, an agent's
``docker run`` and image loading run under Docker inside the machine. Commands run as the machine's ``boxd`` user,
who has passwordless sudo, so the ``sudo`` agent-env prefixes is kept. boxd applies the egress allowlist outside the
machine, so one allowlist covers ``dockerd`` and every container it runs.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import shlex
from typing import Any
from urllib.parse import urlparse

from boxd import APIConnectionError, BoxdError, NotFoundError

from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import CURL_RETRY_FLAGS, NetworkMode, NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM

logger = logging.getLogger(__name__)

READY_TIMEOUT_SEC = 180
# The image starts dockerd at boot; a machine exec reaches within milliseconds can be ahead of it.
DOCKER_READY = r"""
for _ in $(seq 90); do
  docker info >/dev/null 2>&1 && exec docker compose version
  sleep 1
done
echo "docker daemon did not become ready" >&2
exit 1
"""
# boxd stops relaying a command that prints nothing for 30 minutes, and its exec never returns. A NUL on stderr every
# minute keeps a quiet command's stream alive; exec drops them from what it returns. The loop's sleep holds none of
# the exec's streams, so a finished command returns at once.
KEEPALIVE_SEC = 60
_KEEPALIVE = ("( while sleep {interval} </dev/null >/dev/null 2>&1; do printf '\\0' >&2; done ) & k=$!; "
              "{line}; c=$?; kill $k 2>/dev/null; exit $c")
# What exec returns when the stream drops without an exit status; exec_script retries only this.
_UNKNOWN_EXIT_CODE = -1
# Statuses of a stream boxd ended before the command did, which a rerun can get past.
_LOST_STREAM_CODES = frozenset({"deadline_exceeded", "internal", "unknown"})


def allows_host(policy: NetworkPolicy, host: str) -> bool:
    """Whether ``policy`` lets the machine reach ``host`` by boxd's rules: an exact name, a ``*.domain`` entry, which
    covers every name under ``domain`` but not ``domain`` itself, or an address in a listed address or range."""
    if not policy.restricts_egress:
        return True
    host = host.lower().rstrip(".")
    for entry in policy.allow_hosts:
        entry = entry.lower().rstrip(".")
        if entry == host or (entry.startswith("*.") and host.endswith(entry[1:])):
            return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(
        address in ipaddress.ip_network(entry, strict=False)
        for entry in (*policy.allow_cidrs, *policy.allow_hosts)
        if _is_address(entry)
    )


def policy_from_entries(entries: list[str]) -> NetworkPolicy:
    """The policy a machine's egress allowlist applies; boxd reads an empty allowlist as unrestricted."""
    if not entries:
        return NetworkPolicy()
    return NetworkPolicy(
        mode=NetworkMode.ALLOWLIST,
        allow_hosts=tuple(entry for entry in entries if not _is_address(entry)),
        allow_cidrs=tuple(entry for entry in entries if _is_address(entry)),
    )


def _is_address(entry: str) -> bool:
    try:
        ipaddress.ip_network(entry, strict=False)
    except ValueError:
        return False
    return True


def _is_lost_stream(exc: BoxdError) -> bool:
    return isinstance(exc, APIConnectionError) or exc.code in _LOST_STREAM_CODES


class _BytesReader:
    """The small ``StreamReader`` subset consumed by ``Sandbox.exec_with_output``."""

    def __init__(self, value: bytes):
        self._value = value

    async def read(self) -> bytes:
        return self._value


class _CompletedProcess:
    """A finished boxd exec in the common process protocol."""

    def __init__(self, stdout: bytes, stderr: bytes, exit_code: int):
        self.stdout = _BytesReader(stdout)
        self.stderr = _BytesReader(stderr)
        self._exit_code = exit_code

    async def wait(self) -> int:
        return self._exit_code


class BoxdSandbox(VmSandbox):
    """A boxd machine with Docker; its client is its provider's, shared with the provider's other sandboxes."""

    type = "boxd"

    def __init__(
        self,
        client: Any,
        machine_id: str,
        *,
        tunnel_urls: dict[int, str],
        network_policy: NetworkPolicy | None,
    ):
        self._client = client
        self.sandbox_id = machine_id
        self.mode = SANDBOX_MODE_VM
        self.vnc_url = None
        self.tunnel_urls = tunnel_urls
        self.network_policy = network_policy

    async def terminate(self) -> None:
        await delete_machine(self._client, self.sandbox_id)

    async def exec(self, *command: str) -> _CompletedProcess:
        """Run argv to completion with no input, keeping its stream alive while it is quiet. A stream lost before the
        command's exit status is exit -1, which ``exec_script`` retries.

        Standard input is ``/dev/null``: boxd's end of input can miss a command when several of one connection's
        execs read it at once, and the command then waits on its input forever."""
        line = _KEEPALIVE.format(interval=KEEPALIVE_SEC, line=f"{shlex.join(command)} </dev/null")
        try:
            result = await self._client.machines.exec(self.sandbox_id, line, encoding="buffer")
        except BoxdError as exc:
            if not _is_lost_stream(exc):
                raise
            return _CompletedProcess(b"", f"{type(exc).__name__}: {exc}".encode(), _UNKNOWN_EXIT_CODE)
        return _CompletedProcess(result.stdout, result.stderr.replace(b"\0", b""), result.exit_code)

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        """An exec whose stream was cut (exit -1) returns no stdout, so a caller that skips the exit code can't mistake
        partial output for the whole."""
        process = await self.exec(*args)
        stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
        exit_code = await process.wait()
        kept = b"" if exit_code == _UNKNOWN_EXIT_CODE else stdout
        return exit_code, kept.decode(errors="replace"), stderr.decode(errors="replace")

    async def wait_for_vm(self) -> None:
        """Wait until exec reaches the machine and its Docker daemon answers, and check for Compose v2, which
        ``deploy_env`` needs."""
        await self._client.machines.wait_until_ready(self.sandbox_id, timeout=READY_TIMEOUT_SEC)
        try:
            await self.exec_script(DOCKER_READY, max_retries=2)
        except RuntimeError as exc:
            raise RuntimeError(f"Docker not ready on boxd machine {self.sandbox_id}: {exc}") from exc
        logger.info("Docker ready on boxd machine %s", self.sandbox_id)

    async def setup_vm_for_gateway(self, exposed_ports: list[int] | None = None) -> None:
        """Ready Docker; boxd routes each exposed port through its proxy, so no firewall."""
        await self.wait_for_vm()

    def _can_download(self, url: str | None) -> bool:
        """Whether the machine can fetch ``url`` itself; an unknown policy counts as no."""
        if url is None or self.network_policy is None:
            return False
        host = urlparse(url).hostname
        return host is not None and allows_host(self.network_policy, host)

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        """Curl a signed URL the allowlist admits; push anything else over exec, which needs no egress."""
        object_store = get_config().get_object_store_at(object_url)
        signed = await asyncio.to_thread(object_store.signed_get_url, object_url)
        if self._can_download(signed):
            await self.exec_script(f"curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(signed)} -o {shlex.quote(vm_path)}")
            return
        await self._write_unsigned_object(object_store, object_url, vm_path)

    async def _load_tarballs(self, artifacts: list) -> None:
        """Load image tarballs, pushing over exec those the machine's allowlist can't download."""
        signed_urls = await self._signed_image_urls(artifacts)
        reachable = [url if self._can_download(url) else None for url in signed_urls]
        await self._load_docker_images(artifacts, reachable)


async def delete_machine(client: Any, machine_id: str) -> None:
    """Delete a machine; one that is already gone is not an error."""
    try:
        await client.machines.delete(machine_id)
    except NotFoundError:
        logger.info("boxd machine %s was already gone at terminate", machine_id)
