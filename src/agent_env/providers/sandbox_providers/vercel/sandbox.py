"""AgentEnv adapter for an asynchronous Vercel Sandbox."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import shlex
import ssl
import uuid
import logging
import weakref
from ipaddress import ip_network
from typing import Any
from urllib.parse import urlparse

from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import (
    CURL_RETRY_FLAGS,
    NetworkMode,
    NetworkPolicy,
    VmSandbox,
)
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM

logger = logging.getLogger(__name__)

_STREAM_CHUNK_BYTES = 2 * 1024 * 1024
_DOCKER_SETUP_TIMEOUT = 300
_policy_locks: weakref.WeakValueDictionary[
    tuple[asyncio.AbstractEventLoop, str], asyncio.Lock
] = weakref.WeakValueDictionary()
_MISSING_FINAL_METADATA = "Sandbox process response is missing final metadata"
_MISSING_RETURN_CODE = (
    "Sandbox process response final metadata is missing a return code"
)


def network_policy_from_vercel(policy: Any) -> NetworkPolicy | None:
    """Translate a Vercel policy, or return None for one agent-env cannot represent."""
    mode = getattr(policy, "mode", None)
    if mode == "allow-all":
        return NetworkPolicy()
    if mode == "deny-all":
        return NetworkPolicy(mode=NetworkMode.ALLOWLIST)
    if mode != "custom":
        return None

    allow = getattr(policy, "allow", None)
    if not hasattr(allow, "items"):
        return None
    hosts = []
    for domain, rules in allow.items():
        if not isinstance(domain, str) or not domain or tuple(rules or ()):
            return None
        hosts.append(domain)

    subnets = getattr(policy, "subnets", None)
    cidrs: list[str] = []
    if subnets is not None:
        allowed = getattr(subnets, "allow", None)
        denied = getattr(subnets, "deny", None)
        if denied:
            return None
        if allowed is not None:
            for entry in allowed:
                if not isinstance(entry, str) or not entry:
                    return None
                try:
                    ip_network(entry, strict=False)
                except ValueError:
                    return None
                cidrs.append(entry)
    return NetworkPolicy(
        mode=NetworkMode.ALLOWLIST,
        allow_hosts=tuple(hosts),
        allow_cidrs=tuple(cidrs),
    )


def vercel_network_policy(policy: NetworkPolicy) -> Any:
    """Translate an agent-env policy into the pinned Vercel SDK models."""
    from vercel.sandbox import NetworkPolicy as VercelNetworkPolicy, NetworkPolicySubnets

    if policy.mode is NetworkMode.ALLOW_ALL:
        return VercelNetworkPolicy.allow_all()
    if not policy.allow_hosts and not policy.allow_cidrs:
        return VercelNetworkPolicy.deny_all()
    return VercelNetworkPolicy.custom(
        allow={host: () for host in policy.allow_hosts},
        subnets=NetworkPolicySubnets(allow=policy.allow_cidrs)
        if policy.allow_cidrs
        else None,
    )


def _is_missing_resource(error: BaseException) -> bool:
    return getattr(error, "status_code", None) == 404


def _is_lost_process_transport(error: BaseException) -> bool:
    if isinstance(error, (ConnectionError, ssl.SSLEOFError)):
        return True
    try:
        from anyio import BrokenResourceError, EndOfStream
        from httpx2 import ReadError, WriteError, RemoteProtocolError, ReadTimeout, WriteTimeout
        from vercel.sandbox import SandboxResponseError
    except ImportError:
        return False
    if isinstance(error, (BrokenResourceError, EndOfStream, ReadError, WriteError, RemoteProtocolError, ReadTimeout, WriteTimeout)):
        return True
    return isinstance(error, SandboxResponseError) and str(error) in {
        _MISSING_FINAL_METADATA,
        _MISSING_RETURN_CODE,
    }


def _returncode(result: Any) -> int:
    returncode = getattr(result, "returncode", None)
    if returncode is None:
        logger.warning("Vercel process result is missing a return code; mapping to exit -1")
        return -1
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise TypeError(f"Vercel process returned a malformed exit code: {returncode!r}")
    return returncode


class _OutputSink:
    def __init__(self, reader: asyncio.StreamReader):
        self.reader = reader

    def write(self, text: str) -> int:
        self.reader.feed_data(text.encode())
        return len(text)

    def flush(self) -> None:
        pass

    def writable(self) -> bool:
        return True


class _VercelProcess:
    """Route one native command log stream into AgentEnv's two byte readers."""

    def __init__(self, operation: Any, tasks: set[asyncio.Task]):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()

        async def run() -> int:
            try:
                result = await operation(stdout=_OutputSink(self.stdout), stderr=_OutputSink(self.stderr))
                return _returncode(result)
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                if _is_lost_process_transport(error):
                    logger.warning(
                        "Vercel command transport ended without a final process result; "
                        "mapping to exit -1: %s",
                        error,
                    )
                    return -1
                self.stdout.set_exception(error)
                self.stderr.set_exception(error)
                raise
            finally:
                self.stdout.feed_eof()
                self.stderr.feed_eof()

        self._task = asyncio.create_task(run())
        tasks.add(self._task)
        self._task.add_done_callback(tasks.discard)
        self._task.add_done_callback(lambda task: None if task.cancelled() else task.exception())

    async def wait(self) -> int:
        return await asyncio.shield(self._task)


class VercelSandbox(VmSandbox):
    type = "vercel"

    def __init__(
        self,
        sandbox: Any,
        *,
        client: Any,
        tunnel_urls: dict[int, str],
        network_policy: NetworkPolicy | None,
    ):
        self._sandbox = sandbox
        self._client = client
        self._processes: set[asyncio.Task] = set()
        self.sandbox_id = sandbox.name
        self.tunnel_urls = tunnel_urls
        self.vnc_url = None
        self.mode = SANDBOX_MODE_VM
        self.network_policy = network_policy

    async def terminate(self) -> None:
        try:
            await self._sandbox.destroy(delete_orphan_snapshots=True)
        except BaseException as error:
            if not _is_missing_resource(error):
                raise
            logger.info("Vercel sandbox %s was already gone at terminate", self.sandbox_id)

    async def exec(self, *command: str) -> _VercelProcess:
        return await self._run(*command)

    async def _run(self, *command: str, timeout: float | None = None) -> _VercelProcess:
        argv = list(command)
        sudo = argv[:1] == ["sudo"]
        if sudo:
            argv = argv[1:]
        if not argv:
            raise ValueError("a Vercel exec requires a command")
        async def run(**output: Any) -> Any:
            return await self._sandbox.run_process(
                argv[0], argv[1:], sudo=sudo, kill_after=timeout, **output
            )

        return _VercelProcess(run, self._processes)

    async def _exec_with_output(
        self, *command: str, timeout: float | None = None
    ) -> tuple[int, str, str]:
        process = await self._run(*command, timeout=timeout)
        stdout, stderr = await asyncio.gather(
            process.stdout.read(),
            process.stderr.read(),
        )
        return await process.wait(), stdout.decode(errors="replace"), stderr.decode(errors="replace")

    async def wait_for_vm(self) -> None:
        script = """
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  exit 0
fi
if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
if ! command -v apt-get >/dev/null 2>&1; then
  echo 'Vercel image has neither Docker Compose v2 nor apt-get; use a prepared Docker-capable image' >&2
  exit 2
fi
DEBIAN_FRONTEND=noninteractive apt-get update -qq &&
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io docker-compose-v2 ||
  { echo 'Docker installation failed; use a prepared Docker-capable image for restricted egress' >&2; exit 3; }
fi
docker info >/dev/null 2>&1 || (nohup dockerd >/var/log/agentenv-dockerd.log 2>&1 </dev/null &)
for i in $(seq 1 30); do
  docker info >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 && exit 0
  sleep 1
done
tail -40 /var/log/agentenv-dockerd.log >&2
exit 1
"""
        exit_code, _, stderr = await self._exec_with_output(
            "sudo", "bash", "-c", script, timeout=_DOCKER_SETUP_TIMEOUT
        )
        if exit_code != 0:
            raise RuntimeError(
                f"Docker and Compose are unavailable in Vercel sandbox {self.sandbox_id} "
                f"(exit {exit_code}): {stderr[-1000:]}"
            )
        logger.info("Docker and Compose ready in Vercel sandbox %s", self.sandbox_id)

    async def setup_vm_for_gateway(self, exposed_ports: list[int] | None = None) -> None:
        await self.wait_for_vm()

    async def _write_bytes_to_vm_path(self, data: bytes, vm_path: str) -> None:
        if len(data) <= _STREAM_CHUNK_BYTES:
            await self._sandbox.fs.write_bytes(vm_path, data)
        else:
            await self._write_stream_to_vm(io.BytesIO(data), vm_path)

    async def _write_unsigned_object(self, object_store: Any, object_url: str, vm_path: str) -> None:
        with contextlib.closing(await asyncio.to_thread(object_store.open, object_url)) as source:
            await self._write_stream_to_vm(source, vm_path)

    async def _write_stream_to_vm(self, source: Any, vm_path: str) -> None:
        """Assemble bounded SDK uploads and verify contents before replacing the destination."""
        staged = f"/tmp/agentenv-upload-{uuid.uuid4().hex}"
        chunk_path = staged + ".part"
        digest = hashlib.sha256()
        try:
            await self.exec_script(f": > {shlex.quote(staged)}")
            while chunk := await asyncio.to_thread(source.read, _STREAM_CHUNK_BYTES):
                digest.update(chunk)
                for attempt in range(3):
                    try:
                        await self._sandbox.fs.write_bytes(chunk_path, chunk)
                        break
                    except Exception:
                        # SDK transport failures have several types; only the replaceable part is retried.
                        if attempt == 2:
                            raise
                        await asyncio.sleep(0.25 * 2**attempt)
                await self.exec_script(f"cat {shlex.quote(chunk_path)} >> {shlex.quote(staged)}")
            await self.exec_script(
                f"printf '%s  %s\n' {digest.hexdigest()} {shlex.quote(staged)} | sha256sum -c - >/dev/null "
                f"&& mv -- {shlex.quote(staged)} {shlex.quote(vm_path)}"
            )
        finally:
            await self._remove_vm_temp_file(staged, chunk_path)

    def _policy_lock(self) -> asyncio.Lock:
        return _policy_locks.setdefault(
            (asyncio.get_running_loop(), self.sandbox_id), asyncio.Lock()
        )

    async def apply_network_policy(self, policy: NetworkPolicy) -> None:
        async with self._policy_lock():
            await self._sandbox.update_network_policy(vercel_network_policy(policy))
            self.network_policy = policy

    async def _allow_download_hosts(self, urls: list[str | None]) -> None:
        hosts = {parsed.hostname for url in urls if url and (parsed := urlparse(url)).hostname}
        async with self._policy_lock():
            current = await self._client.get_sandbox(name=self.sandbox_id)
            self.network_policy = network_policy_from_vercel(current.network_policy)
            policy = self.network_policy
            if policy is None:
                raise RuntimeError(
                    f"Cannot download artifacts in Vercel sandbox {self.sandbox_id}: "
                    "its applied network policy is unknown, so signed download hosts cannot be added safely"
                )
            if not policy.restricts_egress or not hosts:
                return
            widened = policy.with_hosts(sorted(hosts))
            if widened != policy:
                await self._sandbox.update_network_policy(vercel_network_policy(widened))
                self.network_policy = widened

    async def _load_tarballs(self, artifacts: list) -> None:
        signed_urls = await self._signed_image_urls(artifacts)
        await self._allow_download_hosts(signed_urls)
        await self._load_docker_images(artifacts, signed_urls)

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        object_store = get_config().get_object_store_at(object_url)
        signed = await asyncio.to_thread(object_store.signed_get_url, object_url)
        if signed is None:
            await self._write_unsigned_object(object_store, object_url, vm_path)
            return
        await self._allow_download_hosts([signed])
        await self.exec_script(
            f"curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(signed)} -o {shlex.quote(vm_path)}"
        )
