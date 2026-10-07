"""A Sailbox (a Sail Research Linux VM with Docker) presented through the :class:`VmSandbox` contract."""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
import uuid
import weakref
from ipaddress import ip_network
from typing import Any, Optional
from urllib.parse import urlparse

from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sail.model_key import DOCKER_SHIM_PATH, ModelKeyInjection, docker_shim
from agent_env.providers.sandbox_providers.sandbox import CURL_RETRY_FLAGS, NetworkMode, NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM

logger = logging.getLogger(__name__)

#: Sail's limit on entries in one egress allowlist.
MAX_ALLOWLIST_ENTRIES = 128

# One lock per Sailbox and event loop, shared by every handle to it while an update is in flight.
_policy_locks: weakref.WeakValueDictionary[tuple[asyncio.AbstractEventLoop, str], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)


def egress_document(policy: NetworkPolicy, injection: ModelKeyInjection | None = None) -> dict[str, Any]:
    """``policy`` as a Sail egress-policy document (``{}`` allows everything), with ``injection``'s rules."""
    document: dict[str, Any] = {}
    if policy.mode is NetworkMode.ALLOWLIST:
        document["allowlist"] = [*policy.allow_hosts, *policy.allow_cidrs]
    if injection is not None:
        document["rules"] = injection.rules()
    return document


def policy_from_document(document: Any, injection: ModelKeyInjection | None = None) -> NetworkPolicy | None:
    """The agent-env policy a Sail egress document applies, or None when it can't be represented. Rules are
    representable only as ``injection``'s, which add headers and leave egress alone."""
    if not isinstance(document, dict):
        return None
    document = dict(document)
    rules = document.pop("rules", None)
    if rules is not None and (injection is None or rules != injection.rules()):
        return None
    if not document:
        return NetworkPolicy()
    entries = document.get("allowlist")
    if set(document) != {"allowlist"} or not isinstance(entries, list) or not all(isinstance(e, str) and e for e in entries):
        return None
    hosts, cidrs = [], []
    for entry in entries:
        try:
            ip_network(entry, strict=False)
        except ValueError:
            hosts.append(entry)
        else:
            cidrs.append(entry)
    return NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=tuple(hosts), allow_cidrs=tuple(cidrs))


def _allows_host(policy: NetworkPolicy, host: str) -> bool:
    """Whether an allowlist entry already admits ``host``: an exact match, or a ``*.domain`` it sits under."""
    return any(entry == host or (entry.startswith("*.") and host.endswith(entry[1:])) for entry in policy.allow_hosts)


async def delete_saved_policy(sdk: Any, policy_id: str | None) -> None:
    """Delete a saved egress policy, best effort: a leftover one names a secret but holds no value."""
    if policy_id is None:
        return
    try:
        await (await sdk.EgressPolicy.get.aio(policy_id)).delete.aio()
    except sdk.NotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 - reported, never masks the caller's outcome
        logger.warning("Could not delete egress policy %s: %s", policy_id, exc)


class _BytesReader:
    def __init__(self, value: bytes):
        self._value = value

    async def read(self) -> bytes:
        return self._value


class _CompletedProcess:
    """An exec that failed before it produced output, in the process shape ``VmSandbox`` reads."""

    def __init__(self, stdout: bytes, stderr: bytes, exit_code: int):
        self.stdout = _BytesReader(stdout)
        self.stderr = _BytesReader(stderr)
        self._exit_code = exit_code

    async def wait(self) -> int:
        return self._exit_code


class _Stream:
    """One output stream, pumped from the moment the exec starts so no byte is dropped: iterate it to
    stream (``collect_artifacts`` does), or ``read()`` it whole. The queue is bounded, so a slow reader
    pauses the command rather than growing memory; a stream nobody claims is drained by ``wait()``. A
    transient failure just ends the stream, leaving ``wait()`` to report exit -1 for ``exec_script`` to
    retry; any other is raised to the reader."""

    _MAX_CHUNKS = 64

    def __init__(self, chunks, transient: tuple[type[BaseException], ...]):
        self._transient = transient
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=self._MAX_CHUNKS)
        self.claimed = False
        self.error: BaseException | None = None
        self.pump = asyncio.ensure_future(self._pump(chunks))

    async def _pump(self, chunks) -> None:
        try:
            async for chunk in chunks:
                await self._queue.put(chunk)
        except Exception as exc:  # noqa: BLE001 - re-raised to the reader and by wait()
            self.error = exc
        finally:
            await self._queue.put(None)

    async def _chunks(self):
        while (chunk := await self._queue.get()) is not None:
            yield chunk
        if self.error is not None and not isinstance(self.error, self._transient):
            raise self.error

    def __aiter__(self):
        self.claimed = True
        return self._chunks()

    async def read(self) -> bytes:
        return b"".join([chunk async for chunk in self])

    async def drain(self) -> None:
        if not self.claimed:
            self.claimed = True
            async for _ in self._chunks():
                pass


class _SailProcess:
    """A running Sail exec; a lost host or transport while it runs is exit -1, which ``exec_script`` retries."""

    def __init__(self, process: Any, sdk: Any):
        self._process = process
        self._transient = (sdk.SailboxHostLostError, sdk.TransportError)
        self.stdout = _Stream(process.stdout_bytes, self._transient)
        self.stderr = _Stream(process.stderr_bytes, self._transient)

    async def wait(self) -> int:
        await asyncio.gather(self.stdout.drain(), self.stderr.drain(), return_exceptions=True)
        await asyncio.gather(self.stdout.pump, self.stderr.pump)
        try:
            exit_code = int((await self._process.wait()).exit_code)
        except self._transient:
            return -1
        for error in (self.stdout.error, self.stderr.error):
            if isinstance(error, self._transient):
                return -1
            if error is not None:
                raise error
        return exit_code


class SailSandbox(VmSandbox):
    """A Sailbox from the Docker-capable devbox image; commands run as root. With a model-key ``injection``,
    the key's value never enters the Sailbox: every command and file is scrubbed of it."""

    type = "sail"
    _DOCKER_PROBE_TIMEOUT = 10

    def __init__(
        self, sailbox: Any, *, sdk: Any, tunnel_urls: dict[int, str], network_policy: NetworkPolicy | None,
        injection: ModelKeyInjection | None = None,
    ):
        self._sailbox = sailbox
        self._sdk = sdk
        self._injection = injection
        self.sandbox_id = sailbox.sailbox_id
        self.tunnel_urls = tunnel_urls
        self.vnc_url = None
        self.mode = SANDBOX_MODE_VM
        self.network_policy = network_policy

    async def terminate(self) -> None:
        """Terminate the Sailbox, then delete its model-key policy and, unless another Sailbox's policy
        still names it, the key's secret."""
        try:
            await self._sailbox.terminate.aio()
        except self._sdk.NotFoundError:
            logger.info("Sailbox %s was already gone at terminate", self.sandbox_id)
        if self._injection is not None:
            await self._delete_policy(self._injection.policy_id)
            await self._delete_secret(self._injection.secret)

    async def _delete_policy(self, policy_id: str | None) -> None:
        await delete_saved_policy(self._sdk, policy_id)

    async def _delete_secret(self, name: str) -> None:
        try:
            await (await self._sdk.Secret.get.aio(name)).delete.aio()
        except self._sdk.SecretInUseError:
            logger.info("Model-key secret %s is still used by another Sailbox; leaving it", name)
        except self._sdk.NotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001 - the next Sailbox using this key re-sets it; report and go on
            logger.warning("Could not delete model-key secret %s: %s", name, exc)

    async def install_container_trust(self) -> None:
        """Have every container started on this Sailbox trust the CA Sail injects the model key behind."""
        await self._sailbox.fs.write.aio(DOCKER_SHIM_PATH, docker_shim(), mode=0o755)

    async def exec(self, *command: str) -> _SailProcess | _CompletedProcess:
        """Run argv to completion. A leading ``sudo`` is dropped (commands already run as root, and the
        guest's hostname doesn't resolve, so sudo warns on every call); a lost host or transport maps to
        exit -1, which ``exec_script`` retries."""
        return await self._run(*command)

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        """``(exit_code, stdout, stderr)``; a command whose output stream was cut (exit -1) returns no stdout,
        so a caller that skips the exit code can't mistake partial output for the whole."""
        process = await self.exec(*args)
        stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
        exit_code = await process.wait()
        return exit_code, "" if exit_code == -1 else stdout.decode(), stderr.decode()

    async def _run(self, *command: str, timeout: Optional[int] = None) -> _SailProcess | _CompletedProcess:
        argv = list(command[1:] if command[:1] == ("sudo",) else command)
        if self._injection is not None:
            argv = [self._injection.scrub(arg) for arg in argv]
        try:
            process = await self._sailbox.exec.aio(
                argv, timeout=timeout, output_mode="pipe", idempotency_key=uuid.uuid4().hex,
            )
        except (self._sdk.SailboxHostLostError, self._sdk.TransportError) as exc:
            return _CompletedProcess(b"", f"{type(exc).__name__}: {exc}".encode(), -1)
        return _SailProcess(process, self._sdk)

    async def _exec_with_output(self, *command: str, timeout: int) -> tuple[int, str, str]:
        process = await self._run(*command, timeout=timeout)
        stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
        return await process.wait(), stdout.decode(errors="replace"), stderr.decode(errors="replace")

    async def wait_for_vm(self) -> None:
        """Poll ``docker info`` until the daemon answers, starting dockerd once if it is not running."""
        deadline = time.monotonic() + self._VM_READY_TIMEOUT
        started = False
        attempts = 0
        detail = ""
        while True:
            attempts += 1
            exit_code, _, stderr = await self._exec_with_output(
                "docker", "info", "--format", "{{.ServerVersion}}", timeout=self._DOCKER_PROBE_TIMEOUT,
            )
            if exit_code == 0:
                logger.info("Docker ready in Sailbox %s after %s probe(s)", self.sandbox_id, attempts)
                return
            detail = stderr.strip()
            if time.monotonic() >= deadline:
                break
            if not started:
                await self.exec_script("pgrep -x dockerd > /dev/null || (nohup dockerd > /var/log/dockerd.log 2>&1 &)")
                started = True
            await asyncio.sleep(self._VM_READY_POLL_INTERVAL)
        raise RuntimeError(
            f"Docker not ready in Sailbox {self.sandbox_id} after {self._VM_READY_TIMEOUT}s ({attempts} probes): {detail[-500:]}"
        )

    async def setup_vm_for_gateway(self, exposed_ports: Optional[list[int]] = None) -> None:
        """Wait for Docker and require Compose v2. Sail routes exposed ports itself, so no firewall rules."""
        await self.wait_for_vm()
        exit_code, stdout, stderr = await self.exec_with_output("docker", "compose", "version")
        if exit_code != 0:
            raise RuntimeError(f"Sailbox {self.sandbox_id} has no Docker Compose v2: {(stderr or stdout).strip()}")

    async def _write_bytes_to_vm_path(self, data: bytes, vm_path: str) -> None:
        if self._injection is not None:
            data = self._injection.scrub_bytes(data)
        await self._sailbox.fs.write.aio(vm_path, data)

    async def apply_network_policy(self, policy: NetworkPolicy) -> None:
        """Replace the Sailbox's egress policy; applies to new connections. A model-key injection needs a
        saved policy (only those can name a secret), so a new one replaces the old, which is deleted."""
        if self._injection is None:
            await self._sailbox.set_egress_policy.aio(egress_document(policy))
        else:
            saved = await self._sdk.EgressPolicy.create.aio(f"agentenv-{uuid.uuid4().hex}", egress_document(policy, self._injection))
            await self._sailbox.set_egress_policy.aio(saved)
            previous, self._injection.policy_id = self._injection.policy_id, saved.id
            await self._delete_policy(previous)
        self.network_policy = policy

    def adopt_applied_policy(self, applied: Any) -> NetworkPolicy | None:
        """The agent-env policy Sail reports applied (``Sailbox.egress_policy``), or None when it can't be
        represented: a saved policy counts only when it carries this Sailbox's model-key injection."""
        if applied is None:
            return None
        document = getattr(applied, "document", None)
        if getattr(applied, "policy_id", None) is not None:
            saved = ModelKeyInjection.from_document(document, applied.policy_id)
            if self._injection is None or not self._injection.matches(saved):
                return None
            self._injection.policy_id = applied.policy_id
        return policy_from_document(document, self._injection)

    def _known_policy(self, purpose: str) -> NetworkPolicy:
        if self.network_policy is None:
            raise RuntimeError(
                f"Cannot {purpose} in Sailbox {self.sandbox_id}: its applied egress policy is unknown, "
                "so the signed download hosts cannot be added safely"
            )
        return self.network_policy

    async def _allow_download_hosts(self, urls: list[str | None], purpose: str) -> None:
        """Add the hosts of signed download ``urls`` to the Sailbox's applied egress policy. Hosts the cached
        policy already admits need nothing (it only ever lags the applied one); a new host re-reads the
        applied policy under a per-Sailbox lock, so concurrent downloads don't drop each other's hosts.
        Refuses before exceeding Sail's limit."""
        hosts = {parsed.hostname for url in urls if url and (parsed := urlparse(url)).hostname}
        cached = self._known_policy(purpose)
        if not cached.restricts_egress or all(_allows_host(cached, host) for host in hosts):
            return
        lock = _policy_locks.setdefault((asyncio.get_running_loop(), self.sandbox_id), asyncio.Lock())
        async with lock:
            self.network_policy = self.adopt_applied_policy((await self._sdk.Sailbox.get.aio(self.sandbox_id)).egress_policy)
            policy = self._known_policy(purpose)
            if not policy.restricts_egress:
                return
            missing = sorted(host for host in hosts if not _allows_host(policy, host))
            if not missing:
                return
            if len(policy.allow_hosts) + len(policy.allow_cidrs) + len(missing) > MAX_ALLOWLIST_ENTRIES:
                raise RuntimeError(
                    f"Cannot {purpose} in Sailbox {self.sandbox_id}: adding {missing} would exceed "
                    f"Sail's {MAX_ALLOWLIST_ENTRIES}-entry egress allowlist"
                )
            await self.apply_network_policy(policy.with_hosts(missing))

    async def load_docker_images(self, artifacts: list) -> None:
        """Load images, first adding their signed-download hosts to a restrictive policy."""
        if not artifacts:
            return
        self._known_policy("load Docker images")
        signed_urls = await self._signed_image_urls(artifacts)
        await self._allow_download_hosts(signed_urls, "load Docker images")
        await self._load_docker_images(artifacts, signed_urls)

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        """Download through a signed URL whose host a restrictive policy now allows, or stream the bytes
        through Sail's filesystem API when the store cannot sign one."""
        object_store = get_config().get_object_store()
        signed = await asyncio.to_thread(object_store.signed_get_url, object_url)
        if signed is None:
            await self._write_unsigned_object(object_store, object_url, vm_path)
            return
        await self._allow_download_hosts([signed], "download an object")
        await self.exec_script(f"curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(signed)} -o {shlex.quote(vm_path)}")
