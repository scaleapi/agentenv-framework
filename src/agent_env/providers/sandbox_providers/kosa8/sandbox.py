"""A kosa8 Cloud sandbox presented through the :class:`VmSandbox` contract.

Each sandbox is a microVM booted with ``docker: true`` from kosa8's universal Ubuntu 24.04 image, which ships dockerd
and the Compose v2 plugin, so the gateway's docker-compose, an agent's ``docker run`` and image loading run under Docker
inside it. Commands run as root, so the ``sudo`` agent-env prefixes is dropped. kosa8 applies the egress policy on the
host, outside the VM, so one policy covers ``dockerd`` and every container it runs.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import shlex
import uuid
import weakref
from typing import Any
from urllib.parse import urlparse

import httpx

from agent_env.config import get_config
from agent_env.providers.sandbox_providers.sandbox import CURL_RETRY_FLAGS, NetworkMode, NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM, SandboxNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.kosa8.com"
# dockerd is installed in the image but not started at boot; the sandbox is persistent, so a daemon started by one
# command keeps running after it returns.
DOCKER_READY = r"""
if ! docker info >/dev/null 2>&1; then
  setsid nohup dockerd >/var/log/dockerd.log 2>&1 </dev/null &
fi
for _ in $(seq 90); do
  docker info >/dev/null 2>&1 && exec docker compose version
  sleep 1
done
echo "docker daemon did not become ready" >&2
tail -n 20 /var/log/dockerd.log >&2
exit 1
"""
# What exec returns when no exit status arrives; exec_script retries only this.
UNKNOWN_EXIT_CODE = -1
# kosa8 takes at most 256 argv entries of at most 8 KiB each. A command past either limit is written to a file through
# the files API and read back as argv in the sandbox.
MAX_ARGS = 256
MAX_ARG_BYTES = 8000
# The files API reaches /workspace only, and takes at most 8 MiB a request.
STAGING_DIR = "/workspace/.agentenv"
FILE_PIECE_BYTES = 4 * 1024 * 1024
_NO_INPUT = 'exec "$@" </dev/null'
_FROM_FILE = 'mapfile -d "" -t a < "$1" && rm -f -- "$1" && exec "${a[@]}" </dev/null'
# Answers that say nothing was done: a sandbox still starting, a rate limit, a host that refused before starting.
RESENDABLE = frozenset({409, 429, 503})
# Answers after which a call that changes nothing, or sets a value, is sent again.
IDEMPOTENT_RESENDABLE = frozenset({409, 429, 500, 502, 503, 504})
RESEND_DELAYS = (1, 2, 4, 8, 16)
# A streamed exec sends a heartbeat frame every 15 s while the command prints nothing.
_TIMEOUT = httpx.Timeout(30.0, read=120.0)


class Kosa8Error(RuntimeError):
    """A kosa8 API call that failed, with the status and error code it answered."""

    def __init__(self, response: httpx.Response):
        self.status_code = response.status_code
        self.code, message = _error_of(response)
        super().__init__(f"kosa8 {response.request.method} {response.request.url.path} answered "
                         f"{response.status_code} {self.code}: {message}")


def _error_of(response: httpx.Response) -> tuple[str, str]:
    """``(code, message)`` of kosa8's ``{"error": "<code>", "message": "..."}`` error answer."""
    try:
        body = response.json()
    except ValueError:
        return "unknown", response.text[:500]
    if not isinstance(body, dict):
        return "unknown", str(body)[:500]
    return str(body.get("error", "unknown")), str(body.get("message", ""))


class Kosa8Api:
    """The kosa8 Cloud calls a sandbox makes.

    The key comes from the provider's resolved config and is sent only to ``base_url``. ``transport`` is injectable so
    unit tests never open a connection."""

    def __init__(self, api_key: str, *, base_url: str = DEFAULT_BASE_URL,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        # One client per event loop: a client's connections belong to the loop that opened them, and the CLI runs
        # several loops in one process.
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
            weakref.WeakKeyDictionary())

    def _client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            client = self._clients[loop] = httpx.AsyncClient(
                base_url=self._base_url, transport=self._transport, timeout=_TIMEOUT,
                headers={"Authorization": f"Bearer {self._api_key}", "User-Agent": "agentenv-framework"},
            )
        return client

    async def close(self) -> None:
        client = self._clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.aclose()

    async def _call(self, method: str, path: str, *, resend: frozenset[int] = IDEMPOTENT_RESENDABLE,
                    resend_lost: bool = True, **kwargs: Any) -> httpx.Response:
        """One call, sent again after an answer in ``resend`` or, with ``resend_lost``, a lost connection."""
        delays = iter(RESEND_DELAYS)
        while True:
            try:
                response = await self._client().request(method, path, **kwargs)
            except httpx.TransportError:
                if not resend_lost or (delay := next(delays, None)) is None:
                    raise
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in resend or (delay := next(delays, None)) is None:
                    raise Kosa8Error(response)
            logger.debug("kosa8 %s %s got no usable answer; sending it again in %ss", method, path, delay)
            await asyncio.sleep(delay)

    async def create_sandbox(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create a sandbox. Only answers that say nothing was made are resent; a lost answer is raised, and the
        caller finds what the create may have made by its name."""
        response = await self._call("POST", "/v1/sandboxes", json=body, resend=RESENDABLE, resend_lost=False)
        return response.json()["sandbox"]

    async def get_sandbox(self, sandbox_id: str) -> dict[str, Any]:
        try:
            sandbox = (await self._call("GET", f"/v1/sandboxes/{sandbox_id}")).json()["sandbox"]
        except Kosa8Error as exc:
            if exc.status_code == 404:
                raise SandboxNotFoundError(f"kosa8 sandbox {sandbox_id!r} was not found") from exc
            raise
        if sandbox.get("state") == "deleted":
            raise SandboxNotFoundError(f"kosa8 sandbox {sandbox_id!r} was deleted")
        return sandbox

    async def sandboxes_named(self, name: str) -> list[dict[str, Any]]:
        response = await self._call("GET", "/v1/sandboxes", params={"name": name})
        return [sandbox for sandbox in response.json()["sandboxes"] if sandbox.get("state") != "deleted"]

    async def delete_sandbox(self, sandbox_id: str) -> None:
        """Delete a sandbox; one that is already gone is not an error."""
        try:
            await self._call("DELETE", f"/v1/sandboxes/{sandbox_id}")
        except Kosa8Error as exc:
            if exc.status_code != 404:
                raise
            logger.info("kosa8 sandbox %s was already gone at terminate", sandbox_id)

    async def enforced_egress(self, sandbox_id: str) -> str:
        """The policy kosa8 enforces on the sandbox, compiled: ``any``, ``deny``, ``allow:…[;deny:…]`` or
        ``block:…``. It is the org's default when the sandbox has none of its own."""
        response = await self._call("GET", f"/v1/sandboxes/{sandbox_id}/egress")
        return str(response.json()["egress"]["enforced"])

    async def open_port(self, sandbox_id: str, port: int) -> str:
        """The public preview URL of ``port``; opening one that is open answers with it again."""
        response = await self._call("POST", f"/v1/sandboxes/{sandbox_id}/previews", json={"port": port, "public": True})
        return _origin(response.json()["preview"]["url"])

    async def open_ports(self, sandbox_id: str) -> dict[int, str]:
        response = await self._call("GET", f"/v1/sandboxes/{sandbox_id}/previews")
        return {int(preview["port"]): _origin(preview["url"]) for preview in response.json()["previews"]
                if preview.get("public")}

    async def make_dir(self, sandbox_id: str, path: str) -> None:
        await self._call("POST", f"/v1/sandboxes/{sandbox_id}/files/mkdir", json={"path": path})

    async def write_file(self, sandbox_id: str, path: str, data: bytes) -> None:
        await self._call("PUT", f"/v1/sandboxes/{sandbox_id}/files", params={"path": path}, content=data,
                         headers={"Content-Type": "application/octet-stream"})

    async def exec(self, sandbox_id: str, argv: list[str]) -> tuple[int, bytes, bytes]:
        """Run ``argv`` streamed, which no time limit holds, and return ``(exit, stdout, stderr)``. A call refused
        before the command started is resent; one whose stream ends without an exit status is exit -1."""
        delays = iter(RESEND_DELAYS)
        while True:
            try:
                async with self._client().stream("POST", f"/v1/sandboxes/{sandbox_id}/exec",
                                                 json={"cmd": argv, "stream": True}) as response:
                    if response.status_code == 200:
                        return await _frames(response)
                    await response.aread()
                    error = Kosa8Error(response)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            except httpx.TransportError as exc:
                return UNKNOWN_EXIT_CODE, b"", f"{type(exc).__name__}: {exc}".encode()
            else:
                if error.status_code not in RESENDABLE:
                    if error.status_code >= 500:
                        return UNKNOWN_EXIT_CODE, b"", str(error).encode()
                    raise error
                failure = str(error)
            delay = next(delays, None)
            if delay is None:
                return UNKNOWN_EXIT_CODE, b"", failure.encode()
            logger.warning("kosa8 exec in %s was refused before it started (%s); resending in %ss",
                           sandbox_id, failure, delay)
            await asyncio.sleep(delay)


async def _frames(response: httpx.Response) -> tuple[int, bytes, bytes]:
    """Read a streamed exec's NDJSON frames: ``{stream, data}`` output, heartbeats, then ``{done, exit}`` or
    ``{done, error}``."""
    stdout, stderr = bytearray(), bytearray()
    async for line in response.aiter_lines():
        if not line.strip():
            continue
        frame = json.loads(line)
        if frame.get("done"):
            if isinstance(frame.get("exit"), int):
                return frame["exit"], bytes(stdout), bytes(stderr)
            stderr += f"kosa8: {frame.get('error') or 'the command ended without an exit status'}".encode()
            return UNKNOWN_EXIT_CODE, bytes(stdout), bytes(stderr)
        if frame.get("data"):
            (stdout if frame.get("stream") == "stdout" else stderr).extend(base64.b64decode(frame["data"]))
    stderr += b"kosa8: the stream ended before the command's exit status"
    return UNKNOWN_EXIT_CODE, bytes(stdout), bytes(stderr)


def _origin(url: str) -> str:
    return url.rstrip("/")


def policy_from_enforced(enforced: str) -> NetworkPolicy:
    """The egress policy a compiled kosa8 policy applies. ``block:…`` lets everything out but a deny list, which an
    allowlist can't express, so it reads as allow-all; ``denied_entries`` keeps the denies."""
    if enforced == "deny":
        return NetworkPolicy(mode=NetworkMode.ALLOWLIST)
    if enforced.startswith("allow:"):
        entries = [entry for entry in enforced.removeprefix("allow:").split(";deny:")[0].split(",") if entry]
        return NetworkPolicy(
            mode=NetworkMode.ALLOWLIST,
            allow_hosts=tuple(entry for entry in entries if not _is_address(entry)),
            allow_cidrs=tuple(entry for entry in entries if _is_address(entry)),
        )
    return NetworkPolicy()


def denied_entries(enforced: str) -> tuple[str, ...]:
    """What a compiled kosa8 policy denies: a ``block:…`` list, or the ``;deny:…`` after an allow list. A
    ``NetworkPolicy`` can't carry either, so the sandbox keeps them beside it."""
    if enforced.startswith("block:"):
        listed = enforced.removeprefix("block:")
    elif ";deny:" in enforced:
        listed = enforced.split(";deny:", 1)[1]
    else:
        return ()
    return tuple(entry for entry in listed.split(",") if entry)


def allows_host(policy: NetworkPolicy, host: str) -> bool:
    """Whether ``policy`` lets the sandbox reach ``host`` by kosa8's rules."""
    return not policy.restricts_egress or matches_host((*policy.allow_hosts, *policy.allow_cidrs), host)


def matches_host(entries: tuple[str, ...], host: str) -> bool:
    """Whether a kosa8 entry names ``host``: an exact name, a ``*.domain`` entry, which covers every name under
    ``domain`` but not ``domain`` itself, or an address in a listed address or range."""
    host = host.lower().rstrip(".")
    for entry in entries:
        entry = entry.lower().rstrip(".")
        if entry == host or (entry.startswith("*.") and host.endswith(entry[1:])):
            return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in ipaddress.ip_network(entry, strict=False) for entry in entries if _is_address(entry))


def _is_address(entry: str) -> bool:
    try:
        ipaddress.ip_network(entry, strict=False)
    except ValueError:
        return False
    return True


class _BytesReader:
    """The small ``StreamReader`` subset consumed by ``Sandbox.exec_with_output``."""

    def __init__(self, value: bytes):
        self._value = value

    async def read(self) -> bytes:
        return self._value


class _CompletedProcess:
    """A finished kosa8 exec in the common process protocol."""

    def __init__(self, stdout: bytes, stderr: bytes, exit_code: int):
        self.stdout = _BytesReader(stdout)
        self.stderr = _BytesReader(stderr)
        self._exit_code = exit_code

    async def wait(self) -> int:
        return self._exit_code


class Kosa8Sandbox(VmSandbox):
    """A kosa8 microVM with Docker; its API is its provider's, shared with the provider's other sandboxes."""

    type = "kosa8"

    def __init__(
        self,
        api: Kosa8Api,
        sandbox_id: str,
        *,
        tunnel_urls: dict[int, str],
        network_policy: NetworkPolicy | None,
    ):
        self._api = api
        self.sandbox_id = sandbox_id
        self.mode = SANDBOX_MODE_VM
        self.vnc_url = None
        self.tunnel_urls = tunnel_urls
        self.network_policy = network_policy
        self.denied_hosts: tuple[str, ...] = ()
        self._staging_ready = False

    def enforce(self, enforced: str) -> None:
        """Take the policy kosa8 enforces, compiled: what it allows as ``network_policy``, and what it denies."""
        self.network_policy = policy_from_enforced(enforced)
        self.denied_hosts = denied_entries(enforced)

    async def terminate(self) -> None:
        await self._api.delete_sandbox(self.sandbox_id)

    async def exec(self, *command: str) -> _CompletedProcess:
        """Run argv to completion as root, with no input. A command past kosa8's argv limits goes through a file. A
        stream lost before the command's exit status is exit -1, which ``exec_script`` retries."""
        argv = list(command[1:] if command[:1] == ("sudo",) else command)
        if len(argv) + 4 <= MAX_ARGS and all(len(arg.encode()) <= MAX_ARG_BYTES for arg in argv):
            wrapped = ["bash", "-c", _NO_INPUT, "kosa8-exec", *argv]
        else:
            path = f"{STAGING_DIR}/argv-{uuid.uuid4().hex}"
            await self._upload(path, b"".join(arg.encode() + b"\0" for arg in argv))
            wrapped = ["bash", "-c", _FROM_FILE, "kosa8-exec", path]
        exit_code, stdout, stderr = await self._api.exec(self.sandbox_id, wrapped)
        return _CompletedProcess(stdout, stderr, exit_code)

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        """An exec whose stream was cut (exit -1) returns no stdout, so a caller that skips the exit code can't mistake
        partial output for the whole."""
        process = await self.exec(*args)
        stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
        exit_code = await process.wait()
        kept = b"" if exit_code == UNKNOWN_EXIT_CODE else stdout
        return exit_code, kept.decode(errors="replace"), stderr.decode(errors="replace")

    async def _upload(self, path: str, data: bytes) -> None:
        if not self._staging_ready:
            await self._api.make_dir(self.sandbox_id, STAGING_DIR)
            self._staging_ready = True
        await self._api.write_file(self.sandbox_id, path, data)

    async def _write_bytes_to_vm_path(self, data: bytes, vm_path: str) -> None:
        """Upload the bytes through the files API, a piece per request, and put them together at ``vm_path``. The bytes
        never enter a command line, so a secret written this way stays out of the sandbox's process list."""
        stage = f"{STAGING_DIR}/upload-{uuid.uuid4().hex}"
        pieces = [f"{stage}.{index}" for index in range(max(1, -(-len(data) // FILE_PIECE_BYTES)))]
        quoted = " ".join(shlex.quote(piece) for piece in pieces)
        try:
            for index, piece in enumerate(pieces):
                await self._upload(piece, data[index * FILE_PIECE_BYTES:(index + 1) * FILE_PIECE_BYTES])
        except BaseException:
            await self._remove_vm_temp_file(*pieces)
            raise
        await self.exec_script(f"cat {quoted} > {shlex.quote(vm_path)}; c=$?; rm -f {quoted}; exit $c")

    async def wait_for_vm(self) -> None:
        """Start dockerd and wait until it answers, and check for Compose v2, which ``deploy_env`` needs."""
        try:
            await self.exec_script(DOCKER_READY, max_retries=2)
        except RuntimeError as exc:
            raise RuntimeError(f"Docker not ready on kosa8 sandbox {self.sandbox_id}: {exc}") from exc
        logger.info("Docker ready on kosa8 sandbox %s", self.sandbox_id)

    async def setup_vm_for_gateway(self, exposed_ports: list[int] | None = None) -> None:
        """Ready Docker; kosa8's preview proxy routes each exposed port into the VM, so no firewall."""
        await self.wait_for_vm()

    def _can_download(self, url: str | None) -> bool:
        """Whether the sandbox can fetch ``url`` itself: its policy allows the host and nothing denies it. An unknown
        policy counts as no."""
        if url is None or self.network_policy is None:
            return False
        host = urlparse(url).hostname
        return (host is not None and allows_host(self.network_policy, host)
                and not matches_host(self.denied_hosts, host))

    async def _download_object_to_vm(self, object_url: str, vm_path: str) -> None:
        """Curl a signed URL the policy admits; push anything else over exec, which needs no egress."""
        object_store = get_config().get_object_store_at(object_url)
        signed = await asyncio.to_thread(object_store.signed_get_url, object_url)
        if self._can_download(signed):
            await self.exec_script(f"curl -fsSL {CURL_RETRY_FLAGS} {shlex.quote(signed)} -o {shlex.quote(vm_path)}")
            return
        await self._write_unsigned_object(object_store, object_url, vm_path)

    async def _load_tarballs(self, artifacts: list) -> None:
        """Load image tarballs, pushing over exec those the sandbox's policy can't download."""
        signed_urls = await self._signed_image_urls(artifacts)
        reachable = [url if self._can_download(url) else None for url in signed_urls]
        await self._load_docker_images(artifacts, reachable)
