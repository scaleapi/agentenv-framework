"""An Agent37 VM presented through the :class:`VmSandbox` contract.

Agent37's ``agent37-ubuntu`` VMs boot Ubuntu with their own kernel and with Docker and Compose v2 running, so the
gateway's docker-compose, an agent's ``docker run`` and image loading run under Docker inside the machine. Commands run
as root through Agent37's exec API, so the ``sudo`` agent-env prefixes is dropped. Each exposed port is an Agent37
public port, an HTTPS URL under ``agent37.app`` that the edge routes to the VM, so the VM opens no firewall ports.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import socket
import ssl
import uuid
import weakref
from typing import Any

import httpx

from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM, SandboxNotFoundError

logger = logging.getLogger(__name__)

API_URL = "https://api.agent37.com"
# Exec goes straight to the host the instance runs on, a hop shorter than through API_URL.
SANDBOX_URL = "https://sandbox.agent37.app"
# The image starts dockerd as its entrypoint; a command that reaches a fresh VM can be ahead of it.
DOCKER_READY = r"""
for _ in $(seq 90); do
  docker info >/dev/null 2>&1 && exec docker compose version
  sleep 1
done
echo "docker daemon did not become ready" >&2
exit 1
"""
# What exec returns when no answer arrives; exec_script retries only this.
UNKNOWN_EXIT_CODE = -1
# A resend under the same name gets the first run's answer, waiting for it if it is still running, so an exec whose
# connection drops is resent; so is one refused before the command started (503 at the per-instance command cap).
RESENDABLE = frozenset({409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 530})
RESEND_DELAYS = (1, 2, 4, 8, 16)
# Past the 900 s Agent37 lets a command run, so its answer arrives first.
_TIMEOUT = httpx.Timeout(30.0, read=1000.0)
# Instance calls answer in about a second, a create once its VM is running. A create that has not answered by then is
# given up on, and the VM it may still make is found by name and deleted.
API_TIMEOUT_SECONDS = 120
KEEPALIVE_SECONDS = 15


def _keepalive_transport() -> httpx.AsyncHTTPTransport:
    """A transport whose connections send TCP keepalives, so a command that prints nothing for minutes keeps the
    connection waiting on it from being dropped as idle."""
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle = getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None)
    if idle is not None:
        options.append((socket.IPPROTO_TCP, idle, KEEPALIVE_SECONDS))
    if hasattr(socket, "TCP_KEEPINTVL"):
        options.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, KEEPALIVE_SECONDS))
    return httpx.AsyncHTTPTransport(socket_options=options)


class Agent37Error(RuntimeError):
    """An Agent37 API call that failed, with the status and error code it answered."""

    def __init__(self, response: httpx.Response):
        self.status_code = response.status_code
        self.code, message = _error_of(response)
        super().__init__(f"Agent37 {response.request.method} {response.request.url.path} answered "
                         f"{response.status_code} {self.code}: {message}")


def _error_of(response: httpx.Response) -> tuple[str, str]:
    """``(code, message)`` of an error answer: the API's ``{"error": {"code", "message"}}`` envelope, or the flat
    ``{"error": "<code>"}`` the edge answers with before a request reaches an instance."""
    try:
        error = response.json().get("error")
    except ValueError:
        return "unknown", response.text[:500]
    if isinstance(error, dict):
        return str(error.get("code", "unknown")), str(error.get("message", ""))
    return str(error), ""


class Agent37Api:
    """The Agent37 calls a sandbox makes: instances on the control plane, and exec on the host running one.

    The key comes from the provider's resolved config and is sent only to Agent37. ``transport`` is injectable so unit
    tests never open a connection."""

    def __init__(self, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
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
                transport=self._transport or _keepalive_transport(), timeout=_TIMEOUT,
                headers={"User-Agent": "agentenv-framework"},
            )
        return client

    async def close(self) -> None:
        client = self._clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.aclose()

    async def _call(self, method: str, path: str, *, json: Any = None) -> httpx.Response:
        return await self._client().request(method, f"{API_URL}{path}", json=json,
                                            headers={"Authorization": f"Bearer {self._api_key}"},
                                            timeout=httpx.Timeout(30.0, read=API_TIMEOUT_SECONDS))

    async def create_instance(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create an instance; Agent37 answers once it is running."""
        response = await self._call("POST", "/v1/instances", json=body)
        if response.status_code != 201:
            raise Agent37Error(response)
        return response.json()

    async def get_instance(self, instance_id: str) -> dict[str, Any]:
        response = await self._call("GET", f"/v1/instances/{instance_id}")
        if response.status_code == 404:
            raise SandboxNotFoundError(f"Agent37 instance {instance_id!r} was not found")
        if response.status_code != 200:
            raise Agent37Error(response)
        return response.json()

    async def list_instances(self) -> list[dict[str, Any]]:
        response = await self._call("GET", "/v1/instances")
        if response.status_code != 200:
            raise Agent37Error(response)
        return response.json()["data"]

    async def delete_instance(self, instance_id: str) -> None:
        """Delete an instance; one that is already gone is not an error."""
        response = await self._call("DELETE", f"/v1/instances/{instance_id}")
        if response.status_code == 404:
            logger.info("Agent37 instance %s was already gone at terminate", instance_id)
        elif response.status_code != 200:
            raise Agent37Error(response)

    async def exec(self, instance_id: str, body: dict[str, Any]) -> httpx.Response:
        return await self._client().post(f"{SANDBOX_URL}/v1/instances/{instance_id}/exec", json=body,
                                         headers={"X-Agent37-Key": self._api_key})


def applied_policy(network: Any) -> NetworkPolicy:
    """The egress policy an instance's ``network`` field applies: ``allow-all``, ``deny-all`` or ``{"allow": [...]}``,
    a list of IPv4 addresses and ranges."""
    if network == "deny-all":
        return NetworkPolicy(mode=NetworkMode.ALLOWLIST)
    if isinstance(network, dict):
        return NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_cidrs=tuple(network.get("allow") or ()))
    return NetworkPolicy()


def public_urls(instance: dict[str, Any]) -> dict[int, str]:
    """``{port: url}`` of an instance's public ports."""
    return {int(entry["port"]): entry["url"] for entry in instance.get("public_ports") or []}


class _BytesReader:
    """The small ``StreamReader`` subset consumed by ``Sandbox.exec_with_output``."""

    def __init__(self, value: str):
        self._value = value.encode()

    async def read(self) -> bytes:
        return self._value


class _CompletedProcess:
    """A finished Agent37 exec in the common process protocol."""

    def __init__(self, stdout: str, stderr: str, exit_code: int):
        self.stdout = _BytesReader(stdout)
        self.stderr = _BytesReader(stderr)
        self._exit_code = exit_code

    async def wait(self) -> int:
        return self._exit_code


class Agent37Sandbox(VmSandbox):
    """An Agent37 VM with Docker; its API is its provider's, shared with the provider's other sandboxes."""

    type = "agent37"

    def __init__(
        self,
        api: Agent37Api,
        instance_id: str,
        *,
        tunnel_urls: dict[int, str],
        network_policy: NetworkPolicy | None,
    ):
        self._api = api
        self.sandbox_id = instance_id
        self.mode = SANDBOX_MODE_VM
        self.vnc_url = None
        self.tunnel_urls = tunnel_urls
        self.network_policy = network_policy

    async def terminate(self) -> None:
        await self._api.delete_instance(self.sandbox_id)

    async def exec(self, *command: str) -> _CompletedProcess:
        """Run argv to completion as root, with no input; Agent37 stops a command at 900 s. Every resend carries the
        first call's ``name``, so Agent37 answers it with that run's result instead of running the command again; when no
        resend gets an answer, the exit code is -1, which ``exec_script`` retries."""
        if command[:1] == ("sudo",):
            command = command[1:]
        body = {"command": shlex.join(command), "user": "root", "name": uuid.uuid4().hex}
        resends = iter(RESEND_DELAYS)
        while True:
            try:
                response = await self._api.exec(self.sandbox_id, body)
            # httpx passes some TLS failures through unwrapped, such as a peer's bad-record-MAC alert.
            except (httpx.TransportError, ssl.SSLError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    result = response.json()
                    if result.get("truncated"):
                        logger.warning("Agent37 cut the output of a command in %s to 512 KB per stream: %.200s",
                                       self.sandbox_id, body["command"])
                    return _CompletedProcess(result["stdout"], result["stderr"], int(result["exit_code"]))
                if response.status_code not in RESENDABLE:
                    raise Agent37Error(response)
                failure = str(Agent37Error(response))
            delay = next(resends, None)
            if delay is None:
                return _CompletedProcess("", failure, UNKNOWN_EXIT_CODE)
            logger.warning("Agent37 exec in %s got no answer (%s); resending in %ss", self.sandbox_id, failure, delay)
            await asyncio.sleep(delay)

    async def wait_for_vm(self) -> None:
        """Wait until the VM's Docker daemon answers, and check for Compose v2, which ``deploy_env`` needs."""
        try:
            await self.exec_script(DOCKER_READY, max_retries=2)
        except RuntimeError as exc:
            raise RuntimeError(f"Docker not ready on Agent37 instance {self.sandbox_id}: {exc}") from exc
        logger.info("Docker ready on Agent37 instance %s", self.sandbox_id)

    async def setup_vm_for_gateway(self, exposed_ports: list[int] | None = None) -> None:
        """Ready Docker; the edge routes each public port to the VM, so no firewall."""
        await self.wait_for_vm()
