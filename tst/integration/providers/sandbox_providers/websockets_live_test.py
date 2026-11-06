"""Whether each remote sandbox provider's ``WEBSOCKETS`` declaration is right: whether its tunnel carries what a
website browser's CDP address needs. Through the tunnel, a CDP command is answered over a WebSocket, a 10 MB frame comes
back intact, and after five idle minutes a connection pinged every 20 s, as agentenv's driver and gateway ping, still
answers. A connection beside it that is never pinged records the tunnel's own idle limit, and doesn't decide. The server
behind the tunnel is a stdlib one this test starts in a sandbox of the provider, a VM where it makes them, so only
the tunnel is under test.

A case runs when the resolved config builds its provider (the ``remote_sandbox`` capability) and skips otherwise. Each
prints its findings as a ``websockets-check`` JSON line. No test reads or prints a key."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import httpx
import pytest
import websockets
from websockets.asyncio.client import connect

from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, pytest.mark.asyncio]

#: The remote providers, each of which declares whether it carries WebSockets.
REMOTE = ("e2b", "tensorlake", "vercel", "coreweave", "daytona", "boat", "modal", "modal_vm", "boxd", "sail_vm", "tenki",
          "agent37")
PORT = 8765
FRAME_BYTES = 10 * 1024 * 1024
IDLE_S = 300
PING_S = 20.0
GET_VERSION = {"id": 1, "method": "Browser.getVersion"}
#: What the server answers ``Browser.getVersion`` with.
PRODUCT = "agentenv-websockets-check/1"
SANDBOX_CPU = 1
SANDBOX_MEMORY_MB = 2048
SANDBOX_LIFETIME_S = 1800
SERVED_WITHIN_S = 300
SERVED_POLL_S = 3
HTTP_TIMEOUT_S = 10
OPEN_TIMEOUT_S = 60
ANSWER_TIMEOUT_S = 60
FRAME_TIMEOUT_S = 120
ERROR_CHARS = 300

#: A WebSocket server in the standard library alone: it answers ``Browser.getVersion`` as a browser would, echoes every
#: other message, answers pings, and answers a plain GET with 200 for a tunnel's health check.
SERVER = r'''
import base64, hashlib, json, socketserver, struct, sys

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def read_message(stream):
    """A whole message, its fragments joined: (opcode, payload), or (None, b"") once the connection ends."""
    opcode, parts = None, []
    while True:
        head = stream.read(2)
        if len(head) < 2:
            return None, b""
        fin, code, masked, length = head[0] & 0x80, head[0] & 0x0F, head[1] & 0x80, head[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", stream.read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", stream.read(8))[0]
        mask = stream.read(4) if masked else b""
        payload = stream.read(length)
        if mask:
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        if code >= 8:  # a control frame, never fragmented, may come between fragments
            return code, payload
        opcode = code if code else opcode
        parts.append(payload)
        if fin:
            return opcode, b"".join(parts)


def send(conn, opcode, payload):
    length = len(payload)
    if length < 126:
        head = struct.pack(">BB", 0x80 | opcode, length)
    elif length < 1 << 16:
        head = struct.pack(">BBH", 0x80 | opcode, 126, length)
    else:
        head = struct.pack(">BBQ", 0x80 | opcode, 127, length)
    conn.sendall(head + payload)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        conn, data = self.request, b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                return
            data += chunk
        lines = data.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")
        headers = {name.strip().lower(): value.strip() for name, _, value in (line.partition(":") for line in lines[1:])}
        key = headers.get("sec-websocket-key")
        if not key:
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            return
        accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        conn.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      "Sec-WebSocket-Accept: " + accept + "\r\n\r\n").encode())
        stream = conn.makefile("rb")
        while True:
            opcode, payload = read_message(stream)
            if opcode is None or opcode == 8:
                if opcode == 8:
                    send(conn, 8, payload[:2])
                return
            if opcode == 9:
                send(conn, 10, payload)
            elif opcode == 1 and json.loads(payload).get("method") == "Browser.getVersion":
                answer = {"id": json.loads(payload)["id"], "result": {"product": sys.argv[2]}}
                send(conn, 1, json.dumps(answer).encode())
            elif opcode in (1, 2):
                send(conn, opcode, payload)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


Server(("0.0.0.0", int(sys.argv[1])), Handler).serve_forever()
'''

PYTHON_IMAGE = "public.ecr.aws/docker/library/python:3.12-alpine"
DIR = "/tmp/websockets-check"


def _start_script() -> str:
    """Writes the server and starts it detached, in Docker where the sandbox has it, else on its own Python."""
    encoded = base64.b64encode(SERVER.encode()).decode()
    return f"""set -e
mkdir -p {DIR}
echo {encoded} | base64 -d > {DIR}/server.py
if command -v docker >/dev/null 2>&1; then
  docker run -d --name websockets-check -p {PORT}:{PORT} -v {DIR}:/check {PYTHON_IMAGE} python /check/server.py {PORT} {PRODUCT} >/dev/null
else
  setsid nohup python3 {DIR}/server.py {PORT} {PRODUCT} > {DIR}/server.log 2>&1 < /dev/null &
fi
"""


@pytest.mark.parametrize("name", [pytest.param(name, marks=skip_without_remote_sandbox(name)) for name in REMOTE])
async def test_a_providers_websockets_declaration_matches_what_its_tunnel_carries(name):
    provider = build_sandbox_provider(name)
    found: dict = {"provider": name, "declared": provider.WEBSOCKETS}
    try:
        sandbox = await _sandbox(provider)
        try:
            await _serve(sandbox)
            url = sandbox.tunnel_urls[PORT]
            await _until_served(url)
            found.update(await _checked("ws" + url.removeprefix("http")))
        finally:
            await sandbox.terminate()
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            await close()
    found["carries"] = all(found.get(check) is True for check in ("cdp", "frame", "idle_pinged"))
    print("websockets-check", json.dumps(found))

    assert found["carries"] == found["declared"], (
        f"{name} declares WEBSOCKETS = {found['declared']}, but its tunnel "
        f"{'carries' if found['carries'] else 'does not carry'} what a website browser's CDP address needs: {found}")


async def _sandbox(provider):
    """A VM, or on a provider that makes only containers, a container of a stock Python image."""
    try:
        return await provider.create_vm(cpu=SANDBOX_CPU, memory=SANDBOX_MEMORY_MB, exposed_ports=[PORT],
                                        timeout=SANDBOX_LIFETIME_S)
    except NotImplementedError:
        return await provider.create_container(image_name=PYTHON_IMAGE, port=PORT, env={}, cpu=SANDBOX_CPU,
                                               memory=SANDBOX_MEMORY_MB, timeout=SANDBOX_LIFETIME_S)


async def _serve(sandbox) -> None:
    if isinstance(sandbox, VmSandbox):
        await sandbox.exec_script(_start_script())
        return
    exit_code, stdout, stderr = await sandbox.exec_with_output("sh", "-c", _start_script())
    assert exit_code == 0, f"starting the server failed (exit {exit_code}): {stdout[-500:]} {stderr[-500:]}"


async def _until_served(url: str, timeout_s: float = SERVED_WITHIN_S) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(url)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(SERVED_POLL_S)
    raise TimeoutError(f"nothing answered at {url} within {timeout_s:.0f} s")


async def _checked(ws_url: str) -> dict:
    """What the tunnel at ``ws_url`` carries: a CDP answer, a 10 MB frame back intact, and, after ``IDLE_S`` idle, an
    answer on a pinged connection, and how long one never pinged lasted."""
    found: dict = {}
    try:
        async with connect(ws_url, max_size=None, open_timeout=OPEN_TIMEOUT_S, ping_interval=None) as conn:
            found["cdp"] = await _version(conn) == PRODUCT
            frame = os.urandom(FRAME_BYTES)
            await conn.send(frame)
            found["frame"] = await asyncio.wait_for(conn.recv(), FRAME_TIMEOUT_S) == frame
    except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as e:
        found.setdefault("cdp", False)
        found.setdefault("frame", False)
        found["error"] = _error(e)
        return found
    try:
        pinged = await connect(ws_url, max_size=None, open_timeout=OPEN_TIMEOUT_S, ping_interval=PING_S,
                               ping_timeout=PING_S)
    except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as e:
        found["idle_pinged"] = False
        found["idle_pinged_error"] = _error(e)
        return found
    try:
        silent = await connect(ws_url, max_size=None, open_timeout=OPEN_TIMEOUT_S, ping_interval=None)
    except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as e:
        silent = None
        found["idle_unpinged_error"] = _error(e)
    started = time.monotonic()

    async def lasted(conn) -> int:
        await conn.wait_closed()
        return round(time.monotonic() - started)

    silent_lasted = asyncio.ensure_future(lasted(silent)) if silent is not None else None
    try:
        await asyncio.sleep(IDLE_S)
        if silent_lasted is not None:
            found["idle_unpinged_s"] = silent_lasted.result() if silent_lasted.done() else None
        try:
            found["idle_pinged"] = await _version(pinged) == PRODUCT
        except (asyncio.TimeoutError, websockets.WebSocketException) as e:
            found["idle_pinged"] = False
            found["idle_pinged_error"] = _error(e)
    finally:
        if silent_lasted is not None:
            silent_lasted.cancel()
            await silent.close()
        await pinged.close()
    return found


async def _version(conn) -> str:
    await conn.send(json.dumps(GET_VERSION))
    return json.loads(await asyncio.wait_for(conn.recv(), ANSWER_TIMEOUT_S))["result"]["product"]


def _error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:ERROR_CHARS]
