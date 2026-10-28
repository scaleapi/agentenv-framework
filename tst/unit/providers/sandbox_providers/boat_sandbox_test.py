"""Offline unit tests for a boat sandbox: exec, its failures and cancellation, Docker readiness and file staging."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import threading
from types import SimpleNamespace

import httpx
import pytest
import urllib3

from agent_env.providers.sandbox_providers.boat import lifecycle
from agent_env.providers.sandbox_providers.boat import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.boat.sandbox import DOCKER_READY, UNKNOWN_EXIT_CODE, BoatSandbox
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy

from tst.unit.providers.sandbox_providers.boat_fakes import FakeBoatApi, FakeCommands, api_error

RETRY = lifecycle.Retry(attempts=1, base_delay=0, max_delay=0, delete_attempts=1, delete_delay=0, cancel_wait=1)


def sandbox(commands: FakeCommands, api: FakeBoatApi | None = None) -> BoatSandbox:
    return BoatSandbox(api or FakeBoatApi(), "bx_1", tunnel_urls={}, network_policy=NetworkPolicy(), retry=RETRY,
                       run_command=commands)


@pytest.mark.asyncio
async def test_exec_sends_argv_as_one_quoted_command_and_keeps_the_streams_apart():
    commands = FakeCommands(handler=lambda argv: (3, "out\n", "err\n"))
    code, out, err = await sandbox(commands).exec_with_output("echo", "a b", "c'd")
    assert commands.commands == [["echo", "a b", "c'd"]]
    assert (code, out, err) == (3, "out\n", "err\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [urllib3.exceptions.ReadTimeoutError(None, "/", "timed out"),
                                     ConnectionResetError("reset"), api_error(502, "sandbox_direct_failed"),
                                     api_error(429, "IP_RATE_LIMIT_EXCEEDED")])
async def test_a_dropped_transport_or_a_failing_boat_reads_as_an_unknown_exit(failure):
    code, _, err = await sandbox(FakeCommands(handler=lambda argv: failure)).exec_with_output("true")
    assert code == UNKNOWN_EXIT_CODE and err


@pytest.mark.asyncio
async def test_a_command_boat_refuses_is_raised_not_retried():
    with pytest.raises(Exception, match="machine_not_running"):
        await sandbox(FakeCommands(handler=lambda argv: api_error(409, "machine_not_running"))).exec("true")


@pytest.mark.asyncio
async def test_an_error_frame_or_a_stream_with_no_exit_reads_as_an_unknown_exit():
    class ErrorFrame(FakeCommands):
        def __call__(self, api, sandbox_id, command, cwd=None):
            yield {"type": "started", "processId": 7}
            yield {"type": "stdout", "data": "partial"}
            yield {"type": "error", "message": "process lost"}

    code, out, err = await sandbox(ErrorFrame()).exec_with_output("true")
    assert (code, out, err) == (UNKNOWN_EXIT_CODE, "partial", "process lost")

    class NoExit(FakeCommands):
        def __call__(self, api, sandbox_id, command, cwd=None):
            yield {"type": "started", "processId": 7}

    code, _, err = await sandbox(NoExit()).exec_with_output("true")
    assert code == UNKNOWN_EXIT_CODE and "without an exit" in err


@pytest.mark.asyncio
async def test_a_cancelled_exec_stops_its_command():
    api = FakeBoatApi()
    commands = FakeCommands(gate=threading.Event())
    task = asyncio.create_task(sandbox(commands, api).exec("sleep", "600"))
    assert await asyncio.to_thread(commands.started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert api.stopped == [("bx_1", 100)]
    commands.gate.set()


@pytest.mark.asyncio
async def test_a_command_cancelled_before_it_started_is_stopped_when_it_starts():
    api = FakeBoatApi()
    release = threading.Event()

    class SlowStart(FakeCommands):
        def __call__(self, api, sandbox_id, command, cwd=None):
            assert release.wait(5)
            yield {"type": "started", "processId": 42}
            yield {"type": "exit", "exitCode": 0}

    task = asyncio.create_task(sandbox(SlowStart(), api).exec("true"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    for _ in range(100):
        if api.stopped:
            break
        await asyncio.sleep(0.01)
    assert api.stopped == [("bx_1", 42)]


@pytest.mark.asyncio
async def test_setup_waits_for_docker_and_needs_compose():
    commands = FakeCommands(handler=lambda argv: (0, "Docker Compose version v5.3.1\n", ""))
    await sandbox(commands).setup_vm_for_gateway([8000])
    assert commands.commands == [["bash", "-c", DOCKER_READY]]
    failing = FakeCommands(handler=lambda argv: (1, "", "docker daemon did not become ready\n"))
    with pytest.raises(RuntimeError, match="not ready in boat sandbox bx_1"):
        await sandbox(failing).wait_for_vm()


@pytest.fixture
def received(monkeypatch) -> dict[str, bytes]:
    """The bodies PUT to upload URLs, by URL, through a transport that answers 200."""
    bodies: dict[str, bytes] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies[str(request.url)] = await request.aread()
        return httpx.Response(200, json={"success": True})

    real = httpx.AsyncClient
    monkeypatch.setattr(sandbox_module.httpx, "AsyncClient",
                        lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs))
    return bodies


def moved(payload: bytes, digest: str | None = None) -> FakeCommands:
    """Commands that answer the move into place with ``digest``, the payload's own by default."""
    sha = digest or hashlib.sha256(payload).hexdigest()
    return FakeCommands(handler=lambda argv: (0, f"{sha}  {argv[-1].split()[-1]}\n" if "sha256sum" in argv[-1] else "", ""))


@pytest.mark.asyncio
async def test_bytes_one_exec_carries_go_in_it(received):
    api, commands = FakeBoatApi(), FakeCommands()
    await sandbox(commands, api).write_host_file(b"small", "/opt/smoke/small.bin")
    assert api.uploads == [] and received == {}
    assert any("base64 -d" in argv[-1] for argv in commands.commands)


@pytest.mark.asyncio
async def test_more_bytes_go_through_an_upload_url_and_are_moved_into_place_as_root(received):
    api, payload = FakeBoatApi(), os.urandom(300_000)
    commands = moved(payload)
    await sandbox(commands, api).write_host_file(payload, "/opt/smoke/payload.bin")
    (staged,) = api.uploads
    assert staged.startswith("/tmp/") and list(received.values()) == [payload]
    script = commands.commands[-1]
    assert script[:3] == ["sudo", "bash", "-c"] and f"mv -f {staged} /opt/smoke/payload.bin" in script[-1]
    assert "chown root:root" in script[-1]


@pytest.mark.asyncio
async def test_an_upload_that_arrives_changed_is_refused(received):
    payload = os.urandom(300_000)
    with pytest.raises(RuntimeError, match="does not match the upload"):
        await sandbox(moved(payload, digest="0" * 64)).write_host_file(payload, "/opt/smoke/payload.bin")


@pytest.mark.asyncio
async def test_an_object_the_store_cannot_sign_streams_through_an_upload_url(received):
    payload = os.urandom(9 * 1024 * 1024 + 17)
    store = SimpleNamespace(open=lambda url: io.BytesIO(payload))
    await sandbox(moved(payload))._write_unsigned_object(store, "file:///images/env.tar.gz", "/opt/images/env.tar.gz")
    assert list(received.values()) == [payload]
