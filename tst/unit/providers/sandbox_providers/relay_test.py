"""The relay local Docker runs beside a container with no network joins each connection to the container's port, both
ways, and closes a connection whose target refuses it."""

import asyncio
import socket

import pytest
from pytest_socket import enable_socket

from agent_env.providers.sandbox_providers.local_docker.relay import relay


@pytest.fixture(autouse=True)
def _disable_network():
    """Override the suite-wide socket block (tst/unit/conftest.py) for this module: loopback is the point here."""
    enable_socket()
    yield


@pytest.mark.asyncio
async def test_the_relay_joins_each_connection_to_its_target_both_ways():
    async def echo(reader, writer):
        writer.write(b"echo:" + await reader.readline())
        await writer.drain()
        writer.close()

    target = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = _free_port()
    relaying = asyncio.create_task(relay.main([f"{port}=127.0.0.1:{target.sockets[0].getsockname()[1]}"]))
    try:
        reader, writer = await _connect(port)
        writer.write(b"hello\n")
        await writer.drain()
        assert await reader.read() == b"echo:hello\n"
    finally:
        relaying.cancel()
        target.close()


@pytest.mark.asyncio
async def test_a_connection_whose_target_refuses_it_is_closed():
    port = _free_port()
    relaying = asyncio.create_task(relay.main([f"{port}=127.0.0.1:{_free_port()}"]))
    try:
        reader, _ = await _connect(port)
        assert await reader.read() == b""
    finally:
        relaying.cancel()


async def _connect(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    for _ in range(100):
        try:
            return await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.02)
    raise TimeoutError(f"the relay never listened on {port}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
