"""Exercise HTTPX's TCP, TLS and read timers against a loopback peer."""

import asyncio
import sys
from contextlib import asynccontextmanager
from textwrap import dedent

import anyio
import httpx
import pytest

from agentenv_protocol import client


_CARD = {"capabilities": {"extensions": [{"uri": "urn:example:setup", "params": {"endpoint": "/setup"}}]}}


async def call(url, timeout):
    with client.retry_extension_connections():
        return await client.invoke_extension(url, _CARD, "urn:example:setup", {}, timeout=timeout)


@asynccontextmanager
async def listen(handler):
    writers, tasks = [], []

    def accept(reader, writer):
        writers.append(writer)
        tasks.append(asyncio.create_task(handler(reader, writer)))

    async with await asyncio.start_server(accept, "127.0.0.1", 0) as server:
        try:
            yield f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
        finally:
            for writer in writers:
                writer.close()
            await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_asyncio_extension_without_sniffio(monkeypatch):
    monkeypatch.setenv("no_proxy", "*")
    probe = dedent("""
        import asyncio
        import sys

        sys.modules["sniffio"] = None
        try:
            import anyio
            import httpx
        except ModuleNotFoundError as exc:
            if exc.name == "sniffio":
                sys.exit(77)
            raise

        from agentenv_protocol import client
        card = {"capabilities": {"extensions": [{"uri": "urn:example:setup", "params": {"endpoint": "/setup"}}]}}
        with client.retry_extension_connections():
            result = asyncio.run(client.invoke_extension(sys.argv[1], card, "urn:example:setup", {}, timeout=3))
        assert result == {"name": "items"}
    """)

    async def reply(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 16\r\n\r\n{"name":"items"}')
        await writer.drain()

    async with listen(reply) as address:
        child = await asyncio.create_subprocess_exec(sys.executable, "-c", probe, f"http://{address}",
                                                     stderr=asyncio.subprocess.PIPE)
        _, stderr = await child.communicate()
    if child.returncode == 77:
        pytest.skip("the installed AnyIO or HTTPX version requires sniffio")
    assert child.returncode == 0, stderr.decode()


@pytest.mark.asyncio
async def test_stalled_tcp_connect_leaves_time_to_retry(monkeypatch):
    monkeypatch.setenv("no_proxy", "*")
    connect_tcp = anyio.connect_tcp
    attempts, requests = [], []

    async def stall_first(*args, **kwargs):
        attempts.append(None)
        if len(attempts) == 1:
            await asyncio.Event().wait()
        return await connect_tcp(*args, **kwargs)

    async def reply(reader, writer):
        requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
        await writer.drain()

    monkeypatch.setattr(anyio, "connect_tcp", stall_first)
    async with listen(reply) as address:
        assert await call(f"http://{address}", timeout=2) == {}
    assert len(attempts) == 2 and len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_stalled_tls_closes_each_connection(monkeypatch, cancel):
    monkeypatch.setenv("no_proxy", "*")
    hellos, closed = asyncio.Event(), []

    async def hold(reader, writer):
        peer_closed = asyncio.Event()
        closed.append(peer_closed)
        assert await reader.read(4096)
        hellos.set()
        await reader.read()
        peer_closed.set()

    async with listen(hold) as address:
        task = asyncio.create_task(call(f"https://{address}", timeout=1.5))
        if cancel:
            await asyncio.wait_for(hellos.wait(), 2)
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else httpx.ConnectTimeout):
            await task
        assert len(closed) == (1 if cancel else 2)
        await asyncio.wait_for(asyncio.gather(*(peer.wait() for peer in closed)), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("progress", [False, True])
async def test_buffered_reply_keeps_its_read_inactivity_timeout(monkeypatch, progress):
    monkeypatch.setenv("no_proxy", "*")

    async def reply(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n{")
        if progress:
            for piece in (b" ", b" ", b" ", b"}"):
                await asyncio.sleep(0.1)
                writer.write(piece)
                await writer.drain()
        else:
            await reader.read()

    async with listen(reply) as address:
        if progress:
            assert await call(f"http://{address}", timeout=0.3) == {}
        else:
            with pytest.raises(httpx.ReadTimeout):
                await call(f"http://{address}", timeout=0.3)
