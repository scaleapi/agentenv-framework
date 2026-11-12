"""A front that ends TLS for a server on this machine, as a sandbox provider's tunnel can: it presents a certificate the
local CA signs and passes each connection's bytes on to the server over plain TCP, adding nothing that says how it was
reached, so the server sees the tunnel's host over http."""
from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator

from agent_env.store.object_store.local.tls import LocalCA, server_context

FRONT_HOST = "127.0.0.1"
CHUNK_BYTES = 64 * 1024


@contextlib.asynccontextmanager
async def tls_front(ca: LocalCA, host: str, port: int) -> AsyncIterator[str]:
    """``https://127.0.0.1:<port>``, ending TLS for the server at ``host``:``port`` until the block ends."""
    writers: set[asyncio.StreamWriter] = set()

    async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writers.add(writer)
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
        except OSError:
            writer.close()
            return
        writers.add(upstream_writer)
        await asyncio.gather(_pipe(reader, upstream_writer), _pipe(upstream_reader, writer))

    server = await asyncio.start_server(forward, FRONT_HOST, 0, ssl=server_context(ca, [FRONT_HOST]))
    try:
        yield f"https://{FRONT_HOST}:{server.sockets[0].getsockname()[1]}"
    finally:
        server.close()
        for writer in writers:
            writer.close()
        await server.wait_closed()


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(CHUNK_BYTES):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()
