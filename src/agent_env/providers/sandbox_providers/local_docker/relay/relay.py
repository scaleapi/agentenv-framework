"""Relay TCP connections: each argument ``PORT=HOST:PORT`` accepts connections on the first port and joins each to the
second address, both ways, until either side closes."""

import asyncio
import sys


async def main(routes: list[str]) -> None:
    servers = []
    for route in routes:
        port, target = route.split("=")
        host, _, target_port = target.rpartition(":")
        servers.append(await asyncio.start_server(
            lambda reader, writer, host=host, target_port=int(target_port): join(reader, writer, host, target_port),
            "0.0.0.0", int(port)))
    await asyncio.gather(*(server.serve_forever() for server in servers))


async def join(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, host: str, port: int) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
    except OSError:
        writer.close()
        return
    await asyncio.gather(pipe(reader, upstream_writer), pipe(upstream_reader, writer), return_exceptions=True)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
