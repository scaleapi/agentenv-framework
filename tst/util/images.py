"""Small real images for tests that move media: a PNG whose header a reader can size, built without Pillow."""

from __future__ import annotations

import struct
import zlib


def png(width: int, height: int) -> bytes:
    """A black ``width`` by ``height`` RGB PNG."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    pixels = zlib.compress(b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height)))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", pixels) + chunk(b"IEND", b""))
