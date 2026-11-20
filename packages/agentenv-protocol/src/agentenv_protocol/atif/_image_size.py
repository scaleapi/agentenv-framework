"""An image's width and height, read from its header for the four image types ATIF has parts for, so the profile can
carry them without Pillow. A header this can't read gives None: the image is stored all the same."""

from __future__ import annotations

import struct

_PNG = b"\x89PNG\r\n\x1a\n"
#: JPEG's start-of-frame markers, the segments that hold the frame's size: every SOFn but DHT, JPG and DAC.
_JPEG_FRAMES = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
#: JPEG markers that stand alone, with no length after them.
_JPEG_BARE = frozenset({0x01, *range(0xD0, 0xD9)})


def image_size(data: bytes, media_type: str) -> tuple[int, int] | None:
    """``(width, height)`` of a PNG, JPEG, GIF or WebP image from its header, or None."""
    read = {"image/png": _png, "image/jpeg": _jpeg, "image/gif": _gif, "image/webp": _webp}.get(media_type)
    try:
        size = read(data) if read is not None else None
    except (IndexError, struct.error):
        return None
    return size if size is not None and size[0] > 0 and size[1] > 0 else None


def _png(data: bytes) -> tuple[int, int] | None:
    if data[:8] != _PNG or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def _gif(data: bytes) -> tuple[int, int] | None:
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    return struct.unpack("<HH", data[6:10])


def _jpeg(data: bytes) -> tuple[int, int] | None:
    if data[:2] != b"\xff\xd8":
        return None
    at = 2
    while at + 4 <= len(data):
        if data[at] != 0xFF:
            return None
        marker = data[at + 1]
        if marker == 0xFF:
            at += 1
            continue
        if marker in _JPEG_BARE:
            at += 2
            continue
        (length,) = struct.unpack(">H", data[at + 2:at + 4])
        if marker in _JPEG_FRAMES:
            height, width = struct.unpack(">HH", data[at + 5:at + 9])
            return width, height
        if marker == 0xDA:
            return None
        at += 2 + length
    return None


def _webp(data: bytes) -> tuple[int, int] | None:
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L" and data[20] == 0x2F:
        (bits,) = struct.unpack("<I", data[21:25])
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
    return None
