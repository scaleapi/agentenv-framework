"""What the probe reads from an MP4 without decoding it: the video track's size, frame count and duration, from the
``moov`` box's ``tkhd``, ``mdhd`` and ``stts``."""
from __future__ import annotations

import struct
from collections.abc import Iterator


def read_mp4(data: bytes) -> dict:
    """``{"width", "height", "frames", "duration_s"}`` of the first video track. Raises ValueError for a file that
    isn't an MP4 with one."""
    top = dict(_boxes(data, 0, len(data)))
    if b"ftyp" not in top or b"moov" not in top:
        raise ValueError(f"not an MP4: its top-level boxes are {sorted(t.decode(errors='replace') for t in top)}")
    for trak in _children(data, top[b"moov"], b"trak"):
        mdia = _child(data, trak, b"mdia")
        hdlr = _child(data, mdia, b"hdlr")
        if data[hdlr[0] + 8:hdlr[0] + 12] != b"vide":
            continue
        width, height = _tkhd_size(data, _child(data, trak, b"tkhd"))
        timescale, duration = _mdhd(data, _child(data, mdia, b"mdhd"))
        stts = _child(data, _child(data, _child(data, mdia, b"minf"), b"stbl"), b"stts")
        entries = struct.unpack(">I", data[stts[0] + 4:stts[0] + 8])[0]
        frames = sum(struct.unpack(">I", data[stts[0] + 8 + 8 * i:stts[0] + 12 + 8 * i])[0] for i in range(entries))
        return {"width": width, "height": height, "frames": frames, "duration_s": duration / timescale}
    raise ValueError("the MP4 has no video track")


def _boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, tuple[int, int]]]:
    """Each box between ``start`` and ``end``: its type, and where its payload starts and ends."""
    at = start
    while at + 8 <= end:
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        header = 8
        if size == 1:
            size, header = struct.unpack(">Q", data[at + 8:at + 16])[0], 16
        elif size == 0:
            size = end - at
        if size < header or at + size > end:
            raise ValueError(f"a {kind.decode(errors='replace')} box runs past its container")
        yield kind, (at + header, at + size)
        at += size


def _children(data: bytes, box: tuple[int, int], kind: bytes) -> list[tuple[int, int]]:
    return [span for k, span in _boxes(data, *box) if k == kind]


def _child(data: bytes, box: tuple[int, int], kind: bytes) -> tuple[int, int]:
    found = _children(data, box, kind)
    if not found:
        raise ValueError(f"no {kind.decode()} box where the MP4 needs one")
    return found[0]


def _tkhd_size(data: bytes, box: tuple[int, int]) -> tuple[int, int]:
    at = box[0] + (88 if data[box[0]] == 1 else 76)
    width, height = struct.unpack(">II", data[at:at + 8])
    return width >> 16, height >> 16


def _mdhd(data: bytes, box: tuple[int, int]) -> tuple[int, int]:
    if data[box[0]] == 1:
        return struct.unpack(">IQ", data[box[0] + 20:box[0] + 32])
    return struct.unpack(">II", data[box[0] + 12:box[0] + 20])
