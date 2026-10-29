"""A scripted `gemini` process for the bridge tests: fixed stdout, stderr and exit code. ``hang``
keeps its output open, as a CLI that never finishes."""
from __future__ import annotations

import asyncio


def _reader(data: bytes, *, eof: bool = True) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    if eof:
        reader.feed_eof()
    return reader


class FakeCLI:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0, *, hang: bool = False):
        self.stdout = _reader(stdout, eof=not hang)
        self.stderr = _reader(stderr, eof=not hang)
        self.returncode = returncode
        self.killed = False

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode
