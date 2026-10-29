"""The base class's Linux defaults for the OS operations. The driver runs as the session user inside the desktop
session, so both act directly: a command inherits the session's environment, a file lands as that user."""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import tempfile
from pathlib import Path

from .errors import InvalidArgument
from .operations import ExecResult

#: How long output is still kept after the command exits. An app it started in the background may hold its stdout
#: or stderr open; that stream is then drained, and discarded, until the app closes it, so the app never blocks on
#: a full pipe or dies writing to a closed one.
DRAIN_S = 1.0
#: How often a command is checked for having exited.
POLL_S = 0.02
#: How much of a stream is read at a time.
READ_BYTES = 65536
_draining: set = set()


async def run_in_session(argv: list[str], cwd: str | None, env: dict[str, str] | None, timeout_s: float, *,
                         output_bytes: int) -> ExecResult:
    """Run ``argv`` with the driver's own environment plus ``env``, in ``cwd`` (relative to the home), keeping at
    most ``output_bytes`` of each stream; past ``timeout_s`` its process group is killed."""
    home = Path.home()
    workdir = home / cwd if cwd else home
    try:
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=workdir, env={**os.environ, **(env or {})}, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
    except (FileNotFoundError, PermissionError, NotADirectoryError) as e:
        raise InvalidArgument(f"cannot run {argv[0]!r}: {e.strerror or e}") from None
    outputs = [_Capped(output_bytes), _Capped(output_bytes)]
    readers = [asyncio.ensure_future(out.read(stream)) for out, stream in zip(outputs, (process.stdout, process.stderr))]
    timed_out = not await _exited(process, timeout_s)
    if timed_out:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await _exited(process, DRAIN_S)
    _, pending = await asyncio.wait(readers, timeout=DRAIN_S)
    for out in outputs:
        out.returned = True
    for reader in pending:
        _draining.add(reader)
        reader.add_done_callback(_draining.discard)
    stdout, stderr = outputs
    return ExecResult(exit_code=None if timed_out else process.returncode, stdout=bytes(stdout.kept),
                      stderr=bytes(stderr.kept), timed_out=timed_out, truncated=stdout.cut or stderr.cut)


async def _exited(process: asyncio.subprocess.Process, timeout_s: float) -> bool:
    """Whether the process exits within ``timeout_s``. Read from ``returncode``, which is set when the process exits:
    ``Process.wait()`` also waits for the pipes to close, which a background child can hold open."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while process.returncode is None:
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(POLL_S)
    return True


class _Capped:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.kept = bytearray()
        self.cut = False
        self.returned = False

    async def read(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(READ_BYTES):
            if self.returned:
                continue
            room = self.limit - len(self.kept)
            if len(chunk) > room:
                self.cut = True
            self.kept += chunk[:max(room, 0)]


def write_files(files: dict[str, bytes]) -> None:
    """Write each file under the home atomically: a temporary file beside it, then a rename."""
    home = Path.home().resolve()
    for relative, data in files.items():
        destination = home / relative
        if not destination.resolve().is_relative_to(home):
            raise InvalidArgument(f"{relative!r} is outside the home")
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".part")
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(data)
            os.replace(temporary, destination)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise
