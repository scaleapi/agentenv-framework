"""The base class's defaults for the OS operations on Android, for a driver beside the device: each goes through adb
(``AGENTENV_ADB``, else the one on PATH, with adb's own ``ANDROID_SERIAL`` and ``ADB_SERVER_SOCKET``) as the device's
``shell`` user, under /sdcard."""
from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from ._linux import DRAIN_S, _Capped
from .errors import DeliveryFailed, DeviceUnavailable, InvalidArgument
from .operations import ExecResult

#: Who the device's commands run as, and the home they run in, which the computer skill names to the agent.
USER = "shell"
HOME = PurePosixPath("/sdcard")
#: How long one adb call the defaults make for their own bookkeeping may take; a push or pull, ``TRANSFER_TIMEOUT_S``.
ADB_TIMEOUT_S = 60
TRANSFER_TIMEOUT_S = 600
#: What the device's shell prints on stderr before it runs a command, so what adb printed before it is adb's own.
STARTED = b"agentenv-started\n"
#: The most of stderr adb's own messages may fill before ``STARTED``, as when it starts its server.
ADB_PREAMBLE_BYTES = 4096
#: What the device's shell prints on stderr, exiting ``NOT_FOUND_EXIT``, when the folder or the program isn't there.
CANNOT_RUN = "agentenv-cannot-run:"
NOT_FOUND_EXIT = 127
#: How much of what the device answered an error quotes.
SHOWN_BYTES = 500
#: How many hex characters name the partial file a write pushes beside its target before renaming it over it.
PARTIAL_ID_CHARS = 12
#: The broadcast that has the media scanner list a written file, so gallery and file apps see it.
MEDIA_SCAN = "android.intent.action.MEDIA_SCANNER_SCAN_FILE"


async def run_in_session(argv: list[str], cwd: str | None, env: dict[str, str] | None, timeout_s: float, *,
                         output_bytes: int) -> ExecResult:
    """The device's shell prints ``STARTED`` first, after any lines adb prints starting its server, so adb failing
    is told apart from the command failing; and ``CANNOT_RUN`` when the folder or program isn't there."""
    folder, program = shlex.quote(str(HOME / (cwd or ""))), shlex.quote(argv[0])
    assignments = "".join(f"{shlex.quote(f'{name}={value}')} " for name, value in (env or {}).items())
    lookup = f"PATH={shlex.quote(env['PATH'])}; " if env and "PATH" in env else ""
    command = (f"echo agentenv-started >&2; "
               f"cd {folder} 2>/dev/null || {_refused('no such folder')}; "
               f"({lookup}command -v {program}) >/dev/null 2>&1 || {_refused('not found')}; "
               f"exec {'env ' + assignments if assignments else ''}{shlex.join(argv)}")
    process = await _spawn("shell", command)
    stdout, stderr = _Capped(output_bytes), _Capped(output_bytes + len(STARTED) + ADB_PREAMBLE_BYTES)
    readers = asyncio.gather(stdout.read(process.stdout), stderr.read(process.stderr))
    timed_out = False
    try:
        await asyncio.wait_for(process.wait(), timeout_s)
    except BaseException as e:
        await _stopped(process)
        if not isinstance(e, asyncio.TimeoutError):
            readers.cancel()
            raise
        timed_out = True
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(readers, DRAIN_S)
    start = stderr.kept.find(STARTED, 0, ADB_PREAMBLE_BYTES + len(STARTED))
    if start < 0 and not timed_out:
        raise DeviceUnavailable(f"adb didn't start {argv[0]!r} on the device: "
                                f"{stderr.kept.decode(errors='replace').strip()[:SHOWN_BYTES]}")
    said = bytes(stderr.kept[start + len(STARTED):] if start >= 0 else stderr.kept)
    if process.returncode == NOT_FOUND_EXIT and said.startswith(CANNOT_RUN.encode()):
        why = said.decode(errors="replace").removeprefix(CANNOT_RUN).strip()
        raise InvalidArgument(f"cannot run {argv[0]!r} in {cwd or '~'!r}: {why} on the device")
    return ExecResult(exit_code=None if timed_out else process.returncode, stdout=bytes(stdout.kept),
                      stderr=said[:output_bytes], timed_out=timed_out,
                      truncated=stdout.cut or stderr.cut or len(said) > output_bytes)


async def write_files(files: dict[str, bytes]) -> None:
    """Each file is pushed beside its target and renamed over it, so it lands whole or not at all, then announced to
    the media scanner so the gallery and file apps list it."""
    for relative, data in files.items():
        target = HOME / relative
        partial = target.parent / f".agentenv-{uuid.uuid4().hex[:PARTIAL_ID_CHARS]}.part"
        with tempfile.TemporaryDirectory() as scratch:
            local = Path(scratch) / target.name
            local.write_bytes(data)
            made = await _adb("shell", f"mkdir -p {shlex.quote(str(target.parent))} && "
                                       f"if [ -d {shlex.quote(str(target))} ]; then echo folder; fi")
            if made.strip() == b"folder":
                raise InvalidArgument(f"{relative!r} is a folder on the device, not a file")
            try:
                await _adb("push", str(local), str(partial), timeout_s=TRANSFER_TIMEOUT_S)
                await _adb("shell", f"mv -f {shlex.quote(str(partial))} {shlex.quote(str(target))}")
            except BaseException:
                with contextlib.suppress(Exception):
                    await _adb("shell", f"rm -f {shlex.quote(str(partial))}")
                raise
        await _adb("shell", f"am broadcast -a {MEDIA_SCAN} -d {shlex.quote('file://' + str(target))}")


async def read_files(paths: list[str], into: Path) -> None:
    """The device answers whether each path exists, so a missing one is told apart from adb failing."""
    for relative in paths:
        remote = str(HOME / relative)
        answer = (await _adb("shell", f"if [ -e {shlex.quote(remote)} ]; then echo yes; else echo no; fi")).strip()
        if answer == b"no":
            continue
        if answer != b"yes":
            raise DeliveryFailed(f"the device didn't say whether {relative!r} exists: {answer[:SHOWN_BYTES]!r}")
        (into / relative).parent.mkdir(parents=True, exist_ok=True)
        await _adb("pull", remote, str((into / relative).parent), timeout_s=TRANSFER_TIMEOUT_S)


def _refused(why: str) -> str:
    return f"{{ echo '{CANNOT_RUN} {why}' >&2; exit {NOT_FOUND_EXIT}; }}"


async def _spawn(*args: str) -> asyncio.subprocess.Process:
    adb = os.environ.get("AGENTENV_ADB") or "adb"
    try:
        return await asyncio.create_subprocess_exec(adb, *args, stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError:
        raise DeviceUnavailable(f"{adb} is not installed") from None


async def _adb(*args: str, timeout_s: float = ADB_TIMEOUT_S) -> bytes:
    process = await _spawn(*args)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
    except BaseException as e:
        await _stopped(process)
        if isinstance(e, asyncio.TimeoutError):
            raise DeliveryFailed(f"adb {args[0]} gave no answer in {timeout_s:g}s") from None
        raise
    if process.returncode != 0:
        raise DeliveryFailed(f"adb {args[0]}: {(stderr or stdout).decode(errors='replace').strip()}")
    return stdout


async def _stopped(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    await process.wait()
