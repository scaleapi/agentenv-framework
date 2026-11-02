"""What a computer app's own server does in the desktop session, the Linux way: wait for the session, launch the app
into it and stop it, add its menu launcher and list the session's windows. Another OS type brings its own."""
from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
from pathlib import Path

#: How often a starting session is looked for.
POLL_S = 0.2
#: The processes launched into the session, kept until they exit so each is reaped.
_running: set[asyncio.Task] = set()


async def wait_until_up() -> None:
    """Until the X display's socket is there."""
    display = os.environ["DISPLAY"].lstrip(":").split(".")[0]
    while not Path(f"/tmp/.X11-unix/X{display}").exists():
        await asyncio.sleep(POLL_S)


async def launch(argv: list[str], cwd: Path, env: dict[str, str]) -> int:
    """Start ``argv`` in ``cwd``, in a session of its own so it outlives the server, with the server's environment and
    ``env``; its pid."""
    process = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env={**os.environ, **env}, start_new_session=True, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    reaped = asyncio.ensure_future(process.wait())
    _running.add(reaped)
    reaped.add_done_callback(_running.discard)
    return process.pid


async def stop(pid: int, timeout_s: float) -> None:
    """End the session ``launch`` started as ``pid``, every process in it: asked to exit, then killed after
    ``timeout_s``."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(_exited(pid), timeout_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)


def alive(pid: int) -> bool:
    """Whether a process of the session ``launch`` started as ``pid`` is running, its first or one it started."""
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _exited(pid: int) -> None:
    while alive(pid):
        await asyncio.sleep(POLL_S)


def add_launcher(name: str, folder: Path) -> Path:
    """The desktop entry that launches the app ``name`` in ``folder`` through its own server, written where the
    session user's menu finds it."""
    entry = Path.home() / ".local/share/applications" / f"agentenv-app-{folder.name}.desktop"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(f"[Desktop Entry]\nType=Application\nName={name}\n"
                     f"Exec={folder}/venv/bin/python -m agentenv_protocol.computer.launch {folder}\nTerminal=false\n")
    return entry


async def windows() -> list[dict[str, str]]:
    """The window manager's top-level windows, each as its WM_CLASS ``instance`` and ``class`` and its ``title``, read
    with ``xprop``: ``_NET_WM_NAME``, or ``WM_NAME`` for a window that sets only that; a window that closes while it's
    read is left out."""
    listed = await _xprop("-root", "_NET_CLIENT_LIST")
    found = []
    for window in re.findall(r"0x[0-9a-f]+", listed.partition("#")[2]):
        props = await _xprop("-id", window, "WM_CLASS", "_NET_WM_NAME", "WM_NAME")
        names = dict(re.findall(r'^(WM_CLASS|_NET_WM_NAME|WM_NAME)[^=]*= (.*)$', props, re.MULTILINE))
        parts = _strings(names.get("WM_CLASS", ""))
        if len(parts) == 2:
            title = _strings(names.get("_NET_WM_NAME") or names.get("WM_NAME", ""))
            found.append({"instance": parts[0], "class": parts[1], "title": title[0] if title else ""})
    return found


def _strings(value: str) -> list[str]:
    """The quoted strings of an ``xprop`` value, as the window holds them, without the escapes ``xprop`` prints."""
    return [re.sub(r"\\(.)", r"\1", text) for text in re.findall(r'"((?:[^"\\]|\\.)*)"', value)]


async def _xprop(*args: str) -> str:
    process = await asyncio.create_subprocess_exec("xprop", *args, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        raise
    return out.decode(errors="replace") if process.returncode == 0 else ""
