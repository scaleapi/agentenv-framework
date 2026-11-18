"""What a computer app's own server does in the desktop session on Linux and macOS (``session``): wait for the X
session, or on Hyprland for its Wayland and IPC sockets, launch the app into a session of its own and stop or pause it
by its process group, add its freedesktop menu launcher and list the session's windows, X's with ``xprop`` or
Hyprland's with ``hyprctl``."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Iterator, Optional

from . import _reaping

#: How often a starting session is looked for.
POLL_S = 0.2
#: Moves the fds an app is launched with into place, run on its own so it loads nothing but the standard library.
WITH_FDS = Path(__file__).with_name("_with_fds.py")


async def wait_until_up() -> None:
    """Until the session's sockets are there: on Hyprland its Wayland display's and its IPC's, else the X display's."""
    if signature := _hyprland():
        runtime = Path(os.environ["XDG_RUNTIME_DIR"])
        sockets = [runtime / os.environ["WAYLAND_DISPLAY"], runtime / "hypr" / signature / ".socket.sock"]
    else:
        sockets = [Path(f"/tmp/.X11-unix/X{os.environ['DISPLAY'].lstrip(':').split('.')[0]}")]
    while not all(socket.exists() for socket in sockets):
        await asyncio.sleep(POLL_S)


async def launch(argv: list[str], cwd: Path, env: dict[str, str], fds: Optional[dict[int, int]] = None) -> int:
    """Start ``argv`` in ``cwd``, in a session of its own so it outlives the server, with the server's environment and
    ``env``, and ``fds``, the server's fds it inherits by the number each gets in it; its pid."""
    if fds:
        argv = [sys.executable, "-I", "-S", str(WITH_FDS), *(f"{source}={target}" for target, source in fds.items()),
                "--", *argv]
    process = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env={**os.environ, **env}, start_new_session=True, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, pass_fds=tuple((fds or {}).values()))
    _reaping.reap(process)
    return process.pid


async def stop(pid: int, timeout_s: float) -> None:
    """End the session ``launch`` started as ``pid``, every process in it: asked to exit, a paused one continued so it
    can, then killed after ``timeout_s``. No one asks how it exited once it's stopped."""
    _reaping.forget(pid)
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):  # macOS refuses a group of zombies alone
        os.killpg(pid, signal.SIGCONT)
    try:
        await asyncio.wait_for(_exited(pid), timeout_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)


@contextlib.contextmanager
def paused(pid: int) -> Iterator[None]:
    """Every process of the session ``launch`` started as ``pid`` stopped while the block runs, and continued after."""
    try:
        os.killpg(pid, signal.SIGSTOP)
    except (ProcessLookupError, PermissionError):
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGCONT)


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
    read is left out. On Hyprland, its mapped clients as its IPC lists them, native Wayland and XWayland alike: its
    ``class``, a Wayland app ID or an X window's WM_CLASS class, and its ``title``, and as its ``instance``, which a
    Wayland window hasn't, its program's file name, as on Windows."""
    if _hyprland():
        return await _clients()
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


def _hyprland() -> str | None:
    """The Hyprland instance whose session this is, by its signature; None outside one."""
    return os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or None


async def _clients() -> list[dict[str, str]]:
    """Hyprland's mapped clients, by their program, class and title; none when its IPC doesn't answer."""
    try:
        clients = json.loads(await _output("hyprctl", "-j", "clients") or "[]")
    except ValueError:
        return []
    return [{"instance": _program(client.get("pid")), "class": client.get("class") or "",
             "title": client.get("title") or ""}
            for client in clients if isinstance(client, dict) and client.get("mapped")]


def _program(pid: object) -> str:
    """The file name of the program process ``pid`` runs, or "" when it can't be read."""
    try:
        return Path(os.readlink(f"/proc/{int(pid)}/exe")).name
    except (OSError, TypeError, ValueError):
        return ""


async def _xprop(*args: str) -> str:
    return await _output("xprop", *args)


async def _output(*argv: str) -> str:
    """What ``argv`` prints, or nothing when it fails; killed when the listing it's part of is cut short."""
    process = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        raise
    return out.decode(errors="replace") if process.returncode == 0 else ""
