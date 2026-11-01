"""A computer app's own server at work: it launches the app into the desktop session, marking each launch for the rest
of the boot, stops the app it launched, and says whether the app is ready, its checks run at once, each within
``CHECK_TIMEOUT_S`` (``AppReadiness``). ``ComputerAppEnvironment`` keeps one as ``self._computer_app``."""
from __future__ import annotations

import asyncio
import inspect
import os
import shlex
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict

from . import session

if TYPE_CHECKING:
    from .app import Launch

#: Where an app's server marks that it launched the app since the computer booted, at boot or on demand.
LAUNCHED = Path("/run/agentenv/apps")
#: How long one readiness check may take; an app's checks run at once.
CHECK_TIMEOUT_S = 10.0
#: Where an app's server keeps the pid of the app it launched last, in the app's folder, so a server that restarts can
#: still stop it.
PID = Path(".agentenv/app.pid")
#: How long a stopped app may take to exit before it's killed.
STOP_TIMEOUT_S = 10.0


class AppReadiness(BaseModel):
    """What an app's server says of its readiness, ``computer-app/v1``'s ``ready``: the ``problems`` that keep it from
    being ``ready``, whether it was ``launched`` since the computer booted, and ``waited_s`` since then, or for an app
    with no launch since its server started, against its ``timeout_s``."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    problems: list[str]
    launched: bool
    waited_s: float
    timeout_s: float


class AppRuntime:
    """An app's server's state: its ``launch``, its ``@ready_check`` methods by name, bound to the app, how long it may
    take to be ready, why its last launch failed, and each sync check still running in its thread."""

    def __init__(self, launch: Optional[Launch], ready_checks: dict[str, Callable[[], Any]], timeout_s: float) -> None:
        self.launch = launch
        self.ready_checks = ready_checks
        self.timeout_s = timeout_s
        self.started = time.time()
        self.launch_failed: Optional[str] = None
        self.checking: dict[str, asyncio.Future] = {}

    async def start(self) -> None:
        """Once the desktop session is up, add the app's menu launcher and launch a boot app, unless the computer has
        launched it since it booted, at boot or on demand: a server that restarts leaves the app be."""
        if self.launch is None:
            return
        await session.wait_until_up()
        session.add_launcher(os.environ["ENVIRONMENT_NAME"], Path.cwd())
        if self.launch.at == "boot" and _launched_at() is None:
            try:
                await self.launch_app()
            except Exception as e:
                self.launch_failed = f"{type(e).__name__}: {e}"

    async def launch_app(self) -> int:
        """Start the app from its folder, marked as launched; its pid."""
        pid = await session.launch(self.launch.command, Path.cwd(), {"ENVIRONMENT_NAME": os.environ["ENVIRONMENT_NAME"]})
        self.launch_failed = None
        PID.parent.mkdir(parents=True, exist_ok=True)
        PID.write_text(str(pid))
        _mark().parent.mkdir(parents=True, exist_ok=True)
        _mark().touch()
        return pid

    async def stop_app(self) -> bool:
        """Stop the app this server launched last, with everything it started; whether any of it was running."""
        try:
            pid = int(PID.read_text())
        except (FileNotFoundError, ValueError):
            return False
        PID.unlink()
        if not session.alive(pid):
            return False
        await session.stop(pid, STOP_TIMEOUT_S)
        return True

    async def readiness(self) -> AppReadiness:
        ready = self.launch.ready if self.launch is not None else None
        checks: list[tuple[str, Awaitable[Optional[str]]]] = []
        if ready is not None and ready.window:
            checks.append(("window", _window(self.launch.window)))
        if ready is not None and ready.port is not None:
            checks.append(("port", _port(ready.port)))
        if ready is not None and ready.command:
            checks.append(("command", _command(ready.command)))
        checks += [(f"@ready_check {name}", self._ready_check(name, fn)) for name, fn in self.ready_checks.items()]
        found = await asyncio.gather(*(_bounded(name, check) for name, check in checks))
        problems = [problem for problem in found if problem is not None]
        launched = _launched_at()
        if self.launch is not None and launched is None:
            problems.insert(0, f"its launch failed: {self.launch_failed}" if self.launch_failed else "not launched yet")
        return AppReadiness(ready=not problems, problems=problems, launched=launched is not None,
                            waited_s=round(time.time() - (launched or self.started), 1), timeout_s=self.timeout_s)

    async def _ready_check(self, name: str, fn: Callable[[], Any]) -> Optional[str]:
        """A sync check runs in a thread, so it can't hold up the server, and once at a time: a thread can't be
        stopped, so a poll while one runs past its bound waits on that same call."""
        if inspect.iscoroutinefunction(fn):
            passed = await fn()
        else:
            running = self.checking.get(name)
            if running is None or running.done():
                running = self.checking[name] = asyncio.ensure_future(asyncio.to_thread(fn))
                running.add_done_callback(lambda done: done.cancelled() or done.exception())
            passed = await asyncio.shield(running)
        return None if passed else f"@ready_check {name} returned {passed!r}"


def _mark() -> Path:
    return LAUNCHED / Path.cwd().name


def _launched_at() -> Optional[float]:
    """When the app was last launched since the computer booted, by its mark."""
    try:
        return _mark().stat().st_mtime
    except FileNotFoundError:
        return None


async def _bounded(name: str, check: Awaitable[Optional[str]]) -> Optional[str]:
    """What ``check`` finds wrong, within ``CHECK_TIMEOUT_S``; a check that runs past it or raises says so."""
    try:
        return await asyncio.wait_for(check, CHECK_TIMEOUT_S)
    except TimeoutError:
        return f"{name} ran past {CHECK_TIMEOUT_S:g} s"
    except Exception as e:
        return f"{name}: {type(e).__name__}: {e}"


async def _window(match: dict[str, str]) -> Optional[str]:
    if any(all(window.get(key) == value for key, value in match.items()) for window in await session.windows()):
        return None
    return f"no window with {', '.join(f'{key} {value!r}' for key, value in match.items())}"


async def _port(port: int) -> Optional[str]:
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return f"nothing answers on port {port}"
    writer.close()
    return None


async def _command(argv: list[str]) -> Optional[str]:
    process = await asyncio.create_subprocess_exec(*argv, cwd=Path.cwd(), stdout=asyncio.subprocess.DEVNULL,
                                                   stderr=asyncio.subprocess.PIPE)
    try:
        _, err = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        raise
    if process.returncode == 0:
        return None
    reason = (err.decode(errors="replace").strip().splitlines() or [""])[-1]
    return f"{shlex.join(argv)} exited {process.returncode}" + (f": {reason}" if reason else "")
