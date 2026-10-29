"""The apps composed onto a computer, as its driver finds them, and whether each is ready. The composer writes each
app's ``.agentenv/app.json`` into its folder under ``APPS``. An app that starts with the computer counts toward the
computer's readiness: a ``boot`` app, or one with no launch whose ``@ready_check`` methods watch its services."""
from __future__ import annotations

import asyncio
import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .app import Launch, Ready
from .operations import ReadyCheck

if TYPE_CHECKING:
    from .runtime import ComputerRuntime

#: Where a composed computer installs its apps, a folder each.
APPS = Path("/opt/agentenv/apps")
#: An app's description of itself for the driver, in its folder.
APP_FILE = ".agentenv/app.json"
#: How long a readiness command or ``@ready_check`` may run.
CHECK_TIMEOUT_S = 10.0
#: How long a port gets to answer.
CONNECT_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class InstalledApp:
    name: str
    folder: Path
    module: str
    launch: Optional[Launch]
    ready_checks: tuple[str, ...]

    @property
    def counted(self) -> bool:
        """Whether the computer is ready only once this app is, because it starts with the computer."""
        return self.launch.at == "boot" if self.launch is not None else bool(self.ready_checks)


def installed() -> list[InstalledApp]:
    """The apps composed onto this computer, by their folders' names."""
    found = []
    for path in sorted(APPS.glob(f"*/{APP_FILE}")):
        data = json.loads(path.read_text())
        found.append(InstalledApp(name=data["name"], folder=path.parent.parent, module=data["module"],
                                  launch=Launch.model_validate(data["launch"]) if data.get("launch") else None,
                                  ready_checks=tuple(data.get("ready_checks", ()))))
    return found


async def check(runtime: ComputerRuntime, app: InstalledApp, since: float) -> ReadyCheck:
    """Whether ``app`` is ready, by its launch's ``ready`` and its ``@ready_check`` methods; once past its timeout, the
    check says so with what it still waits on."""
    ready = app.launch.ready if app.launch is not None else None
    problems = []
    if ready is not None and ready.window:
        problems += await _window(runtime, app.launch.window)
    if ready is not None and ready.port is not None:
        problems += await _port(ready.port)
    if ready is not None and ready.command:
        problems += await _run(runtime, app, ready.command, shlex.join(ready.command))
    for method in app.ready_checks:
        file, _, name = app.module.partition(":")
        argv = [str(app.folder / "venv/bin/python"), "-m", "agentenv_protocol.computer._ready", str(app.folder), file,
                name, method]
        problems += await _run(runtime, app, argv, f"@ready_check {method}")
    if not problems:
        return ReadyCheck(f"app {app.name}", True, "ready")
    waited = time.monotonic() - since
    limit = ready.timeout_s if ready is not None else Ready.model_fields["timeout_s"].default
    when = f"not ready after {limit:.0f} s" if waited > limit else f"not ready yet, {waited:.0f} s in"
    return ReadyCheck(f"app {app.name}", False, f"{when}: {'; '.join(problems)}")


async def _window(runtime: ComputerRuntime, match: dict[str, str]) -> list[str]:
    if not runtime.has("list_windows"):
        return ["this driver can't list windows"]
    windows = await runtime.call("list_windows")
    if any(all(window.get(key) == value for key, value in match.items()) for window in windows):
        return []
    return [f"no window with {', '.join(f'{key} {value!r}' for key, value in match.items())}"]


async def _port(port: int) -> list[str]:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), CONNECT_TIMEOUT_S)
    except (OSError, TimeoutError):
        return [f"nothing answers on port {port}"]
    writer.close()
    return []


async def _run(runtime: ComputerRuntime, app: InstalledApp, argv: list[str], what: str) -> list[str]:
    if not runtime.has("run_in_session"):
        return [f"this driver can't run {what}"]
    result = await runtime.call("run_in_session", argv, str(app.folder), {"ENVIRONMENT_NAME": app.name},
                                CHECK_TIMEOUT_S)
    if result.exit_code == 0:
        return []
    if result.timed_out:
        return [f"{what} ran past {CHECK_TIMEOUT_S:.0f} s"]
    reason = (result.stderr.decode(errors="replace").strip().splitlines() or [""])[-1]
    return [f"{what} exited {result.exit_code}" + (f": {reason}" if reason else "")]
