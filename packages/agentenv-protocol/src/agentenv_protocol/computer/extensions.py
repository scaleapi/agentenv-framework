"""The framework extensions agentenv calls, never agents: readiness, display, coordinate spaces, keyboard settings
and commands in the session."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Optional

from ..agent_env_environment import extension
from .errors import Internal, InvalidArgument, Unsupported
from .operations import Display, ExecResult
from .spaces import check_space

COMPUTER = "urn:agentenv:computer/v1"
COMPUTER_READY = "urn:agentenv:computer-ready/v1"
COMPUTER_DISPLAY = "urn:agentenv:computer-display/v1"
COMPUTER_SPACES = "urn:agentenv:computer-spaces/v1"
COMPUTER_POINTER = "urn:agentenv:computer-pointer/v1"
COMPUTER_KEYBOARD = "urn:agentenv:computer-keyboard/v1"
COMPUTER_EXEC = "urn:agentenv:computer-exec/v1"

MAX_EXEC_TIMEOUT_S = 3600
#: How many command results are kept for a repeated request_id.
KEEP_EXEC_RESULTS = 256
KEYBOARD_RANGES = {"tap_hold_ms": (0, 1000), "type_delay_ms": (0, 1000), "max_hold_ms": (0, 600_000)}


class Extensions:
    """The extension handlers, mixed into ComputerDriverEnvironment; one whose operation the driver lacks is
    withdrawn when the driver is built."""

    @extension(COMPUTER_READY, op="status", method="GET",
               description="Whether the computer is usable now, by real checks; never sends input.")
    async def _computer_ready_status(self) -> dict:
        checks = await self._computer_ready_checks()
        return {"ready": all(c.ok for c in checks),
                "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks]}

    @extension(COMPUTER_DISPLAY, op="get", method="GET", description="Get or set the display's resolution and scale.")
    async def _computer_display_get(self) -> dict:
        return self._computer_display_data(await self._computer_display_now())

    @extension(COMPUTER_DISPLAY, op="set")
    async def _computer_display_set(self, width: int, height: int, scale: Optional[float] = None) -> dict:
        if [width, height] not in [list(m) for m in self.display_modes]:
            modes = ", ".join(f"{w}×{h}" for w, h in self.display_modes)
            raise InvalidArgument(f"{width}×{height} is not one of this computer's display modes: {modes}")
        now = await self._computer_display_now()
        changed = await self._computer_call("set_display", width, height, now.scale if scale is None else scale)
        if not isinstance(changed, Display):
            changed = await self._computer_display_now()
        return self._computer_display_data(changed)

    @extension(COMPUTER_SPACES, op="get", method="GET",
               description="Get or set the default coordinate space, written into the tool descriptions.")
    async def _computer_spaces_get(self) -> dict:
        return {"default_space": self.settings.default_space, "default_norm_scale": self.settings.default_norm_scale}

    @extension(COMPUTER_SPACES, op="set")
    async def _computer_spaces_set(self, default_space: str, default_norm_scale: Optional[int] = None) -> dict:
        space, scale = check_space(default_space, default_norm_scale, self.settings.default_space,
                                   self.settings.default_norm_scale)
        self.settings = self.settings.model_copy(update={"default_space": space, "default_norm_scale": scale})
        self._computer_describe_tools()
        return await self._computer_spaces_get()

    @extension(COMPUTER_KEYBOARD, op="get", method="GET",
               description="Get or set the tap hold, typing delay, layout and maximum hold.")
    async def _computer_keyboard_get(self) -> dict:
        s = self.settings
        return {"tap_hold_ms": s.tap_hold_ms, "type_delay_ms": s.type_delay_ms, "layout": s.layout,
                "max_hold_ms": s.max_hold_ms}

    @extension(COMPUTER_KEYBOARD, op="set")
    async def _computer_keyboard_set(self, tap_hold_ms: Optional[int] = None, type_delay_ms: Optional[int] = None,
                            layout: Optional[str] = None, max_hold_ms: Optional[int] = None) -> dict:
        given = {"tap_hold_ms": tap_hold_ms, "type_delay_ms": type_delay_ms, "max_hold_ms": max_hold_ms}
        for name, value in given.items():
            low, high = KEYBOARD_RANGES[name]
            if value is not None and not low <= value <= high:
                raise InvalidArgument(f"{name} is from {low} to {high}, got {value}")
        if layout is not None and layout not in self.layouts:
            raise Unsupported(f"this computer types in the layouts {list(self.layouts)}, not {layout!r}")
        update = {k: v for k, v in {**given, "layout": layout}.items() if v is not None}
        self.settings = self.settings.model_copy(update=update)
        return await self._computer_keyboard_get()

    @extension(COMPUTER_EXEC, op="run",
               description="Run a command inside the desktop session as the session user; a repeated request_id "
                           "returns the first result.")
    async def _computer_exec_run(self, argv: list, request_id: str, cwd: Optional[str] = None, env: Optional[dict] = None,
                        timeout_s: Optional[float] = None) -> dict:
        if not (isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv)):
            raise InvalidArgument("argv is a non-empty list of strings; name a shell to use one: ['bash', '-lc', '...']")
        if not (isinstance(request_id, str) and request_id):
            raise InvalidArgument("request_id is a non-empty string")
        if env is not None and not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise InvalidArgument("env maps names to string values")
        timeout = self.settings.exec_timeout_s if timeout_s is None else timeout_s
        if not 0 < timeout <= MAX_EXEC_TIMEOUT_S:
            raise InvalidArgument(f"timeout_s is above 0 and at most {MAX_EXEC_TIMEOUT_S}, got {timeout}")
        runs: OrderedDict = self._computer_exec_runs
        task = runs.get(request_id)
        if task is None:
            task = asyncio.ensure_future(self._computer_run_command(argv, cwd, env, timeout))
            runs[request_id] = task

            def forget_a_failure(done: asyncio.Task) -> None:
                if done.cancelled() or done.exception() is not None:
                    runs.pop(request_id, None)

            task.add_done_callback(forget_a_failure)
            while len(runs) > KEEP_EXEC_RESULTS:
                runs.popitem(last=False)
        return await asyncio.shield(task)

    async def _computer_run_command(self, argv: list, cwd: Optional[str], env: Optional[dict], timeout: float) -> dict:
        result = await self._computer_call("run_in_session", argv, cwd, env, timeout)
        if not isinstance(result, ExecResult):
            raise Internal(f"@run_in_session returned {type(result).__name__}, not an ExecResult")
        cap = self.settings.exec_output_bytes
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout[:cap].decode("utf-8", errors="replace"),
            "stderr": result.stderr[:cap].decode("utf-8", errors="replace"),
            "truncated": result.truncated or len(result.stdout) > cap or len(result.stderr) > cap,
            "timed_out": result.timed_out,
        }

    def _computer_display_data(self, display: Display) -> dict:
        return {"width": display.width, "height": display.height, "scale": display.scale,
                "modes": [list(m) for m in self.display_modes]}
