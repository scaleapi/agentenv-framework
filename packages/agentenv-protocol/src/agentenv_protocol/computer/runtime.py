"""What ComputerDriverEnvironment keeps for one driver, and the steps its tools and extensions share: calling the
driver's operations, frames, the input queue, readiness, held keys and command runs. A driver reaches it as
``self._computer``; it is the base class's own, so a driver's members can't replace any of it."""
from __future__ import annotations

import asyncio
import contextlib
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from mcp.types import CallToolResult

from ..errors import AgentEnvError
from ..tool_context import ToolContext
from . import apps
from .errors import DeliveryFailed, DeviceUnavailable, InputBusy, Internal
from .frames import Frame, FrameStore, to_pil
from .operations import GROUPS, INPUT, Display, ReadyCheck, resolve
from .queue import InputQueue
from .recording import RecordHints, Recordings
from .results import EffectData, InputResult, send

if TYPE_CHECKING:
    from .driver import ComputerDriverEnvironment

#: How many finished command results are kept for a repeated request_id; a running command is always joined.
KEEP_EXEC_RESULTS = 256


class ComputerRuntime:
    def __init__(self, driver: ComputerDriverEnvironment) -> None:
        self.driver = driver
        self.operations = resolve(type(driver), driver.platform)
        self.frames = FrameStore()
        self.queue = InputQueue()
        self.ready = False
        self.held: dict[str, list[str]] = {}
        self._ready_lock = asyncio.Lock()
        self._releases: dict[str, asyncio.Task] = {}
        self._commands_running: dict[str, asyncio.Task] = {}
        self._commands_finished: OrderedDict = OrderedDict()
        self.recordings = Recordings(self)
        self.apps = apps.expected()

    async def app_paths(self) -> list[str]:
        """The paths under the session user's home that the computer's apps keep their data in."""
        return await apps.owned_paths(self.apps) if self.apps else []

    def has(self, capability: str) -> bool:
        if capability in GROUPS:
            return all(op in self.operations for op in GROUPS[capability])
        return capability in self.operations

    async def call(self, operation: str, *args: Any) -> Any:
        """Run one of the driver's operations: an AgentEnvError passes through, anything else is an input that may
        have been delivered (delivery_failed) or an internal failure."""
        method = getattr(self.driver, self.operations[operation])
        try:
            result = method(*args)
            return await result if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future) else result
        except AgentEnvError:
            raise
        except Exception as e:
            error = DeliveryFailed if operation in INPUT else Internal
            raise error(f"@{operation} failed: {type(e).__name__}: {e}") from e

    def stream(self, hints: RecordHints) -> AsyncIterator:
        """The driver's ``@record`` stream for one recording."""
        return getattr(self.driver, self.operations["record"])(hints)

    async def display_now(self) -> Display:
        display = await self.call("display")
        if not isinstance(display, Display) or display.width <= 0 or display.height <= 0 or display.scale <= 0:
            raise Internal(f"@display returned {display!r}, not a Display with a positive size and scale")
        return display

    async def capture(self, identity: str) -> Frame:
        """A new frame of the screen. The input count is taken before the capture, so input that lands while it runs
        counts as after the frame, which the frame may not show."""
        inputs = self.queue.snapshot()
        display = await self.display_now()
        image = to_pil(await self.call("capture"), display)
        return self.frames.add(image, display, inputs, identity=identity)

    async def pointer(self) -> tuple[int, int]:
        where = await self.call("pointer_position")
        try:
            x, y = where
            return int(x), int(y)
        except (TypeError, ValueError):
            raise Internal(f"@pointer_position returned {where!r}, not (x, y)") from None

    async def act(self, ctx: ToolContext, operation: str, args: tuple, *, frame: Frame | None, screenshot: bool,
                  summary: str, mapped: bool = False, at_pointer: bool = False, repeat: int = 1,
                  hold: tuple | None = None, move_to: tuple | None = None, then: Callable | None = None) -> CallToolResult:
        """One input call: its turn in the queue, the operation (after a move, repeated or held), then the result
        with others' input since ``frame`` and, when asked, a fresh frame. Whatever the call's turn can change is read
        once it has the turn: a point ``mapped`` from ``frame`` is checked again, since the display may have changed
        while it waited, and ``at_pointer`` reads the pointer and puts it before ``args``. ``then`` runs as soon as
        the input is delivered."""
        identity = self.driver.caller_identity(ctx)
        sent = False
        async with self.queue.turn(self.driver.settings.queue_wait_s):
            if mapped and frame is not None:
                self.frames.check_current(frame, await self.display_now())
            if at_pointer:
                sx, sy = await self.pointer()
                args, summary = (sx, sy, *args), f"{summary} at the pointer, screen ({sx}, {sy})"
            others = self.queue.others_since(frame.inputs, identity) if frame is not None else 0
            effect = None
            try:
                if move_to is not None:
                    await self.call("pointer_move", *move_to)
                    sent = True
                for _ in range(repeat):
                    effect = await self.call(operation, *args)
                    sent = True
                if hold is not None:
                    await self._hold_for(*hold)
            except AgentEnvError as e:
                if sent:
                    e.applied = "unknown"
                raise
            finally:
                if sent:
                    self.queue.delivered(identity)
            if then is not None:
                then()
        result = InputResult(summary=summary, applied="yes", delivery="foreground", others_since_frame=others,
                             effect=EffectData.of(effect))
        if not screenshot:
            return send(result)
        try:
            new = await self.capture(identity)
        except AgentEnvError as e:
            result.summary = f"{summary}; the screenshot after it failed ({e.code}: {e.message})"
            return send(result)
        result.summary, result.frame = f"{summary}, then captured {new.id}", new.result()
        return send(result, new.encode())

    async def _hold_for(self, ms: int, keys: list[str]) -> None:
        """Keep ``keys`` down for ``ms``, then release them, also when the call is cancelled meanwhile."""
        release = ("key_up", list(reversed(keys)))
        try:
            await asyncio.sleep(ms / 1000)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await asyncio.shield(self.call(*release))
            raise
        await self.call(*release)

    async def ensure_ready(self) -> None:
        """Refuse with device_unavailable until the ready checks have passed once; run them if none has yet."""
        if self.ready:
            return
        async with self._ready_lock:
            if self.ready:
                return
            failed = [c for c in await self.ready_checks() if not c.ok]
        if failed:
            raise DeviceUnavailable("the computer isn't ready: " + "; ".join(f"{c.name}: {c.detail}" for c in failed))

    async def ready_checks(self) -> list[ReadyCheck]:
        checks = []
        try:
            display = await self.display_now()
            checks.append(ReadyCheck("display", True, f"{display.width}×{display.height} at scale {display.scale:g}"))
        except AgentEnvError as e:
            display = None
            checks.append(ReadyCheck("display", False, e.message))
        if display is None:
            checks.append(ReadyCheck("capture", False, "the display didn't answer"))
        else:
            try:
                image = to_pil(await self.call("capture"), display)
                checks.append(ReadyCheck("capture", True, f"{image.size[0]}×{image.size[1]}"))
            except AgentEnvError as e:
                checks.append(ReadyCheck("capture", False, e.message))
        try:
            checks += list(await self.driver.ready_checks())
        except Exception as e:
            checks.append(ReadyCheck("driver", False, f"{type(e).__name__}: {e}"))
        checks += await asyncio.gather(*(apps.check(folder) for folder in self.apps))
        if all(c.ok for c in checks):
            self.ready = True
        return checks

    def hold(self, identity: str, keys: list[str]) -> None:
        held = self.held.setdefault(identity, [])
        held += [k for k in keys if k not in held]
        previous = self._releases.pop(identity, None)
        if previous is not None:
            previous.cancel()
        self._releases[identity] = asyncio.ensure_future(
            self._release_after(identity, self.driver.settings.max_hold_ms))

    def release(self, identity: str, keys: list[str]) -> None:
        held = [k for k in self.held.get(identity, []) if k not in keys]
        if held:
            self.held[identity] = held
            return
        self.held.pop(identity, None)
        release = self._releases.pop(identity, None)
        if release is not None and release is not asyncio.current_task():
            release.cancel()

    async def _release_after(self, identity: str, ms: int) -> None:
        """Release what ``identity`` still holds once ``max_hold_ms`` has passed, waiting as long as other input
        keeps the queue busy."""
        await asyncio.sleep(ms / 1000)
        while keys := list(reversed(self.held.get(identity, []))):
            try:
                async with self.queue.turn(self.driver.settings.queue_wait_s):
                    await self.call("key_up", keys)
            except InputBusy:
                continue
            self.release(identity, keys)

    async def command(self, request_id: str, start: Callable[[], Awaitable[dict]]) -> dict:
        """The result of the command ``request_id`` names: a running one is joined, one of the last
        ``KEEP_EXEC_RESULTS`` finished is answered again, and anything else is started."""
        finished, running = self._commands_finished, self._commands_running
        if request_id in finished:
            return finished[request_id]
        task = running.get(request_id)
        if task is None:
            task = asyncio.ensure_future(start())
            running[request_id] = task

            def done(ran: asyncio.Task) -> None:
                if running.get(request_id) is ran:
                    del running[request_id]
                if not ran.cancelled() and ran.exception() is None:
                    finished[request_id] = ran.result()
                    while len(finished) > KEEP_EXEC_RESULTS:
                        finished.popitem(last=False)

            task.add_done_callback(done)
        return await asyncio.shield(task)
