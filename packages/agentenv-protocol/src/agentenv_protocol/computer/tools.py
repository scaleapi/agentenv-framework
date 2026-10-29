"""The seven computer_* tools and the pointer and keyboard groups' tools: fixed names, parameters and results, built
from the driver's operations. Every result is structured, with a one-line summary repeated as its text block."""
from __future__ import annotations

import asyncio
import unicodedata
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, Optional

from mcp.types import CallToolResult
from pydantic import Field

from ..agent_env_environment import tool
from ..tool_context import ToolContext
from .errors import InvalidArgument, Unsupported
from .frames import Frame
from .keys import chord
from .results import (
    InputResult,
    Point,
    PointerData,
    PointerListResult,
    PointerPositionResult,
    ScreenshotResult,
    WaitResult,
    send,
)
from .spaces import check_space, frame_point, frame_region

if TYPE_CHECKING:
    from .runtime import ComputerRuntime
    from .settings import ComputerSettings

_READ = {"readOnlyHint": True, "openWorldHint": False}
_INPUT = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}

Number = Annotated[float, Field(description="A coordinate in the chosen space")]
AtPointer = Annotated[Optional[float], Field(
    description="A coordinate in the chosen space. Required unless the computer declares computer-pointer/v1; then "
                "leave out x and y to act at the pointer")]
MoveFirst = Annotated[Optional[float], Field(description="Move here first")]
FrameId = Annotated[Optional[str], Field(description="The frame the point refers to; defaults to your latest")]
SpaceArg = Annotated[Optional[Literal["frame", "norm"]], Field(
    description="frame: pixels of the frame's image; norm: 0 to norm_scale over it")]
NormScale = Annotated[Optional[int], Field(description="The norm grid's size, for space=norm")]
Modifiers = Annotated[Optional[list[str]], Field(description="W3C key names held for this action, such as ['Shift']")]
Screenshot = Annotated[bool, Field(description="Return a fresh frame after acting")]
Button = Annotated[Optional[str], Field(description="Omitted, the primary press; others only where declared")]


class Tools:
    """The tools, mixed into ComputerDriverEnvironment."""

    _computer: ComputerRuntime
    settings: ComputerSettings
    buttons: tuple[str, ...]
    keys: Any
    max_click_count: int
    max_text_length: int
    max_wait_ms: int
    max_drag_points: int
    max_duration_ms: int
    max_key_repeat: int
    max_scroll_amount: int
    has: Callable[[str], bool]
    caller_identity: Callable[[ToolContext], str]

    @tool(annotations=_READ)
    async def computer_screenshot(
        self,
        ctx: ToolContext,
        region: Annotated[Optional[list[float]], Field(
            description="[x0, y0, x1, y1] within a frame: return that part at native resolution as a new frame")] = None,
        frame_id: Annotated[Optional[str], Field(description="The frame region refers to; defaults to your latest")] = None,
        format: Annotated[Literal["png", "jpeg"], Field(description="The image encoding")] = "png",
        quality: Annotated[Optional[int], Field(description="JPEG quality, 1 to 95")] = None,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, ScreenshotResult]:
        """Capture the screen as a frame: an image plus a frame_id you can point into later, the image size, and the
        display resolution and scale. With region, crop an existing frame at native resolution (zoom)."""
        computer = self._computer
        await computer.ensure_ready()
        identity = self.caller_identity(ctx)
        if quality is not None and not (format == "jpeg" and 1 <= quality <= 95):
            raise InvalidArgument("quality is for format=jpeg, from 1 to 95")
        if region is None:
            if frame_id is not None:
                raise InvalidArgument("frame_id names the frame a region crops; give a region too")
            frame = await computer.capture(identity)
            summary = f"Captured {frame.id}: {frame.width}×{frame.height} of the {_display(frame)} display"
        else:
            parent = computer.frames.get(frame_id, identity)
            computer.frames.check_current(parent, await computer.display_now())
            box = frame_region(parent, region, *_space(self, space, norm_scale))
            frame = computer.frames.add(parent.image.crop(box), parent.display, parent.inputs,
                                        identity=identity, parent=parent, origin=(box[0], box[1]))
            summary = f"Cropped {frame.id}: {frame.width}×{frame.height} of {parent.id} at ({box[0]}, {box[1]})"
        result = ScreenshotResult(summary=summary, **frame.result().model_dump())
        return send(result, frame.encode(format, quality))

    @tool(annotations=_INPUT)
    async def computer_click(
        self,
        ctx: ToolContext,
        x: AtPointer = None,
        y: AtPointer = None,
        frame_id: FrameId = None,
        button: Button = None,
        count: Annotated[int, Field(description="1 by default; double and triple clicks where declared")] = 1,
        modifiers: Modifiers = None,
        screenshot: Screenshot = False,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, InputResult]:
        """Click at a point of a frame."""
        await self._computer.ensure_ready()
        button = _button(self, button)
        _check_range("count", count, 1, self.max_click_count)
        held = _modifiers(self, modifiers)
        frame, screen, where, mapped = await _point(self, ctx, x, y, frame_id, space, norm_scale)
        times = f" ×{count}" if count > 1 else ""
        return await self._computer.act(ctx, "click", (*screen, button, count, held), frame=frame, mapped=mapped,
                                        at_pointer=not screen, screenshot=screenshot,
                                        summary=f"Clicked {button}{times}{where}")

    @tool(annotations=_INPUT)
    async def computer_drag(
        self,
        ctx: ToolContext,
        path: Annotated[list[list[float]], Field(description="Points [[x, y], ...]: at least a start and an end")],
        frame_id: FrameId = None,
        button: Button = None,
        duration_ms: Annotated[Optional[int], Field(description="How long the move takes")] = None,
        modifiers: Modifiers = None,
        screenshot: Screenshot = False,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, InputResult]:
        """Press at the first point, move through the path, and release at the last."""
        computer = self._computer
        await computer.ensure_ready()
        button = _button(self, button)
        if not isinstance(path, list) or not 2 <= len(path) <= self.max_drag_points:
            raise InvalidArgument(f"path has from 2 to {self.max_drag_points} points on this computer")
        duration = self.settings.drag_duration_ms if duration_ms is None else duration_ms
        _check_range("duration_ms", duration, 0, self.max_duration_ms)
        held = _modifiers(self, modifiers)
        frame = computer.frames.get(frame_id, self.caller_identity(ctx))
        computer.frames.check_current(frame, await computer.display_now())
        space, scale = _space(self, space, norm_scale)
        screen = []
        for point in path:
            if not (isinstance(point, list) and len(point) == 2):
                raise InvalidArgument("each path point is [x, y]")
            screen.append(frame.to_screen(*frame_point(frame, point[0], point[1], space, scale)))
        return await computer.act(ctx, "drag", (screen, button, duration, held), frame=frame, mapped=True,
                                  screenshot=screenshot,
                                  summary=f"Dragged {button} through {len(path)} points of {frame.id} in {duration} ms")

    @tool(annotations=_INPUT)
    async def computer_scroll(
        self,
        ctx: ToolContext,
        direction: Annotated[Literal["up", "down", "left", "right"], Field(
            description="Where the view moves through the content: down shows what is further down")],
        x: AtPointer = None,
        y: AtPointer = None,
        frame_id: FrameId = None,
        amount: Annotated[int, Field(description="How many steps (wheel notches)")] = 3,
        modifiers: Modifiers = None,
        screenshot: Screenshot = False,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, InputResult]:
        """Scroll the content under a point."""
        await self._computer.ensure_ready()
        _check_range("amount", amount, 1, self.max_scroll_amount)
        held = _modifiers(self, modifiers)
        frame, screen, where, mapped = await _point(self, ctx, x, y, frame_id, space, norm_scale)
        return await self._computer.act(ctx, "scroll", (*screen, direction, amount, held), frame=frame, mapped=mapped,
                                        at_pointer=not screen, screenshot=screenshot,
                                        summary=f"Scrolled {direction} {amount}{where}")

    @tool(annotations=_INPUT)
    async def computer_type(
        self,
        ctx: ToolContext,
        text: Annotated[str, Field(description="Any Unicode; \\n types Enter and \\t types Tab")],
        screenshot: Screenshot = False,
    ) -> Annotated[CallToolResult, InputResult]:
        """Type text into whatever has keyboard focus. Text that can't be typed whole is refused before anything is
        typed; the clipboard is left as it was."""
        computer = self._computer
        await computer.ensure_ready()
        if not text:
            raise InvalidArgument("text is empty")
        if len(text) > self.max_text_length:
            raise InvalidArgument(f"text is at most {self.max_text_length} characters on this computer, got {len(text)}")
        control = sorted({f"U+{ord(c):04X}" for c in text if c not in "\n\t" and unicodedata.category(c) == "Cc"})
        if control:
            raise InvalidArgument(f"text has control characters {control}; press keys with computer_key")
        return await computer.act(ctx, "type_text", (text,), frame=computer.frames.latest(self.caller_identity(ctx)),
                                  screenshot=screenshot, summary=f"Typed {len(text)} characters")

    @tool(annotations=_INPUT)
    async def computer_key(
        self,
        ctx: ToolContext,
        keys: Annotated[list[str], Field(
            description="One chord, W3C key names pressed in order and released in reverse: ['Control', 's'] or "
                        "['Enter']")],
        repeat: Annotated[int, Field(description="How many times to press the chord")] = 1,
        screenshot: Screenshot = False,
    ) -> Annotated[CallToolResult, InputResult]:
        """Press one key chord."""
        computer = self._computer
        await computer.ensure_ready()
        names = chord(keys, self.keys)
        _check_range("repeat", repeat, 1, self.max_key_repeat)
        times = f" ×{repeat}" if repeat > 1 else ""
        return await computer.act(ctx, "press", (names,), frame=computer.frames.latest(self.caller_identity(ctx)),
                                  screenshot=screenshot, repeat=repeat, summary=f"Pressed {'+'.join(names)}{times}")

    @tool(annotations=_READ)
    async def computer_wait(
        self,
        ctx: ToolContext,
        ms: Annotated[int, Field(description="How long to wait, in milliseconds")] = 1000,
        screenshot: Screenshot = False,
    ) -> Annotated[CallToolResult, WaitResult]:
        """Wait, then optionally return a fresh frame."""
        await self._computer.ensure_ready()
        _check_range("ms", ms, 0, self.max_wait_ms)
        await asyncio.sleep(ms / 1000)
        summary = f"Waited {ms} ms"
        if not screenshot:
            return send(WaitResult(summary=summary, waited_ms=ms))
        frame = await self._computer.capture(self.caller_identity(ctx))
        return send(WaitResult(summary=f"{summary}, then captured {frame.id}", waited_ms=ms, frame=frame.result()),
                    frame.encode())

    @tool(annotations=_INPUT)
    async def computer_pointer_move(
        self,
        ctx: ToolContext,
        x: Number,
        y: Number,
        frame_id: FrameId = None,
        screenshot: Screenshot = False,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, InputResult]:
        """Move the pointer to a point; hover effects happen."""
        await self._computer.ensure_ready()
        frame, screen, where, mapped = await _point(self, ctx, x, y, frame_id, space, norm_scale)
        return await self._computer.act(ctx, "pointer_move", screen, frame=frame, mapped=mapped,
                                        screenshot=screenshot, summary=f"Moved the pointer{where}")

    @tool(annotations=_INPUT)
    async def computer_pointer_down(
        self,
        ctx: ToolContext,
        button: Button = None,
        x: MoveFirst = None,
        y: MoveFirst = None,
        frame_id: FrameId = None,
        screenshot: Screenshot = False,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, InputResult]:
        """Press a button, optionally after moving to a point."""
        return await _button_action(self, ctx, "pointer_down", "Pressed", button, x, y, frame_id, screenshot, space,
                                    norm_scale)

    @tool(annotations=_INPUT)
    async def computer_pointer_up(
        self,
        ctx: ToolContext,
        button: Button = None,
        x: MoveFirst = None,
        y: MoveFirst = None,
        frame_id: FrameId = None,
        screenshot: Screenshot = False,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, InputResult]:
        """Release a button, optionally after moving to a point."""
        return await _button_action(self, ctx, "pointer_up", "Released", button, x, y, frame_id, screenshot, space,
                                    norm_scale)

    @tool(annotations=_READ)
    async def computer_pointer_position(
        self,
        ctx: ToolContext,
        frame_id: Annotated[Optional[str], Field(description="The frame to report in; defaults to your latest")] = None,
    ) -> Annotated[CallToolResult, PointerPositionResult]:
        """Where the pointer is, in a frame's pixels and in screen units, and whether it is inside that frame."""
        computer = self._computer
        await computer.ensure_ready()
        sx, sy = await computer.pointer()
        screen = Point(x=sx, y=sy)
        identity = self.caller_identity(ctx)
        frame = computer.frames.get(frame_id, identity) if frame_id is not None else computer.frames.latest(identity)
        if frame is None:
            return send(PointerPositionResult(summary=f"The pointer is at screen ({sx}, {sy})", screen=screen))
        fx, fy = frame.from_screen(sx, sy)
        inside = frame.contains(fx, fy)
        where = f"({fx}, {fy}) in {frame.id}" if inside else f"outside {frame.id}, at ({fx}, {fy}) in its pixels"
        return send(PointerPositionResult(summary=f"The pointer is {where}, screen ({sx}, {sy})", screen=screen,
                                          frame_id=frame.id, x=fx, y=fy, inside_frame=inside))

    @tool(annotations=_READ)
    async def computer_pointer_list(self) -> Annotated[CallToolResult, PointerListResult]:
        """The pointers you may use, and which one is your default."""
        pointer = PointerData(id="default", kind="foreground", label="The computer's pointer", default=True)
        return send(PointerListResult(summary="1 pointer: default (foreground)", pointers=[pointer]))

    @tool(annotations=_INPUT)
    async def computer_keyboard_down(
        self,
        ctx: ToolContext,
        keys: Annotated[list[str], Field(description="W3C key names, pressed in the order given")],
        hold_ms: Annotated[Optional[int], Field(
            description="Hold this long, then release in reverse order; without it they stay down until "
                        "computer_keyboard_up")] = None,
        screenshot: Screenshot = False,
    ) -> Annotated[CallToolResult, InputResult]:
        """Press keys and keep them down."""
        computer = self._computer
        await computer.ensure_ready()
        names = chord(keys, self.keys)
        if hold_ms is not None and not 0 <= hold_ms <= self.settings.max_hold_ms:
            raise InvalidArgument(f"hold_ms is from 0 to {self.settings.max_hold_ms}, got {hold_ms}")
        identity = self.caller_identity(ctx)
        frame = computer.frames.latest(identity)
        if hold_ms is not None:
            return await computer.act(ctx, "key_down", (names,), frame=frame, screenshot=screenshot,
                                      hold=(hold_ms, names), summary=f"Held {'+'.join(names)} for {hold_ms} ms")
        return await computer.act(ctx, "key_down", (names,), frame=frame, screenshot=screenshot,
                                  summary=f"Pressed and held {'+'.join(names)}",
                                  then=lambda: computer.hold(identity, names))

    @tool(annotations=_INPUT)
    async def computer_keyboard_up(
        self,
        ctx: ToolContext,
        keys: Annotated[Optional[list[str]], Field(
            description="The keys to release, in order; without it, every key you are holding")] = None,
        screenshot: Screenshot = False,
    ) -> Annotated[CallToolResult, InputResult]:
        """Release held keys."""
        computer = self._computer
        await computer.ensure_ready()
        identity = self.caller_identity(ctx)
        names = chord(keys, self.keys) if keys is not None else list(computer.held.get(identity, []))
        if not names:
            raise InvalidArgument("you are holding no keys")
        return await computer.act(ctx, "key_up", (names,), frame=computer.frames.latest(identity),
                                  screenshot=screenshot, summary=f"Released {'+'.join(names)}",
                                  then=lambda: computer.release(identity, names))


async def _button_action(driver: Tools, ctx: ToolContext, op: str, verb: str, button: str | None, x: float | None,
                         y: float | None, frame_id: str | None, screenshot: bool, space: str | None,
                         norm_scale: int | None) -> CallToolResult:
    computer = driver._computer
    await computer.ensure_ready()
    button = _button(driver, button)
    if (x is None) != (y is None):
        raise InvalidArgument("give both x and y, or neither")
    if x is None:
        return await computer.act(ctx, op, (button,), frame=computer.frames.latest(driver.caller_identity(ctx)),
                                  screenshot=screenshot, summary=f"{verb} {button}")
    frame, screen, where, mapped = await _point(driver, ctx, x, y, frame_id, space, norm_scale)
    return await computer.act(ctx, op, (button,), frame=frame, mapped=mapped, screenshot=screenshot,
                              move_to=screen, summary=f"{verb} {button}{where}")


async def _point(driver: Tools, ctx: ToolContext, x: float | None, y: float | None, frame_id: str | None,
                 space: str | None, norm_scale: int | None) -> tuple[Frame | None, tuple[int, ...], str, bool]:
    """The frame, the screen pixel, a phrase for the point and whether it was mapped from the frame. With no point
    and the pointer group, no pixel and no phrase: the call acts at the pointer, read once it has its input turn."""
    computer = driver._computer
    identity = driver.caller_identity(ctx)
    if x is None and y is None and driver.has("pointer") and frame_id is None:
        return computer.frames.latest(identity), (), "", False
    if x is None or y is None:
        raise InvalidArgument("give x and y" + (", or neither to act at the pointer" if driver.has("pointer") else ""))
    frame = computer.frames.get(frame_id, identity)
    computer.frames.check_current(frame, await computer.display_now())
    space, scale = _space(driver, space, norm_scale)
    fx, fy = frame_point(frame, x, y, space, scale)
    in_space = f" ({space} {scale})" if space == "norm" else ""
    return frame, frame.to_screen(fx, fy), f" at ({x:g}, {y:g}){in_space} in {frame.id}", True


def _space(driver: Tools, space: str | None, norm_scale: int | None) -> tuple[str, int]:
    return check_space(space, norm_scale, driver.settings.default_space, driver.settings.default_norm_scale)


def _button(driver: Tools, button: str | None) -> str:
    button = button or "left"
    if button not in driver.buttons:
        raise Unsupported(f"this computer has no {button} button; it has {', '.join(driver.buttons)}")
    return button


def _modifiers(driver: Tools, modifiers: list | None) -> list[str]:
    if not modifiers:
        return []
    if not isinstance(modifiers, list):
        raise InvalidArgument("modifiers is a list of W3C key names")
    return chord(modifiers, driver.keys)


def _check_range(name: str, value: int, low: int, high: int) -> None:
    if not low <= value <= high:
        raise InvalidArgument(f"{name} is from {low} to {high} on this computer, got {value}")


def _display(frame: Frame) -> str:
    d = frame.display
    return f"{d.width}×{d.height}" + (f" at scale {d.scale:g}" if d.scale != 1 else "")
