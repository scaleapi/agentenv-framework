"""The seven computer_* tools and the pointer and keyboard groups' tools: fixed names, parameters and results, built
from the driver's operations. Every result is structured, with a one-line summary repeated as its text block."""
from __future__ import annotations

import asyncio
import unicodedata
from typing import Annotated, Literal, Optional

from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel, Field

from ..agent_env_environment import tool
from ..tool_context import ToolContext
from .errors import InvalidArgument, Unsupported
from .frames import Frame
from .keys import chord
from .operations import Effect
from .spaces import check_space, frame_point, frame_region

_READ = {"readOnlyHint": True, "openWorldHint": False}
_INPUT = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}

Number = Annotated[float, Field(description="A coordinate in the chosen space")]
FrameId = Annotated[Optional[str], Field(description="The frame the point refers to; defaults to your latest")]
SpaceArg = Annotated[Optional[Literal["frame", "norm"]], Field(
    description="frame: pixels of the frame's image; norm: 0 to norm_scale over it")]
NormScale = Annotated[Optional[int], Field(description="The norm grid's size, for space=norm")]
Modifiers = Annotated[Optional[list[str]], Field(description="W3C key names held for this action, such as ['Shift']")]
Screenshot = Annotated[bool, Field(description="Return a fresh frame after acting")]
Button = Annotated[Optional[str], Field(description="Omitted, the primary press; others only where declared")]


class DisplayData(BaseModel):
    width: int
    height: int
    scale: float


class FrameData(BaseModel):
    frame_id: str
    width: int
    height: int
    display: DisplayData
    parent_frame_id: Optional[str] = None
    captured_at: str


class ScreenshotResult(FrameData):
    summary: str


class EffectData(BaseModel):
    focus_changed: Optional[bool] = None
    pointer_moved: Optional[bool] = None
    clipboard_used: Optional[bool] = None
    detail: Optional[str] = None


class InputResult(BaseModel):
    summary: str
    applied: Literal["yes", "no", "unknown"]
    delivery: Literal["foreground", "background"]
    others_since_frame: int
    effect: Optional[EffectData] = None
    frame: Optional[FrameData] = None


class WaitResult(BaseModel):
    summary: str
    waited_ms: int
    frame: Optional[FrameData] = None


class Point(BaseModel):
    x: int
    y: int


class PointerPositionResult(BaseModel):
    summary: str
    screen: Point
    frame_id: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    inside_frame: Optional[bool] = None


class PointerData(BaseModel):
    id: str
    kind: Literal["foreground", "background"]
    label: str
    default: bool


class PointerListResult(BaseModel):
    summary: str
    pointers: list[PointerData]


#: The tools a group serves, and the tools whose description names the default coordinate space.
GROUP_TOOLS = {
    "pointer": ("computer_pointer_move", "computer_pointer_down", "computer_pointer_up", "computer_pointer_position",
                "computer_pointer_list"),
    "keyboard": ("computer_keyboard_down", "computer_keyboard_up"),
}
POINT_TOOLS = ("computer_screenshot", "computer_click", "computer_drag", "computer_scroll", "computer_pointer_move",
               "computer_pointer_down", "computer_pointer_up")
POINT_OPTIONAL_WITH_POINTER = ("computer_click", "computer_scroll")


def _result(summary: str, data: dict, image: tuple[str, str] | None = None) -> CallToolResult:
    content: list = [TextContent(type="text", text=summary)]
    if image is not None:
        content.append(ImageContent(type="image", data=image[0], mimeType=image[1]))
    return CallToolResult(content=content, structuredContent={"summary": summary, **data})


def _fmt(x: float, y: float) -> str:
    return f"({x:g}, {y:g})"


class Tools:
    """The tools, mixed into ComputerDriverEnvironment."""

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
        await self._ensure_ready()
        identity = self.caller_identity(ctx)
        if quality is not None and not (format == "jpeg" and 1 <= quality <= 95):
            raise InvalidArgument("quality is for format=jpeg, from 1 to 95")
        if region is None:
            if frame_id is not None:
                raise InvalidArgument("frame_id names the frame a region crops; give a region too")
            frame = await self._capture(identity)
            summary = f"Captured {frame.id}: {frame.width}×{frame.height} of the {_display(frame)} display"
        else:
            parent = self._frames.get(frame_id, identity)
            self._frames.check_current(parent, await self._display_now())
            space, scale = check_space(space, norm_scale, self.settings.default_space, self.settings.default_norm_scale)
            box = frame_region(parent, region, space, scale)
            frame = self._frames.add(parent.image.crop(box), parent.display, parent.inputs, identity=identity,
                                     parent=parent, origin=(box[0], box[1]))
            summary = f"Cropped {frame.id}: {frame.width}×{frame.height} of {parent.id} at ({box[0]}, {box[1]})"
        return _result(summary, frame.data(), frame.encode(format, quality))

    @tool(annotations=_INPUT)
    async def computer_click(
        self,
        ctx: ToolContext,
        x: Annotated[Optional[float], Field(description="A coordinate in the chosen space")] = None,
        y: Annotated[Optional[float], Field(description="A coordinate in the chosen space")] = None,
        frame_id: FrameId = None,
        button: Button = None,
        count: Annotated[int, Field(description="1 by default; double and triple clicks where declared")] = 1,
        modifiers: Modifiers = None,
        screenshot: Screenshot = False,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, InputResult]:
        """Click at a point of a frame."""
        await self._ensure_ready()
        button = self._button(button)
        if not 1 <= count <= self.max_click_count:
            raise InvalidArgument(f"count is from 1 to {self.max_click_count} on this computer, got {count}")
        held = self._modifiers(modifiers)
        frame, (sx, sy), where = await self._point(ctx, x, y, frame_id, space, norm_scale)
        times = f" ×{count}" if count > 1 else ""
        return await self._act(ctx, "click", (sx, sy, button, count, held), frame=frame, screenshot=screenshot,
                               summary=f"Clicked {button}{times} {where}")

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
        await self._ensure_ready()
        button = self._button(button)
        if not isinstance(path, list) or not 2 <= len(path) <= self.max_drag_points:
            raise InvalidArgument(f"path has from 2 to {self.max_drag_points} points on this computer")
        duration = self.settings.drag_duration_ms if duration_ms is None else duration_ms
        if not 0 <= duration <= self.max_duration_ms:
            raise InvalidArgument(f"duration_ms is from 0 to {self.max_duration_ms} on this computer, got {duration}")
        held = self._modifiers(modifiers)
        identity = self.caller_identity(ctx)
        frame = self._frames.get(frame_id, identity)
        self._frames.check_current(frame, await self._display_now())
        space, scale = check_space(space, norm_scale, self.settings.default_space, self.settings.default_norm_scale)
        screen = []
        for point in path:
            if not (isinstance(point, list) and len(point) == 2):
                raise InvalidArgument("each path point is [x, y]")
            screen.append(frame.to_screen(*frame_point(frame, point[0], point[1], space, scale)))
        return await self._act(ctx, "drag", (screen, button, duration, held), frame=frame, screenshot=screenshot,
                               summary=f"Dragged {button} through {len(path)} points of {frame.id} in {duration} ms")

    @tool(annotations=_INPUT)
    async def computer_scroll(
        self,
        ctx: ToolContext,
        direction: Annotated[Literal["up", "down", "left", "right"], Field(
            description="Where the view moves through the content: down shows what is further down")],
        x: Annotated[Optional[float], Field(description="A coordinate in the chosen space")] = None,
        y: Annotated[Optional[float], Field(description="A coordinate in the chosen space")] = None,
        frame_id: FrameId = None,
        amount: Annotated[int, Field(description="How many steps (wheel notches)")] = 3,
        modifiers: Modifiers = None,
        screenshot: Screenshot = False,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, InputResult]:
        """Scroll the content under a point."""
        await self._ensure_ready()
        if not 1 <= amount <= self.max_scroll_amount:
            raise InvalidArgument(f"amount is from 1 to {self.max_scroll_amount} on this computer, got {amount}")
        held = self._modifiers(modifiers)
        frame, (sx, sy), where = await self._point(ctx, x, y, frame_id, space, norm_scale)
        return await self._act(ctx, "scroll", (sx, sy, direction, amount, held), frame=frame, screenshot=screenshot,
                               summary=f"Scrolled {direction} {amount} {where}")

    @tool(annotations=_INPUT)
    async def computer_type(
        self,
        ctx: ToolContext,
        text: Annotated[str, Field(description="Any Unicode; \\n types Enter and \\t types Tab")],
        screenshot: Screenshot = False,
    ) -> Annotated[CallToolResult, InputResult]:
        """Type text into whatever has keyboard focus. Text that can't be typed whole is refused before anything is
        typed; the clipboard is left as it was."""
        await self._ensure_ready()
        if not text:
            raise InvalidArgument("text is empty")
        if len(text) > self.max_text_length:
            raise InvalidArgument(f"text is at most {self.max_text_length} characters on this computer, got {len(text)}")
        control = sorted({f"U+{ord(c):04X}" for c in text if c not in "\n\t" and unicodedata.category(c) == "Cc"})
        if control:
            raise InvalidArgument(f"text has control characters {control}; press keys with computer_key")
        identity = self.caller_identity(ctx)
        return await self._act(ctx, "type_text", (text,), frame=self._frames.latest(identity), screenshot=screenshot,
                               summary=f"Typed {len(text)} characters")

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
        await self._ensure_ready()
        names = chord(keys, self.keys)
        if not 1 <= repeat <= self.max_key_repeat:
            raise InvalidArgument(f"repeat is from 1 to {self.max_key_repeat} on this computer, got {repeat}")
        identity = self.caller_identity(ctx)
        times = f" ×{repeat}" if repeat > 1 else ""
        return await self._act(ctx, "press", (names,), frame=self._frames.latest(identity), screenshot=screenshot,
                               repeat=repeat, summary=f"Pressed {'+'.join(names)}{times}")

    @tool(annotations=_READ)
    async def computer_wait(
        self,
        ctx: ToolContext,
        ms: Annotated[int, Field(description="How long to wait, in milliseconds")] = 1000,
        screenshot: Screenshot = False,
    ) -> Annotated[CallToolResult, WaitResult]:
        """Wait, then optionally return a fresh frame."""
        await self._ensure_ready()
        if not 0 <= ms <= self.max_wait_ms:
            raise InvalidArgument(f"ms is from 0 to {self.max_wait_ms} on this computer, got {ms}")
        await asyncio.sleep(ms / 1000)
        summary = f"Waited {ms} ms"
        if not screenshot:
            return _result(summary, {"waited_ms": ms})
        frame = await self._capture(self.caller_identity(ctx))
        return _result(f"{summary}, then captured {frame.id}", {"waited_ms": ms, "frame": frame.data()}, frame.encode())

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
        await self._ensure_ready()
        frame, (sx, sy), where = await self._point(ctx, x, y, frame_id, space, norm_scale)
        return await self._act(ctx, "pointer_move", (sx, sy), frame=frame, screenshot=screenshot,
                               summary=f"Moved the pointer {where}")

    @tool(annotations=_INPUT)
    async def computer_pointer_down(
        self,
        ctx: ToolContext,
        button: Button = None,
        x: Annotated[Optional[float], Field(description="Move here first")] = None,
        y: Annotated[Optional[float], Field(description="Move here first")] = None,
        frame_id: FrameId = None,
        screenshot: Screenshot = False,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, InputResult]:
        """Press a button, optionally after moving to a point."""
        return await self._button_action(ctx, "pointer_down", "Pressed", button, x, y, frame_id, screenshot, space,
                                         norm_scale)

    @tool(annotations=_INPUT)
    async def computer_pointer_up(
        self,
        ctx: ToolContext,
        button: Button = None,
        x: Annotated[Optional[float], Field(description="Move here first")] = None,
        y: Annotated[Optional[float], Field(description="Move here first")] = None,
        frame_id: FrameId = None,
        screenshot: Screenshot = False,
        space: SpaceArg = None,
        norm_scale: NormScale = None,
    ) -> Annotated[CallToolResult, InputResult]:
        """Release a button, optionally after moving to a point."""
        return await self._button_action(ctx, "pointer_up", "Released", button, x, y, frame_id, screenshot, space,
                                         norm_scale)

    @tool(annotations=_READ)
    async def computer_pointer_position(
        self,
        ctx: ToolContext,
        frame_id: Annotated[Optional[str], Field(description="The frame to report in; defaults to your latest")] = None,
    ) -> Annotated[CallToolResult, PointerPositionResult]:
        """Where the pointer is, in a frame's pixels and in screen units, and whether it is inside that frame."""
        await self._ensure_ready()
        sx, sy = await self._pointer_screen()
        identity = self.caller_identity(ctx)
        frame = self._frames.get(frame_id, identity) if frame_id is not None else self._frames.latest(identity)
        data: dict = {"screen": {"x": sx, "y": sy}}
        if frame is None:
            return _result(f"The pointer is at screen ({sx}, {sy})", data)
        fx, fy = frame.from_screen(sx, sy)
        inside = frame.contains(fx, fy)
        data.update(frame_id=frame.id, x=fx, y=fy, inside_frame=inside)
        where = f"({fx}, {fy}) in {frame.id}" if inside else f"outside {frame.id}, at ({fx}, {fy}) in its pixels"
        return _result(f"The pointer is {where}, screen ({sx}, {sy})", data)

    @tool(annotations=_READ)
    async def computer_pointer_list(self) -> Annotated[CallToolResult, PointerListResult]:
        """The pointers you may use, and which one is your default."""
        pointers = [{"id": "default", "kind": "foreground", "label": "The computer's pointer", "default": True}]
        return _result("1 pointer: default (foreground)", {"pointers": pointers})

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
        await self._ensure_ready()
        names = chord(keys, self.keys)
        if hold_ms is not None and not 0 <= hold_ms <= self.settings.max_hold_ms:
            raise InvalidArgument(f"hold_ms is from 0 to {self.settings.max_hold_ms}, got {hold_ms}")
        identity = self.caller_identity(ctx)
        if hold_ms is not None:
            return await self._act(ctx, "key_down", (names,), frame=self._frames.latest(identity), screenshot=screenshot,
                                   hold=(hold_ms, names), summary=f"Held {'+'.join(names)} for {hold_ms} ms")
        result = await self._act(ctx, "key_down", (names,), frame=self._frames.latest(identity), screenshot=screenshot,
                                 summary=f"Pressed and held {'+'.join(names)}")
        self._hold(identity, names)
        return result

    @tool(annotations=_INPUT)
    async def computer_keyboard_up(
        self,
        ctx: ToolContext,
        keys: Annotated[Optional[list[str]], Field(
            description="The keys to release, in order; without it, every key you are holding")] = None,
        screenshot: Screenshot = False,
    ) -> Annotated[CallToolResult, InputResult]:
        """Release held keys."""
        await self._ensure_ready()
        identity = self.caller_identity(ctx)
        names = chord(keys, self.keys) if keys is not None else list(self._held.get(identity, []))
        if not names:
            raise InvalidArgument("you are holding no keys")
        result = await self._act(ctx, "key_up", (names,), frame=self._frames.latest(identity), screenshot=screenshot,
                                 summary=f"Released {'+'.join(names)}")
        self._release(identity, names)
        return result

    async def _button_action(self, ctx, op, verb, button, x, y, frame_id, screenshot, space, norm_scale):
        await self._ensure_ready()
        button = self._button(button)
        if (x is None) != (y is None):
            raise InvalidArgument("give both x and y, or neither")
        if x is None:
            return await self._act(ctx, op, (button,), frame=self._frames.latest(self.caller_identity(ctx)),
                                   screenshot=screenshot, summary=f"{verb} {button}")
        frame, (sx, sy), where = await self._point(ctx, x, y, frame_id, space, norm_scale)
        return await self._act(ctx, op, (button,), frame=frame, screenshot=screenshot, move_to=(sx, sy),
                               summary=f"{verb} {button} {where}")

    async def _point(self, ctx, x, y, frame_id, space, norm_scale) -> tuple:
        """The frame, the screen pixel and a phrase for a point; with no point and the pointer group, the pointer."""
        identity = self.caller_identity(ctx)
        if x is None and y is None and self.has("pointer") and frame_id is None:
            sx, sy = await self._pointer_screen()
            return self._frames.latest(identity), (sx, sy), f"at the pointer, screen ({sx}, {sy})"
        if x is None or y is None:
            raise InvalidArgument("give x and y" + (", or neither to act at the pointer" if self.has("pointer") else ""))
        frame = self._frames.get(frame_id, identity)
        self._frames.check_current(frame, await self._display_now())
        space, scale = check_space(space, norm_scale, self.settings.default_space, self.settings.default_norm_scale)
        fx, fy = frame_point(frame, x, y, space, scale)
        in_space = f" ({space} {scale})" if space == "norm" else ""
        return frame, frame.to_screen(fx, fy), f"at {_fmt(x, y)}{in_space} in {frame.id}"

    def _button(self, button: str | None) -> str:
        button = button or "left"
        if button not in self.buttons:
            raise Unsupported(f"this computer has no {button} button; it has {', '.join(self.buttons)}")
        return button

    def _modifiers(self, modifiers: list | None) -> list[str]:
        if not modifiers:
            return []
        if not isinstance(modifiers, list):
            raise InvalidArgument("modifiers is a list of W3C key names")
        return chord(modifiers, self.keys)


def _display(frame: Frame) -> str:
    d = frame.display
    return f"{d.width}×{d.height}" + (f" at scale {d.scale:g}" if d.scale != 1 else "")


def effect_data(effect: object) -> dict | None:
    if not isinstance(effect, Effect):
        return None
    data = {k: v for k, v in vars(effect).items() if v is not None}
    return data or None

