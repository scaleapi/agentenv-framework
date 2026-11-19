"""What an annotated recording (computer-recording/v1 ``start`` with ``annotate``) draws over its frames: where each
input landed and what it typed or pressed, in the colour of the role that sent it, on the frames captured while its
mark shows. ``ComputerRuntime.act`` marks every ``computer_*`` input as it is delivered; a call to one of the
driver's own or forwarded tools, which acts without it, shows as a chip naming the tool and its main argument."""
from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import av
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from .errors import InvalidArgument

#: The options ``annotate`` takes: none yet, so ``{}`` draws every mark.
OPTIONS: tuple[str, ...] = ()
#: The role of a call that names none: the agent.
DEFAULT_ROLE = "default"
#: Each role's colour: the agent's first, then each other role's in the order it first acts on the recording.
COLOURS = ((255, 138, 0), (0, 184, 148), (175, 82, 222), (255, 59, 105), (52, 120, 246), (250, 204, 21))
#: How long a mark shows from its input, in seconds; a drag's, after its moves.
SHOW_S = {"click": 0.8, "down": 0.8, "up": 0.8, "move": 0.5, "scroll": 0.9, "drag": 0.8, "chip": 1.6}
#: The last part of a mark's time, in seconds, over which it fades out.
FADE_S = 0.3
#: How long a mark no frame has drawn is kept, in seconds: past it, the recording's frames have stopped coming.
KEEP_S = 30.0
#: The inputs whose marks are drawn at a point of the display.
POINTED = ("click", "drag", "scroll", "pointer_move", "pointer_down", "pointer_up")
#: The most a chip of typed text shows, which grows with its length from the chip's own time.
MOST_TEXT_S = 4.0
#: The most of a chip's text it shows; the rest is cut with an ellipsis.
CHIP_CHARS = 60
#: The most chips shown at once, the newest lowest.
MOST_CHIPS = 4
#: How far above the bottom edge, as a fraction of the frame's height, the lowest chip ends.
CHIPS_ABOVE = 0.1
#: A frame's longer edge that marks are sized for; on a larger or smaller frame they scale with it.
UNIT_EDGE = 1280
#: Arguments a tool's chip shows, the first one a call has.
MAIN_ARGUMENTS = ("text", "keys", "key", "url", "query", "label", "name", "element", "title", "code", "script")
#: Key names as chips show them.
KEY_NAMES = {"Control": "Ctrl", "Escape": "Esc", "ArrowLeft": "Left", "ArrowRight": "Right", "ArrowUp": "Up",
             "ArrowDown": "Down", " ": "Space"}


@dataclass(frozen=True)
class Mark:
    """One input on a recording: ``kind`` is click, down, up, move, scroll, drag or chip; ``points`` are fractions of
    the display, so they land on any frame size; it shows from ``at`` until ``until``, a drag moving for its first
    ``moving_s``."""

    kind: str
    at: datetime
    until: datetime
    role: str
    points: tuple[tuple[float, float], ...] = ()
    button: str = "left"
    count: int = 1
    direction: str = ""
    label: str = ""
    moving_s: float = 0.0


def check_annotate(annotate: object) -> Optional[dict]:
    """``start``'s ``annotate``: None, or an object of the options this driver takes."""
    if annotate is None:
        return None
    if not isinstance(annotate, dict):
        raise InvalidArgument(f"annotate is an object of options, got {type(annotate).__name__}")
    if unknown := sorted(set(annotate) - set(OPTIONS)):
        raise InvalidArgument(f"annotate takes no options yet, got {unknown}; give {{}} to draw every input")
    return dict(annotate)


class Annotations:
    """One annotated recording's marks, from their inputs until they stop showing, and the colour of each role. Each
    mark shows on every frame captured while it lasts, and on the first one after its input however late that comes,
    as at a low fps or from a slow capture."""

    def __init__(self) -> None:
        self.marks: list[Mark] = []
        self.drawn: set[int] = set()
        self.colours: dict[str, tuple[int, int, int]] = {DEFAULT_ROLE: COLOURS[0]}

    def add(self, marks: Sequence[Mark]) -> None:
        for mark in marks:
            self.colours.setdefault(mark.role, COLOURS[len(self.colours) % len(COLOURS)])
            stale = mark.at - timedelta(seconds=KEEP_S)
            self._keep([kept for kept in self.marks if kept.until > stale])
            self.marks.append(mark)

    def showing(self, at: datetime) -> list[tuple[Mark, tuple[int, int, int], float]]:
        """The marks a frame captured ``at`` shows, each with its colour and the age, in seconds, it is drawn at: a
        mark over by then that no frame has drawn yet is drawn once, at the age it is last fully shown. A stream's
        frames come in the order they were captured, so a mark that is over and drawn is over for every later one."""
        shown = []
        for mark in self.marks:
            if mark.at > at:
                continue
            age = (at - mark.at).total_seconds()
            if at >= mark.until:
                if id(mark) in self.drawn:
                    continue
                age = max(0.0, (mark.until - mark.at).total_seconds() - FADE_S)
            shown.append((mark, self.colours[mark.role], age))
            self.drawn.add(id(mark))
        self._keep([mark for mark in self.marks if mark.until > at or id(mark) not in self.drawn])
        return shown

    def _keep(self, marks: list[Mark]) -> None:
        self.marks = marks
        self.drawn &= {id(mark) for mark in marks}


def input_marks(operation: str, args: tuple, *, role: Optional[str], at: datetime, size: tuple[int, int],
                point: Optional[tuple] = None, repeat: int = 1, hold_ms: Optional[int] = None) -> list[Mark]:
    """The marks of one ``ComputerRuntime.act`` input: ``args`` are the operation's, in screen units on a display of
    ``size``; ``point`` is where an input with no point of its own acts (its ``move_to``, or the pointer)."""
    role = role or DEFAULT_ROLE
    width, height = size

    def where(x: float, y: float) -> tuple[float, float]:
        return (x / width, y / height)

    def shown(kind: str, extra_s: float = 0.0) -> datetime:
        return at + timedelta(seconds=SHOW_S[kind] + extra_s)

    def chip(label: str, seconds: Optional[float] = None) -> Mark:
        until = at + timedelta(seconds=seconds) if seconds is not None else shown("chip")
        return Mark("chip", at, until, role, label=label if role == DEFAULT_ROLE else f"{role}: {label}")

    held = args[-1] if operation in ("click", "drag", "scroll") and args and isinstance(args[-1], list) else []
    marks: list[Mark] = []
    if operation == "click":
        x, y, button, count = args[:4]
        marks.append(Mark("click", at, shown("click"), role, (where(x, y),), button=button, count=count))
    elif operation == "drag":
        path, button, duration = args[:3]
        marks.append(Mark("drag", at, shown("drag", duration / 1000), role, tuple(where(*p) for p in path),
                          button=button, moving_s=duration / 1000))
    elif operation == "scroll":
        x, y, direction, amount = args[:4]
        marks.append(Mark("scroll", at, shown("scroll"), role, (where(x, y),), direction=direction, count=amount))
    elif operation == "pointer_move":
        marks.append(Mark("move", at, shown("move"), role, (where(*args[:2]),)))
    elif operation in ("pointer_down", "pointer_up"):
        kind = "down" if operation == "pointer_down" else "up"
        if point is not None:
            marks.append(Mark(kind, at, shown(kind), role, (where(*point[:2]),), button=args[0]))
        else:
            marks.append(chip(f"{'press' if kind == 'down' else 'release'} {args[0]}"))
    elif operation == "type_text":
        text = args[0]
        marks.append(chip(f'type "{_shown_text(text)}"', min(MOST_TEXT_S, SHOW_S["chip"] + len(text) / 25)))
    elif operation == "press":
        times = f" x{repeat}" if repeat > 1 else ""
        marks.append(chip(f"key {_chord(args[0])}{times}"))
    elif operation == "key_down":
        for_ms = f" {hold_ms} ms" if hold_ms is not None else ""
        marks.append(chip(f"hold {_chord(args[0])}{for_ms}"))
    elif operation == "key_up":
        marks.append(chip(f"release {_chord(args[0])}"))
    elif operation == "navigate":
        marks.append(chip(f"open {_cut(args[0])}"))
    elif operation == "history":
        marks.append(chip(args[0]))
    elif operation == "tabs":
        action, tab, url = args
        if action != "list":
            marks.append(chip(" ".join(str(part) for part in ("tab", action, tab or _cut(url or "")) if part)))
    elif operation == "dialog":
        marks.append(chip("dialog OK" if args[0] else "dialog Cancel"))
    elif operation == "upload":
        marks.append(chip(f"upload {_cut(', '.join(args[0]))}"))
    else:
        marks.append(chip(operation))
    if held:
        marks.append(chip(f"{_chord(held)} + {operation}"))
    return marks


def tool_mark(tool: str, arguments: Mapping[str, Any], *, role: Optional[str], at: datetime) -> Mark:
    """The chip of a call to one of the driver's own or forwarded tools: its name and its main argument."""
    role = role or DEFAULT_ROLE
    main = next((arguments[name] for name in MAIN_ARGUMENTS if arguments.get(name) not in (None, "", [])), None)
    if isinstance(main, list) and all(isinstance(part, str) for part in main):
        shown = _chord(main)
    elif isinstance(main, str):
        shown = f'"{_shown_text(main)}"' if main.strip() else ""
    else:
        shown = "" if main is None else _cut(str(main))
    label = f"{tool} {shown}".strip()
    return Mark("chip", at, at + timedelta(seconds=SHOW_S["chip"]), role,
                label=label if role == DEFAULT_ROLE else f"{role}: {label}")


def draw(image: Any, marks: Sequence[tuple[Mark, tuple[int, int, int], float]]) -> PILImage.Image:
    """``image``, a Pillow image or a PyAV frame, with ``marks`` drawn, each in its colour at its age."""
    base = (image.to_image() if isinstance(image, av.VideoFrame) else image).convert("RGBA")
    overlay = PILImage.new("RGBA", base.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(overlay)
    unit = max(base.size) / UNIT_EDGE
    chips = []
    for mark, colour, age in marks:
        span = max((mark.until - mark.at).total_seconds(), 1e-3)
        if mark.kind == "chip":
            chips.append((mark, colour))
            continue
        fade = 1.0 - max(0.0, min(1.0, (age - (span - FADE_S)) / FADE_S))
        points = [(fx * base.width, fy * base.height) for fx, fy in mark.points]
        _DRAWERS[mark.kind](pen, mark, points, colour, unit, age, span, fade)
    if chips:
        _draw_chips(pen, chips[-MOST_CHIPS:], base.size, unit)
    return PILImage.alpha_composite(base, overlay).convert("RGB")


def _draw_click(pen, mark, points, colour, unit, age, span, fade) -> None:
    x, y = points[0]
    grow = min(1.0, age / 0.35)
    alpha = round(255 * fade)
    for ring in range(max(1, min(mark.count, 3))):
        r = unit * (12 + 18 * grow + 8 * ring)
        box = (x - r, y - r, x + r, y + r)
        shape = pen.rectangle if mark.button == "right" else pen.ellipse
        shape(box, outline=(255, 255, 255, alpha), width=max(1, round(5 * unit)))
        shape(box, outline=(*colour, alpha), width=max(1, round(3 * unit)))
    dot = unit * 5
    pen.ellipse((x - dot, y - dot, x + dot, y + dot), fill=(*colour, alpha), outline=(255, 255, 255, alpha),
                width=max(1, round(1.5 * unit)))
    if mark.button not in ("left", "right"):
        _label(pen, (x + unit * 16, y + unit * 16), mark.button, colour, unit, alpha)


def _draw_press(pen, mark, points, colour, unit, age, span, fade) -> None:
    x, y = points[0]
    alpha = round(255 * fade)
    r = unit * (10 if mark.kind == "down" else 16)
    fill = (*colour, round(alpha * 0.7)) if mark.kind == "down" else None
    pen.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(255, 255, 255, alpha), width=max(1, round(4 * unit)))
    pen.ellipse((x - r, y - r, x + r, y + r), outline=(*colour, alpha), width=max(1, round(2.5 * unit)))


def _draw_move(pen, mark, points, colour, unit, age, span, fade) -> None:
    x, y = points[0]
    alpha = round(200 * fade)
    arm = unit * 9
    for dx, dy in ((arm, 0), (0, arm)):
        pen.line((x - dx, y - dy, x + dx, y + dy), fill=(255, 255, 255, alpha), width=max(1, round(4 * unit)))
        pen.line((x - dx, y - dy, x + dx, y + dy), fill=(*colour, alpha), width=max(1, round(2 * unit)))


def _draw_scroll(pen, mark, points, colour, unit, age, span, fade) -> None:
    x, y = points[0]
    alpha = round(255 * fade)
    dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.get(mark.direction, (0, 1))
    length = unit * 46
    end = (x + dx * length, y + dy * length)
    for width, fill in ((7, (255, 255, 255, alpha)), (4, (*colour, alpha))):
        pen.line((x, y, *end), fill=fill, width=max(1, round(width * unit)))
        _arrowhead(pen, (x, y), end, unit, fill, width)
    _label(pen, (end[0] + unit * 10, end[1] + unit * 4), f"x{mark.count}", colour, unit, alpha)


def _draw_drag(pen, mark, points, colour, unit, age, span, fade) -> None:
    alpha = round(255 * fade)
    shown = points if age >= mark.moving_s else _along(points, age / mark.moving_s)
    for width, fill in ((7, (255, 255, 255, alpha)), (4, (*colour, alpha))):
        if len(shown) > 1:
            pen.line([coord for point in shown for coord in point], fill=fill, width=max(1, round(width * unit)),
                     joint="curve")
            _arrowhead(pen, shown[-2], shown[-1], unit, fill, width)
    sx, sy = points[0]
    r = unit * 6
    pen.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(*colour, alpha), outline=(255, 255, 255, alpha),
                width=max(1, round(1.5 * unit)))


_DRAWERS = {"click": _draw_click, "down": _draw_press, "up": _draw_press, "move": _draw_move, "scroll": _draw_scroll,
            "drag": _draw_drag}


def _draw_chips(pen, chips, size, unit) -> None:
    """Chips stacked centred above the bottom edge, clear of a panel or dock there, the newest lowest."""
    font = _font(max(10, round(20 * unit)))
    pad, gap = unit * 10, unit * 8
    y = size[1] - max(unit * 48, size[1] * CHIPS_ABOVE)
    for mark, colour in reversed(chips):
        left, top, right, bottom = pen.textbbox((0, 0), mark.label, font=font)
        w, h = right - left + 2 * pad, bottom - top + 2 * pad * 0.7
        x0 = (size[0] - w) / 2
        y0 = y - h
        pen.rounded_rectangle((x0, y0, x0 + w, y), radius=unit * 8, fill=(*colour, 235),
                              outline=(255, 255, 255, 235), width=max(1, round(1.5 * unit)))
        pen.text((x0 + pad - left, y0 + pad * 0.7 - top), mark.label, font=font, fill=_ink(colour))
        y = y0 - gap


def _label(pen, at: tuple[float, float], text: str, colour, unit, alpha) -> None:
    font = _font(max(9, round(15 * unit)))
    left, top, right, bottom = pen.textbbox(at, text, font=font)
    pad = unit * 4
    pen.rounded_rectangle((left - pad, top - pad, right + pad, bottom + pad), radius=unit * 4,
                          fill=(*colour, round(alpha * 0.9)))
    pen.text(at, text, font=font, fill=(*_ink(colour)[:3], alpha))


def _arrowhead(pen, start, end, unit, fill, width) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = unit * (14 + width)
    left = (end[0] - size * math.cos(angle - 0.5), end[1] - size * math.sin(angle - 0.5))
    right = (end[0] - size * math.cos(angle + 0.5), end[1] - size * math.sin(angle + 0.5))
    pen.polygon([end, left, right], fill=fill)


def _along(points: list[tuple[float, float]], fraction: float) -> list[tuple[float, float]]:
    """The first ``fraction`` of the path through ``points``, by length."""
    legs = [math.dist(a, b) for a, b in zip(points, points[1:])]
    left = sum(legs) * max(0.0, min(1.0, fraction))
    shown = [points[0]]
    for (a, b), leg in zip(zip(points, points[1:]), legs):
        if left >= leg:
            shown.append(b)
            left -= leg
            continue
        t = left / leg if leg else 0.0
        shown.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        break
    return shown


def _ink(colour: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Black or white, whichever reads on ``colour``."""
    r, g, b = colour
    return (20, 20, 20, 255) if 0.299 * r + 0.587 * g + 0.114 * b > 160 else (255, 255, 255, 255)


def _chord(keys: Sequence[str]) -> str:
    return "+".join(KEY_NAMES.get(key, key.upper() if len(key) == 1 else key) for key in keys)


def _shown_text(text: str) -> str:
    return _cut(text.replace("\n", "[Enter]").replace("\t", "[Tab]"))


def _cut(text: str) -> str:
    return text if len(text) <= CHIP_CHARS else text[:CHIP_CHARS - 1] + "…"


@functools.lru_cache(maxsize=16)
def _font(size: int) -> Any:
    return ImageFont.load_default(size=size)
