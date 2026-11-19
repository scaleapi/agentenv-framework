"""What an annotated recording (computer-recording/v1 ``start`` with ``annotate``) draws over its frames: where each
input landed and what it typed or pressed, in the colour of the role that sent it, on the frames captured while its
mark shows. ``ComputerRuntime.act`` marks every ``computer_*`` input as it is delivered; a call to one of the
driver's own or forwarded tools, which acts without it, shows as a chip naming the tool and its main argument. Where
the frames can't show the pointer themselves, as frames from screenshots that leave it out, the agent's pointer is
drawn where its inputs left it."""
from __future__ import annotations

import dataclasses
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
#: The pointer an input acts through when it names none: the computer's own, the only one the SDK drives yet.
DEFAULT_POINTER = "default"
#: Each role's colour: the agent's first, then each other role's in the order it first acts on the recording.
COLOURS = ((255, 138, 0), (0, 184, 148), (175, 82, 222), (255, 59, 105), (52, 120, 246), (250, 204, 21))
#: How long a mark shows from its input, in seconds; a drag's, after its moves.
SHOW_S = {"click": 0.8, "down": 0.8, "up": 0.8, "move": 0.5, "scroll": 0.9, "drag": 0.8, "chip": 1.6}
#: The last part of a mark's time, in seconds, over which it fades out.
FADE_S = 0.3
#: How long a mark no frame has drawn is kept, in seconds: past it, the recording's frames have stopped coming.
KEEP_S = 30.0
#: The marks of inputs that leave the pointer at their last point, a drag's after moving it along its path.
MOVES_POINTER = ("click", "drag", "scroll", "move", "down", "up")
#: How long after the pointer jumps a line shows where it came from, in seconds.
JUMP_S = 0.4
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
    """One input on a recording: ``kind`` is click, down, up, move, scroll, drag or chip, or pointer for where a
    pointer was and then went; ``points`` are fractions of the display, so they land on any frame size; it shows from
    ``at`` until ``until``, a drag moving for its first ``moving_s``. ``pointer`` is the pointer it acted through;
    ``faded``, a pointer drawn fainter, as one that didn't move last."""

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
    pointer: str = DEFAULT_POINTER
    faded: bool = False


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
    as at a low fps or from a slow capture. With ``pointer``, as for frames that can't show the pointer themselves,
    each frame also draws a pointer for each role and pointer that has acted, where its last input before the frame
    left it: the one that moved last at full strength, the others fainter, each named once there are several."""

    def __init__(self, pointer: bool = False) -> None:
        self.marks: list[Mark] = []
        self.drawn: set[int] = set()
        self.colours: dict[str, tuple[int, int, int]] = {DEFAULT_ROLE: COLOURS[0]}
        self.pointer = pointer
        self.moves: dict[tuple[str, str], list[Mark]] = {}  # by role and pointer: each move, where it was and went

    def add(self, marks: Sequence[Mark]) -> None:
        for mark in marks:
            self.colours.setdefault(mark.role, COLOURS[len(self.colours) % len(COLOURS)])
            stale = mark.at - timedelta(seconds=KEEP_S)
            self._keep([kept for kept in self.marks if kept.until > stale])
            self.marks.append(mark)
            if self.pointer and mark.kind in MOVES_POINTER and mark.points:
                moves = self.moves.setdefault((mark.role, mark.pointer), [])
                path = mark.points if mark.kind == "drag" else mark.points[-1:]
                was = moves[-1].points[-1] if moves else path[0]
                moves.append(Mark("pointer", mark.at, mark.at, mark.role, (was, *path), moving_s=mark.moving_s,
                                  pointer=mark.pointer))

    def _pointers(self, at: datetime) -> list[tuple[Mark, tuple[int, int, int], float]]:
        """The pointers a frame captured ``at`` shows, each as its last move before the frame at its age, the one that
        moved last drawn over the others; none for a role or pointer that hadn't acted by then."""
        current = []
        for key, moves in self.moves.items():
            last = next((i for i in range(len(moves) - 1, -1, -1) if moves[i].at <= at), None)
            if last is not None:
                self.moves[key] = moves[last:]  # the frames come in order, so an earlier move won't show again
                current.append(moves[last])
        current.sort(key=lambda move: move.at)
        several = len(current) > 1
        shown = [dataclasses.replace(move, faded=move is not current[-1],
                                     label=_pointer_label(move) if several else "") for move in current]
        return [(move, self.colours[move.role], (at - move.at).total_seconds()) for move in shown]

    def showing(self, at: datetime) -> list[tuple[Mark, tuple[int, int, int], float]]:
        """The marks a frame captured ``at`` shows, each with its colour and the age, in seconds, it is drawn at: a
        mark over by then that no frame has drawn yet is drawn once, at the age it is last fully shown. A stream's
        frames come in the order they were captured, so a mark that is over and drawn is over for every later one."""
        due = []
        for mark in self.marks:
            if mark.at > at:
                continue
            age = (at - mark.at).total_seconds()
            if at >= mark.until:
                if id(mark) in self.drawn:
                    continue
                age = max(0.0, (mark.until - mark.at).total_seconds() - FADE_S)
            due.append((mark, self.colours[mark.role], age))
        chips = [mark for mark, _colour, _age in due if mark.kind == "chip"]
        waiting = {id(mark) for mark in chips[:-MOST_CHIPS]}  # older chips than a frame shows wait for a later one
        shown = [entry for entry in due if id(entry[0]) not in waiting]
        self.drawn |= {id(mark) for mark, _colour, _age in shown}
        self._keep([mark for mark in self.marks if mark.until > at or id(mark) not in self.drawn])
        return shown + (self._pointers(at) if self.pointer else [])

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
        fade = 1.0 if mark.kind == "pointer" else 1.0 - max(0.0, min(1.0, (age - (span - FADE_S)) / FADE_S))
        points = [(fx * base.width, fy * base.height) for fx, fy in mark.points]
        _DRAWERS[mark.kind](pen, mark, points, colour, unit, age, fade)
    if chips:
        _draw_chips(pen, chips, base.size, unit)
    return PILImage.alpha_composite(base, overlay).convert("RGB")


@dataclass(frozen=True)
class Style:
    """How marks look: sizes in pixels of a frame whose longer edge is UNIT_EDGE, scaled with the frame; each coloured
    stroke over a wider white halo, so a mark reads on any background."""

    stroke: float = 3  # a ring's and a cross's coloured stroke
    halo: float = 5  # the white band under a ring's stroke
    line: float = 4  # a trail's and an arrow's coloured stroke
    line_halo: float = 7
    outline: float = 1.5  # a dot's white outline
    ring_grow_s: float = 0.35  # how long a click's ring takes to grow to its full size, in seconds
    ring_from: float = 12  # a click's ring's radius as it starts
    ring_grows: float = 18  # and how much it grows
    ring_apart: float = 8  # between the rings of a double or triple click
    most_rings: int = 3
    dot: float = 5  # the hollow dot at a click's point, which leaves the pointer's tip there in sight
    dot_stroke: float = 2
    pointer: float = 1.6  # the agent's pointer, as a multiple of the usual arrow cursor's size
    pointer_outline: float = 2
    pointer_faded_alpha: int = 120  # a pointer that didn't move last
    jump_alpha: int = 170  # the line from where the pointer jumped, at its start
    press_down: float = 10  # a press's filled ring's radius
    press_up: float = 16  # a release's ring's radius
    press_fill: float = 0.7  # a press's fill opacity, as a share of its stroke's
    cross: float = 9  # each arm of a move's cross
    cross_alpha: int = 200  # a move's cross is fainter than the marks of inputs that act
    arrow: float = 46  # a scroll's arrow
    arrowhead: float = 14  # an arrowhead's sides, beside the stroke's own width
    arrowhead_angle: float = 0.5  # radians either side of its line
    start_dot: float = 6  # the dot where a drag starts
    chip_text: float = 20
    chip_pad: float = 10  # beside a chip's text; above and below it, CHIP_PAD_HIGH of that
    chip_gap: float = 8  # between stacked chips
    chip_radius: float = 8
    chip_alpha: int = 235
    chip_low: float = 48  # the least the lowest chip ends above the bottom edge
    label_text: float = 15  # a scroll's count and a button's name beside its mark
    label_pad: float = 4
    label_from: float = 16  # a button's name, right and down from its click
    label_alpha: float = 0.9
    least_text: int = 9  # the smallest text, in pixels, on any frame


STYLE = Style()
#: A chip's padding above and below its text, as a share of the padding beside it.
CHIP_PAD_HIGH = 0.7
WHITE = (255, 255, 255)


def _width(width: float, unit: float) -> int:
    return max(1, round(width * unit))


def _draw_click(pen, mark, points, colour, unit, age, fade) -> None:
    x, y = points[0]
    grow = min(1.0, age / STYLE.ring_grow_s)
    alpha = round(255 * fade)
    for ring in range(max(1, min(mark.count, STYLE.most_rings))):
        r = unit * (STYLE.ring_from + STYLE.ring_grows * grow + STYLE.ring_apart * ring)
        box = (x - r, y - r, x + r, y + r)
        shape = pen.rectangle if mark.button == "right" else pen.ellipse
        shape(box, outline=(*WHITE, alpha), width=_width(STYLE.halo, unit))
        shape(box, outline=(*colour, alpha), width=_width(STYLE.stroke, unit))
    dot = unit * STYLE.dot
    pen.ellipse((x - dot, y - dot, x + dot, y + dot), outline=(*colour, alpha), width=_width(STYLE.dot_stroke, unit))
    if mark.button not in ("left", "right"):
        _label(pen, (x + unit * STYLE.label_from, y + unit * STYLE.label_from), mark.button, colour, unit, alpha)


def _draw_press(pen, mark, points, colour, unit, age, fade) -> None:
    x, y = points[0]
    alpha = round(255 * fade)
    r = unit * (STYLE.press_down if mark.kind == "down" else STYLE.press_up)
    fill = (*colour, round(alpha * STYLE.press_fill)) if mark.kind == "down" else None
    pen.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(*WHITE, alpha), width=_width(STYLE.halo, unit))
    pen.ellipse((x - r, y - r, x + r, y + r), outline=(*colour, alpha), width=_width(STYLE.stroke, unit))


def _draw_move(pen, mark, points, colour, unit, age, fade) -> None:
    x, y = points[0]
    alpha = round(STYLE.cross_alpha * fade)
    arm = unit * STYLE.cross
    for dx, dy in ((arm, 0), (0, arm)):
        pen.line((x - dx, y - dy, x + dx, y + dy), fill=(*WHITE, alpha), width=_width(STYLE.halo, unit))
        pen.line((x - dx, y - dy, x + dx, y + dy), fill=(*colour, alpha), width=_width(STYLE.stroke, unit))


def _draw_scroll(pen, mark, points, colour, unit, age, fade) -> None:
    x, y = points[0]
    alpha = round(255 * fade)
    dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.get(mark.direction, (0, 1))
    end = (x + dx * unit * STYLE.arrow, y + dy * unit * STYLE.arrow)
    for width, fill in ((STYLE.line_halo, (*WHITE, alpha)), (STYLE.line, (*colour, alpha))):
        pen.line((x, y, *end), fill=fill, width=_width(width, unit))
        _arrowhead(pen, (x, y), end, unit, fill, width)
    _label(pen, (end[0] + unit * STYLE.label_pad * 2, end[1] + unit * STYLE.label_pad), f"x{mark.count}", colour,
           unit, alpha)


def _draw_drag(pen, mark, points, colour, unit, age, fade) -> None:
    alpha = round(255 * fade)
    shown = points if age >= mark.moving_s else _along(points, age / mark.moving_s)
    for width, fill in ((STYLE.line_halo, (*WHITE, alpha)), (STYLE.line, (*colour, alpha))):
        if len(shown) > 1:
            pen.line([coord for point in shown for coord in point], fill=fill, width=_width(width, unit),
                     joint="curve")
            _arrowhead(pen, shown[-2], shown[-1], unit, fill, width)
    sx, sy = points[0]
    r = unit * STYLE.start_dot
    pen.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(*colour, alpha), outline=(*WHITE, alpha),
                width=_width(STYLE.outline, unit))


#: The usual arrow cursor, its tip at (0, 0), in pixels of a UNIT_EDGE frame.
ARROW = ((0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12))


def _draw_pointer(pen, mark, points, colour, unit, age, fade) -> None:
    """The agent's pointer: along a drag's path while it moves, else where its input left it, after a fading line
    from where it jumped."""
    was, path = points[0], points[1:]
    tip = _along(path, age / mark.moving_s)[-1] if len(path) > 1 and age < mark.moving_s else path[-1]
    if age < JUMP_S and math.dist(was, path[0]) > unit * STYLE.cross:
        alpha = round(STYLE.jump_alpha * (1 - age / JUMP_S))
        pen.line((*was, *path[0]), fill=(*colour, alpha), width=_width(STYLE.stroke, unit))
    scale = unit * STYLE.pointer
    arrow = [(tip[0] + dx * scale, tip[1] + dy * scale) for dx, dy in ARROW]
    alpha = STYLE.pointer_faded_alpha if mark.faded else 255
    pen.polygon(arrow, fill=(*colour, alpha), outline=(*WHITE, alpha), width=_width(STYLE.pointer_outline, unit))
    if mark.label:
        corner = max(dx for dx, _dy in ARROW), max(dy for _dx, dy in ARROW)
        _label(pen, (tip[0] + corner[0] * scale, tip[1] + corner[1] * scale), mark.label, colour, unit, alpha)


_DRAWERS = {"click": _draw_click, "down": _draw_press, "up": _draw_press, "move": _draw_move, "scroll": _draw_scroll,
            "drag": _draw_drag, "pointer": _draw_pointer}


def _draw_chips(pen, chips, size, unit) -> None:
    """Chips stacked centred above the bottom edge, clear of a panel or dock there, the newest lowest."""
    font = _font(max(STYLE.least_text, round(STYLE.chip_text * unit)))
    pad, high, gap = unit * STYLE.chip_pad, unit * STYLE.chip_pad * CHIP_PAD_HIGH, unit * STYLE.chip_gap
    y = size[1] - max(unit * STYLE.chip_low, size[1] * CHIPS_ABOVE)
    for mark, colour in reversed(chips):
        left, top, right, bottom = pen.textbbox((0, 0), mark.label, font=font)
        w, h = right - left + 2 * pad, bottom - top + 2 * high
        x0, y0 = (size[0] - w) / 2, y - h
        pen.rounded_rectangle((x0, y0, x0 + w, y), radius=unit * STYLE.chip_radius, fill=(*colour, STYLE.chip_alpha),
                              outline=(*WHITE, STYLE.chip_alpha), width=_width(STYLE.outline, unit))
        pen.text((x0 + pad - left, y0 + high - top), mark.label, font=font, fill=_ink(colour))
        y = y0 - gap


def _label(pen, at: tuple[float, float], text: str, colour, unit, alpha) -> None:
    font = _font(max(STYLE.least_text, round(STYLE.label_text * unit)))
    left, top, right, bottom = pen.textbbox(at, text, font=font)
    pad = unit * STYLE.label_pad
    pen.rounded_rectangle((left - pad, top - pad, right + pad, bottom + pad), radius=pad,
                          fill=(*colour, round(alpha * STYLE.label_alpha)))
    pen.text(at, text, font=font, fill=(*_ink(colour)[:3], alpha))


def _arrowhead(pen, start, end, unit, fill, width) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = unit * (STYLE.arrowhead + width)
    sides = [(end[0] - size * math.cos(angle + turn), end[1] - size * math.sin(angle + turn))
             for turn in (-STYLE.arrowhead_angle, STYLE.arrowhead_angle)]
    pen.polygon([end, *sides], fill=fill)


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


def _pointer_label(move: Mark) -> str:
    """A pointer's name beside it: its role's, the agent's as agent, and its own when it isn't the default."""
    role = "agent" if move.role == DEFAULT_ROLE else move.role
    return role if move.pointer == DEFAULT_POINTER else f"{role} {move.pointer}"


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
