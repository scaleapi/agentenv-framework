"""The operations a driver implements, marked with decorators; the base class builds the contract from them.

Each operation is called with screen pixels, positionally, in the order below, and may be sync or async:

    display() -> Display                                  set_display(width, height, scale) -> Display
    capture() -> Image | PIL.Image.Image                  the whole display, at native resolution
    click(x, y, button, count, modifiers) -> Effect | None
    drag(path, button, duration_ms, modifiers) -> Effect | None      path: [(x, y), ...]
    scroll(x, y, direction, amount, modifiers) -> Effect | None
    type_text(text) -> Effect | None                      press(keys) -> Effect | None, one chord
    pointer_move(x, y), pointer_down(button), pointer_up(button), pointer_position() -> (x, y)
    key_down(keys), key_up(keys)
    run_in_session(argv, cwd, env, timeout_s) -> ExecResult            write_files({path: bytes})
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .platform import Platform, matches, normalize, overlap

REQUIRED = ("display", "capture", "click", "drag", "scroll", "type_text", "press")
GROUPS = {
    "pointer": ("pointer_move", "pointer_down", "pointer_up", "pointer_position"),
    "keyboard": ("key_down", "key_up"),
}
OPTIONAL = ("set_display", "run_in_session", "write_files")
INPUT = frozenset({"click", "drag", "scroll", "type_text", "press", "pointer_move", "pointer_down", "pointer_up",
                   "key_down", "key_up"})

_OP_ATTR = "_aec_operation"


@dataclass(frozen=True)
class Display:
    """The display: its resolution in screen pixels, and how many image pixels a capture has per screen pixel."""
    width: int
    height: int
    scale: float = 1.0


@dataclass(frozen=True)
class Image:
    """A capture: encoded ``png`` or ``jpeg`` bytes, or raw pixels in a Pillow raw mode (``RGB``, ``RGBA``, ``BGRA``,
    ``BGRX``)."""
    width: int
    height: int
    data: bytes
    mode: str = "png"


@dataclass(frozen=True)
class Effect:
    """What a driver can confirm an input did; unknown fields stay None."""
    focus_changed: bool | None = None
    pointer_moved: bool | None = None
    clipboard_used: bool | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ExecResult:
    exit_code: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class ReadyCheck:
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class _Mark:
    operation: str
    predicates: dict
    default: bool = False


def _marker(operation: str) -> Callable:
    def decorate(fn: Callable | None = None, /, **predicates: Any) -> Callable:
        mark = _Mark(operation, normalize(predicates))

        def deco(f: Callable) -> Callable:
            setattr(f, _OP_ATTR, mark)
            return f

        return deco(fn) if fn is not None else deco

    decorate.__name__ = decorate.__qualname__ = operation
    decorate.__doc__ = (f"Mark the method that performs ``{operation}``. Use bare, or with predicates "
                        f"(os, distribution, version, arch, display) to choose it per platform.")
    return decorate


display = _marker("display")
set_display = _marker("set_display")
capture = _marker("capture")
click = _marker("click")
drag = _marker("drag")
scroll = _marker("scroll")
type_text = _marker("type_text")
press = _marker("press")
pointer_move = _marker("pointer_move")
pointer_down = _marker("pointer_down")
pointer_up = _marker("pointer_up")
pointer_position = _marker("pointer_position")
key_down = _marker("key_down")
key_up = _marker("key_up")
run_in_session = _marker("run_in_session")
write_files = _marker("write_files")


def default_for(operation: str, **predicates: Any) -> Callable:
    """Mark one of the base class's per-OS defaults, which a driver's own method for the operation replaces."""
    def deco(f: Callable) -> Callable:
        setattr(f, _OP_ATTR, _Mark(operation, normalize(predicates), default=True))
        return f
    return deco


def declared(cls: type) -> dict[str, list[tuple[str, dict]]]:
    """Every operation the class marks, as (method name, predicates), with the base class's defaults dropped for
    an operation the driver implements itself."""
    found: dict[str, list[tuple[str, _Mark]]] = {}
    for name in dir(cls):
        if name.startswith("__"):
            continue
        mark = getattr(inspect.getattr_static(cls, name, None), _OP_ATTR, None)
        if isinstance(mark, _Mark):
            found.setdefault(mark.operation, []).append((name, mark))
    out = {}
    for operation, marks in found.items():
        own = [(name, mark) for name, mark in marks if not mark.default]
        out[operation] = [(name, mark.predicates) for name, mark in (own or marks)]
    return out


def check_class(cls: type) -> None:
    """The class-definition checks: every required operation, whole groups, one method per operation per
    platform. Raises TypeError naming what is wrong."""
    ops = declared(cls)
    problems = []
    missing = [op for op in REQUIRED if op not in ops]
    if missing:
        problems.append(f"missing required operation(s): {', '.join('@' + op for op in missing)}")
    for group, members in GROUPS.items():
        have = [op for op in members if op in ops]
        if have and len(have) != len(members):
            lacking = [op for op in members if op not in ops]
            problems.append(f"the {group} group needs all of {', '.join('@' + op for op in members)}; "
                            f"missing {', '.join('@' + op for op in lacking)}")
    for op, methods in ops.items():
        for i, (a, pa) in enumerate(methods):
            for b, pb in methods[i + 1:]:
                if overlap(pa, pb):
                    problems.append(f"@{op} is marked on both {a} and {b}, and some platform matches both")
    if problems:
        raise TypeError(f"{cls.__qualname__}: " + "; ".join(problems))


def resolve(cls: type, running: Platform) -> dict[str, str]:
    """The method each operation uses on ``running``, by name. A required operation with no method for it raises
    RuntimeError; a group is kept only when all of it matches."""
    chosen = {}
    for op, methods in declared(cls).items():
        hits = [name for name, predicates in methods if matches(predicates, running)]
        if hits:
            chosen[op] = hits[0]
    missing = [op for op in REQUIRED if op not in chosen]
    if missing:
        raise RuntimeError(f"{cls.__qualname__} has no method for {', '.join('@' + op for op in missing)} "
                           f"on this platform ({running.as_dict()})")
    for members in GROUPS.values():
        if not all(op in chosen for op in members):
            for op in members:
                chosen.pop(op, None)
    return chosen
