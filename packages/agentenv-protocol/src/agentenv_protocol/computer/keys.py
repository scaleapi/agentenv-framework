"""W3C UI Events key values: a single character, or a named key, case-insensitive and passed on in its W3C case.
Model spellings (ctrl, Return, ctrl+s) are an adapter's to map; a refused name gets a hint, never a remapping."""
from __future__ import annotations

import difflib

from .errors import InvalidArgument, Unsupported

NAMED = (
    "Alt", "AltGraph", "CapsLock", "Control", "Fn", "FnLock", "Hyper", "Meta", "NumLock", "ScrollLock", "Shift",
    "Super", "Symbol", "SymbolLock",
    "Enter", "Tab",
    "ArrowDown", "ArrowLeft", "ArrowRight", "ArrowUp", "End", "Home", "PageDown", "PageUp",
    "Backspace", "Clear", "Copy", "CrSel", "Cut", "Delete", "EraseEof", "ExSel", "Insert", "Paste", "Redo", "Undo",
    "Accept", "Again", "Attn", "Cancel", "ContextMenu", "Escape", "Execute", "Find", "Help", "Pause", "Play",
    "Props", "Select", "ZoomIn", "ZoomOut",
    "BrightnessDown", "BrightnessUp", "Eject", "LogOff", "Power", "PowerOff", "PrintScreen", "Hibernate", "Standby",
    "WakeUp",
    *(f"F{n}" for n in range(1, 25)),
    "AudioVolumeDown", "AudioVolumeMute", "AudioVolumeUp", "MediaPlayPause", "MediaStop", "MediaTrackNext",
    "MediaTrackPrevious",
    "BrowserBack", "BrowserFavorites", "BrowserForward", "BrowserHome", "BrowserRefresh", "BrowserSearch",
    "BrowserStop",
)
_BY_LOWER = {name.lower(): name for name in NAMED}
_SPELLINGS = {
    "ctrl": "Control", "control_l": "Control", "cmd": "Meta", "command": "Meta", "win": "Meta", "windows": "Meta",
    "super_l": "Meta", "option": "Alt", "opt": "Alt", "return": "Enter", "esc": "Escape", "del": "Delete",
    "space": "' '", "spacebar": "' '", "up": "ArrowUp", "down": "ArrowDown",
    "left": "ArrowLeft", "right": "ArrowRight", "pgup": "PageUp", "pgdn": "PageDown", "ins": "Insert",
    "menu": "ContextMenu", "print": "PrintScreen",
}


def canonical(key: object) -> str:
    """``key`` as its W3C value, or InvalidArgument with a hint naming the closest valid one."""
    if not isinstance(key, str) or not key:
        raise InvalidArgument(f"a key is a W3C key value, got {key!r}")
    if len(key) == 1:
        return key
    name = _BY_LOWER.get(key.lower())
    if name is not None:
        return name
    raise InvalidArgument(f"{key!r} is not a W3C key value", hint=_hint(key))


def chord(keys: object, supported: object) -> list[str]:
    """A chord's keys in W3C case, refused whole when one isn't a key or isn't one ``supported`` names."""
    if not isinstance(keys, list) or not keys:
        raise InvalidArgument("keys is a non-empty list of W3C key values, such as ['Control', 's']")
    names = [canonical(k) for k in keys]
    if supported != "all":
        lacking = [n for n in names if n not in supported]
        if lacking:
            raise Unsupported(f"this computer doesn't support the key(s) {lacking}")
    return names


def _hint(key: str) -> str:
    lower = key.lower()
    if "+" in key or " " in key.strip():
        return "one key per entry: keys is a list, such as ['Control', 's']"
    if lower in _SPELLINGS:
        return f"did you mean {_SPELLINGS[lower]}?"
    close = difflib.get_close_matches(lower, list(_BY_LOWER), n=1, cutoff=0.6)
    return f"did you mean {_BY_LOWER[close[0]]}?" if close else "named keys are W3C UI Events key values, such as Enter"
