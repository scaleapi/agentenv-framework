"""A TOML writer for the tomls a bundle import writes, since ``tomllib`` only reads."""

from __future__ import annotations

import math
import re
from collections.abc import Collection
from typing import Any

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def dumps(document: dict[str, Any], *, tables: Collection[str] = ()) -> str:
    """``document`` as TOML, keys in ``tables`` as ``[table]``s after the rest; ValueError for what TOML can't hold."""
    lines = [f"{_key(key)} = {_value(value)}" for key, value in document.items() if key not in tables]
    for key in (key for key in document if key in tables):
        lines += ["", f"[{_key(key)}]", *(f"{_key(k)} = {_value(v)}" for k, v in document[key].items())]
    return "\n".join(lines).lstrip("\n") + "\n" if lines else ""


def _key(key: str) -> str:
    return key if _BARE_KEY.fullmatch(key) else _string(key)


def _value(value: Any, *, top: bool = True) -> str:
    """``value`` as TOML; a list of several items spans lines only at the top, as an inline table can't."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{value!r} is not a TOML number")
        return repr(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, (list, tuple)):
        items = [_value(item, top=False) for item in value]
        if top and len(items) > 1:
            return "[\n" + "".join(f"  {item},\n" for item in items) + "]"
        return f"[{', '.join(items)}]"
    if isinstance(value, dict):
        pairs = ", ".join(f"{_key(k)} = {_value(v, top=False)}" for k, v in value.items())
        return f"{{ {pairs} }}" if pairs else "{}"
    raise ValueError(f"TOML can't hold {value!r}")


def _string(text: str) -> str:
    out = []
    for char in text:
        if char in _ESCAPES:
            out.append(_ESCAPES[char])
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'
