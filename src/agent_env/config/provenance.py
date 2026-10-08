"""Where a resolved config value came from, and how to say it in one phrase.

The resolvers in ``runtime`` compute a value and, on the traced path, the ordered list of
layers that could have supplied it. Reporting reuses that list rather than re-deriving the
precedence rule, so the two can never drift.

Display lives here too, for the same reason: a winning layer and a shadowed one are the
same kind of thing said at different lengths, so they share one renderer instead of each
consumer growing its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

LayerKind = Literal["env", "file", "default", "installed"]

KIND_ENV: LayerKind = "env"
KIND_FILE: LayerKind = "file"
KIND_DEFAULT: LayerKind = "default"
# Set on the Config itself by `configure()`, above the file and below the environment.
KIND_INSTALLED: LayerKind = "installed"


def class_name(impl: Any) -> Optional[str]:
    """The class alone — ``LocalFilesystemObjectStore``, not the dotted path."""
    return impl.rpartition(":")[2] if isinstance(impl, str) else None


def lines(value: Any, *, seam: bool = True) -> list[str]:
    """A config node in full: its class and scalars first, then everything nested, indented.

    Nothing is elided. `config show` is a diagnostic, so a summary that hid a nested table
    behind its key names hid the values an operator ran the command to read.

    ``seam=False`` shows a table exactly as written: a plugin's own table gives ``impl`` and
    ``config`` no meaning agent-env knows, so neither is read as a class and its settings.
    """
    if isinstance(value, list):
        return [""] + ([line for item in value for line in _entry(item, seam)] or ["(empty)"])
    if not isinstance(value, dict):
        return ["(unset)" if value is None else _scalar(value)]
    impl = class_name(value.get("impl")) if seam else None
    if not impl:
        return [_scalars(value)] + _nested(value, seam)
    # An impl's `config` reads at the same level as the class it configures; anything else
    # beside them is a key at the wrong level, which is the thing worth seeing. `config`
    # wins a collision, because that is the one the reader takes.
    config = value.get("config")
    hoisted = config if isinstance(config, dict) else {}
    # a `config` that is not a table is an ordinary key, and hiding it would elide it
    skip = ("impl", "config") if hoisted else ("impl",)
    table = {**{k: v for k, v in value.items() if k not in skip}, **hoisted}
    return ["  ".join(filter(None, [_scalar(impl), _scalars(table)]))] + _nested(table, seam)


def headline(value: Any, *, seam: bool = True) -> str:
    """What a config node says at shadowed length: the class it names, else the keys it sets."""
    if isinstance(value, dict):
        impl = class_name(value.get("impl")) if seam else None
        return (_scalar(impl) if impl else "") or _scalars(value) or ", ".join(value) or "(unset)"
    return "(unset)" if value is None else _scalar(value)


def _scalar(value: Any) -> str:
    """A value on one physical line: TOML strings may carry newlines, the layout may not."""
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def _scalars(table: dict) -> str:
    return "  ".join(f"{k}={_scalar(v)}" for k, v in table.items()
                     if not isinstance(v, (dict, list)))


def _nested(table: dict, seam: bool) -> list[str]:
    """Each table or list under `table`, beneath its own heading."""
    out: list[str] = []
    for key, value in table.items():
        if isinstance(value, list):
            out.append(f"{key}:")
            out += [f"  {line}" for item in value for line in _entry(item, seam)] or ["  (empty)"]
        elif isinstance(value, dict):
            out.append(f"{key}:")
            out += [f"  {row}" for row in _rows(value, seam)] or ["  (empty)"]
    return out


def _rows(table: dict, seam: bool) -> list[str]:
    """One entry per line: `name=value` for a scalar, else the node's own lines under it."""
    out: list[str] = []
    for name, node in table.items():
        if not isinstance(node, (dict, list)):
            out.append(f"{name}={_scalar(node)}")
            continue
        body = [line for line in lines(node, seam=seam) if line]
        # a one-key table inlines fine; a one-element list would read as a scalar
        if len(body) == 1 and not isinstance(node, list):
            out.append(f"{name}  {body[0]}")
        else:
            out.append(f"{name}:")
            out += [f"  {line}" for line in body]
    return out


def _entry(item: Any, seam: bool) -> list[str]:
    """One list entry, as its own block: continuation lines indent under the first."""
    if not isinstance(item, (dict, list)):
        return [_scalar(item)]
    body = [line for line in lines(item, seam=seam) if line]
    return body[:1] + [f"  {line}" for line in body[1:]]


@dataclass(frozen=True)
class Layer:
    """One candidate source for a value, resolved or not."""

    kind: LayerKind
    where: str
    raw: Any = None
    error: Optional[str] = None

    def summary(self) -> str:
        """The candidate at shadowed length: what was written, not what it became."""
        return f"(unreadable: {self.error})" if self.error is not None else headline(self.raw)


@dataclass(frozen=True)
class Traced:
    """A resolved value plus the layer that won and the ones it beat, highest priority first."""

    value: Any
    winner: Layer
    shadowed: tuple[Layer, ...] = ()
