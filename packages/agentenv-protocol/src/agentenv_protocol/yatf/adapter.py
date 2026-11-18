"""Writing a yatf/1 adapter: a class, declared with ``@yatf_adapter``, whose ``@native``, ``@aux`` and ``@finish``
methods turn an agent's native records into YATF events through the YatfTurn each one is handed. It reads only what the
run stores, so an offline replay makes the events the agent served live."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Sequence, TypeVar

from .models import Event, YatfError, is_extension_type

if TYPE_CHECKING:
    from .runner import YatfTurn

#: A native format id: ``<name>/<version>``, as ``claude-code-stream-json/1``.
NATIVE_FORMAT = re.compile(r"^[a-z0-9][a-z0-9._-]*/[A-Za-z0-9._-]+$")
_MISSING = object()

A = TypeVar("A", bound="Adapter")


@dataclass(frozen=True)
class Native:
    """What ``from_yatf`` gives back: each turn's native log in turn order, for the round trip, and the files the CLI
    resumes from, by name, or None when this adapter can't seed a session."""

    log: list[list[Any]]
    session: dict[str, bytes] | None = None


@dataclass(frozen=True)
class Selector:
    """The records a handler takes: native ones when ``aux`` is None, else aux file ``aux``'s, whose fields at each
    dotted path in ``conditions`` hold one of the listed values."""

    aux: str | None
    conditions: tuple[tuple[str, tuple[Any, ...]], ...]
    media_type: str | None = field(default=None, compare=False)

    def matches(self, record: Any) -> bool:
        return all(_lookup(record, path) in allowed for path, allowed in self.conditions)


def _lookup(record: Any, path: str) -> Any:
    value = record
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _values(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (tuple, list, set, frozenset)) else (value,)


def _handler(aux_name: str | None, kind: Any, where: Mapping[str, Any] | None,
             media_type: str | None) -> Callable[[Callable], Callable]:
    conditions = ([("type", _values(kind))] if kind is not None else []) + [
        (path, _values(value)) for path, value in (where or {}).items()]
    selector = Selector(aux_name, tuple(conditions), media_type)

    def mark(method: Callable) -> Callable:
        method.__dict__.setdefault("__yatf_selectors__", []).append(selector)
        return method
    return mark


def native(*, type: Any = None, where: Mapping[str, Any] | None = None) -> Callable[[Callable], Callable]:
    """Handle native records whose ``type`` is ``type`` (one value or several) and whose fields at the dotted ``where``
    paths hold the given values. Of the handlers that match a record, the one with the most conditions takes it."""
    return _handler(None, type, where, None)


def aux(name: str, *, media_type: str = "application/jsonl", type: Any = None,
        where: Mapping[str, Any] | None = None) -> Callable[[Callable], Callable]:
    """Handle records of the aux file ``name``, matched as ``native`` matches; this also declares the file. Aux
    records no handler takes are dropped, since the run stores them anyway."""
    return _handler(name, type, where, media_type)


def finish(method: Callable) -> Callable:
    """Run at turn end, after the last record: emit what's held and reset per-turn state."""
    method.__yatf_finish__ = True
    return method


@dataclass(frozen=True)
class AdapterSpec:
    """What ``@yatf_adapter`` declares, and its handlers by selector."""

    name: str
    native_format: str
    version: str
    usage_input_includes_cache: bool | None
    tested_with: tuple[str, ...]
    lossy: tuple[str, ...]
    namespace: str
    aux: Mapping[str, str]
    handlers: tuple[tuple[Selector, str], ...]
    finish: str | None

    def handler(self, aux_name: str | None, record: Any) -> str | None:
        """The method that takes ``record``: the most specific matching handler, None when none matches."""
        matches = [(len(sel.conditions), method) for sel, method in self.handlers
                   if sel.aux == aux_name and sel.matches(record)]
        if not matches:
            return None
        best = max(count for count, _ in matches)
        methods = sorted({method for count, method in matches if count == best})
        if len(methods) > 1:
            raise YatfError(f"a record matches {', '.join(methods)} equally; make one of them more specific")
        return methods[0]


def yatf_adapter(*, native_format: str, version: str, usage_input_includes_cache: bool | None,
                 tested_with: Sequence[str] = (), lossy: Sequence[str] = (), name: str | None = None,
                 namespace: str | None = None) -> Callable[[type[A]], type[A]]:
    """Declare an Adapter subclass for ``native_format``.

    ``usage_input_includes_cache`` says whether the native input token count already includes cache reads and writes
    (None when the native reports no usage). ``tested_with`` names the native producer versions its recorded fixtures
    come from, ``lossy`` only what the native never records, and ``namespace`` its extension types
    (``<namespace>/raw``), by default the native format's name."""
    if not isinstance(native_format, str) or not NATIVE_FORMAT.match(native_format):
        raise YatfError(f"native_format {native_format!r} isn't <name>/<version>")
    if native_format.startswith("yatf/"):
        raise YatfError(f"native_format {native_format!r}: yatf is the format adapters produce, not a native one")
    if not isinstance(version, str) or not version:
        raise YatfError("version is the adapter's own version string")
    if usage_input_includes_cache not in (True, False, None):
        raise YatfError("usage_input_includes_cache is true, false or None")
    ns = namespace or native_format.split("/", 1)[0]
    if not is_extension_type(f"{ns}/raw"):
        raise YatfError(f"namespace {ns!r} can't name extension types")

    def declare(cls: type[A]) -> type[A]:
        if not (isinstance(cls, type) and issubclass(cls, Adapter)):
            raise TypeError("@yatf_adapter decorates an Adapter subclass")
        members: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            members.update(vars(klass))
        owners: dict[Selector, str] = {}
        aux_types: dict[str, str] = {}
        finishes = []
        for attr, member in members.items():
            if getattr(member, "__yatf_finish__", False):
                finishes.append(attr)
            for selector in getattr(member, "__yatf_selectors__", ()):
                if selector in owners and owners[selector] != attr:
                    raise YatfError(f"{cls.__name__}.{owners[selector]} and .{attr} take the same records")
                owners[selector] = attr
                if selector.aux is not None:
                    known = aux_types.setdefault(selector.aux, selector.media_type)
                    if known != selector.media_type:
                        raise YatfError(f"aux file {selector.aux!r} is declared as both {known} and "
                                        f"{selector.media_type}")
        if len(finishes) > 1:
            raise YatfError(f"{cls.__name__} has more than one @finish method: {', '.join(finishes)}")
        cls.spec = AdapterSpec(
            name=name or f"{ns}-yatf", native_format=native_format, version=version,
            usage_input_includes_cache=usage_input_includes_cache, tested_with=tuple(tested_with),
            lossy=tuple(lossy), namespace=ns, aux=dict(aux_types),
            handlers=tuple((selector, attr) for selector, attr in owners.items()),
            finish=finishes[0] if finishes else None)
        return cls
    return declare


class Adapter:
    """Base of a yatf/1 adapter. One instance serves an A2A context across its turns; per-turn state resets in
    ``@finish``. To emit an event later than its record, keep ``turn.source`` and pass it as ``derived_from``."""

    spec: ClassVar[AdapterSpec]

    def __init__(self, turn: YatfTurn) -> None:
        """Set up for the context whose first turn here is ``turn``."""

    def unmatched(self, record: Any, turn: YatfTurn) -> None:
        """A native record no handler takes, kept as a ``<namespace>/raw`` event so nothing is lost."""
        turn.raw(record)

    @classmethod
    def from_yatf(cls, events: list[Event]) -> Native:
        """A context's events, every turn so far, back to native."""
        raise NotImplementedError(f"{cls.__name__} doesn't convert yatf/1 back to its native format")

    @classmethod
    def normal_form(cls, records: list[Any]) -> list[Any]:
        """What the round trip compares: by default the parsed records themselves."""
        return list(records)
