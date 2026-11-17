"""Running an adapter: feed one A2A context's records through it, a turn at a time, and collect the events. The SDK runs
it live and the round-trip kit replays it offline, so the same records, fed in the same order, make the same events."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .adapter import Adapter, AdapterSpec, TurnContext, YatfTurn
from .models import DerivedFrom, Part, YatfError


@dataclass(frozen=True)
class Fed:
    """One record as it was fed: a native record, or a record of the aux file ``aux``."""

    record: Any
    aux: str | None = None


@dataclass(frozen=True, kw_only=True)
class RecordedTurn:
    """A stored turn: its context, its records in the order they were fed, and how it ended."""

    context: TurnContext
    records: Sequence[Fed] = ()
    state: str = "completed"
    final: Sequence[Part | Mapping[str, Any]] | None = None
    ended_at: str | None = None


class AdapterRun:
    """One A2A context's conversion. Each call returns the events it emitted, in order, so a live reader sees them as
    they come; ``events`` holds every event so far. Every read is a fresh copy, so a caller can't rewrite history."""

    def __init__(self, adapter: type[Adapter]) -> None:
        spec = vars(adapter).get("spec")
        if not isinstance(spec, AdapterSpec):
            raise YatfError(f"{adapter.__name__} isn't declared with @yatf_adapter")
        self._cls = adapter
        self._spec = spec
        self._adapter: Adapter | None = None
        self._turn: YatfTurn | None = None
        self._events: list[str] = []
        self._native = 0
        self._aux: dict[str, int] = {}

    @property
    def events(self) -> list[dict[str, Any]]:
        return [json.loads(text) for text in self._events]

    def start_turn(self, context: TurnContext) -> list[dict[str, Any]]:
        """Open a turn: write ``turn.start`` and the inbound message. The context's first turn creates the adapter."""
        if self._turn is not None:
            raise YatfError(f"turn {self._turn.index} is still open")
        turn = YatfTurn(context, self._spec)
        self._turn, self._native, self._aux = turn, 0, {}
        turn._start()
        if self._adapter is None:
            self._adapter = self._cls(turn)
        return self._drain(turn)

    def native(self, record: Any) -> list[dict[str, Any]]:
        """Feed the turn's next native record."""
        turn = self._open()
        source, self._native = DerivedFrom(index=self._native), self._native + 1
        self._handle(turn, source, self._spec.handler(None, record) or "unmatched", record)
        return self._drain(turn)

    def aux(self, name: str, record: Any) -> list[dict[str, Any]]:
        """Feed the turn's next record of aux file ``name``."""
        turn = self._open()
        if name not in self._spec.aux:
            raise YatfError(f"{self._cls.__name__} declares no aux file {name!r}")
        index = self._aux.get(name, 0)
        self._aux[name] = index + 1
        method = self._spec.handler(name, record)
        if method is not None:
            self._handle(turn, DerivedFrom(index=index, aux=name), method, record)
        return self._drain(turn)

    def end_turn(self, *, state: str = "completed", final: Sequence[Part | Mapping[str, Any]] | None = None,
                 ended_at: str | None = None) -> list[dict[str, Any]]:
        """Close the turn: run the adapter's ``@finish`` method, then write ``final_response`` and ``turn.end``. The
        frame is written even when ``@finish`` raises, and the error is raised after it."""
        turn = self._open()
        try:
            if self._spec.finish is not None:
                getattr(self._adapter, self._spec.finish)(turn)
        finally:
            turn._end(state, final, ended_at)
            self._turn = None
            new = self._drain(turn)
        return new

    def _open(self) -> YatfTurn:
        if self._turn is None:
            raise YatfError("no turn is open: call start_turn first")
        return self._turn

    def _handle(self, turn: YatfTurn, source: DerivedFrom, method: str, record: Any) -> None:
        turn.source = source
        try:
            getattr(self._adapter, method)(record, turn)
        finally:
            turn.source = None

    def _drain(self, turn: YatfTurn) -> list[dict[str, Any]]:
        new = turn._drain()
        self._events.extend(new)
        return [json.loads(text) for text in new]


def replay(adapter: type[Adapter], turns: Iterable[RecordedTurn]) -> list[dict[str, Any]]:
    """A context's stored turns run through ``adapter`` again, as the agent ran them live."""
    run = AdapterRun(adapter)
    for turn in turns:
        run.start_turn(turn.context)
        for fed in turn.records:
            if fed.aux is None:
                run.native(fed.record)
            else:
                run.aux(fed.aux, fed.record)
        run.end_turn(state=turn.state, final=turn.final, ended_at=turn.ended_at)
    return run.events
