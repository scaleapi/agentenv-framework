"""`virtual_time` on trajectory events: FakeNow-driven Clock, with the
unarmed case pinned byte-identical for clock-less deployments."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.gateway import Gateway

T0 = "2026-01-01T00:00:00Z"
# The reference rate: 1 real second == 1 virtual day.
RATE = 86400


class FakeNow:
    """Monotonic-clock stand-in, so tests never sleep."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def gateway(tmp_path, monkeypatch) -> Gateway:
    """A Gateway logging to tmp_path (module attr patched before construction —
    the env var is read at import time, so patching it here would be too late)."""
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    gw = Gateway(
        host="127.0.0.1",
        port=0,
        server_name="AgentEnvGateway",
        internal_mcp_servers=[],
    )
    gw._clock = Clock(now_fn=FakeNow())
    yield gw
    gw._close_trajectory_file()


def _logged(gw: Gateway) -> list[dict]:
    """Every event the gateway has flushed, parsed back off disk."""
    gw._trajectory_file.flush()
    with open(gw._trajectory_file.name) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_unarmed_clock_leaves_events_byte_identical(gateway):
    """No `virtual_time` key at all when the clock is unarmed — not a null."""
    event_id = asyncio.run(gateway._log_event({"event_type": "tool_call"}))

    (event,) = _logged(gateway)
    assert "virtual_time" not in event
    assert set(event) == {"event_type", "event_id", "timestamp_utc"}
    assert event["event_id"] == event_id == "event_1"


def test_armed_clock_stamps_the_virtual_instant(gateway):
    """The stamp is the clock's own reading, not an extrapolation."""
    now = FakeNow()
    gateway._clock = Clock(now_fn=now)
    gateway._clock.set_time(T0, RATE)

    now.tick(1.0)  # one real second == one virtual day at 86400x
    asyncio.run(gateway._log_event({"event_type": "tool_call"}))

    (event,) = _logged(gateway)
    assert datetime.fromisoformat(event["virtual_time"]) == datetime(
        2026, 1, 2, tzinfo=timezone.utc
    )
    # The real stamp is still there and still real — the two are not redundant.
    assert (
        datetime.now(timezone.utc) - datetime.fromisoformat(event["timestamp_utc"])
    ) < timedelta(minutes=5)


def test_virtual_time_advances_across_events_while_wall_clock_barely_moves(gateway):
    """The point of the stamp: ordering in task-world time, over a ~ms wall run."""
    now = FakeNow()
    gateway._clock = Clock(now_fn=now)
    gateway._clock.set_time(T0, RATE)

    async def _three() -> None:
        for _ in range(3):
            await gateway._log_event({"event_type": "tool_call"})
            now.tick(1.0)

    asyncio.run(_three())

    stamps = [datetime.fromisoformat(e["virtual_time"]) for e in _logged(gateway)]
    assert stamps == [
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 2, tzinfo=timezone.utc),
        datetime(2026, 1, 3, tzinfo=timezone.utc),
    ]


def test_clearing_the_clock_stops_the_stamp(gateway):
    """Disarming returns to the unarmed shape rather than freezing the last read."""
    gateway._clock.set_time(T0, RATE)
    asyncio.run(gateway._log_event({"event_type": "tool_call"}))
    gateway._clock.clear()
    asyncio.run(gateway._log_event({"event_type": "tool_call_result"}))

    armed, disarmed = _logged(gateway)
    assert "virtual_time" in armed
    assert "virtual_time" not in disarmed


def test_stamp_is_independent_of_event_type(gateway):
    """`trigger_fired` markers and result events get it too, not just tool calls."""
    gateway._clock.set_time(T0, RATE)

    async def _each() -> None:
        for kind in ("tool_call", "tool_call_result", "trigger_fired"):
            await gateway._log_event({"event_type": kind})

    asyncio.run(_each())

    events = _logged(gateway)
    assert [e["event_type"] for e in events] == [
        "tool_call",
        "tool_call_result",
        "trigger_fired",
    ]
    assert all("virtual_time" in e for e in events)
