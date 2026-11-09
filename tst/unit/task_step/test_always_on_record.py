"""What the always_on step keeps in the run's metadata (a short summary, small enough for a running job to report back)
and what it stores in full (the timeline, in the object store)."""

from __future__ import annotations

import json

import pytest

from agent_env.config import get_config
from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")


def _chatty(n, texts, rig):
    rig.message(f"update {n}: " + "x" * 2000)
    return "sent"


@pytest.mark.asyncio
async def test_the_run_record_is_a_summary_and_the_full_timeline_is_stored(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_chatty,
              answers=[{"outcome": "reply", "message": f"ok {i}"} for i in range(30)])
    await rig.arm()
    ctx = await rig.run(rig.step(max_wakes=20))

    record = rig.record(ctx)
    assert "timeline" not in record
    assert len(json.dumps(record, default=str)) < 10_000
    stored = rig.stored(ctx)
    assert stored["end_reason"] == record["end_reason"] == "max_wakes"
    assert len(stored["timeline"]) == record["timeline_events"] > 60
    assert sum(1 for e in stored["timeline"] if e["event"] == "wake") == 20


@pytest.mark.asyncio
async def test_when_the_timeline_cannot_be_stored_the_record_keeps_its_last_events(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_chatty,
              answers=[{"outcome": "reply", "message": f"ok {i}"} for i in range(30)])
    await rig.arm()
    store = get_config().get_object_store()

    def fail(*a, **kw):
        raise OSError("bucket gone")

    monkeypatch.setattr(type(store), "put", fail)
    ctx = await rig.run(rig.step(max_wakes=20))

    record = rig.record(ctx)
    assert "object_url" not in record and record["timeline_store_error"] == "OSError: bucket gone"
    assert len(record["timeline_tail"]) == ao.TIMELINE_TAIL and record["timeline_tail"][-1]["event"] == "end"
    assert not any(set(e) & set(ao.BULKY_EVENT_FIELDS) for e in record["timeline_tail"])
