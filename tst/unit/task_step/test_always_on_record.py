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


@pytest.mark.asyncio
async def test_the_process_log_carries_no_one_s_words_only_the_timeline_does(monkeypatch, tmp_path, caplog):
    def agent(n, texts, rig):
        if n == 1:
            rig.message("SECRET-SENT house at 12 Oak St, $855")
            return "SECRET-PLAIN I'll book it"
        rig.quiet()
        return "SECRET-CLOSING bye"

    rig = Rig(monkeypatch, tmp_path, agent=agent,
              answers=[{"outcome": "reply", "message": "SECRET-REPLY yes, card ending 4242", "used": "SECRET-USED"}])
    await rig.arm()
    rig.triggers(rig.notify_at("sam", "2026-11-09T15:00:00Z", "SECRET-ALERT Sam's address", channel="noor 30"))
    caplog.set_level("INFO", logger=ao.logger.name)
    step = ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt_id="ask", agent_name="solver")
    ctx = rig.context(user={"opening": "SECRET-PROMPT sort the cdmx weekend"})
    ctx = await rig.run(step, ctx)

    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == ao.logger.name)
    assert "always_on wake" in logged and "always_on user_sim_read" in logged
    assert "SECRET" not in logged
    stored = json.dumps(rig.timeline(ctx))
    for marker in ("SECRET-SENT", "SECRET-PLAIN", "SECRET-REPLY", "SECRET-ALERT", "SECRET-PROMPT"):
        assert marker in stored


def _fails_after_fifty_long_messages(n, texts, rig):
    for i in range(50):
        rig.message(f"option {i}: " + "x" * 2000)
    raise TimeoutError("the agent's turn ran out")


@pytest.mark.asyncio
async def test_when_storing_fails_the_kept_events_are_small_at_any_depth(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_fails_after_fifty_long_messages)
    await rig.arm()
    store = get_config().get_object_store()

    def fail(*a, **kw):
        raise OSError("bucket gone")

    monkeypatch.setattr(type(store), "put", fail)
    ctx = await rig.run(rig.step())

    record = rig.record(ctx)
    assert record["pending_at_end"]["unread_by_user"]["count"] == 50
    end = record["timeline_tail"][-1]
    assert end["event"] == "end" and end["pending"]["unread_by_user"][-1] == "(and 30 more)"

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for v in value.values():
                yield from strings(v)
        elif isinstance(value, list):
            for v in value:
                yield from strings(v)

    assert max(len(s) for s in strings(record)) <= ao.PENDING_TEXT_CHARS + 1
    assert len(json.dumps(record, default=str)) < 60_000


@pytest.mark.asyncio
async def test_the_prompt_response_keeps_each_wake_s_parts_cut_to_size(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "TURN_TEXT_TOTAL_CHARS", 9000)
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    for day in (10, 11, 12):
        rig.triggers(rig.notify_at(f"news-{day}", f"2026-11-{day}T14:00:00Z", "y" * 10_000, channel="Gmail"))
    ctx = await rig.run(rig.step(wake_reminder=True))  # its parts add up past the total with the reminder

    kept = ctx.prompt_responses[0].source_agent_per_turn_prompt_parts
    assert len(kept) == 4
    alert = next(p["text"] for p in kept[1] if p["text"].startswith("[Tue Nov 10"))
    assert len(alert) == ao.TURN_TEXT_CHARS + len("… (cut)") and alert.endswith("… (cut)")
    assert kept[3][-1]["text"] == "(cut: the run's stored timeline has this wake's full text)"
    assert "y" * 10_000 in rig.events(ctx, "wake")[1]["parts_text"]  # the timeline keeps it whole


@pytest.mark.asyncio
async def test_reads_whose_answer_broke_the_format_are_counted(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        rig.message(f"question {n}?") if n <= 2 else rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent,
              answers=[{"outcome": "reply", "message": "hm", "schema_errors": ["no JSON object in the answer"]},
                       {"outcome": "no_reply", "message": "", "schema_errors": []}])
    await rig.arm()
    ctx = await rig.run(rig.step())

    assert rig.record(ctx)["counts"]["user_sim_malformed_answers"] == 1


@pytest.mark.asyncio
async def test_a_read_retried_after_a_malformed_first_answer_keeps_both_and_is_counted(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        rig.message("question?") if n == 1 else rig.quiet()
        return ""

    first = {"outcome": "reply", "message": "{oops", "schema_errors": ["no JSON object in the answer"]}
    rig = Rig(monkeypatch, tmp_path, agent=agent,
              answers=[{"outcome": "no_reply", "message": "", "schema_errors": [], "retried": True,
                        "first_answer": first, "turns": 2}])
    await rig.arm()
    ctx = await rig.run(rig.step())

    read = next(e for e in rig.timeline(ctx) if e["event"] == "user_sim_read")
    assert read["retried"] is True and read["first_answer"] == first and read["user_sim_turns"] == 2
    assert rig.record(ctx)["counts"]["user_sim_malformed_answers"] == 1
