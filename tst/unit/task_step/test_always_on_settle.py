"""The always_on step waits for the world to settle by following AI-played contacts' real progress: a contact's reply
is in progress until it lands or its time limit passes, one trigger's replies run side by side, contacts that set no
time limit get 300 s, and replies still in progress at the end are recorded."""

from __future__ import annotations

import random

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")


def _estimate(rig, n):
    rig.alert(f"Live Oak: estimate {n} attached, $2,400", channel="Gmail · Live Oak Arbor")


def _ask_live_oak(at: str) -> dict:
    return {"id": "liveoak-estimate", "when": {"type": "time", "at": at},
            "actions": [{"type": "nl", "executor": "liveoak", "instruction": "send Maya your estimate"}]}


@pytest.mark.asyncio
async def test_a_contact_slower_than_a_fixed_wait_still_wakes_the_agent_with_its_reply(monkeypatch, tmp_path):
    """Scaled down: settle_timeout_seconds (0.2 s) stands in for the old fixed wait and the old 30 s cut-off; the
    contact takes 0.6 s, well inside its own 5 s limit."""
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    liveoak = rig.contact("liveoak", seconds=0.6, reply=_estimate, timeout_seconds=5)
    rig.triggers(_ask_live_oak("2026-11-09T15:00:00Z"), executors={"liveoak": liveoak})
    ctx = await rig.run(rig.step(settle_timeout_seconds=0.2))

    wakes = rig.events(ctx, "wake")
    assert [w["virtual_time"] for w in wakes][:2] == ["2026-11-09T14:00:00Z", "2026-11-09T15:00:00Z"]
    assert "Live Oak: estimate 1 attached, $2,400" in wakes[1]["parts_text"]
    [jump] = [j for j in rig.events(ctx, "clock_jump") if j["to"] == "2026-11-09T15:00:00Z"]
    assert jump["settled"] and jump["settle"]["seconds"] >= 0.5 and jump["settle"]["contact_calls_seen"] == 1
    assert rig.events(ctx, "settle_timeout") == []


@pytest.mark.asyncio
async def test_the_wait_stops_once_a_contact_overruns_its_limit_and_says_what_was_still_at_work(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "CONTACT_GRACE_SECONDS", 0.2)
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    stuck = rig.contact("liveoak", seconds=5, reply=_estimate, timeout_seconds=0.3, keeps_to_its_limit=False)
    rig.triggers(_ask_live_oak("2026-11-09T15:00:00Z"), executors={"liveoak": stuck})
    try:
        ctx = await rig.run(rig.step(settle_timeout_seconds=0.2))
    finally:
        await rig.close()

    [jump] = [j for j in rig.events(ctx, "clock_jump") if j["to"] == "2026-11-09T15:00:00Z"]
    assert not jump["settled"] and 0.4 <= jump["settle"]["seconds"] < 2
    [[trigger]] = [env["triggers"] for env in jump["still_at_work"]]
    assert trigger["id"] == "liveoak-estimate" and trigger["executor_calls"][0]["executor"] == "liveoak"


def _crew_texted(rig, n):
    rig.alert(f"Sam: reply {n}", channel="Messages · noor 30")


CREW_DM = {"id": "crew-dm", "when": {"type": "action", "tool": "messaging_send_message", "repeat": True},
           "actions": [{"type": "nl", "executor": "crew", "instruction": "answer Linda's newest text",
                        "delay": {"dist": "exp", "mean": "PT40M", "seed": 30}}]}


@pytest.mark.parametrize("parallel", [True, False], ids=["side by side", "one at a time"])
@pytest.mark.asyncio
async def test_one_trigger_s_replies_run_side_by_side_so_a_quick_one_lands_on_time(monkeypatch, tmp_path, parallel):
    draws = random.Random(30)
    slow, quick = (round(draws.expovariate(1 / 2400)) for _ in range(2))  # 1859 s, then 819 s
    assert quick < slow

    def texts_two_friends(n, texts, rig):
        if n == 1:
            rig.provoke("messaging_send_message", {"user_id": "+13475550167"})
            rig.provoke("messaging_send_message", {"user_id": "+16465550123"})
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=texts_two_friends)
    await rig.arm()
    rig.app_tools("messaging_send_message")
    rig.triggers(CREW_DM, executors={"crew": rig.contact("crew", seconds=0, reply=_crew_texted)})
    ctx = await rig.run(rig.step(contact_replies_in_parallel=parallel))

    woke = [(w["virtual_time"][:19], w["parts_text"].count("Sam: reply")) for w in rig.events(ctx, "wake")[1:]]
    if parallel:
        assert woke == [("2026-11-09T14:13:39", 1), ("2026-11-09T14:30:58", 1)]
    else:
        assert woke == [("2026-11-09T14:30:58", 2)]
    [env] = rig.record(ctx)["settings"]["contacts"]["envs"]
    assert env == {"env_id": "world", "applied": True, "concurrent_fires": parallel,
                   "executor_timeout_seconds": 300.0}


@pytest.mark.asyncio
async def test_a_contact_that_sets_no_limit_gets_300_s_and_one_that_does_keeps_its_own(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    rig.triggers(_ask_live_oak("2026-11-09T15:00:00Z"),
                 {**_ask_live_oak("2026-11-09T16:00:00Z"), "id": "northwick-quote",
                  "actions": [{"type": "nl", "executor": "northwick", "instruction": "quote Tess"}]},
                 executors={"liveoak": rig.contact("liveoak", seconds=0, reply=_estimate),
                            "northwick": rig.contact("northwick", seconds=0, reply=_estimate, timeout_seconds=120)})
    ctx = await rig.run(rig.step())

    assert [(c["contact"], c["timeout_seconds"]) for c in rig.contact_calls] == [("liveoak", 300), ("northwick", 120)]
    assert rig.record(ctx)["settings"]["contacts"]["timeout_seconds"] == ao.CONTACT_TIMEOUT_SECONDS == 300


@pytest.mark.asyncio
async def test_contact_replies_still_in_progress_at_the_end_are_recorded(monkeypatch, tmp_path):
    def texts_a_friend(n, texts, rig):
        rig.provoke("messaging_send_message", {"user_id": "+13475550167"})
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=texts_a_friend)
    await rig.arm()
    rig.app_tools("messaging_send_message")
    rig.triggers(CREW_DM, executors={"crew": rig.contact("crew", seconds=0, reply=_crew_texted)})
    try:
        ctx = await rig.run(rig.step(max_wakes=1))
    finally:
        await rig.close()

    [env] = rig.record(ctx)["pending_at_end"]["contact_replies_in_progress"]["items"]
    [trigger] = env["triggers"]
    assert trigger["id"] == "crew-dm" and trigger["running"] == 1
    assert trigger["waits"] == [{"resumes_at": "2026-11-09T14:30:58.882035Z"}]


def test_the_contact_settings_survive_a_round_trip_and_a_bad_limit_is_refused():
    step = ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt="hi", prompt_id="p", agent_name="a",
                               contact_timeout_seconds=600, contact_replies_in_parallel=False)
    again = ao.AlwaysOnTaskStep.from_dict(step.to_dict())
    assert (again.contact_timeout_seconds, again.contact_replies_in_parallel) == (600.0, False)
    with pytest.raises(ValueError, match="contact_timeout_seconds"):
        ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt="hi", agent_name="a", contact_timeout_seconds=0)
