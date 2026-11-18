"""The always_on step waits for the world to settle by following AI-played contacts' real progress: a contact's reply
is in progress until it lands or its time limit passes (the one it was registered with), one trigger's replies run side
by side, and replies still in progress at the end are recorded."""

from __future__ import annotations

import random
from datetime import timedelta

import pytest

from agent_env.env.gateway.clock import _iso, _parse_rfc3339
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
                   "contact_timeouts": [{"name": "crew", "timeout_seconds": 120.0}]}


@pytest.mark.asyncio
async def test_a_reply_that_notifies_the_agent_still_runs_side_by_side_so_the_jump_stops_at_the_quick_one(
        monkeypatch, tmp_path):
    """As a host's reply on a booking site: the contact answers, then the app notifies the agent. Both replies go to one
    contact conversation. The second, with the shorter delay, used to queue behind the first and show no due time, so
    the step jumped straight to the first's and the quick reply landed with it, 17 task minutes late."""
    draws = random.Random(30)
    slow, quick = (draws.expovariate(1 / 2400) for _ in range(2))  # 1859 s, then 819 s
    quick_due = _iso(_parse_rfc3339("2026-11-09T14:00:00Z") + timedelta(seconds=quick))

    def texts_two_hosts(n, texts, rig):
        if n == 1:
            rig.provoke("messaging_send_message", {"conversation_id": "CV-000004"})
            rig.provoke("messaging_send_message", {"conversation_id": "CV-000002"})
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=texts_two_hosts)
    await rig.arm()
    rig.app_tools("messaging_send_message")
    host_replies = {**CREW_DM, "id": "host-replies",
                    "actions": [*CREW_DM["actions"], {"type": "notify", "channel": "Stayncook", "text": "New message"}]}
    rig.triggers(host_replies, executors={"crew": rig.contact("crew", seconds=0, reply=_crew_texted)})
    ctx = await rig.run(rig.step())

    first_jump = next(j for j in rig.events(ctx, "clock_jump") if j["from"] != j["to"])
    assert first_jump["to"] <= quick_due
    assert [c["at"][:19] for c in rig.contact_calls] == [quick_due[:19], "2026-11-09T14:30:58"]
    woke = [(w["virtual_time"][:19], w["parts_text"].count("Sam: reply")) for w in rig.events(ctx, "wake")[1:]]
    assert woke == [("2026-11-09T14:13:39", 1), ("2026-11-09T14:30:58", 1)]
    assert slow > quick


@pytest.mark.asyncio
async def test_each_contact_keeps_the_limit_it_was_registered_with_and_the_longest_caps_a_settle(monkeypatch,
                                                                                                    tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    rig.triggers(_ask_live_oak("2026-11-09T15:00:00Z"),
                 {**_ask_live_oak("2026-11-09T16:00:00Z"), "id": "northwick-quote",
                  "actions": [{"type": "nl", "executor": "northwick", "instruction": "quote Tess"}]},
                 executors={"liveoak": rig.contact("liveoak", seconds=0, reply=_estimate),
                            "northwick": rig.contact("northwick", seconds=0, reply=_estimate, timeout_seconds=600)})
    step = rig.step()
    ctx = await rig.run(step)

    # liveoak was registered with no limit of its own: the gateway's 120 s
    assert [(c["contact"], c["timeout_seconds"]) for c in rig.contact_calls] == [("liveoak", 120), ("northwick", 600)]
    settings = rig.record(ctx)["settings"]
    assert settings["contacts"]["longest_timeout_seconds"] == 600
    assert settings["wake_reserve_seconds"] == step.wake_reserve_seconds(600) > step.wake_reserve_seconds()


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


def test_the_contact_setting_survives_a_round_trip_and_the_step_sets_no_contact_time_limit():
    step = ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt_id="p", agent_name="a",
                               contact_replies_in_parallel=False)
    stored = step.to_dict()
    assert ao.AlwaysOnTaskStep.from_dict(stored).contact_replies_in_parallel is False
    assert "contact_timeout_seconds" not in stored  # a contact's limit is register_env_triggers' executor_timeout_seconds
    with pytest.raises(TypeError, match="contact_timeout_seconds"):
        ao.AlwaysOnTaskStep("week", None, user_sim="linda", agent_name="a", contact_timeout_seconds=600)
