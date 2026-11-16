"""always_on with ``user_sim_gate``: an answer of the user sim's that reads as the assistant's is held back, the user
sim answers again in its session, and an answer that still fails never reaches the agent and ends the run."""

from __future__ import annotations

import json

import pytest

from tst.unit.task_step.always_on_harness import Rig
from tst.unit.task_step.test_prompt_agent_user_side import _SWAP

pytestmark = pytest.mark.usefixtures("local_stores")

NOTE = "[Conversation system note, not from the assistant.] Your last message was not sent"


def _answer(outcome: str, message: str = "") -> str:
    return json.dumps({"outcome": outcome, "message": message, "used": "none"})


def _messages_once(n, texts, rig):
    if n == 1:
        rig.message("here's the plan for the weekend")
        return "sent"
    rig.quiet()
    return ""


def _gated(rig: Rig, **gate):
    return rig.step(max_wakes=4, user_sim_gate={"patterns": True, **gate})


def _got(rig: Rig, text: str) -> bool:
    return any(text in t for wake in rig.sent for t in wake)


@pytest.mark.asyncio
async def test_an_answer_that_reads_as_the_assistants_is_held_back_and_the_rewrite_reaches_the_agent(monkeypatch,
                                                                                                    tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_messages_once, answers=[{"outcome": "reply", "message": _SWAP}],
              user_texts=(_answer("reply", "do those numbers hold up?"),))
    await rig.arm()
    ctx = await rig.run(_gated(rig))

    assert [c["parts"][0]["text"][:len(NOTE)] for c in rig.user_calls] == [NOTE]   # asked again, in her session
    assert _got(rig, "do those numbers hold up?") and not _got(rig, "None of the three reported numbers")
    (read, *_) = rig.events(ctx, "user_sim_read")
    assert read["message"] == "do those numbers hold up?"
    gate = read["user_sim_gate"]
    assert [(v["attempt"], v["passed"]) for v in gate["verdicts"]] == [(0, False), (1, True)]
    assert gate["first_message"] == _SWAP and gate["invalid"] is False
    assert [r["message"] for r in gate["rewrites"]] == ["do those numbers hold up?"]
    assert "user_sim_invalid" not in ctx.metadata
    assert rig.record(ctx)["end_reason"] != "user_sim_invalid"


@pytest.mark.asyncio
async def test_an_answer_that_still_fails_never_reaches_the_agent_and_ends_the_run_as_an_error(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_messages_once, answers=[{"outcome": "reply", "message": _SWAP}],
              user_texts=(_answer("reply", _SWAP), _answer("reply", _SWAP)))
    await rig.arm()
    ctx = await rig.run(_gated(rig))

    assert not _got(rig, "None of the three reported numbers")
    assert len(rig.sent) == 1                      # no wake after hers: the run ended at her read
    record = rig.record(ctx)
    assert record["end_reason"] == "user_sim_invalid"
    assert record["error_type"] == "user_sim_invalid"   # verifiers skip a run that broke, as on a user sim error
    assert ctx.metadata["user_sim_invalid"]["week"]["wake"] == 1
    (read,) = rig.events(ctx, "user_sim_read")
    assert read["outcome"] == "no_reply" and read["user_sim_gate"]["invalid"] is True
    assert [v["passed"] for v in read["user_sim_gate"]["verdicts"]] == [False, False, False]


@pytest.mark.asyncio
async def test_on_fail_end_ends_the_run_without_marking_it_an_error(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_messages_once, answers=[{"outcome": "reply", "message": _SWAP}],
              user_texts=(_answer("reply", _SWAP),))
    await rig.arm()
    ctx = await rig.run(_gated(rig, max_rewrites=1, on_fail="end"))

    record = rig.record(ctx)
    assert record["end_reason"] == "user_sim_invalid" and record["error_type"] is None
    assert not _got(rig, "None of the three reported numbers")


@pytest.mark.asyncio
async def test_an_in_role_answer_passes_with_no_rewrite(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_messages_once, answers=[{"outcome": "reply", "message": "shorter, please"}])
    await rig.arm()
    ctx = await rig.run(_gated(rig))
    (read, *_) = rig.events(ctx, "user_sim_read")
    assert read["user_sim_gate"] == {"verdicts": [{"attempt": 0, "chars": 15, "passed": True}], "rewrites": [],
                                     "invalid": False}
    assert rig.user_calls == []


@pytest.mark.asyncio
async def test_without_the_gate_nothing_is_checked(monkeypatch, tmp_path):
    plain = Rig(monkeypatch, tmp_path, agent=_messages_once, answers=[{"outcome": "reply", "message": _SWAP}])
    await plain.arm()
    ctx = await plain.run(plain.step(max_wakes=4))
    assert "user_sim_gate" not in plain.events(ctx, "user_sim_read")[0]
    assert _got(plain, "None of the three reported numbers")   # unchanged without the gate
