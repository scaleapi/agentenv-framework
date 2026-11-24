"""always_on with ``user_sim_gate``: an answer of the user sim's that reads as the assistant's is held back, the user
sim answers again in its session, and an answer that still fails never reaches the agent and ends the run."""

from __future__ import annotations

import json

import pytest

from tst.unit.task_step.always_on_harness import Rig
from tst.unit.task_step.test_prompt_agent_user_side import _SWAP, pattern_judge  # noqa: F401

pytestmark = pytest.mark.usefixtures("local_stores", "pattern_judge")

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
    return rig.step(max_wakes=4, user_sim_gate={"patterns": True, "judge_model": "judge", **gate})


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
    assert read["first_message"] == _SWAP   # where prompt_agent's read keeps it too
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
    assert record["end_reason"] == "user_sim_error"     # a failed user sim: the run is redone
    assert record["error_type"] == "user_sim_error"
    assert ctx.metadata["user_sim_invalid"]["week"]["wake"] == 1
    (read,) = rig.events(ctx, "user_sim_read")
    assert read["outcome"] == "no_reply" and read["user_sim_gate"]["invalid"] is True
    assert [v["passed"] for v in read["user_sim_gate"]["verdicts"]] == [False, False, False]


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", [_answer("no_reply"), _answer("reply", "  ")])
async def test_a_rewrite_with_nothing_to_send_holds_the_message_back_and_fails_the_run(monkeypatch, tmp_path, rewrite):
    rig = Rig(monkeypatch, tmp_path, agent=_messages_once, answers=[{"outcome": "reply", "message": _SWAP}],
              user_texts=(rewrite,))
    await rig.arm()
    ctx = await rig.run(_gated(rig))

    assert not _got(rig, "None of the three reported numbers")
    record = rig.record(ctx)
    assert record["end_reason"] == "user_sim_error" and record["error_type"] == "user_sim_error"
    assert ctx.metadata["user_sim_invalid"]["week"]["reasons"][-1] == "its rewrite sent nothing in its place"
    (read,) = rig.events(ctx, "user_sim_read")
    assert read["outcome"] == "no_reply" and read["user_sim_gate"]["invalid"] is True


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
    assert read["user_sim_gate"] == {"verdicts": [{"attempt": 0, "chars": 15, "passed": True, "judge_in_role": True,
                                                   "judge_reason": "written as the user"}],
                                     "rewrites": [], "invalid": False}
    assert rig.user_calls == []


@pytest.mark.asyncio
async def test_a_gate_without_a_judge_records_pattern_matches_and_holds_nothing_back(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_messages_once, answers=[{"outcome": "reply", "message": _SWAP}])
    await rig.arm()
    ctx = await rig.run(rig.step(max_wakes=4, user_sim_gate={}))
    (read, *_) = rig.events(ctx, "user_sim_read")
    assert read["user_sim_gate"]["verdicts"][0]["passed"] is True and read["user_sim_gate"]["verdicts"][0]["pattern_reasons"]
    assert _got(rig, "None of the three reported numbers") and rig.user_calls == []


@pytest.mark.asyncio
async def test_the_judge_reads_the_user_sims_own_instructions(monkeypatch, tmp_path):
    from agent_env.task_step.task_steps.user_sim_gate import UserSimGate

    seen = []

    async def judge(self, candidate, history, call_config, instructions=None, **_):
        seen.append(instructions)
        return {"in_role": True, "reason": "asks"}

    monkeypatch.setattr(UserSimGate, "_judge", judge)
    rig = Rig(monkeypatch, tmp_path, agent=_messages_once, answers=[{"outcome": "reply", "message": "shorter, please"}])
    await rig.arm()
    await rig.run(_gated(rig))
    assert len(seen) == 1 and seen[0] and "Linda" in seen[0]


@pytest.mark.asyncio
async def test_without_the_gate_nothing_is_checked(monkeypatch, tmp_path):
    plain = Rig(monkeypatch, tmp_path, agent=_messages_once, answers=[{"outcome": "reply", "message": _SWAP}])
    await plain.arm()
    ctx = await plain.run(plain.step(max_wakes=4))
    assert "user_sim_gate" not in plain.events(ctx, "user_sim_read")[0]
    assert _got(plain, "None of the three reported numbers")   # unchanged without the gate



@pytest.mark.asyncio
async def test_a_gate_whose_judge_cant_be_set_up_fails_the_step(monkeypatch, tmp_path):
    # A setup failure won't recover, and only the judge holds an answer back.
    def no_endpoint(model, **kw):
        raise KeyError("no endpoint configured for judge-model")

    monkeypatch.setattr("agent_env.config.model_call.resolve_route", no_endpoint)
    rig = Rig(monkeypatch, tmp_path, agent=_messages_once, answers=[{"outcome": "reply", "message": "looks good"}])
    await rig.arm()

    with pytest.raises(Exception, match="judge setup: KeyError.*only the judge holds"):
        await rig.run(rig.step(max_wakes=4, user_sim_gate={"judge_model": "judge-model"}))
    assert not _got(rig, "looks good")


class _OwnFormatPlugin:
    """A plugin whose user sim answers "REPLY: <text>" instead of our JSON, with its own parse."""

    def __init__(self, answers):
        self.answers = list(answers)

    def setup(self, version, person):
        return {"system_prompt": "you are linda", "output_format": None}

    async def answer(self, version, person, history, read, ask):
        return self.answers.pop(0)

    @staticmethod
    def parse(raw):
        text = raw if isinstance(raw, str) else ""
        return {"outcome": "reply", "message": text.removeprefix("REPLY: "), "used": "none"} if text else {}


@pytest.mark.asyncio
async def test_a_plugins_rewrite_is_read_with_its_own_parse_and_reaches_the_agent(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_messages_once, user_texts=("REPLY: do those numbers hold up?",))
    rig.plugin = _OwnFormatPlugin([{"outcome": "reply", "message": _SWAP}])
    await rig.arm()
    ctx = await rig.run(_gated(rig))

    assert _got(rig, "do those numbers hold up?")      # parsed its way, not read as an empty no_reply
    (read, *_) = rig.events(ctx, "user_sim_read")
    assert read["message"] == "do those numbers hold up?"


@pytest.mark.asyncio
async def test_a_gate_with_a_plugin_that_cant_read_its_rewrites_is_refused_when_the_run_starts(monkeypatch, tmp_path):
    class NoParse:
        @staticmethod
        def setup(version, person):
            return {"system_prompt": "you are linda", "output_format": None}

        @staticmethod
        async def answer(version, person, history, read, ask):
            return {"outcome": "reply", "message": "ok"}

    rig = Rig(monkeypatch, tmp_path, agent=_messages_once)
    rig.plugin = NoParse
    await rig.arm()
    with pytest.raises(ValueError, match="user_sim_gate needs the user sim's plugin to define parse"):
        await rig.run(_gated(rig))
