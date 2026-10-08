"""rubric_evidence_with_mistakes: one judge call, a second trajectory_mistakes-shaped entry under the derived key."""
from __future__ import annotations

import json

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    JudgeOutputFormat,
    JudgeResponseDiscrepancy,
    ResultRows,
    diagnose_rubric_evidence_mistakes_response,
    diagnose_rubric_evidence_response,
    diagnose_trajectory_mistakes_response,
    format_evidence_judge_correction_prompt,
    get_judge_output_format_spec,
    trajectory_mistakes_rows_from_evidence,
)
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    TRAJECTORY_MISTAKES_SUFFIX,
    RubricsVerifierTaskStep,
    trajectory_mistakes_key,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import (
    CompactionType,
    FrameStrategy,
    TrajectoryFilter,
    TrajectoryText,
)

_JPEG = "/9j/" + "A" * 300
CRITERIA = [{"id": "table-reserved", "description": "A table is reserved"},
            {"id": "party-size", "description": "Party size is four"},
            {"id": "no-autocomplete", "description": "No suggestion was accepted", "weight": -1}]
MISTAKES = {"findings": [{"severity": "high", "step": 4, "should": "type the name in full",
                          "description": "typed 'piz' and tapped the suggestion"},
                         {"severity": "low", "step": 9, "should": "scroll once", "description": "scrolled twice"}],
            "botched": False, "botched_reason": "", "summary": "mostly fine, one shortcut"}
WITH_MISTAKES = JudgeOutputFormat.RUBRIC_EVIDENCE_WITH_MISTAKES
# The key the second entry of the step below (verifier_id="rubric") lands under.
KEY = trajectory_mistakes_key("rubric")


def _span(tool: str, args: dict) -> dict:
    return {"name": tool, "attributes": {"gen_ai.operation.name": "execute_tool",
                                         "gen_ai.prompt": json.dumps({"tool": tool, "input": args}),
                                         "gen_ai.completion": json.dumps({"result": "ok", "screenshot": _JPEG})}}


def _filter(**kw) -> TrajectoryFilter:
    base: dict = dict(compaction_type=CompactionType.SCREENSHOT, frame_strategy=FrameStrategy.PER_CRITERION,
                      frame_budget=6, trajectory_text=TrajectoryText.ACTION_LOG)
    base.update(kw)
    return TrajectoryFilter(**base)


def _verifier(**kw) -> RubricsVerifierTaskStep:
    base: dict = dict(id="verify", version=1, criteria=CRITERIA, prompt_id="p1", use_agent_judge=False,
                      use_trajectory=True, trajectory_filter=_filter(), verifier_id="rubric",
                      output_format=WITH_MISTAKES)
    base.update(kw)
    return RubricsVerifierTaskStep(**base)


def _ctx(**pr) -> TaskStepContext:
    base = dict(prompt_id="p1", response="done", prompt_text="reserve", agent_trajectory_object_url="s3://b/raw.json")
    base.update(pr)
    return TaskStepContext(prompt_responses=[PromptResponse(**base)])


def _judged_rows(criteria: list[dict], *, mistakes: dict | None = MISTAKES) -> ResultRows:
    """What the with-mistakes diagnoser hands back: c1 seen, c2 missing, c3 (negative) not_applicable."""
    status = {"c1": "visible", "c2": "missing", "c3": "not_applicable"}
    rows = [{**c, "frame": f"[{c['id']}] action 3" if status[c["id"]] == "visible" else "NOT SHOWN",
             "evidence_status": status[c["id"]], "justification": f"about {c['id']}",
             "score": 0.0 if status[c["id"]] == "missing" else 1.0, "result": status[c["id"]] != "missing"}
            for c in criteria]
    return ResultRows(rows, reasoning="the frames", checks=[], mistakes=mistakes)


def _standalone_mistakes_step(**kw) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(id="m", version=1, criteria=CRITERIA, prompt_id="p1", use_agent_judge=False,
                                   use_trajectory=True, trajectory_filter=_filter(), verifier_id="standalone",
                                   output_format=JudgeOutputFormat.TRAJECTORY_MISTAKES, **kw)


def _capture_judge(monkeypatch, v: RubricsVerifierTaskStep, rows_for, retries: int = 0, discrepancies: list | None = None) -> dict:
    monkeypatch.setattr(v, "_read_trajectory_text",
                        lambda uri: json.dumps([_span("gui_tap", {"x": i}) for i in range(1, 12)]))
    captured: dict = {"calls": 0}

    async def fake_run(*, eval_prompt, context, model, criteria, image_blocks=None, **kw):
        captured["calls"] += 1
        captured.update(eval_prompt=eval_prompt, criteria=criteria)
        return (rows_for(criteria), retries, discrepancies or [], "s3://judge/trajectory.json")
    monkeypatch.setattr(v, "_run_judge_with_output_retries", fake_run)
    return captured


def _mistakes_prompt() -> str:
    return get_judge_output_format_spec("trajectory_mistakes").prompt_template_no_trajectory.substitute(
        agent_prompt="p", agent_response="r", criteria_json="[]")


# --------------------------------------------------------------------------- prompt + schema

@pytest.mark.asyncio
async def test_with_mistakes_prompt_carries_the_mistake_rules_once_and_asks_for_no_coverage(monkeypatch):
    v = _verifier()
    captured = _capture_judge(monkeypatch, v, _judged_rows)
    await v.execute(_ctx())
    text = captured["eval_prompt"]
    standalone = _mistakes_prompt()
    # The prose is the trajectory_mistakes block, present exactly once (anchored on its distinctive rules).
    for anchor in ("HARNESS FAULT IS NOT THE AGENT'S MISTAKE", "SEVERITY — choose the rung",
                   "WHAT YOU CANNOT SEE IS NOT A MISTAKE"):
        assert anchor in standalone and text.count(anchor) == 1, anchor
    for field in ('"findings"', '"botched"', '"botched_reason"', '"summary"', '"frame"', '"evidence_status"'):
        assert field in text, field
    assert '"coverage"' not in text and "Empty array if no criteria were provided" not in text
    assert text.index('"evidence_status"') < text.index("## Trajectory mistakes") < text.index('"findings"')
    assert captured["calls"] == 1


@pytest.mark.asyncio
async def test_plain_rubric_evidence_keeps_the_plain_prompt_and_schema(monkeypatch):
    v = _verifier(output_format=JudgeOutputFormat.RUBRIC_EVIDENCE)
    captured = _capture_judge(monkeypatch, v, lambda criteria: _judged_rows(criteria, mistakes=None))
    ctx = _ctx()
    await v.execute(ctx)
    text = captured["eval_prompt"]
    assert "## Trajectory mistakes" not in text and "HARNESS FAULT" not in text and '"findings"' not in text
    plain = get_judge_output_format_spec("rubric_evidence")
    assert plain.output_format["schema"]["required"] == ["reasoning", "checks", "results"]
    assert v._spec() is plain
    assert set(ctx.metadata["verifications"]) == {"rubric"}


def test_the_with_mistakes_format_is_its_own_registry_entry_with_the_trajectory_mistakes_fields_added():
    spec = get_judge_output_format_spec(WITH_MISTAKES)
    assert get_judge_output_format_spec("rubric_evidence_with_mistakes") is spec
    assert spec is not get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_EVIDENCE)
    combined = spec.output_format["schema"]
    mistakes = get_judge_output_format_spec("trajectory_mistakes").output_format["schema"]
    assert list(combined["properties"]) == ["reasoning", "checks", "results", "findings", "botched", "botched_reason", "summary"]
    assert combined["required"] == list(combined["properties"])
    assert combined["properties"]["findings"] == mistakes["properties"]["findings"]
    for name in ("botched", "botched_reason", "summary"):
        assert combined["properties"][name] == mistakes["properties"][name]
    assert "coverage" not in combined["properties"]
    # Stored exactly as rubric_evidence's entry is, and grounded the same way (the rows cite frames).
    assert spec.stored_format == "rubric_binary" and spec.cites_evidence
    plain = get_judge_output_format_spec(JudgeOutputFormat.RUBRIC_EVIDENCE)
    assert (plain.stored_format, plain.cites_evidence) == (spec.stored_format, spec.cites_evidence)
    assert combined["properties"]["results"] == plain.output_format["schema"]["properties"]["results"]


# --------------------------------------------------------------------------- the second entry

@pytest.mark.asyncio
@pytest.mark.parametrize("fmt, expected_keys", [
    (JudgeOutputFormat.RUBRIC_EVIDENCE, {"rubric"}),
    (WITH_MISTAKES, {"rubric", "rubric-trajectory-mistakes"}),
], ids=["rubric_evidence writes no second entry", "rubric_evidence_with_mistakes writes one"])
async def test_only_the_with_mistakes_format_writes_a_second_entry(monkeypatch, fmt, expected_keys):
    v = _verifier(output_format=fmt)
    _capture_judge(monkeypatch, v, lambda criteria: _judged_rows(criteria, mistakes=MISTAKES if fmt is WITH_MISTAKES else None))
    ctx = _ctx()
    await v.execute(ctx)
    assert set(ctx.metadata["verifications"]) == expected_keys
    assert ctx.metadata["verifications"]["rubric"]["judge_output_format"] == fmt.value


@pytest.mark.asyncio
async def test_execute_writes_the_mistakes_entry_from_the_same_call_under_the_derived_key(monkeypatch):
    v = _verifier()
    captured = _capture_judge(monkeypatch, v, _judged_rows)
    ctx = _ctx()
    await v.execute(ctx)
    assert captured["calls"] == 1
    assert KEY == "rubric" + TRAJECTORY_MISTAKES_SUFFIX == "rubric-trajectory-mistakes"
    main, second = ctx.metadata["verifications"]["rubric"], ctx.metadata["verifications"][KEY]

    # The main entry is stored as a rubric_evidence entry: no mistake fields leak onto it.
    assert main["format"] == "rubric_binary" and main["judge_output_format"] == "rubric_evidence_with_mistakes"
    assert main["reasoning"] == "the frames" and main["checks"] == []
    assert not {"findings", "summary", "botched", "source_verifier_id"} & set(main)
    assert [r["id"] for r in main["results"]] == ["table-reserved", "party-size", "no-autocomplete"]

    # The second entry: trajectory_mistakes shape, the same pointers, provenance.
    assert second["format"] == "trajectory_mistakes" and "judge_output_format" not in second
    assert second["source_verifier_id"] == "rubric"
    for legacy, neutral in (("judge_trajectory_s3_uri", "judge_trajectory_object_url"),
                            ("compact_trajectory_s3_uri", "compact_trajectory_object_url")):
        assert second[legacy] == second[neutral] == main[legacy] == main[neutral]
    assert main["judge_trajectory_object_url"] == "s3://judge/trajectory.json"
    assert not {"reasoning", "checks"} & set(second)

    # ...its rows are exactly what the standalone parser builds for the same findings + derived coverage
    # (coverage ids are the REAL criterion ids; the notes are the judge's justifications, written against
    # the short keys it saw).
    expected, disc = diagnose_trajectory_mistakes_response(json.dumps({
        **MISTAKES,
        "coverage": [{"id": "table-reserved", "shown": True, "note": "about c1"},
                     {"id": "party-size", "shown": False, "note": "about c2"},
                     {"id": "no-autocomplete", "shown": True, "note": "about c3"}],
    }), CRITERIA)
    assert disc is None and second["results"] == expected
    assert [r["id"] for r in second["results"]] == [
        "summary", "botched", "mistake_0", "mistake_1",
        "coverage_table-reserved", "coverage_party-size", "coverage_no-autocomplete"]
    mistake = second["results"][2]
    assert mistake["severity"] == "high" and mistake["step"] == 4 and mistake["weight"] < 0
    assert mistake["justification"].startswith("[high] type the name in full — typed 'piz'")
    assert second["results"][4]["justification"] == "shown: about c1"
    assert second["results"][5]["justification"] == "MISSING: about c2"
    # ...scored the way a trajectory_mistakes step scores (WEIGHTED_AVERAGE, as the downstream configures it)
    assert second["score"] == aggregate_score(expected, ScoreAggregator.WEIGHTED_AVERAGE) and 0 < second["score"] < 1


@pytest.mark.asyncio
async def test_the_mistakes_score_matches_a_standalone_trajectory_mistakes_step(monkeypatch):
    v = _verifier()
    _capture_judge(monkeypatch, v, _judged_rows)
    ctx = _ctx()
    await v.execute(ctx)
    second = ctx.metadata["verifications"][KEY]

    standalone = _standalone_mistakes_step(score_aggregator=ScoreAggregator.WEIGHTED_AVERAGE)
    _capture_judge(monkeypatch, standalone, lambda criteria: [dict(r) for r in second["results"]])
    ctx2 = _ctx()
    await standalone.execute(ctx2)
    assert ctx2.metadata["verifications"]["standalone"]["score"] == second["score"]
    assert ctx2.metadata["verifications"]["standalone"]["results"] == second["results"]


@pytest.mark.asyncio
async def test_coverage_rows_carry_the_real_ids_where_a_standalone_step_keeps_the_short_ids_the_judge_echoed(monkeypatch):
    v = _verifier()
    _capture_judge(monkeypatch, v, _judged_rows)
    ctx = _ctx()
    await v.execute(ctx)
    combined = ctx.metadata["verifications"][KEY]["results"]
    assert [r["id"] for r in combined if r["id"].startswith("coverage_")] == [f"coverage_{c['id']}" for c in CRITERIA]

    # A standalone trajectory_mistakes step stores what the judge wrote: the short keys it was shown.
    standalone = _standalone_mistakes_step()

    def echoed(criteria):
        rows, disc = diagnose_trajectory_mistakes_response(json.dumps({
            **MISTAKES, "coverage": [{"id": c["id"], "shown": True, "note": ""} for c in criteria]}), criteria)
        assert disc is None
        return rows
    _capture_judge(monkeypatch, standalone, echoed)
    ctx2 = _ctx()
    await standalone.execute(ctx2)
    rows = ctx2.metadata["verifications"]["standalone"]["results"]
    assert [r["id"] for r in rows if r["id"].startswith("coverage_")] == ["coverage_c1", "coverage_c2", "coverage_c3"]
    # ...the rows are otherwise the same shape, row for row
    assert [set(r) for r in rows] == [set(r) for r in combined]


@pytest.mark.asyncio
async def test_both_entries_carry_the_judge_retry_fields_when_there_were_retries(monkeypatch):
    disc = [{"expected_count": 3, "returned_count": 2}]

    async def run(retries: int, discrepancies: list) -> tuple[dict, dict]:
        v = _verifier()
        _capture_judge(monkeypatch, v, _judged_rows, retries=retries, discrepancies=discrepancies)
        ctx = _ctx()
        await v.execute(ctx)
        return ctx.metadata["verifications"]["rubric"], ctx.metadata["verifications"][KEY]

    main, second = await run(2, disc)
    assert main["judge_output_retries"] == second["judge_output_retries"] == 2
    assert main["judge_output_discrepancies"] == second["judge_output_discrepancies"] == disc
    main, second = await run(0, [])
    assert not {"judge_output_retries", "judge_output_discrepancies"} & (set(main) | set(second))


def test_coverage_is_derived_from_the_evidence_rows():
    def row(cid: str, status: str, score: float) -> dict:
        return {"id": cid, "evidence_status": status, "justification": f"j-{cid}", "score": score, "result": score == 1.0}
    rows = ResultRows([row("seen", "visible", 1.0), row("gone", "missing", 0.0), row("na", "not_applicable", 1.0),
                       row("seen-but-failed", "visible", 0.0), row("unsure", "ambiguous", 1.0),
                       row("blocked", "blocked", 0.0)],
                      mistakes={"findings": [], "botched": False, "botched_reason": "", "summary": "clean"})
    out = trajectory_mistakes_rows_from_evidence(rows)
    coverage = {r["id"]: (r["result"], r["justification"]) for r in out if r["id"].startswith("coverage_")}
    assert coverage == {
        "coverage_seen": (True, "shown: j-seen"),
        "coverage_gone": (False, "MISSING: j-gone"),
        "coverage_na": (True, "shown: j-na"),
        "coverage_seen-but-failed": (False, "MISSING: j-seen-but-failed"),   # a failed row is never "shown"
        "coverage_unsure": (False, "MISSING: j-unsure"),                     # only visible / not_applicable count
        "coverage_blocked": (False, "MISSING: j-blocked"),
    }
    assert [r["id"] for r in out[:2]] == ["summary", "botched"] and out[0]["result"] is True and out[0]["justification"] == "clean"
    assert aggregate_score(out, ScoreAggregator.WEIGHTED_AVERAGE) == pytest.approx((1 + 2) / 7)   # botched ok + 2 shown of 6
    with pytest.raises(ValueError, match="no mistakes"):
        trajectory_mistakes_rows_from_evidence(ResultRows(list(rows)))


@pytest.mark.asyncio
async def test_a_judge_response_without_findings_writes_neither_entry(monkeypatch):
    def entries(ctx: TaskStepContext) -> set:
        return set(ctx.metadata.get("verifications", {})) & {"rubric", KEY}
    v = _verifier()
    _capture_judge(monkeypatch, v, lambda criteria: _judged_rows(criteria, mistakes=None))
    ctx = _ctx()
    with pytest.raises(ValueError, match="no mistakes"):
        await v.execute(ctx)
    assert entries(ctx) == set()                          # the rubric entry is not left behind without its twin
    v = _verifier()
    _capture_judge(monkeypatch, v, lambda criteria: list(_judged_rows(criteria)))     # a plain list
    ctx = _ctx()
    with pytest.raises(RuntimeError, match="no mistake findings"):
        await v.execute(ctx)
    assert entries(ctx) == set()


@pytest.mark.asyncio
async def test_prompt_error_skips_both_entries_the_same_way():
    ctx = _ctx(response="", error_type="timeout")
    await _verifier().execute(ctx)
    main, second = ctx.metadata["verifications"]["rubric"], ctx.metadata["verifications"][KEY]
    assert main["format"] == "rubric_binary" and main["judge_output_format"] == "rubric_evidence_with_mistakes"
    assert second["format"] == "trajectory_mistakes" and second["source_verifier_id"] == "rubric"
    assert main["results"] == second["results"] and main["results"] is not second["results"]
    assert second["results"][0]["id"] == "prompt_error" and second["score"] == 0 and main["score"] == 0
    assert "judge_output_format" not in second
    ctx = _ctx(response="", error_type="timeout")
    await _verifier(output_format=JudgeOutputFormat.RUBRIC_EVIDENCE).execute(ctx)
    assert set(ctx.metadata["verifications"]) == {"rubric"}


# --------------------------------------------------------------------------- the derived key

@pytest.mark.asyncio
async def test_two_steps_write_under_their_own_derived_keys_and_a_rerun_replaces_its_own_entry(monkeypatch):
    ctx = _ctx()
    # A standalone trajectory_mistakes entry under a key of its own is untouched.
    ctx.metadata["verifications"] = {"standalone": {"format": "trajectory_mistakes", "results": [], "score": 0.4}}
    for vid in ("rubric", "other-rubric"):
        v = _verifier(verifier_id=vid)
        _capture_judge(monkeypatch, v, _judged_rows)
        await v.execute(ctx)
    verifications = ctx.metadata["verifications"]
    assert set(verifications) == {"standalone", "rubric", "rubric-trajectory-mistakes", "other-rubric",
                                  "other-rubric-trajectory-mistakes"}
    assert verifications["standalone"]["score"] == 0.4
    for vid in ("rubric", "other-rubric"):
        assert verifications[trajectory_mistakes_key(vid)]["source_verifier_id"] == vid
        assert "trajectory_mistakes_conflict" not in verifications[vid]
    # a re-run of the same step replaces its own second entry
    v = _verifier()
    _capture_judge(monkeypatch, v, lambda criteria: _judged_rows(criteria, mistakes={**MISTAKES, "findings": []}))
    await v.execute(ctx)
    assert verifications[KEY]["score"] > verifications["other-rubric-trajectory-mistakes"]["score"]
    assert [r["id"] for r in verifications[KEY]["results"] if r["id"].startswith("mistake_")] == []


def test_trajectory_mistakes_key_is_the_verifier_id_plus_the_suffix():
    assert TRAJECTORY_MISTAKES_SUFFIX == "-trajectory-mistakes"
    assert trajectory_mistakes_key("abc") == "abc-trajectory-mistakes"
    v = _verifier(verifier_id=None)                                             # a generated id derives the same way
    assert trajectory_mistakes_key(v.verifier_id) == v.verifier_id + TRAJECTORY_MISTAKES_SUFFIX


# --------------------------------------------------------------------------- config

@pytest.mark.parametrize("fmt", [JudgeOutputFormat.RUBRIC_EVIDENCE, WITH_MISTAKES])
@pytest.mark.parametrize("bad", [
    dict(trajectory_filter=None),
    dict(trajectory_filter=TrajectoryFilter()),
    dict(trajectory_filter=_filter(frame_strategy=FrameStrategy.FINAL_FRAMES)),
    dict(use_agent_judge=True),
    dict(use_trajectory=False),
])
def test_both_evidence_formats_require_per_criterion_frames_on_the_direct_judge(fmt, bad):
    with pytest.raises(ValueError, match=f"output_format={fmt.value} requires the direct LLM judge"):
        _verifier(output_format=fmt, **bad)
    _verifier(output_format=fmt)                                                # the valid combination constructs


def test_the_old_knob_is_gone():
    with pytest.raises(TypeError):
        _verifier(include_trajectory_mistakes=True)
    assert "include_trajectory_mistakes" not in _verifier().to_dict()


def test_the_with_mistakes_format_round_trips_through_to_dict_and_from_dict():
    d = _verifier().to_dict()
    assert d["output_format"] == "rubric_evidence_with_mistakes"
    back = RubricsVerifierTaskStep.from_dict({**d, "default_model": "m"})
    assert back.output_format is WITH_MISTAKES
    assert back._spec().output_format["schema"]["required"][-1] == "summary" and back._writes_mistakes_entry
    # ...and the plain format string on a stored step selects it too
    stored = RubricsVerifierTaskStep.from_dict({
        "id": "x", "version": 1, "criteria": CRITERIA, "prompt_id": "p", "default_model": "m", "verifier_id": "r",
        "use_agent_judge": False, "trajectory_filter": _filter().to_dict(), "output_format": "rubric_evidence_with_mistakes"})
    assert stored.output_format is WITH_MISTAKES and stored._spec() is back._spec()
    plain = RubricsVerifierTaskStep.from_dict({**d, "default_model": "m", "output_format": "rubric_evidence"})
    assert plain.output_format is JudgeOutputFormat.RUBRIC_EVIDENCE and not plain._writes_mistakes_entry
    legacy = RubricsVerifierTaskStep.from_dict({"id": "x", "version": 1, "criteria": CRITERIA, "prompt_id": "p",
                                                "default_model": "m"})
    assert legacy.output_format is JudgeOutputFormat.RUBRIC_BINARY and not legacy._writes_mistakes_entry


# --------------------------------------------------------------------------- parser + correction prompt

_CRIT = [{"id": "c1", "description": "a"}, {"id": "c2", "description": "b"}]


def _response(**extra) -> str:
    return json.dumps({"reasoning": "r", "checks": [], "results": [
        {"id": "c1", "frame": "[c1] action 3", "evidence_status": "visible", "justification": "j", "score": 1.0},
        {"id": "c2", "frame": "NOT SHOWN", "evidence_status": "missing", "justification": "j", "score": 1.0}], **extra})


def test_with_mistakes_parser_requires_the_mistake_fields_and_the_plain_one_ignores_them():
    rows, disc = diagnose_rubric_evidence_mistakes_response(_response(**MISTAKES), _CRIT)
    assert disc is None and rows.mistakes == MISTAKES and rows.reasoning == "r"
    assert [r["result"] for r in rows] == [True, False]                       # the evidence rules still apply
    rows, disc = diagnose_rubric_evidence_response(_response(**MISTAKES), _CRIT)
    assert disc is None and rows.mistakes is None                             # plain format: tolerated, not kept
    rows, disc = diagnose_rubric_evidence_response(_response(), _CRIT)
    assert disc is None and rows.mistakes is None
    # with mistakes: a missing / mistyped field is a discrepancy the correction loop retries
    rows, disc = diagnose_rubric_evidence_mistakes_response(_response(), _CRIT)
    assert rows is None and "findings, botched, summary" in disc.parse_error
    rows, disc = diagnose_rubric_evidence_mistakes_response(_response(**{**MISTAKES, "botched": "false"}), _CRIT)
    assert rows is None and disc.parse_error.endswith("botched")
    # botched_reason may be omitted; non-dict findings are dropped
    rows, _ = diagnose_rubric_evidence_mistakes_response(
        _response(findings=[MISTAKES["findings"][0], "junk"], botched=True, summary="s"), _CRIT)
    assert rows.mistakes == {"findings": [MISTAKES["findings"][0]], "botched": True, "botched_reason": "", "summary": "s"}
    # a bare results array has no mistakes to offer
    bare = json.dumps([{"id": "c1", "score": 1, "justification": "j"}, {"id": "c2", "score": 0, "justification": "j"}])
    assert diagnose_rubric_evidence_mistakes_response(bare, _CRIT)[0] is None
    assert diagnose_rubric_evidence_response(bare, _CRIT)[1] is None
    # the registry wires the two diagnosers to the two formats
    assert get_judge_output_format_spec(WITH_MISTAKES).diagnose_response is diagnose_rubric_evidence_mistakes_response
    assert get_judge_output_format_spec("rubric_evidence").diagnose_response is diagnose_rubric_evidence_response


def test_with_mistakes_correction_prompt_lists_the_mistake_fields():
    disc = JudgeResponseDiscrepancy(expected_count=2, returned_count=0, expected_ids=["c1", "c2"], missing_ids=[],
                                    duplicate_ids=[], unknown_ids=[], parse_error="missing or mistyped field(s): findings")
    kw = dict(eval_prompt="EVAL", previous_response="{}", discrepancy=disc)
    combined = get_judge_output_format_spec("rubric_evidence_with_mistakes").format_correction_prompt(**kw)
    plain = format_evidence_judge_correction_prompt(**kw)
    for name in ("findings", "botched", "botched_reason", "summary"):
        assert f'"{name}"' in combined and f'"{name}"' not in plain
    assert "`reasoning`, `checks`, `findings`, `botched`, `botched_reason`, `summary` and a `results` array" in combined
    assert "`reasoning`, `checks` and a `results` array" in plain
    assert combined == format_evidence_judge_correction_prompt(**kw, with_mistakes=True)
    assert get_judge_output_format_spec("rubric_evidence").format_correction_prompt(**kw) == plain
