"""Per-criterion frames on the direct judge: filter knobs, execute wiring, context facts, the rubric_evidence format."""
from __future__ import annotations

import json

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    EVIDENCE_STATUSES,
    JudgeOutputFormat,
    JudgeResponseDiscrepancy,
    ResultRows,
    diagnose_rubric_evidence_response,
    evidence_checks_instruction,
    format_evidence_judge_correction_prompt,
    format_judge_correction_prompt,
    get_judge_output_format_spec,
    per_criterion_grounding,
)
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    CONTEXT_FACT_MAX_CHARS,
    CONTEXT_FACT_NOT_AVAILABLE,
    RubricsVerifierTaskStep,
    _restore_frame_labels,
    resolve_context_facts,
    trajectory_mistakes_key,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import (
    DEFAULT_ACTION_LOG_MAX_LINES,
    MAX_FRAME_BUDGET,
    MIN_ACTION_LOG_MAX_LINES,
    MIN_FRAME_BUDGET,
    CompactionType,
    FrameStrategy,
    TrajectoryFilter,
    TrajectoryText,
)

_JPEG = "/9j/" + "A" * 300
_BUDGET = 8
_PINS = {"typed": r"^ios_type\b"}


def _span(tool: str, args: dict, shot: str | None = _JPEG) -> dict:
    comp = {"result": "ok"}
    if shot:
        comp["screenshot"] = shot
    return {"name": tool, "attributes": {"gen_ai.operation.name": "execute_tool",
                                         "gen_ai.prompt": json.dumps({"tool": tool, "input": args}),
                                         "gen_ai.completion": json.dumps(comp)}}


def _raw_run(n: int = 30) -> str:
    spans = [{"attributes": {"gen_ai.operation.name": "chat",
                             "gen_ai.completion": json.dumps({"content": [
                                 {"type": "thinking", "thinking": "let me think"},
                                 {"type": "text", "text": "I will now book the table"}]})}}]
    spans += [_span("ios_tap", {"x": i, "y": i}) for i in range(1, n + 1)]
    spans[5] = _span("ios_type", {"text": "party of four"})                 # action 5
    spans[12] = _span("ios_tap_element", {"label": "Reserve table"})          # action 12
    return json.dumps(spans)


CRITERIA = [{"id": "table-reserved", "description": "A table is reserved", "backend": "final_state"},
            {"id": "party-size", "description": "Party size is four", "backend": "intermediate"}]


def _filter(**kw) -> TrajectoryFilter:
    base = dict(compaction_type=CompactionType.SCREENSHOT, frame_strategy=FrameStrategy.PER_CRITERION,
                frame_budget=_BUDGET, trajectory_text=TrajectoryText.ACTION_LOG, always_show_actions=_PINS)
    base.update(kw)
    return TrajectoryFilter(**base)


# --------------------------------------------------------------------------- filter knobs

def test_filter_defaults_preserve_old_behavior_and_round_trip():
    old = TrajectoryFilter.from_dict({"compaction_type": "screenshot", "screenshot_last_n": 3})
    assert old.frame_strategy == FrameStrategy.FINAL_FRAMES and old.trajectory_text == TrajectoryText.FULL
    assert old.always_show_actions is None and old.to_dict()["always_show_actions"] is None
    f = _filter(evidence_exclude_pattern=r"home screen")
    back = TrajectoryFilter.from_dict(f.to_dict())
    assert back == f and back.frame_strategy == FrameStrategy.PER_CRITERION
    assert back.trajectory_text == TrajectoryText.ACTION_LOG and back.frame_budget == _BUDGET
    assert back.always_show_actions == _PINS and back.to_dict()["always_show_actions"] == _PINS


def test_per_criterion_defaults_the_text_to_the_action_log_unless_set_explicitly():
    """The full text (up to MAX_JUDGE_TRAJECTORY_CHARS) next to ~90 frames is never what a per_criterion task
    wants, so it should not have to say so; final_frames keeps the full text and an explicit value always wins."""
    pc = dict(compaction_type=CompactionType.SCREENSHOT, frame_strategy=FrameStrategy.PER_CRITERION)
    assert TrajectoryFilter(**pc).trajectory_text is TrajectoryText.ACTION_LOG
    assert TrajectoryFilter(**pc, trajectory_text=TrajectoryText.FULL).trajectory_text is TrajectoryText.FULL
    assert TrajectoryFilter(**pc, trajectory_text="full").trajectory_text is TrajectoryText.FULL
    assert TrajectoryFilter(compaction_type=CompactionType.SCREENSHOT).trajectory_text is TrajectoryText.FULL
    assert TrajectoryFilter().trajectory_text is TrajectoryText.FULL and TrajectoryFilter().to_dict()["trajectory_text"] == "full"
    # from_dict: a document that omits the key gets the strategy's default; one that names `full` keeps it
    doc = {"compaction_type": "screenshot", "frame_strategy": "per_criterion"}
    assert TrajectoryFilter.from_dict(doc).trajectory_text is TrajectoryText.ACTION_LOG
    assert TrajectoryFilter.from_dict({**doc, "trajectory_text": "full"}).trajectory_text is TrajectoryText.FULL
    # to_dict writes the resolved value, so the round trip is exact and the stored document is self-describing
    f = TrajectoryFilter.from_dict(doc)
    assert f.to_dict()["trajectory_text"] == "action_log" and TrajectoryFilter.from_dict(f.to_dict()) == f
    with pytest.raises(ValueError):
        TrajectoryFilter(**pc, trajectory_text="nope")


def test_the_pre_rename_last_n_spelling_is_accepted_and_written_back_as_final_frames():
    for f in (TrajectoryFilter.from_dict({"compaction_type": "screenshot", "frame_strategy": "last_n"}),
              TrajectoryFilter(compaction_type="screenshot", frame_strategy="last_n")):
        assert f.frame_strategy is FrameStrategy.FINAL_FRAMES and f.to_dict()["frame_strategy"] == "final_frames"
    assert FrameStrategy("last_n") is FrameStrategy.FINAL_FRAMES
    with pytest.raises(ValueError):
        TrajectoryFilter(frame_strategy="nope")


def test_filter_validates_the_per_criterion_budget_and_the_regex_knobs():
    with pytest.raises(ValueError):
        _filter(frame_budget=MAX_FRAME_BUDGET + 1)
    with pytest.raises(ValueError):
        _filter(frame_budget=MIN_FRAME_BUDGET - 1)     # the final-frame reservation needs room left
    _filter(frame_budget=MIN_FRAME_BUDGET)
    for bad in (90.0, "90", True):                                    # JSON can hand over a float or a string
        with pytest.raises(ValueError, match="frame_budget must be an integer"):
            _filter(frame_budget=bad)
    # ...whatever the strategy: a task document may set the per_criterion knobs while toggling the strategy
    for other in (dict(frame_strategy=FrameStrategy.FINAL_FRAMES, compaction_type=CompactionType.SCREENSHOT),
                  dict(compaction_type=CompactionType.DEFAULT)):
        with pytest.raises(ValueError, match="frame_budget must be an integer"):
            TrajectoryFilter(frame_budget=MAX_FRAME_BUDGET + 1, **other)
        with pytest.raises(ValueError, match="evidence_exclude_pattern"):
            TrajectoryFilter(evidence_exclude_pattern="(unclosed", **other)
        TrajectoryFilter(frame_budget=MIN_FRAME_BUDGET, always_show_actions=_PINS, **other)   # ignored, but valid
    with pytest.raises(ValueError, match="evidence_exclude_pattern"):
        _filter(evidence_exclude_pattern="(unclosed")
    with pytest.raises(ValueError, match=r"always_show_actions\['typed'\] is not a valid regex"):
        _filter(always_show_actions={"typed": "(unclosed"})
    for bad in ({"": r"^ios_type"}, {"typed": 3}, ["^ios_type"], "^ios_type", {3: "x"}):
        with pytest.raises(ValueError, match="always_show_actions"):
            _filter(always_show_actions=bad)
    _filter(always_show_actions={})                                   # nothing pinned is a valid configuration
    # last_n bounds are only enforced for FINAL_FRAMES; per_criterion ignores them
    _filter(screenshot_last_n=999)


def test_action_log_line_cap_round_trips_and_is_validated():
    assert TrajectoryFilter.from_dict(_filter(action_log_max_lines=40).to_dict()).action_log_max_lines == 40
    assert TrajectoryFilter.from_dict({"compaction_type": "screenshot"}).action_log_max_lines == DEFAULT_ACTION_LOG_MAX_LINES
    for bad in (MIN_ACTION_LOG_MAX_LINES - 1, 0, -5, "40", 40.0, True):
        with pytest.raises(ValueError, match="action_log_max_lines"):
            _filter(action_log_max_lines=bad)
    _filter(action_log_max_lines=MIN_ACTION_LOG_MAX_LINES)


# --------------------------------------------------------------------------- execute wiring

def _verifier(**kw) -> RubricsVerifierTaskStep:
    base: dict = dict(id="verify", version=1, criteria=CRITERIA, prompt_id="p1", use_agent_judge=False,
                use_trajectory=True, trajectory_filter=_filter(), verifier_id="vid",
                output_format=JudgeOutputFormat.RUBRIC_EVIDENCE,
                grading_policy_prompt="Type every value in full; never accept an autocomplete suggestion.")
    base.update(kw)
    return RubricsVerifierTaskStep(**base)


def _ctx(**pr) -> TaskStepContext:
    base = dict(prompt_id="p1", response="done", prompt_text="reserve", agent_trajectory_s3_uri="s3://b/raw.json")
    base.update(pr)
    return TaskStepContext(prompt_responses=[PromptResponse(**base)])


def _passing_list(criteria: list[dict]) -> list[dict]:
    return [{**c, "score": 1.0, "result": True} for c in criteria]


def _passing_rows(criteria: list[dict]) -> ResultRows:
    return ResultRows(_passing_list(criteria))


def _capture_judge(monkeypatch, v: RubricsVerifierTaskStep, rows_for) -> dict:
    """Route the verifier at a canned judge; returns the dict the judge call is captured into."""
    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: _raw_run())
    monkeypatch.setattr(v, "_filter_trajectory", lambda *a, **k: pytest.fail("no default compaction expected"))
    captured: dict = {}

    async def fake_run(*, eval_prompt, context, model, criteria, image_blocks=None, **kw):
        captured.update(eval_prompt=eval_prompt, image_blocks=image_blocks, criteria=criteria)
        return (rows_for(criteria), 0, [], None)
    monkeypatch.setattr(v, "_run_judge_with_output_retries", fake_run)
    return captured


@pytest.mark.asyncio
async def test_execute_attaches_labelled_per_criterion_frames_with_an_action_log(monkeypatch):
    v = _verifier()

    def rows_for(criteria):
        rows = [{"id": c["id"], "frame": "[c1] action 14", "evidence_status": "visible", "justification": "seen",
                 "score": 1.0, "result": True} for c in criteria]
        return ResultRows(rows, reasoning="the table shows as reserved on frame 14",
                          checks=[{"requirement": "typed in full", "required_value": "party of four",
                                   "observed_value": "party of four", "frame": "[c2] action 6",
                                   "evidence_status": "visible", "note": "ok"}])
    captured = _capture_judge(monkeypatch, v, rows_for)
    ctx = _ctx()
    await v.execute(ctx)

    blocks = captured["image_blocks"]
    images = [b for b in blocks if b["type"] == "image_url"]
    labels = [b["text"] for b in blocks if b["type"] == "text"]
    assert len(images) == _BUDGET and len(labels) == len(images)     # 30 frames trimmed to the budget, a label each
    assert all(label.startswith("FRAME: ") for label in labels)
    assert "FRAME: [c2] typed action 5" in labels                               # short keys, as the judge sees them
    assert "FRAME: [c1] action 12" in labels and "FRAME: [c1] action 13" in labels
    assert "FRAME: final action 30" in labels
    for i in range(0, len(blocks), 2):                                          # label THEN image
        assert blocks[i]["type"] == "text" and blocks[i + 1]["type"] == "image_url"

    text = captured["eval_prompt"]
    assert "/9j/" not in text and "let me think" not in text                      # no frames or thinking in the text
    assert 'agent said: "I will now book the table"' in text                     # the agent's words, labelled as such
    assert text.index("agent said:") < text.index("1. ios_tap")                  # ...in run order
    assert "12. ios_tap_element" in text and '"label": "Reserve table"' in text and ' -> {"result": "ok"}' in text
    assert "`agent said:`" in text                                              # the grounding explains the agent's lines
    assert "harness" not in per_criterion_grounding(1, cites_evidence=True)     # no one bridge's conventions in the prompt
    assert "## Grading policy" in text and "never accept an autocomplete suggestion" in text
    assert evidence_checks_instruction(has_policy=True) in text
    assert per_criterion_grounding(len(images), cites_evidence=True) in text
    assert "Only the FINAL frames are attached" not in text

    entry = ctx.metadata["verifications"]["vid"]
    assert entry["format"] == "rubric_binary" and entry["judge_output_format"] == "rubric_evidence"
    assert entry["score"] == 1.0 and entry["reasoning"].startswith("the table shows")
    assert [r["id"] for r in entry["results"]] == ["table-reserved", "party-size"]         # real ids restored
    assert entry["results"][0]["frame"] == "[table-reserved] action 14"                      # ...inside labels too
    assert entry["checks"][0]["frame"] == "[party-size] action 6"
    assert type(entry["results"]) is list


async def _frame_labels(monkeypatch, v: RubricsVerifierTaskStep) -> list[str]:
    captured = _capture_judge(monkeypatch, v, _passing_rows)
    await v.execute(_ctx())
    return [b["text"] for b in captured["image_blocks"] if b["type"] == "text"]


@pytest.mark.asyncio
async def test_exclude_pattern_drops_actions_as_evidence_case_insensitively(monkeypatch):
    # action 12 is `ios_tap_element {"label": "Reserve table"}` — the only evidence for c1 (table-reserved)
    assert "FRAME: [c1] action 12" in await _frame_labels(monkeypatch, _verifier())
    labels = await _frame_labels(monkeypatch, _verifier(trajectory_filter=_filter(evidence_exclude_pattern=r"reserve TABLE")))
    assert not any(label.startswith("FRAME: [c1]") for label in labels)        # mixed-case action, mixed-case pattern
    assert "FRAME: final action 30" in labels


@pytest.mark.asyncio
async def test_always_show_actions_pins_frames_case_insensitively_and_none_pins_nothing(monkeypatch):
    # action 5 is `ios_type {"text": "party of four"}`; the pin is spelled in upper case and over the arguments
    pins = {"TYPED": r"^IOS_TYPE\b", "party": r'"text": "PARTY'}
    labels = await _frame_labels(monkeypatch, _verifier(trajectory_filter=_filter(always_show_actions=pins)))
    assert "FRAME: [c2] TYPED party action 5" in labels
    labels = await _frame_labels(monkeypatch, _verifier(trajectory_filter=_filter(always_show_actions=None)))
    assert not any("typed" in label.lower() for label in labels)               # no pins: no reservation, no label
    assert len(labels) == _BUDGET


@pytest.mark.asyncio
async def test_action_log_line_cap_keeps_the_final_actions_in_the_prompt(monkeypatch):
    v = _verifier(trajectory_filter=_filter(action_log_max_lines=MIN_ACTION_LOG_MAX_LINES))
    captured = _capture_judge(monkeypatch, v, _passing_rows)
    await v.execute(_ctx())
    text = captured["eval_prompt"]
    # 1 agent message + 30 tool calls = 31 lines; 10 kept — the message and calls 1-4, then calls 26-30
    assert 'agent said: "I will now book the table"\n1. ios_tap' in text and "\n4. ios_tap" in text
    assert "\n... (21 lines not shown) ...\n26. ios_tap" in text and "\n30. ios_tap" in text
    assert "\n5. ios_type" not in text and "\n25. ios_tap" not in text
    v = _verifier()                                                             # the default cap elides nothing here
    default = _capture_judge(monkeypatch, v, _passing_rows)
    await v.execute(_ctx())
    assert "not shown" not in default["eval_prompt"] and "\n5. ios_type" in default["eval_prompt"]


@pytest.mark.asyncio
async def test_rubric_evidence_refuses_to_run_when_an_override_drops_the_filter(monkeypatch):
    v = _verifier()
    captured = _capture_judge(monkeypatch, v, lambda criteria: pytest.fail("the judge must not be called"))
    ctx = _ctx()
    ctx.metadata["user_overrides"] = {"apply_trajectory_filter": False}
    with pytest.raises(RuntimeError, match="rubric_evidence needs per-criterion frames"):
        await v.execute(ctx)
    assert "eval_prompt" not in captured and "verifications" not in ctx.metadata
    # the same override on rubric_binary is fine: no frames is a valid view for a format that cites none
    v = _verifier(output_format=JudgeOutputFormat.RUBRIC_BINARY, grading_policy_prompt=None)
    captured = _capture_judge(monkeypatch, v, _passing_list)
    ctx = _ctx()
    ctx.metadata["user_overrides"] = {"apply_trajectory_filter": False}
    await v.execute(ctx)
    assert captured["image_blocks"] == [] and ctx.metadata["verifications"]["vid"]["score"] == 1.0


_NO_SCREENSHOTS = json.dumps([_span("ios_tap", {"x": i}, shot=None) for i in range(1, 6)])


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [_NO_SCREENSHOTS, "not json"], ids=["no screenshots", "malformed"])
async def test_rubric_evidence_writes_a_skipped_entry_when_the_trajectory_has_no_frames(monkeypatch, raw):
    v = _verifier(output_format=JudgeOutputFormat.RUBRIC_EVIDENCE_WITH_MISTAKES)
    captured = _capture_judge(monkeypatch, v, lambda criteria: pytest.fail("the judge must not be called"))
    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: raw)
    ctx = _ctx()
    await v.execute(ctx)                                                        # a failed run, not a broken task
    assert "eval_prompt" not in captured
    main = ctx.metadata["verifications"]["vid"]
    second = ctx.metadata["verifications"][trajectory_mistakes_key("vid")]
    assert main["score"] == 0 and main["results"][0]["id"] == "no_screenshots" and main["results"][0]["result"] is False
    assert "no screenshots" in main["results"][0]["message"]
    assert main["format"] == "rubric_binary" and main["judge_output_format"] == "rubric_evidence_with_mistakes"
    assert second["results"] == main["results"] and second["format"] == "trajectory_mistakes"


@pytest.mark.asyncio
async def test_a_format_without_evidence_fields_degrades_to_the_text_only_prompt_when_no_frames(monkeypatch):
    v = _verifier(output_format=JudgeOutputFormat.RUBRIC_BINARY, grading_policy_prompt=None)
    captured = _capture_judge(monkeypatch, v, _passing_list)
    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: _NO_SCREENSHOTS)
    ctx = _ctx()
    await v.execute(ctx)
    text = captured["eval_prompt"]
    assert captured["image_blocks"] == [] and "1. ios_tap" in text             # the action log still goes...
    assert "## Screenshots" not in text and "FRAME:" not in text                # ...with no frame rules to follow
    assert ctx.metadata["verifications"]["vid"]["score"] == 1.0


@pytest.mark.asyncio
async def test_evidence_prompt_without_a_policy_does_not_mention_one(monkeypatch):
    v = _verifier(grading_policy_prompt=None)
    captured = _capture_judge(monkeypatch, v, _passing_rows)
    await v.execute(_ctx())
    text = captured["eval_prompt"]
    assert "## Grading policy" not in text and "grading policy below" not in text
    assert evidence_checks_instruction(has_policy=False) in text


@pytest.mark.parametrize("fmt", [JudgeOutputFormat.RUBRIC_BINARY, JudgeOutputFormat.RUBRIC_PARTIAL,
                                 JudgeOutputFormat.TRAJECTORY_MISTAKES])
@pytest.mark.asyncio
async def test_per_criterion_frames_with_a_format_without_evidence_fields(monkeypatch, fmt):
    """Another format on a per_criterion filter: the labels are orientation only, never fields its schema forbids."""
    v = _verifier(output_format=fmt, grading_policy_prompt=None)
    captured = _capture_judge(monkeypatch, v, _passing_list)
    ctx = _ctx()
    await v.execute(ctx)
    text = captured["eval_prompt"]
    images = [b for b in captured["image_blocks"] if b["type"] == "image_url"]
    labels = [b["text"] for b in captured["image_blocks"] if b["type"] == "text"]
    assert len(images) == _BUDGET and "FRAME: final action 30" in labels
    assert per_criterion_grounding(len(images), cites_evidence=False) in text
    assert "`evidence_status`" not in text and "NOT SHOWN" not in text and "$checks_instruction" not in text
    entry = ctx.metadata["verifications"]["vid"]
    assert entry["format"] == fmt.value and "judge_output_format" not in entry
    assert "reasoning" not in entry and "checks" not in entry


@pytest.mark.asyncio
async def test_final_frames_strategy_with_the_action_log_text_view(monkeypatch):
    v = _verifier(trajectory_filter=_filter(frame_strategy=FrameStrategy.FINAL_FRAMES, screenshot_last_n=2),
                  output_format=JudgeOutputFormat.RUBRIC_BINARY, grading_policy_prompt=None)
    captured = _capture_judge(monkeypatch, v, _passing_list)
    ctx = _ctx()
    await v.execute(ctx)
    assert [b["type"] for b in captured["image_blocks"]] == ["image_url", "image_url"]   # plain last-2, no labels
    assert "1. ios_tap" in captured["eval_prompt"] and "Final-state screenshots" in captured["eval_prompt"]
    assert "<image omitted" not in captured["eval_prompt"]                     # the action log, not stripped text
    assert "## Grading policy" not in captured["eval_prompt"]
    assert ctx.metadata["verifications"]["vid"]["format"] == "rubric_binary"
    assert "judge_output_format" not in ctx.metadata["verifications"]["vid"]


@pytest.mark.asyncio
async def test_per_criterion_frames_refuse_the_agent_judge():
    v = _verifier(use_agent_judge=True, default_model_api_base=None, output_format=JudgeOutputFormat.RUBRIC_BINARY)
    with pytest.raises(RuntimeError, match="per_criterion only applies to the direct LLM judge"):
        await v.execute(_ctx())


@pytest.mark.parametrize("bad", [
    dict(trajectory_filter=None),
    dict(trajectory_filter=TrajectoryFilter()),
    dict(trajectory_filter=_filter(frame_strategy=FrameStrategy.FINAL_FRAMES)),
    dict(use_agent_judge=True),
    dict(use_trajectory=False),
])
def test_rubric_evidence_requires_per_criterion_frames_on_the_direct_judge(bad):
    with pytest.raises(ValueError, match="rubric_evidence requires"):
        _verifier(**bad)
    _verifier()                                                                  # the valid combination constructs


@pytest.mark.asyncio
async def test_prompt_error_entry_carries_the_same_format_fields_as_a_graded_one():
    ctx = _ctx(response="", error_type="timeout")
    await _verifier().execute(ctx)
    entry = ctx.metadata["verifications"]["vid"]
    assert entry["format"] == "rubric_binary" and entry["judge_output_format"] == "rubric_evidence"
    assert entry["results"][0]["id"] == "prompt_error" and entry["score"] == 0
    ctx = _ctx(response="", error_type="timeout")
    await _verifier(output_format=JudgeOutputFormat.RUBRIC_BINARY).execute(ctx)
    entry = ctx.metadata["verifications"]["vid"]
    assert entry["format"] == "rubric_binary" and "judge_output_format" not in entry


def test_policy_prompt_round_trips_and_defaults_to_none():
    v = _verifier()
    d = v.to_dict()
    assert d["grading_policy_prompt"].startswith("Type every value") and d["output_format"] == "rubric_evidence"
    back = RubricsVerifierTaskStep.from_dict({**d, "default_model": "m"})
    assert back.grading_policy_prompt == v.grading_policy_prompt and back.output_format == JudgeOutputFormat.RUBRIC_EVIDENCE
    assert RubricsVerifierTaskStep.from_dict({"id": "x", "version": 1, "criteria": CRITERIA, "prompt_id": "p",
                                              "default_model": "m"}).grading_policy_prompt is None


def test_restore_frame_labels_rewrites_every_tag_without_prefix_collisions():
    real_by_short = {f"c{i}": f"real-{i}" for i in range(1, 11)}
    rows = [{"frame": "[c1][c10] action 3"}, {"frame": "NOT SHOWN"}, {"frame": ""}, {"id": "x"}]
    _restore_frame_labels(rows, real_by_short)
    assert [r.get("frame") for r in rows] == ["[real-1][real-10] action 3", "NOT SHOWN", "", None]


# --------------------------------------------------------------------------- context_facts_for_judge

_FACTS = {"autocomplete fragments": "verifications.ios-mechanical-flail.typing_fragments",
          "first tap": "verifications.ios-mechanical-flail.taps.0.target",
          "missing": "verifications.nope.count"}
_FLAIL = {"typing_fragments": {"zeta": 1, "alpha": ["ca", "caf"]}, "taps": [{"target": "Café"}, {"target": "b"}]}


def _facts_ctx() -> TaskStepContext:
    ctx = _ctx()
    ctx.metadata["verifications"] = {"ios-mechanical-flail": _FLAIL}
    return ctx


def test_resolve_context_facts_walks_dicts_and_lists_and_marks_what_does_not_resolve():
    meta = {"verifications": {"ios-mechanical-flail": _FLAIL}, "n": 3, "long": "x" * 1000}
    facts = {**_FACTS, "count": "n", "long value": "long",
             "index past the end": "verifications.ios-mechanical-flail.taps.7.target",
             "key under a scalar": "n.child", "key under a list": "verifications.ios-mechanical-flail.taps.target"}
    resolved = dict(resolve_context_facts(facts, meta))
    assert list(resolved) == list(facts)                                             # requested order kept
    assert resolved["autocomplete fragments"] == '{"alpha": ["ca", "caf"], "zeta": 1}'   # compact JSON, sorted keys
    assert resolved["first tap"] == '"Café"'                                          # list index; not ascii-escaped
    assert resolved["count"] == "3"
    for label in ("missing", "index past the end", "key under a scalar", "key under a list"):
        assert resolved[label] == CONTEXT_FACT_NOT_AVAILABLE
    assert len(resolved["long value"]) == CONTEXT_FACT_MAX_CHARS + 1 and resolved["long value"].endswith("…")
    assert resolved["long value"].startswith('"xxx')
    assert resolve_context_facts({}, meta) == []


def test_resolve_context_facts_renders_values_json_cannot_sort_or_serialise():
    meta = {"mixed": {"a": 1, 1: 2}, "odd": {(1, 2): 3}}
    resolved = dict(resolve_context_facts({"m": "mixed", "o": "odd"}, meta))
    assert resolved["m"] == '{"a": 1, "1": 2}'                                    # keys that cannot be sorted together: stored order
    assert resolved["o"] == "{(1, 2): 3}"                                         # not JSON at all: str()


def _section_positions(text: str) -> tuple[int, int]:
    assert text.count("## Context facts") == 1
    return text.index("## Context facts"), text.index("## Grading policy")


@pytest.mark.asyncio
async def test_execute_shows_context_facts_to_the_direct_judge_before_the_grading_policy(monkeypatch):
    v = _verifier(context_facts_for_judge=_FACTS)
    captured = _capture_judge(monkeypatch, v, _passing_rows)
    ctx = _facts_ctx()
    await v.execute(ctx)
    text = captured["eval_prompt"]
    facts_at, policy_at = _section_positions(text)
    assert text.index("## Criteria") < facts_at < policy_at < text.index("never accept an autocomplete suggestion")
    section = text[facts_at:policy_at]
    assert '- autocomplete fragments: {"alpha": ["ca", "caf"], "zeta": 1}' in section
    assert '- first tap: "Café"' in section
    assert f"- missing: {CONTEXT_FACT_NOT_AVAILABLE}" in section
    assert "measured from the run's record" in section
    assert "context_facts_for_judge" not in ctx.metadata["verifications"]["vid"]            # facts are prompt-only


@pytest.mark.asyncio
async def test_context_facts_section_only_appears_when_facts_are_configured(monkeypatch):
    v = _verifier(context_facts_for_judge=None)
    captured = _capture_judge(monkeypatch, v, _passing_rows)
    await v.execute(_facts_ctx())
    assert "## Context facts" not in captured["eval_prompt"] and "Café" not in captured["eval_prompt"]
    # ...and needs no policy: facts without a policy still render, all-missing metadata still renders
    v = _verifier(context_facts_for_judge={"first tap": _FACTS["first tap"]}, grading_policy_prompt=None)
    captured = _capture_judge(monkeypatch, v, _passing_rows)
    await v.execute(_ctx())
    text = captured["eval_prompt"]
    assert text.count("## Context facts") == 1 and "## Grading policy" not in text
    assert f"- first tap: {CONTEXT_FACT_NOT_AVAILABLE}" in text


def test_agent_judge_prompt_places_context_facts_after_artifacts_and_before_the_policy():
    v = RubricsVerifierTaskStep(id="v", version=1, criteria=CRITERIA, prompt_id="p1", verifier_id="vid",
                                grading_policy_prompt="Type in full.", context_facts_for_judge=_FACTS)
    kw = dict(agent_prompt="a", agent_response="r", criteria_json="[]", trajectory_path="/tmp/x",
              loaded_artifacts=[{"id": "art", "files": ["f"]}])
    text = v._build_eval_prompt(**kw, context_facts=[("first tap", '"Café"')])
    facts_at, policy_at = _section_positions(text)
    assert text.index("## Files Available For Inspection") < facts_at < policy_at
    assert '- first tap: "Café"' in text[facts_at:policy_at]
    assert "## Context facts" not in v._build_eval_prompt(**kw)                      # default: unchanged prompt
    assert "## Context facts" not in v._build_eval_prompt(**kw, context_facts=[])


def test_context_facts_for_judge_round_trip_and_default_to_none():
    v = _verifier(context_facts_for_judge=_FACTS)
    d = v.to_dict()
    assert d["context_facts_for_judge"] == _FACTS
    back = RubricsVerifierTaskStep.from_dict({**d, "default_model": "m"})
    assert back.context_facts_for_judge == _FACTS
    assert _verifier().to_dict()["context_facts_for_judge"] is None
    assert RubricsVerifierTaskStep.from_dict({"id": "x", "version": 1, "criteria": CRITERIA, "prompt_id": "p",
                                              "default_model": "m"}).context_facts_for_judge is None


@pytest.mark.parametrize("bad", [
    {"": "a.b"},                      # empty label
    {"label": ""},                    # empty path
    {"label": "a..b"},                # empty segment
    {"label": ".a"},
    {"label": "a b"},                 # whitespace
    {"label": "a.b["},                # not an identifier
    {"label": 3},                     # non-string path
    {3: "a.b"},                       # non-string label
    ["verifications.x"],              # not a mapping
])
def test_context_facts_for_judge_ctor_validation(bad):
    with pytest.raises(ValueError, match="context_facts_for_judge"):
        _verifier(context_facts_for_judge=bad)
    _verifier(context_facts_for_judge={"ok": "a-b.c_d.0.Z9"})


# --------------------------------------------------------------------------- rubric_evidence format

_CRIT = [{"id": "c1", "description": "a"}, {"id": "c2", "description": "b"}]


def _evidence_response(*rows: dict, reasoning: str = "r", checks: list | None = None) -> str:
    return json.dumps({"reasoning": reasoning, "checks": checks or [], "results": list(rows)})


def _row(cid: str, score, status: str = "visible", frame: str = "[c1] action 3") -> dict:
    return {"id": cid, "frame": frame, "evidence_status": status, "justification": "j", "score": score}


def test_rubric_evidence_rows_carry_evidence_and_bad_evidence_fails_the_row():
    spec = get_judge_output_format_spec("rubric_evidence")
    assert spec.stored_format == "rubric_binary" and spec.cites_evidence
    rows, disc = diagnose_rubric_evidence_response(
        _evidence_response(_row("c1", 1.0), _row("c2", 1.0, status="missing", frame="NOT SHOWN")), _CRIT)
    assert disc is None and isinstance(rows, ResultRows) and rows.reasoning == "r" and rows.checks == []
    assert rows[0]["result"] is True and rows[0]["frame"] == "[c1] action 3" and rows[0]["description"] == "a"
    assert rows[1]["result"] is False and rows[1]["score"] == 0.0        # missing evidence beats the judge's 1.0
    # id integrity is the shared machinery: a dropped criterion is a discrepancy, not a pass
    rows, disc = diagnose_rubric_evidence_response(_evidence_response(_row("c1", 1.0)), _CRIT)
    assert rows is None and disc.missing_ids == ["c2"]
    schema = spec.output_format["schema"]
    assert list(schema["properties"]) == ["reasoning", "checks", "results"]            # evidence first, verdict last
    assert list(schema["properties"]["results"]["items"]["properties"]) == [
        "id", "frame", "evidence_status", "justification", "score"]


def test_rubric_evidence_scores_are_snapped_to_binary_and_unknown_statuses_are_ambiguous():
    rows, disc = diagnose_rubric_evidence_response(
        _evidence_response(_row("c1", 0.5), _row("c2", 1, status="Definitely Seen")), _CRIT)
    assert disc is None
    assert rows[0]["score"] == 0.0 and rows[0]["result"] is False and rows[0]["raw_score"] == 0.5
    assert aggregate_score(rows, ScoreAggregator.WEIGHTED_AVERAGE) == 0.5   # the 0.5 earns nothing: (0 + 1) / 2
    assert rows[1]["score"] == 1.0 and rows[1]["result"] is True and rows[1]["evidence_status"] == "ambiguous"
    assert "raw_score" not in rows[1]                                          # 1 is binary: nothing was snapped
    rows, _ = diagnose_rubric_evidence_response(_evidence_response(_row("c1", 2), _row("c2", -1)), _CRIT)
    assert [(r["score"], r["result"], r["raw_score"]) for r in rows] == [(1.0, True, 2.0), (0.0, False, -1.0)]
    rows, _ = diagnose_rubric_evidence_response(_evidence_response(_row("c1", "one"), _row("c2", None)), _CRIT)
    assert [(r["score"], r["result"]) for r in rows] == [(0.0, False), (0.0, False)]   # non-numeric -> fail
    for status in EVIDENCE_STATUSES:
        rows, _ = diagnose_rubric_evidence_response(_evidence_response(_row("c1", 1.0, status), _row("c2", 1.0)), _CRIT)
        assert rows[0]["evidence_status"] == status


def test_rubric_evidence_reasoning_and_checks_are_capped_and_typed():
    checks = [{"requirement": str(i)} for i in range(20)] + ["not a dict"]
    rows, _ = diagnose_rubric_evidence_response(
        _evidence_response(_row("c1", 1.0), _row("c2", 1.0), reasoning="x" * 5000, checks=checks), _CRIT)
    assert len(rows.reasoning) == 2000 and len(rows.checks) == 12 and all(isinstance(c, dict) for c in rows.checks)


def test_evidence_correction_prompt_asks_for_the_evidence_fields_and_the_binary_one_does_not():
    disc = JudgeResponseDiscrepancy(expected_count=2, returned_count=1, expected_ids=["c1", "c2"],
                                    missing_ids=["c2"], duplicate_ids=[], unknown_ids=[])
    kw = dict(eval_prompt="EVAL", previous_response="{}", discrepancy=disc)
    evidence = format_evidence_judge_correction_prompt(**kw)
    binary = format_judge_correction_prompt(**kw)
    for name in ("frame", "evidence_status", "reasoning", "checks"):
        assert f'"{name}"' in evidence and f'"{name}"' not in binary
    assert "Return JSON with `reasoning`, `checks` and a `results` array containing exactly 2 objects" in evidence
    assert "Return JSON with a `results` array containing exactly 2 objects" in binary
    parse = JudgeResponseDiscrepancy(expected_count=2, returned_count=0, expected_ids=["c1", "c2"], missing_ids=[],
                                     duplicate_ids=[], unknown_ids=[], parse_error="boom")
    assert "Return only a JSON object with `reasoning`, `checks` and a `results` array" in \
        format_evidence_judge_correction_prompt(**{**kw, "discrepancy": parse})
    spec = get_judge_output_format_spec("rubric_evidence")
    assert spec.format_correction_prompt is format_evidence_judge_correction_prompt
