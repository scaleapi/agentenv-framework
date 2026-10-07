"""Unit tests for the trajectory-mistakes "golden judge" output format (a
RubricsVerifierTaskStep with ``output_format="trajectory_mistakes"``, not a bespoke step
type): the output format itself and the deploy-safety property (a task carrying it still
loads on an agent-env that doesn't know the format)."""
from __future__ import annotations

import json

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    _TRAJECTORY_MISTAKES_INSTRUCTIONS,
    SCREENSHOT_GROUNDING,
    MISTAKE_SEVERITY_PENALTY,
    JudgeOutputFormat,
    diagnose_trajectory_mistakes_response,
    get_judge_output_format_spec,
)
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep


def _score(n_coverage: int, severities: list[str], *, botched: bool = False,
           coverage_shown: bool = True, weights: dict | None = None) -> float:
    """Score a synthetic judge response the way the runtime does."""
    resp = json.dumps({
        "findings": [{"severity": s, "step": i, "should": "x", "description": "y"}
                     for i, s in enumerate(severities)],
        "coverage": [{"id": f"c{i}", "shown": coverage_shown, "note": ""}
                     for i in range(n_coverage)],
        "botched": botched, "botched_reason": "", "summary": "s",
    })
    rows, disc = diagnose_trajectory_mistakes_response(resp, [], weights)
    assert disc is None
    return aggregate_score(rows, ScoreAggregator.WEIGHTED_AVERAGE)


# ── output format ─────────────────────────────────────────────────────────────

def test_format_registered_and_step_roundtrips():
    assert "trajectory_mistakes" in [f.value for f in JudgeOutputFormat]
    step = RubricsVerifierTaskStep(
        id="s1", version=1, prompt_id="p.prompt", criteria=[{"id": "c1", "description": "x"}],
        output_format="trajectory_mistakes", use_agent_judge=False, fail_task_on_error=False,
    )
    d = step.to_dict()
    assert d["type"] == "rubrics_verifier" and d["output_format"] == "trajectory_mistakes"
    back = RubricsVerifierTaskStep.from_dict(d)
    assert back.output_format is JudgeOutputFormat.TRAJECTORY_MISTAKES
    assert back.fail_task_on_error is False


def test_unknown_format_raises():
    # coerce_output_format was removed in review (FIX #4): an unknown format now RAISES
    # (main's behavior) instead of degrading to rubric_binary.
    d = RubricsVerifierTaskStep(
        id="s", version=1, prompt_id="p", criteria=[], output_format="rubric_binary",
    ).to_dict()
    d["output_format"] = "some_future_format"
    with pytest.raises(ValueError):
        RubricsVerifierTaskStep.from_dict(d)


def test_findings_map_to_rubric_shaped_rows():
    resp = json.dumps({
        "findings": [{"severity": "high", "step": 3, "should": "open Notes",
                      "description": "tapped wrong app"}],
        "coverage": [{"id": "c1", "shown": False, "note": "never shown"}],
        "botched": True, "botched_reason": "flailed", "summary": "bad run",
    })
    rows, disc = diagnose_trajectory_mistakes_response(resp, [{"id": "c1"}])
    assert disc is None
    ids = {r["id"] for r in rows}
    assert {"summary", "botched", "mistake_0", "coverage_c1"} <= ids
    # Rubric-shaped rows -> the existing hub rubric renderer displays them unchanged.
    assert all(all(k in r for k in ("id", "score", "result", "justification")) for r in rows)
    assert next(r for r in rows if r["id"] == "botched")["result"] is False
    assert next(r for r in rows if r["id"] == "coverage_c1")["result"] is False


def test_bad_json_flags_parse_error_for_retry():
    rows, disc = diagnose_trajectory_mistakes_response("not json at all", [])
    assert rows is None and disc is not None and disc.parse_error


# ── scoring: severity is priced, and the ceiling is rubric-size independent ────

def test_clean_run_scores_one():
    assert _score(4, []) == pytest.approx(1.0)


def test_severity_changes_the_score():
    """The old shape scored every mistake a flat 0.0, so these were indistinguishable."""
    low, medium, high = _score(6, ["low"]), _score(6, ["medium"]), _score(6, ["high"])
    assert low > medium > high
    assert low == pytest.approx(1.0 - MISTAKE_SEVERITY_PENALTY["low"])
    assert medium == pytest.approx(1.0 - MISTAKE_SEVERITY_PENALTY["medium"])
    assert high == pytest.approx(1.0 - MISTAKE_SEVERITY_PENALTY["high"])


def test_penalty_is_independent_of_coverage_count():
    """Regression: the old ceiling was (C+1)/(C+2+M), so a C=2 task scored 0.60 for a single
    `low` while a C=11 task scored 0.86 for the same trajectory quality."""
    scores = {c: _score(c, ["low"]) for c in (2, 4, 6, 11, 20)}
    assert len(set(round(s, 9) for s in scores.values())) == 1, scores


def test_small_rubric_can_reach_the_golden_bar_with_one_minor_mistake():
    """Regression: at C<=4 the old maximum with any mistake was 0.714 — unreachable."""
    for c in (1, 2, 3, 4):
        assert _score(c, ["low"]) >= 0.75
        assert _score(c, ["medium"]) >= 0.75


def test_critical_floors_the_score():
    assert _score(8, ["critical"]) == pytest.approx(0.0)


def test_penalties_accumulate_and_clamp_at_zero():
    assert _score(8, ["medium", "medium"]) == pytest.approx(1.0 - 2 * MISTAKE_SEVERITY_PENALTY["medium"])
    assert _score(8, ["high"] * 9) == pytest.approx(0.0)


def test_summary_row_does_not_double_charge():
    """`summary` flips to 0.0 on any finding; scoring it charged the same mistake twice."""
    rows, _ = diagnose_trajectory_mistakes_response(json.dumps({
        "findings": [{"severity": "low", "step": 1, "should": "x", "description": "y"}],
        "coverage": [{"id": "c0", "shown": True, "note": ""}],
        "botched": False, "botched_reason": "", "summary": "s",
    }), [])
    summary = next(r for r in rows if r["id"] == "summary")
    assert summary["weight"] == 0.0
    assert summary["result"] is False  # still rendered as a failing row in the hub


def test_missing_coverage_still_lowers_the_score():
    """Coverage remains the positive signal — penalties are additive on top of it."""
    assert _score(4, [], coverage_shown=False) < _score(4, [])


def test_botched_still_penalised():
    assert _score(4, [], botched=True) < _score(4, [])


def test_unknown_severity_scores_as_medium():
    assert _score(6, ["bogus"]) == pytest.approx(_score(6, ["medium"]))


# ── the weights are caller-owned, so they can be tuned without a release ──────

def test_caller_supplied_weights_override_the_default():
    caller = {"low": 0.5, "medium": 0.5, "high": 0.5, "critical": 0.5}
    assert _score(6, ["low"], weights=caller) == pytest.approx(0.5)
    assert _score(6, ["high"], weights=caller) == pytest.approx(0.5)
    # ...and differ from the built-in fallback.
    assert _score(6, ["low"]) != pytest.approx(_score(6, ["low"], weights=caller))


def test_unknown_severity_falls_back_within_caller_table():
    caller = {"low": 0.01, "medium": 0.3, "high": 0.4, "critical": 1.0}
    assert _score(6, ["bogus"], weights=caller) == pytest.approx(_score(6, ["medium"], weights=caller))


def test_override_is_a_pure_parameter_not_step_config():
    """The override is a parameter on the parse helper only. Carrying it on the step would
    mean widening the shared RubricsVerifierTaskStep, which every team's rubric verifier
    goes through — deliberately not done."""
    import inspect
    assert "severity_weights" in inspect.signature(
        diagnose_trajectory_mistakes_response).parameters
    assert "mistake_severity_weights" not in inspect.signature(
        RubricsVerifierTaskStep.__init__).parameters


# ── the shared rules are platform-neutral; product policy arrives via grading_policy_prompt ──

# What a task now puts in its `grading_policy_prompt` to get the rule the shared text used to carry.
_AUTOCOMPLETE_POLICY = """ALWAYS A MISTAKE — autocomplete / suggestion shortcuts. The task prompt is the reference: the
agent must type the prompt's target out IN FULL and end up on it. Severity "high", never lower.
A recent search, suggestion chip, Siri Suggestion, Spotlight top hit, AutoFill prompt, or QuickType row
filling or submitting the field is a violation."""
_PLATFORM_WORDS = ("autocomplete", "siri", "spotlight", "autofill", "quicktype", "pizza hut", "see above")


def test_shared_mistake_rules_carry_no_platform_policy():
    """The autocomplete / suggestion-shortcut rule (Siri Suggestion, Spotlight top hit, QuickType...) was
    one platform's product policy living in the shared instructions, so every trajectory_mistakes and
    rubric_evidence_with_mistakes user got it whether or not it applied. It now belongs in the task's
    grading_policy_prompt; the shared text keeps only what is true of any trajectory."""
    shared = [_TRAJECTORY_MISTAKES_INSTRUCTIONS]
    spec = get_judge_output_format_spec("rubric_evidence_with_mistakes")
    shared += [t.template for t in (spec.prompt_template_path, spec.prompt_template_inline,
                                    spec.prompt_template_no_trajectory)]
    for text in shared:
        low = text.lower()
        for word in _PLATFORM_WORDS:
            assert word not in low, word
    # ...and the generic parts stayed
    low = _TRAJECTORY_MISTAKES_INSTRUCTIONS.lower()
    for kept in ("what you cannot see is not a mistake", "harness fault is not the agent's mistake",
                 "severity — choose the rung", '"findings"', '"coverage"', '"botched"'):
        assert kept in low, kept


def _mistakes_verifier(**kw) -> RubricsVerifierTaskStep:
    base: dict = dict(id="verify", version=1, criteria=[{"id": "c1", "description": "x"}], prompt_id="p1",
                      use_agent_judge=False, use_trajectory=True, verifier_id="vid",
                      output_format="trajectory_mistakes", grading_policy_prompt=_AUTOCOMPLETE_POLICY)
    base.update(kw)
    return RubricsVerifierTaskStep(**base)


@pytest.mark.asyncio
async def test_a_grading_policy_restores_the_autocomplete_rule_on_the_trajectory_mistakes_path(monkeypatch):
    """A task that wants the rule sets `grading_policy_prompt`; on the trajectory_mistakes format it reaches
    the judge as a "## Grading policy" section on both judge paths (the direct judge via execute, the agent
    judge via the path template)."""
    v = _mistakes_verifier()
    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: json.dumps([]))
    monkeypatch.setattr(v, "_filter_trajectory", lambda uri, tf: ("s3://b/compact.json", []))
    captured: dict = {}

    async def fake_run(*, eval_prompt, criteria, **kw):
        captured["eval_prompt"] = eval_prompt
        return ([{**c, "score": 1.0, "result": True, "justification": "j"} for c in criteria], 0, [], None)
    monkeypatch.setattr(v, "_run_judge_with_output_retries", fake_run)
    ctx = TaskStepContext(prompt_responses=[PromptResponse(prompt_id="p1", response="done", prompt_text="reserve",
                                                           agent_trajectory_object_url="s3://b/raw.json")])
    await v.execute(ctx)
    text = captured["eval_prompt"]
    assert text.count("## Grading policy") == 1 and text.count(_AUTOCOMPLETE_POLICY) == 1
    assert text.index("## Instructions") < text.index("## Grading policy")   # the policy follows the format's rules
    assert text.count("Siri Suggestion") == 1                                 # from the policy only, not the shared text

    agent_path = _mistakes_verifier(use_agent_judge=True)._build_eval_prompt(
        agent_prompt="a", agent_response="r", criteria_json="[]", trajectory_path="/tmp/x")
    assert "## Grading policy\n" + _AUTOCOMPLETE_POLICY in agent_path
    # ...and without a policy, nothing about autocomplete reaches the judge
    bare = _mistakes_verifier(grading_policy_prompt=None)._build_eval_prompt(
        agent_prompt="a", agent_response="r", criteria_json="[]", trajectory_path="/tmp/x")
    assert "## Grading policy" not in bare and "autocomplete" not in bare.lower()


def _instructions_flat():
    """Instructions with whitespace collapsed — the prose is hard-wrapped, so a phrase can
    straddle a newline + indent and a naive substring check misses it."""
    return " ".join(_TRAJECTORY_MISTAKES_INSTRUCTIONS.lower().split())


def test_mid_interaction_frames_are_not_the_outcome():
    """A booking run selected Nov 17 after a couple of tries; an attached frame caught the
    calendar still sitting on 'Tue, Dec 1'. BOTH judges read that transient frame as the chosen
    date — the rubric failed `departure-date` AND `departure-origin` on it, and the trajectory
    judge filed a [critical]. Frames of an open picker show where the control was, not what was
    picked. SCREENSHOT_GROUNDING is appended for every judge that gets images, so this rule
    has to live there rather than in one format's instructions."""
    flat = " ".join(SCREENSHOT_GROUNDING.lower().split())
    assert "mid-interaction" in flat
    assert "transient" in flat
    assert "never let one override the trajectory" in flat


def test_absence_of_evidence_is_not_a_mistake():
    """The judge sees a small sample of frames and picker values leave no text trace, so
    'the trajectory does not show X' says nothing about whether the agent did X. It was
    reporting those as mistakes."""
    flat = _instructions_flat()
    assert "what you cannot see is not a mistake" in flat
    assert "in the run's favour" in flat
    assert "not clearly demonstrated" in flat


def test_severity_ladder_is_defined_for_every_rung():
    """Severity is priced (see MISTAKE_SEVERITY_PENALTY) and gates promotion, so the judge has
    to be told what earns each rung — the enum previously offered four values and defined none,
    and the same behaviour came back as "high" on one run and "medium" on the next."""
    low = _TRAJECTORY_MISTAKES_INSTRUCTIONS.lower()
    assert "severity" in low
    for rung in MISTAKE_SEVERITY_PENALTY:
        assert f'"{rung}":' in low, f"severity ladder does not define {rung!r}"


def test_high_severity_finding_renders_the_marker_the_gate_reads():
    """A downstream gate greps "[high]" out of the justification; keep that contract pinned from
    this side."""
    resp = json.dumps({
        "findings": [{"severity": "high", "step": 5, "should": "type the full query",
                      "description": "typed 'piz' then tapped the suggestion"}],
        "coverage": [], "botched": False, "botched_reason": "", "summary": "took a suggestion",
    })
    rows, disc = diagnose_trajectory_mistakes_response(resp, [])
    assert disc is None
    m = next(r for r in rows if r["id"] == "mistake_0")
    assert m["justification"].startswith("[high] ") and m["result"] is False


# --- harness faults must not be charged to the agent (2026-08-12) --------------------------------
# In a corpus of recorded runs, 178 have judge findings quoting a harness
# error, and 9 of 1161 [high]/[critical] findings ARE the harness error itself. An independent harness
# review of 16 runs found tool 500s / Server-disconnected as its #2 ranked bug and noted "invisible
# tool errors need tool-log detectors, not vision-only prompting" — i.e. the agent is being penalised
# for infrastructure it cannot see or avoid.


def _mistakes_prompt():
    from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import _TRAJECTORY_MISTAKES_INSTRUCTIONS
    return _TRAJECTORY_MISTAKES_INSTRUCTIONS


def test_harness_faults_are_carved_out():
    p = _mistakes_prompt()
    assert "HARNESS FAULT IS NOT THE AGENT'S MISTAKE" in p
    for phrase in ("Server disconnected", "verified:false"):
        assert phrase in p, f"the carve-out must name {phrase} — it is what the judge actually sees"


def test_carve_out_does_not_excuse_blind_repetition():
    """The fault is free; an unthinking reaction to it is not. Without this the carve-out would
    launder genuine flail as 'the tool errored'."""
    p = _mistakes_prompt()
    assert "blindly repeating" in p
    assert "The fault is free" in p


def test_verified_false_is_explicitly_not_a_mistake():
    """The agent is INSTRUCTED to treat verified:false as entered (by its harness's system prompt), so
    flagging it punishes the agent for following its own instructions."""
    p = _mistakes_prompt()
    i = p.index("verified:false in particular")
    assert "does NOT mean the text failed to land" in p[i:i + 200]


def test_the_severity_ladder_follows_the_carve_out():
    """Ordering guard: the carve-out sits between the evidence rule and the severity ladder, so a finding
    is filtered for harness noise before it is priced."""
    p = _mistakes_prompt()
    assert p.index("WHAT YOU CANNOT SEE") < p.index("HARNESS FAULT IS NOT") < p.index("SEVERITY — choose the rung")
