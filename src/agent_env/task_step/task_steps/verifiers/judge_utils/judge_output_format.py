"""Registry of judge output formats: prompt templates, schemas, and parsers."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from string import Template
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

_SCAFFOLD_PATH = """An agent was given the following prompt and generated the trajectory and response below. Analyze the agent's trajectory and response against the following criteria.

## Agent Prompt
<agent_prompt>
$agent_prompt
</agent_prompt>

## Agent Trajectory
The agent's trajectory is available at $trajectory_path

## Agent Response
<agent_response>
$agent_response
</agent_response>

## Criteria
$criteria_json

"""

_SCAFFOLD_INLINE = """An agent was given the following prompt and generated the trajectory and response below. Analyze the agent's trajectory and response against the following criteria.

## Agent Prompt
<agent_prompt>
$agent_prompt
</agent_prompt>

## Agent Trajectory
<agent_trajectory>
$trajectory_content
</agent_trajectory>

## Agent Response
<agent_response>
$agent_response
</agent_response>

## Criteria
$criteria_json

"""

_SCAFFOLD_NO_TRAJECTORY = """An agent was given the following prompt and generated the response below. Analyze the agent's response against the following criteria.

## Agent Prompt
<agent_prompt>
$agent_prompt
</agent_prompt>

## Agent Response
<agent_response>
$agent_response
</agent_response>

## Criteria
$criteria_json

"""

# rubric_binary instructions — binary pass/fail. path/inline judge "behavior",
# no-trajectory judges "response"; everything above is the shared scaffold.
_BINARY_INSTRUCTIONS_BEHAVIOR = """## Instructions
For each criterion, determine whether the agent's behavior PASSES or FAILS.

For each criterion, return a result with:
- "id": the criterion's id
- "score": exactly 1.0 if the criterion is met, exactly 0.0 if it is not. Binary only — do NOT use partial scores like 0.5. If the criterion is partially met, choose the closer of 1.0 or 0.0 based on whether it would be considered "met" for grading purposes.
  Exception — `"negative": true` criteria (read carefully; this is the single most common place graders slip): the criterion text describes an anti-pattern the response should NOT exhibit, so the score is INVERTED relative to your instinct. Decide in two explicit, ordered steps and do NOT collapse them:
    1. First judge ONLY whether the named anti-pattern is PRESENT or ABSENT in the agent's output, and state that verdict verbatim ("anti-pattern ABSENT" or "anti-pattern PRESENT") in your justification.
    2. Then map that verdict mechanically to the score: ABSENT → 1.0 (good — the agent avoided the anti-pattern), PRESENT → 0.0 (bad — the agent exhibited it).
  Your score MUST match the verdict you wrote — absent ⇒ 1.0, present ⇒ 0.0, no exceptions. Never let a justification that concludes the anti-pattern is absent end on 0.0 (or one that concludes it is present end on 1.0). Grade only the anti-pattern the criterion names, and grade near-identical outputs identically. Same binary-only rule applies (exactly 1.0 or 0.0).
- "justification": a brief explanation of why you chose this score"""

_BINARY_INSTRUCTIONS_RESPONSE = """## Instructions
For each criterion, determine whether the agent's response PASSES or FAILS.

For each criterion, return a result with:
- "id": the criterion's id
- "score": exactly 1.0 if the criterion is met, exactly 0.0 if it is not. Binary only — do NOT use partial scores like 0.5. If the criterion is partially met, choose the closer of 1.0 or 0.0 based on whether it would be considered "met" for grading purposes.
  Exception — `"negative": true` criteria (read carefully; this is the single most common place graders slip): the criterion text describes an anti-pattern the response should NOT exhibit, so the score is INVERTED relative to your instinct. Decide in two explicit, ordered steps and do NOT collapse them:
    1. First judge ONLY whether the named anti-pattern is PRESENT or ABSENT in the agent's output, and state that verdict verbatim ("anti-pattern ABSENT" or "anti-pattern PRESENT") in your justification.
    2. Then map that verdict mechanically to the score: ABSENT → 1.0 (good — the agent avoided the anti-pattern), PRESENT → 0.0 (bad — the agent exhibited it).
  Your score MUST match the verdict you wrote — absent ⇒ 1.0, present ⇒ 0.0, no exceptions. Never let a justification that concludes the anti-pattern is absent end on 0.0 (or one that concludes it is present end on 1.0). Grade only the anti-pattern the criterion names, and grade near-identical outputs identically. Same binary-only rule applies (exactly 1.0 or 0.0).
- "justification": a brief explanation of why you chose this score"""

_RUBRIC_BINARY_PROMPT_PATH = Template(_SCAFFOLD_PATH + _BINARY_INSTRUCTIONS_BEHAVIOR)
_RUBRIC_BINARY_PROMPT_INLINE = Template(_SCAFFOLD_INLINE + _BINARY_INSTRUCTIONS_BEHAVIOR)
_RUBRIC_BINARY_PROMPT_NO_TRAJECTORY = Template(_SCAFFOLD_NO_TRAJECTORY + _BINARY_INSTRUCTIONS_RESPONSE)


# ---------------------------------------------------------------------------
# Rubric PARTIAL (partial-credit) templates.
#
# Unlike RUBRIC_BINARY, the judge may return any score in [0.0, 1.0] to express
# partial credit, or — when a criterion carries an `outcomes` list — must pick
# exactly one listed outcome and return its score (labeled/discrete grading).
# Score anchors are spelled out so the decimals stay calibrated across runs.
# ---------------------------------------------------------------------------
_RUBRIC_PARTIAL_INSTRUCTIONS = """## Instructions
For each criterion, assign a score reflecting HOW FULLY the agent met it. Partial credit is allowed.

For each criterion, return a result with:
- "id": the criterion's id
- "score": a numeric rating for the criterion, chosen as follows:
  * DEFAULT (criterion has no "outcomes"): a number from 0.0 to 1.0 (inclusive). Anchor your value:
    1.0 = fully met/correct; ~0.75 = met with minor gaps; ~0.5 = partially met; ~0.25 = mostly unmet;
    0.0 = absent/incorrect.
  * If "outcomes" is a LIST of labeled options: choose exactly one and return that outcome's "score"
    value verbatim — do not invent a value between the listed outcomes.
  * If "outcomes" is an OBJECT with "min" and "max" anchors (a continuous scale, e.g. min value
    -1.0 labeled "ugly" to max value 1.0 labeled "beautiful"): return a number anywhere within that
    [min, max] range using the anchors. Report it ON THAT SCALE — do NOT rescale to 0-1 yourself.
  Exception — "negative": true criteria (read carefully; this is where graders most often slip): the
  criterion text describes an anti-pattern the response should NOT exhibit, so the score is INVERTED
  relative to your instinct. Decide in two ordered steps: (1) first judge how present the named
  anti-pattern is and state that verdict in your justification; (2) then map it — fully ABSENT → 1.0
  (desirable), fully PRESENT → 0.0 (undesirable), partial presence → an intermediate value where a
  HIGHER score means MORE absent. Your score MUST agree with that verdict — never end a justification
  that concludes the anti-pattern is absent on a low score (or vice versa). Grade near-identical
  outputs identically.
- "justification": a brief explanation of why you chose this score"""

_RUBRIC_PARTIAL_PROMPT_PATH = Template(_SCAFFOLD_PATH + _RUBRIC_PARTIAL_INSTRUCTIONS)
_RUBRIC_PARTIAL_PROMPT_INLINE = Template(_SCAFFOLD_INLINE + _RUBRIC_PARTIAL_INSTRUCTIONS)
_RUBRIC_PARTIAL_PROMPT_NO_TRAJECTORY = Template(_SCAFFOLD_NO_TRAJECTORY + _RUBRIC_PARTIAL_INSTRUCTIONS)


# The mistake / botched / severity rules, shared by the trajectory_mistakes and rubric_evidence_with_mistakes
# formats. Platform-neutral on purpose: a product-specific rule (e.g. "an autocomplete shortcut is always a
# high-severity mistake") belongs in the task's ``grading_policy_prompt``, which every format appends as a
# "## Grading policy" section.
_MISTAKE_RULES = """Identify MISTAKES in the agent's trajectory — actions it should not have made, wrong
turns, ignored errors, redundant or looping behavior, premature stops, or fabricated
success — and assess whether the run was BOTCHED (the agent flailed without making real
progress). Report only real, defensible mistakes; do NOT invent problems in a clean run.

WHAT YOU CANNOT SEE IS NOT A MISTAKE. You get a SMALL SAMPLE of screens, and picker-set values
(date, guest count, size, filter) leave no trace in the actions. "The trajectory does not show
X" limits YOUR EVIDENCE — it is not evidence the agent failed.
- Never file a finding resting on absence ("should have verified X", "no confirmation of X")
  when the end state is consistent with X having been done.
- Resolve circumstantial evidence IN THE RUN'S FAVOUR: "Mon, Oct 5 – Sat, Oct 10" matching the
  requested year confirms it, it is not an open question.
- File a mistake only for something you can SEE go wrong. If you are writing "not clearly
  demonstrated", drop the finding.

A HARNESS FAULT IS NOT THE AGENT'S MISTAKE. When a tool errors — "Server disconnected without
sending a response", a 500, a type reporting verified:false, a screenshot timing out — that is
infrastructure failing, not the agent behaving badly. The agent cannot avoid it and prompting
cannot fix it.
- Do NOT file a finding for the error itself, and do NOT count the retries it forced as
  redundant/looping behaviour. A run that had to redo an action three times because the tool
  failed three times is not a flailing run.
- verified:false in particular does NOT mean the text failed to land: the field just could not be
  read back. The agent is told to treat it as entered, so doing so is CORRECT, not a mistake.
- What IS still a mistake: responding to an error by blindly repeating the same action many times
  without checking the screen. The fault is free; an unthinking reaction to it is not.
- If a tool error changed what you would otherwise have concluded, say so in the finding you do
  file, so a reader can tell agent behaviour from infrastructure noise.

SEVERITY — choose the rung by IMPACT ON THE RESULT, not by how odd the action looked. Each
finding subtracts a fixed penalty from the run's score and "high"/"critical" block the run from
being used as training data, so this choice decides whether an otherwise good run is thrown away:
- "critical": the run is unusable — the end state is wrong or the goal was defeated. Acted on the
  wrong entity (wrong address, store, account, item), committed when told to stop short, ended
  with nothing staged, or asserted an outcome the screens contradict.
- "high": the end state is broadly right, but a REQUIRED element is wrong or was never
  established — a stated constraint (price cap, quantity, date, size, "closest"/"cheapest") went
  unhonored or unchecked.
- "medium": the goal was met and the constraints hold, but the agent got there sloppily — a
  skipped confirmation, an avoidable detour, or steps taken out of order.
- "low": cosmetic only — a few redundant taps or a needless scroll, with no bearing on the result.
Judge each finding on its own. When one fits two rungs, take the LOWER unless the end state is
itself affected; reserve "critical" for runs a reviewer would discard outright."""

# The response fields the rules produce; rubric_evidence_with_mistakes derives `coverage` from the evidence rows
# instead of asking for it (`trajectory_mistakes_rows_from_evidence`).
_FINDINGS_FIELD = """- "findings": array of mistakes. Each object: {"severity": one of "low"|"medium"|"high"|"critical",
  "step": the 0-based action index where it occurred (or null), "should": what the agent
  should have done instead, "description": what actually went wrong}. Empty array if clean."""
_COVERAGE_FIELD = """- "coverage": for each criterion listed above, {"id": the criterion id, "shown": true if the
  trajectory clearly shows the agent doing/achieving it else false, "note": brief evidence}.
  Empty array if no criteria were provided."""
_BOTCHED_SUMMARY_FIELDS = """- "botched": true if the agent flailed — repeated the same failing action, stalled for long
  stretches, or never made real progress toward the goal — else false.
- "botched_reason": brief explanation when botched, else "".
- "summary": one or two sentences on overall trajectory quality."""

_TRAJECTORY_MISTAKES_INSTRUCTIONS = (
    "## Instructions\n" + _MISTAKE_RULES + "\n\nReturn a JSON object with:\n"
    + _FINDINGS_FIELD + "\n" + _COVERAGE_FIELD + "\n" + _BOTCHED_SUMMARY_FIELDS
)

_TRAJECTORY_MISTAKES_PROMPT_PATH = Template(_SCAFFOLD_PATH + _TRAJECTORY_MISTAKES_INSTRUCTIONS)
_TRAJECTORY_MISTAKES_PROMPT_INLINE = Template(_SCAFFOLD_INLINE + _TRAJECTORY_MISTAKES_INSTRUCTIONS)
_TRAJECTORY_MISTAKES_PROMPT_NO_TRAJECTORY = Template(_SCAFFOLD_NO_TRAJECTORY + _TRAJECTORY_MISTAKES_INSTRUCTIONS)


# Appended to a rubric_binary eval prompt ONLY when screenshots are attached to the
# judge (never mutates the templates above, so other consumers are unaffected).
# `.format(n=<num frames>)` before use.
SCREENSHOT_GROUNDING = """

## Final-state screenshots
The final {n} screenshot(s) from the agent's trajectory are attached, in chronological \
order; the LAST image is the final on-screen state. Only the FINAL frames are attached — \
earlier screens the agent visited are not shown here.

Grounding rules:
- For a criterion tagged `"backend": "final_state"` (an OUTCOME): PREFER the screenshots. \
If the outcome IS visible in an attached frame, that frame is authoritative — trust it over \
the agent's narration. If the outcome is NOT visible in any attached frame, FALL BACK to the \
trajectory text / action log to decide; do NOT score it 0.0 solely because it isn't in the \
attached frames (the evidence may be on an earlier screen that wasn't attached).
- For a criterion tagged `"backend": "intermediate"`, `"backend": "process"` \
(deprecated alias), or with no backend tag (a BEHAVIOR the agent performed): judge it from \
the trajectory text / action log as usual; the screenshots are only supplementary for these.
- A frame showing a control MID-INTERACTION — a calendar open, a picker part-scrolled, a \
dropdown expanded — is a TRANSIENT state, not the outcome. Whatever it has highlighted is \
where the control happened to sit when the frame was taken, not what the agent finally chose. \
Never read a final selection off such a frame, and never let one override the trajectory."""


# Replaces SCREENSHOT_GROUNDING when the frames were chosen per criterion (each image preceded by a
# "FRAME: <label>" text block). `{frame_rule}` is filled by `per_criterion_grounding`.
_SCREENSHOT_GROUNDING_PER_CRITERION = """

## Screenshots
{n} screenshot(s) from the agent's trajectory are attached, in chronological order. They were chosen \
across the WHOLE run for the criteria you are grading, not only from the end. Each image is preceded by \
a text block `FRAME: <label>`; a frame that serves several purposes carries every tag \
(`[c1][c2] final submitted action K`):
- `[cN] action K` — the frame after action K, chosen because action K most likely shows criterion cN \
(and the frame after it, since the effect of an action usually appears on the frame after it);
- `<label> action K` (e.g. `submitted action K`) — the screen right after an action the task configuration \
asked to always show; check what actually landed on screen against the action log's line K;
- `final action K` — the settled end state;
- `action K` — spread over the run for context.

Grounding rules:
{frame_rule}
- The action log is the record of what the agent DID (tool calls with their arguments and results) — use \
it for facts the pixels cannot show (the exact arguments passed, which application or page was opened and \
how). Lines starting `agent said:` are the agent's own words — claims to check against the frames, never \
evidence on their own.
- A frame showing a control MID-INTERACTION — a calendar open, a picker part-scrolled, a dropdown \
expanded — is a TRANSIENT state, not the outcome. Never read a final selection off such a frame."""

_PER_CRITERION_FRAME_RULE = """\
- Grade each criterion from the frames FIRST; when no attached frame shows it, decide from the \
trajectory text / action log — do NOT score it 0.0 solely because it is not in the attached frames."""

EVIDENCE_STATUSES = ["visible", "missing", "contradicted", "blocked", "unavailable", "ambiguous", "not_applicable"]
# Statuses that mean the criterion is NOT met whatever score the judge wrote.
BAD_EVIDENCE = frozenset({"missing", "contradicted", "blocked", "unavailable"})
# Every prompt that names the statuses derives its wording from these lists, so adding a status is one line.
_EVIDENCE_STATUS_GLOSS = {
    "visible": "the outcome is on screen",
    "missing": "a frame that should show it does not",
    "contradicted": "a frame shows the opposite",
    "blocked": "a dialog / error prevented it",
    "unavailable": "no frame covers it",
}
assert set(_EVIDENCE_STATUS_GLOSS) <= set(EVIDENCE_STATUSES) and BAD_EVIDENCE <= set(EVIDENCE_STATUSES)
_BAD_EVIDENCE_ORDERED = [s for s in EVIDENCE_STATUSES if s in BAD_EVIDENCE]
_EVIDENCE_STATUS_LIST = " / ".join(EVIDENCE_STATUSES)


def _or_list(items: Sequence[str]) -> str:
    """``"a, b or c"``."""
    return ", ".join(items[:-1]) + " or " + items[-1] if len(items) > 1 else "".join(items)


_PER_CRITERION_EVIDENCE_RULE = (
    "- Grade each criterion from the frames FIRST. In every result cite the label of the frame you relied "
    "on in `frame`, or `NOT SHOWN` if no attached frame shows it.\n"
    "- `evidence_status` says what the frames established: "
    + ", ".join(f"`{s}` ({_EVIDENCE_STATUS_GLOSS[s]})" if s in _EVIDENCE_STATUS_GLOSS else f"`{s}`"
                for s in EVIDENCE_STATUSES)
    + ". A criterion with " + _or_list([f"`{s}`" for s in _BAD_EVIDENCE_ORDERED]) + " evidence is NOT met."
)


def per_criterion_grounding(n: int, *, cites_evidence: bool) -> str:
    """The per-criterion screenshot section for ``n`` attached frames; only a format whose rows carry
    `frame` / `evidence_status` (``JudgeOutputFormatSpec.cites_evidence``) is told to cite the labels."""
    rule = _PER_CRITERION_EVIDENCE_RULE if cites_evidence else _PER_CRITERION_FRAME_RULE
    return _SCREENSHOT_GROUNDING_PER_CRITERION.format(n=n, frame_rule=rule)


def _rubric_score_output_format() -> dict:
    """Wire schema shared by rubric_binary and rubric_partial.

    `score` is already a free `number`, so the binary-vs-partial difference lives
    in the prompt and the result builder, not the schema — the judge is never
    schema-blocked from emitting a fraction.
    """
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "score": {"type": "number"},
                            "justification": {"type": "string"},
                        },
                        "required": ["id", "score", "justification"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    }


# An evidence_status outside EVIDENCE_STATUSES (possible without structured output) is stored as this.
_UNKNOWN_EVIDENCE_STATUS = "ambiguous"
# Caps on what the parser keeps from a rubric_evidence response (the verification entry is stored).
_MAX_REASONING_CHARS = 2000
_MAX_CHECKS = 12

# One mistake, as the trajectory_mistakes and rubric_evidence_with_mistakes formats both ask for it.
_FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "step": {"type": ["integer", "null"]},
        "should": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["severity", "step", "should", "description"],
    "additionalProperties": False,
}


def _rubric_evidence_output_format(*, with_mistakes: bool = False) -> dict:
    """Wire schema for rubric_evidence: binary rows that cite their evidence, plus `reasoning` and the
    policy `checks` (unscored; they ride on the verification entry). Field order is load-bearing:
    structured output is generated in schema order, so evidence comes before the verdict in every row and
    `reasoning` before the rows. ``with_mistakes`` (rubric_evidence_with_mistakes) adds the trajectory_mistakes
    fields after the rows."""
    check = {
        "type": "object",
        "properties": {
            "requirement": {"type": "string"},
            "required_value": {"type": "string"},
            "observed_value": {"type": "string"},
            "frame": {"type": "string"},
            "evidence_status": {"type": "string", "enum": EVIDENCE_STATUSES},
            "note": {"type": "string"},
        },
        "required": ["requirement", "required_value", "observed_value", "frame", "evidence_status", "note"],
        "additionalProperties": False,
    }
    properties: dict = {
        "reasoning": {"type": "string"},
        # No maxItems: not every provider's strict structured-output mode accepts it; the
        # parser caps the list at _MAX_CHECKS instead.
        "checks": {"type": "array", "items": check},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "frame": {"type": "string"},
                    "evidence_status": {"type": "string", "enum": EVIDENCE_STATUSES},
                    "justification": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["id", "frame", "evidence_status", "justification", "score"],
                "additionalProperties": False,
            },
        },
    }
    if with_mistakes:
        properties.update({
            "findings": {"type": "array", "items": _FINDING_SCHEMA},
            "botched": {"type": "boolean"},
            "botched_reason": {"type": "string"},
            "summary": {"type": "string"},
        })
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


# rubric_evidence adds the evidence fields to the rubric_binary instructions; `$checks_instruction` is
# `evidence_checks_instruction`, filled by the verifier.
_EVIDENCE_FIELDS = f"""
- "frame": the label of the attached screenshot you relied on (exactly as written after `FRAME:`), or "NOT SHOWN"
- "evidence_status": one of {_EVIDENCE_STATUS_LIST} — what the frames established (see the screenshot rules). A criterion whose evidence is {_or_list(_BAD_EVIDENCE_ORDERED)} is NOT met: score it 0.0.

Also return:
- "reasoning": a short account of what the frames show, written BEFORE the results
- "checks": $checks_instruction"""

_CHECKS_INSTRUCTION_WITH_POLICY = (
    "one policy check per requirement the grading policy below asks you to verify — each with the "
    "requirement, the value it required, the value you observed, the frame you saw it in, its "
    "evidence_status and a note. Return [] if it asks for none."
)
_CHECKS_INSTRUCTION_NO_POLICY = "[] (no grading policy was given)"


def evidence_checks_instruction(*, has_policy: bool) -> str:
    """The `checks` instruction of the rubric_evidence prompt: only mentions a grading policy when one was given."""
    return _CHECKS_INSTRUCTION_WITH_POLICY if has_policy else _CHECKS_INSTRUCTION_NO_POLICY


_RUBRIC_EVIDENCE_PROMPT_PATH = Template(_SCAFFOLD_PATH + _BINARY_INSTRUCTIONS_BEHAVIOR + _EVIDENCE_FIELDS)
_RUBRIC_EVIDENCE_PROMPT_INLINE = Template(_SCAFFOLD_INLINE + _BINARY_INSTRUCTIONS_BEHAVIOR + _EVIDENCE_FIELDS)
_RUBRIC_EVIDENCE_PROMPT_NO_TRAJECTORY = Template(
    _SCAFFOLD_NO_TRAJECTORY + _BINARY_INSTRUCTIONS_RESPONSE + _EVIDENCE_FIELDS)

# rubric_evidence_with_mistakes: the same call also detects mistakes.
_EVIDENCE_MISTAKES_SECTION = (
    "\n\n## Trajectory mistakes\n" + _MISTAKE_RULES + "\n\nAlso return, on the same JSON object:\n"
    + _FINDINGS_FIELD + "\n" + _BOTCHED_SUMMARY_FIELDS
)
_RUBRIC_EVIDENCE_MISTAKES_PROMPT_PATH = Template(
    _SCAFFOLD_PATH + _BINARY_INSTRUCTIONS_BEHAVIOR + _EVIDENCE_FIELDS + _EVIDENCE_MISTAKES_SECTION)
_RUBRIC_EVIDENCE_MISTAKES_PROMPT_INLINE = Template(
    _SCAFFOLD_INLINE + _BINARY_INSTRUCTIONS_BEHAVIOR + _EVIDENCE_FIELDS + _EVIDENCE_MISTAKES_SECTION)
_RUBRIC_EVIDENCE_MISTAKES_PROMPT_NO_TRAJECTORY = Template(
    _SCAFFOLD_NO_TRAJECTORY + _BINARY_INSTRUCTIONS_RESPONSE + _EVIDENCE_FIELDS + _EVIDENCE_MISTAKES_SECTION)


class ResultRows(list):
    """The validated rows of a rubric_evidence response plus what rides on the verification entry: the
    judge's ``reasoning`` and policy ``checks``, and for rubric_evidence_with_mistakes its ``mistakes``
    (``findings`` / ``botched`` / ``botched_reason`` / ``summary``; None otherwise). Other formats return a
    plain list."""

    reasoning: str
    checks: list[dict]
    mistakes: dict | None

    def __init__(self, rows: list, *, reasoning: str = "", checks: list[dict] | None = None,
                 mistakes: dict | None = None):
        super().__init__(rows)
        self.reasoning = reasoning
        self.checks = list(checks or [])
        self.mistakes = mistakes


# Default pass/fail threshold used to derive the boolean `result` from a partial
# score. `result` feeds all_pass/any_pass and the pass/fail display; partial
# rubrics should aggregate with `weighted_average`, which reads `score` directly.
PARTIAL_PASS_THRESHOLD = 0.5


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _continuous_range(criterion: dict) -> tuple[float, float] | None:
    """Return (min, max) if the criterion declares a continuous `outcomes` range.

    Mirrors the RFP shape ``outcomes: {"min": {"value": -1.0}, "max": {"value": 1.0}}``
    and also tolerates a flat ``{"min": -1.0, "max": 1.0}``. A *list* `outcomes` is the
    discrete/labeled case and returns None here (handled as-is, no rescaling).
    """
    outcomes = criterion.get("outcomes")
    if not isinstance(outcomes, dict):
        return None

    def _val(x: object) -> object:
        return x.get("value") if isinstance(x, dict) else x

    try:
        lo = float(_val(outcomes.get("min")))  # type: ignore[arg-type]
        hi = float(_val(outcomes.get("max")))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return (lo, hi)


def _discrete_outcome_scores(criterion: dict) -> list[float] | None:
    """Return the allowed scores if the criterion lists discrete `outcomes`.

    A *list* `outcomes` (e.g. `[{"label": "good", "score": 0.67}, ...]`) is the
    discrete/labeled case; the judge must return one of these scores. Returns None
    for the continuous cases (dict range or no `outcomes`).
    """
    outcomes = criterion.get("outcomes")
    if not isinstance(outcomes, list):
        return None
    scores: list[float] = []
    for o in outcomes:
        if isinstance(o, dict) and "score" in o:
            try:
                scores.append(float(o["score"]))
            except (TypeError, ValueError):
                continue
    return scores or None


def _pass_threshold(criterion: dict, rng: tuple[float, float] | None) -> float:
    """Per-criterion pass cutoff for the boolean `result`, as a normalized [0,1] value.

    An optional `pass_threshold` is read in the criterion's OWN units — the raw
    min..max scale when the criterion declares a continuous range, else [0,1] — and
    normalized the same way the score is, so threshold and score are always compared
    on the same scale. Absent/invalid values fall back to the global default.
    """
    raw = criterion.get("pass_threshold")
    if raw is None:
        return PARTIAL_PASS_THRESHOLD
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return PARTIAL_PASS_THRESHOLD
    if rng is not None and rng[1] > rng[0]:
        lo, hi = rng
        return _clamp01((raw - lo) / (hi - lo))
    return _clamp01(raw)


_CORRECTION_PROMPT_MAX_PREVIOUS_RESPONSE_CHARS = 8000


@dataclass(frozen=True)
class JudgeResponseDiscrepancy:
    expected_count: int
    returned_count: int
    expected_ids: list[str]
    missing_ids: list[str]
    duplicate_ids: list[str]
    unknown_ids: list[str]
    parse_error: str | None = None
    non_finite_score_ids: list[str] = field(default_factory=list)
    # What's wrong inside rows whose ids are right (user_sim_checks: a fail citing no graded turn, a verdict its own
    # reason contradicts, ...), one problem per entry naming its row.
    row_problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {
            "expected_count": self.expected_count,
            "returned_count": self.returned_count,
            "expected_ids": list(self.expected_ids),
            "missing_ids": list(self.missing_ids),
            "duplicate_ids": list(self.duplicate_ids),
            "unknown_ids": list(self.unknown_ids),
        }
        if self.parse_error is not None:
            out["parse_error"] = self.parse_error
        if self.non_finite_score_ids:
            out["non_finite_score_ids"] = list(self.non_finite_score_ids)
        if self.row_problems:
            out["row_problems"] = list(self.row_problems)
        return out

    def __str__(self) -> str:
        if self.parse_error:
            return f"invalid JSON ({self.parse_error})"
        if self.non_finite_score_ids:
            return _non_finite_score_problem(self.non_finite_score_ids)
        if self.row_problems:
            return "; ".join(self.row_problems)
        return (
            f"expected {self.expected_count} results but got {self.returned_count}"
            f" (missing={self.missing_ids}, duplicate={self.duplicate_ids}, unknown={self.unknown_ids})"
        )


def _non_finite_score_problem(ids: list[str]) -> str:
    return f"Each `score` must be a finite number; criterion id(s) {', '.join(ids)} were not"


def _default_failure_results(criteria: list[dict]) -> list[dict]:
    return [
        {**c, "score": 0.0, "result": False, "justification": ""}
        for c in criteria
    ]


def _parse_failure_discrepancy(criteria: list[dict], parse_error: str) -> JudgeResponseDiscrepancy:
    return JudgeResponseDiscrepancy(
        expected_count=len(criteria),
        returned_count=0,
        expected_ids=[c["id"] for c in criteria],
        missing_ids=[],
        duplicate_ids=[],
        unknown_ids=[],
        parse_error=parse_error,
    )


def _extract_rubric_binary_results(parsed: object) -> list | None:
    if isinstance(parsed, dict) and "results" in parsed:
        return parsed["results"]
    if isinstance(parsed, list):
        return parsed
    return None


def _analyze_result_ids(results: list, criteria: list[dict]) -> JudgeResponseDiscrepancy | None:
    criteria_ids = {c["id"] for c in criteria}
    result_ids = [item.get("id") for item in results]

    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for result_id in result_ids:
        if result_id in seen and result_id not in duplicate_ids:
            duplicate_ids.append(result_id)
        if isinstance(result_id, str):
            seen.add(result_id)

    result_id_set = {rid for rid in result_ids if isinstance(rid, str)}
    missing_ids = sorted(criteria_ids - result_id_set)
    unknown_ids = sorted(result_id_set - criteria_ids)

    if missing_ids or duplicate_ids or unknown_ids or len(results) != len(criteria):
        return JudgeResponseDiscrepancy(
            expected_count=len(criteria),
            returned_count=len(results),
            expected_ids=[c["id"] for c in criteria],
            missing_ids=missing_ids,
            duplicate_ids=duplicate_ids,
            unknown_ids=unknown_ids,
        )
    return None


def _build_validated_results(results: list, criteria: list[dict]) -> list[dict]:
    criteria_by_id = {c["id"]: c for c in criteria}
    validated = []
    for item in results:
        score = item.get("score", 0.0)
        criterion = criteria_by_id.get(item.get("id"), {})
        entry = {**criterion, **item, "score": score, "result": score == 1.0}
        validated.append(entry)
    return validated


def _build_validated_results_partial(results: list, criteria: list[dict]) -> list[dict]:
    """Like `_build_validated_results` but keeps partial-credit scores.

    The normalized `score` is always in [0.0, 1.0] (so aggregation math is
    unchanged); `result` is a threshold-derived boolean (partial rubrics should
    aggregate via `weighted_average`, which uses `score`). The pass cutoff is the
    criterion's own `pass_threshold` when set, else the global default.

    If a criterion declares a continuous `outcomes` range (e.g. RFP's −1.0→1.0),
    the judge's raw value on that scale is min-max normalized into [0,1] and the
    original value is preserved under `raw_score` for display/debugging. An unusable
    range (min >= max) logs a warning and falls back to a clamped raw score. If a
    criterion lists discrete `outcomes`, an off-list score is snapped to the nearest
    listed value (with a warning) to enforce the discrete constraint.
    """
    criteria_by_id = {c["id"]: c for c in criteria}
    validated = []
    for item in results:
        criterion = criteria_by_id.get(item.get("id"), {})
        try:
            raw = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            raw = 0.0

        entry = {**criterion, **item}
        rng = _continuous_range(criterion)
        discrete_scores = _discrete_outcome_scores(criterion)
        if rng is not None:
            # A continuous range was declared, so the judge scored on that scale —
            # always preserve its on-scale value, even if the range is unusable.
            lo, hi = rng
            entry["raw_score"] = raw
            if hi > lo:
                score = _clamp01((raw - lo) / (hi - lo))
            else:
                logger.warning(
                    "Rubric criterion %r declares an unusable continuous range "
                    "(min=%s, max=%s); cannot normalize, using clamped raw score.",
                    criterion.get("id"), lo, hi,
                )
                score = _clamp01(raw)
        elif discrete_scores is not None:
            # Enforce the discrete constraint: snap to the nearest listed outcome.
            snapped = min(discrete_scores, key=lambda s: abs(s - raw))
            if snapped != raw:
                logger.warning(
                    "Rubric criterion %r: judge score %s is not one of the listed "
                    "outcome scores %s; snapping to nearest (%s).",
                    criterion.get("id"), raw, discrete_scores, snapped,
                )
            score = _clamp01(snapped)
        else:
            score = _clamp01(raw)
        entry["score"] = score
        entry["result"] = score >= _pass_threshold(criterion, rng)
        validated.append(entry)
    return validated


def _is_non_finite_number(value: Any) -> bool:
    try:
        return not math.isfinite(float(value))
    except OverflowError:
        return True
    except (TypeError, ValueError):
        return False


def _parse_judge_results(
    response_text: str,
    criteria: list[dict],
) -> tuple[object, list | None, JudgeResponseDiscrepancy | None]:
    """Shared parse/validate path: JSON -> results array -> id check. Returns ``(parsed, results,
    discrepancy)`` — the parsed JSON so a format can read its other keys, and the id-checked results array
    or the discrepancy."""
    text = response_text.strip()

    try:
        parsed = json.loads(text)
    except ValueError as e:  # JSONDecodeError, or an integer literal past sys.get_int_max_str_digits()
        logger.error(f"Failed to parse judge response as JSON: {e}")
        logger.error(f"Raw response: {response_text[:500]}")
        return None, None, _parse_failure_discrepancy(criteria, str(e))

    results = _extract_rubric_binary_results(parsed)
    if results is None:
        logger.error(f"Unexpected judge response shape: {type(parsed)}")
        return parsed, None, _parse_failure_discrepancy(
            criteria, f"unexpected response shape: {type(parsed).__name__}"
        )

    discrepancy = _analyze_result_ids(results, criteria)
    if discrepancy is not None:
        return parsed, None, discrepancy

    non_finite_ids = [item["id"] for item in results if _is_non_finite_number(item.get("score"))]
    if non_finite_ids:
        return parsed, None, JudgeResponseDiscrepancy(
            expected_count=len(criteria),
            returned_count=len(results),
            expected_ids=[c["id"] for c in criteria],
            missing_ids=[],
            duplicate_ids=[],
            unknown_ids=[],
            non_finite_score_ids=non_finite_ids,
        )

    return parsed, results, None


def _diagnose_response(
    response_text: str,
    criteria: list[dict],
    build_results: Callable[[list, list[dict]], list[dict]],
) -> tuple[list[dict] | None, JudgeResponseDiscrepancy | None]:
    """`_parse_judge_results` -> rows. `build_results` decides how each row's `score`/`result` are
    derived, so binary and partial formats reuse the same JSON/id-integrity handling."""
    _, results, discrepancy = _parse_judge_results(response_text, criteria)
    if results is None:
        return None, discrepancy
    return build_results(results, criteria), None


def diagnose_rubric_binary_response(
    response_text: str,
    criteria: list[dict],
) -> tuple[list[dict] | None, JudgeResponseDiscrepancy | None]:
    """Parse a rubric-binary judge response, returning validated rows or a discrepancy."""
    return _diagnose_response(response_text, criteria, _build_validated_results)


def _evidence_status(item: dict) -> str:
    status = str(item.get("evidence_status") or "").strip().lower()
    return status if status in EVIDENCE_STATUSES else _UNKNOWN_EVIDENCE_STATUS


def _build_validated_results_evidence(results: list, criteria: list[dict]) -> list[dict]:
    """rubric_binary rows that also carry ``frame`` / ``evidence_status``. The score is snapped to binary
    (the entry is read as rubric_binary, and a stray 0.5 from a judge without structured output would earn
    partial credit under a weighted average), the raw value kept under ``raw_score``; bad evidence forces
    the row to fail whatever score the judge wrote."""
    criteria_by_id = {c["id"]: c for c in criteria}
    validated = []
    for item in results:
        try:
            raw = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            raw = 0.0
        score = 1.0 if raw >= 1.0 - 1e-9 else 0.0
        criterion = criteria_by_id.get(item.get("id"), {})
        entry = {**criterion, **item, "evidence_status": _evidence_status(item)}
        if raw != score:
            logger.warning("Rubric criterion %r: judge score %s is not binary; snapping to %s.",
                           item.get("id"), raw, score)
            entry["raw_score"] = raw
        if entry["evidence_status"] in BAD_EVIDENCE:
            score = 0.0
        validated.append({**entry, "score": score, "result": score == 1.0})
    return validated


def _mistakes_from_parsed(parsed: dict) -> tuple[dict | None, list[str]]:
    """The mistakes part of a rubric_evidence_with_mistakes response, or ``(None, [missing or mistyped field
    names])`` so the correction loop can ask again. ``botched_reason`` is optional."""
    missing = [name for name, ok in (
        ("findings", isinstance(parsed.get("findings"), list)),
        ("botched", isinstance(parsed.get("botched"), bool)),
        ("summary", isinstance(parsed.get("summary"), str)),
    ) if not ok]
    if missing:
        return None, missing
    return {
        "findings": [f for f in parsed["findings"] if isinstance(f, dict)],
        "botched": parsed["botched"],
        "botched_reason": str(parsed.get("botched_reason") or ""),
        "summary": parsed["summary"],
    }, []


def diagnose_rubric_evidence_response(
    response_text: str, criteria: list[dict], *, with_mistakes: bool = False,
) -> tuple[list[dict] | None, JudgeResponseDiscrepancy | None]:
    """Parse a rubric_evidence judge response into ``ResultRows`` (rows + ``reasoning`` + ``checks``) or a
    discrepancy. ``with_mistakes`` (rubric_evidence_with_mistakes) also requires ``findings`` / ``botched`` /
    ``summary`` — a missing one is a discrepancy, retried like invalid JSON — and lands them on
    ``ResultRows.mistakes``; otherwise those fields are ignored."""
    parsed, results, discrepancy = _parse_judge_results(response_text, criteria)
    if results is None:
        return None, discrepancy
    top = parsed if isinstance(parsed, dict) else {}
    reasoning = top["reasoning"][:_MAX_REASONING_CHARS] if isinstance(top.get("reasoning"), str) else ""
    checks = [c for c in top["checks"] if isinstance(c, dict)][:_MAX_CHECKS] if isinstance(top.get("checks"), list) else []
    mistakes = None
    if with_mistakes:
        mistakes, missing = _mistakes_from_parsed(top)
        if mistakes is None:
            return None, _parse_failure_discrepancy(criteria, f"missing or mistyped field(s): {', '.join(missing)}")
    rows = _build_validated_results_evidence(results, criteria)
    return ResultRows(rows, reasoning=reasoning, checks=checks, mistakes=mistakes), None


diagnose_rubric_evidence_mistakes_response = partial(diagnose_rubric_evidence_response, with_mistakes=True)


# Evidence on which a passing row counts as SHOWN in a derived coverage row.
_SHOWN_EVIDENCE = frozenset({"visible", "not_applicable"})


def trajectory_mistakes_rows_from_evidence(rows: ResultRows) -> list[dict]:
    """The trajectory_mistakes rows (the shape a standalone step writes) for a rubric_evidence_with_mistakes
    response: the judge's findings / botched / summary plus one coverage item per evidence row — ``shown``
    when the row passed on visible or not_applicable evidence, its ``note`` the row's justification. Coverage
    rows are keyed by whatever id the evidence rows carry when this is called."""
    if rows.mistakes is None:
        raise ValueError("rows carry no mistakes: not a rubric_evidence_with_mistakes response")
    coverage = [{
        "id": r.get("id"),
        "shown": r.get("result") is True and r.get("evidence_status") in _SHOWN_EVIDENCE,
        "note": r.get("justification", ""),
    } for r in rows]
    return _trajectory_mistakes_rows({**rows.mistakes, "coverage": coverage})


def diagnose_rubric_partial_response(
    response_text: str,
    criteria: list[dict],
) -> tuple[list[dict] | None, JudgeResponseDiscrepancy | None]:
    """Parse a rubric-partial (partial-credit) judge response into rows or a discrepancy."""
    return _diagnose_response(response_text, criteria, _build_validated_results_partial)


_BINARY_SCORE_INSTRUCTION = '- "score": exactly 1.0 or 0.0'
_PARTIAL_SCORE_INSTRUCTION = (
    '- "score": a numeric rating — 0.0 to 1.0 by default, one listed outcome\'s score, '
    "or a value within the criterion's min/max range"
)


def format_judge_correction_prompt(
    *,
    eval_prompt: str,
    previous_response: str,
    discrepancy: JudgeResponseDiscrepancy,
    max_previous_response_chars: int = _CORRECTION_PROMPT_MAX_PREVIOUS_RESPONSE_CHARS,
    score_instruction: str = _BINARY_SCORE_INSTRUCTION,
    extra_result_fields: Sequence[tuple[str, str]] = (),
    extra_object_fields: Sequence[tuple[str, str]] = (),
) -> str:
    """Build a follow-up prompt asking the judge to fix its output. ``extra_result_fields`` /
    ``extra_object_fields`` are ``(name, description)`` pairs a format requires beyond id / score /
    justification on each result, or beyond `results` on the object."""
    truncated_response = previous_response.strip()
    if len(truncated_response) > max_previous_response_chars:
        truncated_response = truncated_response[:max_previous_response_chars] + "\n... [truncated]"

    object_shape = "a `results` array"
    object_lines: list[str] = []
    if extra_object_fields:
        object_shape = ", ".join(f"`{name}`" for name, _ in extra_object_fields) + " and " + object_shape
        object_lines = ["", "The object must also include:",
                        *[f'- "{name}": {description}' for name, description in extra_object_fields]]

    if discrepancy.parse_error:
        problems = [
            f"Your response was not valid JSON: {discrepancy.parse_error}",
            f"Return only a JSON object with {object_shape} — no preamble, markdown fences, or extra text.",
        ]
    elif discrepancy.non_finite_score_ids:
        problems = [f"{_non_finite_score_problem(discrepancy.non_finite_score_ids)}."]
    else:
        problems = [
            f"You returned {discrepancy.returned_count} result(s) but there are "
            f"{discrepancy.expected_count} criteria."
        ]
        if discrepancy.missing_ids:
            problems.append(f"Missing result(s) for criterion id(s): {', '.join(discrepancy.missing_ids)}")
        if discrepancy.duplicate_ids:
            problems.append(f"Duplicate result(s) for criterion id(s): {', '.join(discrepancy.duplicate_ids)}")
        if discrepancy.unknown_ids:
            problems.append(f"Unknown criterion id(s) in your results: {', '.join(discrepancy.unknown_ids)}")

    correction_block = "\n".join([
        "## Correction Required",
        "",
        "Your previous response did not satisfy the output requirements:",
        *[f"- {problem}" for problem in problems],
        "",
        f"Return JSON with {object_shape} containing exactly {discrepancy.expected_count} objects — "
        "one per criterion, no duplicates, no extras.",
        f"Required criterion ids (in any order): {', '.join(discrepancy.expected_ids)}",
        "",
        "Each result must include:",
        '- "id": the criterion id',
        score_instruction,
        '- "justification": a brief explanation',
        *[f'- "{name}": {description}' for name, description in extra_result_fields],
        *object_lines,
        "",
        "## Your Previous Response",
        truncated_response,
    ])
    return f"{eval_prompt.rstrip()}\n\n{correction_block}"


def format_partial_judge_correction_prompt(
    *,
    eval_prompt: str,
    previous_response: str,
    discrepancy: JudgeResponseDiscrepancy,
    max_previous_response_chars: int = _CORRECTION_PROMPT_MAX_PREVIOUS_RESPONSE_CHARS,
) -> str:
    """Correction prompt variant for the partial format (allows [0,1] scores)."""
    return format_judge_correction_prompt(
        eval_prompt=eval_prompt,
        previous_response=previous_response,
        discrepancy=discrepancy,
        max_previous_response_chars=max_previous_response_chars,
        score_instruction=_PARTIAL_SCORE_INSTRUCTION,
    )


_EVIDENCE_RESULT_FIELDS = (
    ("frame", 'the label of the attached frame you relied on (as written after `FRAME:`), or "NOT SHOWN"'),
    ("evidence_status", "one of " + _EVIDENCE_STATUS_LIST),
)
_EVIDENCE_OBJECT_FIELDS = (
    ("reasoning", "a short account of what the frames show"),
    ("checks", "the policy checks (an array; [] if there are none)"),
)
_MISTAKES_OBJECT_FIELDS = (
    ("findings", "the mistakes (an array; [] if the run is clean), each with severity, step, should and description"),
    ("botched", "true or false"),
    ("botched_reason", 'a brief explanation when botched, else ""'),
    ("summary", "one or two sentences on overall trajectory quality"),
)


def format_evidence_judge_correction_prompt(
    *,
    eval_prompt: str,
    previous_response: str,
    discrepancy: JudgeResponseDiscrepancy,
    max_previous_response_chars: int = _CORRECTION_PROMPT_MAX_PREVIOUS_RESPONSE_CHARS,
    with_mistakes: bool = False,
) -> str:
    """The rubric_evidence correction prompt: asks for the evidence fields too, and with ``with_mistakes``
    (rubric_evidence_with_mistakes) the mistakes fields."""
    return format_judge_correction_prompt(
        eval_prompt=eval_prompt,
        previous_response=previous_response,
        discrepancy=discrepancy,
        max_previous_response_chars=max_previous_response_chars,
        extra_result_fields=_EVIDENCE_RESULT_FIELDS,
        extra_object_fields=_EVIDENCE_OBJECT_FIELDS + (_MISTAKES_OBJECT_FIELDS if with_mistakes else ()),
    )


def _rows_or_raise(diagnose: DiagnoseResponse, response_text: str, criteria: list[dict]) -> list[dict]:
    """Verification rows from ``diagnose``: invalid JSON or a non-finite score degrades to all-fail rows,
    a wrong row set (missing / duplicate / unknown ids) raises."""
    results, discrepancy = diagnose(response_text, criteria)
    if discrepancy is not None:
        if discrepancy.parse_error or discrepancy.non_finite_score_ids:
            return _default_failure_results(criteria)
        raise ValueError(
            f"Judge returned {discrepancy.returned_count} results but expected "
            f"{discrepancy.expected_count} criteria"
        )
    assert results is not None
    return results


_parse_rubric_binary_response = partial(_rows_or_raise, diagnose_rubric_binary_response)
_parse_rubric_partial_response = partial(_rows_or_raise, diagnose_rubric_partial_response)
_parse_rubric_evidence_response = partial(_rows_or_raise, diagnose_rubric_evidence_response)
_parse_rubric_evidence_mistakes_response = partial(_rows_or_raise, diagnose_rubric_evidence_mistakes_response)


def _trajectory_mistakes_output_format() -> dict:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "findings": {"type": "array", "items": _FINDING_SCHEMA},
                "coverage": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "shown": {"type": "boolean"},
                            "note": {"type": "string"},
                        },
                        "required": ["id", "shown", "note"],
                        "additionalProperties": False,
                    },
                },
                "botched": {"type": "boolean"},
                "botched_reason": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["findings", "coverage", "botched", "botched_reason", "summary"],
            "additionalProperties": False,
        },
    }


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


# Fallback only. How much each mistake severity subtracts from the coverage pass-rate,
# applied via a NEGATIVE row weight (see below) so these are absolute penalties on the
# final 0-1 score. The owning pipeline should pass its own table via the verifier step's
# ``mistake_severity_weights`` — these numbers want calibrating against human review, and
# that shouldn't need an agent-env release. This default exists so an unconfigured step
# still prices severity rather than silently treating every finding alike.
MISTAKE_SEVERITY_PENALTY = {
    "low": 0.02,
    "medium": 0.08,
    "high": 0.25,
    "critical": 1.0,  # a critical finding alone floors the score
}
_DEFAULT_SEVERITY = "medium"


def _trajectory_mistakes_rows(
    parsed: dict,
    severity_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Map a parsed mistakes response to RUBRIC-SHAPED rows so the existing hub
    rubric renderer displays them unchanged (id/score/result/justification).

    Scoring (WEIGHTED_AVERAGE, which the caller configures) is
    ``coverage_pass_rate - sum(severity penalties)``. Two properties matter:

    * **Severity is priced.** Mistake rows previously scored a flat ``0.0`` at weight 1,
      so a ``low`` nitpick cost exactly what a ``critical`` did. They now carry a negative
      weight, which ``aggregate_score`` treats as a penalty instead of a zero-scored row
      dragging down the mean.
    * **The reachable maximum no longer depends on rubric size.** Because a zero-scored row
      also grew the denominator, the old ceiling was ``(C+1)/(C+2+M)`` — a task emitting
      C<=4 coverage items could not reach 0.75 with even one mistake (best 0.714), while a
      C=11 task absorbed three. Scaling the weight by ``pos_den`` cancels that division, so
      the penalty is the same absolute hit regardless of how many coverage items a rubric
      happens to emit.

    ``severity_weights`` maps a severity to its penalty; the owning pipeline supplies it so
    the numbers can be tuned without an agent-env release. Falls back to
    ``MISTAKE_SEVERITY_PENALTY``, and an unrecognised severity falls back to the table's
    ``medium`` rather than scoring free.

    ``severity``, ``result`` and the ``"[sev] ..."`` justification prefix are unchanged —
    a downstream gate greps that marker out of the justification.
    """
    weights = severity_weights or MISTAKE_SEVERITY_PENALTY
    fallback = weights.get(_DEFAULT_SEVERITY, MISTAKE_SEVERITY_PENALTY[_DEFAULT_SEVERITY])
    findings = parsed.get("findings") or []
    coverage = parsed.get("coverage") or []
    botched = bool(parsed.get("botched"))
    clean = not findings and not botched

    # Positive rows: botched + one per coverage item. Mistake penalties are scaled by this
    # so that `(pos_num - penalty*pos_den) / pos_den` leaves the penalty undivided.
    pos_den = 1 + len(coverage)

    rows: list[dict] = [
        {
            "id": "summary",
            "score": 1.0 if clean else 0.0,
            "result": clean,
            # Informational only (weight 0). It flips to 0.0 on ANY finding, so scoring it
            # charged for the same mistake twice — once here, once on the mistake row.
            "weight": 0.0,
            "justification": parsed.get("summary", ""),
        },
        {
            "id": "botched",
            "score": 0.0 if botched else 1.0,
            "result": not botched,
            "justification": parsed.get("botched_reason", "") or ("Run botched" if botched else "No flailing detected"),
        },
    ]
    for i, f in enumerate(findings):
        sev = (f.get("severity") or "medium").lower()
        should = f.get("should", "")
        desc = f.get("description", "")
        penalty = weights.get(sev, fallback)
        rows.append({
            "id": f"mistake_{i}",
            "score": 0.0,
            "result": False,
            "severity": sev,
            "step": f.get("step"),
            "weight": -(penalty * pos_den),
            "justification": f"[{sev}] {should} — {desc}".strip(" —"),
        })
    for c in coverage:
        shown = bool(c.get("shown"))
        rows.append({
            "id": f"coverage_{c.get('id', '?')}",
            "score": 1.0 if shown else 0.0,
            "result": shown,
            "justification": ("shown: " if shown else "MISSING: ") + (c.get("note", "") or ""),
        })
    return rows


def diagnose_trajectory_mistakes_response(
    response_text: str,
    criteria: list[dict],
    severity_weights: dict[str, float] | None = None,
) -> tuple[list[dict] | None, JudgeResponseDiscrepancy | None]:
    """Parse a trajectory-mistakes judge response. Unlike rubric_binary this does NOT
    id-match against criteria (findings are open-ended); it only flags invalid JSON so
    the correction loop can retry.

    ``severity_weights`` is the caller's severity→penalty table (see
    ``_trajectory_mistakes_rows``); None uses the built-in fallback."""
    try:
        parsed = json.loads(_strip_json_fences(response_text))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse trajectory-mistakes judge response as JSON: {e}")
        return None, _parse_failure_discrepancy(criteria, str(e))
    if not isinstance(parsed, dict):
        return None, _parse_failure_discrepancy(criteria, f"unexpected response shape: {type(parsed).__name__}")
    return _trajectory_mistakes_rows(parsed, severity_weights), None


def _parse_trajectory_mistakes_response(
    response_text: str,
    criteria: list[dict],
    severity_weights: dict[str, float] | None = None,
) -> list[dict]:
    rows, discrepancy = diagnose_trajectory_mistakes_response(
        response_text, criteria, severity_weights)
    if discrepancy is not None:
        # Degrade to a single parse-error row rather than failing the task.
        return [{"id": "parse_error", "score": 0.0, "result": False,
                 "justification": f"Judge output unparseable: {discrepancy.parse_error}"}]
    assert rows is not None
    return rows


def format_trajectory_mistakes_correction(
    *,
    eval_prompt: str,
    previous_response: str,
    discrepancy: JudgeResponseDiscrepancy,
    max_previous_response_chars: int = _CORRECTION_PROMPT_MAX_PREVIOUS_RESPONSE_CHARS,
) -> str:
    truncated = previous_response.strip()
    if len(truncated) > max_previous_response_chars:
        truncated = truncated[:max_previous_response_chars] + "\n... [truncated]"
    correction_block = "\n".join([
        "## Correction Required",
        "",
        f"Your previous response was not valid JSON ({discrepancy.parse_error}).",
        "Return ONLY a JSON object with keys: findings, coverage, botched, botched_reason, summary.",
        "No preamble, no markdown fences, no extra text.",
        "",
        "## Your Previous Response",
        truncated,
    ])
    return f"{eval_prompt.rstrip()}\n\n{correction_block}"


# ---------------------------------------------------------------------------
# user_sim_checks: grades the user sim (the AI playing the user), not the assistant.
#
# Each criterion is one question about the user sim's run: a check (guidance, oversharing, ...) or one of the task's
# own rubric items. Fields the prompt and the parser read on a criterion, beyond `id`:
# - `question`: what to answer, with what fails, what doesn't, when n/a and when could_change_grade is true;
# - `kind` (task items): "hard" or "general"; `na_when`: when it never came up, "never" when n/a isn't allowed;
# - `graded_turns`: the turns marked GRADED, which a fail must cite (any cited turn counts when it's absent);
# - `weight`: 0 records the row but never lets it lower the score (style).
# The run goes in the template's trajectory placeholder, so the framing and the run come first, the same for every
# call on a run, and the criteria last: a prompt cache can share everything before them.
# ---------------------------------------------------------------------------
USER_SIM_VERDICTS = ("pass", "fail", "n/a")
_USER_SIM_NA_NEVER = "never"

_USER_SIM_FRAMING = """You check a user simulator ("the user sim"): an AI playing a real person who messages their \
personal AI assistant. You grade the user sim, never the assistant. You are given:
1. THE USER SIM'S INSTRUCTIONS: exactly what it was told about who it is, what it knows, its decisions, its rules for \
the assistant and what is private. Hold it to these lines only.
2. THE RUN: every check-in it got, word for word ("what it was sent": the time, its new and earlier emails, texts and \
notifications, the lines it sent, and the assistant's new messages), each followed by its answer: "reply" (the \
message reaches the assistant), "no_reply" (nothing is sent) or "done" (a sign-off).
3. THE CRITERIA: the questions to answer about it.

How to read the run:
- What it was sent at a turn, and at every earlier turn, is what it knew; a fact it was never shown is not one it \
forgot.
- Turns marked CONTEXT ONLY are fixed history: never fail the user sim for them, but they still count as what it said.
- Grade only the turns marked GRADED (each criterion's "graded_turns"). Cite turns by their number.
- The assistant's own words are the assistant's, even when they repeat an email.

How to answer:
- Answer each criterion's "question" only, narrowly, by its own definition; other problems belong to other criteria. \
A failure needs clear evidence in a graded answer.
- A criterion's "question" says what fails, what is not a failure, when to answer "n/a" and when "could_change_grade" \
is true.
- A task rubric item gives its "kind". A "hard" item is an exact rule from the user sim's instructions: it fails on \
any breach of the list or value it names, however small. A "general" item is a judgement about how this person would \
behave: it fails only on a clear instance.
- An item's "na_when" is the situation in which it never came up: answer "n/a" for it then, and only then. \
"na_when": "never" means it always applies: answer pass or fail.
- Reason first and decide last. Write the reason, then the turns it rests on, and only then the verdict, which \
follows from the reason: a reason that finds no clear failure gives "pass" (or "n/a"), never "fail". Keep every \
reason to at most 40 words: the evidence and the conclusion, not your working.
- "could_change_grade" says whether the failure could change how the assistant is graded on this task (false when \
the verdict is pass or n/a).
- "assistant_asked" is true when the assistant asked for or invited the behaviour you graded (e.g. asked the user \
to write a message or to give a fact it should look up); false otherwise. Give it on every result, false when \
nothing was asked; it is recorded only and doesn't change the verdict.

Return only this JSON object, with one result per criterion, each result's keys in this order:
{"results": [{"id": "the criterion's id", "reason": "at most 40 words", "turns": [turn numbers the reason rests on], \
"verdict": "pass" | "fail" | "n/a", "could_change_grade": true | false, "assistant_asked": true | false}]}"""

_USER_SIM_CRITERIA = """

# THE CRITERIA
$criteria_json

Return only the JSON object."""

# `$trajectory_content` is the user sim's instructions and its run as the user sim judge renders them.
_USER_SIM_CHECKS_PROMPT_INLINE = Template(_USER_SIM_FRAMING + "\n\n$trajectory_content" + _USER_SIM_CRITERIA)
_USER_SIM_CHECKS_PROMPT_PATH = Template(
    _USER_SIM_FRAMING + "\n\n# THE USER SIM'S INSTRUCTIONS AND THE RUN\nBoth are in the file at $trajectory_path: "
    "read all of it before you answer." + _USER_SIM_CRITERIA)
# Without a trajectory the run comes in the sections the verifier adds after the template, so after the criteria.
_USER_SIM_CHECKS_PROMPT_NO_TRAJECTORY = Template(
    _USER_SIM_FRAMING + _USER_SIM_CRITERIA + "\n\nThe user sim's instructions and the run follow.")


def _user_sim_checks_output_format() -> dict:
    """Reason before verdict in every row: structured output is generated in schema order, so the judge decides after
    reasoning. The parser reads the keys by name, in any order."""
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "reason": {"type": "string"},
                            "turns": {"type": "array", "items": {"type": "integer"}},
                            "verdict": {"type": "string", "enum": list(USER_SIM_VERDICTS)},
                            "could_change_grade": {"type": "boolean"},
                            "assistant_asked": {"type": "boolean"},
                        },
                        "required": ["id", "reason", "turns", "verdict", "could_change_grade", "assistant_asked"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    }


# A reason concluding nothing failed ("No clear delegation failures found"); a backstop to the reason-first prompt.
_NO_FAILURE_FOUND = re.compile(
    r"\bno\s+(?:clear\s+)?(?:\w+\s+)?failures?\s+(?:was\s+|were\s+)?(?:found|detected|identified)\b"
    r"|\bno\s+clear\s+(?:\w+\s+)?failures?\b|\bnothing\s+fails\b", re.I)


def _reason_says_no_failure(reason: str) -> str | None:
    """What in the reason's conclusion says no failure was found, or None. Only its last sentence is its conclusion:
    "no clear failure at turn 1; at turn 2 she leaked it" is a real fail."""
    sentences = [x for x in re.split(r"[.;!?]+", reason) if x.strip()]
    m = _NO_FAILURE_FOUND.search(sentences[-1]) if sentences else None
    return m.group(0) if m else None


def _cited_turns(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        try:
            out.append(int(v))
        except (TypeError, ValueError, OverflowError):  # OverflowError: an infinite number
            continue
    return out


def _graded_turns(criterion: dict) -> set[int] | None:
    value = criterion.get("graded_turns")
    return set(_cited_turns(value)) if isinstance(value, list) else None


def _user_sim_json(text: str) -> Any:
    """The JSON in an answer, tolerating a markdown fence or text around the object; None when there is none."""
    s = str(text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except ValueError:
            return None


def _never_gates(criterion: dict) -> bool:
    try:
        return float(criterion.get("weight", 1.0)) == 0.0
    except (TypeError, ValueError):
        return False


def _user_sim_row(item: dict, criterion: dict) -> tuple[dict | None, list[str]]:
    """One result checked against its criterion: ``(row, [])``, or ``(None, problems)`` naming the row. The row
    lowers the score (score 0.0, result false) only on a fail with could_change_grade, and never at weight 0;
    ``verdict`` and ``could_change_grade`` keep what the judge said either way. ``assistant_asked`` (the assistant
    asked for or invited the behaviour) is required and recorded only: it never changes the score or a gate."""
    rid = item.get("id")
    problems = []
    verdict = str(item.get("verdict", "")).strip().lower()
    if verdict not in USER_SIM_VERDICTS:
        problems.append(f"{rid}: verdict {item.get('verdict')!r} is not one of {list(USER_SIM_VERDICTS)}")
    change = item.get("could_change_grade")
    if not isinstance(change, bool):
        problems.append(f"{rid}: could_change_grade must be true or false")
    asked = item.get("assistant_asked")
    if not isinstance(asked, bool):
        problems.append(f"{rid}: assistant_asked must be true or false")
    if verdict == "n/a" and str(criterion.get("na_when") or "").strip().lower() == _USER_SIM_NA_NEVER:
        problems.append(f"{rid} may not be n/a: it always applies, so answer pass or fail")
    if not isinstance(item.get("reason"), str):
        problems.append(f"{rid}: reason must be text")
    if not isinstance(item.get("turns"), list):
        problems.append(f"{rid}: turns must be a list of turn numbers")
    turns = _cited_turns(item.get("turns"))
    reason = item["reason"].strip() if isinstance(item.get("reason"), str) else ""
    graded = _graded_turns(criterion)
    if verdict == "fail":
        if not any(graded is None or t in graded for t in turns):
            problems.append(f"{rid}: a fail must cite at least one graded turn; cited {turns}"
                            + (f", graded {sorted(graded)}" if graded is not None else ""))
        said = _reason_says_no_failure(reason)
        if said:
            problems.append(f"{rid}: a fail whose reason says {said!r} contradicts itself; the verdict must follow "
                            "the reason")
    if problems:
        return None, problems
    change = change and verdict == "fail"
    lowers = change and not _never_gates(criterion)
    row = {**criterion, "id": rid, "reason": reason,
           "turns": [t for t in turns if graded is None or t in graded],
           "verdict": verdict, "could_change_grade": change, "assistant_asked": asked,
           "justification": reason, "score": 0.0 if lowers else 1.0, "result": not lowers}
    outside = [t for t in turns if graded is not None and t not in graded]
    if outside:
        row["turns_outside_graded"] = outside
    return row, []


def diagnose_user_sim_checks_response(
    response_text: str, criteria: list[dict],
) -> tuple[list[dict] | None, JudgeResponseDiscrepancy | None]:
    """Parse a user_sim_checks judge response into rows, or a discrepancy for the correction prompt: no JSON
    object, the wrong row set, or a row the user sim judge can't use: an id, reason or turns of the wrong type, a
    verdict, could_change_grade or assistant_asked out of range, an n/a where ``na_when`` is never, a fail citing no
    graded turn, or a fail whose reason concludes nothing failed."""
    parsed = _user_sim_json(response_text)
    if parsed is None:
        return None, _parse_failure_discrepancy(criteria, "the answer is not a JSON object")
    results = _extract_rubric_binary_results(parsed)
    if not isinstance(results, list):
        shape = ("`results` must be a list" if results is not None
                 else f"unexpected response shape: {type(parsed).__name__}")
        return None, _parse_failure_discrepancy(criteria, shape)
    if not all(isinstance(r, dict) for r in results):
        return None, _parse_failure_discrepancy(criteria, "every result must be a JSON object")
    # The shared id check hashes each id, so one that isn't text (a list, an object) would raise instead of being
    # asked again.
    bad_ids = [f"result {i}: id must be a criterion id (text), not {r.get('id')!r}"
               for i, r in enumerate(results, 1) if not isinstance(r.get("id"), str)]
    if bad_ids:
        return None, JudgeResponseDiscrepancy(
            expected_count=len(criteria), returned_count=len(results), expected_ids=[c["id"] for c in criteria],
            missing_ids=[], duplicate_ids=[], unknown_ids=[], row_problems=bad_ids)
    discrepancy = _analyze_result_ids(results, criteria)
    if discrepancy is not None:
        return None, discrepancy
    by_id = {c["id"]: c for c in criteria}
    rows: list[dict] = []
    problems: list[str] = []
    for item in results:
        row, its_problems = _user_sim_row(item, by_id[item["id"]])
        problems += its_problems
        if row is not None:
            rows.append(row)
    if problems:
        return None, JudgeResponseDiscrepancy(
            expected_count=len(criteria), returned_count=len(results), expected_ids=[c["id"] for c in criteria],
            missing_ids=[], duplicate_ids=[], unknown_ids=[], row_problems=problems)
    return rows, None


def _parse_user_sim_checks_response(response_text: str, criteria: list[dict]) -> list[dict]:
    """Rows, or ValueError: an unusable answer has no verdict, so it never degrades to fails that would gate."""
    rows, discrepancy = diagnose_user_sim_checks_response(response_text, criteria)
    if discrepancy is not None:
        raise ValueError(f"user_sim_checks answer not usable: {discrepancy}")
    assert rows is not None
    return rows


def format_user_sim_checks_correction(
    *,
    eval_prompt: str,
    previous_response: str,
    discrepancy: JudgeResponseDiscrepancy,
    max_previous_response_chars: int = _CORRECTION_PROMPT_MAX_PREVIOUS_RESPONSE_CHARS,
) -> str:
    """The user_sim_checks correction prompt: what was wrong, then the row shape again."""
    truncated = previous_response.strip()
    if len(truncated) > max_previous_response_chars:
        truncated = truncated[:max_previous_response_chars] + "\n... [truncated]"
    if discrepancy.parse_error:
        problems = [f"Your response could not be read: {discrepancy.parse_error}",
                    "Return only a JSON object with a `results` array — no preamble, markdown fences, or extra text."]
    elif discrepancy.row_problems:
        problems = list(discrepancy.row_problems)
    else:
        problems = [f"You returned {discrepancy.returned_count} result(s) but there are "
                    f"{discrepancy.expected_count} criteria."]
        for ids, what in ((discrepancy.missing_ids, "Missing result(s) for criterion id(s)"),
                          (discrepancy.duplicate_ids, "Duplicate result(s) for criterion id(s)"),
                          (discrepancy.unknown_ids, "Unknown criterion id(s) in your results")):
            if ids:
                problems.append(f"{what}: {', '.join(ids)}")
    correction_block = "\n".join([
        "## Correction Required",
        "",
        "Your previous response did not satisfy the output requirements:",
        *[f"- {problem}" for problem in problems],
        "",
        f"Return JSON with a `results` array containing exactly {discrepancy.expected_count} objects — "
        "one per criterion, no duplicates, no extras.",
        f"Required criterion ids (in any order): {', '.join(discrepancy.expected_ids)}",
        "",
        "Each result must include, in this order:",
        '- "id": the criterion id',
        '- "reason": at most 40 words, written before the verdict',
        '- "turns": the turn numbers the reason rests on; a fail cites at least one graded turn',
        '- "verdict": "pass", "fail" or "n/a", following from the reason: a reason that finds no clear failure is '
        'never "fail"',
        '- "could_change_grade": true or false',
        '- "assistant_asked": true when the assistant asked for or invited the behaviour you graded, else false',
        "",
        "## Your Previous Response",
        truncated,
    ])
    return f"{eval_prompt.rstrip()}\n\n{correction_block}"


class JudgeOutputFormat(str, Enum):
    RUBRIC_BINARY = "rubric_binary"
    RUBRIC_PARTIAL = "rubric_partial"
    TRAJECTORY_MISTAKES = "trajectory_mistakes"
    # rubric_binary rows that cite their evidence (frame + evidence_status); stored as format "rubric_binary".
    RUBRIC_EVIDENCE = "rubric_evidence"
    # rubric_evidence whose judge call also detects trajectory mistakes (findings / botched / summary on the same
    # object); the entry is stored exactly as rubric_evidence's and the verifier writes a second,
    # trajectory_mistakes-shaped entry from the findings.
    RUBRIC_EVIDENCE_WITH_MISTAKES = "rubric_evidence_with_mistakes"
    # Grades the user sim (the AI playing the user), not the assistant: per criterion a reason, the turns it cites, a
    # pass / fail / n/a verdict, could_change_grade and assistant_asked (recorded only); a row lowers the score only on a fail that could change the grade.
    USER_SIM_CHECKS = "user_sim_checks"


DiagnoseResponse = Callable[[str, list[dict]], tuple[list[dict] | None, JudgeResponseDiscrepancy | None]]
FormatCorrectionPrompt = Callable[..., str]


@dataclass(frozen=True)
class JudgeOutputFormatSpec:
    output_format: dict
    prompt_template_path: Template
    prompt_template_inline: Template
    prompt_template_no_trajectory: Template
    parse_response: Callable[[str, list[dict]], list[dict]]
    diagnose_response: DiagnoseResponse
    format_correction_prompt: FormatCorrectionPrompt
    # The `format` written on the verification entry when it differs from the enum value (rubric_binary-shaped
    # rows store themselves as rubric_binary so existing readers see no change); both evidence formats set it.
    stored_format: str | None = None
    # Rows carry `frame` / `evidence_status`, so the per-criterion grounding asks the judge to cite them; a format
    # with this set needs the labelled per-criterion frames (the verifier enforces the configuration).
    cites_evidence: bool = False


JUDGE_OUTPUT_FORMAT_REGISTRY: dict[JudgeOutputFormat, JudgeOutputFormatSpec] = {
    JudgeOutputFormat.RUBRIC_BINARY: JudgeOutputFormatSpec(
        output_format=_rubric_score_output_format(),
        prompt_template_path=_RUBRIC_BINARY_PROMPT_PATH,
        prompt_template_inline=_RUBRIC_BINARY_PROMPT_INLINE,
        prompt_template_no_trajectory=_RUBRIC_BINARY_PROMPT_NO_TRAJECTORY,
        parse_response=_parse_rubric_binary_response,
        diagnose_response=diagnose_rubric_binary_response,
        format_correction_prompt=format_judge_correction_prompt,
    ),
    JudgeOutputFormat.RUBRIC_PARTIAL: JudgeOutputFormatSpec(
        output_format=_rubric_score_output_format(),
        prompt_template_path=_RUBRIC_PARTIAL_PROMPT_PATH,
        prompt_template_inline=_RUBRIC_PARTIAL_PROMPT_INLINE,
        prompt_template_no_trajectory=_RUBRIC_PARTIAL_PROMPT_NO_TRAJECTORY,
        parse_response=_parse_rubric_partial_response,
        diagnose_response=diagnose_rubric_partial_response,
        format_correction_prompt=format_partial_judge_correction_prompt,
    ),
    JudgeOutputFormat.RUBRIC_EVIDENCE: JudgeOutputFormatSpec(
        output_format=_rubric_evidence_output_format(),
        prompt_template_path=_RUBRIC_EVIDENCE_PROMPT_PATH,
        prompt_template_inline=_RUBRIC_EVIDENCE_PROMPT_INLINE,
        prompt_template_no_trajectory=_RUBRIC_EVIDENCE_PROMPT_NO_TRAJECTORY,
        parse_response=_parse_rubric_evidence_response,
        diagnose_response=diagnose_rubric_evidence_response,
        format_correction_prompt=format_evidence_judge_correction_prompt,
        stored_format=JudgeOutputFormat.RUBRIC_BINARY.value,
        cites_evidence=True,
    ),
    JudgeOutputFormat.RUBRIC_EVIDENCE_WITH_MISTAKES: JudgeOutputFormatSpec(
        output_format=_rubric_evidence_output_format(with_mistakes=True),
        prompt_template_path=_RUBRIC_EVIDENCE_MISTAKES_PROMPT_PATH,
        prompt_template_inline=_RUBRIC_EVIDENCE_MISTAKES_PROMPT_INLINE,
        prompt_template_no_trajectory=_RUBRIC_EVIDENCE_MISTAKES_PROMPT_NO_TRAJECTORY,
        parse_response=_parse_rubric_evidence_mistakes_response,
        diagnose_response=diagnose_rubric_evidence_mistakes_response,
        format_correction_prompt=partial(format_evidence_judge_correction_prompt, with_mistakes=True),
        stored_format=JudgeOutputFormat.RUBRIC_BINARY.value,
        cites_evidence=True,
    ),
    JudgeOutputFormat.TRAJECTORY_MISTAKES: JudgeOutputFormatSpec(
        output_format=_trajectory_mistakes_output_format(),
        prompt_template_path=_TRAJECTORY_MISTAKES_PROMPT_PATH,
        prompt_template_inline=_TRAJECTORY_MISTAKES_PROMPT_INLINE,
        prompt_template_no_trajectory=_TRAJECTORY_MISTAKES_PROMPT_NO_TRAJECTORY,
        parse_response=_parse_trajectory_mistakes_response,
        diagnose_response=diagnose_trajectory_mistakes_response,
        format_correction_prompt=format_trajectory_mistakes_correction,
    ),
    JudgeOutputFormat.USER_SIM_CHECKS: JudgeOutputFormatSpec(
        output_format=_user_sim_checks_output_format(),
        prompt_template_path=_USER_SIM_CHECKS_PROMPT_PATH,
        prompt_template_inline=_USER_SIM_CHECKS_PROMPT_INLINE,
        prompt_template_no_trajectory=_USER_SIM_CHECKS_PROMPT_NO_TRAJECTORY,
        parse_response=_parse_user_sim_checks_response,
        diagnose_response=diagnose_user_sim_checks_response,
        format_correction_prompt=format_user_sim_checks_correction,
    ),
}


def get_judge_output_format_spec(fmt: JudgeOutputFormat | str) -> JudgeOutputFormatSpec:
    if isinstance(fmt, str):
        try:
            fmt = JudgeOutputFormat(fmt)
        except ValueError as e:
            raise ValueError(f"Unknown judge output format: {fmt}") from e
    try:
        return JUDGE_OUTPUT_FORMAT_REGISTRY[fmt]
    except KeyError as e:
        raise ValueError(f"Unknown judge output format: {fmt}") from e
