"""Registry of judge output formats: prompt templates, schemas, and parsers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from functools import partial
from string import Template
from typing import Callable, Sequence

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
        return out

    def __str__(self) -> str:
        if self.parse_error:
            return f"invalid JSON ({self.parse_error})"
        return (
            f"expected {self.expected_count} results but got {self.returned_count}"
            f" (missing={self.missing_ids}, duplicate={self.duplicate_ids}, unknown={self.unknown_ids})"
        )


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
    except json.JSONDecodeError as e:
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
    """Verification rows from ``diagnose``: invalid JSON degrades to all-fail rows, a wrong row set
    (missing / duplicate / unknown ids) raises."""
    results, discrepancy = diagnose(response_text, criteria)
    if discrepancy is not None:
        if discrepancy.parse_error:
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
    the CUA harness repo's golden gate greps that marker to compute ``fb_high``.
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
