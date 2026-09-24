"""Prompt + rubric construction for the file-based universe agent-judge (v1).

The agent-judge complements the programmatic verifier (``compare_dicts``). It is handed the
original + exported universes as files on its filesystem (``/universes/original``,
``/universes/export_1``, ``/universes/export_2``) and asked to flag only *major* (semantic)
differences — i.e. real data loss — while ignoring representation noise the programmatic side
already owns.

This module is pure (strings + a small criteria builder); it performs no I/O so it can be
unit-tested without external services.
"""

from __future__ import annotations

# Where LoadArtifactTaskStep stages the FileArtifactUniverse onto the judge agent.
UNIVERSES_DIR = "/universes"

# Prefix for the per-service rubric criteria; shared so judge_issues_from_verdict can recover the
# service name from a flagged criterion id.
SERVICE_CRITERION_PREFIX = "service_preserved__"
# Synthetic service bucket for cross-cutting (non-service-scoped) judge findings, so they render
# through the same per-service UNIVERSE_COMPATIBILITY path PV uses.
CROSS_SERVICE_KEY = "(cross-service)"
# Issue type/phase stamped on judge-derived issues (mirrors PV's issue shape).
JUDGE_ISSUE_TYPE = "agent_judge"
JUDGE_PHASE = "agent_judge"

# The definition of "semantically equivalent" — the IGNORE preamble (representation noise the
# programmatic verifier owns) and the FLAG list (real semantic/relational drift = data loss).
JUDGE_SYSTEM_PROMPT = f"""\
You are a meticulous data-fidelity judge. You are given two (sometimes three) snapshots of the \
same multi-service universe as JSON files on disk:

- `{UNIVERSES_DIR}/original/<service>.json` — the source-of-truth universe before any round-trip.
- `{UNIVERSES_DIR}/export_1/<service>.json` — the data after one load -> export-state cycle.
- `{UNIVERSES_DIR}/export_2/<service>.json` — the data after re-loading export_1 and exporting \
again (present only when available).

Your job is to decide whether the round-trip is **semantically lossless**: does the exported data \
preserve the *meaning* of the original? Compare `original` against `export_1` (and use `export_2` \
to corroborate stability). Read the files with your tools; do not guess.

VERIFY QUANTITATIVE CLAIMS PROGRAMMATICALLY — you have a code-execution environment, so use it. \
For anything involving math or counting — deciding whether two timestamps denote the same instant \
(across timezone offsets or epoch-seconds vs ISO), whether two numbers are equivalent, record \
counts, or whether a field stays present/populated across many records — WRITE AND RUN A SCRIPT \
that loads the JSON and checks it. Do NOT convert timestamps, compare numbers, or count records by \
hand: manual math and eyeballing large/sparse arrays are error-prone and are the most common source \
of both missed losses and false flags.

IGNORE the following — these are benign representation differences, NOT data loss:
- Timezone representation that resolves to the same instant (e.g. `-04:00` vs `Z`).
- Numeric formatting: `"12.50"` vs `12.5`, trailing zeros, int-vs-float, epoch-seconds vs ISO \
timestamp that denote the same moment.
- Server-added audit/enrichment fields present in the export but absent in the original (e.g. \
`created_at`, derived ids) — unless they overwrite or replace real data.
- Empty-equivalent values: `null` / `""` / `{{}}` / `[]`.
- Record or list ordering when the set of records is unchanged.
- Generation/scaffolding fields that are clearly build-time provenance (e.g. `noise_theme`) being \
dropped.

FLAG the following as a MAJOR difference (real data loss / corruption):
- A meaningful record present in the original is missing from the export (dropped record).
- A content-bearing value (amount, name, description, date meaning) changed or was lost.
- A populated business field in the original is null/absent in the export.
- Numeric precision loss beyond representation rounding.
- Free-text (description/memo/body) truncated, paraphrased, or semantically altered.
- Broken referential integrity: a reference that resolved in the original now dangles.
- Structural reshaping that drops information.
- Server-added values that are nonsensical placeholders replacing real data.

When in doubt about whether a dropped/changed field is *meaningful*, lean toward flagging it and \
explain your reasoning. Be specific: cite the service, entity, field, and an example value.\
"""


def build_user_prompt() -> str:
    """The task message handed to the judge agent."""
    return (
        f"Inspect the universe snapshots under `{UNIVERSES_DIR}/`. For each service, compare "
        f"`original/<service>.json` against `export_1/<service>.json` (corroborate with "
        f"`export_2/<service>.json` when present) and identify any MAJOR differences — real data "
        f"loss or corruption — applying the IGNORE/FLAG rules from your instructions. "
        f"Produce a concise per-service report: for each service state whether it is semantically "
        f"preserved, and list every major difference you found with the service, entity, field, "
        f"and an example (original value vs exported value). If a service is fully preserved "
        f"(only benign representation differences), say so explicitly."
    )


# Cross-cutting FLAG criteria (1.0 = the drift class is ABSENT / data preserved).
_CROSS_CUTTING_CRITERIA: list[dict] = [
    {"id": "record_preservation", "title": "Every meaningful record in the original (by identity) is present in the export; no real records were dropped (server-added records are fine)."},
    {"id": "value_fidelity", "title": "Content-bearing values (amounts, names, descriptions, dates) are preserved in MEANING across the round-trip; ignore timezone/number/format representation."},
    {"id": "referential_integrity", "title": "References between records/services still resolve after the round-trip; no orphaned or dangling references were introduced."},
    {"id": "numeric_precision", "title": "Numeric values are preserved to full precision; no sub-decimal loss beyond representation rounding."},
    {"id": "freetext_fidelity", "title": "Free-text fields (description/memo/body) are not truncated, paraphrased, or semantically altered."},
    {"id": "enrichment_sanity", "title": "Server-added fields/values are plausible and consistent, not nonsensical placeholders replacing real data."},
    {"id": "structure_preserved", "title": "Any structural reshaping (e.g. blob expanded into columns) preserves all information without loss."},
]


def build_criteria(environment_names: list[str]) -> list[dict]:
    """Build the rubric: one per-service preservation criterion plus the cross-cutting FLAG set.

    Each criterion is a dict with ``id``/``title``/``rubric_category``/``rubric_target`` as
    expected by ``RubricsVerifierTaskStep``. A score of 1.0 means the data was preserved (good).
    """
    criteria: list[dict] = []
    for name in environment_names:
        criteria.append({
            "id": f"{SERVICE_CRITERION_PREFIX}{name}",
            "title": (
                f"All data for the '{name}' service is semantically preserved between the original "
                f"and exported universes — no real records or content-bearing values were lost "
                f"(ignore representation noise and benign server enrichment)."
            ),
            "rubric_category": "Data Fidelity",
            "rubric_target": "Outcome",
        })
    for crit in _CROSS_CUTTING_CRITERIA:
        criteria.append({**crit, "rubric_category": "Data Fidelity", "rubric_target": "Outcome"})
    return criteria


def judge_issues_from_verdict(verdict: dict) -> dict[str, list[dict]]:
    """Map a rubric verdict's FLAGGED criteria into PV-shaped issues, grouped by service.

    Returns ``{environment_key: [issue, ...]}`` where each issue mirrors the programmatic verifier's
    issue shape (``entity``/``field``/``type``/``phase``/``critical``/``detail``) so it renders
    through the same UNIVERSE_COMPATIBILITY surface. Per-service criteria attach to their service;
    cross-cutting criteria attach under :data:`CROSS_SERVICE_KEY`. A criterion is "flagged" when its
    boolean ``result`` is False (or, absent that, its score is below 1.0). Flagged = the judge found
    a major (semantic) difference, so the issue is ``critical`` — gating the same way a PV critical
    does.
    """
    grouped: dict[str, list[dict]] = {}
    for r in verdict.get("results", []):
        result = r.get("result")
        flagged = result is False or (result is None and r.get("score", 1.0) < 1.0)
        if not flagged:
            continue
        cid = r.get("id", "")
        detail = r.get("justification") or "agent-judge flagged a major difference"
        if cid.startswith(SERVICE_CRITERION_PREFIX):
            service = cid[len(SERVICE_CRITERION_PREFIX):]
            field = "(semantic)"
        else:
            service = CROSS_SERVICE_KEY
            field = cid
        grouped.setdefault(service, []).append({
            "entity": service,
            "field": field,
            "type": JUDGE_ISSUE_TYPE,
            "phase": JUDGE_PHASE,
            "critical": True,
            "detail": detail,
        })
    return grouped


def apply_judge_verdict(compat: dict, verdict: dict) -> dict:
    """Merge a rubric verdict into a programmatic UNIVERSE_COMPATIBILITY result, in place.

    Flagged criteria become per-service critical issues (cross-cutting ones under
    :data:`CROSS_SERVICE_KEY`); any flag flips the affected service's and the overall ``compatible``
    to False — so the judge gates exactly the way a programmatic critical does. Existing programmatic
    issues are preserved (judge issues are appended). The raw verdict is stored under ``agent_judge``
    for audit. Returns the same (mutated) ``compat`` dict.
    """
    grouped = judge_issues_from_verdict(verdict)
    services = compat.setdefault("services", {})
    for environment_key, issues in grouped.items():
        entry = services.setdefault(environment_key, {"compatible": True, "issues": []})
        entry.setdefault("issues", []).extend(issues)
        entry["compatible"] = False
    if grouped:
        compat["compatible"] = False
    compat["agent_judge"] = verdict
    return compat
