"""The rubric judge echoes short keys (c1..cN), not 36-char UUIDs.

The judge used to be handed criteria keyed by 36-char UUIDs and matched by exact
string; models drop or transpose a character when copying a UUID, and one unknown
id discarded the entire judge response. The judge is now shown short positional
keys and the real ids are restored on the graded rows.
"""

import json

import pytest

from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import diagnose_rubric_binary_response
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    _restore_criterion_ids,
    _to_short_keyed_criteria,
)

# Real-shaped 36-char UUID ids — the strings the judge used to have to transcribe.
UUID_A = "cd93072d-4c77-4a1b-9f30-2b1c8de5a740"
UUID_B = "7cafc429-ae5e-40d9-84f6-7741bfd96e9c"
UUID_C = "1f0b6a35-9d21-4c88-b7e2-55aa0c9f31de"


def _criteria(ids):
    return [{"id": i, "criterion": f"criterion {i}", "weight": 50} for i in ids]


def _binary_response(pairs):
    return json.dumps(
        {"results": [{"id": i, "score": s, "justification": "j"} for i, s in pairs]}
    )


class TestToShortKeyedCriteria:
    def test_rekeys_to_c1_cn_and_maps_back(self):
        short, real_by_short = _to_short_keyed_criteria(_criteria([UUID_A, UUID_B, UUID_C]))
        assert [c["id"] for c in short] == ["c1", "c2", "c3"]
        assert real_by_short == {"c1": UUID_A, "c2": UUID_B, "c3": UUID_C}

    def test_judge_never_sees_a_uuid(self):
        short, _ = _to_short_keyed_criteria(_criteria([UUID_A, UUID_B]))
        assert all(c["id"] in {"c1", "c2"} for c in short)

    def test_preserves_other_fields_and_does_not_mutate_input(self):
        criteria = _criteria([UUID_A])
        criteria[0]["weight"] = -60
        short, _ = _to_short_keyed_criteria(criteria)
        assert short[0]["weight"] == -60
        assert short[0]["criterion"] == f"criterion {UUID_A}"
        assert criteria[0]["id"] == UUID_A  # input untouched


class TestGuardOnMalformedIds:
    """Malformed criteria (missing/duplicate/non-string ids) fail fast rather than
    being masked by generated short keys."""

    def test_missing_id_raises(self):
        with pytest.raises(ValueError):
            _to_short_keyed_criteria([{"id": UUID_A, "weight": 50}, {"weight": 50}])

    def test_duplicate_ids_raise(self):
        with pytest.raises(ValueError):
            _to_short_keyed_criteria(_criteria([UUID_A, UUID_A]))

    def test_non_string_id_raises(self):
        with pytest.raises(ValueError):
            _to_short_keyed_criteria([{"id": 7, "weight": 50}])


class TestRestoreCriterionIds:
    def test_restores_short_keys_to_real_ids(self):
        rows = [{"id": "c1", "score": 1.0}, {"id": "c2", "score": 0.0}]
        _restore_criterion_ids(rows, {"c1": UUID_A, "c2": UUID_B})
        assert [r["id"] for r in rows] == [UUID_A, UUID_B]

    def test_leaves_unknown_ids_untouched(self):
        rows = [{"id": "c9", "score": 1.0}, {"id": UUID_A, "score": 0.0}]
        _restore_criterion_ids(rows, {"c1": UUID_A})
        assert [r["id"] for r in rows] == ["c9", UUID_A]


class TestRoundTripThroughDiagnose:
    def test_short_key_response_grades_every_criterion(self):
        criteria = _criteria([UUID_A, UUID_B, UUID_C])
        short, real_by_short = _to_short_keyed_criteria(criteria)

        # The judge echoes the short keys it was shown — one per criterion.
        rows, discrepancy = diagnose_rubric_binary_response(
            _binary_response([("c1", 1.0), ("c2", 0.0), ("c3", 1.0)]), short
        )

        assert discrepancy is None
        assert rows is not None and len(rows) == 3
        _restore_criterion_ids(rows, real_by_short)
        by_id = {r["id"]: r for r in rows}
        assert set(by_id) == {UUID_A, UUID_B, UUID_C}
        assert by_id[UUID_A]["score"] == 1.0
        assert by_id[UUID_B]["score"] == 0.0
        assert by_id[UUID_A]["weight"] == 50  # real fields survive the round-trip

    def test_short_keys_survive_out_of_order_echo(self):
        criteria = _criteria([UUID_A, UUID_B, UUID_C])
        short, real_by_short = _to_short_keyed_criteria(criteria)
        rows, discrepancy = diagnose_rubric_binary_response(
            _binary_response([("c3", 1.0), ("c1", 0.0), ("c2", 1.0)]), short
        )
        assert discrepancy is None
        _restore_criterion_ids(rows, real_by_short)
        assert {r["id"] for r in rows} == {UUID_A, UUID_B, UUID_C}
