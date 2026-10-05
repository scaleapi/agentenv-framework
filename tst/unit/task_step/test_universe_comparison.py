"""Unit tests for the universe load/export roundtrip comparison.

Locks in the semantic-comparison behavior, including two correctness cases:
  * large financial amounts in the epoch range must NOT be coerced to timestamps
    (otherwise distinct amounts collapse to the same second — a silent false-negative);
  * a DROPPED non-empty nested-dict key is real data loss and must fail, while an
    ADDED key is benign server enrichment.
"""

from agent_env.task_step.task_steps.multienv_validator.universe_comparison import (
    _vals_equal,
    classify_issues,
    compare_dicts,
)


class TestFinancialAmountsNotTimestamps:
    """Numbers in the epoch range are amounts, not timestamps."""

    def test_distinct_amounts_in_epoch_range_differ(self):
        # Both fall in the old 1.4e9–2.0e9 "epoch seconds" window; sub-second precision
        # must be preserved so they don't compare equal.
        assert _vals_equal(1_500_000_001.50, 1_500_000_001.75) is False

    def test_distinct_amount_strings_in_epoch_range_differ(self):
        assert _vals_equal("1500000001", 1500000002) is False

    def test_equal_amounts_still_equal(self):
        assert _vals_equal(1_500_000_001.50, 1_500_000_001.50) is True

    def test_numeric_string_matches_number(self):
        assert _vals_equal("12.50", 12.5) is True


class TestNestedDictKeyAsymmetry:
    """Dropped universe keys fail; added export keys are enrichment."""

    def test_dropped_nonempty_key_fails(self):
        assert _vals_equal(
            {"ledger": {"balance": 1000, "currency": "USD"}},
            {"ledger": {"balance": 1000}},
        ) is False

    def test_added_key_is_enrichment(self):
        assert _vals_equal(
            {"ledger": {"balance": 1000}},
            {"ledger": {"balance": 1000, "id": "x"}},
        ) is True

    def test_dropping_empty_value_is_harmless(self):
        assert _vals_equal({"a": 1, "note": None}, {"a": 1}) is True


class TestRepresentationTolerance:
    """Intended noise-suppression must keep working (no regressions)."""

    def test_tz_offset_vs_z_same_instant(self):
        assert _vals_equal(
            "2026-06-30T20:00:00-04:00", "2026-07-01T00:00:00Z"
        ) is True

    def test_universe_null_vs_export_value_is_enrichment(self):
        assert _vals_equal(None, "stamped") is True

    def test_export_null_vs_universe_value_is_loss(self):
        assert _vals_equal("real", None) is False

    def test_date_vs_datetime_same_day(self):
        assert _vals_equal("2026-06-30", "2026-06-30T14:00:00Z") is True


class TestProgrammaticIssueShape:
    """Pin the PV issue dict shape (compare_dicts → classify_issues). This is the *other* half of the
    persisted UNIVERSE_COMPATIBILITY doc the frontend renders (alongside agent_judge issues), and the
    shape the seed/UI hand-authors must match. A rename here should fail loudly."""

    # The keys env-detail-page.tsx renders for every issue (see TestFrontendContract in
    # test_verify_universe_agent_judge.py for the merged-doc contract).
    REQUIRED = {"entity", "field", "type", "phase", "critical", "detail"}

    def test_compare_dicts_emits_core_issue_keys(self):
        # one dropped field (non-empty → real loss) + one added field (enrichment)
        a = {"users": [{"id": 1, "name": "Ada", "ssn": "123"}]}
        b = {"users": [{"id": 1, "name": "Ada", "created_at": "2026-01-01"}]}
        issues = compare_dicts(a, b, "universe", "export1")
        types = {i["type"] for i in issues}
        assert {"dropped_field", "added_field"} <= types
        # compare_dicts emits entity/field/type/detail; dropped_field also carries non_empty_count
        for i in issues:
            assert {"entity", "field", "type", "detail"} <= set(i)
        dropped = next(i for i in issues if i["type"] == "dropped_field")
        assert "non_empty_count" in dropped

    def test_classify_issues_annotates_phase_and_critical(self):
        a = {"users": [{"id": 1, "ssn": "123"}]}
        b = {"users": [{"id": 1, "created_at": "2026-01-01"}]}
        load_issues = compare_dicts(a, b, "universe", "export1")
        annotated, is_compatible = classify_issues(load_issues, [])
        for i in annotated:
            assert self.REQUIRED <= set(i), f"PV issue missing required keys: {set(i)}"
            assert isinstance(i["critical"], bool)
        # a dropped non-empty field is a real loss → critical → incompatible
        assert any(i["type"] == "dropped_field" and i["critical"] for i in annotated)
        assert is_compatible is False
        # added_field is benign enrichment → non-critical, phase "export"
        added = next(i for i in annotated if i["type"] == "added_field")
        assert added["critical"] is False and added["phase"] == "export"

    def test_idempotency_issues_are_phase_idempotency_and_critical(self):
        export1 = {"users": [{"id": 1, "status": "open"}]}
        export2 = {"users": [{"id": 1, "status": "closed"}]}
        idem = compare_dicts(export1, export2, "export1", "export2")
        annotated, is_compatible = classify_issues([], idem)
        assert annotated and all(i["phase"] == "idempotency" and i["critical"] for i in annotated)
        assert is_compatible is False
