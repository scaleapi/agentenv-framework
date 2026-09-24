"""Unit tests for the pure gate roll-up backing ValidationGateAggregatorStep.

Exercises `aggregate_gate` directly (no task context / DB): required gates that pass,
fail, or never ran, and the fail-closed treatment of a missing verdict.
"""

from agent_env.task_step.task_steps.mcp_env_validator.validation_gate_aggregator import (
    DEFAULT_REQUIRED_GATES,
    aggregate_gate,
)


def _all_pass() -> dict:
    return {g: {"passed": True} for g in DEFAULT_REQUIRED_GATES}


class TestAggregateGate:
    def test_all_required_pass(self):
        summary = aggregate_gate(_all_pass(), DEFAULT_REQUIRED_GATES)
        assert summary["passed"] is True
        assert summary["failed_gates"] == []
        assert summary["missing_gates"] == []

    def test_one_gate_failed_blocks(self):
        v = _all_pass()
        v["mcp_tool_schema"] = {"passed": False, "tools_missing_description": ["foo"]}
        summary = aggregate_gate(v, DEFAULT_REQUIRED_GATES)
        assert summary["passed"] is False
        assert "mcp_tool_schema" in summary["failed_gates"]

    def test_missing_gate_is_fail_closed(self):
        v = _all_pass()
        del v["mcp_tool_correctness"]  # gate never recorded a verdict
        summary = aggregate_gate(v, DEFAULT_REQUIRED_GATES)
        assert summary["passed"] is False
        assert "mcp_tool_correctness" in summary["missing_gates"]

    def test_extra_non_required_verdicts_ignored(self):
        v = _all_pass()
        v["environment_card"] = {"validated_environment_card": {}}  # advisory, not required
        summary = aggregate_gate(v, DEFAULT_REQUIRED_GATES)
        assert summary["passed"] is True

    def test_empty_verifications_fails(self):
        summary = aggregate_gate({}, DEFAULT_REQUIRED_GATES)
        assert summary["passed"] is False
        assert set(summary["missing_gates"]) == set(DEFAULT_REQUIRED_GATES)

    def test_skipped_gate_is_advisory_and_does_not_block(self):
        # A required gate that ran but had nothing to check must not block.
        v = _all_pass()
        v["mcp_tool_correctness"] = {"passed": False, "skipped": True, "reason": "nothing to check"}
        summary = aggregate_gate(v, DEFAULT_REQUIRED_GATES)
        assert summary["passed"] is True
        assert summary["skipped_gates"] == ["mcp_tool_correctness"]
        assert summary["failed_gates"] == []

    def test_skip_does_not_mask_a_sibling_failure(self):
        v = _all_pass()
        v["mcp_tool_correctness"] = {"passed": False, "skipped": True, "reason": "nothing to check"}
        v["mcp_tool_schema"] = {"passed": False}
        summary = aggregate_gate(v, DEFAULT_REQUIRED_GATES)
        assert summary["passed"] is False
        assert summary["failed_gates"] == ["mcp_tool_schema"]

    def test_narrowed_required_gates(self):
        # Per-env `required_gates` from env metadata narrows the gate to a subset.
        v = {"mcp_tool_schema": {"passed": True}, "mcp_tool_correctness": {"passed": True}}
        summary = aggregate_gate(v, ["mcp_tool_schema"])
        assert summary["passed"] is True
        assert summary["missing_gates"] == []


class TestSpecConformanceIsAdvisory:
    """Recorded but not required: enforced at generation time; post-deploy it only
    measures staleness, which must not block a republish."""

    def test_spec_conformance_is_not_a_default_required_gate(self):
        assert "spec_conformance" not in DEFAULT_REQUIRED_GATES

    def test_failing_spec_conformance_does_not_block_by_default(self):
        v = _all_pass()
        v["spec_conformance"] = {
            "passed": False,
            "blocking_findings": 3,
            "findings": [{"kind": "missing_tool", "blocking": True}],
        }
        summary = aggregate_gate(v, DEFAULT_REQUIRED_GATES)
        assert summary["passed"] is True
        assert summary["failed_gates"] == []

    def test_env_can_opt_back_into_enforcement_via_required_gates(self):
        # {"required_gates": [..., "spec_conformance"]} in env metadata, no code change.
        v = _all_pass()
        v["spec_conformance"] = {"passed": False, "blocking_findings": 1, "findings": []}
        summary = aggregate_gate(v, [*DEFAULT_REQUIRED_GATES, "spec_conformance"])
        assert summary["passed"] is False
        assert summary["failed_gates"] == ["spec_conformance"]

    def test_opted_in_but_never_ran_is_fail_closed(self):
        summary = aggregate_gate(_all_pass(), [*DEFAULT_REQUIRED_GATES, "spec_conformance"])
        assert summary["passed"] is False
        assert summary["missing_gates"] == ["spec_conformance"]
