"""Unit tests for the pure release-gate decision behind `mcp-server put`.

`_decide_release` is the single source of the promote/block/override call; these pin
its logic without touching the store, docker, or validation task.
"""

from agent_env.cli.env.mcp_server import _decide_release

_PASS = {"passed": True, "required_gates": ["mcp_tool_schema", "mcp_tool_correctness"],
         "failed_gates": [], "missing_gates": []}
_FAIL = {"passed": False, "required_gates": ["mcp_tool_schema", "mcp_tool_correctness"],
         "failed_gates": ["mcp_tool_schema"], "missing_gates": []}


class TestDecideRelease:
    def test_all_pass_promotes(self):
        d = _decide_release(_PASS, {"passed": True}, override=False)
        assert d["promote"] is True and d["blocked"] is False and d["overridden"] is False

    def test_failed_gate_blocks(self):
        d = _decide_release(_FAIL, {"passed": True}, override=False)
        assert d["promote"] is False and d["blocked"] is True
        assert any("mcp_tool_schema" in r for r in d["reasons"])

    def test_failed_gate_override_promotes_with_audit(self):
        d = _decide_release(_FAIL, {"passed": True}, override=True)
        assert d["promote"] is True and d["blocked"] is True and d["overridden"] is True
        assert d["reasons"]  # reasons preserved for the audit stamp

    def test_missing_validation_gate_blocks(self):
        d = _decide_release(None, {"passed": True}, override=False)
        assert d["blocked"] is True
        assert any("did not run" in r for r in d["reasons"])

    def test_failed_unit_tests_block(self):
        d = _decide_release(_PASS, {"passed": False}, override=False)
        assert d["blocked"] is True
        assert any("unit tests" in r for r in d["reasons"])

    def test_unknown_unit_tests_do_not_block(self):
        # A manual put with no build report → unit-test verdict is None → not a blocker.
        d = _decide_release(_PASS, None, override=False)
        assert d["promote"] is True and d["blocked"] is False
