"""Unit tests for the spec-conformance section of `mcp-server validate` output.

The display is keyed off the presence of findings, not off `passed`. That distinction
matters because the spec is a floor: a fully conformant server still reports its
framework-registered tools (`<service>_health`, the async-job pair) as informational
`extra_*` findings, so keying off `passed` reported a clean match and hid real drift.
"""

import pytest

from agent_env.cli.env.mcp_server import _echo_spec_conformance


@pytest.fixture
def render(capsys):
    """Return what `_echo_spec_conformance` prints for a stored verdict."""
    def _render(spec) -> str:
        _echo_spec_conformance(spec)
        return capsys.readouterr().out
    return _render


def _finding(kind: str, tool: str, blocking: bool, param: str | None = None) -> dict:
    f = {"kind": kind, "tool": tool, "blocking": blocking, "detail": f"{kind} on {tool}"}
    if param:
        f["param"] = param
    return f


class TestNoVerdict:
    def test_absent_and_skipped_verdicts_short_circuit(self, render):
        """Both early returns, together — neither reaches the findings display."""
        assert "(did not run)" in render(None)
        out = render({"skipped": True, "reason": "no spec served", "findings": []})
        assert "SKIPPED" in out and "no spec served" in out


class TestCleanServer:
    def test_no_findings_reports_a_plain_match(self, render):
        out = render({"passed": True, "spec_operations": 4, "findings": [], "blocking_findings": 0})
        assert "PASSED: 4 spec operation(s) match the live tool surface" in out
        assert "informational" not in out


class TestInformationalDriftIsVisible:
    """The regression flagged in review: `passed=True` with informational findings used to
    print a clean match and list nothing, so operators could not see the drift."""

    def test_passes_but_still_counts_and_lists_the_drift(self, render):
        out = render({
            "passed": True,
            "spec_operations": 1,
            "blocking_findings": 0,
            "findings": [
                _finding("extra_tool", "city_health", blocking=False),
                _finding("extra_tool", "city_get_job_status", blocking=False),
            ],
        })
        # Nothing blocks, so the verdict genuinely passed — don't cry wolf.
        assert "PASSED" in out
        assert "2 informational finding(s)" in out
        # ...but the findings are still listed, which is what keying off `passed` hid.
        assert "city_health" in out and "city_get_job_status" in out


class TestBlockingFindings:
    def test_counts_blocking_and_informational_separately(self, render):
        out = render({
            "passed": False, "spec_operations": 2, "blocking_findings": 1,
            "findings": [
                _finding("extra_tool", "svc_health", blocking=False),
                _finding("missing_tool", "reset", blocking=True),
            ],
        })
        assert "1 blocking finding(s), 1 informational" in out
        assert "PASSED" not in out

    def test_blocking_findings_are_listed_before_informational(self, render):
        # The list truncates, so signal must survive and noise must be what gets dropped.
        out = render({
            "passed": False, "spec_operations": 1, "blocking_findings": 1,
            "findings": [
                _finding("extra_param", "svc_get_x", blocking=False, param="b"),
                _finding("missing_param", "svc_get_x", blocking=True, param="a"),
            ],
        })
        assert out.index("missing_param") < out.index("extra_param")


class TestLegacyVerdicts:
    def test_verdict_without_blocking_findings_key_still_renders(self, render):
        # Records written before the blocking/informational split carry neither
        # `blocking_findings` nor per-finding `blocking`; treat them all as blocking.
        out = render({
            "passed": False, "spec_operations": 1,
            "findings": [{"kind": "missing_tool", "tool": "reset", "detail": "gone"}],
        })
        assert "1 blocking finding(s), 0 informational" in out
        assert "reset" in out
