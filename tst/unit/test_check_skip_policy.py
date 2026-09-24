"""The CI skip-policy check in ``.github/scripts/check_skip_policy.py``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "check_skip_policy.py"
ALLOWED = "agentenv-capability-missing: model_endpoint_configured"


def _report(tmp_path: Path, cases: list[tuple[str, str, str | None]], skip_type: str = "pytest.skip") -> Path:
    body = "".join(
        f'<testcase classname="{classname}" name="{name}">'
        + (f'<skipped type="{skip_type}" message="{message}"/>' if message is not None else "")
        + "</testcase>"
        for classname, name, message in cases
    )
    report = tmp_path / "junit.xml"
    report.write_text(f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>')
    return report


def _check(report: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(report), *args], capture_output=True, text=True)


def test_passes_when_every_skip_is_an_allowed_capability(tmp_path):
    report = _report(tmp_path, [("tst.integration.a_test", "ran", None), ("tst.integration.b_test", "gated", ALLOWED)])
    result = _check(report, "--allow", "model_endpoint_configured")
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "message",
    [
        "temporary workaround: agentenv-capability-missing: model_endpoint_configured",
        "could not import 'anthropic'",
        "agentenv-capability-missing: remote_sandbox",
    ],
    ids=["marker-not-at-the-start", "prose-reason", "capability-not-allowed-here"],
)
def test_rejects_a_skip_that_is_not_a_declared_capability_gap(tmp_path, message):
    result = _check(_report(tmp_path, [("tst.integration.b_test", "gated", message)]), "--allow", "model_endpoint_configured")
    assert result.returncode == 1 and "tst/integration/b_test::gated" in result.stdout, result.stdout


def test_rejects_any_skip_under_a_no_skips_path(tmp_path):
    report = _report(tmp_path, [("tst.integration.store.local_x_test", "gated", ALLOWED)])
    result = _check(report, "--allow", "model_endpoint_configured", "--no-skips-under", "tst/integration/store/")
    assert result.returncode == 1 and "skips are not allowed here" in result.stdout


def test_rejects_a_report_in_which_nothing_ran(tmp_path):
    assert _check(_report(tmp_path, [])).returncode == 1


def test_ignores_an_expected_failure(tmp_path):
    report = _report(tmp_path, [("tst.integration.b_test", "known_gap", "validator chain drifts")], skip_type="pytest.xfail")
    assert _check(report).returncode == 0
