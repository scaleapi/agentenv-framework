"""The slow-tier durations refresh in ``.github/scripts/slow_test_durations.py``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "slow_test_durations.py"
SPLIT = "tst/integration/env/gateway/gateway_test.py"


def _checkout(tmp_path: Path) -> Path:
    for path in ("tst/integration/cli/cli_test.py", "tst/integration/env/task_step/task_steps_test.py", SPLIT):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("")
    table = tmp_path / "slow_durations.json"
    table.write_text(json.dumps({"_comment": "kept", "split_by_test": [SPLIT], "seconds": {"tst/integration/gone_test.py": 9}}))
    return table


def _report(tmp_path: Path, name: str, cases: list[tuple[str, str, float]]) -> Path:
    body = "".join(f'<testcase classname="{classname}" name="{test}" time="{time}"/>' for classname, test, time in cases)
    report = tmp_path / name
    report.write_text(f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>')
    return report


def _refresh(table: Path, root: Path, *reports: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, reports), "--table", str(table), "--root", str(root)],
                          capture_output=True, text=True)


def test_sums_each_files_tests_and_keeps_the_split_files_tests_apart(tmp_path):
    table = _checkout(tmp_path)
    one = _report(tmp_path, "one.xml", [("tst.integration.cli.cli_test.TestCliIntegration", "test_deploy", 30.0),
                                        ("tst.integration.cli.cli_test.TestCliIntegration", "test_put", 12.5)])
    two = _report(tmp_path, "two.xml", [("tst.integration.env.gateway.gateway_test", "test_mode[scale]", 41.0),
                                        ("tst.integration.env.gateway.gateway_test", "test_other", 7.0),
                                        ("tst.integration.env.task_step.task_steps_test", "test_load", 3.0)])
    result = _refresh(table, tmp_path, one, two)
    assert result.returncode == 0, result.stdout
    written = json.loads(table.read_text())
    assert written["seconds"] == {
        "tst/integration/cli/cli_test.py": 42.5,
        f"{SPLIT}::test_mode[scale]": 41.0,
        f"{SPLIT}::test_other": 7.0,
        "tst/integration/env/task_step/task_steps_test.py": 3.0,
    }
    assert written["split_by_test"] == [SPLIT] and written["_comment"] == "kept"


def test_a_case_whose_file_is_gone_is_left_out(tmp_path):
    table = _checkout(tmp_path)
    report = _report(tmp_path, "r.xml", [("tst.integration.cli.cli_test", "test_deploy", 5.0),
                                         ("tst.integration.renamed_test", "test_x", 99.0)])
    result = _refresh(table, tmp_path, report)
    assert result.returncode == 0
    assert "tst.integration.renamed_test" in result.stdout
    assert json.loads(table.read_text())["seconds"] == {"tst/integration/cli/cli_test.py": 5.0}


def test_reports_with_no_cases_leave_the_table_as_it_was(tmp_path):
    table = _checkout(tmp_path)
    before = table.read_text()
    result = _refresh(table, tmp_path, _report(tmp_path, "empty.xml", []))
    assert result.returncode == 1
    assert table.read_text() == before
