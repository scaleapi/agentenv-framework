import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tst.benchmarks import version_lookup


def test_missing_pinned_baseline_has_fetch_instructions(monkeypatch):
    def missing_revision(command, **kwargs):
        assert command == [
            "git",
            "show",
            f"{version_lookup.BASELINE_REVISION}:src/agent_env/store/document_store/document_store.py",
        ]
        raise subprocess.CalledProcessError(128, command, stderr="fatal: bad object")

    monkeypatch.setattr(version_lookup.subprocess, "check_output", missing_revision)
    with pytest.raises(SystemExit, match="is unavailable in this checkout") as error:
        version_lookup._baseline_versioned_store()
    assert f"git fetch origin {version_lookup.BASELINE_REVISION}" in str(error.value)


def test_shallow_clone_reports_how_to_fetch_the_pinned_baseline(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, text=True).strip()
    clone = tmp_path / "shallow"
    command = ["git", "clone", "--depth=1"]
    if branch != "HEAD":
        command.extend(["--branch", branch])
    command.extend([f"file://{repo}", str(clone)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    missing = subprocess.run(
        ["git", "cat-file", "-e", f"{version_lookup.BASELINE_REVISION}^{{commit}}"],
        cwd=clone,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0, "the shallow clone unexpectedly contains the pinned baseline"

    pythonpath = os.pathsep.join((str(repo), str(repo / "src"), str(repo / "packages/agentenv-protocol/src")))
    result = subprocess.run(
        [sys.executable, str(repo / "tst/benchmarks/version_lookup.py")],
        cwd=clone,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert f"git fetch origin {version_lookup.BASELINE_REVISION}" in result.stderr
    assert "Traceback" not in result.stderr


def test_baseline_loader_uses_the_pinned_source(monkeypatch):
    source = """class VersionedEntityStore:
    def get(self, id, version=None):
        return (id, version)
    def next_version(self, id):
        return 3
    def put(self, entity, max_retries=5):
        return 4
"""
    commands = []

    def baseline_source(command, **kwargs):
        commands.append(command)
        return source

    monkeypatch.setattr(version_lookup.subprocess, "check_output", baseline_source)
    baseline, source_hash = version_lookup._baseline_versioned_store()

    assert commands == [[
        "git",
        "show",
        f"{version_lookup.BASELINE_REVISION}:src/agent_env/store/document_store/document_store.py",
    ]]
    assert source_hash == hashlib.sha256(source.encode()).hexdigest()
    instance = baseline.__new__(baseline)
    assert instance.get("sample") == ("sample", None)
    assert instance.next_version("sample") == 3
    assert instance.put({"id": "sample"}) == 4
