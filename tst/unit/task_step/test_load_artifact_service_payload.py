"""Tests for the sandbox-side service-payload expansion used by
LoadArtifactTaskStep when an EnvironmentArtifact targets an agent/container:
the stdlib-only script must honor the synthetic filesystem server's load
contract — `root/` members first (real files take precedence), then
`data.json` `files[]` entries (text or base64), with zip-slip guarded.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from agent_env.task_step.task_steps.load_artifact import _ENVIRONMENT_PAYLOAD_EXPAND_SCRIPT


def _run_expand(payload: Path, dest: Path) -> subprocess.CompletedProcess:
    # Test-only, no shell: argv is the current interpreter, a module-level
    # constant script, and two pytest tmp_path locations — nothing
    # attacker-controlled reaches the command line.
    # nosemgrep: dangerous-subprocess-use-audit
    return subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit
        [sys.executable, "-c", _ENVIRONMENT_PAYLOAD_EXPAND_SCRIPT, str(payload), str(dest)],
        capture_output=True,
        text=True,
    )


def test_data_json_text_and_base64_files(tmp_path):
    payload = tmp_path / "data.json"
    payload.write_text(json.dumps({
        "files": [
            {"path": "notes/readme.txt", "content": "hello world"},
            {
                "path": "docs/report.pdf",
                "content": base64.b64encode(b"%PDF-1.4 fake").decode("ascii"),
                "encoding": "base64",
            },
        ]
    }))
    dest = tmp_path / "out"
    result = _run_expand(payload, dest)
    assert result.returncode == 0, result.stderr
    assert (dest / "notes/readme.txt").read_text() == "hello world"
    assert (dest / "docs/report.pdf").read_bytes() == b"%PDF-1.4 fake"
    assert sorted(result.stdout.split()) == ["docs/report.pdf", "notes/readme.txt"]


def test_zip_bundle_root_takes_precedence_over_data_json(tmp_path):
    payload = tmp_path / "bundle.zip"
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("root/shared.txt", "from root/")
        zf.writestr("root/sub/dir/deep.bin", b"\x00\x01binary")
        zf.writestr("data.json", json.dumps({
            "files": [
                {"path": "shared.txt", "content": "from data.json (must lose)"},
                {"path": "json-only.txt", "content": "from data.json"},
            ]
        }))
    dest = tmp_path / "out"
    result = _run_expand(payload, dest)
    assert result.returncode == 0, result.stderr
    # root/ member wins over the data.json entry for the same path
    assert (dest / "shared.txt").read_text() == "from root/"
    assert (dest / "sub/dir/deep.bin").read_bytes() == b"\x00\x01binary"
    assert (dest / "json-only.txt").read_text() == "from data.json"
    assert sorted(result.stdout.split()) == ["json-only.txt", "shared.txt", "sub/dir/deep.bin"]


def test_zip_bundle_without_data_json(tmp_path):
    payload = tmp_path / "bundle.zip"
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("root/a.txt", "a")
        zf.writestr("ignored-not-under-root.txt", "x")
    dest = tmp_path / "out"
    result = _run_expand(payload, dest)
    assert result.returncode == 0, result.stderr
    assert (dest / "a.txt").read_text() == "a"
    assert not (dest / "ignored-not-under-root.txt").exists()


@pytest.mark.parametrize("bad_path", ["../escape.txt", "a/../../escape.txt"])
def test_zip_slip_and_dotdot_paths_rejected(tmp_path, bad_path):
    payload = tmp_path / "data.json"
    payload.write_text(json.dumps({"files": [{"path": bad_path, "content": "evil"}]}))
    dest = tmp_path / "out"
    result = _run_expand(payload, dest)
    assert result.returncode != 0
    assert "unsafe" in result.stderr
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("bad_member", ["root/../../../etc/passwd", "root/a/../../escape.txt"])
def test_zip_member_traversal_rejected(tmp_path, bad_member):
    # Zip-file vector of the same guard: a `root/`-prefixed member whose
    # post-prefix portion climbs out of the destination must abort the
    # expansion before anything escapes the destination dir.
    payload = tmp_path / "bundle.zip"
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("root/ok.txt", "fine")
        zf.writestr(bad_member, "evil")
    dest = tmp_path / "out"
    result = _run_expand(payload, dest)
    assert result.returncode != 0
    assert "unsafe" in result.stderr
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "etc").exists()


def test_empty_payload_stages_nothing(tmp_path):
    payload = tmp_path / "data.json"
    # Non-filesystem service shape (collections, no files[]) → no files staged.
    payload.write_text(json.dumps({"users": [{"id": 1}]}))
    dest = tmp_path / "out"
    result = _run_expand(payload, dest)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
