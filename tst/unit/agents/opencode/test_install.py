"""install.sh: a host install's folder in the shared /tmp, which holds the model key, is this user's alone; the agent
it starts keeps OpenCode's state in that folder and runs its own CLI, with its Node and Python only where the target
has none."""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import uuid
from pathlib import Path

import pytest

from tst.unit.agents.opencode.agent import AGENT_DIR

def _install(tmp_path: Path, root_named) -> subprocess.CompletedProcess:
    """``install.sh host`` from a copy of the agent's folder, with a curl that fails so it stops at its first download.
    ``root_named`` is given the folder it will use, before it runs."""
    folder = tmp_path / "agent"
    folder.mkdir()
    for name in ("install.sh", "Dockerfile"):
        shutil.copy(AGENT_DIR / name, folder / name)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "curl").write_text("#!/bin/sh\nexit 22\n")
    (tmp_path / "bin" / "curl").chmod(0o755)
    agent, port, sandbox = f"t-{uuid.uuid4().hex[:8]}", "18999", f"sb-{uuid.uuid4().hex[:8]}"
    root_named(Path(f"/tmp/agentenv-agent-{agent}-{port}-{sandbox}"))
    return subprocess.run(["bash", str(folder / "install.sh"), "host", agent, port, str(tmp_path / "app")],
                          env={"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "HOME": str(tmp_path),
                               "AGENTENV_SANDBOX": sandbox},
                          capture_output=True, text=True, timeout=60)


_LINUX_ONLY = pytest.mark.skipif(platform.system() != "Linux", reason="install.sh installs onto a Linux host only")


@_LINUX_ONLY
def test_a_host_install_makes_its_folder_this_users_alone(tmp_path):
    roots: list[Path] = []
    try:
        result = _install(tmp_path, roots.append)
        (root,) = roots
        assert root.is_dir() and stat.S_IMODE(root.stat().st_mode) == 0o700, result.stderr
        assert root.stat().st_uid == os.getuid()
    finally:
        for root in roots:
            shutil.rmtree(root, ignore_errors=True)


@_LINUX_ONLY
def test_a_host_install_refuses_a_folder_it_did_not_make_in_its_place(tmp_path):
    planted = tmp_path / "elsewhere"
    planted.mkdir()
    roots: list[Path] = []

    def plant(root: Path) -> None:
        root.symlink_to(planted)  # as another user could, pointing where they read
        roots.append(root)

    try:
        result = _install(tmp_path, plant)
    finally:
        for root in roots:
            root.unlink()

    assert result.returncode == 1 and "isn't a folder of this user's own" in result.stderr, result.stderr
    assert not any(planted.iterdir())


def test_an_installed_agent_runs_its_own_cli_and_its_node_and_python_only_where_the_target_has_none():
    run_sh = (AGENT_DIR / "run.sh").read_text()
    install_sh = (AGENT_DIR / "install.sh").read_text()

    assert 'export PATH="$root/bin:$PATH:$root/node/bin:$root/venv/bin"' in run_sh
    assert 'put "$stage/opencode" "$root/bin/opencode"' in install_sh
    assert "node_modules/opencode-ai/bin/opencode.exe" in install_sh


def test_an_installed_agent_keeps_opencodes_state_and_its_workspace_where_the_install_says():
    install_sh = (AGENT_DIR / "install.sh").read_text()

    assert "OPENCODE_WORKSPACE_DIR=%s" in install_sh and 'HOME=%s' in install_sh
    assert "for dir in CONFIG:config CACHE:cache DATA:share STATE:state" in install_sh
    assert 'rm -rf $r/data $r/node $r/npm' in install_sh
