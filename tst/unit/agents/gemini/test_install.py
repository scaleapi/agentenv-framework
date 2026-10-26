"""install/v1: every placeholder the commands use is a declared parameter, the endpoint reaches install.sh as
environment, and a host install's folder in the shared /tmp, which holds the model key, is this user's alone."""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import uuid
from pathlib import Path

import pytest

from tst.unit.agents.gemini.agent import AGENT_DIR, a2a_server

_ENDPOINT = ["litellm_api_key", "litellm_base_url", "litellm_extra_headers", "trust_env"]


@pytest.mark.parametrize(("commands", "params"), [
    (a2a_server._INSTALL_COMMANDS, ["container", "work_dir", "agent_ctx_tar", "a2a_port", *_ENDPOINT]),
    (a2a_server._INSTALL_COMMANDS_HOST, ["work_dir", "agent_ctx_tar", "a2a_port", "agent_name", *_ENDPOINT]),
])
def test_every_placeholder_is_a_declared_parameter(commands, params):
    used = {name for command in commands for name in re.findall(r"\{(\w+)\}", command)}

    assert used == set(params)
    for name in ("litellm_api_key", "litellm_base_url", "litellm_extra_headers"):
        assert f"{name.upper()}={{{name}}}" in commands[-1]
    assert "AGENT_TRUST_ENV={trust_env}" in commands[-1]


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


def test_an_installed_agent_runs_its_own_clis_and_its_node_and_python_only_where_the_target_has_none():
    run_sh = (AGENT_DIR / "run.sh").read_text()

    assert 'export PATH="$root/bin:$PATH:$root/node/bin:$root/venv/bin"' in run_sh
