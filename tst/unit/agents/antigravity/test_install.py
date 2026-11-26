"""install.sh: a host install's folder in the shared /tmp, which holds the model key, is this user's alone; the agent it
starts keeps the SDK's conversations and skills in that folder, runs from its own Python, and trusts certifi's CAs only
on a target with none of its own."""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import uuid
from pathlib import Path

import pytest

from tst.unit.agents.antigravity.agent import AGENT_DIR

INSTALL_SH = (AGENT_DIR / "install.sh").read_text()


def _install(tmp_path: Path, root_named) -> subprocess.CompletedProcess:
    """``install.sh host`` from a copy of the agent's folder, with a curl that fails so it stops at its first download.
    ``root_named`` is given the folder it will use, before it runs."""
    folder = tmp_path / "agent"
    folder.mkdir()
    shutil.copy(AGENT_DIR / "install.sh", folder / "install.sh")
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


def test_an_installed_agent_runs_from_its_own_python_with_its_state_in_its_folder():
    run_sh = (AGENT_DIR / "run.sh").read_text()

    assert 'export PATH="$root/bin:$PATH:$root/venv/bin" PYTHONPATH="$root/app"' in run_sh
    assert 'exec "$root/venv/bin/uvicorn" a2a_server:app' in run_sh
    assert "ANTIGRAVITY_WORKSPACE_DIR=%s\\nANTIGRAVITY_HOME=%s\\nHOME=%s" in INSTALL_SH
    assert '"$(quoted "$root/data/.antigravity")" "$(quoted "$root/data")"' in INSTALL_SH
    assert 'on "rm -rf $r/data && mkdir -p $r/bin $r/app $r/data' in INSTALL_SH


CERTIFI = "SSL_CERT_FILE='/opt/agent/venv/lib/python3.12/site-packages/certifi/cacert.pem'\n"


@pytest.mark.parametrize("bundle, content, trust, written", [
    (None, "", "", CERTIFI),
    ("etc/ssl/certs/ca-certificates.crt", "PEM", "", ""),
    ("etc/pki/tls/certs/ca-bundle.crt", "PEM", "", ""),
    ("etc/ssl/cert.pem", "PEM", "", ""),
    ("etc/ssl/certs/ca-certificates.crt", "", "", CERTIFI),
    (None, "", "SSL_CERT_FILE='/trust/ca-bundle.pem'", ""),
], ids=["no-bundle", "debian-bundle", "rhel-bundle", "alpine-bundle", "an-empty-bundle", "trust-lines-name-one"])
def test_an_installed_agent_trusts_certifis_cas_only_on_a_target_with_none_and_no_bundle_named(
        tmp_path, bundle, content, trust, written):
    """The relay and the harness read the system's CA certificates: the agent's environment names certifi's only where
    the target has no bundle and the trust lines name none."""
    if bundle:
        (tmp_path / bundle).parent.mkdir(parents=True)
        (tmp_path / bundle).write_text(content)

    assert _ca_line(tmp_path, trust) == written


def _ca_line(target: Path, trust: str) -> str:
    """What install.sh writes to the agent's environment for its CA bundle, its decision run against a target whose
    ``/etc`` is ``target``'s."""
    lines = INSTALL_SH.splitlines()
    quoted = next(line for line in lines if line.startswith("quoted() {"))
    decision = INSTALL_SH[INSTALL_SH.index("certifi=$root/venv"):]
    decision = decision[:decision.index("needs_cas= ;; esac") + len("needs_cas= ;; esac")]
    written = next(line.strip() for line in lines if '"$needs_cas" ] || printf' in line)
    on = f'on() {{ sh -c "$(printf %s "$1" | sed "s#/etc/#{target}/etc/#g")"; }}'
    script = "\n".join([quoted, on, "root=/opt/agent PYTHON_VERSION=3.12", decision, written])
    return subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin", "AGENT_TRUST_ENV": trust},
                          capture_output=True, text=True, check=True).stdout


def test_uv_is_pinned_by_version_and_checksum():
    assert "UV_VERSION=0.13.1" in INSTALL_SH
    assert 'checked "$stage/$uv_archive" "$uv_sha256"' in INSTALL_SH
