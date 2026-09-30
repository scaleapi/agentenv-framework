"""``agent-env plugin add / remove``: the installer each environment uses, the exact commands, and
what happens around them (preview, verification, rollback, the removal checks).

Nothing here installs anything: the installer runs, the fresh-interpreter snapshots and
`plugin show` are replaced through ``Tools``. The real installers are exercised end to end by
the installer tier.
"""

import json
import os
import signal
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import _installers, _plugin_changes
from agent_env.cli._installers import (
    HATCH, PDM, PIPX, POETRY, SYSTEM, UV_PROJECT, UV_TOOL, VIRTUALENV, Environment, InstallerError, ToolReceipt,
    add_plan, detect, forced, remove_plan, requirement_name, with_refusal,
)
from agent_env.cli.plugin import plugin
from agent_env.config import get_config
from agent_env.config.paths import state_root
from tst.unit.plugins_test import _fresh, _use_config  # noqa: F401  (_fresh: autouse)

PY = Path("/env/bin/python")


def _env(kind: str, tmp_path: Path, **kwargs) -> Environment:
    prefix = kwargs.pop("prefix", tmp_path / "env")
    prefix.mkdir(parents=True, exist_ok=True)
    return Environment(kind, kwargs.pop("location", prefix), prefix, PY, **kwargs)


def _receipt(prefix: Path, body: str) -> Path:
    prefix.mkdir(parents=True, exist_ok=True)
    path = prefix / "uv-receipt.toml"
    path.write_text(body)
    return path


def _pipx(prefix: Path, pip_args=(), injected=()) -> None:
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "pipx_metadata.json").write_text(json.dumps({
        "main_package": {"package": "agentenv-framework", "pip_args": list(pip_args)},
        "injected_packages": {name: {} for name in injected},
    }))


# ---------------------------------------------------------------- which installer owns the environment


def test_each_installer_is_recognised_by_what_it_leaves_behind(tmp_path):
    base = tmp_path / "base"
    tool = tmp_path / "tools" / "agentenv-framework"
    _receipt(tool, "[tool]\nrequirements = [{ name = \"agentenv-framework\" }]\n")
    pipx = tmp_path / "pipx" / "venvs" / "agentenv-framework"
    _pipx(pipx)
    projects = {}
    for name, extra, table in (("uvp", "uv.lock", ""), ("poetry", None, "[tool.poetry]\n"), ("pdm", None, "[tool.pdm]\n")):
        root = tmp_path / name
        (root / ".venv").mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname = 'x'\n" + table)
        if extra:
            (root / extra).write_text("")
        projects[name] = root
    cached = tmp_path / "Caches" / "pypoetry" / "virtualenvs" / "demo-py3.12"
    hatch = tmp_path / "hatch" / "env" / "virtual" / "demo"

    assert (detect(tool, base, PY).kind, detect(tool, base, PY).location) == (UV_TOOL, tool)
    assert detect(pipx, base, PY).kind == PIPX
    assert (detect(projects["uvp"] / ".venv", base, PY).kind, detect(projects["uvp"] / ".venv", base, PY).location) == (
        UV_PROJECT, projects["uvp"])
    assert detect(projects["poetry"] / ".venv", base, PY).kind == POETRY
    assert detect(projects["pdm"] / ".venv", base, PY).kind == PDM
    assert detect(cached, base, PY).kind == POETRY
    assert detect(hatch, base, PY).kind == HATCH
    assert detect(tmp_path / "venv", base, PY).kind == VIRTUALENV
    assert detect(base, base, PY).kind == SYSTEM


def test_a_system_managed_python_is_refused(tmp_path):
    stdlib = tmp_path / "stdlib"
    stdlib.mkdir()
    (stdlib / "EXTERNALLY-MANAGED").write_text("[externally-managed]\n")
    env = with_refusal(_env(SYSTEM, tmp_path), stdlib=stdlib, site_packages=tmp_path)

    with pytest.raises(InstallerError, match="PEP 668"):
        add_plan(env, ["agentenv-browser"])


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes to read-only directories")
def test_read_only_site_packages_are_refused(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    site.chmod(0o555)
    try:
        env = with_refusal(_env(VIRTUALENV, tmp_path), stdlib=tmp_path, site_packages=site)
        with pytest.raises(InstallerError, match="read-only"):
            remove_plan(env, ["agentenv-browser"])
    finally:
        site.chmod(0o755)


def test_installer_can_be_forced_only_where_it_applies(tmp_path):
    root = tmp_path / "proj"
    (root / ".venv").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (root / "uv.lock").write_text("")
    env = detect(root / ".venv", tmp_path, PY)

    as_pip = forced(env, "pip")

    assert (as_pip.kind, as_pip.location) == (VIRTUALENV, root / ".venv")
    with pytest.raises(InstallerError, match="uv-receipt.toml"):
        forced(env, "uv-tool")


# ---------------------------------------------------------------- the commands


def test_uv_tool_add_passes_back_the_receipts_requirements_python_and_options(tmp_path):
    prefix = tmp_path / "tool"
    local = tmp_path / "src" / "local"
    local.mkdir(parents=True)
    _receipt(prefix, f"""
[tool]
requirements = [
    {{ name = "agentenv-framework", specifier = ">=0.9" }},
    {{ name = "agentenv-platform", specifier = "==0.26.0" }},
    {{ name = "agentenv-grader", extras = ["essays"], marker = "python_version >= '3.11'" }},
    {{ name = "agentenv-local", editable = "{local}" }},
    {{ name = "agentenv-git", git = "https://github.com/o/r?rev=v1&subdirectory=pkg" }},
    {{ name = "agentenv-url", url = "https://h/repo.tar.gz", subdirectory = "sub" }},
]
python = "3.12"

[tool.options]
find-links = ["file:///wheels"]
index-url = "https://example.test/simple"
no-build = true
""")

    plan = add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["agentenv-browser==1.0"], index_url=None)

    assert plan.commands == ((
        "uv", "tool", "install", "agentenv-framework>=0.9",
        "--with", "agentenv-platform==0.26.0",
        "--with", "agentenv-grader[essays] ; python_version >= '3.11'",
        "--with-editable", str(local),
        "--with", "agentenv-git @ git+https://github.com/o/r@v1#subdirectory=pkg",
        "--with", "agentenv-url @ https://h/repo.tar.gz#subdirectory=sub",
        "--with", "agentenv-browser==1.0",
        "--python", "3.12",
        "--find-links", "file:///wheels", "--no-build", "--default-index", "https://example.test/simple",
    ),)


def test_uv_tool_add_of_a_listed_package_replaces_its_entry(tmp_path):
    prefix = tmp_path / "tool"
    _receipt(prefix, '[tool]\nrequirements = [{ name = "agentenv-framework" }, { name = "agentenv-browser" }]\n')

    (command,) = add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["agentenv_browser==2.0"]).commands

    assert command == ("uv", "tool", "install", "agentenv-framework", "--with", "agentenv_browser==2.0")


def test_uv_tool_remove_drops_only_that_requirement(tmp_path):
    prefix = tmp_path / "tool"
    _receipt(prefix, '[tool]\nrequirements = [{ name = "agentenv-framework" }, { name = "agentenv-browser" },'
                     ' { name = "agentenv-platform" }]\npython = "3.12"\n')
    env = _env(UV_TOOL, tmp_path, prefix=prefix)

    (command,) = remove_plan(env, ["agentenv_browser"]).commands

    assert command == ("uv", "tool", "install", "agentenv-framework", "--with", "agentenv-platform", "--python", "3.12")
    with pytest.raises(InstallerError, match="the tool itself"):
        remove_plan(env, ["agentenv-framework"])
    with pytest.raises(InstallerError, match="dependency of another package"):
        remove_plan(env, ["pydantic"])


@pytest.mark.parametrize("body", [
    '[tool]\nrequirements = [{ name = "agentenv-framework" }]\n[tool.options]\nconfig-settings = { a = "b" }\n',
    '[tool]\nrequirements = [{ name = "agentenv-framework", index = "private" }]\n',
])
def test_a_receipt_agent_env_cannot_reproduce_refuses_rather_than_dropping_it(tmp_path, body):
    prefix = tmp_path / "tool"
    _receipt(prefix, body)

    with pytest.raises(InstallerError, match="make this change with uv yourself"):
        add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["agentenv-browser"])


def test_pipx_injects_with_the_venvs_own_pip_arguments(tmp_path):
    prefix = tmp_path / "venvs" / "agentenv-framework"
    _pipx(prefix, pip_args=["--no-index", "--find-links", "/wheels"])
    env = _env(PIPX, tmp_path, prefix=prefix)

    assert add_plan(env, ["agentenv-browser"]).commands == ((
        "pipx", "inject", "agentenv-framework", "agentenv-browser", "--pip-args", "--no-index --find-links /wheels"),)
    assert remove_plan(env, ["agentenv-browser"]).commands == (
        ("pipx", "uninject", "--leave-deps", "agentenv-framework", "agentenv-browser"),)
    with pytest.raises(InstallerError, match="pipx uninstall"):
        remove_plan(env, ["agentenv-framework"])


def test_a_uv_project_adds_and_removes_through_uv(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["agentenv-browser"]\n\n'
                                         '[dependency-groups]\ndev = ["agentenv-grader"]\n')
    env = _env(UV_PROJECT, tmp_path, location=root)

    # Never an exact sync: that would uninstall extras and anything installed outside the lock.
    assert add_plan(env, ["agentenv-browser"], index_url="https://x.test/simple").commands == (
        ("uv", "add", "--no-sync", "--project", str(root), "agentenv-browser", "--default-index", "https://x.test/simple"),
        ("uv", "sync", "--inexact", "--project", str(root)),
    )
    assert remove_plan(env, ["agentenv-browser"]).commands == (
        ("uv", "remove", "--no-sync", "--project", str(root), "agentenv-browser"),
        ("uv", "pip", "uninstall", "--python", str(PY), "agentenv-browser"),
    )
    # A dependency group is named, so `uv remove` finds the package where it is declared.
    assert remove_plan(env, ["agentenv-grader"]).commands[0] == (
        "uv", "remove", "--no-sync", "--project", str(root), "--group", "dev", "agentenv-grader")
    with pytest.raises(InstallerError, match="does not declare agentenv-other"):
        remove_plan(env, ["agentenv-other"])


def test_a_plain_venv_uses_its_pip_else_uv_pip(tmp_path, monkeypatch):
    with_pip = _env(VIRTUALENV, tmp_path)
    without_pip = _env(VIRTUALENV, tmp_path, has_pip=False)
    monkeypatch.setattr(_installers.shutil, "which", lambda name: "/bin/uv" if name == "uv" else None)

    assert add_plan(with_pip, ["agentenv-browser"]).commands == ((str(PY), "-m", "pip", "install", "agentenv-browser"),)
    assert remove_plan(with_pip, ["agentenv-browser"]).commands == (
        (str(PY), "-m", "pip", "uninstall", "-y", "agentenv-browser"),)
    assert add_plan(without_pip, ["agentenv-browser"]).commands == (
        ("uv", "pip", "install", "--python", str(PY), "agentenv-browser"),)

    monkeypatch.setattr(_installers.shutil, "which", lambda name: None)
    with pytest.raises(InstallerError, match="neither pip nor uv"):
        add_plan(without_pip, ["agentenv-browser"])


@pytest.mark.parametrize("kind, command", [(POETRY, ("poetry", "add", "agentenv-browser")), (PDM, ("pdm", "add", "agentenv-browser"))])
def test_poetry_and_pdm_are_only_told_the_command(tmp_path, kind, command):
    plan = add_plan(_env(kind, tmp_path), ["agentenv-browser"])

    assert (plan.runs, plan.commands) == (False, (command,))


def test_hatch_is_told_to_edit_its_dependencies(tmp_path):
    plan = remove_plan(_env(HATCH, tmp_path), ["agentenv-browser"])

    assert (plan.runs, plan.commands) == (False, ())
    assert "hatch env prune" in plan.note


@pytest.mark.parametrize("spec, name", [
    ("agentenv-browser", "agentenv-browser"), ("Agentenv_Browser[x]>=1 ; python_version>'3'", "agentenv-browser"),
    ("agentenv-browser @ file:///w/a.whl", "agentenv-browser"), ("agentenv-browser@git+https://h/r@v1", "agentenv-browser"),
    ("./dist/agentenv_browser-1.0.0-py3-none-any.whl", "agentenv-browser"), ("git+https://github.com/o/r@main", None),
    ("https://h/agentenv_browser-1.0.tar.gz", "agentenv-browser"), ("/w/my%20dir/agentenv_x-2.0.zip", "agentenv-x"),
    ("./checkouts/agentenv-browser/", None),
])
def test_requirement_names(spec, name):
    assert requirement_name(spec) == name


# ---------------------------------------------------------------- add and remove, around the installer


def _dist(name, version="1.0.0", entry_points=(), requires=()):
    return {_installers.normalize(name): {"name": name, "version": version, "requires": list(requires),
                                          "entry_points": [list(ep) for ep in entry_points]}}


CORE = _dist("agentenv-framework", "0.9.1")
BROWSER = _dist("agentenv-browser", entry_points=[("agent_env.envs", "browser"), ("agent_env.task_steps", "browser_navigate")])


class _Fake(_plugin_changes.Tools):
    """Records commands, and plays back snapshots in order (the last one repeats)."""

    def __init__(self, snapshots, report=None, fail=(), config_set_by=None, installs=None, broken=None, reports=None):
        self.commands, self._snapshots, self._report, self._fail = [], list(snapshots), report, set(fail)
        self._reports = reports or {}
        super().__init__(run=self._run, snapshot=self._snapshot, inspect=self._inspect,
                         config_set_by=config_set_by or (lambda name: None),
                         installs=installs or (lambda name, path: False), broken=broken or set)

    def _run(self, command, cwd):
        self.commands.append(command)
        return 1 if command[:4] in self._fail or command[:3] in self._fail else 0

    def _snapshot(self, python):
        return self._snapshots.pop(0) if len(self._snapshots) > 1 else self._snapshots[0]

    def _inspect(self, python, dist):
        return self._reports.get(dist) or self._report or {"plugins": [{"contributions": [
            {"group": "agent_env.envs", "name": "browser", "status": "active", "reason": None}]}], "config_effect": None}


@pytest.fixture
def venv(tmp_path, monkeypatch):
    env = _env(VIRTUALENV, tmp_path)
    monkeypatch.setattr(_plugin_changes, "current_environment", lambda installer=None: env)
    monkeypatch.setattr(_plugin_changes, "_preview", lambda env, specs, index_url: True)
    return env


@pytest.fixture
def invoke(monkeypatch):
    """Run `agent-env plugin ...` with ``tools`` in place of the real installer and probes."""
    def run(args, tools, input=None):
        monkeypatch.setattr(_plugin_changes, "Tools", lambda: tools)
        return CliRunner().invoke(plugin, list(args), input=input)
    return run


def test_add_installs_verifies_and_reports(venv, invoke):
    tools = _Fake([CORE, {**CORE, **BROWSER}])

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 0, result.output
    assert tools.commands == [(str(PY), "-m", "pip", "install", "agentenv-browser")]
    assert "+ agentenv-browser 1.0.0" in result.output
    assert "agentenv-browser: envs browser active" in result.output


@pytest.mark.parametrize("core", [CORE, _dist("agentenv-framework", "0.9.2", entry_points=[("agent_env.bundles", "hello")])],
                         ids=["core unchanged", "core upgraded with a bundle"])
def test_add_of_a_package_that_is_not_a_plugin_is_rolled_back(venv, invoke, core):
    after = {**core, **_dist("six")}
    tools = _Fake([CORE, after, after, CORE])

    result = invoke(["add", "six", "--yes"], tools)

    assert result.exit_code == 1
    assert "not an agent-env plugin" in result.output
    assert (str(PY), "-m", "pip", "uninstall", "-y", "six") in tools.commands
    assert "Restored." in result.output


def test_add_whose_plugin_does_not_take_effect_is_rolled_back_unless_kept(venv, invoke):
    report = {"plugins": [{"contributions": [
        {"group": "agent_env.envs", "name": "browser", "status": "conflict", "reason": "another installed package registers this name"}]}]}
    after = {**CORE, **BROWSER}

    rolled = _Fake([CORE, after, after, CORE], report)
    kept = _Fake([CORE, after], report)
    result = invoke(["add", "agentenv-browser", "--yes"], rolled)
    kept_result = invoke(["add", "agentenv-browser", "--yes", "--keep"], kept)

    assert result.exit_code == kept_result.exit_code == 1
    assert "browser is conflict" in result.output
    assert rolled.commands[-1] == (str(PY), "-m", "pip", "uninstall", "-y", "agentenv-browser")
    assert kept.commands == [(str(PY), "-m", "pip", "install", "agentenv-browser")]
    assert "Kept as installed (--keep)." in kept_result.output


@pytest.mark.parametrize(("report", "problem"), [
    ({"format_version": 2, "plugins": []}, "reports plugins in format 2, but this agent-env reads format 1"),
    ({"format_version": 1}, "`plugin show agentenv-browser --json` reported no plugin package"),
    ({"format_version": 1, "plugins": "none"}, "`plugin show agentenv-browser --json` reported no plugin package"),
])
def test_add_rolls_back_when_it_cannot_read_the_new_report(venv, invoke, report, problem):
    after = {**CORE, **BROWSER}
    tools = _Fake([CORE, after, after, CORE], report)

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 1
    assert problem in result.output
    assert "Restored." in result.output


def test_add_reads_format_1_and_ignores_what_it_does_not_know(venv, invoke):
    report = {"format_version": 1, "added_later": {}, "plugins": [{"contributions": [
        {"group": "agent_env.envs", "name": "browser", "status": "active", "code": "added-later", "reason": "a note"}]}]}
    tools = _Fake([CORE, {**CORE, **BROWSER}], report)

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 0, result.output


def test_inspect_reports_output_that_is_not_json_as_an_error(tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\necho 'hello from a plugin'\n")
    python.chmod(0o755)

    report = _plugin_changes.inspect(python, "agentenv-browser")

    assert report["error"].startswith("`plugin show agentenv-browser --json` printed no JSON report")


def test_an_add_that_upgrades_the_core_checks_the_bundles_the_core_ships(venv, invoke):
    upgraded = {**_dist("agentenv-framework", "0.9.2", entry_points=[("agent_env.bundles", "hello")]), **BROWSER}

    def core(status):
        return {"plugins": [{"contributions": [
            {"group": "agent_env.bundles", "name": "hello", "status": status, "reason": None}]}], "config_effect": None}

    kept = _Fake([CORE, upgraded], reports={"agentenv-framework": core("active")})
    rolled = _Fake([CORE, upgraded, upgraded, CORE], reports={"agentenv-framework": core("failed")})
    kept_result = invoke(["add", "agentenv-browser", "--yes"], kept)
    rolled_result = invoke(["add", "agentenv-browser", "--yes"], rolled)

    assert kept_result.exit_code == 0, kept_result.output
    assert "agentenv-framework: bundles hello active" in kept_result.output
    assert rolled_result.exit_code == 1
    assert "agentenv-framework: hello is failed" in rolled_result.output
    assert "Restored." in rolled_result.output


def test_an_upgrade_the_plugin_caused_is_reverted_too(venv, invoke):
    upgraded = {**_dist("agentenv-framework", "0.9.2"), **BROWSER}
    report = {"plugins": [{"contributions": [{"group": "agent_env.envs", "name": "browser", "status": "failed", "reason": "x"}]}]}
    tools = _Fake([CORE, upgraded, upgraded, CORE], report)

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert "~ agentenv-framework 0.9.1 -> 0.9.2 (agent-env itself)" in result.output
    assert tools.commands[-2:] == [
        (str(PY), "-m", "pip", "uninstall", "-y", "agentenv-browser"),
        (str(PY), "-m", "pip", "install", "--no-deps", "agentenv-framework==0.9.1"),
    ]


def test_a_failed_installer_run_restores_the_environment(venv, invoke):
    tools = _Fake([CORE], fail=[(str(PY), "-m", "pip", "install")])

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 1
    assert "problem: the installer failed" in result.output


def test_dry_run_and_a_declined_prompt_change_nothing(venv, invoke):
    dry, declined = _Fake([CORE]), _Fake([CORE])

    dry_result = invoke(["add", "agentenv-browser", "--dry-run"], dry)
    declined_result = invoke(["add", "agentenv-browser"], declined, input="n\n")

    assert dry_result.exit_code == 0 and dry.commands == []
    assert "Will run:" in dry_result.output
    assert declined_result.exit_code == 1 and declined.commands == []
    assert "Nothing changed." in declined_result.output


def test_a_uv_tool_rollback_rebuilds_from_the_old_receipt_and_restores_it_byte_for_byte(tmp_path, monkeypatch, invoke):
    prefix = tmp_path / "tool"
    receipt = _receipt(prefix, '[tool]\nrequirements = [{ name = "agentenv-framework" }]\npython = "3.12"\n')
    original = receipt.read_bytes()
    env = _env(UV_TOOL, tmp_path, prefix=prefix)
    monkeypatch.setattr(_plugin_changes, "current_environment", lambda installer=None: env)
    after = {**CORE, **_dist("six")}
    tools = _Fake([CORE, after, after, CORE])
    real_run = tools._run

    def run(command, cwd):
        receipt.write_text("rewritten by uv")
        return real_run(command, cwd)
    tools.run = run

    result = invoke(["add", "six", "--yes"], tools)

    assert result.exit_code == 1
    # Pinned to the versions from before: the old requirements alone keep newer ones that fit.
    assert tools.commands[-1][:6] == ("uv", "tool", "install", "agentenv-framework", "--python", "3.12")
    assert tools.commands[-1][6] == "--constraints"
    assert receipt.read_bytes() == original


def test_a_uv_project_rollback_restores_pyproject_and_lock_and_the_packages(tmp_path, monkeypatch, invoke):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\ndependencies = []\n")
    (root / "uv.lock").write_text("lock v1\n")
    before = {p: p.read_bytes() for p in (root / "pyproject.toml", root / "uv.lock")}
    env = _env(UV_PROJECT, tmp_path, location=root)
    monkeypatch.setattr(_plugin_changes, "current_environment", lambda installer=None: env)
    six = {**CORE, **_dist("six")}
    tools = _Fake([CORE, six, six, CORE])
    real_run = tools._run

    def run(command, cwd):
        if command[:2] == ("uv", "add"):
            (root / "pyproject.toml").write_text("[project]\nname = 'x'\ndependencies = ['six']\n")
            (root / "uv.lock").write_text("lock v2\n")
        return real_run(command, cwd)
    tools.run = run

    result = invoke(["add", "six", "--yes"], tools)

    assert "+dependencies = ['six']" in result.output
    assert tools.commands[-1] == (str(PY), "-m", "pip", "uninstall", "-y", "six")
    assert {p: p.read_bytes() for p in before} == before


def test_a_print_only_installer_runs_nothing(tmp_path, monkeypatch, invoke):
    env = _env(POETRY, tmp_path)
    monkeypatch.setattr(_plugin_changes, "current_environment", lambda installer=None: env)
    tools = _Fake([CORE])

    result = invoke(["add", "agentenv-browser"], tools)

    assert result.exit_code == 0 and tools.commands == []
    assert "Run: cd" in result.output and "poetry add agentenv-browser" in result.output


def test_remove_uninstalls_a_plugin_and_confirms_it_is_gone(venv, invoke):
    tools = _Fake([{**CORE, **BROWSER}, CORE])

    result = invoke(["remove", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 0, result.output
    assert tools.commands == [(str(PY), "-m", "pip", "uninstall", "-y", "agentenv-browser")]
    assert "Removed." in result.output


@pytest.mark.parametrize("name, message", [
    ("agentenv-framework", "agent-env itself"), ("agentenv-missing", "not installed"), ("pydantic", "not a plugin"),
    ("agent-env", "agent-env itself"),
])
def test_remove_refuses_what_is_not_a_removable_plugin(venv, name, message, invoke):
    tools = _Fake([{**CORE, **BROWSER, **_dist("pydantic")}])

    result = invoke(["remove", name, "--yes"], tools)

    assert result.exit_code == 1 and message in result.output
    assert tools.commands == []


def test_remove_is_blocked_by_a_package_that_requires_it_unless_forced(venv, invoke):
    installed = {**CORE, **BROWSER, **_dist("agentenv-suite", requires=["agentenv-browser"])}
    blocked, forced_ = _Fake([installed]), _Fake([installed, {**CORE, **_dist("agentenv-suite")}])

    result = invoke(["remove", "agentenv-browser", "--yes"], blocked)
    forced_result = invoke(["remove", "agentenv-browser", "--yes", "--force"], forced_)

    assert result.exit_code == 1 and "agentenv-browser is required by agentenv-suite" in result.output
    assert blocked.commands == []
    assert forced_result.exit_code == 0 and "warning: agentenv-browser is required by agentenv-suite" in forced_result.output


def test_remove_is_blocked_while_the_config_names_what_the_plugin_provides(venv, monkeypatch, tmp_path, invoke):
    box = _dist("agentenv-box", entry_points=[("agent_env.sandbox_providers", "box"), ("agent_env.envs", "boxed")])
    box["agentenv-box"]["modules"] = ["agentenv_box", "agentenv_box.env"]
    _use_config(monkeypatch, tmp_path, """
        [sandbox]
        default = "box"
        agent_default = "box,local"

        [sandbox.providers.box]
        config = {}

        [envs]
        impls = ["agentenv_box.env:BoxedEnv"]
    """)
    tools = _Fake([{**CORE, **box}])

    result = invoke(["remove", "agentenv-box", "--yes"], tools)

    assert result.exit_code == 1
    for expected in ("[sandbox] default = 'box'", "[sandbox] agent_default = 'box,local'",
                     "[sandbox.providers.box] configures its provider", "[envs] impls names 'agentenv_box.env:BoxedEnv'"):
        assert expected in result.output


def test_remove_is_blocked_by_stored_documents_of_its_types_in_a_local_store(venv, invoke):
    store = get_config().get_document_store()
    store.insert("envs", {"id": "e1", "version": 1, "type": "browser"})
    store.insert("tasks", {"id": "t1", "version": 1, "steps": [{"id": "s", "type": "browser_navigate"}]})
    tools = _Fake([{**CORE, **BROWSER}])

    result = invoke(["remove", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 1
    assert "1 stored env(s) use its types (browser)" in result.output
    assert "1 stored task(s) use its step types (browser_navigate)" in result.output


def test_remove_is_blocked_by_stored_envs_and_instances_that_deploy_through_its_provider(venv, invoke):
    """The provider may come from another package than the env, so its own types would not show the dependency."""
    store = get_config().get_document_store()
    store.insert("envs", {"id": "e1", "version": 1, "type": "hosted_mcp", "env_provider_type": "hosted"})
    store.insert("env_instances", {"instance_id": "e1-abc", "env_id": "e1", "env_version": 1, "env_provider_type": "hosted"})
    hosted = _dist("agentenv-hosted", entry_points=[("agent_env.env_providers", "hosted")])

    result = invoke(["remove", "agentenv-hosted", "--yes"], _Fake([{**CORE, **hosted}]))

    assert result.exit_code == 1
    assert "1 stored env(s) use its environment providers (hosted)" in result.output
    assert "1 stored env instance(s) use its environment providers (hosted)" in result.output


def test_the_stored_document_check_never_creates_a_local_store(venv, tmp_path, invoke):
    tools = _Fake([{**CORE, **BROWSER}, CORE])

    result = invoke(["remove", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 0, result.output
    assert not (state_root() / "document_store").exists()


def test_a_remote_store_is_only_checked_with_check_usage(venv, monkeypatch, invoke):
    monkeypatch.setattr(type(get_config()), "trace_section", lambda self, name: type("T", (), {
        "value": {"impl": "agent_env.store.document_store:MongoDocumentStore", "config": {}}})())
    tools = _Fake([{**CORE, **BROWSER}, CORE])

    result = invoke(["remove", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 0
    assert "--check-usage does" in result.output


def test_the_remedy_for_a_conflict_names_plugin_remove():
    from agent_env.plugins._registration import conflict_message
    from agent_env.plugins._discovery import Plugin

    message = conflict_message("browser", "agent_env.envs", [Plugin("browser", "a:A", "a", "1"), Plugin("browser", "b:B", "b", "1")])

    assert message.endswith("Remove all but one with `agent-env plugin remove <package>`.")


def test_the_real_snapshot_reads_this_interpreter():
    installed = _plugin_changes.snapshot(Path(sys.executable))

    assert installed["agentenv-framework"]["name"] == "agentenv-framework"
    assert all({"name", "version", "requires", "entry_points"} <= set(d) for d in installed.values())


def test_a_receipt_round_trips_its_own_install(tmp_path):
    prefix = tmp_path / "tool"
    _receipt(prefix, '[tool]\nrequirements = [{ name = "agentenv-framework" }, { name = "agentenv-platform" }]\n')

    assert ToolReceipt.read(prefix / "uv-receipt.toml").install_args() == (
        "agentenv-framework", "--with", "agentenv-platform")


# ---------------------------------------------------------------- what the adversarial round found


def test_an_editable_tool_and_an_editable_plugin_stay_editable(tmp_path):
    prefix = tmp_path / "tool"
    core, local = tmp_path / "src" / "ae", tmp_path / "src" / "local"
    core.mkdir(parents=True)
    local.mkdir()
    _receipt(prefix, f'[tool]\nrequirements = [{{ name = "agentenv-framework", editable = "{core}" }},'
                     f' {{ name = "agentenv-local", editable = "{local}" }}]\n')

    (command,) = add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["agentenv-browser"]).commands

    assert command == ("uv", "tool", "install", "-e", str(core), "--with-editable", str(local),
                       "--with", "agentenv-browser")


@pytest.mark.parametrize("extra", [
    'constraints = [{ name = "six", specifier = "==1.16.0" }]',
    'overrides = [{ name = "pyyaml", specifier = "==6.0.1" }]',
    'entrypoints = [{ name = "plug", install-path = "/b/plug", from = "agentenv-plug" }]',
])
def test_a_receipt_with_settings_uv_install_cannot_take_back_refuses(tmp_path, extra):
    prefix = tmp_path / "tool"
    _receipt(prefix, f'[tool]\nrequirements = [{{ name = "agentenv-framework" }}]\n{extra}\n')

    with pytest.raises(InstallerError, match="make this change with uv yourself"):
        add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["agentenv-browser"])


def test_index_url_replaces_the_receipts_default_index_instead_of_repeating_it(tmp_path):
    prefix = tmp_path / "tool"
    _receipt(prefix, '[tool]\nrequirements = [{ name = "agentenv-framework" }]\n'
                     '[tool.options]\nindex-url = "https://a.test/simple"\n')

    (command,) = add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["x"], index_url="https://b.test/simple").commands

    assert command.count("--default-index") == 1
    assert command[-2:] == ("--default-index", "https://b.test/simple")


def test_a_path_spec_replaces_the_receipt_entry_of_the_same_package(tmp_path):
    prefix = tmp_path / "tool"
    _receipt(prefix, '[tool]\nrequirements = [{ name = "agentenv-framework" }, { name = "agentenv-grader",'
                     ' path = "/w/agentenv_grader-0.3.0-py3-none-any.whl" }]\n')

    (command,) = add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["/w/agentenv_grader-0.4.0-py3-none-any.whl"]).commands

    assert command == ("uv", "tool", "install", "agentenv-framework", "--with", "/w/agentenv_grader-0.4.0-py3-none-any.whl")


def test_pipx_forces_the_inject_of_a_package_it_already_has(tmp_path):
    prefix = tmp_path / "venvs" / "agentenv-framework"
    _pipx(prefix, injected=["agentenv-browser"])
    env = _env(PIPX, tmp_path, prefix=prefix)

    assert "--force" in add_plan(env, ["agentenv-browser==2.0"]).commands[0]
    assert "--force" not in add_plan(env, ["agentenv-grader"]).commands[0]
    # A bare URL's name is unknown until built, and it could build as agent-env itself.
    assert "--force" not in add_plan(env, ["git+https://github.com/mycorp/agentenv-browser@v2.0"]).commands[0]


def test_forcing_uv_project_needs_the_projects_own_venv(tmp_path):
    root = tmp_path / "proj"
    (root / "venv").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")

    with pytest.raises(InstallerError, match="own .venv"):
        forced(detect(root / "venv", tmp_path, PY), "uv-project")


def test_a_second_change_to_the_same_environment_is_refused_while_the_first_runs(venv, invoke):
    with _plugin_changes._locked(venv):
        result = invoke(["add", "agentenv-browser", "--yes"], _Fake([CORE]))

    assert result.exit_code == 1 and "another `agent-env plugin add` or `remove`" in result.output


def test_an_interrupted_add_restores_the_environment(venv, invoke):
    tools = _Fake([CORE, {**CORE, **BROWSER}, CORE])

    def interrupted(command, cwd):
        tools.commands.append(command)
        if command[3] == "install":
            raise KeyboardInterrupt
        return 0
    tools.run = interrupted

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 130
    assert "Interrupted; restoring" in result.output
    assert tools.commands[-1] == (str(PY), "-m", "pip", "uninstall", "-y", "agentenv-browser")


def test_adding_what_is_already_installed_changes_nothing_and_succeeds(venv, invoke):
    tools = _Fake([{**CORE, **BROWSER}])

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 0, result.output
    assert "Nothing changed" in result.output
    assert tools.commands == [(str(PY), "-m", "pip", "install", "agentenv-browser")]


def test_a_rollback_reinstalls_each_package_from_where_it_came_from(venv, invoke):
    local = {**_dist("agentenv-framework", "0.9.1"), **BROWSER}
    local["agentenv-framework"]["direct_url"] = {"url": "file:///w/agentenv_framework-0.9.1-py3-none-any.whl",
                                                 "archive_info": {}}
    local["agentenv-browser"] = {**BROWSER["agentenv-browser"],
                                 "direct_url": {"url": "file:///src/browser", "dir_info": {"editable": True}}}
    upgraded = {**_dist("agentenv-framework", "0.9.2"), **_dist("agentenv-browser", "2.0.0"),
                **_dist("agentenv-grader", entry_points=[("agent_env.task_steps", "grade")])}
    report = {"plugins": [{"contributions": [{"group": "agent_env.task_steps", "name": "grade", "status": "failed"}]}]}
    tools = _Fake([local, upgraded, upgraded, local], report)

    invoke(["add", "agentenv-grader", "--yes"], tools)

    assert tools.commands[-1] == (str(PY), "-m", "pip", "install", "--no-deps",
                                  "agentenv-framework @ file:///w/agentenv_framework-0.9.1-py3-none-any.whl",
                                  "-e", "/src/browser")


def test_a_name_another_package_still_provides_does_not_block_removal(venv, invoke):
    web = _dist("agentenv-web", entry_points=[("agent_env.envs", "browser")])
    get_config().get_document_store().insert("envs", {"id": "e1", "version": 1, "type": "browser"})
    tools = _Fake([{**CORE, **BROWSER, **web}, {**CORE, **BROWSER}])

    result = invoke(["remove", "agentenv-web", "--yes"], tools)

    assert result.exit_code == 0, result.output
    assert "blocked" not in result.output


def test_config_references_match_whole_module_paths_and_every_impl_pointer(venv, monkeypatch, tmp_path, invoke):
    one = _dist("acme-one", entry_points=[("agent_env.sandbox_providers", "one")])
    one["acme-one"]["modules"] = ["acme.one", "acme.one.env"]
    _use_config(monkeypatch, tmp_path, """
        [envs]
        impls = ["acme.two.env:TwoEnv"]

        [sandbox.providers.custom]
        impl = "acme.one.env:Sandbox"

        [stores.document]
        impl = "acme.one.env:Documents"
    """)
    tools = _Fake([{**CORE, **one}])

    result = invoke(["remove", "acme-one", "--yes"], tools)

    assert result.exit_code == 1
    assert "acme.two" not in result.output
    assert "[sandbox.providers.custom] impl 'acme.one.env:Sandbox' is from it" in result.output
    assert "[stores.document] impl 'acme.one.env:Documents' is from it" in result.output


def test_stored_usage_reads_an_env_path_and_turns_store_errors_into_blockers(venv, monkeypatch, tmp_path, invoke):
    bad = tmp_path / "garbage.db"
    bad.write_text("not a database")
    monkeypatch.setenv("DOCS_DB", str(bad))
    _use_config(monkeypatch, tmp_path, """
        [stores.document]
        impl = "agent_env.store.document_store:LocalSqliteDocumentStore"
        config = { path = "env:DOCS_DB" }
    """)
    installed = {**CORE, **BROWSER}

    blocked = invoke(["remove", "agentenv-browser", "--yes"], _Fake([installed]))
    forced_result = invoke(["remove", "agentenv-browser", "--yes", "--force"], _Fake([installed, CORE]))

    assert blocked.exit_code == 1 and "the document store cannot be read" in blocked.output
    assert forced_result.exit_code == 0, forced_result.output



def test_a_pipx_rollback_uninjects_what_the_add_injected(tmp_path, monkeypatch, invoke):
    prefix = tmp_path / "venvs" / "agentenv-framework"
    _pipx(prefix)
    metadata = prefix / "pipx_metadata.json"
    original = metadata.read_bytes()
    env = _env(PIPX, tmp_path, prefix=prefix, has_pip=False)
    monkeypatch.setattr(_plugin_changes, "current_environment", lambda installer=None: env)
    report = {"plugins": [{"contributions": [{"group": "agent_env.envs", "name": "browser", "status": "failed"}]}]}
    after = {**CORE, **BROWSER}
    tools = _Fake([CORE, after, after, CORE, CORE], report)
    real_run = tools._run

    def run(command, cwd):
        if command[:2] == ("pipx", "inject"):
            _pipx(prefix, injected=["agentenv-browser"])
        return real_run(command, cwd)
    tools.run = run

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 1
    assert ("pipx", "uninject", "--leave-deps", "agentenv-framework", "agentenv-browser") in tools.commands
    assert metadata.read_bytes() == original


@pytest.mark.parametrize("broken", ["snapshot", "inspect"])
def test_a_check_that_fails_after_the_installer_ran_still_rolls_back(venv, invoke, broken):
    tools = _Fake([CORE, {**CORE, **BROWSER}, {**CORE, **BROWSER}, CORE])
    if broken == "snapshot":
        snapshots = iter([CORE, TimeoutError("the probe timed out"), {**CORE, **BROWSER}, CORE])

        def snapshot(python):
            value = next(snapshots)
            if isinstance(value, Exception):
                raise value
            return value
        tools.snapshot = snapshot
    else:
        tools.inspect = lambda python, dist: json.loads("not json")

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 1
    assert "could not finish the change" in result.output
    assert tools.commands[-1] == (str(PY), "-m", "pip", "uninstall", "-y", "agentenv-browser")


@pytest.mark.parametrize("command, planner", [(["add", "agentenv-browser", "--yes"], "add_plan"),
                                              (["remove", "agentenv-browser", "--yes"], "remove_plan")])
def test_the_plan_is_built_under_the_lock(venv, monkeypatch, invoke, command, planner):
    held = []
    real = getattr(_installers, planner)

    def plan(env, *args, **kwargs):
        try:
            with _plugin_changes._locked(env):
                held.append(False)
        except InstallerError:
            held.append(True)
        return real(env, *args, **kwargs)
    monkeypatch.setattr(_installers, planner, plan)

    invoke(command, _Fake([{**CORE, **BROWSER}, CORE] if planner == "remove_plan" else [CORE, {**CORE, **BROWSER}]))

    assert held == [True]


def test_a_uv_project_remove_whose_uninstall_fails_is_put_back(tmp_path, monkeypatch, invoke):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\ndependencies = ['agentenv-browser']\n")
    (root / "uv.lock").write_text("lock v1\n")
    before = {p: p.read_bytes() for p in (root / "pyproject.toml", root / "uv.lock")}
    env = _env(UV_PROJECT, tmp_path, location=root, has_pip=False)
    monkeypatch.setattr(_plugin_changes, "current_environment", lambda installer=None: env)
    monkeypatch.setattr(_installers.shutil, "which", lambda name: f"/bin/{name}")
    installed = {**CORE, **BROWSER}
    tools = _Fake([installed, CORE, installed], fail={("uv", "pip", "uninstall")})
    real_run = tools._run

    def run(command, cwd):
        if command[:2] == ("uv", "remove"):
            (root / "pyproject.toml").write_text("[project]\nname = 'x'\ndependencies = []\n")
            (root / "uv.lock").write_text("lock v2\n")
        return real_run(command, cwd)
    tools.run = run

    result = invoke(["remove", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 1 and "the installer failed" in result.output
    assert tools.commands[-1] == ("uv", "pip", "install", "--python", str(PY), "--no-deps", "agentenv-browser==1.0.0")
    assert {p: p.read_bytes() for p in before} == before
    assert "Restored." in result.output


GIT = {"url": "https://github.com/acme/agentenv-browser", "vcs_info": {"vcs": "git", "commit_id": "aaa"}}


@pytest.mark.parametrize("a, b, same", [
    (GIT, {**GIT, "vcs_info": {"vcs": "git", "commit_id": "bbb"}}, False),
    (GIT, {**GIT, "vcs_info": {"vcs": "git", "commit_id": "aaa", "requested_revision": "main"}}, True),
    ({"url": "file:///src/b", "dir_info": {}}, {"url": "file:///src/b", "dir_info": {"editable": True}}, False),
    ({**GIT, "subdirectory": "one"}, {**GIT, "subdirectory": "two"}, False),
    ({"url": "file:///w/b.whl", "archive_info": {"hashes": {"sha256": "1"}}},
     {"url": "file:///w/b.whl", "archive_info": {"hashes": {"sha256": "2"}}}, False),
    ({"url": "file:///w/b.whl", "archive_info": {"hashes": {"sha256": "1"}}},
     {"url": "file:///w/b.whl", "archive_info": {}}, True),
])
def test_an_install_is_the_same_only_from_the_same_source(a, b, same):
    entry = BROWSER["agentenv-browser"]

    assert _plugin_changes._same({**entry, "direct_url": a}, {**entry, "direct_url": b}) is same


def test_re_adding_a_git_plugin_at_a_new_commit_is_a_change(venv, invoke):
    old, new = ({**BROWSER["agentenv-browser"], "direct_url": {**GIT, "vcs_info": {"vcs": "git", "commit_id": c}}}
                for c in ("aaa", "bbb"))
    tools = _Fake([{**CORE, "agentenv-browser": old}, {**CORE, "agentenv-browser": new}])

    result = invoke(["add", "git+https://github.com/acme/agentenv-browser@main", "--yes"], tools)

    assert result.exit_code == 0, result.output
    assert "Nothing changed" not in result.output
    assert "from a different source" in result.output


def test_the_lock_is_a_private_file_in_the_users_state_directory(venv):
    with _plugin_changes._locked(venv):
        (lock,) = (state_root() / "locks").iterdir()

    assert lock.parent.stat().st_mode & 0o777 == 0o700
    assert lock.stat().st_mode & 0o777 == 0o600


def test_a_symlink_planted_at_the_lock_path_is_refused_and_its_target_untouched(venv, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    lock = _plugin_changes._lock_path(venv)
    lock.parent.mkdir(parents=True)
    lock.symlink_to(victim)

    with pytest.raises(InstallerError, match="cannot take the lock"):
        with _plugin_changes._locked(venv):
            pass

    assert victim.read_text() == "keep me"


@pytest.mark.parametrize("spec, message", [
    ("agentenv-framework==0.9.2", "is agent-env itself"),
    ("agentenv-framework @ git+https://github.com/mycorp/agentenv-framework@main", "is agent-env itself"),
    ("./dist/agentenv_framework-0.9.2-py3-none-any.whl", "is agent-env itself"),
    ("agentenv-framework-protocol==0.1.1", "is agent-env itself"),
    ("agentenv-protocol==0.1.1", "is agent-env itself"),
    ("agent-env", "is agent-env itself"),
])
def test_add_refuses_agent_env_itself_before_running_anything(venv, invoke, spec, message):
    tools = _Fake([CORE])

    result = invoke(["add", "agentenv-browser", spec, "--yes"], tools)

    assert result.exit_code == 1 and message in result.output
    assert tools.commands == []


def test_an_unnamed_pipx_spec_that_changes_nothing_says_how_to_upgrade(tmp_path, monkeypatch, invoke):
    prefix = tmp_path / "venvs" / "agentenv-framework"
    _pipx(prefix, injected=["agentenv-browser"])
    env = _env(PIPX, tmp_path, prefix=prefix, has_pip=False)
    monkeypatch.setattr(_plugin_changes, "current_environment", lambda installer=None: env)
    tools = _Fake([{**CORE, **BROWSER}])

    result = invoke(["add", "git+https://github.com/mycorp/agentenv-browser@v2.0", "--yes"], tools)

    assert result.exit_code == 0, result.output
    assert "Nothing changed" in result.output and "'NAME @ URL'" in result.output


def test_a_bare_spec_that_builds_as_agent_env_itself_is_rolled_back(venv, invoke):
    replaced = {**_dist("agentenv-framework", "9.9.9"), **BROWSER}
    replaced["agentenv-framework"]["direct_url"] = {"url": "https://github.com/mycorp/agentenv-framework",
                                                    "vcs_info": {"vcs": "git", "commit_id": "abc"}}
    tools = _Fake([CORE, replaced, replaced, CORE])

    specs = ["git+https://github.com/mycorp/agentenv-framework@main", "agentenv-browser"]
    result = invoke(["add", *specs, "--yes"], tools)

    assert result.exit_code == 1
    assert "a spec built as agent-env itself, which `plugin add` does not replace" in result.output
    assert (str(PY), "-m", "pip", "uninstall", "-y", "agentenv-browser") in tools.commands
    assert tools.commands[-1] == (str(PY), "-m", "pip", "install", "--no-deps", "agentenv-framework==0.9.1")


def test_a_plugin_that_needs_a_newer_agent_env_from_the_index_is_added(venv, invoke):
    tools = _Fake([CORE, {**_dist("agentenv-framework", "0.9.2"), **BROWSER}])

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 0, result.output
    assert "~ agentenv-framework 0.9.1 -> 0.9.2 (agent-env itself)" in result.output


INDEX = "https://pypi.example.test/simple/"


@pytest.mark.parametrize("options", [
    # uv records a configured index and the same index passed as a flag side by side.
    f'index = [{{ url = "{INDEX}", default = true }}, {{ name = "corp", url = "{INDEX}", default = true }}]',
    f'index-url = "{INDEX}"\nindex = [{{ name = "corp", url = "{INDEX}", default = true }}]',
])
def test_a_default_index_the_receipt_records_twice_is_passed_once_by_name(tmp_path, options):
    prefix = tmp_path / "tool"
    _receipt(prefix, f'[tool]\nrequirements = [{{ name = "agentenv-framework" }}]\n\n[tool.options]\n{options}\n')

    (command,) = add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["agentenv-browser"]).commands

    assert command.count("--default-index") == 1
    assert command[-2:] == ("--default-index", f"corp={INDEX}")
    assert "--index" not in command


def test_other_indexes_are_passed_once_each_beside_the_default(tmp_path):
    prefix = tmp_path / "tool"
    _receipt(prefix, f"""[tool]
requirements = [{{ name = "agentenv-framework" }}]

[tool.options]
index = [{{ name = "corp", url = "{INDEX}", default = true }}, {{ name = "extra", url = "https://extra.test/simple/" }},
         {{ url = "https://extra.test/simple/" }}]
""")

    (command,) = add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["agentenv-browser"]).commands

    assert command[-4:] == ("--index", "extra=https://extra.test/simple/", "--default-index", f"corp={INDEX}")
    assert command.count("--index") == 1


def test_a_config_the_plugin_installs_and_selects_does_not_block_its_removal(venv, monkeypatch, tmp_path, invoke):
    box = _dist("agentenv-box", entry_points=[("agent_env.sandbox_providers", "box")])
    box["agentenv-box"]["modules"] = ["agentenv_box"]
    _use_config(monkeypatch, tmp_path, """
        [sandbox]
        default = "box"
    """)
    own = str(get_config().config_path())

    def selects(name):
        return own if name == "agentenv-box" else None

    def installed_with(name, path):
        return name == "agentenv-box"

    owned = _Fake([{**CORE, **box}, CORE], config_set_by=selects, installs=installed_with)
    # A shared file the package points at stays after it goes, so its references still count.
    shared = _Fake([{**CORE, **box}], config_set_by=selects)
    elsewhere = _Fake([{**CORE, **box}], config_set_by=lambda name: str(tmp_path / "elsewhere.toml"),
                      installs=installed_with)

    removed = invoke(["remove", "agentenv-box", "--yes"], owned)
    results = [invoke(["remove", "agentenv-box", "--yes"], tools) for tools in (shared, elsewhere)]

    assert removed.exit_code == 0, removed.output
    assert "agentenv-box installs the config file in effect" in removed.output and "blocked" not in removed.output
    for blocked in results:
        assert blocked.exit_code == 1 and "[sandbox] default = 'box'" in blocked.output


def test_a_config_probe_that_cannot_tell_leaves_the_block_in_place(venv, monkeypatch, tmp_path, invoke):
    box = _dist("agentenv-box", entry_points=[("agent_env.sandbox_providers", "box")])
    _use_config(monkeypatch, tmp_path, '[sandbox]\ndefault = "box"\n')

    def cannot_tell(name):
        raise _plugin_changes.ProbeError("importing it took more than 120s")

    result = invoke(["remove", "agentenv-box", "--yes"],
                    _Fake([{**CORE, **box}], config_set_by=cannot_tell, installs=lambda name, path: True))

    assert result.exit_code == 1 and "[sandbox] default = 'box'" in result.output


def test_installs_is_true_only_for_a_file_the_distribution_recorded(tmp_path):
    site = tmp_path / "site"
    info = site / "agentenv_rec-1.0.dist-info"
    info.mkdir(parents=True)
    (site / "agentenv_rec").mkdir()
    (site / "agentenv_rec" / "config.toml").write_text("")
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: agentenv-rec\nVersion: 1.0\n")
    (info / "RECORD").write_text("agentenv_rec/config.toml,,\nagentenv_rec-1.0.dist-info/METADATA,,\n")
    (tmp_path / "shared.toml").write_text("")
    sys.path.insert(0, str(site))
    try:
        assert _plugin_changes.installs("agentenv-rec", site / "agentenv_rec" / "config.toml")
        assert not _plugin_changes.installs("agentenv-rec", tmp_path / "shared.toml")
        assert not _plugin_changes.installs("agentenv-missing", tmp_path / "shared.toml")
    finally:
        sys.path.remove(str(site))


# ---------------------------------------------------------------- what the round against the release found


def _tool(tmp_path: Path, monkeypatch, names) -> tuple[Environment, Path]:
    """A uv tool, with ``names`` in its receipt, that agent-env is running from."""
    prefix = tmp_path / "tool"
    receipt = _receipt(prefix, "[tool]\nrequirements = [" + ", ".join(f'{{ name = "{n}" }}' for n in names) + "]\n")
    env = _env(UV_TOOL, tmp_path, prefix=prefix)
    monkeypatch.setattr(_plugin_changes, "current_environment", lambda installer=None: env)
    return env, receipt


GRADER = _dist("agentenv-grader", entry_points=[("agent_env.task_steps", "grade")])
SUITE = _dist("agentenv-suite", entry_points=[("agent_env.envs", "suite")], requires=["agentenv-grader"])
PLATFORM = _dist("agentenv-platform", entry_points=[("agent_env.envs", "platform")], requires=["platform-pin"])


def test_a_bare_name_already_in_the_receipt_keeps_its_pin(tmp_path):
    prefix = tmp_path / "tool"
    _receipt(prefix, '[tool]\nrequirements = [{ name = "agentenv-framework" },'
                     ' { name = "agentenv-platform", specifier = "==0.26.0" }]\n')

    (command,) = add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["Agentenv_Platform"]).commands

    assert command == ("uv", "tool", "install", "agentenv-framework", "--with", "agentenv-platform==0.26.0")


def test_a_plugin_whose_recorded_file_is_gone_is_named_before_anything_runs(tmp_path):
    prefix = tmp_path / "tool"
    gone = tmp_path / "Downloads" / "agentenv_local-1.0-py3-none-any.whl"
    _receipt(prefix, f'[tool]\nrequirements = [{{ name = "agentenv-framework" }},'
                     f' {{ name = "agentenv-local", path = "{gone}" }}]\n')
    env = _env(UV_TOOL, tmp_path, prefix=prefix)

    with pytest.raises(InstallerError, match="no longer exists") as refused:
        add_plan(env, ["agentenv-browser"])

    assert "`agent-env plugin remove agentenv-local`" in str(refused.value)
    assert remove_plan(env, ["agentenv-local"]).commands == (("uv", "tool", "install", "agentenv-framework"),)


def test_a_uv_project_whose_environment_is_elsewhere_is_still_the_project(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (root / "uv.lock").write_text("")
    venv = tmp_path / "opt" / "venv"
    venv.mkdir(parents=True)

    absolute = detect(venv, tmp_path, PY, project_environment=str(venv), cwd=root / "src")
    relative = detect(root / "env", tmp_path, PY, project_environment="env", cwd=tmp_path)
    outside = detect(venv, tmp_path, PY, project_environment=str(venv), cwd=tmp_path)
    unrelated = detect(tmp_path / "venv", tmp_path, PY, project_environment=str(venv), cwd=root)

    assert (absolute.kind, absolute.location) == (UV_PROJECT, root)
    assert (relative.kind, relative.location) == (UV_PROJECT, root)
    assert outside.kind == VIRTUALENV and "run this from the project's directory" in outside.refusal
    assert (unrelated.kind, unrelated.refusal) == (VIRTUALENV, None)


def _workspace(tmp_path: Path, *declaring: str) -> Path:
    """A uv workspace whose root has no [project] table, with members app and lib."""
    root = tmp_path / "ws"
    (root / ".venv").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["packages/*"]\n')
    (root / "uv.lock").write_text("")
    for name, declared in (("app", "agentenv-browser"), ("lib", "agentenv-grader")):
        member = root / "packages" / name
        member.mkdir(parents=True)
        dependencies = [declared, *(["agentenv-framework>=0.9"] if name in declaring else [])]
        (member / "pyproject.toml").write_text(f'[project]\nname = "{name}"\ndependencies = {dependencies!r}\n')
    return root


def test_a_virtual_workspace_changes_the_member_that_declares_agent_env(tmp_path):
    root = _workspace(tmp_path, "app")
    env = detect(root / ".venv", tmp_path, PY)

    assert env.member == ("app", root / "packages" / "app")
    assert add_plan(env, ["agentenv-browser"]).commands == (
        ("uv", "add", "--no-sync", "--project", str(root), "--package", "app", "agentenv-browser"),
        ("uv", "sync", "--inexact", "--project", str(root), "--package", "app"),
    )
    assert remove_plan(env, ["agentenv-browser"]).commands[0] == (
        "uv", "remove", "--no-sync", "--project", str(root), "--package", "app", "agentenv-browser")
    # A plugin another member declares is removed from that member.
    assert remove_plan(env, ["agentenv-grader"]).commands[0] == (
        "uv", "remove", "--no-sync", "--project", str(root), "--package", "lib", "agentenv-grader")
    assert _plugin_changes._record(env) == [
        root / "pyproject.toml", root / "packages" / "app" / "pyproject.toml",
        root / "packages" / "lib" / "pyproject.toml", root / "uv.lock"]


@pytest.mark.parametrize("declaring", [(), ("app", "lib")])
def test_a_virtual_workspace_that_does_not_say_which_member_is_refused(tmp_path, declaring):
    env = detect(_workspace(tmp_path, *declaring) / ".venv", tmp_path, PY)

    with pytest.raises(InstallerError, match="uv add --package"):
        add_plan(env, ["agentenv-browser"])


def test_a_hatch_environment_kept_in_the_project_is_hatchs(tmp_path):
    root, plain = tmp_path / "proj", tmp_path / "plain"
    for project, tool in ((root, "[tool.hatch.envs.default]\npath = \".venv\"\n"), (plain, "[tool.hatch.build]\n")):
        (project / ".venv").mkdir(parents=True)
        (project / "pyproject.toml").write_text(f'[project]\nname = "x"\n\n{tool}')

    hatch = detect(root / ".venv", tmp_path, PY)

    assert (hatch.kind, hatch.location) == (HATCH, root)
    assert detect(plain / ".venv", tmp_path, PY).kind == VIRTUALENV


def test_a_package_recorded_next_to_a_plugin_can_be_removed_with_it(tmp_path, monkeypatch, invoke):
    _tool(tmp_path, monkeypatch, ["agentenv-framework", "agentenv-platform", "platform-pin"])
    installed = {**CORE, **PLATFORM, **_dist("platform-pin")}
    both = _Fake([installed, CORE])

    alone = invoke(["remove", "platform-pin", "--yes"], _Fake([installed]))
    plugin_only = invoke(["remove", "agentenv-platform", "--dry-run"], _Fake([installed]))
    together = invoke(["remove", "agentenv-platform", "platform-pin", "--yes"], both)
    unrecorded = invoke(["remove", "pydantic", "--yes"], _Fake([{**installed, **_dist("pydantic")}]))

    assert alone.exit_code == 1 and "platform-pin is required by agentenv-platform" in alone.output
    assert "note: platform-pin stays installed" in plugin_only.output
    assert together.exit_code == 0, together.output
    assert both.commands == [("uv", "tool", "install", "agentenv-framework")]
    assert unrecorded.exit_code == 1 and "not a plugin" in unrecorded.output


def test_a_plugin_only_the_removed_one_needed_is_named_and_checked(tmp_path, monkeypatch, invoke):
    _tool(tmp_path, monkeypatch, ["agentenv-framework", "agentenv-suite"])
    store = get_config().get_document_store()
    store.insert("tasks", {"id": "t1", "version": 1, "steps": [{"id": "s", "type": "grade"}]})
    tools = _Fake([{**CORE, **SUITE, **GRADER}])

    result = invoke(["remove", "agentenv-suite", "--yes"], tools)

    assert result.exit_code == 1 and tools.commands == []
    assert "This also removes agentenv-grader" in result.output
    assert "1 stored task(s) use its step types (grade)" in result.output


def test_a_uv_project_uninstalls_the_plugins_its_lock_drops_with_the_one_removed(tmp_path, monkeypatch, invoke):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["agentenv-framework", "agentenv-suite"]\n')
    (root / "uv.lock").write_text("")
    env = _env(UV_PROJECT, tmp_path, location=root)
    monkeypatch.setattr(_plugin_changes, "current_environment", lambda installer=None: env)
    tools = _Fake([{**CORE, **SUITE, **GRADER}, CORE])

    result = invoke(["remove", "agentenv-suite", "--yes"], tools)

    assert result.exit_code == 0, result.output
    assert tools.commands[1] == ("uv", "pip", "uninstall", "--python", str(PY), "agentenv-suite", "agentenv-grader")


def test_an_add_that_removes_another_plugin_is_rolled_back(venv, invoke):
    tools = _Fake([{**CORE, **GRADER}, {**CORE, **BROWSER}, {**CORE, **BROWSER}, {**CORE, **GRADER}])

    result = invoke(["add", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 1
    assert "the installer removed agentenv-grader, which this change did not ask for" in result.output
    assert "Restored." in result.output


def test_a_record_changed_while_waiting_for_confirmation_is_not_overwritten(tmp_path, monkeypatch, invoke):
    _, receipt = _tool(tmp_path, monkeypatch, ["agentenv-framework"])

    def confirm(yes):
        receipt.write_text(receipt.read_text().replace("]", ', { name = "agentenv-grader" }]'))
        return True
    monkeypatch.setattr(_plugin_changes, "_proceed", confirm)
    tools = _Fake([CORE])

    result = invoke(["add", "agentenv-browser"], tools)

    assert result.exit_code == 1 and "changed while this waited to be confirmed" in result.output
    assert tools.commands == []


def _rewriting(tools: _Fake, receipt: Path) -> _Fake:
    """``tools`` whose installer rewrites the receipt and changes no package."""
    real = tools._run

    def run(command, cwd):
        receipt.write_text(receipt.read_text() + "# rewritten\n")
        return real(command, cwd)
    tools.run = run
    return tools


def test_an_add_that_only_rewrites_the_record_is_still_checked(tmp_path, monkeypatch, invoke):
    _, receipt = _tool(tmp_path, monkeypatch, ["agentenv-framework", "agentenv-browser"])
    original = receipt.read_bytes()
    installed = {**CORE, **BROWSER, **_dist("six")}

    refused = invoke(["add", "six", "--yes"], _rewriting(_Fake([installed]), receipt))
    after_refused = receipt.read_bytes()
    recorded = invoke(["add", "agentenv-browser>=1", "--yes"], _rewriting(_Fake([installed]), receipt))

    assert refused.exit_code == 1 and "not an agent-env plugin" in refused.output
    assert after_refused == original
    assert recorded.exit_code == 0 and "Recorded. The installed packages did not change." in recorded.output


def _interrupting(tools: _Fake, first=lambda: None) -> _Fake:
    """``tools`` whose first installer run is interrupted, after ``first`` has run."""
    def run(command, cwd):
        tools.commands.append(command)
        if len(tools.commands) == 1:
            first()
            raise KeyboardInterrupt
        return 0
    tools.run = run
    return tools


def test_records_an_interrupted_installer_left_are_cleared_before_restoring(tmp_path, monkeypatch, invoke):
    _tool(tmp_path, monkeypatch, ["agentenv-framework", "agentenv-browser"])
    site = tmp_path / "site-packages"
    site.mkdir()
    half = site / "lxml-6.1.3.dist-info"
    tools = _interrupting(_Fake([{**CORE, **BROWSER}], broken=lambda: {
        p for p in site.glob("*.dist-info") if not (p / "METADATA").is_file()}), first=half.mkdir)

    result = invoke(["remove", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 130
    assert not half.exists()
    assert "Restored." in result.output


def test_a_restore_that_cannot_finish_says_how_to_repair_the_environment(tmp_path, monkeypatch, invoke):
    _tool(tmp_path, monkeypatch, ["agentenv-framework", "agentenv-browser"])
    tools = _interrupting(_Fake([{**CORE, **BROWSER}, CORE]))

    result = invoke(["remove", "agentenv-browser", "--yes"], tools)

    assert result.exit_code == 130
    assert "Could not restore exactly; these differ from before: agentenv-browser" in result.output
    assert "`uv tool upgrade --reinstall tool` reinstalls what the installer records." in result.output


def test_the_preview_marks_agent_env_by_package_name_not_by_path():
    assert _plugin_changes._names_in("+ agentenv-framework==0.9.2") == {"agentenv-framework"}
    assert _plugin_changes._names_in(
        "+ agentenv-grader @ file:///home/me/agent-env/dist/agentenv_grader-1-py3-none-any.whl") == {"agentenv-grader"}
    assert _plugin_changes._names_in("Would install agentenv-framework-0.9.2 six-1.16.0") == {"agentenv-framework", "six"}


def _extras_tool(tmp_path: Path, monkeypatch, other_extras: str) -> None:
    prefix = tmp_path / "tool"
    _receipt(prefix, '[tool]\nrequirements = [{ name = "agentenv-framework" }, '
                     f'{{ name = "agentenv-other"{other_extras} }}, {{ name = "agentenv-suite" }}]\n')
    env = _env(UV_TOOL, tmp_path, prefix=prefix)
    monkeypatch.setattr(_plugin_changes, "current_environment", lambda installer=None: env)


def test_a_plugin_another_packages_requested_extra_needs_is_not_taken_along(tmp_path, monkeypatch, invoke):
    _extras_tool(tmp_path, monkeypatch, ', extras = ["x"]')
    other = _dist("agentenv-other")
    other["agentenv-other"]["extras"] = {"x": {"agentenv-grader": []}}
    tools = _Fake([{**CORE, **other, **SUITE, **GRADER}, {**CORE, **other, **GRADER}])

    result = invoke(["remove", "agentenv-suite", "--yes"], tools)

    assert result.exit_code == 0, result.output
    assert "also removes" not in result.output


def test_a_plugin_only_the_removed_ones_requested_extra_needs_is_taken_along(tmp_path, monkeypatch, invoke):
    _extras_tool(tmp_path, monkeypatch, "")
    prefix = tmp_path / "tool"
    receipt = prefix / "uv-receipt.toml"
    receipt.write_text(receipt.read_text().replace('{ name = "agentenv-suite" }',
                                                   '{ name = "agentenv-suite", extras = ["grading"] }'))
    suite = _dist("agentenv-suite", entry_points=[("agent_env.envs", "suite")])
    suite["agentenv-suite"]["extras"] = {"grading": {"agentenv-grader": []}}

    result = invoke(["remove", "agentenv-suite", "--dry-run"], _Fake([{**CORE, **_dist("agentenv-other"), **suite,
                                                                       **GRADER}]))

    assert result.exit_code == 0, result.output
    assert "This also removes agentenv-grader" in result.output


def test_the_installer_starts_with_ctrl_c_ignored_and_agent_env_keeps_its_own_handler():
    probe = "import signal, sys; sys.exit(0 if signal.getsignal(signal.SIGINT) == signal.SIG_IGN else 3)"
    handler = signal.getsignal(signal.SIGINT)

    assert _plugin_changes.run((sys.executable, "-c", probe)) == 0
    assert signal.getsignal(signal.SIGINT) is handler


def test_the_extras_an_extra_asks_of_its_requirements_are_followed_too(tmp_path, monkeypatch, invoke):
    # other[x] needs mid[c], and mid's extra c needs the grader, so the grader stays.
    _extras_tool(tmp_path, monkeypatch, ', extras = ["x"]')
    other, mid = _dist("agentenv-other"), _dist("agentenv-mid")
    other["agentenv-other"]["extras"] = {"x": {"agentenv-mid": ["c"]}}
    mid["agentenv-mid"]["extras"] = {"c": {"agentenv-grader": []}}

    result = invoke(["remove", "agentenv-suite", "--dry-run"], _Fake([{**CORE, **other, **mid, **SUITE, **GRADER}]))

    assert result.exit_code == 0, result.output
    assert "also removes" not in result.output


def test_a_project_environment_beside_the_project_is_found_from_inside_it(tmp_path):
    app = tmp_path / "repo" / "app"
    app.mkdir(parents=True)
    (app / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    (app / "uv.lock").write_text("")
    shared = tmp_path / "repo" / "shared-venv"
    shared.mkdir()

    found = detect(shared, tmp_path, PY, project_environment="../shared-venv", cwd=app)

    assert (found.kind, found.location) == (UV_PROJECT, app)


def test_a_hatch_environment_nested_in_the_project_is_hatchs(tmp_path):
    root = tmp_path / "proj"
    (root / ".venvs" / "default").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "x"\n\n[tool.hatch.envs.default]\npath = ".venvs/default"\n')

    hatch = detect(root / ".venvs" / "default", tmp_path, PY)

    assert (hatch.kind, hatch.location) == (HATCH, root)


def test_a_tool_whose_own_source_is_gone_is_told_to_reinstall_it(tmp_path):
    prefix = tmp_path / "tool"
    _receipt(prefix, f'[tool]\nrequirements = [{{ name = "agentenv-framework", editable = "{tmp_path / "gone"}" }}]\n')

    with pytest.raises(InstallerError, match="no longer exists.*reinstall the tool with uv"):
        add_plan(_env(UV_TOOL, tmp_path, prefix=prefix), ["agentenv-browser"])


def test_a_project_environment_run_from_inside_another_project_is_refused(tmp_path):
    owner, other = tmp_path / "owner", tmp_path / "other"
    for project in (owner, other):
        project.mkdir()
        (project / "pyproject.toml").write_text(f"[project]\nname = '{project.name}'\n")
        (project / "uv.lock").write_text("")
    (owner / "env").mkdir()

    refused = detect(owner / "env", tmp_path, PY, project_environment="env", cwd=other)

    assert refused.kind == VIRTUALENV and f"the uv project at {owner}" in refused.refusal
