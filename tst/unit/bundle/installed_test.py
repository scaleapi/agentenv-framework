"""Bundles installed with a package: how an ``agent_env.bundles`` entry point resolves to its folder, which name
runs which bundle, and what ``agent-env run`` lists and runs."""

import importlib
import importlib.machinery
import importlib.util
import json
import logging
import os
import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
from click.testing import CliRunner

import agent_env
from agent_env.bundle import BundleError, BundleKind
from agent_env.bundle import installed as installed_module
from agent_env.bundle.installed import BUNDLES, CORE, checked, find_bundle, installed_bundles, run_name
from agent_env.cli import cli
from agent_env.config.runtime import Config
from agent_env.plugins import _discovery
from agent_env.plugins._report import INCOMPATIBLE_CORE, INVALID_PLUGIN, NAME_CONFLICT
from agent_env.task_step.task_step import TaskStep
from tst.unit.plugins_test import _Dist

TASK = json.dumps([{"id": "check", "type": "noop_installed_test"}])


class _Noop(TaskStep):
    type = "noop_installed_test"
    entity_refs = ()

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        return context


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Noop.type: _Noop}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A folder on sys.path that packages holding bundles are written into."""
    folder = tmp_path / "site"
    folder.mkdir()
    monkeypatch.syspath_prepend(str(folder))
    importlib.invalidate_caches()
    yield folder
    for name in [name for name in sys.modules if name.startswith("demo_bundles")]:
        del sys.modules[name]


@pytest.fixture
def quiet_logs():
    """pytest's live logging swaps its own stdout back in to print a record, so CliRunner loses whatever is
    echoed after the first one."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _package(site, dotted, bundles, init=""):
    """``dotted`` as a package in ``site`` holding a one-task bundle folder per name in ``bundles``."""
    folder = site.joinpath(*dotted.split("."))
    folder.mkdir(parents=True)
    (site / dotted.split(".")[0] / "__init__.py").write_text(init)
    for name in bundles:
        (folder / name / "tasks").mkdir(parents=True)
        (folder / name / "tasks/t.json").write_text(TASK)
        (folder / name / "README.md").write_text(f"The {name} bundle.\n")
    return folder


def _registered(monkeypatch, *eps):
    """Only ``eps`` in the bundles group, whatever else the venv has installed."""
    real = _discovery.entry_points
    monkeypatch.setattr(_discovery, "entry_points", lambda *, group: list(eps) if group == BUNDLES else real(group=group))


def _ep(name, value, dist="demo-bundles", version="1.0", requires=None):
    return EntryPoint(name, value, BUNDLES)._for(_Dist(dist, version, requires))


def test_a_real_installed_package_registers_bundles_rooted_at_its_distribution(site):
    folder = _package(site, "demo_bundles.examples", ["hello"])
    dist_info = site / "Demo_Bundles-1.2.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: Demo_Bundles\nVersion: 1.2\n")
    (dist_info / "entry_points.txt").write_text(f"[{BUNDLES}]\nhello = demo_bundles.examples\n")
    (dist_info / "RECORD").write_text("")
    importlib.invalidate_caches()

    [bundle] = [bundle for bundle in installed_bundles() if bundle.package == "Demo_Bundles"]

    assert (bundle.name, bundle.dist, bundle.version, bundle.root, bundle.problem) == (
        "hello", "demo-bundles", "1.2", folder / "hello", None)
    assert (bundle.qualified, bundle.id_root) == ("demo-bundles/hello", "@local/demo-bundles/hello")


def test_agent_env_ships_hello_rooted_at_its_own_distribution():
    hello = find_bundle("hello")

    assert (hello.dist, hello.value, hello.root, hello.problem) == (
        CORE, "agent_env.examples", Path(agent_env.__file__).parent / "examples" / "hello", None)
    assert (hello.id_root, run_name(hello, installed_bundles())) == ("@local/agentenv-framework/hello", "hello")
    assert [(entry.kind, entry.id, entry.type) for entry in checked(hello).entries] == [
        (BundleKind.ARTIFACT, "@local/agentenv-framework/hello/greeting", "file_artifact_universe"),
        (BundleKind.TASK, "@local/agentenv-framework/hello/hello", "task"),
    ]


def test_the_plugin_report_counts_the_bundle_agent_env_ships(monkeypatch, quiet_logs):
    monkeypatch.setattr(sys.modules["agent_env.cli.plugin"], "config_effect", lambda name: None)

    result = CliRunner().invoke(cli, ["plugin", "show", CORE, "--json"])
    shown = CliRunner().invoke(cli, ["plugin", "show", CORE]).output

    assert result.exit_code == 0, result.output
    (core,) = json.loads(result.stdout)["plugins"]
    assert [(c["group"], c["name"], c["status"]) for c in core["contributions"]] == [(BUNDLES, "hello", "active")]
    assert "\n  agent-env itself\n" in shown and "requires:" not in shown


def test_finding_a_bundle_imports_none_of_its_packages_code(site, monkeypatch):
    _package(site, "demo_bundles.examples", ["hello"], init="raise RuntimeError('imported')\n")
    _registered(monkeypatch, _ep("hello", "demo_bundles.examples"))

    assert find_bundle("hello").root.name == "hello"
    assert "demo_bundles" not in sys.modules


def test_a_name_with_a_hyphen_is_the_folder_of_that_name(site, monkeypatch):
    folder = _package(site, "demo_bundles", ["echo-mcp"])
    _registered(monkeypatch, _ep("echo-mcp", "demo_bundles"))

    assert find_bundle("echo-mcp").root == folder / "echo-mcp"


@pytest.mark.parametrize("value, name, problem", [
    ("demo_bundles:HELLO", "hello", "the entry point's value 'demo_bundles:HELLO' isn't a package name"),
    ("demo-bundles.x", "hello", "the entry point's value 'demo-bundles.x' isn't a package name"),
    ("demo_bundles", "missing", "the package 'demo_bundles' has no folder 'missing'"),
    ("demo_bundles", "Hello", "the package 'demo_bundles' has no folder 'Hello'"),
    ("demo_bundles", "a/hello", "the bundle name 'a/hello' contains '/'"),
    ("demo_bundles_absent", "hello", "'demo_bundles_absent' isn't an installed package"),
], ids=["attr", "not-a-name", "missing-folder", "other-case", "slash", "no-package"])
def test_an_entry_point_that_doesnt_name_a_bundle_folder_is_invalid(site, monkeypatch, value, name, problem):
    _package(site, "demo_bundles", ["hello"])
    _registered(monkeypatch, _ep(name, value))

    [bundle] = installed_bundles()

    assert (bundle.root, bundle.code) == (None, INVALID_PLUGIN)
    assert bundle.problem.startswith(problem)


def test_one_name_twice_in_a_package_and_a_package_needing_a_newer_core_are_refused(site, monkeypatch):
    _package(site, "demo_bundles", ["hello", "other"])
    _registered(monkeypatch, _ep("hello", "demo_bundles"), _ep("hello", "demo_bundles.elsewhere"),
                _ep("other", "demo_bundles", dist="demo-future", requires=["agentenv-framework>=999"]))

    assert [(bundle.name, bundle.code) for bundle in installed_bundles()] == [
        ("hello", NAME_CONFLICT), ("hello", NAME_CONFLICT), ("other", INCOMPATIBLE_CORE)]
    with pytest.raises(BundleError, match="demo-bundles registers the bundle 'hello' more than once"):
        find_bundle("hello")


def test_a_name_two_packages_share_runs_qualified_unless_one_is_agent_envs_own(site, monkeypatch):
    _package(site, "demo_bundles_a", ["hello"])
    _package(site, "demo_bundles_b", ["hello"])
    _registered(monkeypatch, _ep("hello", "demo_bundles_a", dist="demo-a"), _ep("hello", "demo_bundles_b", dist="Demo_B"))
    bundles = installed_bundles()

    with pytest.raises(BundleError, match="several packages install a bundle of that name; run one of "
                                          "'demo-a/hello', 'demo-b/hello'"):
        find_bundle("hello")
    assert find_bundle("DEMO.b/hello").package == "Demo_B"
    assert sorted(run_name(bundle, bundles) for bundle in bundles) == ["demo-a/hello", "demo-b/hello"]

    monkeypatch.setattr(installed_module, "CORE", "demo-a")

    assert find_bundle("hello").package == "demo-a"
    assert sorted(run_name(bundle, bundles) for bundle in bundles) == ["demo-b/hello", "hello"]


def test_an_unknown_name_suggests_the_close_ones(site, monkeypatch):
    _package(site, "demo_bundles", ["hello", "help"])
    _registered(monkeypatch, _ep("hello", "demo_bundles"), _ep("help", "demo_bundles"))

    with pytest.raises(BundleError, match=r"'helo' is neither a folder nor an installed bundle; did you mean "
                                          r"'hello' or 'help'\?"):
        find_bundle("helo")
    _registered(monkeypatch)
    with pytest.raises(BundleError, match="and no bundles are installed"):
        find_bundle("helo")


def test_the_cli_lists_the_installed_bundles_by_the_name_each_runs_by(site, monkeypatch):
    _package(site, "demo_bundles_a", ["hello", "empty", "broken"])
    _package(site, "demo_bundles_b", ["hello"])
    (site / "demo_bundles_a/empty/tasks/t.json").unlink()
    (site / "demo_bundles_a/broken/tasks/t.json").write_text('[{"id": "box", "type": "deploy_sandbox", "sandbox_mode": ["vm"]}]')
    _registered(monkeypatch, _ep("hello", "demo_bundles_a", dist="demo-a"), _ep("empty", "demo_bundles_a", dist="demo-a"),
                _ep("broken", "demo_bundles_a", dist="demo-a"),
                _ep("hello", "demo_bundles_b", dist="demo-b", version="2.0"), _ep("gone", "demo_bundles_b", dist="demo-b"))

    result = CliRunner().invoke(cli, ["run"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split() == ["NAME", "PACKAGE", "CONTENTS", "DESCRIPTION"]
    rows = {line.split("  ")[0]: line for line in lines[1:-2] if not line.startswith(" ")}
    folders = {lines[i - 1].split("  ")[0]: line.strip() for i, line in enumerate(lines) if line.startswith(" ")}
    assert sorted(rows) == ["broken", "demo-a/hello", "demo-b/hello", "empty", "gone"]
    assert "demo-a 1.0" in rows["demo-a/hello"] and "1 task" in rows["demo-a/hello"]
    assert rows["demo-a/hello"].endswith("The hello bundle.")
    assert "demo-b 2.0" in rows["demo-b/hello"]
    assert "invalid" in rows["empty"] and rows["empty"].endswith("this bundle has no tasks or evals to run")
    assert "invalid" in rows["broken"] and "step 'box': deploy_sandbox can't read it" in rows["broken"]
    assert "invalid" in rows["gone"] and rows["gone"].endswith("the package 'demo_bundles_b' has no folder 'gone'")
    assert folders == {"demo-a/hello": str(site / "demo_bundles_a/hello"), "demo-b/hello": str(site / "demo_bundles_b/hello"),
                       "empty": str(site / "demo_bundles_a/empty"), "broken": str(site / "demo_bundles_a/broken")}
    assert lines[-1] == "Run one with agent-env run NAME. To change one, copy its folder and run the copy."


def test_the_cli_says_so_when_no_bundles_are_installed_and_needs_a_bundle_for_its_options(monkeypatch):
    _registered(monkeypatch)

    assert CliRunner().invoke(cli, ["run"]).output == "No bundles are installed. agent-env run PATH runs a folder.\n"
    for option in (["--task", "t"], ["--keep"], ["--dry-run"]):
        usage = CliRunner().invoke(cli, ["run", *option])
        assert usage.exit_code == 2
        assert "--task, --eval, --model, --sandbox, --keep and --dry-run need a BUNDLE to run" in usage.output


def test_the_cli_runs_an_installed_bundle_by_name_with_ids_rooted_at_its_package(site, monkeypatch, quiet_logs,
                                                                                 local_stores):
    _package(site, "demo_bundles", ["hello"])
    _registered(monkeypatch, _ep("hello", "demo_bundles"))

    result = CliRunner().invoke(cli, ["run", "hello"])

    assert result.exit_code == 0, result.output
    assert "tasks/t.json v1: unscored" in result.output
    assert "instance @local/demo-bundles/hello/t-" in result.output


def test_the_cli_dry_runs_an_installed_bundle_by_name_and_writes_nothing(site, monkeypatch, quiet_logs, local_stores):
    _package(site, "demo_bundles", ["hello"])
    _registered(monkeypatch, _ep("hello", "demo_bundles"))

    result = CliRunner().invoke(cli, ["run", "hello", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "tasks/t.json: v1 (new)\n" in result.output
    assert "Would run:\n  tasks/t.json v1\n" in result.output
    assert not local_stores.local_namespace_document_store().path.exists()


def test_a_folder_wins_over_an_installed_name_and_a_folder_that_isnt_a_bundle_names_it(site, monkeypatch, tmp_path,
                                                                                       quiet_logs, local_stores):
    _package(site, "demo_bundles", ["hello"])
    _registered(monkeypatch, _ep("hello", "demo_bundles"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello").mkdir()

    result = CliRunner().invoke(cli, ["run", "hello"])

    assert result.exit_code == 1
    assert "not a bundle" in result.output
    assert "an installed bundle is also named 'hello': run it with agent-env run demo-bundles/hello" in result.output
    assert CliRunner().invoke(cli, ["run", "demo-bundles/hello"]).exit_code == 0


def test_the_cli_reports_installed_metadata_it_cant_read(monkeypatch):
    def unreadable(*, group):
        raise TypeError("Pair.__new__() missing 1 required positional argument: 'value'")

    monkeypatch.setattr(_discovery, "entry_points", unreadable)

    result = CliRunner().invoke(cli, ["run"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: the installed bundles can't be read: TypeError: Pair.__new__()")


def test_a_namespace_location_that_isnt_a_folder_is_skipped(site, monkeypatch):
    """An editable install can put a finder placeholder in a namespace package's path, before a real folder."""
    folder = _package(site, "demo_bundles_ns.examples", ["hello"])
    spec = importlib.machinery.ModuleSpec("demo_bundles_ns", None, is_package=True)
    spec.submodule_search_locations = ["__editable__.demo_finder.__path_hook__", str(site / "demo_bundles_ns")]
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec if name == "demo_bundles_ns" else None)
    _registered(monkeypatch, _ep("hello", "demo_bundles_ns.examples"))

    assert find_bundle("hello").root == folder / "hello"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads anything")
def test_an_unreadable_package_folder_fails_only_its_own_bundle(site, monkeypatch):
    _package(site, "demo_bundles_ok", ["other"])
    locked = _package(site, "demo_bundles_locked", ["hello"])
    _registered(monkeypatch, _ep("hello", "demo_bundles_locked", dist="demo-locked"), _ep("other", "demo_bundles_ok", dist="demo-ok"))
    locked.chmod(0)
    try:
        bundles = installed_bundles()
    finally:
        locked.chmod(0o755)

    assert {bundle.name: (bundle.code, (bundle.problem or "")[-17:]) for bundle in bundles} == {
        "hello": (INVALID_PLUGIN, "permission denied"), "other": (None, "")}


def test_a_linked_folder_is_found_by_its_link_name(site, monkeypatch, tmp_path):
    package = _package(site, "demo_bundles", [])
    target = _package(tmp_path / "elsewhere", "stash", ["hello-v2"])
    (package / "hello").symlink_to(target / "hello-v2", target_is_directory=True)
    _registered(monkeypatch, _ep("hello", "demo_bundles"))

    assert find_bundle("hello").root == package / "hello"


def test_two_installs_with_unreadable_metadata_are_each_unreadable_not_a_conflict(site, monkeypatch):
    _package(site, "demo_bundles", ["hello"])
    _registered(monkeypatch, _ep("hello", "demo_bundles", dist=""), _ep("hello", "demo_bundles.x", dist=""))

    assert {(bundle.code, bundle.problem) for bundle in installed_bundles()} == {
        (INVALID_PLUGIN, "its distribution's metadata can't be read")}


def test_a_home_typo_or_a_qualified_folder_prints_one_line(site, monkeypatch, tmp_path, quiet_logs, local_stores):
    _package(site, "demo_bundles", ["hello"])
    _registered(monkeypatch, _ep("hello", "demo_bundles"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "demo-bundles" / "hello").mkdir(parents=True)

    typo = CliRunner().invoke(cli, ["run", "~nosuchuser_zz/hello"])
    shadowed = CliRunner().invoke(cli, ["run", "demo-bundles/hello"])

    assert typo.exit_code == 1 and typo.output.startswith("Error: ~nosuchuser_zz/hello: ") and "Traceback" not in typo.output
    assert shadowed.exit_code == 1 and "not a bundle" in shadowed.output
    assert "an installed bundle is also named" not in shadowed.output
