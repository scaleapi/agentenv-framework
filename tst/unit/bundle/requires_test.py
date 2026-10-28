"""A bundle's bundle.toml: what it requires is checked before the bundle resolves, a command refused for one this machine
doesn't meet names the command that installs it, and a bundle without one is given one listing the plugins it uses."""

import errno
import json
import os
import sys
import threading
import tomllib
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from packaging.requirements import Requirement

from agent_env.bundle import BundleError, dry_run_bundle, parse_bundle, prepare_bundle, run_bundle
from agent_env.bundle import register as register_module
from agent_env.bundle import requires as requires_module
from agent_env.bundle.register import register_bundle
from agent_env.bundle.requires import (
    Provider, Providers, UnmetRequirements, _create, declare, install_command, unmet, unmet_problems,
)
from agent_env.bundle.resolve import resolve_bundle
from agent_env.cli import _install_prompt, cli
from agent_env.cli._installers import InstallerError
from agent_env.config.runtime import Config
from agent_env.plugins import TASK_STEPS, Contribution, Distribution, Inventory
from agent_env.store.routing import namespace_routing
from agent_env.task_step.task_step import TaskStep
from tst.unit.bundle._support import layout

ACME = "agentenv-acme"
WROTE = ('wrote bundle.toml: requires = ["agentenv-acme>=1.2.0"]; commit it so whoever you share the bundle with knows '
         "what to install")
MISSING = ("bundle.toml: requires 'agentenv-missing>=1', which isn't installed; install it with: agent-env plugin add "
           "'agentenv-missing>=1'")


class _Acme(TaskStep):
    """A step type the plugin agentenv-acme ships."""

    type = "acme_requires_test"
    entity_refs = ()

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        return context


class _AcmeMetadata:
    """agentenv-acme's installed requirements: a base one, an extra each, and an extra naming another of its own."""

    requires = ["packaging", "playwright>=1.40; extra == 'browser'", "torch; extra == 'gpu'",
                "agentenv-acme[gpu]; extra == 'all'"]


class _Configured(_Acme):
    """A step type only this machine's configuration registers."""

    type = "configured_requires_test"


class _Replaced(_Acme):
    """A step type a config file registers over the one agentenv-acme ships under the same name."""

    type = "replaced_requires_test"


def _no_links(src, dst):
    raise PermissionError("links aren't supported here")


def _value(cls):
    return f"{cls.__module__}:{cls.__qualname__}"


@pytest.fixture(autouse=True)
def acme(monkeypatch):
    """agentenv-acme 1.2.0 installed, its step type registered as its entry point registers it, beside a step type only
    configuration registers and one configuration registers over agentenv-acme's."""
    steps = {**Config().task_step_registry(), **{cls.type: cls for cls in (_Acme, _Configured, _Replaced)}}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    contributions = (Contribution(TASK_STEPS, _Acme.type, _value(_Acme), "active"),
                     Contribution(TASK_STEPS, _Replaced.type, _value(_Acme), "replaced"))
    monkeypatch.setattr(requires_module, "inventory", lambda load: Inventory(
        distributions=(Distribution(ACME, "1.2.0", contributions),), group_errors={}, discovery_errors={},
        config_path=None, config_error=None, loaded=load))
    monkeypatch.setattr(requires_module, "load_failures", dict)
    monkeypatch.setattr(requires_module, "installed", {ACME: "1.2.0"}.get)
    monkeypatch.setattr(requires_module, "packages_distributions", dict)


def _bundle(root, *types, toml=None):
    files = {"tasks/t.json": json.dumps([{"id": f"s{i}", "type": step_type} for i, step_type in enumerate(types)])}
    if toml is not None:
        files["bundle.toml"] = toml
    return layout(root, files)


def _declared(root, *, own=True, dry_run=False):
    return declare(resolve_bundle(parse_bundle(root)), own=own, dry_run=dry_run)


class TestProviders:
    def test_what_core_ships_has_no_provider(self):
        assert Providers().of(TASK_STEPS, "run_code", None) is None

    def test_a_plugins_type_is_provided_by_its_distribution_at_its_version(self):
        assert Providers().of(TASK_STEPS, _Acme.type, _Acme) == Provider(ACME, "1.2.0")
        assert Provider(ACME, "1.2.0").requirement() == "agentenv-acme>=1.2.0"

    @pytest.mark.parametrize("version, requirement", [
        ("1.2.0+acme.1", "agentenv-acme>=1.2.0"),
        ("1!2.0.post1", "agentenv-acme>=1!2.0.post1"),
        ("not a version", "agentenv-acme"),
    ], ids=["local-label", "epoch-post", "legacy"])
    def test_a_requirement_is_one_pip_takes_whatever_the_installed_version(self, version, requirement):
        assert Provider(ACME, version).requirement() == requirement
        assert str(Requirement(requirement)) == requirement

    @pytest.mark.parametrize("cls", [_Configured, _Replaced])
    def test_a_type_configuration_registers_is_provided_by_configuration(self, cls):
        assert Providers().of(TASK_STEPS, cls.type, cls) == Provider(None, None)
        assert Provider(None, None).requirement() is None

    def test_configuration_names_the_installed_package_its_class_comes_from(self, monkeypatch):
        monkeypatch.setattr(requires_module, "packages_distributions", lambda: {"tst": ["agentenv-shipper"]})
        monkeypatch.setattr(requires_module, "installed", {ACME: "1.2.0", "agentenv-shipper": "2.0"}.get)

        assert Providers().of(TASK_STEPS, _Configured.type, _Configured) == Provider(None, None, "agentenv-shipper 2.0")

    def test_of_a_namespace_packages_distributions_the_one_holding_the_module_ships_it(self, monkeypatch):
        files = {"shipper-a": ["tst/other.py"], "shipper-b": [f"{__name__.replace('.', '/')}.py"]}
        monkeypatch.setattr(requires_module, "packages_distributions", lambda: {"tst": ["shipper-a", "shipper-b"]})
        monkeypatch.setattr(requires_module, "distribution",
                            lambda name: SimpleNamespace(files=[PurePosixPath(f) for f in files[name]]))
        monkeypatch.setattr(requires_module, "installed", {ACME: "1.2.0", "shipper-b": "3.1"}.get)

        assert Providers().of(TASK_STEPS, _Configured.type, _Configured).shipped_by == "shipper-b 3.1"

    def test_a_plugin_that_failed_to_load_provides_nothing(self, monkeypatch):
        monkeypatch.setattr(requires_module, "load_failures", lambda: {TASK_STEPS: {_Acme.type: "failed to load"}})
        assert Providers().of(TASK_STEPS, _Acme.type, _Acme) == Provider(None, None)


class TestUnmet:
    @pytest.mark.parametrize("line", ["agentenv-acme", "agentenv-acme>=1.0,<2", "AgentEnv_Acme>=1",
                                      "agentenv-missing; sys_platform == 'no-such-platform'"])
    def test_a_requirement_met_here_or_that_doesnt_apply_is_met(self, line):
        assert unmet([line]) == []

    def test_a_package_not_installed_or_at_a_version_it_excludes_is_unmet(self):
        assert unmet(["agentenv-missing", "agentenv-acme>=2"]) == [
            ("agentenv-missing", "which isn't installed"), ("agentenv-acme>=2", "and agentenv-acme 1.2.0 is installed")]

    def test_a_version_pip_cant_compare_meets_only_a_requirement_naming_no_versions(self, monkeypatch):
        monkeypatch.setattr(requires_module, "installed", {ACME: "not a version"}.get)

        assert unmet(["agentenv-acme", "agentenv-acme>=1"]) == [
            ("agentenv-acme>=1", "and agentenv-acme not a version is installed")]

    @pytest.mark.parametrize("line, missing", [
        ("agentenv-acme[browser]>=1", "playwright>=1.40"),
        ("agentenv-acme[all]", "torch"),
        ("agentenv-acme[browser,all]", "torch, playwright>=1.40"),
    ])
    def test_an_extra_is_unmet_without_what_it_needs(self, monkeypatch, line, missing):
        monkeypatch.setattr(requires_module, "distribution", lambda name: _AcmeMetadata)

        assert unmet([line]) == [(line, f"and agentenv-acme 1.2.0 is installed without {missing}, which its extras "
                                        f"need")]

    @pytest.mark.parametrize("line", ["agentenv-acme[browser]", "agentenv-acme[gpu]", "agentenv-acme[undeclared]"])
    def test_an_extra_whose_needs_are_installed_or_that_needs_nothing_is_met(self, monkeypatch, line):
        monkeypatch.setattr(requires_module, "distribution", lambda name: _AcmeMetadata)
        monkeypatch.setattr(requires_module, "installed",
                            {ACME: "1.2.0", "playwright": "1.41", "torch": "2.5"}.get)

        assert unmet([line]) == []

    def test_one_unmet_requirement_names_the_command_that_installs_it(self):
        assert unmet_problems(["agentenv-missing>=1", "agentenv-acme"]) == ([MISSING], ["agentenv-missing>=1"])

    def test_agent_env_itself_is_upgraded_not_added(self, monkeypatch):
        monkeypatch.setattr(requires_module, "installed", {ACME: "1.2.0", "agentenv-framework": "0.9.1375"}.get)

        assert unmet_problems(["agentenv-framework>=0.9.1400", "agentenv-missing>=1"]) == ([
            "bundle.toml: requires 'agentenv-framework>=0.9.1400', and agentenv-framework 0.9.1375 is installed; "
            "upgrade agent-env with the installer that installed it",
            MISSING,
        ], ["agentenv-missing>=1"])

    def test_a_bundle_needing_a_newer_agent_env_is_refused_without_an_install_to_offer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(requires_module, "installed", {ACME: "1.2.0", "agentenv-framework": "0.9.1375"}.get)
        root = _bundle(tmp_path / "b", _Acme.type, toml='requires = ["agentenv-framework>=0.9.1400"]\n')

        with pytest.raises(BundleError) as caught:
            resolve_bundle(parse_bundle(root))

        assert not isinstance(caught.value, UnmetRequirements)
        assert caught.value.problems[0].endswith("upgrade agent-env with the installer that installed it")

    def test_several_name_one_command_that_installs_them_all(self):
        assert unmet_problems(["agentenv-missing>=1", "agentenv-acme>=2"]) == ([
            "bundle.toml: requires 'agentenv-missing>=1', which isn't installed",
            "bundle.toml: requires 'agentenv-acme>=2', and agentenv-acme 1.2.0 is installed",
            "bundle.toml: install what it requires with: agent-env plugin add 'agentenv-missing>=1' 'agentenv-acme>=2'",
        ], ["agentenv-missing>=1", "agentenv-acme>=2"])

    def test_the_command_quotes_what_a_shell_would_split(self):
        assert install_command(["a; sys_platform == 'linux'"]) == \
            "agent-env plugin add 'a; sys_platform == '\"'\"'linux'\"'\"''"

    def test_a_bundle_requiring_what_isnt_here_is_refused_before_it_resolves_with_every_other_problem(self, tmp_path):
        root = _bundle(tmp_path / "b", "acme_plugin_not_here", toml='requires = ["agentenv-missing>=1"]\n')

        with pytest.raises(UnmetRequirements) as caught:
            resolve_bundle(parse_bundle(root))

        assert caught.value.requirements == ("agentenv-missing>=1",)
        assert MISSING in caught.value.problems
        assert any("acme_plugin_not_here" in problem for problem in caught.value.problems), caught.value.problems

    def test_a_bundle_whose_requirements_are_met_resolves(self, tmp_path):
        resolve_bundle(parse_bundle(_bundle(tmp_path / "b", _Acme.type, toml=f'requires = ["{ACME}>=1"]\n')))


class TestDeclare:
    def test_a_bundle_without_one_is_given_a_bundle_toml_listing_the_plugins_it_uses(self, tmp_path):
        root = _bundle(tmp_path / "b", _Acme.type, _Acme.type, "run_code")

        assert _declared(root) == [WROTE]
        assert tomllib.loads((root / "bundle.toml").read_text()) == {"requires": ["agentenv-acme>=1.2.0"]}
        assert parse_bundle(root).requires == ("agentenv-acme>=1.2.0",)
        assert _declared(root) == []

    def test_a_dry_run_says_what_it_would_write(self, tmp_path):
        root = _bundle(tmp_path / "b", _Acme.type)

        assert _declared(root, dry_run=True) == ['would write bundle.toml: requires = ["agentenv-acme>=1.2.0"]']
        assert not (root / "bundle.toml").exists()

    def test_a_bundle_using_only_what_core_ships_is_given_none(self, tmp_path):
        root = _bundle(tmp_path / "b", "run_code")

        assert _declared(root) == []
        assert not (root / "bundle.toml").exists()

    @pytest.mark.parametrize("toml, requires", [
        ("requires = []\n", ["agentenv-acme>=1.2.0"]),
        ('requires = ["agentenv-other"]\n', ["agentenv-acme>=1.2.0", "agentenv-other"]),
    ], ids=["opted-out", "requires-another"])
    def test_an_existing_bundle_toml_is_never_edited_but_told_what_to_declare(self, tmp_path, monkeypatch, toml,
                                                                            requires):
        monkeypatch.setattr(requires_module, "installed", {ACME: "1.2.0", "agentenv-other": "1"}.get)
        root = _bundle(tmp_path / "b", _Acme.type, toml=toml)

        assert _declared(root) == [
            "this bundle uses agentenv-acme 1.2.0 (tasks/t.json: step 's0'), which bundle.toml doesn't require",
            f"declare them in bundle.toml, so whoever it's shared with knows what to install: requires = "
            f"{json.dumps(requires)}"]
        assert (root / "bundle.toml").read_text() == toml

    def test_a_folder_not_its_own_is_told_what_to_declare(self, tmp_path):
        root = _bundle(tmp_path / "b", _Acme.type, _Acme.type)

        assert _declared(root, own=False) == [
            "this bundle uses agentenv-acme 1.2.0 (tasks/t.json: step 's0', and 1 more), which bundle.toml doesn't "
            "require",
            'declare them in bundle.toml, so whoever it\'s shared with knows what to install: requires = '
            '["agentenv-acme>=1.2.0"]']
        assert not (root / "bundle.toml").exists()

    def test_a_type_only_configuration_registers_is_named_and_never_required(self, tmp_path):
        root = _bundle(tmp_path / "b", _Configured.type, _Acme.type, _Replaced.type)

        assert _declared(root) == [
            WROTE,
            "this bundle uses the step type 'configured_requires_test' and the step type 'replaced_requires_test' "
            "(tasks/t.json: step 's0', and 1 more), which this machine's configuration registers; whoever runs it "
            "needs that configuration too"]
        assert parse_bundle(root).requires == ("agentenv-acme>=1.2.0",)

    def test_a_folder_on_a_filesystem_without_links_is_told_what_to_declare(self, tmp_path, monkeypatch):
        monkeypatch.setattr(requires_module.os, "link", _no_links)
        root = _bundle(tmp_path / "b", _Acme.type)

        assert _declared(root)[0] == ("this bundle uses agentenv-acme 1.2.0 (tasks/t.json: step 's0'), which bundle.toml "
                                      "doesn't require")
        assert sorted(os.listdir(root)) == ["tasks"]

    def test_a_type_configuration_registers_from_an_installed_package_names_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(requires_module, "packages_distributions", lambda: {"tst": ["agentenv-shipper"]})
        monkeypatch.setattr(requires_module, "installed", {ACME: "1.2.0", "agentenv-shipper": "2.0"}.get)
        root = _bundle(tmp_path / "b", _Configured.type, _Replaced.type)

        assert _declared(root) == [
            "this bundle uses the step type 'configured_requires_test' and the step type 'replaced_requires_test' "
            "(tasks/t.json: step 's0', and 1 more), which this machine's configuration registers, from "
            "agentenv-shipper 2.0; whoever runs it needs that package and its configuration"]
        assert not (root / "bundle.toml").exists()

    def test_a_bundle_toml_that_appears_meanwhile_is_kept_and_told_what_to_declare(self, tmp_path, monkeypatch):
        root = _bundle(tmp_path / "b", _Acme.type)
        resolved = resolve_bundle(parse_bundle(root))
        (root / "bundle.toml").write_text("requires = []\n")

        assert declare(resolved, own=True, dry_run=False)[0].startswith("this bundle uses agentenv-acme 1.2.0")
        assert (root / "bundle.toml").read_text() == "requires = []\n"


class TestCreate:
    def test_it_never_writes_over_a_file_and_leaves_nothing_staged(self, tmp_path):
        path = tmp_path / "bundle.toml"
        path.write_text("mine\n")

        assert _create(path, "theirs\n") is False
        assert path.read_text() == "mine\n"
        assert os.listdir(tmp_path) == ["bundle.toml"]

    def test_of_runs_racing_to_create_it_one_writes_it_whole(self, tmp_path):
        path = tmp_path / "bundle.toml"
        texts = [f"requires = [{i!r}]\n" * 200 for i in range(16)]
        start = threading.Barrier(len(texts))
        created = []

        def create(text):
            start.wait()
            created.append(_create(path, text))

        threads = [threading.Thread(target=create, args=(text,)) for text in texts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert created.count(True) == 1
        assert path.read_text() in texts
        assert os.listdir(tmp_path) == ["bundle.toml"]

    def test_a_filesystem_without_links_gets_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(requires_module.os, "link", _no_links)

        assert _create(tmp_path / "bundle.toml", "requires = []\n") is False
        assert os.listdir(tmp_path) == []

    def test_a_file_it_cant_write_whole_is_never_left_behind(self, tmp_path, monkeypatch):
        real_open = open

        def full(path, mode="r", **kwargs):
            file = real_open(path, mode, **kwargs)
            file.write = lambda text: (_ for _ in ()).throw(OSError(errno.ENOSPC, "No space left on device"))
            return file

        monkeypatch.setattr(requires_module, "open", full, raising=False)

        assert _create(tmp_path / "bundle.toml", "requires = []\n") is False
        assert os.listdir(tmp_path) == []

    @pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="a folder root can write to anyway")
    def test_a_folder_it_cant_write_to_is_left_as_it_is(self, tmp_path):
        tmp_path.chmod(0o555)
        try:
            assert _create(tmp_path / "bundle.toml", "x\n") is False
            assert os.listdir(tmp_path) == []
        finally:
            tmp_path.chmod(0o755)


@pytest.fixture
def stores(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class TestCommands:
    def test_a_run_writes_it_and_a_dry_run_or_a_service_run_doesnt(self, stores):
        root = _bundle(stores / "b", _Acme.type)
        dry, prepared, ran = [], [], []

        dry_run_bundle(root, on_progress=dry.append)
        with namespace_routing():
            prepare_bundle(root, on_progress=prepared.append)
        assert not (root / "bundle.toml").exists()
        run_bundle(root, on_progress=ran.append)

        assert 'would write bundle.toml: requires = ["agentenv-acme>=1.2.0"]' in dry
        assert prepared[0].startswith("this bundle uses agentenv-acme 1.2.0") and WROTE not in prepared
        assert WROTE in ran
        assert parse_bundle(root).requires == ("agentenv-acme>=1.2.0",)

    @pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="a folder root can write to anyway")
    def test_a_run_of_a_folder_it_cant_write_to_says_what_to_declare(self, stores):
        root = _bundle(stores / "b", _Acme.type)
        lines = []
        root.chmod(0o555)
        try:
            run_bundle(root, on_progress=lines.append)
        finally:
            root.chmod(0o755)

        assert "this bundle uses agentenv-acme 1.2.0 (tasks/t.json: step 's0'), which bundle.toml doesn't require" in lines
        assert not (root / "bundle.toml").exists()

    def test_a_registration_writes_it_unless_a_dry_run(self, stores, monkeypatch):
        monkeypatch.setattr(register_module, "_git", lambda root, *args: None)
        root = _bundle(stores / "b", _Acme.type)

        dry = register_bundle(root, dry_run=True).requires
        assert not (root / "bundle.toml").exists()
        registered = register_bundle(root, dry_run=True, write_requires=True).requires

        assert dry == ('would write bundle.toml: requires = ["agentenv-acme>=1.2.0"]',)
        assert registered == (WROTE,)
        assert parse_bundle(root).requires == ("agentenv-acme>=1.2.0",)
        assert register_bundle(root).requires == ()

    def test_the_cli_registration_writes_it_unless_a_dry_run_and_says_so_once(self, stores, monkeypatch):
        monkeypatch.setattr(register_module, "_git", lambda root, *args: None)
        root = _bundle(stores / "b", _Acme.type)

        dry = CliRunner().invoke(cli, ["bundle", "register", str(root), "--dry-run"])
        assert not (root / "bundle.toml").exists()
        done = CliRunner().invoke(cli, ["bundle", "register", str(root)])

        assert dry.exit_code == 0 and done.exit_code == 0, dry.output + done.output
        assert dry.output.count('  would write bundle.toml: requires = ["agentenv-acme>=1.2.0"]\n') == 1, dry.output
        assert done.output.count(f"  {WROTE}\n") == 1, done.output
        assert parse_bundle(root).requires == ("agentenv-acme>=1.2.0",)

    def test_the_cli_registration_says_what_to_declare_once(self, stores, monkeypatch):
        monkeypatch.setattr(register_module, "_git", lambda root, *args: None)
        root = _bundle(stores / "b", _Acme.type, toml="requires = []\n")

        result = CliRunner().invoke(cli, ["bundle", "register", str(root)])

        assert result.exit_code == 0, result.output
        assert result.output.count("this bundle uses agentenv-acme 1.2.0") == 1, result.output

    def test_the_cli_refuses_a_run_whose_requirements_arent_met_without_a_terminal(self, stores, monkeypatch):
        added = []
        monkeypatch.setattr(_install_prompt._plugin_changes, "add", lambda *a, **k: added.append(a))
        root = _bundle(stores / "b", _Acme.type, toml='requires = ["agentenv-missing>=1"]\n')

        result = CliRunner().invoke(cli, ["run", str(root)])

        assert result.exit_code == 1
        assert f"Error: {MISSING}" in result.output
        assert "Install" not in result.output and added == []


@pytest.fixture
def terminal(stores, monkeypatch):
    """A person at a terminal, an install that records what it's asked for, and a re-exec that stops the test."""
    calls = {"added": [], "exec": []}

    def add(specs, **kwargs):
        calls["added"].append((specs, kwargs["yes"]))
        return calls.get("code", 0)

    def execv(path, argv):
        calls["exec"].append((path, argv, os.environ.get(_install_prompt._ASKED)))
        raise SystemExit(0)

    monkeypatch.setattr(_install_prompt, "_interactive", lambda: True)
    monkeypatch.setattr(_install_prompt._plugin_changes, "add", add)
    monkeypatch.setattr(_install_prompt.os, "execv", execv)
    monkeypatch.setenv(_install_prompt._ASKED, "")
    root = _bundle(stores / "b", _Acme.type, toml='requires = ["agentenv-missing>=1"]\n')
    monkeypatch.setattr(sys, "argv", ["agent-env", "run", str(root)])
    return root, calls


class TestInstallPrompt:
    def test_yes_installs_it_and_runs_the_command_again(self, terminal):
        root, calls = terminal

        result = CliRunner().invoke(cli, ["run", str(root)], input="y\n")

        assert result.exit_code == 0, result.output
        assert result.output.count("Error:") == 1
        assert f"Error: {MISSING}" in result.output
        assert "Install agentenv-missing>=1 with agent-env plugin add? [y/N]: y" in result.output
        assert f"Running it again: agent-env run {root}" in result.output
        assert calls["added"] == [(["agentenv-missing>=1"], True)]
        assert calls["exec"] == [(sys.executable, [sys.executable, *sys.orig_argv[1:]], "1")]

    def test_no_installs_nothing_and_exits_1(self, terminal):
        root, calls = terminal

        result = CliRunner().invoke(cli, ["run", str(root)], input="n\n")

        assert result.exit_code == 1
        assert result.output.count("Error:") == 1
        assert calls["added"] == [] and calls["exec"] == []

    def test_a_command_run_again_after_an_install_never_asks_again(self, terminal, monkeypatch):
        root, calls = terminal
        monkeypatch.setenv(_install_prompt._ASKED, "1")

        result = CliRunner().invoke(cli, ["run", str(root)], input="y\n")

        assert result.exit_code == 1
        assert f"Error: {MISSING}" in result.output and "Install" not in result.output
        assert calls["added"] == []

    def test_a_dry_run_never_asks(self, terminal):
        root, calls = terminal

        result = CliRunner().invoke(cli, ["run", str(root), "--dry-run"], input="y\n")

        assert result.exit_code == 1
        assert "Install" not in result.output and calls["added"] == []

    def test_an_install_that_fails_stops_without_running_it_again(self, terminal, monkeypatch):
        root, calls = terminal

        def refused(specs, **kwargs):
            raise InstallerError("no installer could reach the index")

        monkeypatch.setattr(_install_prompt._plugin_changes, "add", refused)

        result = CliRunner().invoke(cli, ["run", str(root)], input="y\n")

        assert result.exit_code == 1
        assert "Error: no installer could reach the index" in result.output
        assert calls["exec"] == []

    def test_an_install_that_exits_non_zero_exits_with_its_code(self, terminal):
        root, calls = terminal
        calls["code"] = 2

        result = CliRunner().invoke(cli, ["run", str(root)], input="y\n")

        assert result.exit_code == 2
        assert calls["exec"] == []

    def test_a_refusal_for_agent_env_itself_never_asks(self, terminal, monkeypatch):
        root, calls = terminal
        (root / "bundle.toml").write_text('requires = ["agentenv-framework>=0.9.1400"]\n')
        monkeypatch.setattr(requires_module, "installed", {ACME: "1.2.0", "agentenv-framework": "0.9.1375"}.get)

        result = CliRunner().invoke(cli, ["run", str(root)], input="y\n")

        assert result.exit_code == 1
        assert "upgrade agent-env with the installer that installed it" in result.output
        assert "Install" not in result.output and calls["added"] == []

    def test_a_registration_refused_for_it_asks_too(self, terminal):
        root, calls = terminal

        result = CliRunner().invoke(cli, ["bundle", "register", str(root)], input="y\n")

        assert result.exit_code == 0, result.output
        assert calls["added"] == [(["agentenv-missing>=1"], True)]
        assert len(calls["exec"]) == 1


def test_unmet_requirements_are_a_bundle_error():
    assert issubclass(UnmetRequirements, BundleError)
