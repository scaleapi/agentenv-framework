"""Tests for the ``agent_env.cli_plugins`` and ``agent_env.cli_root_options`` loaders."""

import importlib
import os
import subprocess
import sys
from importlib.metadata import EntryPoint, entry_points

import click
import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.plugins import _cli as plugins_mod
from agent_env.plugins._cli import (
    CLI_PLUGINS_GROUP,
    CLI_ROOT_OPTIONS_GROUP,
    RootOptionConflictError,
    load_cli_plugins,
    load_cli_root_options,
)
from agent_env.plugins._discovery import dist_name

_HERE = "tst.unit.cli.plugins_test"

dummy_plugin = click.Group(name="dummy-plugin")
first_dup = click.Group(name="dup")
second_dup = click.Group(name="dup")
not_a_command = object()
nameless_command = click.Command(name=None)


@dummy_plugin.command()
def ping():
    click.echo("pong")


def _ep(name: str, attr: str) -> EntryPoint:
    return EntryPoint(name=name, value=f"{_HERE}:{attr}", group=CLI_PLUGINS_GROUP)


def _fake_entry_points(monkeypatch, eps, expected_group=CLI_PLUGINS_GROUP):
    def fake(*, group):
        assert group == expected_group
        return eps

    monkeypatch.setattr(plugins_mod, "entry_points", fake)


def test_happy_path_registers_plugin_group(monkeypatch, capsys):
    root = click.Group()
    _fake_entry_points(monkeypatch, [_ep("dummy", "dummy_plugin")])

    load_cli_plugins(root)

    assert root.commands["dummy-plugin"] is dummy_plugin
    assert capsys.readouterr().err == ""


def test_broken_entry_point_is_skipped_and_sibling_still_loads(monkeypatch, capsys):
    root = click.Group()
    _fake_entry_points(
        monkeypatch, [_ep("a_broken", "does_not_exist"), _ep("b_good", "dummy_plugin")]
    )

    load_cli_plugins(root)

    assert root.commands["dummy-plugin"] is dummy_plugin
    err = capsys.readouterr().err
    assert "a_broken" in err and "skipped" in err


def test_unimportable_module_is_skipped(monkeypatch, capsys):
    root = click.Group()
    ghost = EntryPoint(name="ghost", value="tst.unit.cli.no_such_module:x", group=CLI_PLUGINS_GROUP)
    _fake_entry_points(monkeypatch, [ghost])

    load_cli_plugins(root)

    assert root.commands == {}
    err = capsys.readouterr().err
    assert "ghost" in err and "skipped" in err


def test_non_command_object_is_skipped(monkeypatch, capsys):
    root = click.Group()
    _fake_entry_points(monkeypatch, [_ep("obj", "not_a_command")])

    load_cli_plugins(root)

    assert root.commands == {}
    assert "not a click.Command" in capsys.readouterr().err


def test_core_name_collision_is_skipped_and_core_wins(monkeypatch, capsys):
    root = click.Group()
    core = click.Group(name="dummy-plugin")
    root.add_command(core)
    _fake_entry_points(monkeypatch, [_ep("dummy", "dummy_plugin")])

    load_cli_plugins(root)

    assert root.commands["dummy-plugin"] is core
    err = capsys.readouterr().err
    assert "clashes" in err and "dummy-plugin" in err


def test_plugin_vs_plugin_first_by_name_wins(monkeypatch, capsys):
    root = click.Group()
    _fake_entry_points(
        monkeypatch, [_ep("z_second", "second_dup"), _ep("a_first", "first_dup")]
    )

    load_cli_plugins(root)

    assert root.commands["dup"] is first_dup
    assert "z_second" in capsys.readouterr().err


class _RaisingNameDist:
    @property
    def name(self):
        raise TypeError("missing package metadata")


class _EpWithBrokenDist:
    name = "broken_dist_meta"
    dist = _RaisingNameDist()

    def load(self):
        return dummy_plugin


def test_broken_dist_metadata_does_not_block_loading(monkeypatch, capsys):
    root = click.Group()
    _fake_entry_points(monkeypatch, [_EpWithBrokenDist()])

    load_cli_plugins(root)

    assert root.commands["dummy-plugin"] is dummy_plugin
    assert capsys.readouterr().err == ""


def test_discovery_failure_warns_and_loads_nothing(monkeypatch, capsys):
    root = click.Group()

    def broken(*, group):
        raise OSError("corrupt distribution metadata")

    monkeypatch.setattr(plugins_mod, "entry_points", broken)

    load_cli_plugins(root)

    assert root.commands == {}
    assert "discovery failed" in capsys.readouterr().err


def test_no_entry_points_is_silent(monkeypatch, capsys):
    root = click.Group()
    _fake_entry_points(monkeypatch, [])

    load_cli_plugins(root)

    assert root.commands == {}
    assert capsys.readouterr().err == ""


def test_warnings_use_stderr_and_leave_stdout_clean(monkeypatch, capsys):
    root = click.Group()
    _fake_entry_points(monkeypatch, [_ep("a_broken", "does_not_exist")])

    load_cli_plugins(root)

    captured = capsys.readouterr()
    assert "a_broken" in captured.err
    assert captured.out == ""


def test_unnamed_command_registers_under_entry_point_name(monkeypatch):
    root = click.Group()
    _fake_entry_points(monkeypatch, [_ep("fallback_name", "nameless_command")])

    load_cli_plugins(root)

    assert root.commands["fallback_name"] is nameless_command


class _FakeDist:
    def __init__(self, name):
        self.name = name


class _FakeEp:
    def __init__(self, name, command, dist_name):
        self.name = name
        self._command = command
        self.dist = _FakeDist(dist_name)

    def load(self):
        return self._command


def test_same_entry_point_name_tie_breaks_by_distribution(monkeypatch, capsys):
    root = click.Group()
    _fake_entry_points(
        monkeypatch,
        [_FakeEp("plugin", second_dup, "zz-dist"), _FakeEp("plugin", first_dup, "aa-dist")],
    )

    load_cli_plugins(root)

    assert root.commands["dup"] is first_dup
    assert "zz-dist" in capsys.readouterr().err


class _BrokenDistEp:
    def __init__(self, name, value, command):
        self.name = name
        self.value = value
        self.dist = _RaisingNameDist()
        self._command = command

    def load(self):
        return self._command


def test_all_metadata_broken_ties_break_by_target(monkeypatch):
    root = click.Group()
    _fake_entry_points(
        monkeypatch,
        [
            _BrokenDistEp("plugin", "zzz_mod:group", second_dup),
            _BrokenDistEp("plugin", "aaa_mod:group", first_dup),
        ],
    )

    load_cli_plugins(root)

    assert root.commands["dup"] is first_dup


def test_plugin_command_invocable_through_root(monkeypatch):
    root = click.Group()
    _fake_entry_points(monkeypatch, [_ep("dummy", "dummy_plugin")])
    load_cli_plugins(root)

    result = CliRunner().invoke(root, ["dummy-plugin", "ping"])

    assert result.exit_code == 0
    assert result.output.strip() == "pong"


def _write_real_dist(tmp_path, module_name, dist_name, *, with_metadata=True):
    command_name = module_name.replace("_", "-")
    (tmp_path / f"{module_name}.py").write_text(
        "import click\n\n"
        f'group = click.Group(name="{command_name}")\n\n\n'
        "@group.command()\n"
        "def ping():\n"
        '    click.echo("pong")\n'
    )
    dist_info = tmp_path / f"{dist_name}-0.0.1.dist-info"
    dist_info.mkdir()
    if with_metadata:
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: 0.0.1\n"
        )
    (dist_info / "entry_points.txt").write_text(
        f"[{CLI_PLUGINS_GROUP}]\n{module_name} = {module_name}:group\n"
    )
    (dist_info / "RECORD").write_text("")
    return command_name


def test_real_distribution_is_discovered_and_loaded(tmp_path, monkeypatch):
    command_name = _write_real_dist(tmp_path, "plugin_real_ok", "plugin-real-ok")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    root = click.Group()

    load_cli_plugins(root)

    assert command_name in root.commands
    ep = next(e for e in entry_points(group=CLI_PLUGINS_GROUP) if e.name == "plugin_real_ok")
    assert dist_name(ep) == "plugin-real-ok"


def test_real_distribution_without_metadata_still_loads(tmp_path, monkeypatch):
    command_name = _write_real_dist(
        tmp_path, "plugin_real_nometa", "plugin-real-nometa", with_metadata=False
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    root = click.Group()

    load_cli_plugins(root)

    assert command_name in root.commands
    ep = next(e for e in entry_points(group=CLI_PLUGINS_GROUP) if e.name == "plugin_real_nometa")
    assert dist_name(ep) == ""


def test_module_import_hookup_registers_real_plugin(tmp_path):
    command_name = _write_real_dist(tmp_path, "plugin_real_sub", "plugin-real-sub")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(tmp_path), env.get("PYTHONPATH", "")] if p
    )
    code = (
        "from agent_env.cli import cli; "
        "print(','.join(sorted(cli.commands))); "
        f"cli(args=['{command_name}', 'ping'])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )

    assert proc.returncode == 0, proc.stderr
    listing, invocation = proc.stdout.strip().splitlines()
    commands = listing.split(",")
    assert command_name in commands
    for core in ["a2a-agent", "artifact", "env", "eval", "task"]:
        assert core in commands
    assert invocation == "pong"


@pytest.mark.skipif(
    len(entry_points(group=CLI_PLUGINS_GROUP)) > 0,
    reason="bare-install assertion only holds when no agent_env.cli_plugins distribution is installed",
)
def test_bare_install_core_commands_unchanged():
    assert sorted(cli.commands) == [
        "a2a-agent",
        "artifact",
        "config",
        "env",
        "eval",
        "plugin",
        "task",
        "up",
    ]


recorded: list[str] = []


def _record_tenant(ctx, param, value):
    if value is not None:
        recorded.append(f"tenant={value}")


def _record_sub_flag(ctx, param, value):
    if value is not None:
        recorded.append(f"sub-parse={value}")
    return value


tenant_option = click.Option(["--tenant"], expose_value=False, callback=_record_tenant)
tenant_dup = click.Option(["--tenant"], expose_value=False)
exposing_option = click.Option(["--exposing"])
required_option = click.Option(["--must"], expose_value=False, required=True)
verbose_clash = click.Option(["--verbose"], expose_value=False)
short_clash = click.Option(["--loud", "-v"], expose_value=False)
help_clash = click.Option(["--help"], expose_value=False)
not_an_option = click.Argument(["thing"])


def _root_ep(name: str, attr: str) -> EntryPoint:
    return EntryPoint(name=name, value=f"{_HERE}:{attr}", group=CLI_ROOT_OPTIONS_GROUP)


def _root_with_sub():
    recorded.clear()
    root = click.Group(params=[click.Option(["--verbose", "-v"], is_flag=True)])

    @root.command()
    @click.option("--flag", callback=_record_sub_flag)
    def sub(flag):
        recorded.append("sub")

    return root


def test_root_option_attaches_and_runs_before_the_subcommand_parses(monkeypatch, capsys):
    root = _root_with_sub()
    _fake_entry_points(monkeypatch, [_root_ep("tenant", "tenant_option")], CLI_ROOT_OPTIONS_GROUP)

    load_cli_root_options(root)
    result = CliRunner().invoke(root, ["--tenant", "acme", "sub", "--flag", "x"])

    assert result.exit_code == 0, result.output
    assert recorded == ["tenant=acme", "sub-parse=x", "sub"]
    assert capsys.readouterr().err == ""


def test_root_option_is_optional_and_stays_off_subcommands(monkeypatch):
    root = _root_with_sub()
    _fake_entry_points(monkeypatch, [_root_ep("tenant", "tenant_option")], CLI_ROOT_OPTIONS_GROUP)
    load_cli_root_options(root)
    runner = CliRunner()

    assert runner.invoke(root, ["sub"]).exit_code == 0
    assert recorded == ["sub"]
    assert "--tenant" in runner.invoke(root, ["--help"]).output
    assert "--tenant" not in runner.invoke(root, ["sub", "--help"]).output
    assert "no such option" in runner.invoke(root, ["sub", "--tenant", "acme"]).output.lower()


@pytest.mark.parametrize(
    "attr,option",
    [("exposing_option", exposing_option), ("required_option", required_option)],
)
def test_root_option_that_exposes_a_value_or_is_required_is_skipped(monkeypatch, capsys, attr, option):
    root = _root_with_sub()
    _fake_entry_points(monkeypatch, [_root_ep("bad", attr)], CLI_ROOT_OPTIONS_GROUP)

    load_cli_root_options(root)

    assert option not in root.params
    assert "expose_value=False" in capsys.readouterr().err


def test_root_option_that_is_not_an_option_is_skipped(monkeypatch, capsys):
    root = _root_with_sub()
    _fake_entry_points(monkeypatch, [_root_ep("thing", "not_an_option")], CLI_ROOT_OPTIONS_GROUP)

    load_cli_root_options(root)

    assert not_an_option not in root.params
    assert "not a click.Option" in capsys.readouterr().err


@pytest.mark.parametrize(
    "attr,option,flag",
    [
        ("verbose_clash", verbose_clash, "--verbose"),
        ("short_clash", short_clash, "-v"),
        ("help_clash", help_clash, "--help"),
    ],
)
def test_root_option_clashing_with_a_root_flag_is_skipped(monkeypatch, capsys, attr, option, flag):
    root = _root_with_sub()
    _fake_entry_points(monkeypatch, [_root_ep("clash", attr)], CLI_ROOT_OPTIONS_GROUP)

    load_cli_root_options(root)

    assert option not in root.params
    err = capsys.readouterr().err
    assert "clashes" in err and flag in err
    assert CliRunner().invoke(root, ["--help"]).exit_code == 0


def test_two_plugins_claiming_the_same_root_flag_abort_startup(monkeypatch):
    root = _root_with_sub()
    _fake_entry_points(
        monkeypatch, [_root_ep("z_second", "tenant_dup"), _root_ep("a_first", "tenant_option")], CLI_ROOT_OPTIONS_GROUP
    )

    with pytest.raises(RootOptionConflictError) as excinfo:
        load_cli_root_options(root)

    message = str(excinfo.value)
    assert "'--tenant'" in message and "'a_first'" in message and "'z_second'" in message
    assert tenant_dup not in root.params


def test_the_same_root_option_object_listed_twice_is_attached_once(monkeypatch, capsys):
    root = _root_with_sub()
    _fake_entry_points(
        monkeypatch, [_root_ep("a_first", "tenant_option"), _root_ep("b_again", "tenant_option")], CLI_ROOT_OPTIONS_GROUP
    )

    load_cli_root_options(root)

    assert root.params.count(tenant_option) == 1
    assert capsys.readouterr().err == ""


def test_broken_root_option_is_skipped_and_sibling_still_loads(monkeypatch, capsys):
    root = _root_with_sub()
    _fake_entry_points(
        monkeypatch,
        [_root_ep("a_broken", "does_not_exist"), _root_ep("b_tenant", "tenant_option")],
        CLI_ROOT_OPTIONS_GROUP,
    )

    load_cli_root_options(root)

    assert tenant_option in root.params
    err = capsys.readouterr().err
    assert "a_broken" in err and "skipped" in err


def _write_real_root_option_dist(tmp_path, module_name, dist_name):
    (tmp_path / f"{module_name}.py").write_text(
        "import click\n\n\n"
        "def pick(ctx, param, value):\n"
        "    if value is not None:\n"
        '        click.echo(f"tenant={value}")\n\n\n'
        'option = click.Option(["--tenant"], expose_value=False, callback=pick)\n'
    )
    # PEP 427 dist-info stem: the normalised name, or importlib.metadata folds two dists into one
    dist_info = tmp_path / f"{dist_name.replace('-', '_')}-0.0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: 0.0.1\n")
    (dist_info / "entry_points.txt").write_text(f"[{CLI_ROOT_OPTIONS_GROUP}]\n{module_name} = {module_name}:option\n")
    (dist_info / "RECORD").write_text("")


def _import_cli_with(tmp_path, code):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(tmp_path), env.get("PYTHONPATH", "")] if p)
    # nosemgrep: dangerous-subprocess-use-audit -- argv is sys.executable + the literal `code`; no external input
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_module_import_hookup_registers_real_root_option(tmp_path):
    _write_real_root_option_dist(tmp_path, "tenant_plugin", "tenant-plugin")
    proc = _import_cli_with(tmp_path, "from agent_env.cli import cli; cli(args=['--tenant', 'acme', 'task', '--help'])")

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[0] == "tenant=acme"
    assert "Usage:" in proc.stdout


def test_two_real_distributions_on_the_same_flag_abort_cli_startup(tmp_path):
    _write_real_root_option_dist(tmp_path, "tenant_plugin_a", "tenant-plugin-a")
    _write_real_root_option_dist(tmp_path, "tenant_plugin_b", "tenant-plugin-b")
    proc = _import_cli_with(tmp_path, "from agent_env.cli import cli; cli(args=['--help'])")

    assert proc.returncode != 0, proc.stdout
    assert "RootOptionConflictError" in proc.stderr
    assert "'tenant_plugin_a' from 'tenant-plugin-a'" in proc.stderr
    assert "'tenant_plugin_b' from 'tenant-plugin-b'" in proc.stderr


@pytest.mark.skipif(
    len(entry_points(group=CLI_ROOT_OPTIONS_GROUP)) > 0,
    reason="bare-install assertion only holds when no agent_env.cli_root_options distribution is installed",
)
def test_bare_install_root_options_unchanged():
    assert [param.name for param in cli.params] == ["verbose"]
