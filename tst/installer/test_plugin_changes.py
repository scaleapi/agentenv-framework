"""`plugin add` and `remove` through each real installer, offline.

The central check is the rebuild: after `add`, the installer recreates the environment the way
it normally would, and the plugin must still be there. A plain `pip install` into a uv tool or
pipx environment fails that; going through the installer passes it.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tst.installer.conftest import PYTHON, Setup, _checked, _platform_marker, build_wheel, make_setup, requires_tool
from tst.installer.toys import BROWSER, GRADE_STEP, GRADER, browser_wheel, grader_wheel
from tst.util.capabilities import missing_capability_reason

pytestmark = [pytest.mark.installer, requires_tool("uv")]

# How long the lock test waits for the first change to take the lock, and to finish once told no.
_LOCK_WAIT_S = 120
_POLL_INTERVAL_S = 0.2

_STEP = GRADE_STEP


@pytest.fixture
def toys(tmp_path) -> dict[str, Path]:
    """The toy plugins, as wheels in a directory of their own."""
    out = tmp_path / "toys"
    out.mkdir()
    requires = ("agentenv-framework",)
    return {
        "grader": grader_wheel(out),
        "grader-0.4.0": grader_wheel(out, "0.4.0"),
        "broken": build_wheel(out, "agentenv-toy-broken", "0.1.0",
                              {"agentenv_toy_broken/__init__.py": "import agentenv_toy_missing_module\n"},
                              {"agent_env.task_steps": {"toy_broken": "agentenv_toy_broken:Step"}}, requires),
        "not-a-plugin": build_wheel(out, "agentenv-toy-helper", "0.1.0", {"agentenv_toy_helper.py": ""}),
        # Claims the grader's step name, so the two conflict.
        "rival": build_wheel(out, "agentenv-toy-rival", "0.1.0",
                             {"agentenv_toy_rival/__init__.py": "", "agentenv_toy_rival/steps.py": _STEP},
                             {"agent_env.task_steps": {"toy_grade": "agentenv_toy_rival.steps:GradeTaskStep"}},
                             requires),
        "clash": build_wheel(out, "agentenv-toy-clash", "0.1.0",
                             {"agentenv_toy_clash/__init__.py": "", "agentenv_toy_clash/steps.py": _STEP},
                             {"agent_env.task_steps": {"deploy_env": "agentenv_toy_clash.steps:GradeTaskStep"}},
                             requires),
        "configurer": build_wheel(out, "agentenv-toy-configurer", "0.1.0", {
            "agentenv_toy_configurer/__init__.py": (
                "import os, pathlib\n"
                "os.environ.setdefault('AGENT_ENV_CONFIG', str(pathlib.Path(__file__).with_name('config.toml')))\n"
            ),
            "agentenv_toy_configurer/config.toml": "",
            "agentenv_toy_configurer/steps.py": _STEP.replace("toy_grade", "toy_configured"),
        }, {"agent_env.task_steps": {"toy_configured": "agentenv_toy_configurer.steps:GradeTaskStep"}}, requires),
        # A root option whose callback fails even when its flag is not given.
        "option": build_wheel(out, "agentenv-toy-option", "0.1.0", {"agentenv_toy_option.py": (
            "import click\n\n"
            "def _fail(ctx, param, value):\n    raise RuntimeError('toy option is broken')\n\n"
            "OPTION = click.Option(['--toy-flag'], expose_value=False, callback=_fail)\n"
        )}, {"agent_env.cli_root_options": {"toy_flag": "agentenv_toy_option:OPTION"}}, requires),
        "box": build_wheel(out, "agentenv-toy-box", "0.1.0", {"agentenv_toy_box.py": "class Box:\n    pass\n"},
                           {"agent_env.sandbox_providers": {"toy_box": "agentenv_toy_box:Box"}}, requires),
        # Like a platform plugin: a CLI plugin, so it is imported at startup, and on import it selects
        # the config it bundles, which names its own sandbox provider.
        "bundler": build_wheel(out, "agentenv-toy-bundler", "0.1.0", {
            "agentenv_toy_bundler/__init__.py": (
                "import os, pathlib\n"
                "os.environ.setdefault('AGENT_ENV_CONFIG',\n"
                "                      str(pathlib.Path(__file__).with_name('config.toml')))\n\n"
                "class Box:\n    pass\n"
            ),
            "agentenv_toy_bundler/config.toml": '[sandbox]\ndefault = "toy_bundled_box"\n',
            "agentenv_toy_bundler/cli.py": (
                "import click\n\n\n@click.command('toy-bundler')\ndef command():\n    pass\n"
            ),
        }, {"agent_env.cli_plugins": {"toy_bundler": "agentenv_toy_bundler.cli:command"},
            "agent_env.sandbox_providers": {"toy_bundled_box": "agentenv_toy_bundler:Box"}}, requires),
        # The same, but the config it selects is a shared file it does not install, so it stays.
        "sharer": build_wheel(out, "agentenv-toy-sharer", "0.1.0", {
            "agentenv_toy_sharer/__init__.py": (
                "import os\n"
                "os.environ.setdefault('AGENT_ENV_CONFIG', os.path.expanduser('~/shared-config.toml'))\n\n"
                "class Box:\n    pass\n"
            ),
            "agentenv_toy_sharer/cli.py": (
                "import click\n\n\n@click.command('toy-sharer')\ndef command():\n    pass\n"
            ),
        }, {"agent_env.cli_plugins": {"toy_sharer": "agentenv_toy_sharer.cli:command"},
            "agent_env.sandbox_providers": {"toy_shared_box": "agentenv_toy_sharer:Box"}}, requires),
    }


@pytest.fixture
def imposter(tmp_path) -> Path:
    """A source directory that builds as agentenv-framework, `agent-env` command included, so
    the installer accepts it: its name only shows once built."""
    root = tmp_path / "imposter"
    (root / "agentenv_framework").mkdir(parents=True)
    (root / "agentenv_framework" / "__init__.py").write_text("class Step:\n    pass\n\n\ndef main():\n    pass\n")
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["flit_core>=3.2"]\nbuild-backend = "flit_core.buildapi"\n\n'
        '[project]\nname = "agentenv-framework"\nversion = "9.9.9"\ndescription = "not agent-env"\n\n'
        '[project.scripts]\nagent-env = "agentenv_framework:main"\n\n'
        '[project.entry-points."agent_env.task_steps"]\nimposter = "agentenv_framework:Step"\n'
    )
    return root


KINDS = [
    "venv",
    "uv-project",
    pytest.param("uv-tool"),
    pytest.param("pipx", marks=requires_tool("pipx")),
]


@pytest.fixture(params=KINDS)
def setup(request, tmp_path, hermetic) -> Setup:
    return make_setup(request.param, tmp_path, hermetic)


def _status(setup: Setup, package: str) -> set[str]:
    return {c["status"] for c in setup.plugins()[package]["contributions"]}


def test_a_plugin_survives_the_installers_own_rebuild_and_is_removed(setup, toys):
    added = setup.cli("plugin", "add", str(toys["grader"]), "--yes")
    assert added.returncode == 0, added.stdout + added.stderr
    assert _status(setup, GRADER) == {"active"}

    setup.rebuild()
    assert _status(setup, GRADER) == {"active"}

    removed = setup.cli("plugin", "remove", GRADER, "--yes")
    assert removed.returncode == 0, removed.stdout + removed.stderr
    assert GRADER not in setup.plugins()
    for record in setup.records:
        assert "toy-grader" not in record.read_text() and "toy_grader" not in record.read_text()


@pytest.mark.parametrize("toy, problem", [
    ("broken", "toy_broken is failed"),
    ("not-a-plugin", "not an agent-env plugin"),
    ("clash", "deploy_env is skipped"),
])
def test_an_add_that_does_not_take_effect_restores_the_environment_exactly(setup, toys, toy, problem):
    before = setup.state()

    result = setup.cli("plugin", "add", str(toys[toy]), "--yes")

    assert result.returncode == 1, result.stdout + result.stderr
    assert problem in result.stderr and "Restored." in result.stderr
    assert setup.state() == before


def test_a_newer_wheel_upgrades_the_installed_plugin(setup, toys):
    assert setup.cli("plugin", "add", str(toys["grader"]), "--yes").returncode == 0

    upgraded = setup.cli("plugin", "add", str(toys["grader-0.4.0"]), "--yes")

    assert upgraded.returncode == 0, upgraded.stdout + upgraded.stderr
    assert setup.plugins()[GRADER]["version"] == "0.4.0"


def test_a_plugin_that_conflicts_with_an_installed_one_is_rolled_back(setup, toys):
    assert setup.cli("plugin", "add", str(toys["grader"]), "--yes").returncode == 0
    before = setup.state()

    result = setup.cli("plugin", "add", str(toys["rival"]), "--yes")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "toy_grade is conflict" in result.stderr and "Restored." in result.stderr
    assert setup.state() == before


def test_a_bare_directory_that_builds_as_agent_env_never_replaces_it(setup, imposter):
    before = setup.state()

    result = setup.cli("plugin", "add", str(imposter), "--yes")

    # pipx skips its own main package; the others install it, and add rolls it back.
    if setup.kind == "pipx":
        assert result.returncode == 0 and "Nothing changed" in result.stdout, result.stdout + result.stderr
    else:
        assert result.returncode == 1, result.stdout + result.stderr
        assert "a spec built as agent-env itself" in result.stderr and "Restored." in result.stderr
    assert setup.state() == before


@pytest.fixture(params=["venv", "uv-tool"])
def either(request, tmp_path, hermetic) -> Setup:
    """For checks that do not depend on the installer: one plain and one recorded setup."""
    return make_setup(request.param, tmp_path, hermetic)


def test_a_plugin_that_sets_the_config_on_import_is_reported(either, toys):
    result = either.cli("plugin", "add", str(toys["configurer"]), "--yes")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "importing it sets AGENT_ENV_CONFIG to" in result.stdout


def test_a_plugin_whose_root_option_breaks_startup_can_still_be_removed(either, toys):
    kept = either.cli("plugin", "add", str(toys["option"]), "--yes", "--keep")
    assert kept.returncode == 1 and "Kept as installed" in kept.stderr

    listed = either.cli("plugin", "list")
    assert listed.returncode == 0 and "toy option is broken" in listed.stderr

    removed = either.cli("plugin", "remove", "agentenv-toy-option", "--yes")
    assert removed.returncode == 0, removed.stdout + removed.stderr
    assert "agentenv-toy-option" not in either.plugins()


def test_agent_env_itself_is_never_added_or_removed(either, checkout_wheels):
    framework = next(checkout_wheels.glob("agentenv_framework-*.whl"))

    added = either.cli("plugin", "add", str(framework), "--yes")
    removed = either.cli("plugin", "remove", "agentenv-framework", "--yes")

    assert added.returncode == removed.returncode == 1
    assert "is agent-env itself" in added.stderr and "is agent-env itself" in removed.stderr


def test_remove_is_blocked_while_the_config_names_its_sandbox_provider(either, toys, tmp_path):
    assert either.cli("plugin", "add", str(toys["box"]), "--yes", "--keep").returncode == 1
    config = tmp_path / "config.toml"
    config.write_text('[sandbox]\ndefault = "toy_box"\n')

    blocked = either.cli("plugin", "remove", "agentenv-toy-box", "--yes", env={"AGENT_ENV_CONFIG": str(config)})
    forced = either.cli("plugin", "remove", "agentenv-toy-box", "--yes", "--force",
                        env={"AGENT_ENV_CONFIG": str(config)})

    assert blocked.returncode == 1 and "[sandbox] default = 'toy_box'" in blocked.stderr
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert "agentenv-toy-box" not in either.plugins()


@pytest.mark.parametrize("kind, command", [
    ("poetry", "poetry add"), ("pdm", "pdm add"), ("hatch", "Hatch has no command for this"),
])
def test_a_project_tool_that_owns_the_environment_is_given_the_command_and_nothing_runs(
        kind, command, tmp_path, hermetic, toys):
    owned = make_setup(kind, tmp_path, hermetic)
    before = owned.state()

    result = owned.cli("plugin", "add", str(toys["grader"]), "--yes")

    assert result.returncode == 0, result.stdout + result.stderr
    assert command in result.stdout
    assert owned.state() == before


# Root writes through a read-only directory, so only an unprivileged user can see the refusal.
@pytest.mark.skipif(os.geteuid() == 0, reason=missing_capability_reason("unprivileged_user"))
def test_read_only_site_packages_are_refused(tmp_path, hermetic, toys):
    venv = make_setup("venv", tmp_path, hermetic)
    (site,) = (tmp_path / "venv" / "lib").glob("python*/site-packages")
    site.chmod(0o555)
    try:
        result = venv.cli("plugin", "add", str(toys["grader"]), "--yes")
    finally:
        site.chmod(0o755)

    assert result.returncode == 1 and "will not change this environment" in result.stderr


def test_a_second_change_waits_for_the_first_to_finish(tmp_path, hermetic, toys):
    venv = make_setup("venv", tmp_path, hermetic)
    first = subprocess.Popen([str(venv.cli_path), "plugin", "add", str(toys["grader"])], env=venv.env, cwd=venv.work,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    locks = Path(hermetic["XDG_STATE_HOME"]) / "agent-env" / "locks"
    deadline = time.monotonic() + _LOCK_WAIT_S
    while not (locks.is_dir() and any(locks.iterdir())) and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
    try:
        second = venv.cli("plugin", "add", str(toys["grader"]), "--dry-run")
    finally:
        first.communicate("n\n", timeout=_LOCK_WAIT_S)

    assert second.returncode == 1 and "another `agent-env plugin add` or `remove`" in second.stderr
    assert first.returncode == 1 and GRADER not in venv.plugins()


def test_a_plugin_whose_bundled_config_is_in_effect_can_be_removed(either, toys):
    kept = either.cli("plugin", "add", str(toys["bundler"]), "--yes", "--keep")
    assert kept.returncode == 1 and "Kept as installed" in kept.stderr

    removed = either.cli("plugin", "remove", "agentenv-toy-bundler", "--yes")

    assert removed.returncode == 0, removed.stdout + removed.stderr
    assert "installs the config file in effect" in removed.stderr and "blocked" not in removed.stderr
    assert "agentenv-toy-bundler" not in either.plugins()


def test_a_uv_tool_whose_index_comes_from_its_config_takes_one_change_after_another(
        tmp_path, hermetic, simple_index, toys):
    tool = make_setup("uv-tool-indexed", tmp_path, {**hermetic, "SIMPLE_INDEX": simple_index})

    first = tool.cli("plugin", "add", str(toys["grader"]), "--yes")
    second = tool.cli("plugin", "add", str(toys["grader-0.4.0"]), "--yes")

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert tool.plugins()[GRADER]["version"] == "0.4.0"

    before = tool.state()
    failed = tool.cli("plugin", "add", str(toys["broken"]), "--yes")
    assert failed.returncode == 1 and "Restored." in failed.stderr, failed.stdout + failed.stderr
    assert tool.state() == before
    assert tool.cli("plugin", "remove", GRADER, "--yes").returncode == 0


def test_a_plugin_that_selects_a_shared_config_file_is_still_blocked_by_it(either, toys):
    (Path(either.env["HOME"]) / "shared-config.toml").write_text('[sandbox]\ndefault = "toy_shared_box"\n')
    assert either.cli("plugin", "add", str(toys["sharer"]), "--yes", "--keep").returncode == 1

    blocked = either.cli("plugin", "remove", "agentenv-toy-sharer", "--yes")

    assert blocked.returncode == 1, blocked.stdout + blocked.stderr
    assert "[sandbox] default = 'toy_shared_box'" in blocked.stderr


def _started(setup: Setup, *args: str) -> tuple[subprocess.Popen, list[str]]:
    """`agent-env ARGS` in a session of its own, as a terminal runs it, with its output collected
    by a thread; `_finished` waits for both."""
    proc = subprocess.Popen([str(setup.cli_path), *args], env=setup.env, cwd=setup.work, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    out: list[str] = []

    def pump():
        while chunk := os.read(proc.stdout.fileno(), 4096):
            out.append(chunk.decode(errors="replace"))
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    proc.reader = reader
    return proc, out


def _finished(proc: subprocess.Popen, answer: bytes = b"") -> None:
    """Answer the prompt, if asked, then wait for the command and for all of its output."""
    if answer:
        proc.stdin.write(answer)
        proc.stdin.flush()
    proc.stdin.close()
    proc.wait(_LOCK_WAIT_S)
    proc.reader.join(_LOCK_WAIT_S)


def _wait_for(condition, what: str) -> None:
    deadline = time.monotonic() + _LOCK_WAIT_S
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(_POLL_INTERVAL_S)


def _broken_records(setup: Setup) -> list[str]:
    (site,) = setup.python.parent.parent.glob("lib/python*/site-packages")
    return sorted(record.name for record in site.glob("*.dist-info") if not (record / "METADATA").is_file())


def test_ctrl_c_lets_the_installer_finish_and_then_restores(tmp_path, hermetic, toys):
    # uv behind a shim that notes how it was started, then waits: an installer stopped halfway
    # leaves packages without metadata, so a Ctrl-C at the terminal must not stop it.
    shim, started = tmp_path / "shim", tmp_path / "started"
    shim.mkdir()
    (shim / "uv").write_text(
        f"#!{sys.executable}\n"
        "import os, signal, sys, time\n"
        f"if sys.argv[1:3] == ['tool', 'install'] and not os.path.exists({str(started)!r}):\n"
        "    ignored = signal.getsignal(signal.SIGINT) == signal.SIG_IGN\n"
        f"    open({str(started)!r}, 'w').write('ignored' if ignored else 'default')\n"
        "    time.sleep(3)\n"
        f"os.execv({shutil.which('uv')!r}, ['uv', *sys.argv[1:]])\n"
    )
    (shim / "uv").chmod(0o755)
    tool = make_setup("uv-tool", tmp_path, hermetic)
    tool.env["PATH"] = f"{shim}:{tool.env['PATH']}"
    before = tool.state()

    proc, out = _started(tool, "plugin", "add", str(toys["grader"]), "--yes")
    _wait_for(lambda: started.exists() and started.read_text(), "the installer to start")
    os.killpg(proc.pid, signal.SIGINT)  # what Ctrl-C at the terminal does: the whole foreground group
    _finished(proc)

    output = "".join(out)
    assert proc.returncode == 130, output
    assert started.read_text() == "ignored", "a Ctrl-C at the terminal could stop the installer halfway"
    assert "letting the installer finish" in output and "Restored." in output
    assert tool.state() == before and _broken_records(tool) == []


def test_a_change_made_elsewhere_while_add_waits_for_confirmation_is_not_undone(tmp_path, hermetic, toys):
    tool = make_setup("uv-tool", tmp_path, hermetic)
    browser = browser_wheel(tmp_path)
    proc, out = _started(tool, "plugin", "add", str(toys["grader"]))
    _wait_for(lambda: "Proceed?" in "".join(out), "the confirmation prompt")

    _checked(["uv", "tool", "install", "-q", "agentenv-framework", "--python", str(tool.python.resolve()),
              "--with", str(browser)], env=tool.env)
    _finished(proc, b"y\n")

    assert proc.returncode == 1 and "changed while this waited to be confirmed" in "".join(out)
    assert set(tool.plugins()) == {"agentenv-framework", BROWSER}


def _with_links(hermetic: dict[str, str], links: Path) -> dict[str, str]:
    """``hermetic`` finding wheels in ``links`` too, for a plugin's own requirements."""
    return {**hermetic, "UV_FIND_LINKS": f"{hermetic['UV_FIND_LINKS']},{links}",
            "PIP_FIND_LINKS": f"{hermetic['PIP_FIND_LINKS']} {links}"}


@pytest.mark.parametrize("kind", ["uv-tool", "uv-project"])
def test_removing_a_plugin_names_and_checks_the_plugins_that_go_with_it(kind, tmp_path, hermetic):
    links = tmp_path / "links"
    links.mkdir()
    grader_wheel(links)
    suite = build_wheel(tmp_path, "agentenv-toy-suite", "1.0.0",
                        {"agentenv_toy_suite.py": _STEP.replace("toy_grade", "toy_suite")},
                        {"agent_env.task_steps": {"toy_suite": "agentenv_toy_suite:GradeTaskStep"}},
                        ("agentenv-framework", GRADER))
    owned = make_setup(kind, tmp_path, _with_links(hermetic, links))
    assert owned.cli("plugin", "add", str(suite), "--yes").returncode == 0
    config = tmp_path / "config.toml"
    config.write_text('[task_steps]\nimpls = ["agentenv_toy_grader.steps:GradeTaskStep"]\n')

    blocked = owned.cli("plugin", "remove", "agentenv-toy-suite", "--yes", env={"AGENT_ENV_CONFIG": str(config)})
    removed = owned.cli("plugin", "remove", "agentenv-toy-suite", "--yes")
    owned.rebuild()

    assert blocked.returncode == 1 and "This also removes agentenv-toy-grader" in blocked.stdout
    assert "impls names 'agentenv_toy_grader.steps:GradeTaskStep'" in blocked.stderr
    assert removed.returncode == 0, removed.stdout + removed.stderr
    assert GRADER not in owned.plugins() and "agentenv-toy-suite" not in owned.plugins()


def test_a_package_added_next_to_a_plugin_is_removed_the_same_way(tmp_path, hermetic, toys):
    tool = make_setup("uv-tool", tmp_path, hermetic)
    needy = build_wheel(tmp_path, "agentenv-toy-needy", "0.1.0",
                        {"agentenv_toy_needy.py": _STEP.replace("toy_grade", "toy_needy")},
                        {"agent_env.task_steps": {"toy_needy": "agentenv_toy_needy:GradeTaskStep"}},
                        ("agentenv-framework", "agentenv-toy-helper"))
    assert tool.cli("plugin", "add", str(needy), str(toys["not-a-plugin"]), "--yes").returncode == 0

    plugin_gone = tool.cli("plugin", "remove", "agentenv-toy-needy", "--yes")
    helper_gone = tool.cli("plugin", "remove", "agentenv-toy-helper", "--yes")

    assert plugin_gone.returncode == 0 and "agentenv-toy-helper stays installed" in plugin_gone.stderr
    assert helper_gone.returncode == 0, helper_gone.stdout + helper_gone.stderr
    assert "toy_helper" not in tool.records[0].read_text() and "toy-helper" not in tool.records[0].read_text()


def test_adding_a_plugin_by_its_bare_name_keeps_the_version_the_tool_pins(tmp_path, hermetic, toys):
    tool = make_setup("uv-tool", tmp_path, _with_links(hermetic, toys["grader"].parent))
    assert tool.cli("plugin", "add", f"{GRADER}==0.3.0", "--yes").returncode == 0
    before = tool.state()

    again = tool.cli("plugin", "add", GRADER, "--yes")

    assert again.returncode == 0 and "Nothing changed" in again.stdout, again.stdout + again.stderr
    assert tool.state()[1] == before[1]
    tool.rebuild()
    assert tool.plugins()[GRADER]["version"] == "0.3.0"


def _project(root: Path, dependencies: str = "[]", tail: str = "") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{root.name}"\nversion = "0.1.0"\nrequires-python = ">={PYTHON}"\n'
        f'dependencies = {dependencies}\n{tail}'
    )


# Absolute, run from inside the project; relative (to the project, as uv reads it), run from outside.
@pytest.mark.parametrize("relative", [False, True])
def test_a_uv_project_with_its_environment_elsewhere_keeps_the_plugin_through_uv_sync(
        tmp_path, hermetic, toys, relative):
    project = tmp_path / "project"
    venv = project / ".venv-alt" if relative else tmp_path / "elsewhere-venv"
    env = {**hermetic, "UV_PROJECT_ENVIRONMENT": ".venv-alt" if relative else str(venv)}
    _project(project, tail=f'\n[tool.uv]\nenvironments = ["{_platform_marker()}"]\n')
    _checked(["uv", "add", "-q", "--python", sys.executable, "agentenv-framework"], env=env, cwd=project)
    work = tmp_path / "work" if relative else project
    owned = Setup("uv-project", venv / "bin" / "python", venv / "bin" / "agent-env",
                  [project / "pyproject.toml", project / "uv.lock"], env, work, project=project,
                  _rebuild=[["uv", "sync", "-q"]])

    added = owned.cli("plugin", "add", str(toys["grader"]), "--yes")
    owned.rebuild()

    assert added.returncode == 0 and "uv project at" in added.stdout, added.stdout + added.stderr
    assert GRADER in owned.plugins()


def test_a_virtual_uv_workspace_adds_the_plugin_to_the_member_that_declares_agent_env(tmp_path, hermetic, toys):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f'[tool.uv.workspace]\nmembers = ["app"]\n\n[tool.uv]\nenvironments = ["{_platform_marker()}"]\n')
    _project(root / "app", '["agentenv-framework"]')
    _checked(["uv", "sync", "-q", "--all-packages", "--python", sys.executable], env=hermetic, cwd=root)
    venv = root / ".venv"
    owned = Setup("uv-project", venv / "bin" / "python", venv / "bin" / "agent-env",
                  [root / "pyproject.toml", root / "app" / "pyproject.toml", root / "uv.lock"], hermetic, root,
                  project=root, _rebuild=[["uv", "sync", "-q", "--all-packages"]])

    added = owned.cli("plugin", "add", str(toys["grader"]), "--yes")
    owned.rebuild()

    assert added.returncode == 0, added.stdout + added.stderr
    assert "agentenv-toy-grader" in (root / "app" / "pyproject.toml").read_text()
    assert GRADER in owned.plugins()
    assert owned.cli("plugin", "remove", GRADER, "--yes").returncode == 0
    assert "agentenv-toy-grader" not in (root / "app" / "pyproject.toml").read_text()


def test_a_uv_project_removes_a_plugin_from_the_group_that_declares_it(tmp_path, hermetic, toys):
    owned = make_setup("uv-project", tmp_path, hermetic)
    _checked(["uv", "add", "-q", "--dev", str(toys["grader"])], env=owned.env, cwd=owned.project)
    assert GRADER in owned.plugins()

    removed = owned.cli("plugin", "remove", GRADER, "--yes")
    owned.rebuild()

    assert removed.returncode == 0, removed.stdout + removed.stderr
    assert "toy-grader" not in owned.records[0].read_text() and GRADER not in owned.plugins()


def test_a_uv_workspace_removes_a_plugin_from_the_member_that_declares_it(tmp_path, hermetic, toys):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f'[tool.uv.workspace]\nmembers = ["app", "lib"]\n\n[tool.uv]\nenvironments = ["{_platform_marker()}"]\n')
    _project(root / "app", '["agentenv-framework"]')
    _project(root / "lib")
    _checked(["uv", "add", "-q", "--package", "lib", str(toys["grader"]), "--python", sys.executable],
             env=hermetic, cwd=root)
    _checked(["uv", "sync", "-q", "--all-packages", "--python", sys.executable], env=hermetic, cwd=root)
    venv = root / ".venv"
    owned = Setup("uv-project", venv / "bin" / "python", venv / "bin" / "agent-env",
                  [root / "pyproject.toml", root / "app" / "pyproject.toml", root / "lib" / "pyproject.toml",
                   root / "uv.lock"], hermetic, root, project=root, _rebuild=[["uv", "sync", "-q", "--all-packages"]])
    assert GRADER in owned.plugins()

    removed = owned.cli("plugin", "remove", GRADER, "--yes")
    owned.rebuild()

    assert removed.returncode == 0, removed.stdout + removed.stderr
    assert "toy-grader" not in (root / "lib" / "pyproject.toml").read_text() and GRADER not in owned.plugins()
