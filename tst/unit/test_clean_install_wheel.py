"""The wheel check in ``.github/scripts/clean_install.py``: every tracked file of each shipped bundle, and every bundle
entry point, must be in the built wheel; and the published images as the checkout makes them, which
``check_clean_install.py`` compares the installed ones' tags with."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

import agent_env
from agent_env.env import bootstrap
from tst.util.scripts import load_script

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "clean_install.py"
sys.path.insert(0, str(SCRIPT.parent))  # as running the script does, so it imports its sibling shipped_bundles
_spec = importlib.util.spec_from_file_location("clean_install", SCRIPT)
clean_install = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clean_install)

TRACKED = ["src/agent_env/examples/hello/README.md", "src/agent_env/examples/hello/tasks/hello.json"]
DECLARED = {"hello": "agent_env.examples"}
ENTRY_POINTS = "[console_scripts]\nagent-env = agent_env.cli:cli\n\n[agent_env.bundles]\nhello = agent_env.examples\n"


def _wheel(tmp_path: Path, files: list[str], entry_points: str | None = ENTRY_POINTS) -> Path:
    wheel = tmp_path / "agentenv_framework-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in files:
            archive.writestr(name, "x")
        if entry_points is not None:
            archive.writestr("agentenv_framework-1.0.dist-info/entry_points.txt", entry_points)
    return wheel


def test_a_wheel_with_every_tracked_example_and_entry_point_passes(tmp_path):
    wheel = _wheel(tmp_path, [path.removeprefix("src/") for path in TRACKED])

    assert clean_install.wheel_problems(wheel, TRACKED, DECLARED) == []


def test_each_tracked_example_file_the_wheel_leaves_out_is_named(tmp_path):
    wheel = _wheel(tmp_path, ["agent_env/examples/hello/tasks/hello.json"])

    assert clean_install.wheel_problems(wheel, TRACKED, DECLARED) == [
        "::error file=src/agent_env/examples/hello/README.md::src/agent_env/examples/hello/README.md is tracked but "
        "not in agentenv_framework-1.0-py3-none-any.whl (hatch leaves out .gitignore'd names)",
    ]


@pytest.mark.parametrize("entry_points", [None, "[console_scripts]\nagent-env = agent_env.cli:cli\n",
                                          "[agent_env.bundles]\nhello = agent_env.samples\n"],
                         ids=["no entry points", "no bundles group", "another package"])
def test_bundle_entry_points_that_differ_from_pyproject_are_named(tmp_path, entry_points):
    wheel = _wheel(tmp_path, [path.removeprefix("src/") for path in TRACKED], entry_points)

    [problem] = clean_install.wheel_problems(wheel, TRACKED, DECLARED)

    assert problem.startswith("::error file=pyproject.toml::agentenv_framework-1.0-py3-none-any.whl registers the "
                              "agent_env.bundles entry points")
    assert problem.endswith("but pyproject.toml declares {'hello': 'agent_env.examples'}")


def test_the_gate_checks_the_folder_of_every_shipped_bundle():
    roots = {name: root.relative_to(SCRIPT.parents[2]).as_posix() for name, root in clean_install.bundle_roots().items()}

    assert roots == {"advance-to-next-event": "src/agent_env/examples/advance-to-next-event",
                     "agents": "src/agent_env/agents", "browsers": "src/agent_env/browsers",
                     "computers": "src/agent_env/computers",
                     "desktop": "src/agent_env/examples/desktop", "drivers": "src/agent_env/drivers",
                     "echo-mcp": "src/agent_env/examples/echo-mcp", "hello": "src/agent_env/examples/hello",
                     "mail-mcp": "src/agent_env/examples/mail-mcp",
                     "next-env-event": "src/agent_env/examples/next-env-event"}


def test_the_checkout_makes_each_published_image_as_the_installed_sources_do_when_they_are_its_own(monkeypatch):
    """Every field of an image is kept in the checkout's copy, its build args too, so the tags differ only when the
    sources do."""
    check = load_script("check_clean_install.py")
    monkeypatch.setattr(check, "CHECKOUT_SRC", Path(agent_env.__file__).parent.parent)

    assert [check._checkout(image) for image in bootstrap.PUBLISHED_IMAGES] == list(bootstrap.PUBLISHED_IMAGES)
    assert any(image.build_args for image in bootstrap.PUBLISHED_IMAGES)
