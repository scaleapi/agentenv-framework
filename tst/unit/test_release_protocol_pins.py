"""The files ``.github/workflows/publish-pypi.yml`` rewrites to pin agentenv-framework-protocol on a release, the
pyproject with its sed and the shipped bundles' env and artifact requirements files with ``shipped_bundles.py pin``: each
must hold exactly one protocol requirement, or the release ships an unpinned one. An agent's keeps its floor."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = (REPO / ".github" / "workflows" / "publish-pypi.yml").read_text()
_spec = importlib.util.spec_from_file_location("shipped_bundles", REPO / ".github" / "scripts" / "shipped_bundles.py")
shipped_bundles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shipped_bundles)


def test_the_pyproject_holds_the_one_quoted_protocol_requirement_the_release_pins():
    assert len(re.findall(r'"agentenv-framework-protocol[^"]*"', (REPO / "pyproject.toml").read_text())) == 1


def _repo(tmp_path: Path, requirements: dict[str, str]) -> Path:
    """A checkout whose protocol is at 1.2.3, shipping the agents bundle with a ``<kind>/<name>`` folder per entry of
    ``requirements``."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n\n[project.entry-points."agent_env.bundles"]\n'
                                             'agents = "agent_env"\n')
    (tmp_path / "packages/agentenv-protocol").mkdir(parents=True)
    (tmp_path / "packages/agentenv-protocol/pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    for folder, text in requirements.items():
        (tmp_path / "src/agent_env/agents" / folder).mkdir(parents=True)
        (tmp_path / "src/agent_env/agents" / folder / "requirements.txt").write_text(text)
    return tmp_path


def _requirements(repo: Path, folder: str) -> str:
    return (repo / "src/agent_env/agents" / folder / "requirements.txt").read_text()


def test_pin_pins_the_protocol_to_the_version_the_release_ships_keeping_its_extras(tmp_path):
    repo = _repo(tmp_path, {"envs/floor": "# a comment\nagentenv-framework-protocol[agent]>=0.1.351\nmcp>=1.25,<2\n",
                            "envs/bare": "agentenv-framework-protocol",
                            "artifacts/old": "agentenv-framework-protocol[agent]==0.1.305  # stale\n",
                            "envs/none": "agentenv-framework-protocol-tools==1\nhttpx\n"})

    pinned = shipped_bundles.pin(repo)

    assert [path.parent.name for path in pinned] == ["old", "bare", "floor"]
    assert _requirements(repo, "envs/floor") == "# a comment\nagentenv-framework-protocol[agent]==1.2.3\nmcp>=1.25,<2\n"
    assert _requirements(repo, "envs/bare") == "agentenv-framework-protocol==1.2.3"
    assert _requirements(repo, "artifacts/old") == "agentenv-framework-protocol[agent]==1.2.3\n"
    assert _requirements(repo, "envs/none") == "agentenv-framework-protocol-tools==1\nhttpx\n"


def test_an_agent_keeps_its_floor_so_its_image_is_built_only_when_its_folder_changes(tmp_path):
    floor = "agentenv-framework-protocol[agent]>=0.1.351\n"
    repo = _repo(tmp_path, {"agents/claude-code": floor, "envs/echo": "agentenv-framework-protocol[server]\n"})

    assert [path.parent.name for path in shipped_bundles.pin(repo)] == ["echo"]
    assert _requirements(repo, "agents/claude-code") == floor


def test_pin_pinned_again_changes_nothing(tmp_path):
    """Each release job pins its own checkout, and so do two pins of one."""
    repo = _repo(tmp_path, {"envs/echo": "agentenv-framework-protocol[server]>=0.1.351\n"})
    shipped_bundles.pin(repo)

    assert [path.parent.name for path in shipped_bundles.pin(repo)] == ["echo"]
    assert _requirements(repo, "envs/echo") == "agentenv-framework-protocol[server]==1.2.3\n"


@pytest.mark.parametrize("requirements", [{}, {"agents/claude-code": "agentenv-framework-protocol[agent]>=0.1.351\n"}],
                         ids=["none", "agents-only"])
def test_pin_fails_when_the_shipped_bundles_hold_no_env_or_artifact_requirements_file(tmp_path, requirements):
    with pytest.raises(ValueError, match="hold no env or artifact requirements.txt"):
        shipped_bundles.pin(_repo(tmp_path, requirements))


def test_the_release_pins_the_shipped_bundles_with_the_script():
    assert WORKFLOW.count("python3 .github/scripts/shipped_bundles.py pin") >= 1
    assert "requirements_files=$(" not in WORKFLOW


def test_each_shipped_requirements_file_the_release_pins_holds_one_protocol_line():
    pinned = {path.relative_to(REPO).as_posix(): [line for line in path.read_text().splitlines()
                                                   if line.startswith("agentenv-framework-protocol")]
              for path in shipped_bundles.requirements_files()}

    assert "src/agent_env/examples/echo-mcp/envs/echo/requirements.txt" in pinned
    assert "src/agent_env/examples/mail-mcp/envs/mail/requirements.txt" in pinned
    assert all(len(lines) == 1 for lines in pinned.values()), pinned


def test_each_shipped_agent_requires_the_protocol_by_a_floor():
    """Nothing pins an agent's at release, so an exact version there would hold its image to that protocol."""
    agents = [path for root in shipped_bundles.bundle_roots().values() for path in root.glob("agents/*/requirements.txt")]
    lines = {path.parent.name: [line for line in path.read_text().splitlines()
                                if line.startswith("agentenv-framework-protocol")] for path in agents}

    assert {"claude-code", "pi"} <= set(lines)
    assert all(len(found) == 1 and ">=" in found[0] and "==" not in found[0] for found in lines.values()), lines


def test_an_env_added_to_a_shipped_bundle_is_pinned_with_the_rest_and_an_agent_isnt(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n\n[project.entry-points."agent_env.bundles"]\n'
                                             'agents = "agent_env"\nhello = "agent_env.examples"\n')
    for folder in ("examples/hello/envs/echo", "examples/hello/envs/another", "agents/agents/claude-code"):
        (tmp_path / "src/agent_env" / folder).mkdir(parents=True)
        (tmp_path / "src/agent_env" / folder / "requirements.txt").write_text("agentenv-framework-protocol[server]\n")

    assert [path.relative_to(tmp_path).as_posix() for path in shipped_bundles.requirements_files(tmp_path)] == [
        "src/agent_env/examples/hello/envs/another/requirements.txt",
        "src/agent_env/examples/hello/envs/echo/requirements.txt",
    ]
