"""The files ``.github/workflows/publish-pypi.yml`` rewrites to pin agentenv-framework-protocol on a release: each must
hold exactly one line its sed matches, or the release fails after the protocol is already on PyPI."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = (REPO / ".github" / "workflows" / "publish-pypi.yml").read_text()
_spec = importlib.util.spec_from_file_location("shipped_bundles", REPO / ".github" / "scripts" / "shipped_bundles.py")
shipped_bundles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shipped_bundles)


def test_the_pyproject_holds_the_one_quoted_protocol_requirement_the_release_pins():
    assert len(re.findall(r'"agentenv-framework-protocol[^"]*"', (REPO / "pyproject.toml").read_text())) == 1


def test_the_release_pins_the_requirements_files_the_shipped_bundles_hold():
    assert "requirements_files=$(python .github/scripts/shipped_bundles.py)" in WORKFLOW
    assert re.search(r"^\s*for requirements in \$requirements_files; do$", WORKFLOW, re.MULTILINE)


def test_each_shipped_requirements_file_the_release_pins_holds_one_protocol_line():
    pinned = {path.relative_to(REPO).as_posix(): [line for line in path.read_text().splitlines()
                                                   if line.startswith("agentenv-framework-protocol")]
              for path in shipped_bundles.requirements_files()}

    assert {"src/agent_env/examples/echo-mcp/envs/echo/requirements.txt",
            "src/agent_env/agents/agents/claude-code/requirements.txt"} <= set(pinned)
    assert all(len(lines) == 1 for lines in pinned.values()), pinned


def test_an_agent_added_to_a_shipped_bundle_is_pinned_with_the_rest(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n\n[project.entry-points."agent_env.bundles"]\n'
                                             'agents = "agent_env"\nhello = "agent_env.examples"\n')
    (tmp_path / "src/agent_env/examples/hello/tasks").mkdir(parents=True)
    for agent in ("claude-code", "another"):
        folder = tmp_path / "src/agent_env/agents/agents" / agent
        folder.mkdir(parents=True)
        (folder / "requirements.txt").write_text("agentenv-framework-protocol[agent]\n")

    assert [path.relative_to(tmp_path).as_posix() for path in shipped_bundles.requirements_files(tmp_path)] == [
        "src/agent_env/agents/agents/another/requirements.txt",
        "src/agent_env/agents/agents/claude-code/requirements.txt",
    ]
