"""An agent installed into a task container (install/v1) works in the container's working directory, where its
``start.sh`` is started, as Claude Code does; one started at the root keeps its image's workspace."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import agent_env

AGENTS = Path(agent_env.__file__).parent / "agents" / "agents"


@pytest.mark.parametrize("agent, variable", [
    ("pi", "PI_A2A_WORKSPACE"), ("openhands", "OPENHANDS_A2A_WORKSPACE"), ("pydantic-ai", "PYDANTIC_AI_A2A_WORKSPACE")])
def test_an_installed_agent_works_in_the_directory_it_is_started_in_unless_that_is_the_root(tmp_path, agent, variable):
    here = tmp_path / f"{agent}-a2a"
    here.mkdir()
    # The agent's start.sh, logging into this test's folder rather than /tmp.
    (here / "start.sh").write_text((AGENTS / agent / "start.sh").read_text().replace("/tmp/", f"{tmp_path}/"))
    (here / "agent.env").write_text("")
    # In place of the server: the workspace it would be served with, written beside it.
    (here / "a2a_server.py").write_text(
        f"import os, pathlib\npathlib.Path(__file__).with_name('seen').write_text(os.environ.get({variable!r}, '-'))\n")
    task = tmp_path / "task_repo"
    task.mkdir()
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    env.pop(variable, None)

    seen = {}
    for cwd in (task, Path("/")):
        subprocess.run(["sh", str(here / "start.sh")], cwd=cwd, env=env, check=True, timeout=60)
        seen[cwd] = (here / "seen").read_text()

    assert seen == {task: os.path.realpath(task), Path("/"): "-"}
