"""``agent-env run`` on a bundle whose agents are built from their folders' Dockerfiles, with real docker builds and
the agents deployed on the local sandbox: the shapes a Dockerfile can take in an agent folder, a rerun that builds
nothing, an edit that rebuilds only its own agent, and a build that fails. Needs a Docker daemon and the local registry."""

import json
import logging
import shutil

import pytest
from click.testing import CliRunner

from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from tst.util.a2a_test_agent import AGENT_DIR, PROTOCOL_DIR

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BASE = (AGENT_DIR / "Dockerfile").read_text().splitlines()[0]  # the echo agent's pinned base image
NO_MODEL = {"LITELLM_API_KEY": "unused", "LITELLM_BASE_URL": "http://unused.invalid"}

# The echo agent, replying with what its build put in it instead of an echo: a greeting (the GREETING env var, else
# greeting.txt beside it) and which of the files a build might leave out it can see.
AGENT_PY = (AGENT_DIR / "agent.py").read_text().replace(
    'reply = f"Echo: {prompt}"',
    'greeting = os.environ.get("GREETING") or (HERE / "greeting.txt").read_text().strip()\n'
    '        seen = [name for name in ("notes.md", "secret.txt", "__pycache__") if (HERE / name).exists()]\n'
    '        reply = f"{greeting} sees {seen}"',
).replace("from typing import Any\n", "import os\nfrom pathlib import Path\nfrom typing import Any\n\nHERE = Path(__file__).parent\n")
assert "HERE = Path" in AGENT_PY and "sees {seen}" in AGENT_PY

# The steps every agent's image starts with, so the build cache shares them.
PREFIX = f'''{BASE}
WORKDIR /app
COPY agentenv-protocol/ ./agentenv-protocol/
RUN pip install --no-cache-dir './agentenv-protocol[agent]'
'''


class Link(str):
    """A symlink to this path, relative to the link."""


def _toml(**fields):
    lines = [f"{key} = {value}" for key, value in fields.items() if key != "env"]
    env = {**NO_MODEL, **fields.get("env", {})}
    return "\n".join([*lines, "[default_env_vars]", *(f'{key} = "{value}"' for key, value in env.items())]) + "\n"


AGENTS = {
    # A Dockerfile at the folder's root and no agent.toml: the deploy step passes the model variables.
    "plain": {
        "Dockerfile": PREFIX + 'COPY agent.py greeting.txt ./\nCMD ["python", "agent.py"]\n',
        "greeting.txt": "plain-v1\n",
    },
    # `COPY .`: the whole folder, less what .dockerignore names. A link is copied as its target, and what the OS or
    # Python leaves behind stays out of the build.
    "whole": {
        "Dockerfile": PREFIX + 'COPY . ./\nCMD ["python", "agent.py"]\n',
        ".dockerignore": "secret.txt\n",
        "data/greeting.txt": "whole-v1\n",
        "greeting.txt": Link("data/greeting.txt"),
        "notes.md": "kept\n",
        "secret.txt": "left out by .dockerignore\n",
        "__pycache__/agent.cpython-312.pyc": "left out of the build\n",
        ".DS_Store": "left out of the build\n",
        "agent.toml": _toml(),
    },
    # A Dockerfile in a subfolder, named by agent.toml; the build context is still the agent's folder. agent.toml's
    # env vars reach the container.
    "subdir": {
        "docker/Dockerfile": PREFIX + 'COPY agent.py ./\nCMD ["python", "agent.py"]\n',
        "agent.toml": _toml(image='{ dockerfile = "docker/Dockerfile" }', env={"GREETING": "subdir-from-toml"}),
    },
    # A multi-stage build under another name.
    "staged": {
        "Dockerfile.agent": f'{BASE} AS greeting\nRUN echo staged-v1 > /greeting.txt\n\n' + PREFIX
                            + 'COPY agent.py ./\nCOPY --from=greeting /greeting.txt ./greeting.txt\n'
                            + 'CMD ["python", "agent.py"]\n',
        "agent.toml": _toml(image='{ dockerfile = "Dockerfile.agent" }'),
    },
}

# How the run names each agent's image: its folder, and the Dockerfile it's built from.
IMAGES = {"plain": "agents/plain (Dockerfile image)", "whole": "agents/whole (Dockerfile image)",
          "subdir": "agents/subdir (docker/Dockerfile image)", "staged": "agents/staged (Dockerfile.agent image)"}

REPLIES = {
    "plain": "plain-v1 sees []",
    "whole": "whole-v1 sees ['notes.md']",
    "subdir": "subdir-from-toml sees []",
    "staged": "staged-v1 sees []",
}


def _task(name, reply):
    deploy = {"id": "deploy", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": name, "sandbox_type": "local"}
    if name == "plain":
        deploy["env_vars"] = NO_MODEL
    return json.dumps([
        deploy,
        {"id": "ask", "type": "prompt_agent", "prompt": "what did your build put in you?", "prompt_id": "ask",
         "depends_on": ["deploy"]},
        {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask", "verifier_id": "reply",
         "criteria": [{"type": "response_contains", "needles": [reply]}], "depends_on": ["ask"]},
    ])


def _agent_folder(root, name, files):
    folder = root / "agents" / name
    for path, text in {"agent.py": AGENT_PY, **files}.items():
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, Link):
            (folder / path).symlink_to(text)
        else:
            (folder / path).write_text(text)
    shutil.copytree(PROTOCOL_DIR, folder / "agentenv-protocol",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "examples", "*.egg-info"))
    return folder


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Local stores under this test's folder. Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    # HOME stays: docker's credential helper (the macOS keychain, for one) can hang a build's base-image lookup
    # when HOME moves.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        reset_artifact_store()
        reset_config()


def _run(root):
    return CliRunner().invoke(cli, ["run", str(root)])


def test_each_shape_of_agent_dockerfile_builds_runs_and_is_rebuilt_only_when_its_folder_changes(state):
    root = state / "agents-bundle"
    for name, files in AGENTS.items():
        _agent_folder(root, name, files)
        (root / "tasks").mkdir(parents=True, exist_ok=True)
        (root / f"tasks/{name}.json").write_text(_task(name, REPLIES[name]))

    first = _run(root)

    assert first.exit_code == 0, first.output
    for name in AGENTS:
        assert f"{IMAGES[name]}: building with docker" in first.output, first.output
        assert f"{IMAGES[name]}: v1 (new)" in first.output, first.output
        assert f"tasks/{name}.json v1: passed" in first.output, first.output

    again = _run(root)

    assert again.exit_code == 0, again.output
    assert "building with docker" not in again.output, again.output
    for name in AGENTS:
        assert f"{IMAGES[name]}: v1, unchanged" in again.output, again.output
        assert f"tasks/{name}.json v1: passed" in again.output, again.output

    (root / "agents/plain/greeting.txt").write_text("plain-v2\n")
    (root / "tasks/plain.json").write_text(_task("plain", "plain-v2 sees []"))
    (root / "agents/whole/__pycache__/agent.cpython-312.pyc").write_text("recompiled\n")
    (root / "agents/whole/.DS_Store").write_text("moved\n")

    edited = _run(root)

    assert edited.exit_code == 0, edited.output
    assert f"{IMAGES['plain']}: v2 (files changed: greeting.txt)" in edited.output, edited.output
    assert "agents/plain: v2 (" in edited.output, edited.output
    for name in ("whole", "subdir", "staged"):
        assert f"{IMAGES[name]}: v1, unchanged" in edited.output, edited.output
    assert "tasks/plain.json v2: passed" in edited.output, edited.output
    assert edited.output.count("building with docker") == 1, edited.output


def test_a_dockerfile_that_fails_to_build_is_one_problem_with_dockers_output_and_writes_no_agent(state):
    root = state / "broken-bundle"
    _agent_folder(root, "broken", {"Dockerfile": f"{BASE}\nRUN echo about to fail && exit 3\n"})
    (root / "tasks").mkdir()
    (root / "tasks/broken.json").write_text(_task("broken", "never"))

    result = _run(root)

    assert result.exit_code == 1, result.output
    assert "Error: agents/broken: docker build of " in result.output, result.output
    assert "about to fail" in result.output, result.output
    assert "Traceback" not in result.output, result.output
    assert "agents/broken: v1" not in result.output, result.output
