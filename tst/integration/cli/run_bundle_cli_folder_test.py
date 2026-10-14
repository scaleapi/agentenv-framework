"""``agent-env run`` on a bundle with a CLI folder, ``artifacts/<name>/`` with ``type = "cli"``: a task deploys an MCP
server env through the gateway and the echo agent, loads the CLI into the agent and attaches it as a skill, and the
agent runs it. The CLI reads a file of its own and the gateway URL installing it wrote beside its entrypoint. The
task also builds the env's own CLI with build_mcp_cli, as the worker does, and the agent adds an item through it and
lists it back. An edit to the folder CLI's files writes it again, and the next run installs the new one. Needs a
Docker daemon and the local registry."""

import json
import logging
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.store.object_store.local.grant_server import grant_server
from tst.util.a2a_test_agent import AGENT_DIR, PROTOCOL_DIR

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

REPO = Path(__file__).resolve().parents[3]
ITEMS = REPO / "tst" / "data" / "agentenv_mcp"  # an in-memory MCP server whose card names it items
NO_MODEL = 'LITELLM_API_KEY = "unused"\nLITELLM_BASE_URL = "http://unused.invalid"\n'
ENTRYPOINT = "/opt/cli/hello/bin/hello"
ITEMS_CLI = "/opt/cli/items/bin/items"  # the CLI build_mcp_cli generates for the items env
# Prints its greeting file and the scheme of the gateway URL install_cli wrote to bin/.env.
HELLO = '''#!/bin/sh
here="$(dirname "$0")"
. "$here/.env"
echo "$(cat "$here/../lib/greeting.txt") from a bundle cli, gateway ${AGENT_ENV_GATEWAY_URL%%://*}"
'''


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
        # The agent moved its trajectory through this process's grant server, which serves a certificate from this
        # test's state root; a later test's agents trust another root's CA, so it must start afresh.
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def _task(greeting):
    return json.dumps([
        {"id": "env", "type": "deploy_env", "env_id": "items", "sandbox_type": "local"},
        {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "echo", "sandbox_type": "local"},
        {"id": "load", "type": "load_artifact", "env_id": "items", "artifact_id": "hello",
         "agent_name": "default-agent", "depends_on": ["env", "agent"]},
        {"id": "build", "type": "build_mcp_cli", "env_id": "items", "command_name": "items",
         "cli_artifact_id": "items-cli", "depends_on": ["env"]},
        {"id": "load-built", "type": "load_artifact", "env_id": "items", "artifact_id": "items-cli",
         "agent_name": "default-agent", "depends_on": ["build", "agent"]},
        {"id": "skill", "type": "add_skills", "cli_artifact_ids": ["hello"], "depends_on": ["load", "load-built"]},
        {"id": "ask", "type": "prompt_agent", "prompt_id": "ask", "depends_on": ["skill"],
         "prompt": f"run {ENTRYPOINT}\nrun {ITEMS_CLI} items-add-item --item from-cli\nrun {ITEMS_CLI} list-items\n"
                   "list-skills"},
        # The added item is quoted only in list-items' JSON, not in the prompt the agent echoes.
        {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask", "verifier_id": "ran",
         "criteria": [{"type": "response_contains",
                       "needles": [f"ran {ENTRYPOINT}: {greeting} from a bundle cli, gateway http",
                                   f"ran {ITEMS_CLI} list-items: ", '"from-cli"', "skill hello-cli: SKILL.md="]}],
         "depends_on": ["ask"]},
    ])


def _bundle(root):
    items = root / "envs/items"
    shutil.copytree(ITEMS, items, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PROTOCOL_DIR / "src" / "agentenv_protocol", items / "agentenv_protocol",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (items / "env.toml").write_text('environment_name = "items"\n')
    echo = root / "agents/echo"
    shutil.copytree(AGENT_DIR, echo, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PROTOCOL_DIR, echo / "agentenv-protocol",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "examples", "*.egg-info"))
    (echo / "agent.toml").write_text(f"[default_env_vars]\n{NO_MODEL}")
    hello = root / "artifacts/hello"
    (hello / "bin").mkdir(parents=True)
    (hello / "lib").mkdir()
    (hello / "bin/hello").write_text(HELLO)
    (hello / "lib/greeting.txt").write_text("hello\n")
    (hello / "artifact.toml").write_text('type = "cli"\ncommand_name = "hello"\nentrypoint = "bin/hello"\n')
    (root / "tasks").mkdir()
    (root / "tasks/hello.json").write_text(_task("hello"))
    return root


def _run(root):
    return CliRunner().invoke(cli, ["run", str(root)])


def test_a_cli_folder_is_installed_into_an_agent_that_runs_it_and_an_edit_installs_the_new_one(state):
    root = _bundle(state / "cli-bundle")

    first = _run(root)

    assert first.exit_code == 0, first.output
    assert "artifacts/hello: v1 (new)" in first.output, first.output
    assert "tasks/hello.json v1: passed (check: 1)" in first.output, first.output

    (root / "artifacts/hello/lib/greeting.txt").write_text("bonjour\n")
    (root / "tasks/hello.json").write_text(_task("bonjour"))

    edited = _run(root)

    assert edited.exit_code == 0, edited.output
    assert "artifacts/hello: v2 (files changed: lib/greeting.txt)" in edited.output, edited.output
    assert "tasks/hello.json v2: passed (check: 1)" in edited.output, edited.output
