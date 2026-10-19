"""``agent-env run`` on bundles using an agent an installed package's bundle ships, with a real docker build and the
agent deployed on the local sandbox: one bundle names it by id, another deploys it as the configured default, and the
second reuses the first's write of it. Needs a Docker daemon and the local registry."""

import importlib
import json
import logging
import shutil
import sys

import pytest
from click.testing import CliRunner

from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle.installed import BUNDLES
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.store.object_store.local.grant_server import grant_server
from tst.util.a2a_test_agent import AGENT_DIR, PROTOCOL_DIR

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

AGENT = "@local/demo-kit/kit/echo"
BASE = (AGENT_DIR / "Dockerfile").read_text().splitlines()[0]  # the echo agent's pinned base image
DOCKERFILE = f'''{BASE}
WORKDIR /app
COPY agentenv-protocol/ ./agentenv-protocol/
RUN pip install --no-cache-dir './agentenv-protocol[agent]'
COPY agent.py ./
CMD ["python", "agent.py"]
'''
NO_MODEL = {"LITELLM_API_KEY": "unused", "LITELLM_BASE_URL": "http://unused.invalid"}


def _task(**deploy):
    return json.dumps([
        {"id": "deploy", "type": "deploy_agent", "env_ids": [], "sandbox_type": "local", "env_vars": NO_MODEL,
         **deploy},
        {"id": "ask", "type": "prompt_agent", "prompt": "hello kit", "prompt_id": "ask", "depends_on": ["deploy"]},
        {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask", "verifier_id": "reply",
         "criteria": [{"type": "response_contains", "needles": ["Echo: hello kit"]}], "depends_on": ["ask"]},
    ])


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A distribution, demo-kit, on sys.path, registering the bundle ``kit`` with one agent folder, ``echo``."""
    site = tmp_path / "site"
    folder = site / "demo_kit" / "kit" / "agents" / "echo"
    folder.mkdir(parents=True)
    (site / "demo_kit" / "__init__.py").write_text("")
    (folder / "Dockerfile").write_text(DOCKERFILE)
    shutil.copy(AGENT_DIR / "agent.py", folder / "agent.py")
    shutil.copytree(PROTOCOL_DIR, folder / "agentenv-protocol",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "examples", "*.egg-info"))
    dist_info = site / "demo_kit-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: demo-kit\nVersion: 1.0\n")
    (dist_info / "entry_points.txt").write_text(f"[{BUNDLES}]\nkit = demo_kit\n")
    (dist_info / "RECORD").write_text("")
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    yield folder
    for name in [name for name in sys.modules if name.startswith("demo_kit")]:
        del sys.modules[name]


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Local stores under this test's folder. Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure(default_a2a_agent_id=AGENT)
    reset_artifact_store()
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def _bundle(root, task):
    (root / "tasks").mkdir(parents=True)
    (root / "tasks/t.json").write_text(task)
    return root


def test_bundles_naming_an_installed_agent_or_defaulting_to_it_build_it_once_and_run_it(state, installed):
    by_id = CliRunner().invoke(cli, ["run", str(_bundle(state / "by-id", _task(a2a_agent_id=AGENT)))])

    assert by_id.exit_code == 0, by_id.output
    assert "demo-kit/kit:agents/echo (Dockerfile image): building with docker" in by_id.output, by_id.output
    assert "demo-kit/kit:agents/echo: v1 (new)" in by_id.output, by_id.output
    assert "tasks/t.json v1: passed" in by_id.output, by_id.output

    by_default = CliRunner().invoke(cli, ["run", str(_bundle(state / "by-default", _task()))])

    assert by_default.exit_code == 0, by_default.output
    assert "building with docker" not in by_default.output, by_default.output
    assert "demo-kit/kit:agents/echo (Dockerfile image): v1, unchanged" in by_default.output, by_default.output
    assert "demo-kit/kit:agents/echo: v1, unchanged" in by_default.output, by_default.output
    assert "tasks/t.json v1: passed" in by_default.output, by_default.output
