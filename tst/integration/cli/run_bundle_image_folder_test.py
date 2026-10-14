"""``agent-env run`` on a bundle whose agents share image folders, ``artifacts/<name>/``: the echo agent, built once
from one folder's Dockerfile, runs as two agents that differ only in their agent.toml, and a third agent runs the echo
agent pushed to this machine's local registry, through a folder naming it with ``ref``. A rerun builds nothing, an
agent.toml edit rewrites only that agent, and an edit to the folder rebuilds its image once and rewrites both agents
over it. Once the ref's tag moves, a new description rewrites the ref folder over the digest it pinned. Needs a Docker
daemon and the local registry."""

import json
import logging
import shutil
import subprocess

import pytest
from click.testing import CliRunner

from agent_env.artifact import DockerImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import BundleKind, parse_bundle
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.store.image_store.registry_api import pin_digest
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.store.routing import namespace_routing
from tst.util.a2a_test_agent import AGENT_DIR, PROTOCOL_DIR, build_test_agent_image

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

NO_MODEL = 'LITELLM_API_KEY = "unused"\nLITELLM_BASE_URL = "http://unused.invalid"\n'
# The echo agent, greeting with its GREETING env var instead of "Echo".
AGENT_PY = (AGENT_DIR / "agent.py").read_text().replace(
    'reply = f"Echo: {prompt}"', 'reply = f"{os.environ[\'GREETING\']}: {prompt}"').replace(
    "from typing import Any\n", "import os\nfrom typing import Any\n")
assert "os.environ['GREETING']" in AGENT_PY and "import os\n" in AGENT_PY
GREETER = "artifacts/greeter (Dockerfile image)"


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
        # The agents moved their trajectories through this process's grant server, which serves a certificate from
        # this test's state root; a later test's agents trust another root's CA, so it must start afresh.
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def _agent(root, name, image, greeting=None):
    (root / "agents" / name).mkdir(parents=True, exist_ok=True)
    env = f'GREETING = "{greeting}"\n' if greeting else ""
    (root / f"agents/{name}/agent.toml").write_text(f'image = "{image}"\n[default_env_vars]\n{env}{NO_MODEL}')
    (root / "tasks").mkdir(exist_ok=True)
    (root / f"tasks/{name}.json").write_text(json.dumps([
        {"id": "deploy", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": name, "sandbox_type": "local"},
        {"id": "ask", "type": "prompt_agent", "prompt": "hi", "prompt_id": "ask", "depends_on": ["deploy"]},
        {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask", "verifier_id": "reply",
         "criteria": [{"type": "response_contains", "needles": [f"{greeting or 'Echo'}: hi"]}], "depends_on": ["ask"]},
    ]))


def _bundle(root, ref):
    folder = root / "artifacts/greeter"
    folder.mkdir(parents=True)
    shutil.copy(AGENT_DIR / "Dockerfile", folder / "Dockerfile")
    (folder / "agent.py").write_text(AGENT_PY)
    (folder / "artifact.toml").write_text('description = "the echo agent, greeting by GREETING"\n')
    shutil.copytree(PROTOCOL_DIR, folder / "agentenv-protocol",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "examples", "*.egg-info"))
    (root / "artifacts/echo").mkdir()
    (root / "artifacts/echo/artifact.toml").write_text(f'type = "docker_image"\nref = "{ref}"\n')
    _agent(root, "en", "greeter", "hello")
    _agent(root, "fr", "greeter", "bonjour")
    _agent(root, "echo", "echo")
    return root


def _run(root):
    return CliRunner().invoke(cli, ["run", str(root)])


def _push(source, ref, label=None):
    """Push ``source``, an image artifact, as ``ref``, or with ``label``, an image that differs from it by that
    label."""
    platform = ["--platform", source.platform] if source.platform else []
    subprocess.run(["docker", "pull", "-q", *platform, source.image_name], check=True, capture_output=True)
    if label is None:
        subprocess.run(["docker", "tag", source.image_name, ref], check=True, capture_output=True)
    else:
        subprocess.run(["docker", "build", "-q", *platform, "-t", ref, "-"], text=True, check=True,
                       input=f"FROM {source.image_name}\nLABEL moved={label}\n", capture_output=True)
    subprocess.run(["docker", "push", "-q", ref], check=True, capture_output=True)


def _pinned(root, name):
    """The image name the image folder ``name``'s latest version records."""
    entry = next(entry for entry in parse_bundle(root).entries
                 if entry.kind is BundleKind.ARTIFACT and entry.name == name)
    with namespace_routing():
        return DockerImageArtifact.get(entry.id).image_name


def _passed(output, fr=1):
    """Whether each task passed, at its version: fr's is rewritten when its greeting changes."""
    return all(f"tasks/{name}.json v{version}: passed" in output
               for name, version in (("en", 1), ("fr", fr), ("echo", 1)))


def test_agents_sharing_an_image_folder_run_its_one_build_and_an_edit_to_it_rebuilds_it_once(state):
    echo = build_test_agent_image("bundle-image-folder-echo")
    assert echo.image_name.startswith("localhost:"), echo.image_name
    # A tag this test owns, so it can move it: the echo image's own is shared by every suite that builds it.
    ref = f"{echo.image_name.split('/', 1)[0]}/bundle-image-folder/echo:moving"
    _push(echo, ref)
    root = _bundle(state / "folders-bundle", ref)

    first = _run(root)

    assert first.exit_code == 0, first.output
    assert first.output.count("building with docker") == 1, first.output
    assert f"{GREETER}: building with docker" in first.output, first.output
    assert f"{GREETER}: v1 (new)" in first.output, first.output
    assert f"artifacts/echo ({ref} image): v1 (new)" in first.output, first.output
    assert _passed(first.output), first.output

    again = _run(root)

    assert again.exit_code == 0, again.output
    assert "building with docker" not in again.output, again.output
    assert f"{GREETER}: v1, unchanged" in again.output, again.output
    assert _passed(again.output), again.output

    _agent(root, "fr", "greeter", "salut")

    retoned = _run(root)

    assert retoned.exit_code == 0, retoned.output
    assert "building with docker" not in retoned.output, retoned.output
    assert "agents/fr: v2 (" in retoned.output and "agents/en: v1, unchanged" in retoned.output, retoned.output
    assert _passed(retoned.output, fr=2), retoned.output

    (root / "artifacts/greeter/agent.py").write_text(AGENT_PY + "# rebuilt\n")

    rebuilt = _run(root)

    assert rebuilt.exit_code == 0, rebuilt.output
    assert rebuilt.output.count("building with docker") == 1, rebuilt.output
    assert f"{GREETER}: v2 (files changed: agent.py)" in rebuilt.output, rebuilt.output
    assert "agents/en: v2 (" in rebuilt.output and "agents/fr: v3 (" in rebuilt.output, rebuilt.output
    assert "agents/echo: v1, unchanged" in rebuilt.output, rebuilt.output
    assert _passed(rebuilt.output, fr=2), rebuilt.output

    pinned = _pinned(root, "echo")
    _push(echo, ref, label="1")
    assert pin_digest(ref, None) != pinned, "the tag should name another image now"
    (root / "artifacts/echo/artifact.toml").write_text(
        f'type = "docker_image"\nref = "{ref}"\ndescription = "the echo agent, by a tag that moves"\n')

    described = _run(root)

    assert described.exit_code == 0, described.output
    assert f"artifacts/echo ({ref} image): v2 (" in described.output, described.output
    assert "agents/echo: v2 (" in described.output, described.output
    assert _pinned(root, "echo") == pinned, described.output
    assert _passed(described.output, fr=2), described.output
