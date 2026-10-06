"""Files loaded into a container sandbox are found there by verify_sandbox and collect_artifacts, on the local provider.

A local container sandbox, a deploy_sandbox one or an agent's, runs beside the host machine, so each step has to reach
into the container rather than act on the host. Every path here is one only the container has. Needs Docker and a
throwaway local registry.
"""

import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx
import pytest

from agent_env.artifact import DockerImageArtifact, FileArtifact, FileArtifactUniverse
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config, set_image_store
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.task import Task
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.verifiers.verify_sandbox import VerifySandboxTaskStep
from tst.util.a2a_test_agent import put_test_agent

pytestmark = pytest.mark.integration


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def local_registry(monkeypatch, tmp_path):
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    port = _free_port()
    name = f"files-local-registry-{uuid.uuid4().hex[:8]}"
    started = _docker("run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:5000", "registry:2")
    assert started.returncode == 0, started.stderr
    host = f"localhost:{port}"
    deadline = time.time() + 45
    while time.time() < deadline:
        try:
            if httpx.get(f"http://{host}/v2/", timeout=2).status_code in (200, 401):
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    configure()
    set_image_store(LocalRegistryImageStore(host))
    reset_artifact_store()
    try:
        yield tmp_path
    finally:
        # Only this test's containers: every local sandbox it deployed has a work dir in its own sandbox root, and its
        # container is named after that sandbox. Images are shared, so matching by image would reach other runs.
        for work_dir in sandboxes.glob("agent-env-*"):
            if m := re.match(r"agent-env-(local-[0-9a-f]+)-", work_dir.name):
                _docker("rm", "-f", f"agent-{m.group(1)}")
        _docker("rm", "-f", name)
        shutil.rmtree(sandboxes, ignore_errors=True)
        reset_artifact_store()
        reset_config()


def _probe_universe(tmp_path, suffix):
    src = tmp_path / "probe.txt"
    src.write_text("hello from the probe\n")
    return FileArtifactUniverse.put(
        id=f"probe-files-{suffix}",
        file_artifacts={"probe.txt": FileArtifact.put(id=f"probe-file-{suffix}", description="probe", file_path=str(src))},
    )


def _load_verify_collect(target: dict, universe, probe_dir: str):
    return [
        LoadArtifactTaskStep(id="load", version=None, artifact_id=universe.id, destination_path=probe_dir, **target),
        VerifySandboxTaskStep(
            id="verify", version=None, verifier_id="probe", base_dir=probe_dir, **target,
            criteria=[{"type": "probe_file_contains", "criterion": "loaded", "paths": ["probe.txt"], "expected": "hello"}],
        ),
        CollectArtifactsTaskStep(id="collect", version=None, base_path=probe_dir, artifact_paths=["probe.txt"], **target),
    ]


def _assert_found_in_the_container(ctx, probe_dir):
    assert not Path(probe_dir).exists()
    assert ctx.metadata["verifications"]["probe"]["score"] == 1.0
    universe = FileArtifactUniverse.get(ctx.metadata["collected_artifacts"]["collect"]["file_artifact_universe"]["id"])
    assert universe.get_file_artifacts()["probe.txt"].load() == b"hello from the probe\n"
    assert Task.get_instance(ctx.instance_id).status == "completed"


@pytest.mark.asyncio
async def test_a_container_mode_deploy_sandbox(local_registry):
    tmp_path = local_registry
    suffix = uuid.uuid4().hex[:8]
    port = _free_port()
    probe_dir = f"/probe-{suffix}"
    with tempfile.TemporaryDirectory() as d:
        # A non-root image user and a root-only probe dir: the steps only reach the loaded files as root.
        (Path(d) / "Dockerfile").write_text(
            f"FROM python:3.12-slim\nRUN mkdir -m 700 {probe_dir}\nUSER nobody\n"
            f'CMD ["python", "-m", "http.server", "{port}"]\n'
        )
        built = _docker("build", "-t", f"files-local-box-{suffix}", d)
        assert built.returncode == 0, built.stderr
    image = DockerImageArtifact.put(id=f"files-local-box-{suffix}", description="box", image_name=f"files-local-box-{suffix}")
    task = Task.put(id=f"files-box-{suffix}", steps=[
        DeploySandboxTaskStep(
            id="deploy", version=None, sandbox_name="box", sandbox_mode="container", sandbox_type="local",
            image=image.image_name, port=port,
        ),
        *_load_verify_collect({"sandbox_name": "box"}, _probe_universe(tmp_path, suffix), probe_dir),
    ])
    ctx = await task.run()

    _assert_found_in_the_container(ctx, probe_dir)


@pytest.mark.asyncio
async def test_a_local_agent(local_registry):
    tmp_path = local_registry
    suffix = uuid.uuid4().hex[:8]
    agent = put_test_agent(f"files-local-agent-{suffix}")
    probe_dir = f"/probe-{suffix}"
    task = Task.put(id=f"files-agent-{suffix}", steps=[
        DeployAgentTaskStep(
            id="deploy", version=None, env_ids=[], a2a_agent_id=agent.id, a2a_agent_version=agent.version,
            agent_name="solver", sandbox_type="local",
        ),
        *_load_verify_collect({"agent_name": "solver"}, _probe_universe(tmp_path, suffix), probe_dir),
    ])
    ctx = await task.run()

    _assert_found_in_the_container(ctx, probe_dir)
