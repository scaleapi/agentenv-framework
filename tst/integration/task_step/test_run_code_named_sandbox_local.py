"""run_code on a deploy_sandbox sandbox: files loaded onto it, a script that builds from them, and the result
collected off it and loaded back by step id, on a local VM-mode sandbox and on a local container-mode sandbox.

A local VM-mode sandbox maps only `/app` paths onto its work dir, and a script's own file reads are not rewritten,
so its paths are absolute host paths under the test's tmp dir. The container-mode run needs Docker and a throwaway
local registry; its paths are inside the container.
"""

import io
import json
import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import httpx

import pytest

from agent_env.artifact import DockerImageArtifact, FileArtifact, FileArtifactUniverse
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config, set_image_store
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.task import Task
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.run_code import RunCodeTaskStep

pytestmark = pytest.mark.integration

_BUILD_PY = """\
import json, pathlib, zipfile

def run(inp):
    args = inp["args"]
    src, out = pathlib.Path(args["src"]), pathlib.Path(args["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in src.iterdir())
    with zipfile.ZipFile(out, "w") as zf:
        for name in names:
            zf.write(src / name, f"bundle/{name}")
    out.chmod(0o600)
    return {"members": names}
"""

_READ_PY = """\
import pathlib, zipfile

def run(inp):
    (path,) = pathlib.Path(inp["args"]["dir"]).iterdir()
    return {"name": path.name, "members": sorted(zipfile.ZipFile(path).namelist())}
"""


@pytest.fixture
def local_backends(monkeypatch, tmp_path):
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    try:
        yield tmp_path
    finally:
        shutil.rmtree(sandboxes, ignore_errors=True)
        reset_artifact_store()
        reset_config()


def _universe(src_dir, uid, files):
    src_dir.mkdir()
    for name, body in files.items():
        (src_dir / name).write_text(body)
    return FileArtifactUniverse.put(
        id=uid,
        file_artifacts={
            name: FileArtifact.put(id=f"{uid}-{name}", description=name, file_path=str(src_dir / name))
            for name in files
        },
    )


def _chain(tmp_path, suffix, deploy, work):
    inputs = _universe(tmp_path / "inputs", f"inputs-{suffix}", {
        "task.json": json.dumps({"initial_prompt": "hi"}),
        "rubrics.json": "[]",
    })
    build = _universe(tmp_path / "build", f"build-{suffix}", {"build.py": _BUILD_PY})
    read = _universe(tmp_path / "read", f"read-{suffix}", {"read.py": _READ_PY})
    return Task.put(id=f"named-sandbox-{suffix}", steps=[
        deploy,
        LoadArtifactTaskStep(
            id="load-inputs", version=None, artifact_id=inputs.id, sandbox_name="box",
            destination_path=f"{work}/inputs",
        ),
        RunCodeTaskStep(
            id="build", version=None, script_artifact_id=build.id, sandbox_name="box",
            args={"src": f"{work}/inputs", "out": f"{work}/out/bundle.zip"},
        ),
        CollectArtifactsTaskStep(
            id="collect", version=None, sandbox_name="box", base_path=f"{work}/out", artifact_paths=["bundle.zip"],
        ),
        LoadArtifactTaskStep(
            id="load-back", version=None, artifact_from_step_id="collect", sandbox_name="box",
            destination_path=f"{work}/loaded",
        ),
        RunCodeTaskStep(
            id="read", version=None, script_artifact_id=read.id, sandbox_name="box",
            args={"dir": f"{work}/loaded"},
        ),
    ])


def _assert_round_trip(ctx):
    results = ctx.metadata["script_results"]
    assert results["build"] == {"members": ["rubrics.json", "task.json"]}
    assert results["read"] == {"name": "bundle.zip", "members": ["bundle/rubrics.json", "bundle/task.json"]}
    universe = FileArtifactUniverse.get(ctx.metadata["collected_artifacts"]["collect"]["file_artifact_universe"]["id"])
    stored = universe.get_file_artifacts()["bundle.zip"].load()
    assert sorted(zipfile.ZipFile(io.BytesIO(stored)).namelist()) == ["bundle/rubrics.json", "bundle/task.json"]
    assert Task.get_instance(ctx.instance_id).status == "completed"


@pytest.mark.asyncio
async def test_a_script_builds_on_a_named_vm_sandbox_and_its_output_round_trips(local_backends):
    tmp_path = local_backends
    deploy = DeploySandboxTaskStep(id="deploy", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local")
    ctx = await _chain(tmp_path, uuid.uuid4().hex[:8], deploy, str(tmp_path / "work")).run()
    _assert_round_trip(ctx)


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def local_registry(local_backends):
    port = _free_port()
    name = f"run-code-registry-{uuid.uuid4().hex[:8]}"
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
    set_image_store(LocalRegistryImageStore(host))
    try:
        yield local_backends
    finally:
        # Only this test's containers: every local sandbox it deployed has a work dir in its own sandbox root, and its
        # container is named after that sandbox. Matching by image could reach another run's containers.
        for work_dir in (local_backends / "sandboxes").glob("agent-env-*"):
            if m := re.match(r"agent-env-(local-[0-9a-f]+)-", work_dir.name):
                _docker("rm", "-f", f"agent-{m.group(1)}")
        _docker("rm", "-f", name)


@pytest.mark.asyncio
async def test_a_script_builds_in_a_named_container_sandbox_and_its_output_round_trips(local_registry):
    tmp_path = local_registry
    suffix = uuid.uuid4().hex[:8]
    port = _free_port()
    with tempfile.TemporaryDirectory() as d:
        # A non-root image user: the steps write as root, so collect has to read as root too.
        (Path(d) / "Dockerfile").write_text(
            f'FROM python:3.12-slim\nUSER nobody\nCMD ["python", "-m", "http.server", "{port}"]\n'
        )
        built = _docker("build", "-t", f"run-code-box-{suffix}", d)
        assert built.returncode == 0, built.stderr
    image = DockerImageArtifact.put(id=f"run-code-box-{suffix}", description="box", image_name=f"run-code-box-{suffix}")
    deploy = DeploySandboxTaskStep(
        id="deploy", version=None, sandbox_name="box", sandbox_mode="container", sandbox_type="local",
        image=image.image_name, port=port,
    )
    try:
        ctx = await _chain(tmp_path, suffix, deploy, "/work").run()
    finally:
        # The tag is unique to this run, as is its registry copy: drop them, and the containers holding them, so
        # repeated runs don't pile up images.
        refs = [r for ref in (f"run-code-box-{suffix}", f"*/run-code-box-{suffix}") for r in _docker(
            "image", "ls", "--format", "{{.Repository}}:{{.Tag}}", "--filter", f"reference={ref}",
        ).stdout.split()]
        for ref in refs:
            if containers := _docker("ps", "-aq", "--filter", f"ancestor={ref}").stdout.split():
                _docker("rm", "-f", *containers)
        if refs:
            _docker("rmi", "-f", *refs)

    (deployed,) = ctx.deployed_sandboxes
    assert deployed.sandbox_mode == "container"
    _assert_round_trip(ctx)
