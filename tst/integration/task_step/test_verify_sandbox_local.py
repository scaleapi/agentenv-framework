"""A task scores against a bare local sandbox with no Docker, model or network.

A local VM-mode sandbox is a host work dir driven by subprocesses, so this needs `bash`
and `python3` on PATH and nothing else. The agent tests start a real container in place of
an agent's, so they need Docker too, but no model.
"""

import io
import json
import shutil
import subprocess
import uuid
import zipfile

import pytest
import pytest_asyncio

from agent_env.artifact import FileArtifact, FileArtifactUniverse
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.a2a_agent.a2a_agent import DeployedA2AAgent
from agent_env.a2a_agent.store import get_a2a_agent_instance_store
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, get_config, reset_config
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.task import Task
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep,
)
from agent_env.task_step.task_steps.verifiers.verify_sandbox import VerifySandboxTaskStep
from tst.util.capabilities import missing_capability_reason

pytestmark = pytest.mark.integration

_CHECK_PY = "import pathlib, sys\nsys.exit(0 if 'hello' in pathlib.Path('hello.txt').read_text() else 1)\n"
# Stands in for an agent's image: a server that stays up, on a Debian userland (bash, GNU find), for any arch.
_AGENT_IMAGE = "mirror.gcr.io/library/nginx:1.27-bookworm"


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
        yield tmp_path, sandboxes
    finally:
        shutil.rmtree(sandboxes, ignore_errors=True)
        reset_artifact_store()
        reset_config()


@pytest_asyncio.fixture
async def local_agent(local_backends):
    """A running local container sandbox recorded as the deployed agent `agent`."""
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    sandbox = await LocalSandboxProvider().create_container(image_name=_AGENT_IMAGE, port=80, env={})
    try:
        yield sandbox, TaskStepContext(instance_id=f"agent-{uuid.uuid4().hex[:8]}", deployed_agents=[
            DeployedAgent(agent_name="agent", api_url=sandbox.tunnel_urls[80], sandbox_id=sandbox.sandbox_id,
                          sandbox_type="local"),
        ])
    finally:
        await sandbox.terminate()


def _greeting_universe(src_dir, suffix):
    src_dir.mkdir()
    (src_dir / "hello.txt").write_text("hello, world\n")
    (src_dir / "check.py").write_text(_CHECK_PY)
    return FileArtifactUniverse.put(
        id=f"greeting-{suffix}",
        file_artifacts={
            name: FileArtifact.put(id=f"greeting-{suffix}-{name}", description=name, file_path=str(src_dir / name))
            for name in ("hello.txt", "check.py")
        },
    )


def _verify(verifier_id, expected, bash_cmd):
    return VerifySandboxTaskStep(
        id=verifier_id, version=None, sandbox_name="box", base_dir="/app/greeting", verifier_id=verifier_id,
        criteria=[
            {"type": "probe_file_contains", "criterion": "greets", "paths": ["hello.txt"], "expected": expected},
            {"type": "bash_cmd_succeeds", "criterion": "check passes", "bash_cmd": bash_cmd},
        ],
    )


@pytest.mark.asyncio
async def test_verify_sandbox_scores_a_bare_local_sandbox(local_backends):
    tmp_path, sandboxes = local_backends
    suffix = uuid.uuid4().hex[:8]
    universe = _greeting_universe(tmp_path / "greeting", suffix)
    task = Task.put(id=f"hello-{suffix}", steps=[
        DeploySandboxTaskStep(id="deploy", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local"),
        LoadArtifactTaskStep(
            id="load", version=None, artifact_id=universe.id, sandbox_name="box", destination_path="/app/greeting",
        ),
        _verify("hello", "hello", "python3 check.py"),
        _verify("control", "goodbye", "python3 -c 'raise SystemExit(1)'"),
    ])

    ctx = await task.run()

    verifications = ctx.metadata["verifications"]
    assert verifications["hello"]["score"] == 1.0
    assert [r["result"] for r in verifications["hello"]["results"]] == [True, True]
    assert verifications["control"]["score"] == 0.0
    assert [r["result"] for r in verifications["control"]["results"]] == [False, False]
    (deployed,) = ctx.deployed_sandboxes
    work_dir = LocalSandbox.find_work_dir(deployed.sandbox_id)
    assert work_dir.parent == sandboxes
    assert (work_dir / "greeting" / "hello.txt").read_text() == "hello, world\n"
    assert Task.get_instance(ctx.instance_id).status == "completed"


@pytest.mark.asyncio
async def test_file_artifacts_stage_beside_each_other(local_backends):
    tmp_path, _ = local_backends
    suffix = uuid.uuid4().hex[:8]
    files = _greeting_universe(tmp_path / "greeting", suffix).get_file_artifacts()
    task = Task.put(id=f"hello-files-{suffix}", steps=[
        DeploySandboxTaskStep(id="deploy", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local"),
        *(
            LoadArtifactTaskStep(
                id=f"load-{i}", version=None, artifact_id=file.id, sandbox_name="box", destination_path="/app/greeting",
            )
            for i, file in enumerate(files.values())
        ),
        _verify("hello", "hello", "python3 check.py"),
    ])

    ctx = await task.run()

    assert ctx.metadata["verifications"]["hello"]["score"] == 1.0
    loaded = ctx.metadata["loaded_file_artifact_universes"]
    assert [(entry["artifact_type"], entry["files"]) for entry in loaded] == [("file", ["hello.txt"]), ("file", ["check.py"])]


@pytest.mark.asyncio
async def test_verify_sandbox_probes_a_local_agent_inside_its_container(local_agent):
    sandbox, context = local_agent
    _write_in_container(sandbox, "/app/out/report.txt", "done")
    step = VerifySandboxTaskStep(
        id="verify", version=None, agent_name="agent", base_dir="/app/out", verifier_id="agent",
        criteria=[
            {"type": "probe_file_exists", "criterion": "wrote it", "paths": ["report.txt"]},
            {"type": "bash_cmd_succeeds", "criterion": "runs in the container", "bash_cmd": 'test "$(uname -s)" = Linux'},
        ],
    )

    ctx = await step.execute(context)

    assert [r["result"] for r in ctx.metadata["verifications"]["agent"]["results"]] == [True, True]
    assert list(sandbox.work_dir.iterdir()) == [sandbox.work_dir / ".agent-container-mode"]


@pytest.mark.asyncio
async def test_collect_artifacts_reads_a_local_agents_file_from_its_container(local_agent):
    sandbox, context = local_agent
    token = uuid.uuid4().hex
    _write_in_container(sandbox, "/app/artifact/report.txt", token)
    step = CollectArtifactsTaskStep(id="collect", version=None, agent_name="agent", artifact_paths=["report.txt"])

    ctx = await step.execute(context)

    url = ctx.metadata["collected_artifacts"]["collect"]["artifacts"]["report.txt"]
    assert get_config().get_object_store().get(url) == f"{token}\n".encode()


@pytest.mark.asyncio
async def test_load_artifact_puts_a_universe_in_a_local_agents_container(local_agent, tmp_path):
    sandbox, context = local_agent
    instance = get_a2a_agent_instance_store().create_instance(DeployedA2AAgent(
        agent_id="agent", agent_version=1, a2a_url=sandbox.tunnel_urls[80], sandbox_id=sandbox.sandbox_id,
        agent_card={}, sandbox_type="local",
    ), 600)
    context.deployed_agents[0].instance_id = instance.instance_id
    universe = _greeting_universe(tmp_path / "greeting", uuid.uuid4().hex[:8])
    step = LoadArtifactTaskStep(
        id="load", version=None, artifact_id=universe.id, agent_name="agent", destination_path="/app/greeting",
    )

    await step.execute(context)

    shown = subprocess.run(["docker", "exec", sandbox.container_name, "cat", "/app/greeting/hello.txt"],
                           capture_output=True, text=True)
    assert (shown.returncode, shown.stdout) == (0, "hello, world\n")
    assert list(sandbox.work_dir.iterdir()) == [sandbox.work_dir / ".agent-container-mode"]


@pytest.mark.asyncio
async def test_load_artifact_stages_an_environment_payload_in_a_local_container(local_agent, tmp_path):
    sandbox, context = local_agent
    _as_deployed_container(sandbox, context)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("root/docs/readme.md", "root doc\n")
        zf.writestr("data.json", json.dumps({"files": [{"path": "notes/a.txt", "content": "from data.json\n"}]}))
    (tmp_path / "payload.zip").write_bytes(payload.getvalue())
    suffix = uuid.uuid4().hex[:8]
    artifact = EnvironmentArtifact.put(
        id=f"payload-{suffix}", environment_name="filesystem",
        file_artifact=FileArtifact.put(id=f"payload-{suffix}-zip", description="payload", file_path=str(tmp_path / "payload.zip")),
    )
    step = LoadArtifactTaskStep(
        id="load", version=None, artifact_id=artifact.id, sandbox_name="box", container_name=sandbox.container_name,
        destination_path="/app/my files",
    )

    await step.execute(context)

    for path, text in (("/app/my files/docs/readme.md", "root doc\n"), ("/app/my files/notes/a.txt", "from data.json\n")):
        shown = subprocess.run(["docker", "exec", sandbox.container_name, "cat", path], capture_output=True, text=True)
        assert (shown.returncode, shown.stdout) == (0, text)


@pytest.mark.asyncio
async def test_the_unit_tests_verifier_runs_in_a_local_container(local_agent):
    sandbox, context = local_agent
    _as_deployed_container(sandbox, context)
    _write_in_container(sandbox, "/app/data/hello.txt", "hello")
    step = RunContainerUnitTestsVerifierTaskStep(
        id="verify", version=None, sandbox_name="box", container_name=sandbox.container_name,
        setup_commands=["mkdir -p /logs && touch /logs/setup-ran"],
        command="test -f /logs/setup-ran && grep -q hello /app/data/hello.txt && echo '{\"reward\": 1}' > /logs/reward.json",
        result_paths=["/logs/reward.json"],
    )

    ctx = await step.execute(context)

    entry = ctx.metadata["verifications"]["verify"]
    assert (entry["exit_code"], entry["extracted_files"]) == (0, {"/logs/reward.json": {"reward": 1}})


def _as_deployed_container(sandbox, context):
    """Record the agent's container as a ``run_docker_container`` container on the deployed sandbox ``box``."""
    context.deployed_sandboxes.append(
        DeployedSandbox(sandbox_name="box", sandbox_id=sandbox.sandbox_id, sandbox_mode="vm", sandbox_type="local"),
    )
    context.metadata.setdefault("deployed_docker_containers", []).append(
        {"container_name": sandbox.container_name, "sandbox_name": "box", "sandbox_id": sandbox.sandbox_id},
    )


def _write_in_container(sandbox, path, text):
    """Write ``text`` at ``path`` inside the sandbox's container, as the agent would."""
    script = f"mkdir -p $(dirname {path}) && echo {text} > {path}"
    subprocess.run(["docker", "exec", sandbox.container_name, "sh", "-c", script], check=True, capture_output=True)
