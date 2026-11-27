"""run_container_unit_tests_verifier reads its result files in the container on a local sandbox: a regular file as it
is, and a link the agent planted at a result path, pointing at a file of the sandbox's host, not at all. A copy of the
link read on the host would have followed it there. Needs Docker and a pull of a small public image."""

import shutil
import subprocess
import uuid

import pytest

from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep,
)
from tst.util.capabilities import missing_capability_reason

pytestmark = pytest.mark.integration

IMAGE = "mirror.gcr.io/library/debian:bookworm-slim"


@pytest.fixture
def local_backends(monkeypatch, tmp_path):
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    configure()
    reset_artifact_store()
    try:
        yield tmp_path
    finally:
        reset_artifact_store()
        reset_config()


@pytest.mark.asyncio
async def test_a_result_file_is_read_in_the_container_and_a_link_to_a_host_file_is_refused(local_backends):
    secret = local_backends / "host-secret.txt"
    secret.write_text(f"host-only-{uuid.uuid4().hex}")
    sandbox = await LocalSandboxProvider().create_vm()
    container = sandbox.scoped_name(f"tests-{uuid.uuid4().hex[:8]}")
    try:
        subprocess.run(["docker", "run", "-d", "--name", container, IMAGE, "sleep", "infinity"], check=True,
                       capture_output=True)
        subprocess.run(["docker", "exec", container, "sh", "-c", f"mkdir -p /logs && ln -s {secret} /logs/link.json"],
                       check=True, capture_output=True)
        context = TaskStepContext(
            deployed_sandboxes=[DeployedSandbox(sandbox_name="box", sandbox_id=sandbox.sandbox_id, sandbox_mode="vm",
                                                sandbox_type="local")],
            metadata={"deployed_docker_containers": [
                {"container_name": container.removesuffix(f"-{sandbox.sandbox_id}"), "sandbox_name": "box"}]})
        step = RunContainerUnitTestsVerifierTaskStep(
            id="tests", version=None, sandbox_name="box", container_name=container.removesuffix(
                f"-{sandbox.sandbox_id}"), verifier_id="tests", command="echo '{\"n\": 1}' > /logs/result.json",
            result_paths=["/logs/result.json", "/logs/link.json"])

        await step.execute(context)

        extracted = context.metadata["verifications"]["tests"]["extracted_files"]
        assert extracted == {"/logs/result.json": {"n": 1}, "/logs/link.json": None}
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        await sandbox.terminate()
