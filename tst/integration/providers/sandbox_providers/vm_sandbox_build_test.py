"""A sandbox that builds as a VM does builds each stored build context as the store holds it: two versions of one image
built on one host, a changed file keeping its size and the stored context's fixed times, each build their own file.
Needs Docker."""

import asyncio
import shutil
import subprocess
import uuid

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.host_images import host_platform
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from tst.util.capabilities import missing_capability_reason

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BASE = "mirror.gcr.io/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf"


@pytest.fixture
def local_store(monkeypatch, tmp_path):
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    configure()
    reset_artifact_store()
    tags = []
    try:
        yield tags
    finally:
        for tag in tags:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
        reset_artifact_store()
        reset_config()


def test_two_versions_of_a_stored_context_built_on_one_host_each_build_their_own_files(local_store, tmp_path):
    id = f"version-{uuid.uuid4().hex[:8]}"
    images = []
    for version in ("1", "2"):
        folder = tmp_path / f"v{version}"
        folder.mkdir()
        (folder / "Dockerfile").write_text(f"FROM {BASE}\nCOPY version.txt /version.txt\n")
        (folder / "version.txt").write_text(f"version {version}\n")
        images.append(DockerImageArtifact.put_context(id, description=id, context_path=str(folder),
                                                      dockerfile_path="Dockerfile", platform=None))
    local_store.extend(image.image_name for image in images)
    assert images[0].image_name != images[1].image_name

    async def build_both():
        sandbox = await LocalSandboxProvider().create_vm(setup_for_gateway=False)
        sandbox.platform = host_platform()  # its own platform, so it builds as a VM does
        try:
            for image in images:
                await sandbox.build_images([image])
        finally:
            await sandbox.terminate()

    asyncio.run(build_both())

    built = [subprocess.run(["docker", "run", "--rm", image.image_name, "cat", "/version.txt"], capture_output=True,
                            text=True, check=True).stdout for image in images]
    assert built == ["version 1\n", "version 2\n"]
