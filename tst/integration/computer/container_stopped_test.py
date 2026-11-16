"""A computer's container that exits before its driver answers, booted on this machine's Docker: the boot fails within
seconds, naming the container's exit status and the last lines it wrote, rather than at the readiness timeout, and
takes the container down. Needs Docker and the local registry."""

import subprocess
import time
import uuid

import pytest

from agent_env.env.envs.computer.utils import boot
from agent_env.providers.sandbox_providers.host_images import host_platform
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider
from agent_env.utils.docker.build import build_image
from tst.integration.computer.ghost_disk import ROOTFS

pytestmark = pytest.mark.integration


@pytest.fixture
def exiting(tmp_path):
    busybox = f"agentenv-test-disk:{uuid.uuid4().hex[:12]}"
    image = f"localhost:5000/agentenv-test-exiting:{uuid.uuid4().hex[:12]}"
    build_image(ROOTFS / "Dockerfile", ROOTFS, busybox, platform=host_platform())
    (tmp_path / "Dockerfile").write_text(f'FROM {busybox}\nCMD ["sh", "-c", "echo starting the desktop; echo Xvfb: no '
                                         'screens found >&2; exit 4"]\n')
    build_image(tmp_path / "Dockerfile", tmp_path, image, platform=host_platform())
    subprocess.run(["docker", "push", "--quiet", image], capture_output=True, check=True)
    yield image
    subprocess.run(["docker", "rmi", "-f", busybox, image], capture_output=True)


class Recording(LocalSandboxProvider):
    async def create_container(self, **kwargs):
        self.made = await super().create_container(**kwargs)
        return self.made


@pytest.mark.asyncio
async def test_a_computers_container_that_exits_fails_its_boot_at_once_naming_its_status_and_last_lines(exiting):
    provider, started = Recording(), time.monotonic()

    with pytest.raises(boot.ComputerStopped) as stopped:
        await boot.boot(provider, image=exiting, port=18765, environment={}, token=boot.control_token(),
                        network_policy=None, timeout=600, attribution=None)

    assert time.monotonic() - started < 30
    said, _, written = str(stopped.value).partition("; its sandbox last wrote:\n")
    assert said.endswith("stopped before it was ready, exiting with 4")
    assert sorted(written.splitlines()) == ["  Xvfb: no screens found", "  starting the desktop"]
    left = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^{provider.made.container_name}$"],
                          capture_output=True, text=True, check=True).stdout
    assert left == ""
