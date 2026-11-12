"""A disk its guest's init refuses, booted on a Modal VM sandbox under nested virtualization: the boot fails once the
launcher booting it exits, rather than at the readiness timeout, naming what the guest's console last said, as on this
machine's Docker; and a launcher that can't forward the driver's port fails the boot as soon as it starts. Needs Docker,
the local registry and Modal credentials, with ``[sandbox.attribution]`` tags for Modal's apps."""

import pytest

from agent_env.env.envs.computer.utils import boot
from agent_env.providers import build_sandbox_provider
from tst.integration.computer.ghost_disk import REFUSED, ghost_disk
from tst.util.capabilities import skip_without_remote_sandbox

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, skip_without_remote_sandbox("modal_vm")]


@pytest.fixture(scope="module")
def ghost(desktop, tmp_path_factory):
    yield from ghost_disk(desktop.metadata["composed"]["disk"]["amd64"]["image"], tmp_path_factory.mktemp("ghost"))


@pytest.mark.asyncio
async def test_a_disk_its_guests_init_refuses_fails_its_boot_on_a_modal_vm_once_the_launcher_exits(desktop, ghost):
    with pytest.raises(boot.ComputerStopped) as stopped:
        await boot.boot(build_sandbox_provider("modal_vm"), image=ghost, port=desktop.driver.port, environment={},
                        token=boot.control_token(), network_policy=None, timeout=1800, attribution=None, form="disk")

    assert "stopped before it was ready, exiting with 0; its sandbox last wrote:\n" in str(stopped.value)
    assert f"  {REFUSED}" in str(stopped.value).splitlines()


@pytest.mark.asyncio
async def test_a_launcher_that_cant_forward_the_drivers_port_fails_the_boot_on_a_modal_vm_saying_why(desktop):
    with pytest.raises(boot.ComputerStopped) as stopped:
        await boot.boot(build_sandbox_provider("modal_vm"), image=desktop.metadata["composed"]["disk"]["amd64"]["image"],
                        port=60000, environment={}, token=boot.control_token(), network_policy=None, timeout=1800,
                        attribution=None, form="disk")

    assert str(stopped.value).endswith(
        "exiting with 2; its sandbox last wrote:\n  launch: port 60000 can't be forwarded: QEMU's forward behind its "
        "proxy would take port 70000, which is above 65535")
