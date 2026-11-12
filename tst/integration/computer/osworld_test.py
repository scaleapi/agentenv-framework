"""OSWorld on a computer env, end to end: the osworld bundle's desk, the ubuntu-24.04 base with the reference driver and
the osworld-server app staged from the guest server's source, deployed; the OSWorld controller deployed beside it by
osworld_deploy_controller, in a container of its own that reaches the guest server only through the endpoint the
computer's card advertises, with the env's control token; osworld_initialize running a classic init_config's execute
on the computer through the controller; and osworld_evaluate grading what that wrote, by the controller's own
check_include_exclude over a vm_command_line getter, 1.0 for a rule it meets and 0.0 for one it doesn't.

Needs Docker, the local registry on localhost:5000, the guest server's source (scale-cua's
``environments/linux/src/vm/setup/osworld_server``) at ``OSWORLD_SERVER_SOURCE``, and a controller image that takes a
remote base URL and headers (scale-cua #360) at ``OSWORLD_CONTROLLER_IMAGE``; skipped without the last two."""

import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.osworld import stage_server
from tst.integration.computer.conftest import run_context

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

OSWORLD = Path(__file__).parent / "osworld"
SOURCE = os.environ.get("OSWORLD_SERVER_SOURCE")
CONTROLLER_IMAGE = os.environ.get("OSWORLD_CONTROLLER_IMAGE")


@pytest.fixture
def bundle(tmp_path) -> Path:
    """The osworld bundle with the app staged from the guest server's source and the controller image named."""
    if not SOURCE or not CONTROLLER_IMAGE:
        pytest.skip("needs OSWORLD_SERVER_SOURCE and OSWORLD_CONTROLLER_IMAGE")
    root = shutil.copytree(OSWORLD, tmp_path / "osworld")
    stage_server(Path(SOURCE), root / "envs" / "osworld-server")
    (root / "artifacts" / "controller").mkdir(parents=True)
    (root / "artifacts" / "controller" / "artifact.toml").write_text(
        f'type = "docker_image"\nref = "{CONTROLLER_IMAGE}"\n')
    return root


def test_a_classic_task_is_set_up_and_graded_on_a_computer_through_the_controller_beside_it(state, bundle):
    ran = CliRunner().invoke(cli, ["run", str(bundle), "--task", "classic", "--no-emulate"])

    assert ran.exit_code == 0, ran.output
    context = run_context(ran.output)
    verifications = context["metadata"]["verifications"]
    assert verifications["pass"]["score"] == 1.0, verifications["pass"]
    assert verifications["fail"]["score"] == 0.0, verifications["fail"]
    (deployed,) = context["deployed_envs"]
    recorded = context["metadata"]["osworld_controllers"][deployed["env_id"]]
    assert recorded["osworld_guest_url"].endswith("/svc/osworld-server/osworld")
    assert recorded["osworld_controller_sandbox_id"] in {s["sandbox_id"] for s in context["deployed_sandboxes"]}
