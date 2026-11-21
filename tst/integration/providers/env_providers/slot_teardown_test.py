"""``StackLayout.teardown_script`` against a real Docker daemon: it ends one slot's containers, volumes, networks, dir and
scratch files, and leaves a neighbouring slot whole. The script runs in a throwaway Docker CLI container that holds the
slots' dirs, so it never touches this machine's /app or /tmp. Needs Docker."""

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from agent_env.providers.env_providers.stack_layout import StackLayout
from tst.util.capabilities import missing_capability_reason

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BUSYBOX = "mirror.gcr.io/library/busybox:1.37"
DOCKER_CLI = "mirror.gcr.io/library/docker:28-cli"
COMPOSE = f"""services:
  web:
    image: {BUSYBOX}
    command: ["sleep", "3600"]
    volumes: ["data:/data", "/scratch"]
volumes:
  data: {{}}
"""
GONE, KEPT = StackLayout.for_slot(43), StackLayout.for_slot(44)


def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True).stdout


def _labelled(kind: str, layout: StackLayout) -> list[str]:
    listing = ["ps", "-aq"] if kind == "container" else [kind, "ls", "-q"]
    return _docker(*listing, "--filter", f"label=com.docker.compose.project={layout.compose_project}").split()


@pytest.fixture
def two_slots(tmp_path: Path):
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(COMPOSE)
    for layout in (GONE, KEPT):
        _docker("compose", "-p", layout.compose_project, "-f", str(compose_file), "up", "-d", "--quiet-pull")
    try:
        yield
    finally:
        for layout in (GONE, KEPT):
            subprocess.run(["docker", "compose", "-p", layout.compose_project, "-f", str(compose_file), "down", "-v"],
                           capture_output=True)


def _run_teardown_twice_beside_both_slots_files() -> str:
    files = f"{GONE.app_dir} {KEPT.app_dir} {GONE.scratch_path('pgdata')} {KEPT.scratch_path('pgdata')} /tmp/s430-other"
    script = "\n".join([
        f"mkdir -p {GONE.app_dir} {KEPT.app_dir} && touch {GONE.scratch_path('pgdata')} {KEPT.scratch_path('pgdata')} "
        "/tmp/s430-other",
        GONE.teardown_script,
        GONE.teardown_script,
        f"for f in {files}; do [ -e $f ] && echo kept $f || echo gone $f; done",
    ])
    return _docker("run", "--rm", "--name", f"slot-teardown-{uuid.uuid4().hex[:8]}",
                   "-v", "/var/run/docker.sock:/var/run/docker.sock", DOCKER_CLI, "sh", "-c", script)


def test_tearing_down_one_slot_ends_its_stack_and_files_and_leaves_its_neighbour_whole(two_slots):
    kept_before = {kind: _labelled(kind, KEPT) for kind in ("container", "volume", "network")}
    assert all(kept_before.values()) and all(_labelled(kind, GONE) for kind in kept_before)

    report = _run_teardown_twice_beside_both_slots_files()

    assert {kind: _labelled(kind, GONE) for kind in kept_before} == {"container": [], "volume": [], "network": []}
    assert {kind: _labelled(kind, KEPT) for kind in kept_before} == kept_before
    assert report.split("\n")[:-1] == [
        f"gone {GONE.app_dir}", f"kept {KEPT.app_dir}", f"gone {GONE.scratch_path('pgdata')}",
        f"kept {KEPT.scratch_path('pgdata')}", "kept /tmp/s430-other",
    ]
