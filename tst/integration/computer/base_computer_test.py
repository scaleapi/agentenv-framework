"""The ubuntu-24.04 base computer, built from its folder: its desktop runs as the session user with the settings a deploy
passes, a setting it can't use stops it at start, and pushed to the local registry it registers as a computer artifact
whose image put reads back. Needs Docker and the local registry on localhost:5000."""

import json
import platform
import subprocess
import time
import tomllib
import uuid
from pathlib import Path

import pytest

import agent_env
from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BASE = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04"
SUPERVISOR = ["supervisorctl", "-c", "/etc/supervisor/supervisord.conf", "status"]
ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]


@pytest.fixture(scope="module")
def image():
    tag = f"localhost:5000/agentenv-test-ubuntu-24.04:{uuid.uuid4().hex[:8]}"
    _docker("build", "-q", "-t", tag, str(BASE))
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


@pytest.fixture
def stores(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    configure()
    reset_artifact_store()
    yield
    reset_artifact_store()
    reset_config()


def test_the_desktop_runs_as_the_session_user_with_the_settings_a_deploy_passes(image):
    name = f"agentenv-test-base-{uuid.uuid4().hex[:8]}"
    _docker("run", "-d", "--name", name, "-e", "AGENTENV_SETTING_RESOLUTION=1600x900",
            "-e", "AGENTENV_SETTING_TIMEZONE=Europe/Paris", image)
    try:
        for _ in range(60):
            running = subprocess.run(["docker", "exec", name, *SUPERVISOR], capture_output=True, text=True).stdout
            if running and all(" RUNNING " in line for line in running.splitlines()):
                break
            time.sleep(1)
        users = set(_exec(name, "ps", "-eo", "user=").split())
        uid = _exec(name, "id", "-u", "user").strip()
        screen = _exec(name, "xdpyinfo")
        wm = _exec(name, "xprop", "-root", "_NET_SUPPORTING_WM_CHECK")
        environment = _exec(name, "sh", "-c", 'tr "\\0" "\\n" < /proc/$(pgrep -x xfwm4)/environ')
        accessibility = _exec(name, "pgrep", "-c", "at-spi2-registr")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    assert {line.split()[0] for line in running.splitlines()} == {"dbus", "desktop", "xvfb"}
    assert all(" RUNNING " in line for line in running.splitlines()), running
    assert users == {"user"} and uid == "1000"
    assert "dimensions:    1600x900 pixels" in screen and "window id #" in wm
    assert "TZ=Europe/Paris" in environment.splitlines() and "DISPLAY=:99" in environment.splitlines()
    assert int(accessibility) == 1


@pytest.mark.parametrize("setting, value, error", [
    ("RESOLUTION", "big", "resolution is <width>x<height>, not 'big'"),
    ("TIMEZONE", "Mars/Olympus", "timezone 'Mars/Olympus' isn't in /usr/share/zoneinfo"),
], ids=["resolution", "timezone"])
def test_a_setting_it_cant_use_stops_it_at_start(image, setting, value, error):
    stopped = subprocess.run(["docker", "run", "--rm", "-e", f"AGENTENV_SETTING_{setting}={value}", image],
                             capture_output=True, text=True, timeout=120)

    assert stopped.returncode == 2 and error in stopped.stderr


def test_pushed_to_a_registry_it_registers_as_a_computer_whose_image_put_reads_back(image, stores):
    _docker("push", "-q", image)
    declared = tomllib.loads((BASE / "artifact.toml").read_text())
    pinned = DockerImageArtifact.put_ref("base-image", description="the base", image_name=image)

    computer = ComputerArtifact.put(id="base", arch=[ARCH], os=declared["os"], session=declared["session"],
                                    config=declared["config"],
                                    forms={"container": {**declared["forms"]["container"],
                                                         "image": {"id": pinned.id, "version": pinned.version}}})

    found = computer.metadata["validated_images"]["container"][ARCH]
    assert (found["status"], found["os_release"]) == ("passed", {"ID": "ubuntu", "VERSION_ID": "24.04"}), json.dumps(found)
    assert found["image"] == pinned.image_name


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=True, capture_output=True)


def _exec(name: str, *argv: str) -> str:
    return subprocess.run(["docker", "exec", name, *argv], check=True, capture_output=True, text=True).stdout
