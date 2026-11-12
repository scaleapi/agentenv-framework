"""The ubuntu-24.04 base computer, built from its folder: it starts as root, which runs only its init, the clock
follower as its own user and the session's supervisord, and its desktop runs as the session user with the settings a
deploy passes and comes back however often X dies; the session user can neither write, replace, read the config of nor
stop the clock, and only the session's programs read universe time, a program marked real reading real time and what it
starts universe time; a follower killed comes back; a setting it can't use stops it at start, and pushed to the local
registry it registers as a computer artifact whose image put reads back. Needs Docker and the local registry on
localhost:5000."""

import json
import platform
import subprocess
import time
import tomllib
import uuid

import pytest

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from tst.integration.computer.conftest import BASE, docker, only_session_and_clock, process_owners

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

SUPERVISOR = ["supervisorctl", "-c", "/etc/supervisor/supervisord.conf", "status"]
FOLLOWER = "/usr/local/lib/agentenv-clock/clock-follower"
PAGE_WRITER = "/usr/local/lib/agentenv-clock/clockpage.py"
ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]


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


def test_the_desktop_runs_as_the_session_user_with_the_settings_a_deploy_passes(base):
    name = f"agentenv-test-base-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, "-e", "AGENTENV_SETTING_RESOLUTION=1600x900",
            "-e", "AGENTENV_SETTING_TIMEZONE=Europe/Paris", base)
    try:
        running = _all_running(name)
        owners = process_owners(name)
        zone = _exec(name, "readlink", "/etc/localtime").strip()
        uid = _exec(name, "id", "-u", "user").strip()
        screen = _exec(name, "xdpyinfo")
        wm = _exec(name, "xprop", "-root", "_NET_SUPPORTING_WM_CHECK")
        environment = _exec(name, "sh", "-c", 'for _ in $(seq 120); do w=$(pgrep -o -x xfwm4) && break; sleep 0.25; done; '
                                              'tr "\\0" "\\n" < /proc/$w/environ')
        accessibility = _exec(name, "pgrep", "-c", "at-spi2-registr")
        folders = set(_exec(name, "ls", "/home/user").split())
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    assert {line.split()[0] for line in running.splitlines()} == {"dbus", "desktop", "xvfb"}
    assert all(" RUNNING " in line for line in running.splitlines()), running
    assert uid == "1000" and only_session_and_clock(owners)
    assert zone == "/usr/share/zoneinfo/Europe/Paris"
    assert "dimensions:    1600x900 pixels" in screen and "window id #" in wm
    assert "TZ=Europe/Paris" in environment.splitlines() and "DISPLAY=:99" in environment.splitlines()
    assert int(accessibility) == 1
    assert {"Desktop", "Documents", "Downloads", "Pictures"} <= folders


def test_the_desktop_comes_back_however_often_x_dies(base):
    name = f"agentenv-test-base-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, base)
    try:
        _all_running(name)
        for _ in range(10):
            killed = _exec(name, "pgrep", "-x", "Xvfb").strip()
            _exec(name, "pkill", "-9", "-x", "Xvfb")
            for _ in range(40):
                time.sleep(0.25)
                back = subprocess.run(["docker", "exec", name, "pgrep", "-x", "Xvfb"], capture_output=True, text=True)
                if back.stdout.strip() not in ("", killed):
                    break
            time.sleep(1.2)
        time.sleep(5)
        running = _all_running(name)
        wm = _exec(name, "xprop", "-root", "_NET_SUPPORTING_WM_CHECK")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    assert all(" RUNNING " in line for line in running.splitlines()), running
    assert "window id #" in wm


def test_the_desktop_comes_back_from_a_restart_that_killed_x_with_its_lock_naming_a_running_pid(base):
    """A restart gives pids out again, so the lock a killed X left can name a running process; here tini, pid 1."""
    name = f"agentenv-test-base-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, base)
    try:
        _all_running(name)
        # As X leaves it, X running as the session user.
        _exec_as(name, "user", "sh", "-c",
                 'printf "%10d\\n" 1 > /tmp/lock && chmod 444 /tmp/lock && mv -f /tmp/lock /tmp/.X99-lock')
        docker("restart", "-t", "0", name)
        running = _all_running(name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    assert all(" RUNNING " in line for line in running.splitlines()), running


@pytest.mark.parametrize("setting, value, error", [
    ("RESOLUTION", "big", "resolution is <width>x<height>, each 1 to 32767 pixels, not 'big'"),
    ("RESOLUTION", "40000x800", "resolution is <width>x<height>, each 1 to 32767 pixels, not '40000x800'"),
    ("RESOLUTION", "1280x0", "resolution is <width>x<height>, each 1 to 32767 pixels, not '1280x0'"),
    ("TIMEZONE", "Mars/Olympus", "timezone 'Mars/Olympus' isn't in /usr/share/zoneinfo"),
], ids=["resolution", "resolution-too-wide", "resolution-zero", "timezone"])
def test_a_setting_it_cant_use_stops_it_at_start(base, setting, value, error):
    stopped = subprocess.run(["docker", "run", "--rm", "-e", f"AGENTENV_SETTING_{setting}={value}", base],
                             capture_output=True, text=True, timeout=120)

    assert stopped.returncode == 2 and error in stopped.stderr


def test_pushed_to_a_registry_it_registers_as_a_computer_whose_image_put_reads_back(base, stores):
    docker("push", "-q", base)
    declared = tomllib.loads((BASE / "artifact.toml").read_text())
    pinned = DockerImageArtifact.put_ref("base-image", description="the base", image_name=base)

    computer = ComputerArtifact.put(id="base", arch=[ARCH], os=declared["os"], session=declared["session"],
                                    config=declared["config"],
                                    forms={"container": {**declared["forms"]["container"],
                                                         "image": {"id": pinned.id, "version": pinned.version}}})

    found = computer.metadata["validated_images"]["container"][ARCH]
    assert found["status"] == "passed", json.dumps(found)
    assert found["os_release"] == {"ID": "ubuntu", "VERSION_ID": "24.04"}
    assert found["image"] == pinned.image_name


def test_the_session_user_can_neither_change_nor_read_the_config_of_nor_stop_the_clock(base):
    name = f"agentenv-test-base-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, base)
    try:
        _all_running(name)
        follower = _follower(name)
        _exec(name, "sh", "-c", "echo '{}' > /etc/agentenv/clock/f && chown root:agentenv-clock /etc/agentenv/clock/f && "
                                "chmod 0640 /etc/agentenv/clock/f && mv /etc/agentenv/clock/f /etc/agentenv/clock/follower.json")
        attempts = {what: _as_session(name, command).returncode for what, command in {
            "write the page": "echo x > /dev/shm/agentenv-clock",
            "replace the page": "rm -f /dev/shm/agentenv-clock",
            "write the clock file": "echo '+0 x1' > /run/agentenv-clock/clock",
            "add a file beside it": "touch /run/agentenv-clock/status.json.x",
            "read the follower's config": "cat /etc/agentenv/clock/follower.json",
            "list its folder": "ls /etc/agentenv/clock",
            "kill the follower": f"kill -9 {follower}",
            "stop it through root's supervisord": "supervisorctl -c /etc/supervisor/root.conf stop clock",
            "change what's preloaded": "echo > /etc/ld.so.preload",
            "change the library's settings": "echo > /etc/agentenv/clock.conf",
        }.items()}
        page = _exec(name, "stat", "-c", "%U %a", "/dev/shm/agentenv-clock").strip()
        still = _follower(name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    assert {what: code for what, code in attempts.items() if code == 0} == {}, attempts
    assert page == "agentenv-clock 644" and still == follower


def test_only_the_sessions_programs_read_universe_time_and_a_marked_one_real_time_but_not_what_it_starts(base):
    name = f"agentenv-test-base-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, base)
    try:
        _all_running(name)
        universe = time.time() - 182 * 86400
        _exec_as(name, "agentenv-clock", "python3", PAGE_WRITER, "set", repr(universe), "1")
        read = {who: float(subprocess.run(["docker", "exec", *flags, name, *command], check=True, capture_output=True,
                                          text=True).stdout) - time.time()
                for who, flags, command in [
                    ("session", ["-u", "user"], ["date", "+%s.%N"]),
                    ("marked", ["-u", "user", "-e", "AGENTENV_REAL_TIME=1"], ["date", "+%s.%N"]),
                    ("started by a marked one", ["-u", "user", "-e", "AGENTENV_REAL_TIME=1"],
                     ["sh", "-c", "date +%s.%N"]),
                    ("root", ["-u", "0"], ["date", "+%s.%N"]),
                    ("the follower's user", ["-u", "agentenv-clock"], ["date", "+%s.%N"])]}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    back = universe - time.time()
    assert {who: abs(offset - back) < 5 for who, offset in read.items()} == {
        "session": True, "marked": False, "started by a marked one": True, "root": False, "the follower's user": False}
    assert all(abs(offset) < 5 for who, offset in read.items() if who in ("marked", "root", "the follower's user"))


def test_a_follower_killed_comes_back_within_2_s(base):
    name = f"agentenv-test-base-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, base)
    try:
        _all_running(name)
        killed = _follower(name)
        _exec(name, "kill", "-9", killed)
        start = time.monotonic()
        while (back := _follower(name)) in ("", killed) and time.monotonic() - start < 10:
            time.sleep(0.05)
        took = time.monotonic() - start
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    assert back not in ("", killed) and took < 2, took


def _follower(name: str) -> str:
    return subprocess.run(["docker", "exec", name, "pgrep", "-u", "agentenv-clock", "-f", FOLLOWER],
                          capture_output=True, text=True).stdout.strip()


def _as_session(name: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "exec", "-u", "user", name, "sh", "-c", command], capture_output=True, text=True)


def _exec_as(name: str, user: str, *argv: str) -> str:
    return subprocess.run(["docker", "exec", "-u", user, name, *argv], check=True, capture_output=True, text=True).stdout


def _all_running(name: str) -> str:
    for _ in range(60):
        running = subprocess.run(["docker", "exec", name, *SUPERVISOR], capture_output=True, text=True).stdout
        if running and all(" RUNNING " in line for line in running.splitlines()):
            break
        time.sleep(1)
    return running


def _exec(name: str, *argv: str) -> str:
    return subprocess.run(["docker", "exec", name, *argv], check=True, capture_output=True, text=True).stdout
