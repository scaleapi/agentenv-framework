"""The ubuntu-24.04 base's clock files agree with what core and the driver read from agentenv_protocol.computer.clock:
its Dockerfile, init and root's supervisord name the follower, its config's folder, its status's folder and its user as
the protocol does, and the library's settings name the page and file its writer writes. A disk form's guest has the file
the driver takes for one, so it serves no sync_time there."""
from __future__ import annotations

import configparser
import importlib.util
from pathlib import Path, PurePosixPath

import agent_env
from agentenv_protocol.computer import clock

COMPUTERS = Path(agent_env.__file__).parent / "computers"
BASE = COMPUTERS / "artifacts" / "ubuntu-24.04"
USER = clock.FOLLOWER_USER


def _settings() -> dict[str, str]:
    return dict(line.split("=", 1) for line in (BASE / "clock" / "clock.conf").read_text().splitlines()
                if "=" in line and not line.startswith("#"))


def _page_writer():
    spec = importlib.util.spec_from_file_location("clockpage_defaults", BASE / "clock" / "clockpage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_bases_dockerfile_installs_the_follower_and_makes_its_user_and_config_folder_as_the_protocol_names_them():
    dockerfile = (BASE / "Dockerfile").read_text()
    config_folder = PurePosixPath(clock.CONFIG).parent

    assert f"clock/clock-follower {PurePosixPath(clock.FOLLOWER).parent}/\n" in dockerfile
    assert f"--shell /usr/sbin/nologin {USER} \\\n" in dockerfile
    assert f"chown root:{USER} {config_folder} && chmod 0750 {config_folder}" in dockerfile


def test_roots_supervisord_runs_the_follower_as_its_user_and_init_makes_its_status_folder_and_page_its_own():
    root = configparser.ConfigParser(interpolation=None)
    root.read_string((BASE / "root.conf").read_text())
    init = (BASE / "start-desktop").read_text()

    assert (root["program:clock"]["command"], root["program:clock"]["user"]) == (
        f"/usr/bin/python3 -I {clock.FOLLOWER}", USER)
    assert f"install -d -o {USER} -g {USER} -m 0755 {PurePosixPath(clock.STATUS).parent}\n" in init
    assert f"clockpage.py create {USER}\n" in init


def test_the_librarys_settings_name_the_page_and_file_the_writer_writes_beside_the_followers_status():
    settings, writer = _settings(), _page_writer()

    assert (settings["AGENTENV_CLOCK_PAGE"], settings["FAKETIME_TIMESTAMP_FILE"]) == (writer.PAGE, writer.FILE)
    assert PurePosixPath(writer.FILE).parent == PurePosixPath(clock.STATUS).parent


def test_a_disk_guest_has_the_file_the_driver_takes_for_one():
    assert f"COPY --from=start /image.env {clock.DISK_GUEST}\n" in (
        COMPUTERS / "disk" / "linux" / "boot" / "Dockerfile").read_text()
