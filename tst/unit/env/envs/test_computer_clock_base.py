"""The clock files of each base that follows the universe clock, ubuntu-24.04 and ubuntu-24.04-browser, agree with
what core and the driver read from agentenv_protocol.computer.clock: its Dockerfile and the root start it installs and
its init runs, and root's supervisord, name the follower, its config's folder, its status's folder and its user as the
protocol does, and the library's settings name the page and file its writer writes. A disk form's guest has the file
the driver takes for one, so it serves no sync_time there."""
from __future__ import annotations

import configparser
import importlib.util
from pathlib import Path, PurePosixPath

import pytest

import agent_env
from agentenv_protocol.computer import clock

COMPUTERS = Path(agent_env.__file__).parent / "computers"
USER = clock.FOLLOWER_USER
#: Each base that follows the clock, by its folder, and its start script, whose init is the root start.
BASES = {"ubuntu-24.04": "start-desktop", "ubuntu-24.04-browser": "start-computer"}
base = pytest.mark.parametrize("base", BASES, ids=list(BASES))


def _folder(base: str) -> Path:
    return COMPUTERS / "artifacts" / base


def _settings(base: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in (_folder(base) / "clock" / "clock.conf").read_text().splitlines()
                if "=" in line and not line.startswith("#"))


def _page_writer(base: str):
    spec = importlib.util.spec_from_file_location("clockpage_defaults", _folder(base) / "clock" / "clockpage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@base
def test_the_bases_dockerfile_installs_the_follower_and_makes_its_user_and_config_folder_as_the_protocol_names_them(
        base):
    dockerfile = (_folder(base) / "Dockerfile").read_text()
    install = (_folder(base) / "root" / "install.sh").read_text()
    config_folder = PurePosixPath(clock.CONFIG).parent

    assert f"clock/clock-follower {PurePosixPath(clock.FOLLOWER).parent}/\n" in dockerfile
    assert ("COPY root/ /usr/local/lib/agentenv-root/\n"
            "RUN sh /usr/local/lib/agentenv-root/install.sh user\n") in dockerfile
    assert f"--shell /usr/sbin/nologin {USER}\n" in install
    assert f"chown root:{USER} {config_folder} && chmod 0750 {config_folder}\n" in install


@base
def test_roots_supervisord_runs_the_follower_as_its_user_and_init_makes_its_status_folder_and_page_its_own(base):
    root = configparser.ConfigParser(interpolation=None)
    root.read_string((_folder(base) / "root.conf").read_text())
    init = (_folder(base) / "root" / "init").read_text()

    assert (root["program:clock"]["command"], root["program:clock"]["user"]) == (
        f"/usr/bin/python3 -I {clock.FOLLOWER}", USER)
    assert f"install -d -o {USER} -g {USER} -m 0755 {PurePosixPath(clock.STATUS).parent}\n" in init
    assert f"clockpage.py create {USER}\n" in init
    assert "    init)\n        exec /usr/local/lib/agentenv-root/init\n" in (_folder(base) / BASES[base]).read_text()


@base
def test_the_librarys_settings_name_the_page_and_file_the_writer_writes_beside_the_followers_status(base):
    settings, writer = _settings(base), _page_writer(base)

    assert (settings["AGENTENV_CLOCK_PAGE"], settings["FAKETIME_TIMESTAMP_FILE"]) == (writer.PAGE, writer.FILE)
    assert PurePosixPath(writer.FILE).parent == PurePosixPath(clock.STATUS).parent


def test_a_disk_guest_has_the_file_the_driver_takes_for_one():
    assert f"COPY --from=start /image.env {clock.DISK_GUEST}\n" in (
        COMPUTERS / "disk" / "linux" / "boot" / "Dockerfile").read_text()
