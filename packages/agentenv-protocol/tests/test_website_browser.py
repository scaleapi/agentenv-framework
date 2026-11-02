"""Chromium's install, shared by agentenv's chromium app and any app that starts Chromium itself: a pinned archive per
arch, its packages, the command, the install step that unpacks it keeping its execute bits and writes the policies, and
another app that inherits that step."""
import json
import os
import zipfile

from agentenv_protocol.computer import ComputerAppEnvironment, Launch, Packages, Ready, computer_app, linux
from agentenv_protocol.website_browser import (CHROMIUM_DOWNLOADS, CHROMIUM_PACKAGES, CHROMIUM_VERSION, POLICIES,
                                               WINDOW_CLASS, ChromiumInstall, chromium_command, install_chromium)


def test_each_arch_gets_its_pinned_archive_of_the_version_in_a_place_of_its_own():
    assert [(d.arch, d.to) for d in CHROMIUM_DOWNLOADS] == [("amd64", "chromium/chrome-amd64.zip"),
                                                            ("arm64", "chromium/chrome-arm64.zip")]
    assert all(f"/chrome-for-testing-public/{CHROMIUM_VERSION}/" in d.url for d in CHROMIUM_DOWNLOADS)
    assert {"libnss3", "libasound2t64", "unzip"} <= set(CHROMIUM_PACKAGES)


def test_the_command_runs_unsandboxed_without_dev_shm_first_runs_or_proxy_and_opens_the_urls_in_order(tmp_path):
    command = chromium_command(tmp_path, profile=tmp_path / "profile", urls=["https://a.test/", "https://b.test/"],
                               flags=["--disable-default-apps"])

    assert command == [str(tmp_path / "chromium/chrome/chrome"), "--no-sandbox", "--disable-dev-shm-usage",
                       "--no-first-run", "--no-default-browser-check", "--class=Chromium",
                       f"--user-data-dir={tmp_path / 'profile'}", "--disable-default-apps", "https://a.test/",
                       "https://b.test/"]
    assert not [arg for arg in command if "proxy" in arg]


def test_the_install_unpacks_the_archive_with_its_execute_bits_in_its_place_and_writes_the_policies(tmp_path):
    (tmp_path / "chromium").mkdir()
    with zipfile.ZipFile(tmp_path / "chromium" / "chrome-amd64.zip", "w") as archive:
        executable = zipfile.ZipInfo("chrome-linux64/chrome")
        executable.external_attr = 0o100755 << 16
        archive.writestr(executable, "#!/bin/sh\n")
        archive.writestr("chrome-linux64/resources.pak", "data")

    install_chromium(tmp_path, tmp_path / "policies")

    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        "chromium/chrome/chrome", "chromium/chrome/resources.pak", "policies/agentenv.json"]
    assert os.access(tmp_path / "chromium/chrome/chrome", os.X_OK)
    policies = json.loads((tmp_path / "policies" / "agentenv.json").read_text())
    assert policies == POLICIES and policies["BrowserSignin"] == 0 and not policies["CommandLineFlagSecurityWarningsEnabled"]


def test_another_app_that_starts_chromium_itself_inherits_its_install_step():
    @computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
    @linux(packages=Packages(apt=[*CHROMIUM_PACKAGES, "xdotool"]), downloads=CHROMIUM_DOWNLOADS,
           launch=Launch(command=["venv/bin/python", "desk.py"], window={"class": WINDOW_CLASS}, ready=Ready(window=True)))
    class Desk(ChromiumInstall, ComputerAppEnvironment):
        pass

    declared = Desk.declaration()

    assert [(step.name, step.scope, step.os) for step in declared.install_steps] == [("chromium", "system", ["linux"])]
    assert [d["to"] for d in declared.os["linux"].install["downloads"]] == [d.to for d in CHROMIUM_DOWNLOADS]
