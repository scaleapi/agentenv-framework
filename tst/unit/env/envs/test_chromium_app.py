"""agentenv's chromium app as a put reads it: the installed ``browsers`` bundle names it ``chromium``; its class declares
the pinned Chromium archive for each arch and its packages, its profile as its data, a launch matched by its window
class and started on its ``start_urls``, none by default, and its install step, which unpacks the archive keeping its
execute bits and writes the policies; and its command."""
import json
import os
import sys
import zipfile

import pytest

from agentenv_protocol.computer._load import load_class
from agentenv_protocol.computer.app import ComputerAppEnvironment

from agent_env.bundle import parse_bundle
from agent_env.bundle.installed import find_bundle


@pytest.fixture
def chromium():
    sys.modules.pop("chromium", None)
    cls = load_class(find_bundle("browsers").root / "envs" / "chromium" / "chromium.py", "Chromium",
                     ComputerAppEnvironment)
    yield sys.modules[cls.__module__]
    sys.modules.pop("chromium", None)


def test_the_browsers_bundle_names_its_chromium_app_chromium():
    bundle = find_bundle("browsers")
    parsed = parse_bundle(bundle.root, id_root=bundle.id_root)

    assert [(entry.id, entry.type) for entry in parsed.entries] == [("@local/agentenv-framework/browsers/chromium",
                                                                     "computer_app")]
    assert parsed.aliases == {"envs": {"chromium": ("chromium",)}}


def test_chromium_declares_its_archives_profile_launch_start_urls_and_install_step(chromium):
    declared = chromium.Chromium.declaration()
    linux = declared.os["linux"]

    assert [(d["arch"], d["to"]) for d in linux.install["downloads"]] == [("amd64", "chromium/chrome-amd64.zip"),
                                                                          ("arm64", "chromium/chrome-arm64.zip")]
    assert all(f"/chrome-for-testing-public/{chromium.VERSION}/" in d["url"] for d in linux.install["downloads"])
    assert {"libnss3", "libasound2t64", "unzip"} <= set(linux.install["packages"]["apt"])
    assert linux.paths == [".config/chromium"]
    assert linux.launch.command == ["venv/bin/python", "chromium.py"] and linux.launch.at == "boot"
    assert linux.launch.window == {"class": "Chromium"} and linux.launch.ready.window
    assert [(step.name, step.scope, step.os) for step in declared.install_steps] == [("chromium", "system", ["linux"])]
    assert declared.settings["properties"]["start_urls"]["default"] == []
    assert declared.supports[0]["arch"] == ["amd64", "arm64"]


def test_its_install_unpacks_the_archive_with_its_execute_bits_in_its_place_and_writes_the_policies(chromium, tmp_path):
    (tmp_path / "chromium").mkdir()
    with zipfile.ZipFile(tmp_path / "chromium" / "chrome-amd64.zip", "w") as archive:
        executable = zipfile.ZipInfo("chrome-linux64/chrome")
        executable.external_attr = 0o100755 << 16
        archive.writestr(executable, "#!/bin/sh\n")
        archive.writestr("chrome-linux64/resources.pak", "data")

    chromium.install_chromium(tmp_path, tmp_path / "policies")

    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        "chromium/chrome/chrome", "chromium/chrome/resources.pak", "policies/agentenv.json"]
    assert os.access(tmp_path / "chromium/chrome/chrome", os.X_OK)
    policies = json.loads((tmp_path / "policies" / "agentenv.json").read_text())
    assert policies == chromium.POLICIES and policies["BrowserSignin"] == 0
    assert not policies["CommandLineFlagSecurityWarningsEnabled"]


def test_its_command_runs_unsandboxed_without_dev_shm_first_runs_or_proxy_and_opens_the_urls_in_order(chromium,
                                                                                                      tmp_path):
    command = chromium.chromium_command(tmp_path, home=tmp_path / "home", urls=["https://a.test/", "https://b.test/"])

    assert command == [str(tmp_path / "chromium/chrome/chrome"), "--no-sandbox", "--disable-dev-shm-usage",
                       "--no-first-run", "--no-default-browser-check", "--class=Chromium",
                       f"--user-data-dir={tmp_path / 'home/.config/chromium'}",
                       f"--disk-cache-dir={tmp_path / 'home/.cache/chromium'}", "https://a.test/", "https://b.test/"]
    assert not [arg for arg in command if "proxy" in arg]
