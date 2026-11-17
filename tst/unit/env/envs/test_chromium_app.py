"""agentenv's chromium app as a put reads it: the installed ``browsers`` bundle names it ``chromium``; its class, a website browser, declares the pinned Chrome for Testing archive for each arch and its
packages, its profile as its data, 1 GiB of shared memory, a launch of Chrome itself matched by its window
class with the flags that stop the requests this build makes of its own, and the policies that do too over
agentenv's, the base's settings, and its install step, which unpacks the archive keeping its execute bits, beside
the base's, which writes the policies."""
import json
import os
import sys
import zipfile

import pytest

from agentenv_protocol.computer._load import load_class
from agentenv_protocol.computer.app import ComputerAppEnvironment
from agentenv_protocol.website_browser import POLICIES, BrowserSettings, WebsiteBrowserEnvironment

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


def test_chromium_is_a_website_browser_that_declares_its_archives_profile_launch_and_install_step(chromium):
    declared = chromium.Chromium.declaration()
    linux = declared.os["linux"]

    assert issubclass(chromium.Chromium, WebsiteBrowserEnvironment)
    assert [(d["arch"], d["to"]) for d in linux.install["downloads"]] == [("amd64", "chromium/chrome-amd64.zip"),
                                                                          ("arm64", "chromium/chrome-arm64.zip")]
    assert all(f"/chrome-for-testing-public/{chromium.VERSION}/" in d["url"] for d in linux.install["downloads"])
    assert {"libnss3", "libasound2t64", "unzip"} <= set(linux.install["packages"]["apt"])
    assert linux.paths == [".config/chromium"] and linux.install["shm_mb"] == chromium.SHM_MB == 1024
    assert linux.launch.command == ["chromium/chrome/chrome", "--no-sandbox", "--class=Chromium",
                                    "--disable-background-networking",
                                    "--component-updater=url-source=file:///dev/null",
                                    "--disable-features=AimEnabled,PreconnectToSearch"]
    assert linux.launch.at == "boot" and linux.launch.window == {"class": "Chromium"} and linux.launch.ready.window
    assert {(step.name, step.scope) for step in declared.install_steps} == {("chromium", "system"),
                                                                           ("write_policies", "system")}
    assert declared.settings == BrowserSettings.model_json_schema()
    assert declared.supports[0]["arch"] == ["amd64", "arm64"]


def test_its_policies_stop_what_this_build_asks_for_of_its_own_beside_agentenvs_and_spellcheck_stays_blocklisted(
        chromium, monkeypatch, tmp_path):
    monkeypatch.setattr(chromium.Chromium, "policy_dir", tmp_path / "policies")

    chromium.Chromium.__new__(chromium.Chromium).write_policies()

    written = json.loads((tmp_path / "policies" / "agentenv.json").read_text())
    assert written == {**POLICIES, **chromium.QUIET_POLICIES}
    assert written["BrowserNetworkTimeQueriesEnabled"] is False and written["AutofillAddressEnabled"] is False
    assert {"en-US", "fr-FR", "de", "es-419", "pt-BR"} <= set(written["SpellcheckLanguageBlocklist"])
    # SpellcheckEnabled false would void the blocklist and still fetch a dictionary; NetworkPredictionOptions would
    # turn off Chrome's preconnects for pages, where a feature stops its search engine's alone.
    assert "SpellcheckEnabled" not in written and "NetworkPredictionOptions" not in written


def test_its_install_unpacks_the_archive_with_its_execute_bits_in_its_place(chromium, tmp_path):
    (tmp_path / "chromium").mkdir()
    with zipfile.ZipFile(tmp_path / "chromium" / "chrome-amd64.zip", "w") as archive:
        executable = zipfile.ZipInfo("chrome-linux64/chrome")
        executable.external_attr = 0o100755 << 16
        archive.writestr(executable, "#!/bin/sh\n")
        archive.writestr("chrome-linux64/resources.pak", "data")

    chromium.unpack(tmp_path)

    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        "chromium/chrome/chrome", "chromium/chrome/resources.pak"]
    assert os.access(tmp_path / "chromium/chrome/chrome", os.X_OK)
