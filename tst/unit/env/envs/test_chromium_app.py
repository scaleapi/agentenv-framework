"""agentenv's chromium app as a put reads it: the installed ``browsers`` bundle names it ``chromium``, and its class declares
the pinned Chromium archives and packages, its profile as its data, a launch matched by its window class and started on
its ``start_urls``, none by default, and Chromium's install step."""
import sys

from agentenv_protocol.computer._load import load_class
from agentenv_protocol.computer.app import ComputerAppEnvironment
from agentenv_protocol.website_browser import CHROMIUM_DOWNLOADS, CHROMIUM_PACKAGES

from agent_env.bundle import parse_bundle
from agent_env.bundle.installed import find_bundle


def test_the_browsers_bundle_names_its_chromium_app_chromium():
    bundle = find_bundle("browsers")
    parsed = parse_bundle(bundle.root, id_root=bundle.id_root)

    assert [(entry.id, entry.type) for entry in parsed.entries] == [("@local/agentenv-framework/browsers/chromium",
                                                                     "computer_app")]
    assert parsed.aliases == {"envs": {"chromium": ("chromium",)}}


def test_chromium_declares_its_archives_profile_launch_start_urls_and_install_step():
    sys.modules.pop("chromium", None)
    chromium = load_class(find_bundle("browsers").root / "envs" / "chromium" / "chromium.py", "Chromium",
                          ComputerAppEnvironment)

    declared = chromium.declaration()
    linux = declared.os["linux"]

    assert linux.install["downloads"] == [d.model_dump(mode="json", exclude_none=True) for d in CHROMIUM_DOWNLOADS]
    assert linux.install["packages"]["apt"] == CHROMIUM_PACKAGES
    assert linux.paths == [".config/chromium"]
    assert linux.launch.command == ["venv/bin/python", "chromium.py"] and linux.launch.at == "boot"
    assert linux.launch.window == {"class": "Chromium"} and linux.launch.ready.window
    assert [(step.name, step.scope) for step in declared.install_steps] == [("chromium", "system")]
    assert declared.settings["properties"]["start_urls"]["default"] == []
    assert declared.supports[0]["arch"] == ["amd64", "arm64"]
