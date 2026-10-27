"""Computer driver envs written from a bundle's env folders: the driver's files stored with each version, its class read
where its own dependencies aren't installed, what it declares for each platform it supports, a new version when a file
changes, refusals before any write, and a deploy that waits for a computer."""

import json

import pytest

from agent_env.bundle import BundleError
from agent_env.bundle.installed import installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.bundle.run import dry_run_bundle
from agent_env.bundle import parse_bundle
from agent_env.env import Env
from agent_env.env.envs import computer_driver
from agent_env.env.envs.computer_driver import ComputerDriverEnv
from tst.unit.bundle._support import layout, plan_of

ROOT = "@local/~/desks"
DRIVER = '''import not_installed_here
from not_installed_here.input import Keyboard

from agentenv_protocol.computer import (ComputerDriverEnvironment, Display, capture, click, computer_driver, display,
                                        drag, press, scroll, type_text)

SHIFT = Keyboard.keysym("Shift_L")
print("loading the desk driver")


@computer_driver(name="desk", supports={supports})
class Desk(ComputerDriverEnvironment):
    buttons = {buttons}

    @display
    def size(self): return Display(10, 10)

    @capture
    def grab(self): return not_installed_here.grab()

    @click
    def click_at(self, x, y, button, count, modifiers): pass

    @drag
    def drag_along(self, path, button, duration_ms, modifiers): pass

    @scroll
    def scroll_at(self, x, y, direction, amount, modifiers): pass

    @type_text
    def type_in(self, text): pass

    @press
    def press_keys(self, keys): pass
'''
LINUX = [{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}]
TOML = 'type = "computer_driver"\nmodule = "driver.py:Desk"\n'


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "desks"


def _bundle(root, toml=TOML, supports=LINUX, buttons=("left", "right"), **extra):
    return layout(root, {"envs/desk/env.toml": toml,
                         "envs/desk/driver.py": DRIVER.format(supports=supports, buttons=buttons), **extra,
                         "tasks/t.json": json.dumps([{"id": "agent", "type": "deploy_agent", "env_ids": ["desk"]}])})


def _run(root):
    return {done.write.id: (done.version, done.reused, done.reasons) for done in materialize(plan_of(root)).writes}


def _problems(root):
    with pytest.raises(BundleError) as caught:
        _run(root)
    return " ".join(caught.value.problems)


def test_a_driver_folder_is_written_with_its_files_and_what_its_class_declares(bundle_dir):
    _bundle(bundle_dir)

    first = _run(bundle_dir)
    env = Env.get(f"{ROOT}/desk")

    assert first[f"{ROOT}/desk"] == (1, False, ("new",))
    assert isinstance(env, ComputerDriverEnv) and env.module == "driver.py:Desk"
    assert sorted(env.files.file_artifact_ids) == ["driver.py"]
    declared = env.declaration
    assert declared["name"] == "desk" and declared["supports"] == LINUX
    assert [p["platform"] for p in declared["platforms"]] == [{"os": "linux", "arch": "amd64", "display": "x11"},
                                                              {"os": "linux", "arch": "arm64", "display": "x11"}]
    linux = declared["platforms"][0]
    assert "urn:agentenv:computer-exec/v1" in linux["provides"] and "urn:agentenv:computer-pointer/v1" not in linux["provides"]
    assert len(linux["tools"]) == 7 and linux["features"]["buttons"] == ["left", "right"]
    assert ComputerDriverEnv.from_dict(env.to_dict()).to_dict() == env.to_dict()


def test_a_driver_is_written_again_only_when_one_of_its_files_changes(bundle_dir):
    _bundle(bundle_dir)
    _run(bundle_dir)

    assert _run(bundle_dir)[f"{ROOT}/desk"][1] is True
    (bundle_dir / "envs/desk/driver.py").write_text(DRIVER.format(supports=LINUX, buttons=("left",)))
    rewritten = _run(bundle_dir)

    assert rewritten[f"{ROOT}/desk"] == (2, False, ("files changed: driver.py",))
    assert Env.get(f"{ROOT}/desk").declaration["platforms"][0]["features"]["buttons"] == ["left"]


@pytest.mark.parametrize("toml, supports, buttons, problem", [
    ('type = "computer_driver"\n', LINUX, ("left",), "module is required"),
    ('type = "computer_driver"\nmodule = "server.py:Desk"\n', LINUX, ("left",),
     "module 'server.py:Desk' names no \"<file>:<class>\" in this folder; it has driver.py"),
    ('type = "computer_driver"\nmodule = "driver.py:Missing"\n', LINUX, ("left",),
     "driver.py has no ComputerDriverEnvironment subclass named Missing"),
    (TOML, LINUX, ("thumb",), "buttons include left"),
    (TOML, [], ("left",), "Desk declares no platform: name them with @computer_driver(supports=[...])"),
    ('type = "computer_driver"\nmodule = "driver.py:Desk"\nimage = "x"\n', LINUX, ("left",), "unknown key 'image'"),
])
def test_a_driver_its_class_cant_be_read_from_is_refused_before_any_write(bundle_dir, toml, supports, buttons,
                                                                          problem):
    _bundle(bundle_dir, toml=toml, supports=supports, buttons=buttons)

    assert problem in _problems(bundle_dir)
    with pytest.raises(Exception):
        Env.get(f"{ROOT}/desk")


def test_a_dry_run_refuses_a_driver_folder_its_env_toml_doesnt_name_a_class_of(bundle_dir):
    _bundle(bundle_dir, toml='type = "computer_driver"\nmodule = "server.py:Desk"\n')

    with pytest.raises(BundleError) as caught:
        dry_run_bundle(bundle_dir)

    assert "module 'server.py:Desk' names no" in " ".join(caught.value.problems)


def test_a_driver_whose_class_takes_too_long_to_read_is_refused(bundle_dir, monkeypatch):
    monkeypatch.setattr(computer_driver, "DECLARE_TIMEOUT_S", 0.001)
    _bundle(bundle_dir)

    assert "reading driver.py:Desk took over 0.001 s" in _problems(bundle_dir)


@pytest.mark.asyncio
async def test_a_driver_env_doesnt_deploy_until_there_is_a_computer_to_deploy_it_in(bundle_dir):
    _bundle(bundle_dir)
    _run(bundle_dir)

    with pytest.raises(NotImplementedError, match="deploys inside a computer, and agent-env can't create a computer"):
        await Env.get(f"{ROOT}/desk").deploy()


def test_a_run_naming_the_reference_driver_writes_the_shipped_one(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    layout(tmp_path / "uses", {"tasks/t.json": json.dumps([
        {"id": "agent", "type": "deploy_agent", "env_ids": ["reference"]}])})
    plan = plan_bundle(resolve_bundle(parse_bundle(tmp_path / "uses")), installed=installed_bundles)

    written = {done.write.id for done in materialize(plan).writes}
    env = Env.get("@local/agentenv-framework/drivers/reference")

    assert "@local/agentenv-framework/drivers/reference" in written
    assert env.declaration["name"] == "reference" and len(env.declaration["platforms"]) == 2
    assert {"urn:agentenv:computer-pointer/v1", "urn:agentenv:computer-keyboard/v1",
            "urn:agentenv:computer-exec/v1"} <= set(env.declaration["platforms"][0]["provides"])
