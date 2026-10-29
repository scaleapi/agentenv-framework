"""Computer driver envs written from a bundle's env folders: the driver's files stored with each version, its class read
where its own dependencies aren't installed, what it declares for each platform it supports, how a computer installs it
and reaches it, a new version when a file changes, refusals before any write, and a deploy that waits for a
computer."""

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
from agent_env.env.envs.computer import driver as computer_driver
from agent_env.env.envs.computer.driver import ComputerDriverEnv
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
SERVE = 'services = [{ name = "driver", command = "venv/bin/python -m agentenv_protocol.computer.serve driver.py:Desk" }]'
INSTALLED = f'{TOML}port = 19000\n\n[install.linux]\nfiles = "linux"\npython = "requirements.txt"\n{SERVE}\n'


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


def test_a_driver_is_written_with_how_a_computer_installs_it_and_the_port_it_serves_on(bundle_dir):
    _bundle(bundle_dir, toml=INSTALLED, **{"envs/desk/linux/requirements.txt": "mss>=9,<11\n"})

    _run(bundle_dir)
    env = Env.get(f"{ROOT}/desk")

    install = env.install["linux"]
    assert (env.port, install.files, install.python) == (19000, "linux", "requirements.txt")
    assert [(service.name, service.scope) for service in install.services] == [("driver", "session")]
    assert sorted(env.files.file_artifact_ids) == ["driver.py", "linux/requirements.txt"]
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
     "driver.py has no ComputerDriverEnvironment or ComputerAppEnvironment subclass named Missing"),
    (TOML, LINUX, ("thumb",), "buttons include left"),
    (TOML, [], ("left",), "Desk declares no platform: name them with @computer_driver(supports=[...])"),
    ('type = "computer_driver"\nmodule = "driver.py:Desk"\nimage = "x"\n', LINUX, ("left",), "unknown key 'image'"),
    (f"{TOML}port = 0\n", LINUX, ("left",), "port must be 1 to 65535, not 0"),
    (f'{TOML}[install.windows]\n{SERVE}\n', LINUX, ("left",),
     "install.windows: an install is for an OS type agent-env composes: linux"),
    (f'{TOML}[install.linux]\npython = "requirements.txt"\n{SERVE}\n', LINUX, ("left",),
     "install.linux.python: 'requirements.txt' isn't in this folder"),
    (f'{TOML}[install.linux]\nfiles = "linux"\n{SERVE}\n', LINUX, ("left",),
     "install.linux.files: this folder has no 'linux' folder"),
    (f'{TOML}[install.linux]\nfiles = "../elsewhere"\n{SERVE}\n', LINUX, ("left",),
     "install.linux.files: Value error, '../elsewhere' must be a path inside the layer's folder"),
    (f'{TOML}[install.linux]\nservices = []\n', LINUX, ("left",), "install.linux.services: List should have at least 1"),
    (f'{TOML}[install.linux]\nservices = [{{ name = "d", command = "x" }}, {{ name = "d", command = "y" }}]\n', LINUX,
     ("left",), "install.linux.services: Value error, names a service twice: ['d', 'd']"),
    (f'{TOML}[install.linux]\nservices = [{{ name = "d", command = "x", scope = "root" }}]\n', LINUX, ("left",),
     "install.linux.services.0.scope: Input should be 'session' or 'system'"),
], ids=["no-module", "module-file", "module-class", "button", "no-platform", "unknown-key", "port", "install-os",
        "install-python", "install-files", "install-escape", "no-service", "service-twice", "service-scope"])
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
async def test_a_driver_env_deploys_only_inside_a_computer_env(bundle_dir):
    _bundle(bundle_dir)
    _run(bundle_dir)

    with pytest.raises(NotImplementedError, match="deploys inside a computer, never on its own"):
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
    assert (env.port, env.install["linux"].python) == (18765, "requirements.txt")
    assert env.install["linux"].services[0].command.endswith("agentenv_protocol.computer.serve driver.py:ReferenceDriver")
