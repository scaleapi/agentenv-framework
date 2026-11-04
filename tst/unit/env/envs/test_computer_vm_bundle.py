"""A macOS computer env, composed on a Mac: its VM bundle's build folder holds the base's readying, each layer's files,
its LaunchAgents and the provision script that installs each layer as its ``[install.macos]`` says; a builder of the
base is booted on a link that boots macOS VM bundles, provisioned, and kept by a name of its key, which a compose of the
same inputs reuses; the env's computer is that VM bundle. The service runner reads the protocol's seed."""

import importlib.machinery
import importlib.util
import plistlib
import stat
from pathlib import Path

import pytest

from agentenv_protocol.computer import DriverDeclaration, seed
from agentenv_protocol.computer.app import Download, RunScript

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.utils import composer, install, vm_bundle
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.envs.computer.utils.install import Grant, MacInstall, MacService
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

BASE_IMAGE = "ghcr.io/trycua/macos:26-x@sha256:" + "7c" * 32
SERVE = "venv/bin/python -m agentenv_protocol.computer.serve driver.py:Desk"
APP = "/Applications/Desk.app"
URL = "https://example.com/desk.tar.gz"


class FakeBuilder:
    def __init__(self, mac, name, build):
        self.mac, self.sandbox_id = mac, name
        mac.provisioned.append((name, (build / "provision.sh").read_text(), {
            path.name: plistlib.loads(path.read_bytes()) for path in (build / "agents").iterdir()},
            stat.S_IMODE((build / ".password").stat().st_mode)))

    async def exec_with_output(self, *command):
        self.mac.ran.append(command)
        return self.mac.exit, "", "boom"

    async def terminate(self):
        self.mac.terminated.append(self.sandbox_id)


class FakeMac(SandboxProvider):
    """A Mac that boots macOS VM bundles, its builders recorded and kept as the bundles it holds."""

    ssh_password = "lume"

    def __init__(self):
        self.opened, self.provisioned, self.ran, self.terminated, self.kept, self.exit = [], [], [], [], set(), 0

    @classmethod
    def native_platforms(cls):
        return frozenset()

    @classmethod
    def vm_bundle_platforms(cls):
        return frozenset({"macos/arm64"})

    async def has_vm_bundle(self, image):
        return image in self.kept

    async def open_builder(self, image, name, build):
        self.opened.append((image, name))
        return FakeBuilder(self, name, build)

    async def keep_builder(self, builder):
        self.kept.add(f"local:{builder.sandbox_id}")
        return f"local:{builder.sandbox_id}"


@pytest.fixture
def mac(local_stores, monkeypatch):
    monkeypatch.setattr(composer, "_probed", lambda env, driver, form, arch, result, provider, progress: {
        **result, "status": "ready"})
    return FakeMac()


def _base() -> ComputerArtifact:
    return ComputerArtifact.put(**{
        "id": "@local/~/kit/macos-26", "arch": ["arm64"], "os": {"type": "macos", "version": "26"},
        "session": {"user": "lume"}, "config": {"resolution": {"default": "1280x800", "phase": "deploy"}},
        "forms": {"vm_bundle": {"image": BASE_IMAGE, "format": "lume", "input_channel": "shared_folder",
                                "min_disk_gb": 150}}})


def _driver(folder: Path, macos: MacInstall | None = None) -> ComputerDriverEnv:
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (("driver.py", "class Desk: ...\n"), ("requirements.txt", "mss\n"), ("install-app.sh", "true\n")):
        (folder / name).write_text(text)
    files = FileArtifactUniverse.put_bundled("@local/~/kit/desk__files", files={
        name: folder / name for name in ("driver.py", "requirements.txt", "install-app.sh")})
    macos = macos or MacInstall(
        python="requirements.txt", downloads=[Download(url=URL, sha256="ab" * 32, to="desk.tar.gz", arch="arm64")],
        run=[RunScript(script="install-app.sh")], grants=[Grant(app=APP, services=["accessibility", "screen_capture"])],
        services=[MacService(name="daemon", command=f"{APP}/Contents/MacOS/desk serve", direct=True,
                             environment={"DO_NOT_TRACK": "1"}),
                  MacService(name="driver", command=SERVE)])
    return ComputerDriverEnv.put(
        id="@local/~/kit/desk", module="driver.py:Desk", files=files, port=18765, install={"macos": macos},
        declaration=DriverDeclaration.model_validate({
            "name": "desk", "supports": [{"os": {"type": "macos", "display": ["aqua"]}, "arch": ["arm64"]}],
            "platforms": [{"platform": {"os": "macos", "arch": "arm64", "display": "aqua"}, "provides": [],
                           "tools": ["computer_screenshot"], "features": {}}]}))


def _put(driver: ComputerDriverEnv, mac: FakeMac) -> ComputerEnv:
    return ComputerEnv.put(id="@local/~/kit/desktop", computer=_base(), driver=driver, environment_name="desktop",
                           composing=Composing(sandbox_provider=mac))


def test_a_macos_env_composes_a_vm_bundle_on_a_mac_each_layer_installed_by_its_provision_script(mac, tmp_path):
    env = _put(_driver(tmp_path / "desk"), mac)

    built = env.metadata["composed"]["vm_bundle"]["arm64"]
    name = built["image"].removeprefix("local:")
    assert built["status"] == "ready" and name.startswith("agentenv-computer-") and not built["reused"]
    assert mac.opened == [(BASE_IMAGE, name)] and mac.ran == [
        ("bash", f"{vm_bundle.SHARED_FILES}/{vm_bundle.BUILD_FOLDER}/provision.sh")]
    _, script, agents, mode = mac.provisioned[0]
    prefix = "/opt/agentenv/drivers/desk"
    for line in ['bash "$build/prepare/prepare.sh"', f"curl -fsSL --retry 3 -o {prefix}/desk.tar.gz {URL}",
                 f'echo "{"ab" * 32}  "{prefix}/desk.tar.gz | shasum -a 256 -c - > /dev/null',
                 f"/opt/agentenv/bin/python3 -m venv {prefix}/venv",
                 f"{prefix}/venv/bin/pip install -q --disable-pip-version-check -r {prefix}/requirements.txt",
                 f"as_root env PREFIX={prefix} bash -c 'cd {prefix} && ./install-app.sh'",
                 f"grant {APP} accessibility screen_capture", 'cp "$build"/agents/*.plist "$HOME/Library/LaunchAgents/"']:
        assert line in script.splitlines()
    daemon, server = agents["dev.agentenv.driver-desk-daemon.plist"], agents["dev.agentenv.driver-desk-driver.plist"]
    assert daemon["ProgramArguments"] == [f"{APP}/Contents/MacOS/desk", "serve"]
    assert daemon["EnvironmentVariables"]["DO_NOT_TRACK"] == "1" and daemon["LimitLoadToSessionType"] == "Aqua"
    assert server["ProgramArguments"] == ["/opt/agentenv/bin/python3", "/opt/agentenv/bin/agentenv-service", prefix,
                                          "MCP_PORT=18765", "--", f"{prefix}/venv/bin/python", "-m",
                                          "agentenv_protocol.computer.serve", "driver.py:Desk"]
    assert "dev.agentenv.set-display.plist" in agents and mode == 0o600
    computer = env.computer_artifact()
    assert computer.os.type == "macos" and computer.forms.vm_bundle.image == built["image"]
    assert env.metadata["computer"]["vm_bundles"] == {"arm64": built["image"]}


def test_a_compose_of_the_same_inputs_reuses_the_vm_bundle_this_mac_keeps(mac, tmp_path):
    driver = _driver(tmp_path / "desk")
    first = _put(driver, mac)
    composer.compose(first, composing=Composing(sandbox_provider=mac))

    assert len(mac.opened) == 1 and first.metadata["composed"]["vm_bundle"]["arm64"]["reused"]


def test_a_failed_provisioning_deletes_its_builder_and_fails_the_pair_with_what_it_said(mac, tmp_path):
    mac.exit = 1

    env = _put(_driver(tmp_path / "desk"), mac)

    built = env.metadata["composed"]["vm_bundle"]["arm64"]
    assert built["status"] == "failed" and "provisioning exited 1: boom" in built["reason"]
    assert mac.terminated == [mac.opened[0][1]] and not mac.kept


def test_a_vm_bundle_is_composed_only_where_a_link_boots_macos_vm_bundles(local_stores, tmp_path):
    env = _put(_driver(tmp_path / "desk"), SandboxProvider())

    built = env.metadata["composed"]["vm_bundle"]["arm64"]
    assert built["status"] == "failed" and "boots macos/arm64 VM bundles" in built["reason"]


@pytest.mark.parametrize("table, problem", [
    ({"services": [{"name": "d", "command": "x"}], "packages": {"apt": ["curl"]}}, "macOS has no apt"),
    ({"services": [{"name": "d", "command": "x", "environment": {"A": "1"}}]}, "only a direct one names its own"),
    ({"services": [{"name": "d", "command": "x", "scope": "system"}]}, "services.0.scope"),
    ({"services": [{"name": "d", "command": "x"}], "grants": [{"app": "/Applications/X", "services": ["accessibility"]}]},
     "grants.0.app"),
], ids=["apt", "environment", "system", "grant-not-an-app"])
def test_a_macos_install_takes_no_apt_and_its_services_are_session_launchagents(table, problem):
    found, problems = install.installs({"macos": table})
    assert not found and any(problem in each for each in problems)


def test_a_macos_base_refuses_a_driver_that_declares_no_install_for_it(mac, tmp_path):
    driver = _driver(tmp_path / "desk")
    driver.install = {}
    with pytest.raises(ValueError, match="declares no install for macos"):
        _put(driver, mac)


def test_the_service_runner_reads_the_protocols_seed_byte_for_byte(tmp_path):
    loader = importlib.machinery.SourceFileLoader("agentenv_service", str(vm_bundle.PREPARE / "agentenv-service"))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    (tmp_path / "user-data").write_text(seed.SAMPLE_USER_DATA)

    assert module.environment(tmp_path) == seed.SAMPLE
    (tmp_path / seed.OLDEST).write_text("2\n")
    with pytest.raises(SystemExit, match="this computer reads seed v1; this agent-env writes v2 or later"):
        module.environment(tmp_path)
