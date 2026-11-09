"""A macOS computer env, composed on a Mac: its VM bundle's build folder holds the base's readying, each layer's files,
its LaunchAgents and the provision script that installs each layer as its ``[install.macos]`` says; a builder of the
base is booted on a link that boots macOS VM bundles, provisioned, and kept by a name of its key, which a compose of the
same inputs reuses; the env's computer is that VM bundle. Deploys on a Mac that lacks it compose it there once and
probe it. The service runner reads the protocol's seed."""

import importlib.machinery
import importlib.util
import json
import logging
import plistlib
import re
import shlex
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentenv_protocol.computer import DriverDeclaration, seed
from agentenv_protocol.computer.app import Download, RunScript

from agent_env import providers
from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.cli import cli
from agent_env.env.env import Env
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.utils import composer, install, macos_vm_bundle
from agent_env.env.envs.computer.utils import params as computer_params
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.envs.computer.utils.install import Grant, MacInstall, MacService
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep

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
        mac.settings.append(json.loads((build / "build-settings.json").read_text()))
        mac.staged.append(sorted(path.relative_to(build).as_posix() for path in build.rglob("*.whl")))

    async def exec_with_output(self, *command):
        self.mac.ran.append(command)
        return self.mac.exit, "", "boom"

    async def terminate(self):
        self.mac.terminated.append(self.sandbox_id)


class FakeResolver:
    """A VM resolving a VM bundle's wheels: one wheel into each folder its script builds wheels into."""

    def __init__(self, mac, name, build):
        self.mac, self.sandbox_id, self.build = mac, name, build

    async def exec_with_output(self, *command):
        script = (self.build / "resolve.sh").read_text()
        self.mac.resolve_scripts.append(script)
        for staged in re.findall(r'--wheel-dir "\$build/([^"]+)"', script):
            (self.build / staged).mkdir(parents=True, exist_ok=True)
            (self.build / staged / "mss-10.1.0-py3-none-any.whl").write_bytes(b"wheel of " + staged.encode())
        return self.mac.resolve_exit, "", "no matching distribution"

    async def terminate(self):
        self.mac.resolve_terminated.append(self.sandbox_id)


class FakeMac(SandboxProvider):
    """A Mac that boots macOS VM bundles, its builders recorded and kept as the bundles it holds."""

    ssh_password = "lume"

    def __init__(self):
        self.opened, self.provisioned, self.ran, self.terminated, self.kept, self.exit = [], [], [], [], set(), 0
        self.settings, self.keep_error, self.delay = [], None, 0
        self.resolves, self.resolve_scripts, self.resolve_terminated, self.resolve_exit, self.staged = [], [], [], 0, []

    @classmethod
    def native_platforms(cls):
        return frozenset()

    @classmethod
    def vm_bundle_platforms(cls):
        return frozenset({"macos/arm64"})

    async def has_vm_bundle(self, image):
        return image in self.kept

    async def delete_vm_bundle(self, image):
        self.kept.discard(image)

    async def open_builder(self, image, name, build):
        time.sleep(self.delay)
        if name.startswith("agentenv-resolve-"):
            self.resolves.append((image, name))
            return FakeResolver(self, name, build)
        self.opened.append((image, name))
        return FakeBuilder(self, name, build)

    async def keep_builder(self, builder):
        if self.keep_error is not None:
            raise self.keep_error
        self.kept.add(f"local:{builder.sandbox_id}")
        return f"local:{builder.sandbox_id}"


@pytest.fixture
def mac(local_stores, cli_routing, monkeypatch):
    monkeypatch.setattr(composer, "_probed", lambda env, driver, form, arch, result, provider, progress: {
        **result, "status": "ready"})
    return FakeMac()


def _base() -> ComputerArtifact:
    return ComputerArtifact.put(**{
        "id": "@local/~/kit/macos-26", "arch": ["arm64"], "os": {"type": "macos", "version": "26"},
        "session": {"user": "lume"}, "config": {"resolution": {"default": "1280x800", "phase": "deploy"},
                                                "theme": {"default": "light", "phase": "build"}},
        "forms": {"vm_bundle": {"image": BASE_IMAGE, "format": "lume", "input_channel": "shared_folder",
                                "min_disk_gb": 150}}})


def _driver(folder: Path, macos: MacInstall | None = None, daemon: dict[str, str] | None = None,
            requirements: str = "mss\n") -> ComputerDriverEnv:
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (("driver.py", "class Desk: ...\n"), ("requirements.txt", requirements),
                       ("install-app.sh", "true\n")):
        (folder / name).write_text(text)
    files = FileArtifactUniverse.put_bundled("@local/~/kit/desk__files", files={
        name: folder / name for name in ("driver.py", "requirements.txt", "install-app.sh")})
    macos = macos or MacInstall(
        python="requirements.txt", downloads=[Download(url=URL, sha256="ab" * 32, to="desk.tar.gz", arch="arm64")],
        run=[RunScript(script="install-app.sh")], grants=[Grant(app=APP, services=["accessibility", "screen_capture"]),
                                                          Grant(executable="venv/bin/python", services=["post_event"])],
        services=[MacService(name="daemon", command=f"{APP}/Contents/MacOS/desk serve", direct=True,
                             environment=daemon or {"DO_NOT_TRACK": "1"}),
                  MacService(name="driver", command=SERVE)])
    put = ComputerDriverEnv.put(
        id="@local/~/kit/desk", module="driver.py:Desk", files=files, port=18765, install={"macos": macos},
        declaration=DriverDeclaration.model_validate({
            "name": "desk", "supports": [{"os": {"type": "macos", "display": ["aqua"]}, "arch": ["arm64"]}],
            "platforms": [{"platform": {"os": "macos", "arch": "arm64", "display": "aqua"}, "provides": [],
                           "tools": ["computer_screenshot"], "features": {}}]}))
    return Env.get(put.id, put.version)


def _put(driver: ComputerDriverEnv, mac: FakeMac, **settings) -> ComputerEnv:
    return ComputerEnv.put(id="@local/~/kit/desktop", computer=_base(), driver=driver, environment_name="desktop",
                           composing=Composing(sandbox_provider=mac), computer_settings=settings or None)


def test_a_macos_env_composes_a_vm_bundle_on_a_mac_each_layer_installed_by_its_provision_script(mac, tmp_path):
    env = _put(_driver(tmp_path / "desk"), mac)

    built = env.metadata["composed"]["vm_bundle"]["arm64"]
    name = built["image"].removeprefix("local:")
    assert built["status"] == "ready" and name.startswith("agentenv-computer-") and not built["reused"]
    assert mac.opened == [(BASE_IMAGE, name)] and mac.ran == [
        ("bash", f"{macos_vm_bundle.SHARED_FILES}/{macos_vm_bundle.BUILD_FOLDER}/provision.sh")]
    _, script, agents, mode = mac.provisioned[0]
    prefix = "/opt/agentenv/drivers/desk"
    for line in ['bash "$build/prepare/prepare.sh"', f"curl -fsSL --retry 3 -o {prefix}/desk.tar.gz {URL}",
                 f'echo "{"ab" * 32}  "{prefix}/desk.tar.gz | shasum -a 256 -c - > /dev/null',
                 f"/opt/agentenv/python/bin/python3 -m venv --copies {prefix}/venv",
                 f"as_root env PREFIX={prefix} bash -c 'cd {prefix} && ./install-app.sh'",
                 f"grant {APP} accessibility screen_capture", f"grant_executable {prefix}/venv/bin/python post_event",
                 'cp "$build"/agents/*.plist "$HOME/Library/LaunchAgents/"',
                 'as_root cp "$build/image.json" "$build/build-settings.json" /etc/agentenv/', "sync"]:
        assert line in script.splitlines()
    wheels = "packages/wheels/driver"
    assert (f'{prefix}/venv/bin/pip install -q --disable-pip-version-check --no-index --no-deps "$build/{wheels}/"'
            "mss-10.1.0-py3-none-any.whl") in script.splitlines()
    assert mac.staged == [[f"{wheels}/mss-10.1.0-py3-none-any.whl"]]
    (resolver,) = mac.resolve_scripts
    assert mac.resolves == [(BASE_IMAGE, mac.resolve_terminated[0])] and "pinned_python" in resolver.splitlines()
    assert any(f'--wheel-dir "$build/{wheels}" -r "$build/context/driver/"requirements.txt "$build/context/protocol"'
               in line for line in resolver.splitlines())
    daemon, server = agents["dev.agentenv.driver-desk-daemon.plist"], agents["dev.agentenv.driver-desk-driver.plist"]
    assert daemon["ProgramArguments"] == [f"{APP}/Contents/MacOS/desk", "serve"]
    assert daemon["EnvironmentVariables"]["DO_NOT_TRACK"] == "1" and daemon["LimitLoadToSessionType"] == "Aqua"
    assert server["ProgramArguments"] == [f"{prefix}/venv/bin/python", "/opt/agentenv/bin/agentenv-service", prefix,
                                          "MCP_PORT=18765", "--", f"{prefix}/venv/bin/python", "-m",
                                          "agentenv_protocol.computer.serve", "driver.py:Desk"]
    assert "dev.agentenv.set-display.plist" in agents and mode == 0o600
    computer = env.computer_artifact()
    assert computer.os.type == "macos" and computer.forms.vm_bundle.image == built["image"]
    assert env.metadata["computer"]["vm_bundles"] == {"arm64": built["image"]}
    spaced = MacInstall(run=[RunScript(script="install app.sh")], services=[MacService(name="app", command="app")])
    ran = next(line for line in macos_vm_bundle._installed(macos_vm_bundle.Part("x", "/opt/x", spaced, {}, {}, False))
               if "PREFIX=" in line)
    assert shlex.split(shlex.split(ran)[-1])[-1] == "./install app.sh"


def test_a_compose_of_the_same_inputs_reuses_the_vm_bundle_this_mac_keeps_and_a_changed_service_composes_another(
        mac, tmp_path):
    driver = _driver(tmp_path / "desk")
    first = _put(driver, mac)
    composer.compose(first, composing=Composing(sandbox_provider=mac))

    assert len(mac.opened) == 1 and first.metadata["composed"]["vm_bundle"]["arm64"]["reused"]
    other = _put(_driver(tmp_path / "desk", daemon={"DO_NOT_TRACK": "0"}), mac)
    assert len(mac.opened) == 2 and (other.metadata["composed"]["vm_bundle"]["arm64"]["image"]
                                     != first.metadata["composed"]["vm_bundle"]["arm64"]["image"])


def test_a_vm_bundles_venvs_install_offline_from_a_lock_resolved_once_for_their_inputs(mac, tmp_path):
    driver = _driver(tmp_path / "desk")
    first = _put(driver, mac)
    image = first.metadata["composed"]["vm_bundle"]["arm64"]["image"]
    other = FakeMac()

    composer.compose(first, [("vm_bundle", "arm64")], Composing(sandbox_provider=other))

    assert len(mac.resolves) == 1 and not other.resolves and other.kept == {image} and other.staged == mac.staged
    changed = _put(_driver(tmp_path / "desk", requirements="mss\npillow\n"), mac)
    assert len(mac.resolves) == 2 and changed.metadata["composed"]["vm_bundle"]["arm64"]["image"] != image
    failing = FakeMac()
    failing.resolve_exit = 1
    broken = _put(_driver(tmp_path / "desk", requirements="mss\nnumpy\n"), failing)
    built = broken.metadata["composed"]["vm_bundle"]["arm64"]
    assert (built["status"], built["stage"]) == ("failed", "resolve") and "no matching distribution" in built["reason"]
    assert failing.resolve_terminated == [name for _, name in failing.resolves] and not failing.opened


def test_the_build_settings_a_vm_bundle_fixes_reach_its_services_and_another_value_composes_another(mac, tmp_path):
    driver = _driver(tmp_path / "desk")
    dark = _put(driver, mac, theme="dark").metadata["composed"]["vm_bundle"]["arm64"]["image"]
    light = _put(driver, mac).metadata["composed"]["vm_bundle"]["arm64"]["image"]

    assert mac.settings == [{"AGENTENV_SETTING_THEME": "dark"}, {"AGENTENV_SETTING_THEME": "light"}]
    assert dark != light


def test_a_mac_that_lacks_the_vm_bundle_its_record_names_composes_it_there_under_that_name(mac, tmp_path):
    env = _put(_driver(tmp_path / "desk"), mac)
    recorded = env.metadata["composed"]["vm_bundle"]["arm64"]["image"]
    other, elsewhere = FakeMac(), FakeMac()

    composer.held_or_composed_here(env, "arm64", other)

    assert other.kept == {recorded} and env.metadata["composed"]["vm_bundle"]["arm64"]["image"] == recorded
    env.metadata["composed"]["vm_bundle"]["arm64"]["image"] = "local:agentenv-computer-elsewhere"
    with pytest.raises(RuntimeError, match="not the local:agentenv-computer-elsewhere its record names, as another "
                                           "agent-env composed it; compose it on this Mac with agent-env run"):
        composer.held_or_composed_here(env, "arm64", elsewhere)
    assert not elsewhere.opened


def test_a_sync_on_a_mac_that_lacks_the_vm_bundle_its_record_names_composes_it_there_once_and_records_what_it_composed(
        mac, tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=composer.__name__)  # live logging would put pytest's stdout back mid-command
    env = _put(_driver(tmp_path / "desk"), mac)
    composed = env.metadata["composed"]["vm_bundle"]["arm64"]["image"]
    other = FakeMac()

    assert composer.pending(env, mac) == [] and composer.pending(env, SandboxProvider()) == []
    assert composer.pending(env, other) == [("vm_bundle", "arm64")]
    env.metadata["composed"]["vm_bundle"]["arm64"]["image"] = "local:agentenv-computer-elsewhere"
    composer.compose(env, composer.pending(env, other), Composing(sandbox_provider=other))

    built = Env.get(env.id, env.version)
    assert other.kept == {composed} and built.metadata["composed"]["vm_bundle"]["arm64"]["image"] == composed
    assert built.computer_artifact().forms.vm_bundle.image == composed and composer.pending(built, other) == []
    slow = FakeMac()
    slow.delay = 0.3
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: composer.compose(Env.get(env.id, env.version), [("vm_bundle", "arm64")],
                                                 Composing(sandbox_provider=slow)), range(2)))
    assert len(slow.opened) == 1 and slow.kept == {composed}
    flaky = FakeMac()
    monkeypatch.setattr(composer, "_probed", lambda env, driver, form, arch, result, provider, progress: {
        **result, "status": "failed", "stage": "probe", "reason": "the probe failed: P4"})
    monkeypatch.setattr(providers, "get_env_sandbox_provider", lambda: flaky)
    rebuilt = CliRunner().invoke(cli, ["env", "build", "--id", env.id])
    assert rebuilt.exit_code == 1 and "vm_bundle/arm64: failed (the probe failed: P4)" in rebuilt.output
    assert len(flaky.opened) == 1 and not flaky.kept


def test_deploys_on_a_mac_that_lacks_the_vm_bundle_compose_it_once_and_none_boots_it_before_its_probe_passed(
        mac, tmp_path, monkeypatch):
    env = _put(_driver(tmp_path / "desk"), mac)
    image = env.metadata["composed"]["vm_bundle"]["arm64"]["image"]
    probed, outcome = [], {"status": "ready"}

    def probe(env, driver, form, arch, result, provider, progress):
        time.sleep(0.3)
        probed.append((provider, result["image"], time.monotonic()))
        return {**result, **outcome}

    monkeypatch.setattr(composer, "_probed", probe)
    other = FakeMac()

    def deploy(_):
        composer.held_or_composed_here(env, "arm64", other)
        return time.monotonic()

    with ThreadPoolExecutor(2) as pool:
        booted = list(pool.map(deploy, range(2)))

    assert len(other.opened) == 1 and other.kept == {image} and [p[:2] for p in probed] == [(other, image)]
    assert min(booted) >= probed[0][2]
    outcome, failing = {"status": "failed", "stage": "probe", "reason": "the probe failed: P4"}, FakeMac()
    with pytest.raises(RuntimeError, match="couldn't be composed and probed on this Mac: the probe failed: P4"):
        composer.held_or_composed_here(env, "arm64", failing)
    assert len(failing.opened) == 1 and not failing.kept


def test_a_failed_provisioning_or_keep_deletes_its_builder_and_a_failed_probe_its_vm_bundle_failing_the_pair(
        mac, tmp_path, monkeypatch):
    mac.exit = 1

    env = _put(_driver(tmp_path / "desk"), mac)

    built = env.metadata["composed"]["vm_bundle"]["arm64"]
    assert built["status"] == "failed" and "provisioning exited 1: boom" in built["reason"]
    assert mac.terminated == [mac.opened[0][1]] and not mac.kept
    stuck = FakeMac()
    stuck.keep_error = TimeoutError("it didn't shut down within 600 s, so it isn't kept")
    composer.compose(env, [("vm_bundle", "arm64")], Composing(sandbox_provider=stuck))
    built = Env.get(env.id, env.version).metadata["composed"]["vm_bundle"]["arm64"]
    assert built["status"] == "failed" and "didn't shut down within 600 s" in built["reason"]
    assert stuck.terminated == [stuck.opened[0][1]] and not stuck.kept
    monkeypatch.setattr(composer, "_probed", lambda env, driver, form, arch, result, provider, progress: {
        **result, "status": "failed", "stage": "probe", "reason": "the probe failed: P4"})
    flaky = FakeMac()
    composer.compose(env, [("vm_bundle", "arm64")], Composing(sandbox_provider=flaky))
    built = Env.get(env.id, env.version).metadata["composed"]["vm_bundle"]["arm64"]
    assert (built["status"], built["stage"]) == ("failed", "probe") and len(flaky.opened) == 1 and not flaky.kept


def test_a_vm_bundle_is_composed_only_where_a_link_boots_macos_vm_bundles(mac, tmp_path):
    env = _put(_driver(tmp_path / "desk"), SandboxProvider())

    built = env.metadata["composed"]["vm_bundle"]["arm64"]
    assert built["status"] == "failed" and "boots macos/arm64 VM bundles" in built["reason"]


@pytest.mark.parametrize("table, problem", [
    ({"services": [{"name": "d", "command": "x"}], "packages": {"apt": ["curl"]}}, "macOS has no apt"),
    ({"services": [{"name": "d", "command": "x", "environment": {"A": "1"}}]}, "only a direct one names its own"),
    ({"services": [{"name": "d", "command": "x", "scope": "system"}]}, "services.0.scope"),
    ({"services": [{"name": "d", "command": "x"}], "grants": [{"app": "/Applications/X", "services": ["accessibility"]}]},
     "grants.0.app"),
    ({"services": [{"name": "d", "command": "x"}], "grants": [{"services": ["accessibility"]}]}, "names one program"),
    ({"services": [{"name": "d", "command": "x"}],
      "grants": [{"app": "/Applications/X.app", "executable": "x", "services": ["accessibility"]}]}, "names one program"),
    ({"services": [{"name": "d", "command": "x"}], "grants": [{"executable": "../x", "services": ["accessibility"]}]},
     "grants.0.executable"),
], ids=["apt", "environment", "system", "grant-not-an-app", "grant-of-nothing", "grant-of-two", "grant-outside"])
def test_a_macos_install_takes_no_apt_and_its_services_are_session_launchagents(table, problem):
    found, problems = install.installs({"macos": table})
    assert not found and any(problem in each for each in problems)


def test_a_macos_base_refuses_a_driver_that_declares_no_install_for_it(mac, tmp_path):
    driver = _driver(tmp_path / "desk")
    driver.install = {}
    with pytest.raises(ValueError, match="declares no install for macos"):
        _put(driver, mac)


def test_site_routing_into_a_macos_computer_is_refused_at_save_and_at_deploy(mac, tmp_path):
    env = _put(_driver(tmp_path / "desk"), mac)
    params = {"site_routing": True, "sites": [{"name": "a", "url": "http://gw/website/a/", "domains": ["a.test"]}]}

    saved = DeployEnvTaskStep(id="d", version=None, env_id=env.id, env_params=params).preflight()
    _, _, problems = computer_params.checked(env, params)

    assert problems == ["env_params.site_routing: site routing reaches Linux computers alone, and this computer runs "
                        "macos"]
    assert len(saved) == 1 and problems[0] in saved[0]


def test_the_service_runner_reads_the_protocols_seed_byte_for_byte(tmp_path):
    loader = importlib.machinery.SourceFileLoader("agentenv_service", str(macos_vm_bundle.PREPARE / "agentenv-service"))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    (tmp_path / "user-data").write_text(seed.SAMPLE_USER_DATA)

    assert module.environment(tmp_path) == seed.SAMPLE
    (tmp_path / seed.OLDEST).write_text("2\n")
    with pytest.raises(SystemExit, match="this computer reads seed v1; this agent-env writes v2 or later"):
        module.environment(tmp_path)
