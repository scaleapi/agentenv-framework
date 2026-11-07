"""A Windows computer env, composed on this machine's Docker with KVM: its base installed from Microsoft's media once and
kept here, then the driver installed over ssh by a provision script as its ``[install.windows]`` says and packed as a
thin layer on the base, under a tag of its key, which a compose of the same inputs reuses while this machine holds it;
the env's computer is a disk of the base's form. A Windows base is amd64, disk-only and started by agentenv-session."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath

import pytest
from pydantic import ValidationError

from agentenv_protocol.computer import DriverDeclaration
from agentenv_protocol.computer.app import Download, Packages, RunScript

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.env import Env
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.utils import composer, gate, windows_disk
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.envs.computer.utils.install import WindowsInstall, WindowsService
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

URL = "https://example.com/desk.zip"
SERVE = "venv/Scripts/python.exe -m agentenv_protocol.computer.serve driver.py:Desk"
WINDOWS = {"type": "windows", "version": "11", "edition": "Enterprise Evaluation"}
DISK = {"init": "agentenv-session", "input_channel": "nocloud", "min_disk_gb": 64}


class FakeLinux(SandboxProvider):
    """This machine with KVM, which boots Windows disks or doesn't."""

    def __init__(self, platforms=frozenset({"linux/amd64", "windows/amd64"})):
        self.platforms = platforms

    @classmethod
    def native_platforms(cls):
        return frozenset({"linux/amd64"})

    def disk_image_platforms(self):
        return self.platforms


class FakeDocker:
    """windows_disk on this machine's Docker: bases installed and disks composed, as images it holds."""

    def __init__(self, tmp_path: Path):
        self.tmp, self.images, self.installs, self.composed = tmp_path, set(), [], []

    def base(self, key, iso, disk_gb, user, say):
        image = f"agentenv-windows-base:{key[7:23]}"
        if image not in self.images:
            self.installs.append((key, iso, disk_gb, user))
            self.images.add(image)
        return windows_disk.Base(image, self.tmp / "keys", datetime.now(timezone.utc) + timedelta(days=89))

    def compose(self, base, build, image, labels, say):
        self.composed.append((base.image, (build / "provision.ps1").read_bytes().decode(),
                              sorted(p.relative_to(build).as_posix() for p in build.rglob("*") if p.is_file()), labels))
        self.images.add(image)


@pytest.fixture
def docker(local_stores, cli_routing, monkeypatch, tmp_path):
    fake = FakeDocker(tmp_path)
    iso = tmp_path / "windows.iso"
    iso.write_bytes(b"iso")
    monkeypatch.setattr(windows_disk, "media", lambda say: iso)
    monkeypatch.setattr(windows_disk, "sha256", lambda path: "ab" * 32)
    monkeypatch.setattr(windows_disk, "base", fake.base)
    monkeypatch.setattr(windows_disk, "compose", fake.compose)
    monkeypatch.setattr(windows_disk, "held", lambda image: image in fake.images)
    monkeypatch.setattr(windows_disk, "image_id", lambda image: f"sha256:{image}")
    monkeypatch.setattr(windows_disk, "remove", fake.images.discard)
    monkeypatch.setattr(composer, "_probed", lambda env, driver, form, arch, result, provider, progress: {
        **result, "status": "ready"})
    return fake


def _base(**changes) -> ComputerArtifact:
    return ComputerArtifact.put(**{
        "id": "@local/~/kit/windows-11", "arch": ["amd64"], "os": WINDOWS, "session": {"user": "user"},
        "config": {"resolution": {"default": "1920x1080", "phase": "deploy"},
                   "theme": {"default": "light", "phase": "build"}},
        "forms": {"disk": DISK}, **changes})


def _driver(folder: Path) -> ComputerDriverEnv:
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (("driver.py", "class Desk: ...\n"), ("requirements-windows.txt", "mss\n"),
                       ("install.ps1", "Expand-Archive desk.zip\n")):
        (folder / name).write_text(text)
    files = FileArtifactUniverse.put_bundled("@local/~/kit/desk__files", files={
        name: folder / name for name in ("driver.py", "requirements-windows.txt", "install.ps1")})
    windows = WindowsInstall(
        python="requirements-windows.txt", downloads=[Download(url=URL, sha256="cd" * 32, to="desk.zip", arch="amd64")],
        run=[RunScript(script="install.ps1")],
        services=[WindowsService(name="daemon", command="venv/Scripts/python.exe driver.py"),
                  WindowsService(name="driver", command=SERVE)])
    put = ComputerDriverEnv.put(
        id="@local/~/kit/desk", module="driver.py:Desk", files=files, port=18765, install={"windows": windows},
        declaration=DriverDeclaration.model_validate({
            "name": "desk", "supports": [{"os": {"type": "windows", "display": ["dwm"]}, "arch": ["amd64"]}],
            "platforms": [{"platform": {"os": "windows", "arch": "amd64", "display": "dwm"}, "provides": [],
                           "tools": ["computer_screenshot"], "features": {}}]}))
    return Env.get(put.id, put.version)


def _put(driver: ComputerDriverEnv, link: SandboxProvider, **settings) -> ComputerEnv:
    return ComputerEnv.put(id="@local/~/kit/desktop", computer=_base(), driver=driver, environment_name="desktop",
                           composing=Composing(sandbox_provider=link), computer_settings=settings or None)


def test_a_windows_env_composes_a_thin_disk_on_its_base_here_the_driver_installed_by_its_provision_script(
        docker, tmp_path):
    env = _put(_driver(tmp_path / "desk"), FakeLinux(), theme="dark")

    built = env.metadata["composed"]["disk"]["amd64"]
    assert built["status"] == "ready" and built["image"].startswith("agentenv-windows:") and not built["reused"]
    (installed,) = docker.installs
    assert installed[1:] == (tmp_path / "windows.iso", 64, "user")
    ((base, script, files, labels),) = docker.composed
    assert base == f"agentenv-windows-base:{installed[0][7:23]}" and labels[composer.KEY_LABEL] == built["key"]
    prefix = PureWindowsPath("C:/agentenv/drivers/desk")
    lines = script.split("\r\n")
    for line in [f"Fetch '{URL}' '{'cd' * 32}' '{prefix}\\desk.zip'",
                 f"Checked {{ & 'C:\\agentenv\\python\\python.exe' -m venv '{prefix}\\venv' }}",
                 f"$env:PREFIX = '{prefix}'; Push-Location '{prefix}'; Checked {{ & powershell -NoProfile "
                 f"-ExecutionPolicy Bypass -File '{prefix}\\install.ps1' }}; Pop-Location",
                 "[IO.File]::WriteAllText('C:\\agentenv\\build-settings.json', '{\"AGENTENV_SETTING_THEME\": \"dark\"}')"]:
        assert line in lines
    assert any(line.startswith(f"Checked {{ & '{prefix}\\venv\\Scripts\\python.exe' -m pip install") and
               f"'-r' '{prefix}\\requirements-windows.txt'" in line for line in lines)
    daemon = next(line for line in lines if "services\\driver-desk-daemon.json" in line)
    assert json.loads(daemon.split(", '", 1)[1].rsplit("')", 1)[0].replace("''", "'")) == {
        "name": "driver-desk-daemon", "prefix": str(prefix), "command": [str(prefix / "venv/Scripts/python.exe"),
                                                                         "driver.py"], "env": {"PYTHONUTF8": "1", "MCP_PORT": "18765"}}
    assert {"provision.ps1", "context/driver/driver.py", "context/driver/install.ps1"} <= set(files)
    computer = env.computer_artifact()
    assert computer.os.type == "windows" and computer.forms.disk.init == "agentenv-session"
    assert computer.forms.disk.image is None and env.metadata["computer"]["disks"] == {"amd64": built["image"]}


def test_a_compose_of_the_same_inputs_reuses_the_disk_this_machine_holds_and_composes_one_it_lost(docker, tmp_path):
    driver = _driver(tmp_path / "desk")
    first = _put(driver, FakeLinux())
    image = first.metadata["composed"]["disk"]["amd64"]["image"]

    again = _put(driver, FakeLinux())
    assert again.metadata["composed"]["disk"]["amd64"]["reused"] and len(docker.composed) == 1
    assert composer.pending(again, FakeLinux()) == []

    docker.images.discard(image)
    assert composer.pending(again, FakeLinux()) == [("disk", "amd64")]
    assert len(docker.installs) == 1


def test_a_machine_that_boots_no_windows_disk_composes_none(docker, tmp_path):
    env = _put(_driver(tmp_path / "desk"), FakeLinux(frozenset({"linux/amd64"})))

    built = env.metadata["composed"]["disk"]["amd64"]
    assert built["status"] == "failed" and "boots windows/amd64 disks" in built["reason"]
    assert docker.installs == [] and docker.composed == []


@pytest.mark.parametrize("changes, refused", [
    ({"arch": ["arm64"]}, "a Windows computer is amd64 alone"),
    ({"forms": {"disk": {**DISK, "init": "supervisord"}}}, "a windows disk's init is agentenv-session"),
    ({"forms": {"container": {"image": {"id": "x", "version": 1}, "init": "supervisord", "input_channel": "env"},
                "disk": DISK}}, "a windows computer's forms are disk"),
    ({"os": {**WINDOWS, "version": "11.0"}}, "should match pattern"),
])
def test_a_windows_base_is_an_amd64_disk_agentenv_session_starts(local_stores, cli_routing, changes, refused):
    with pytest.raises((ValidationError, ValueError), match=refused):
        _base(**changes)


def test_a_linux_disk_is_still_started_by_supervisord(local_stores, cli_routing):
    with pytest.raises(ValidationError, match="a linux disk's init is supervisord"):
        ComputerArtifact(id="x", arch=["amd64"], os={"type": "linux", "distribution": "ubuntu", "version": "24.04",
                                                     "display": "x11"}, session={"user": "user"},
                         forms={"container": {"image": {"id": "x", "version": 1}, "init": "supervisord",
                                              "input_channel": "env"}, "disk": DISK})


@pytest.mark.parametrize("install, refused", [
    ({"packages": Packages(apt=["curl"])}, "Windows has no apt"),
    ({"run": [RunScript(script="install.sh")]}, r"PowerShell \(.ps1\)"),
    ({"run": [RunScript(script="install.ps1", scope="session")]}, "run elevated"),
])
def test_a_windows_install_fetches_by_sha256_and_runs_powershell_elevated(install, refused):
    with pytest.raises(ValidationError, match=refused):
        WindowsInstall(services=[WindowsService(name="driver", command=SERVE)], **install)


def test_a_service_runs_its_relative_program_from_the_layers_prefix_its_quoted_words_whole():
    prefix = PureWindowsPath("C:/agentenv/drivers/desk")

    assert windows_disk.argv("venv/Scripts/python.exe driver.py", prefix) == [
        "C:\\agentenv\\drivers\\desk\\venv\\Scripts\\python.exe", "driver.py"]
    assert windows_disk.argv("C:/Windows/notepad.exe x", prefix) == ["C:/Windows/notepad.exe", "x"]
    assert windows_disk.argv('"C:/Program Files/Tool/tool.exe" --config "settings file.json"', prefix) == [
        "C:/Program Files/Tool/tool.exe", "--config", "settings file.json"]


def test_a_disk_whose_evaluation_nears_its_end_is_composed_again(docker, tmp_path, monkeypatch):
    env = _put(_driver(tmp_path / "desk"), FakeLinux())
    assert composer.pending(env, FakeLinux()) == []

    monkeypatch.setattr(windows_disk, "RENEW_DAYS", 100)
    assert composer.pending(env, FakeLinux()) == [("disk", "amd64")]


@pytest.mark.parametrize("days_left, installs", [(30, 0), (3, 1)])
def test_a_base_is_reused_until_its_evaluation_nears_its_end_then_installed_afresh(monkeypatch, tmp_path, days_left,
                                                                                    installs):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    key = "sha256:" + "12" * 32
    keys = tmp_path / "agentenv" / "windows" / "bases" / key[7:23]
    keys.mkdir(parents=True)
    (keys / "id").write_text("key")
    expires = datetime.now(timezone.utc) + timedelta(days=days_left)
    installed = []
    monkeypatch.setattr(windows_disk, "present", lambda image: True)
    monkeypatch.setattr(windows_disk, "_label", lambda image, name: expires.isoformat(timespec="seconds"))
    monkeypatch.setattr(windows_disk.subprocess, "run", lambda argv, **kwargs: (keys / "id").write_text("new"))
    monkeypatch.setattr(windows_disk, "_installed", lambda *args: installed.append(args) or {
        "grace_minutes": 90 * 24 * 60, "build": "26200.6584"})
    monkeypatch.setattr(windows_disk, "_pack", lambda folder, image, labels, base_image: None)

    base = windows_disk.base(key, tmp_path / "windows.iso", 64, "user", lambda line: None)

    assert len(installed) == installs and base.image == f"agentenv-windows-base:{key[7:23]}"
    assert (base.expires - datetime.now(timezone.utc)).days == (days_left - 1 if not installs else 89)


def test_a_windows_computers_display_is_1920x1080_for_now(local_stores, cli_routing):
    base = _base()

    assert gate.check_computer_settings(base, {"resolution": "1920x1080"}) == ({"resolution": "1920x1080"}, [])
    assert gate.check_computer_settings(base, {"resolution": "1280x800"})[1] == [
        "computer_settings.resolution: a Windows computer's display is 1920x1080 for now, not 1280x800"]
    assert gate._display(base, {"width": 1280, "height": 800}, None) == [
        "display: a Windows computer's display is 1920x1080 for now"]
