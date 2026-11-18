"""A Windows computer env, composed on this machine's Docker with KVM: its base installed from Microsoft's media once and
kept here, then the driver and each app installed over ssh by a provision script as their ``[install.windows]`` says
and packed as a thin layer on the base, under a tag of its key, which a compose of the same inputs reuses while this
machine holds it, its agentenv folder left the admins' and the session user's to read; the env's computer is a disk
of the base's form. An app's own server runs from its venv, its own folder the session user's alone, its install steps
elevated; one it runs as the session user is refused. A Windows base is amd64, disk-only and started by
agentenv-session. The scripts the Windows test bundle's verifiers run inside the computer are Python."""

import ast
import importlib.util
import json
import threading
import time
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath

import pytest
from pydantic import ValidationError

from agentenv_protocol.computer import AppDeclaration, DriverDeclaration
from agentenv_protocol.computer.app import Download, Packages, RunScript

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.env import Env
from agent_env.env.envs.computer.app_env import ComputerAppEnv
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.utils import composer, gate, thin_disk, windows_disk
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.envs.computer.utils.install import WindowsInstall, WindowsService
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

URL = "https://example.com/desk.zip"
BUNDLE = Path(__file__).parents[3] / "integration" / "computer" / "windows"
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


class FakeRegistry:
    """The env's image store as Windows disks are kept in it: what it serves, by tag, pinned."""

    def __init__(self):
        self.held = {}

    def served(self, tag):
        return self.held.get(tag)

    def push(self, local, tag):
        self.held[tag] = f"registry.test/agentenv-computers:{tag}@sha256:{len(self.held):064x}"
        return self.held[tag]


class FakeDocker:
    """windows_disk and thin_disk on one machine's Docker: bases installed and disks composed and pushed to the image
    store."""

    def __init__(self, tmp_path: Path, registry: FakeRegistry):
        self.tmp, self.registry, self.bases, self.installs, self.composed = tmp_path, registry, {}, [], []

    def base(self, key, iso, disk_gb, user, say, registry):
        if key not in self.bases:
            self.installs.append((key, iso(), disk_gb, user))
            self.bases[key] = (datetime.now(timezone.utc) + timedelta(days=89)).isoformat(timespec="seconds")
        return thin_disk.Base(f"agentenv-windows-base:{key[7:23]}", self.tmp / "keys",
                              {windows_disk.EXPIRES_LABEL: self.bases[key]})

    def compose(self, guest, base, build, key, labels, say, registry):
        self.composed.append((base.image, (build / "provision.ps1").read_bytes().decode(),
                              sorted(p.relative_to(build).as_posix() for p in build.rglob("*") if p.is_file()), labels))
        return registry.push(*thin_disk.disk_names(guest, key))


@pytest.fixture
def docker(local_stores, cli_routing, monkeypatch, tmp_path):
    fake = FakeDocker(tmp_path, FakeRegistry())
    iso = tmp_path / "windows.iso"
    iso.write_bytes(b"iso")
    monkeypatch.setattr(windows_disk, "media", lambda say: iso)
    monkeypatch.setattr(windows_disk, "base", fake.base)
    monkeypatch.setattr(thin_disk, "compose", fake.compose)
    monkeypatch.setattr(thin_disk, "Registry", lambda store, repository: fake.registry)
    monkeypatch.setattr(composer, "get_config", lambda: SimpleNamespace(
        get_image_store_for=lambda stored_id: SimpleNamespace(ensure_repository=lambda repository: None)))
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


def _notes(base: ComputerArtifact | None, driver: ComputerDriverEnv | None, folder: Path,
           steps: tuple = (("stamp", "system", None),)) -> ComputerAppEnv:
    """A notes app on Windows: a sync service in the venv its own server gets, an elevated run script, ``steps``, its
    install steps by name, scope and the OS types they run on, and the edition its env fixes."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (("app.py", "class Notes: ...\n"), ("setup.ps1", "'ran' | Set-Content run.txt\n")):
        (folder / name).write_text(text)
    files = FileArtifactUniverse.put_bundled("@local/~/kit/notes__files", files={
        name: folder / name for name in ("app.py", "setup.ps1")})
    declaration = AppDeclaration.model_validate({
        "supports": [{"os": {"type": "windows", "display": ["dwm"]}, "arch": ["amd64"]}], "permissions": [],
        "settings": {"type": "object", "properties": {}}, "skill": None, "data": [], "ready_checks": [],
        "build_settings": {"type": "object", "properties": {"edition": {"type": "string", "default": "home"}}},
        "install_steps": [{"name": name, "scope": scope, "os": os} for name, scope, os in steps],
        "os": {"windows": {"install": {"run": [{"script": "setup.ps1"}], "services": [
            {"name": "sync", "command": "venv/Scripts/python.exe sync.py"}]}}}})
    return ComputerAppEnv.put(id="@local/~/kit/notes", environment_name="notes", module="app.py:Notes", files=files,
                              declaration=declaration, computer=base, driver=driver, compose=False,
                              build_settings={"edition": "pro"})


def _put(driver: ComputerDriverEnv, link: SandboxProvider, apps: list[ComputerAppEnv] | None = None,
         base: ComputerArtifact | None = None, **settings) -> ComputerEnv:
    return ComputerEnv.put(id="@local/~/kit/desktop", computer=base or _base(), driver=driver,
                           environment_name="desktop", composing=Composing(sandbox_provider=link),
                           computer_settings=settings or None, apps=apps or [])


def _written_json(line: str) -> object:
    """The JSON a provision script's line writes to its file (``windows_disk._written``)."""
    return json.loads(line.split(", '", 1)[1].rsplit("')", 1)[0].replace("''", "'"))


def test_a_windows_env_composes_a_thin_disk_on_its_base_here_the_driver_installed_by_its_provision_script(
        docker, tmp_path):
    env = _put(_driver(tmp_path / "desk"), FakeLinux(), theme="dark")

    built = env.metadata["composed"]["disk"]["amd64"]
    assert built["status"] == "ready" and not built["reused"]
    assert built["image"] == docker.registry.held[built["key"].removeprefix("sha256:")]
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
    assert _written_json(daemon) == {
        "name": "driver-desk-daemon", "prefix": str(prefix), "command": [str(prefix / "venv/Scripts/python.exe"),
                                                                         "driver.py"], "env": {"PYTHONUTF8": "1", "MCP_PORT": "18765"}}
    assert lines[-3:] == [
        "Checked { icacls 'C:\\agentenv' /inheritance:r /grant:r '*S-1-5-32-544:(OI)(CI)F' '*S-1-5-18:(OI)(CI)F' "
        "'*S-1-5-32-545:(OI)(CI)RX' '*S-1-15-2-1:(OI)(CI)RX' '*S-1-15-2-2:(OI)(CI)RX' | Out-Null }",
        "Checked { icacls 'C:\\agentenv\\logs' /grant 'user:(OI)(CI)M' | Out-Null }", ""]
    assert {"provision.ps1", "context/driver/driver.py", "context/driver/install.ps1"} <= set(files)
    computer = env.computer_artifact()
    assert computer.os.type == "windows" and computer.forms.disk.init == "agentenv-session"
    assert computer.forms.disk.image is None and env.metadata["computer"]["disks"] == {"amd64": built["image"]}


def test_an_app_is_a_layer_of_its_own_its_server_on_its_venv_its_folder_the_session_users_its_steps_elevated(
        docker, tmp_path):
    base = _base()
    driver = _driver(tmp_path / "desk")
    env = _put(driver, FakeLinux(), apps=[_notes(base, driver, tmp_path / "notes")], base=base)

    assert env.metadata["composed"]["disk"]["amd64"]["status"] == "ready"
    ((_, script, files, _),) = docker.composed
    prefix, lines = PureWindowsPath("C:/agentenv/apps/notes"), script.split("\r\n")
    python = prefix / "venv" / "Scripts" / "python.exe"
    installed = lines[lines.index("# apps/notes") + 1:]
    assert installed[:5] == [
        f"New-Item -ItemType Directory -Force '{prefix}' | Out-Null",
        f"Copy-Item -Recurse -Force \"$build\\context\\apps\\notes\\*\" '{prefix}'",
        f"Checked {{ & 'C:\\agentenv\\python\\python.exe' -m venv '{prefix}\\venv' }}",
        f"Checked {{ & '{python}' -m pip install -q --disable-pip-version-check "
        "'C:\\agentenv\\build\\context\\protocol[app]' }",
        f"Checked {{ & '{python}' -m pip list --format=json | Set-Content -Encoding utf8 '{prefix}\\installed.json' }}"]
    assert installed[5:9] == [
        f"New-Item -ItemType Directory -Force '{prefix}\\.agentenv' | Out-Null",
        f"Checked {{ icacls '{prefix}\\.agentenv' /inheritance:r /grant:r 'user:(OI)(CI)F' '*S-1-5-18:(OI)(CI)F' "
        "'*S-1-5-32-544:(OI)(CI)F' | Out-Null }",
        f"$env:PREFIX = '{prefix}'; Push-Location '{prefix}'; Checked {{ & powershell -NoProfile -ExecutionPolicy "
        f"Bypass -File '{prefix}\\setup.ps1' }}; Pop-Location",
        f"Push-Location '{prefix}'; Checked {{ & '{python}' -m agentenv_protocol.computer.run_install_step "
        "'app.py:Notes' 'stamp' }; Pop-Location"]
    services = {spec["name"]: spec for spec in (_written_json(line) for line in installed if "\\services\\" in line)}
    environment = {"PYTHONUTF8": "1", "ENVIRONMENT_NAME": "notes"}
    assert services == {
        "app-notes": {"name": "app-notes", "prefix": str(prefix), "env": environment,
                      "command": [str(python), "-m", "agentenv_protocol.computer.serve_app", "app.py:Notes"]},
        "app-notes-sync": {"name": "app-notes-sync", "prefix": str(prefix), "env": environment,
                           "command": [str(python), "sync.py"]}}
    driver_lines = lines[lines.index("# driver"):lines.index("# apps/notes")]
    assert not any(".agentenv" in line or "run_install_step" in line for line in driver_lines)
    assert any(line.endswith("\"layers\": [\"driver\", \"apps/notes\"]}')") for line in lines)
    assert {"context/apps/notes/app.py", "context/apps/notes/.agentenv/build_settings.json"} <= set(files)


def test_an_apps_install_step_for_the_session_user_is_refused_on_windows_alone_or_composed(docker, tmp_path):
    base = _base()
    driver = _driver(tmp_path / "desk")
    steps = (("stamp", "system", None), ("remember", "session", ["linux"]), ("greet", "session", None))
    refused = "@local/~/kit/notes's install steps greet run as the session user, which a Windows computer can't"

    with pytest.raises(ValueError, match=f"computer app env '@local/~/kit/notes': app: {refused}"):
        _notes(base, driver, tmp_path / "notes", steps)
    with pytest.raises(ValueError, match=rf"apps\[0\]: {refused}"):
        _put(driver, FakeLinux(), apps=[_notes(None, None, tmp_path / "notes", steps)], base=base)
    assert docker.composed == []


def test_a_compose_of_the_same_inputs_reuses_the_disk_the_image_store_holds(docker, tmp_path):
    driver = _driver(tmp_path / "desk")
    first = _put(driver, FakeLinux())
    image = first.metadata["composed"]["disk"]["amd64"]["image"]

    again = _put(driver, FakeLinux())

    built = again.metadata["composed"]["disk"]["amd64"]
    assert built["reused"] and built["image"] == image and len(docker.composed) == 1 and len(docker.installs) == 1
    assert composer.pending(again, FakeLinux()) == []


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
    """Either way the image store gets it, which a push that failed after an install, or another env's store, lacks."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    key = "sha256:" + "12" * 32
    keys = tmp_path / "agentenv" / "windows" / "bases" / key[7:23]
    keys.mkdir(parents=True)
    (keys / "id").write_text("key")
    expires = datetime.now(timezone.utc) + timedelta(days=days_left)
    installed, said = [], []
    monkeypatch.setattr(thin_disk, "present", lambda image: True)
    monkeypatch.setattr(thin_disk, "local_labels", lambda image: {
        windows_disk.EXPIRES_LABEL: expires.isoformat(timespec="seconds")})
    monkeypatch.setattr(thin_disk.subprocess, "run", lambda argv, **kwargs: (keys / "id").write_text("new"))
    monkeypatch.setattr(windows_disk, "_installed", lambda work, *args: (work / "base.img").write_text("") or
                        installed.append(args) or {"grace_minutes": 90 * 24 * 60, "build": "26200.6584"})
    monkeypatch.setattr(thin_disk, "pack", lambda guest, folder, image, labels, base_image: None)
    registry = FakeRegistry()

    base = windows_disk.base(key, lambda: tmp_path / "windows.iso", 64, "user", said.append, registry)

    assert len(installed) == installs and base.image == f"agentenv-windows-base:{key[7:23]}"
    left = datetime.fromisoformat(base.labels[windows_disk.EXPIRES_LABEL]) - datetime.now(timezone.utc)
    assert left.days == (days_left - 1 if not installs else 89)
    assert (f"its base's evaluation expires {expires:%Y-%m-%d}; building it afresh" in said) == bool(installs)
    assert list(registry.held) == [f"windows-base-{key[7:23]}"]


def test_a_base_this_machine_holds_goes_to_an_image_store_lacking_it_once(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    key = "sha256:" + "56" * 32
    keys = tmp_path / "agentenv" / "windows" / "bases" / key[7:23]
    keys.mkdir(parents=True)
    (keys / "id").write_text("key")
    expires = datetime.now(timezone.utc) + timedelta(days=30)
    monkeypatch.setattr(thin_disk, "present", lambda image: True)
    monkeypatch.setattr(thin_disk, "local_labels", lambda image: {
        windows_disk.EXPIRES_LABEL: expires.isoformat(timespec="seconds")})
    registry, pushed = FakeRegistry(), []
    push = registry.push
    registry.push = lambda local, tag: pushed.append(local) or push(local, tag)

    for _ in range(2):
        windows_disk.base(key, lambda: tmp_path / "windows.iso", 64, "user", lambda line: None, registry)

    assert pushed == [f"agentenv-windows-base:{key[7:23]}"]


def test_composes_of_two_disks_on_a_base_the_store_holds_pull_it_and_take_its_key_once(monkeypatch, tmp_path):
    """The second waits for the first's pull, then finds the base here."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    keys = tmp_path / "keys"
    made, pulls = threading.Event(), []

    def pull(pinned):
        pulls.append(pinned)
        time.sleep(0.2)

    monkeypatch.setattr(thin_disk, "present", lambda image: made.is_set())
    monkeypatch.setattr(thin_disk, "docker", lambda *args: None)
    monkeypatch.setattr(thin_disk, "take_admin_key", lambda image, at: (at.mkdir(), (at / "id").write_text("key"),
                                                                            made.set()))
    registry = SimpleNamespace(pull=pull)
    base = thin_disk.Base("agentenv-windows-base:1", keys, pinned="registry.test/base@sha256:1")

    composes = [threading.Thread(target=thin_disk.here, args=(windows_disk.GUEST, base, registry, lambda line: None))
                for _ in range(2)]
    for compose in composes:
        compose.start()
    for compose in composes:
        compose.join(10)

    assert pulls == ["registry.test/base@sha256:1"] and (keys / "id").read_text() == "key"


def test_a_machine_without_the_base_takes_the_image_stores_unless_its_evaluation_nears_its_end(monkeypatch, tmp_path):
    """Found without the ISO, which only an install fetches; pulled only when a layer is composed on it."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    key = "sha256:" + "34" * 32
    registry = FakeRegistry()
    pinned = registry.push("agentenv-windows-base:x", f"windows-base-{key[7:23]}")
    expires = datetime.now(timezone.utc) + timedelta(days=60)
    registry.labels = lambda ref: {windows_disk.EXPIRES_LABEL: expires.isoformat(timespec="seconds")}
    monkeypatch.setattr(thin_disk, "present", lambda image: False)
    fetched = []

    base = windows_disk.base(key, lambda: fetched.append("iso"), 64, "user", lambda line: None, registry)

    assert base.pinned == pinned and base.image == f"agentenv-windows-base:{key[7:23]}" and fetched == []
    assert (datetime.fromisoformat(base.labels[windows_disk.EXPIRES_LABEL]) - datetime.now(timezone.utc)).days == 59


def test_a_windows_computers_display_is_1920x1080_for_now(local_stores, cli_routing):
    base = _base()

    assert gate.check_computer_settings(base, {"resolution": "1920x1080"}) == ({"resolution": "1920x1080"}, [])
    assert gate.check_computer_settings(base, {"resolution": "1280x800"})[1] == [
        "computer_settings.resolution: a Windows computer's display is 1920x1080 for now, not 1280x800"]
    assert gate._display(base, {"width": 1280, "height": 800}, None) == [
        "display: a Windows computer's display is 1920x1080 for now"]


@pytest.mark.parametrize("backing, over, args", [
    (None, [], ()), ("/disk/base.img", ["-B", "/disk/base.img", "-F", "qcow2"], ()),
    ("/disk/base.img", ["-B", "/disk/base.img", "-F", "qcow2"], ("--mount", "type=image,source=base:1,target=/disk"))],
    ids=["a-base", "a-layer-over-its-base", "a-layer-over-its-base-mounted"])
def test_a_disk_is_rewritten_keeping_only_the_clusters_it_uses(monkeypatch, tmp_path, backing, over, args):
    ran = []

    def run(argv, **kwargs):
        ran.append(argv)
        (tmp_path / "compact.img").write_text("compacted")

    (tmp_path / "disk.img").write_text("as Windows left it")
    monkeypatch.setattr(thin_disk.subprocess, "run", run)

    thin_disk.converted(tmp_path, "machine:1", backing=backing, args=args)

    assert ran == [["docker", "run", "--rm", "-v", f"{tmp_path}:/out", *args, "--entrypoint", "qemu-img", "machine:1",
                    "convert", "-O", "qcow2", *over, "/out/disk.img", "/out/compact.img"]]
    assert (tmp_path / "disk.img").read_text() == "compacted" and not (tmp_path / "compact.img").exists()


@pytest.mark.parametrize("artifact, script", [("check_notes", "FACTS"), ("check_session", "SESSION")])
def test_the_scripts_the_windows_bundles_verifiers_run_inside_the_computer_are_python(artifact, script):
    verifier = ast.parse((BUNDLE / "artifacts" / artifact / "verify.py").read_text())
    (text,) = [node.value.value for node in verifier.body
               if isinstance(node, ast.Assign) and [target.id for target in node.targets] == [script]]

    ast.parse(text)


def test_the_session_points_tls_clients_at_a_seeds_trust_folder_and_at_nothing_without_one(tmp_path):
    spec = importlib.util.spec_from_file_location("agentenv_session", windows_disk.RECIPE / "agentenv-session.py")
    session = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(session)
    trust = tmp_path / "trust"

    assert session.trust(trust) == {}
    trust.mkdir()
    (trust / "ca-bundle.pem").write_text("roots and the local CA")
    (trust / "ca.pem").write_text("the local CA")
    assert session.trust(trust) == {"SSL_CERT_FILE": str(trust / "ca-bundle.pem"),
                                    "REQUESTS_CA_BUNDLE": str(trust / "ca-bundle.pem"),
                                    "NODE_EXTRA_CA_CERTS": str(trust / "ca.pem")}
