"""An Omarchy computer env, composed on this machine's Docker with KVM: its base built once from Omarchy's own image in
this machine's cache, then the driver installed over ssh by a bash provision script as its ``[install."linux/omarchy"]``
says, its services systemd user units of the graphical session started through with-seed, and packed as a thin layer on
the base under a tag of its key. An Omarchy base is a disk alone, built from its source, started by systemd; its apps
are refused for now. The base's build boots Omarchy's image as a linux-desktop guest, root's key a systemd credential,
and runs the recipe, which refuses an image its artifact mislabels."""

import base64
import hashlib
import io
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentenv_protocol.computer import DriverDeclaration
from agentenv_protocol.computer.app import Download, RunScript

from agent_env.artifact.artifacts.computer import ComputerArtifact, DiskForm, DiskSource
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.env import Env
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.utils import boot, composer, gate, omarchy_disk, thin_disk
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.envs.computer.utils.install import Install, OmarchyInstall, Service, installs
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

URL = "https://example.com/desk.tar.gz"
SOURCE = {"url": "https://example.com/omarchy.qcow2", "sha256": "ab" * 32}
OMARCHY = {"type": "linux", "distribution": "omarchy", "version": "4.5.0", "display": "wayland"}
DISK = {"source": SOURCE, "init": "systemd", "input_channel": "nocloud", "min_disk_gb": 40}
SERVE = "venv/bin/python -m agentenv_protocol.computer.serve driver.py:Desk"
OMARCHY_SUPPORTS = [{"os": {"type": "linux", "distribution": ["omarchy"], "display": ["wayland"]}, "arch": ["amd64"]}]


class FakeLinux(SandboxProvider):
    """This machine with KVM, which boots linux/amd64 disks or doesn't."""

    def __init__(self, platforms=frozenset({"linux/amd64"})):
        self.platforms = platforms

    @classmethod
    def native_platforms(cls):
        return frozenset({"linux/amd64"})

    def disk_image_platforms(self):
        return self.platforms


class FakeRegistry:
    """The env's image store as thin disks are kept in it: what it serves, by tag, pinned."""

    def __init__(self):
        self.held = {}

    def served(self, tag):
        return self.held.get(tag)

    def push(self, local, tag):
        self.held[tag] = f"registry.test/agentenv-computers:{tag}@sha256:{len(self.held):064x}"
        return self.held[tag]


class FakeDocker:
    """omarchy_disk and thin_disk on one machine's Docker: bases built and disks composed and pushed to the image
    store."""

    def __init__(self, tmp_path: Path, registry: FakeRegistry):
        self.tmp, self.registry, self.builds, self.composed = tmp_path, registry, [], []

    def base(self, key, image, user, distribution, version, say, registry):
        self.builds.append((key, image(), user, distribution, version))
        return thin_disk.Base(f"agentenv-omarchy-base:{key[7:23]}", self.tmp / "keys")

    def compose(self, guest, base, build, key, labels, say, registry):
        self.composed.append((guest, base.image, (build / "provision.sh").read_text(),
                              sorted(p.relative_to(build).as_posix() for p in build.rglob("*") if p.is_file()), labels))
        return registry.push(*thin_disk.disk_names(guest, key))


@pytest.fixture
def docker(local_stores, cli_routing, monkeypatch, tmp_path):
    fake = FakeDocker(tmp_path, FakeRegistry())
    image = tmp_path / "omarchy.qcow2"
    image.write_bytes(b"qcow2")
    monkeypatch.setattr(omarchy_disk, "source", lambda disk_source, say: image)
    monkeypatch.setattr(omarchy_disk, "base", fake.base)
    monkeypatch.setattr(thin_disk, "compose", fake.compose)
    monkeypatch.setattr(thin_disk, "Registry", lambda store, repository: fake.registry)
    monkeypatch.setattr(composer, "get_config", lambda: SimpleNamespace(
        get_image_store_for=lambda stored_id: SimpleNamespace(ensure_repository=lambda repository: None)))
    monkeypatch.setattr(composer, "_probed", lambda env, driver, form, arch, result, provider, progress: {
        **result, "status": "ready"})
    return fake


def _base(**changes) -> ComputerArtifact:
    return ComputerArtifact.put(**{
        "id": "@local/~/kit/omarchy-4.5", "arch": ["amd64"], "os": OMARCHY, "session": {"user": "omarchy"},
        "config": {"resolution": {"default": "1920x1080", "phase": "deploy"},
                   "timezone": {"default": "UTC", "phase": "deploy"}, "theme": {"default": "light", "phase": "build"}},
        "forms": {"disk": DISK}, **changes})


def _driver(folder: Path, services: list[Service] | None = None) -> ComputerDriverEnv:
    folder.mkdir(parents=True, exist_ok=True)
    names = ("driver.py", "requirements.txt", "install.sh", "settings.sh")
    for name, text in zip(names, ("class Desk: ...\n", "cua-driver==0.44.0\n", "tar xzf desk.tar.gz\n",
                                  "mkdir -p ~/.config/desk\n")):
        (folder / name).write_text(text)
    files = FileArtifactUniverse.put_bundled("@local/~/kit/desk__files", files={name: folder / name for name in names})
    omarchy = OmarchyInstall(
        python="requirements.txt", downloads=[Download(url=URL, sha256="cd" * 32, to="desk.tar.gz", arch="amd64"),
                                              Download(url=URL, sha256="ef" * 32, to="arm.tar.gz", arch="arm64")],
        run=[RunScript(script="install.sh"), RunScript(script="settings.sh", scope="session")],
        services=services or [Service(name="daemon", command="venv/bin/python driver.py"),
                              Service(name="driver", command=SERVE)])
    put = ComputerDriverEnv.put(
        id="@local/~/kit/desk", module="driver.py:Desk", files=files, port=18765, install={"linux/omarchy": omarchy},
        declaration=DriverDeclaration.model_validate({
            "name": "desk", "supports": OMARCHY_SUPPORTS,
            "platforms": [{"platform": {"os": "linux", "distribution": "omarchy", "arch": "amd64", "display": "wayland"},
                           "provides": [], "tools": ["computer_screenshot"], "features": {}}]}))
    return Env.get(put.id, put.version)


def _put(driver: ComputerDriverEnv, link: SandboxProvider, base: ComputerArtifact | None = None,
         **settings) -> ComputerEnv:
    return ComputerEnv.put(id="@local/~/kit/desktop", computer=base or _base(), driver=driver,
                           environment_name="desktop", composing=Composing(sandbox_provider=link),
                           computer_settings=settings or None, apps=[])


def test_an_omarchy_env_composes_a_thin_disk_on_its_base_built_from_omarchys_image_the_driver_as_user_units(
        docker, tmp_path):
    env = _put(_driver(tmp_path / "desk"), FakeLinux(), theme="dark")

    built = env.metadata["composed"]["disk"]["amd64"]
    assert built["status"] == "ready" and not built["reused"]
    assert built["image"] == docker.registry.held[built["key"].removeprefix("sha256:")]
    ((base_key, image, user, distribution, version),) = docker.builds
    assert (image, user, distribution, version) == (tmp_path / "omarchy.qcow2", "omarchy", "omarchy", "4.5.0")
    ((guest, base, script, files, labels),) = docker.composed
    assert guest is omarchy_disk.GUEST and base == f"agentenv-omarchy-base:{base_key[7:23]}"
    assert labels[composer.KEY_LABEL] == built["key"]
    prefix = "/opt/agentenv/drivers/desk"
    lines = script.splitlines()
    for line in [f'cp -a "$build/context/driver/." {prefix}/', f"fetch {URL} {'cd' * 32} {prefix}/desk.tar.gz",
                 f"python3 -m venv {prefix}/venv", f"(cd {prefix} && PREFIX={prefix} bash {prefix}/install.sh)",
                 "user=omarchy", f'(cd {prefix} && runuser -u "$user" -- env HOME="$home" PREFIX={prefix} '
                                 f"bash {prefix}/settings.sh)",
                 "systemctl --global enable --quiet agentenv-driver-desk-daemon.service",
                 "systemctl --global enable --quiet agentenv-driver-desk-driver.service",
                 "systemctl mask --quiet sshd.service sshd.socket"]:
        assert line in lines, line
    assert "printf %s 'export AGENTENV_SETTING_THEME=dark\n' > /opt/agentenv/build-environment" in script
    assert not any("arm.tar.gz" in line for line in lines)
    assert any(line.startswith(f"{prefix}/venv/bin/pip install") and f"-r {prefix}/requirements.txt" in line
               for line in lines)
    unit = omarchy_disk.unit("driver-desk-daemon", PurePosixPath(prefix), [f"{prefix}/venv/bin/python", "driver.py"],
                             {"MCP_PORT": 18765})
    assert f"printf %s {shlex.quote(unit)} > /etc/systemd/user/agentenv-driver-desk-daemon.service" in script
    assert f"ExecStart=/opt/agentenv/with-seed {prefix}/venv/bin/python driver.py" in unit.splitlines()
    assert {"provision.sh", "context/driver/driver.py", "context/driver/install.sh"} <= set(files)
    computer = env.computer_artifact()
    assert computer.os.distribution == "omarchy" and computer.forms.disk.init == "systemd"
    assert env.metadata["computer"]["disks"] == {"amd64": built["image"]}


def test_a_compose_of_the_same_inputs_reuses_the_disk_the_image_store_holds(docker, tmp_path):
    driver = _driver(tmp_path / "desk")
    _put(driver, FakeLinux())
    again = _put(driver, FakeLinux())

    assert again.metadata["composed"]["disk"]["amd64"]["reused"] and len(docker.composed) == 1


def test_a_machine_that_boots_no_linux_disk_composes_no_omarchy_disk(docker, tmp_path):
    env = _put(_driver(tmp_path / "desk"), FakeLinux(frozenset()))

    built = env.metadata["composed"]["disk"]["amd64"]
    assert built["status"] == "failed" and "boots linux/amd64 disks" in built["reason"]
    assert docker.builds == [] and docker.composed == []


def test_an_omarchy_computer_refuses_apps_for_now(local_stores, cli_routing, tmp_path):
    app = SimpleNamespace(id="@local/~/kit/notes", declaration=SimpleNamespace(supports=OMARCHY_SUPPORTS,
                                                                               install_steps=[]),
                          install={"linux/omarchy": Install(services=[Service(name="sync", command="sync")])})

    problems = gate.check_app(_base(), _driver(tmp_path / "desk"), app)

    assert problems == ["app: an Omarchy computer composes its driver alone for now; @local/~/kit/notes would need its "
                        "install composed onto Omarchy's disk, which agent-env doesn't do yet"]


@pytest.mark.parametrize("changes, error", [
    ({"forms": {"disk": DISK, "container": {"image": {"id": "x", "version": 1}, "init": "supervisord",
                                            "input_channel": "env"}}},
     "forms.container: a linux/omarchy computer's forms are disk"),
    ({"forms": {}}, "forms.disk is required: every linux/omarchy computer has one"),
    ({"forms": {"disk": {**DISK, "init": "supervisord"}}},
     "forms.disk.init: a linux/omarchy disk's init is systemd, not supervisord"),
    ({"forms": {"disk": {**DISK, "image": {"id": "x", "version": 1}}}},
     "forms.disk: a disk names an image or the source its base is built from, not both"),
    ({"forms": {"disk": {key: value for key, value in DISK.items() if key != "source"}}},
     "forms.disk: an Omarchy disk's base is built from Omarchy's own image on the machine that composes it, so it "
     "names that image's source and no image"),
    ({"forms": {"disk": {**{key: value for key, value in DISK.items() if key != "source"},
                         "image": {"id": "x", "version": 1}}}},
     "forms.disk: an Omarchy disk's base is built from Omarchy's own image"),
    ({"forms": {"disk": {**DISK, "source": {"url": "http://example.com/omarchy.qcow2", "sha256": "ab" * 32}}}},
     "url"),
], ids=["no-container", "disk-required", "systemd", "image-or-source", "names-its-source", "no-image", "https"])
def test_an_omarchy_base_is_a_disk_alone_built_from_its_source_and_started_by_systemd(
        local_stores, cli_routing, changes, error):
    with pytest.raises(ValidationError, match=error):
        _base(**changes)


def test_an_omarchy_install_names_no_apt_packages_which_a_linux_one_may():
    table = {"packages": {"apt": ["htop"]}, "services": [{"name": "driver", "command": SERVE}]}

    assert installs({"linux/omarchy": table})[1] == [
        "install.linux/omarchy: Value error, Omarchy has no apt: fetch what the layer needs with downloads, checked by "
        "sha256"]
    assert installs({"linux": table})[1] == []


def test_a_driver_service_that_runs_as_root_is_refused_naming_omarchys_systemd(docker, tmp_path):
    with pytest.raises(ValueError, match="driver: its service 'driver' runs as root, and @local/~/kit/omarchy-4.5's "
                                         "systemd runs every program as omarchy"):
        _put(_driver(tmp_path / "desk", services=[Service(name="driver", command=SERVE, scope="system")]),
             FakeLinux())


def test_a_disk_form_naming_no_source_is_written_without_one_so_an_agent_env_from_before_sources_reads_it():
    linux = DiskForm(init="supervisord", input_channel="nocloud", min_disk_gb=16).model_dump(by_alias=True)

    assert "source" not in linux and DiskForm(**DISK).model_dump(by_alias=True)["source"] == SOURCE


def test_a_time_zone_is_checked_against_ianas_before_the_computer_boots(local_stores, cli_routing):
    base = _base()

    assert gate.check_computer_settings(base, {"timezone": "America/Chicago"}) == ({"timezone": "America/Chicago"}, [])
    assert gate.check_computer_settings(base, {"timezone": "Mars/Base"})[1] == [
        "computer_settings.timezone: 'Mars/Base' isn't an IANA time zone, such as UTC or America/Chicago"]


def test_a_windows_disk_names_no_source(local_stores, cli_routing):
    with pytest.raises(ValidationError, match="a Windows disk is installed from Microsoft's media"):
        ComputerArtifact.put(id="@local/~/kit/windows-11", arch=["amd64"],
                             os={"type": "windows", "version": "11", "edition": "Enterprise Evaluation"},
                             session={"user": "user"}, forms={"disk": {**DISK, "init": "agentenv-session"}})


def test_an_omarchy_bases_disk_is_checked_when_its_built_not_at_put(local_stores, cli_routing):
    base = _base()

    (check,) = base.metadata["validated_images"]["disk"].values()
    assert check["status"] == "uninspected" and "built from its source when composed" in check["reason"]


def test_the_shipped_omarchy_base_is_omarchys_stable_image():
    import tomllib

    shipped = tomllib.loads((Path(composer.__file__).parents[4] / "computers" / "artifacts" / "omarchy-4.5" /
                             "artifact.toml").read_text())
    assert shipped["os"] == OMARCHY and shipped["forms"]["disk"]["init"] == "systemd"
    assert DiskSource(**shipped["forms"]["disk"]["source"]).url.endswith("/stable/4.5.0/479/omarchy-4.5.0.qcow2")


def test_an_omarchy_disk_boots_at_the_size_hyprland_wants_and_other_linux_disks_as_before():
    assert boot.entry_for(boot.DISK_SIZES, "linux/omarchy") == (4.0, 8192)
    assert boot.entry_for(boot.DISK_SIZES, "linux/ubuntu") == (boot.DISK_CPU, boot.DISK_MEMORY_MB)
    assert boot.entry_for(boot.DISK_SIZES, "windows") == (4.0, 8192)


def test_a_unit_quotes_what_systemd_would_otherwise_split_or_expand():
    text = omarchy_disk.unit("app-desk-sync", PurePosixPath("/opt/agentenv/apps/desk"),
                             ["/opt/agentenv/apps/desk/venv/bin/python", "sync.py", "--title", "100% $HOME's \"desk\""],
                             {"ENVIRONMENT_NAME": "desk", "GREETING": "hello world"})

    assert ('ExecStart=/opt/agentenv/with-seed /opt/agentenv/apps/desk/venv/bin/python sync.py --title '
            '"100%% $$HOME\'s \\"desk\\""') in text.splitlines()
    assert 'Environment="GREETING=hello world"' in text.splitlines()
    assert "Environment=ENVIRONMENT_NAME=desk" in text.splitlines()
    assert "WantedBy=graphical-session.target" in text.splitlines()


def test_a_service_runs_its_relative_program_from_the_layers_prefix():
    prefix = PurePosixPath("/opt/agentenv/drivers/desk")

    assert omarchy_disk.argv("venv/bin/python -m serve 'driver.py:Desk'", prefix) == [
        "/opt/agentenv/drivers/desk/venv/bin/python", "-m", "serve", "driver.py:Desk"]
    assert omarchy_disk.argv("/usr/bin/foot --server", prefix) == ["/usr/bin/foot", "--server"]


def test_omarchys_image_is_fetched_into_the_cache_once_and_refused_unless_its_sha256_matches(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    body, fetched = b"omarchy's image", []

    def urlopen(url, timeout):
        fetched.append(url)
        return io.BytesIO(body)

    monkeypatch.setattr(omarchy_disk.urllib.request, "urlopen", urlopen)
    right = DiskSource(url="https://example.com/omarchy.qcow2", sha256=hashlib.sha256(body).hexdigest())
    wrong = DiskSource(url="https://example.com/other.qcow2", sha256="00" * 32)

    path = omarchy_disk.source(right, lambda line: None)
    omarchy_disk.source(right, lambda line: None)
    with pytest.raises(RuntimeError, match=f"came with sha256 {hashlib.sha256(body).hexdigest()}, not {'00' * 32}"):
        omarchy_disk.source(wrong, lambda line: None)

    assert path.read_bytes() == body and fetched == [right.url, wrong.url]
    assert not list(omarchy_disk.source_path("00" * 32).parent.glob("*.part"))


class FakeBuilder:
    """A base's build on this machine: the launcher booting Omarchy's image, its guest over ssh, and the recipe's exit."""

    def __init__(self, monkeypatch, recipe_exit=0, recipe_err=""):
        self.launched, self.ran, self.converted, self.removed = None, [], None, []
        self.recipe_exit, self.recipe_err = recipe_exit, recipe_err
        monkeypatch.setattr(omarchy_disk, "launcher", lambda: "launcher:1")
        monkeypatch.setattr(thin_disk, "launched", self.launch)
        monkeypatch.setattr(thin_disk, "ssh_port", lambda container: 2222)
        monkeypatch.setattr(thin_disk, "ssh", self.ssh)
        monkeypatch.setattr(thin_disk, "compacted", lambda guest, port, keys, say: self.ran.append(("compact",)))
        monkeypatch.setattr(thin_disk, "shut_down", lambda guest, container, port, keys: self.ran.append(("off",)))
        monkeypatch.setattr(thin_disk, "converted", self.convert)
        monkeypatch.setattr(thin_disk.subprocess, "run", self.run)

    def launch(self, name, variables, mounts, image):
        self.launched = (name, variables, mounts, image)
        return "container"

    def ssh(self, port, keys, user, command, *, check=True, timeout=120, input=None):
        self.ran.append((user, command))
        code = self.recipe_exit if "recipe.sh" in command else 0
        return subprocess.CompletedProcess([], code, "", self.recipe_err if code else "")

    def convert(self, folder, image, backing=None, args=()):
        self.converted = (image, backing, args)
        (folder / "disk.img").write_text("whole")

    def run(self, argv, **kwargs):
        if argv[:3] == ["docker", "rm", "-f"]:
            self.removed.append(argv[3])
        return subprocess.CompletedProcess(argv, 0)


def test_a_base_is_built_by_booting_omarchys_image_with_roots_key_and_running_the_recipe(monkeypatch, tmp_path):
    builder = FakeBuilder(monkeypatch)
    keys, work, image = tmp_path / "keys", tmp_path / "work", tmp_path / "omarchy.qcow2"
    keys.mkdir()
    work.mkdir()
    (keys / "id.pub").write_text("ssh-ed25519 AAAA agentenv\n")

    omarchy_disk._built(work, image, keys, "omarchy", "omarchy", "4.5.0", lambda line: None)

    name, variables, mounts, launcher = builder.launched
    assert variables["VM_KEEP"] == "/out/disk.img" and variables["VM_DISK"] == "/in/source.qcow2"
    credential = base64.b64decode(variables["VM_CREDENTIALS"].removeprefix("tmpfiles.extra=")).decode()
    assert "f+ /root/.ssh/authorized_keys 0600 root root - ssh-ed25519 AAAA agentenv" in credential
    assert f"{image}:/in/source.qcow2:ro" in mounts and launcher == "launcher:1"
    assert [step for step in builder.ran if step[0] == "root"][-2:] == [
        ("root", "bash /root/agentenv-recipe/recipe.sh omarchy omarchy 4.5.0"), ("root", "rm -rf /root/agentenv-recipe")]
    assert builder.ran[-2:] == [("compact",), ("off",)]
    assert builder.converted == ("launcher:1", None, ("-v", f"{image}:/in/source.qcow2:ro"))
    assert (work / "base.img").read_text() == "whole" and (work / "out" / "guest").read_text() == "linux-desktop"
    assert builder.removed == ["container"]


def test_a_base_built_from_an_image_its_artifact_mislabels_is_refused(monkeypatch, tmp_path):
    FakeBuilder(monkeypatch, recipe_exit=omarchy_disk.MISLABELLED,
                recipe_err="recipe: this image is arch rolling, not omarchy 4.5.0 as its artifact declares\n")
    keys, work = tmp_path / "keys", tmp_path / "work"
    keys.mkdir()
    work.mkdir()
    (keys / "id.pub").write_text("ssh-ed25519 AAAA agentenv\n")

    with pytest.raises(RuntimeError, match="omarchy: recipe: this image is arch rolling, not omarchy 4.5.0"):
        omarchy_disk._built(work, tmp_path / "arch.qcow2", keys, "omarchy", "omarchy", "4.5.0", lambda line: None)
