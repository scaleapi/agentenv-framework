"""Computer envs: gate 1 refuses a base, a driver or a declaration that can't compose, naming every problem, before
anything is stored; the composer builds each form and arch on this machine's Docker, reuses an image of the same key
whichever env built it, records each result as it finishes, and writes the arches built as the env's computer. Docker
and the registry are faked."""

import io
import json
import logging
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts import docker_image
from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.store import get_artifact_store
from agent_env.cli import cli
from agent_env.env import Env
from agent_env.env import env as env_module
from agent_env.env.envs.computer import composer
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.install import Install, Service
from agent_env.env.envs.computer_driver import ComputerDriverEnv
from agent_env.plugins._bundles import ALIASES_FILE, InstalledBundle, installed_aliases
from agent_env.store.base import NotFoundError
from agent_env.store.image_store.image_store import OciRegistryImageStore
from agent_env.store.image_store.local_registry_image_store import LocalRegistryImageStore
from agent_env.store.image_store.oci_registry_credentials import RegistryAuth
from agent_env.utils.docker.build import DockerBuildError
from tst.util.oci_registry import UBUNTU_LAYER, Registry

OS = {"type": "linux", "distribution": "ubuntu", "version": "24.04", "display": "x11"}
CONFIG = {"resolution": {"default": "1280x800", "phase": "deploy"}, "timezone": {"default": "UTC", "phase": "deploy"},
          "theme": {"default": "light", "phase": "build"}}
SERVE = "venv/bin/python -m agentenv_protocol.computer.serve driver.py:Desk"
TOOLS = ["computer_screenshot", "computer_click", "computer_type", "computer_pointer_list"]


class FakeDocker:
    """This machine's Docker for the composer, on arm64 emulating amd64: each build recorded with its Dockerfile, its
    context's files and its programs, each push landing in the fake registry as an image of its arch."""

    def __init__(self, registry: Registry):
        self.registry, self.builds, self.fail, self.emulates, self.labels = registry, [], {}, {"linux/amd64"}, {}

    def build_image(self, dockerfile, context, tag, *, platform, labels=None, build_args=None):
        if platform in self.fail:
            raise DockerBuildError(self.fail[platform])
        self.builds.append({"platform": platform, "tag": tag, "dockerfile": dockerfile.read_text(),
                            "files": sorted(p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()),
                            "conf": next(context.glob("agentenv-*.conf")).read_text()})

    def label(self, local, platform, extra):
        self.labels[local] = extra

    def push(self, local, ref, store):
        self.registry.tags[ref] = self.registry.image(UBUNTU_LAYER, arch=local.rsplit("-", 1)[1],
                                                      labels=self.labels.get(local))

    def joined(self, index, images):
        manifests = {self.registry.manifests[image.rpartition("@")[2]]["config"]["digest"]: image.rpartition("@")[2]
                     for image in images}
        arches = {self._arch(config): digest for config, digest in manifests.items()}
        self.registry.tags[index] = self.registry.index(**arches)

    def _arch(self, config_digest):
        return json.loads(self.registry.blobs[config_digest])["architecture"]


@pytest.fixture
def docker(local_stores, cli_routing, monkeypatch):
    registry = Registry().serve(monkeypatch)
    fake = FakeDocker(registry)
    monkeypatch.setattr(composer, "build_image", fake.build_image)
    monkeypatch.setattr(composer, "push_local_image", fake.push)
    monkeypatch.setattr(composer, "_label", fake.label)
    monkeypatch.setattr(composer, "docker_login", lambda auth: None)
    monkeypatch.setattr(composer, "_joined", fake.joined)
    monkeypatch.setattr(composer, "pin_digest", registry.pin_digest)
    monkeypatch.setattr(docker_image, "pin_digest", registry.pin_digest)
    monkeypatch.setattr(composer, "host_platform", lambda: "linux/arm64")
    monkeypatch.setattr(composer, "_emulated_platforms", lambda: {"linux/arm64", *fake.emulates})
    monkeypatch.setattr(composer, "_installed", lambda local, layer: {"mss": "10.1.0"})
    monkeypatch.setattr(composer, "_protocol_source", lambda: None)
    monkeypatch.setattr(LocalRegistryImageStore, "ensure_repository", lambda self, repository: None)
    return fake


@pytest.fixture
def base(docker):
    index = docker.registry.index(amd64=docker.registry.image(UBUNTU_LAYER),
                                  arm64=docker.registry.image(UBUNTU_LAYER, arch="arm64"))
    image = get_artifact_store().put_document(DockerImageArtifact(id="base-image", description="",
                                                                  image_name=f"ghcr.io/acme/base:1@{index}"))
    return _base(image)


@pytest.fixture
def driver(docker, tmp_path):
    return _driver(tmp_path / "driver")


def _base(image, **changes):
    return ComputerArtifact.put(inspect=False, **{
        "id": "@local/~/kit/ubuntu-24.04", "arch": ["amd64", "arm64"], "os": OS, "session": {"user": "user"},
        "config": CONFIG, "forms": {"container": {"image": {"id": image.id, "version": image.version},
                                                  "init": "supervisord", "input_channel": "env"}}, **changes})


def _driver(folder, requirements="mss>=9,<11\n", supports=None, services=None, install=True, extra=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "driver.py").write_text("class Desk: ...\n")
    (folder / "requirements.txt").write_text(requirements)
    files = FileArtifactUniverse.put_bundled("@local/~/kit/desk__files", files={
        name: folder / name for name in ("driver.py", "requirements.txt")})
    if extra:
        files = FileArtifactUniverse.put(files.id, file_artifacts={**files.get_file_artifacts(), **extra})
    supports = supports or [{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}]
    platforms = [{"platform": {"os": "linux", "arch": arch, "display": "x11"}, "provides": [], "tools": TOOLS,
                  "features": {}} for arch in ("amd64", "arm64")]
    services = services or [Service(name="driver", command=SERVE)]
    return ComputerDriverEnv.put(
        id="@local/~/kit/desk", module="driver.py:Desk", files=files,
        declaration={"name": "desk", "supports": supports, "platforms": platforms},
        install={"linux": Install(python="requirements.txt", services=services)} if install else {}, port=18765)


def _put(base, driver, compose=True, **changes):
    return ComputerEnv.put(**{"id": "@local/~/kit/desktop", "computer": base, "driver": driver,
                              "environment_name": "desktop", "compose": compose, **changes})


def test_a_computer_env_is_composed_for_each_arch_and_written_with_the_computer_its_images_make(base, driver, docker,
                                                                                                local_stores):
    local_stores.set_image_store(OciRegistryImageStore("registry.example"))
    env = _put(base, driver, computer_settings={"timezone": "Europe/Paris", "theme": "dark"},
               display={"width": 1600, "height": 900})

    stored = Env.get(env.id)
    composed = stored.metadata["composed"]["container"]
    assert {arch: (r["status"], r["reused"], r["emulated"]) for arch, r in composed.items()} == {
        "amd64": ("unverified", False, True), "arm64": ("unverified", False, False)}
    assert composed["arm64"]["installed"] == {"mss": "10.1.0"} and composed["arm64"]["key"] != composed["amd64"]["key"]
    assert composed["arm64"]["image"].startswith("localhost:5000/agentenv-computers:")
    computer = ComputerArtifact.get(stored.metadata["computer"]["id"], stored.metadata["computer"]["version"])
    assert computer.id == "@local/~/kit/desktop__computer" and computer.arch == ["amd64", "arm64"]
    assert computer.image().image_name.startswith("localhost:5000/agentenv-computers:index-")
    assert {arch: r["status"] for arch, r in computer.metadata["validated_images"]["container"].items()} == {
        "amd64": "passed", "arm64": "passed"}
    assert (computer.config["resolution"].default, computer.config["timezone"].default) == ("1600x900", "Europe/Paris")
    assert stored.computer_settings == {"timezone": "Europe/Paris", "theme": "dark"}
    arm64 = next(build for build in docker.builds if build["platform"] == "linux/arm64")
    assert arm64["dockerfile"].splitlines() == [
        f"FROM {base.image().image_name}", "USER root", "COPY driver/ /opt/agentenv/drivers/desk/",
        "RUN python3 -m venv /opt/agentenv/drivers/desk/venv && /opt/agentenv/drivers/desk/venv/bin/pip install "
        "--no-cache-dir -r /opt/agentenv/drivers/desk/requirements.txt && /opt/agentenv/drivers/desk/venv/bin/pip list "
        "--format=json > /opt/agentenv/drivers/desk/installed.json",
        "COPY agentenv-desk.conf /etc/supervisor/conf.d/agentenv-desk.conf",
        "RUN test -d /opt/agentenv/drivers/desk && test -f /etc/supervisor/conf.d/agentenv-desk.conf && test -x "
        "/opt/agentenv/drivers/desk/venv/bin/python",
        'ENV AGENTENV_SETTING_THEME="dark"', "USER user", f"LABEL dev.agentenv.computer-key={composed['arm64']['key']}"]
    assert arm64["files"] == ["Dockerfile", "agentenv-desk.conf", "driver/driver.py", "driver/requirements.txt"]
    assert arm64["conf"] == (
        "[program:desk-driver]\ncommand=/opt/agentenv/drivers/desk/venv/bin/python -m agentenv_protocol.computer.serve "
        'driver.py:Desk\ndirectory=/opt/agentenv/drivers/desk\nenvironment=MCP_PORT="18765"\nautorestart=true\n'
        "redirect_stderr=true\nstdout_logfile=/var/log/agentenv/desk-driver.log\n")
    assert ComputerEnv.from_dict(stored.to_dict()).to_dict() == stored.to_dict()


def test_an_env_that_differs_only_in_what_doesnt_change_the_image_reuses_the_one_another_env_built(base, driver,
                                                                                                    docker, monkeypatch):
    auth, docker_calls = RegistryAuth("localhost:5000", "user", "token"), []
    monkeypatch.setattr(LocalRegistryImageStore, "auth", lambda self, ref: auth)
    monkeypatch.setattr(composer, "docker_login", lambda signed_in: docker_calls.append(("login", signed_in)))
    monkeypatch.setattr(composer, "_joined", lambda index, images: (docker_calls.append(("join", index)),
                                                                    docker.joined(index, images)))
    announced = []
    first = _put(base, driver, on_build=lambda: announced.append("first"))
    docker_calls.clear()
    second = _put(base, driver, id="@local/~/kit/other", environment_name="other", display={"width": 640, "height": 480},
                  network={"mode": "allowlist"}, on_build=lambda: announced.append("second"))

    built, reused = (Env.get(env.id).metadata["composed"]["container"] for env in (first, second))
    assert len(docker.builds) == 2 and all(reused[arch]["reused"] for arch in reused) and announced == ["first"]
    assert {arch: reused[arch]["image"] for arch in reused} == {arch: built[arch]["image"] for arch in built}
    assert {arch: (r["installed"], r["emulated"]) for arch, r in reused.items()} == {
        "amd64": ({"mss": "10.1.0"}, True), "arm64": ({"mss": "10.1.0"}, False)}
    index = ComputerArtifact.get(f"{second.id}__computer").image().image_name.partition("@")[0]
    assert docker_calls == [("login", auth), ("join", index)]


@pytest.mark.parametrize("change", ["requirements", "build-setting", "service", "display-read-at-build"])
def test_what_changes_the_image_changes_its_key_and_builds_it_again(base, driver, docker, tmp_path, change):
    if change == "display-read-at-build":
        base = _base(base.image(), config={**CONFIG, "resolution": {"default": "1280x800", "phase": "build"}})
    first = Env.get(_put(base, driver).id).metadata["composed"]["container"]["arm64"]["key"]
    if change == "requirements":
        driver = _driver(tmp_path / "changed", requirements="mss>=10,<11\n")
    elif change == "service":
        driver = _driver(tmp_path / "changed", services=[Service(name="driver", command=f"{SERVE} --verbose")])
    settings = {"theme": "dark"} if change == "build-setting" else {}
    display = {"width": 1600, "height": 900} if change == "display-read-at-build" else None

    again = Env.get(_put(base, driver, computer_settings=settings, display=display).id).metadata["composed"][
        "container"]["arm64"]

    assert again["key"] != first and again["reused"] is False and len(docker.builds) == 4
    if display:
        assert 'ENV AGENTENV_SETTING_RESOLUTION="1600x900"' in docker.builds[-1]["dockerfile"]


def test_a_pair_that_fails_is_recorded_with_its_log_and_the_rest_are_built_and_written(base, driver, docker):
    docker.fail["linux/amd64"] = "#7 ERROR: process \"/bin/sh -c pip install\" did not complete successfully"

    env = Env.get(_put(base, driver).id)

    amd64 = env.metadata["composed"]["container"]["amd64"]
    assert (amd64["status"], amd64["stage"]) == ("failed", "build")
    assert amd64["reason"] == 'the build failed: #7 ERROR: process "/bin/sh -c pip install" did not complete successfully'
    log = FileArtifact.get(amd64["log"]["id"], amd64["log"]["version"])
    assert b"did not complete successfully" in log.load()
    assert env.metadata["composed"]["container"]["arm64"]["status"] == "unverified"
    assert ComputerArtifact.get(env.metadata["computer"]["id"]).arch == ["arm64"]


def test_a_driver_file_named_outside_its_folder_fails_the_build_before_anything_is_written_or_built(base, docker,
                                                                                                    tmp_path):
    escape = FileArtifact.put_bytes("@local/~/kit/escape", description="", filename="agentenv-escape", content=b"x",
                                    content_type="text/plain")
    driver = _driver(tmp_path / "driver", extra={"../../agentenv-escape": escape})

    env = Env.get(_put(base, driver).id)

    assert {(r["stage"], r["reason"]) for r in env.metadata["composed"]["container"].values()} == {
        ("build", "ValueError: driver @local/~/kit/desk: '../../agentenv-escape' would be written outside its folder")}
    assert docker.builds == []


def test_a_machine_that_cant_emulate_the_other_arch_fails_that_arch_alone(base, driver, docker):
    docker.emulates = set()

    env = Env.get(_put(base, driver).id)

    amd64 = env.metadata["composed"]["container"]["amd64"]
    assert amd64["reason"] == "this machine can't build linux/amd64: its Docker has no emulation for it"
    assert env.metadata["composed"]["container"]["arm64"]["status"] == "unverified"
    assert [build["platform"] for build in docker.builds] == ["linux/arm64"]


def test_a_build_that_cant_run_the_other_archs_programs_says_its_emulation_is_missing(base, driver, docker):
    docker.fail["linux/amd64"] = "#6 0.01 exec /bin/sh: exec format error\n#6 ERROR: process did not complete successfully"

    amd64 = Env.get(_put(base, driver).id).metadata["composed"]["container"]["amd64"]

    assert amd64["reason"].startswith("this machine can't run linux/amd64 programs: its Docker has no emulation for it:")
    assert amd64["log"]


def test_a_base_built_from_its_context_is_built_here_for_each_arch_and_keyed_by_its_sources(driver, docker,
                                                                                            monkeypatch):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id="base-image", description="", image_name="local/base-1234:abcd", build_context_object_url="file:///c.tar.gz",
        dockerfile_path="Dockerfile", source_digest="sha256:" + "c" * 64))
    built_for = []
    monkeypatch.setattr(composer, "build_for", lambda image, platform: built_for.append(platform) or
                        f"{image.image_name}-{platform.rsplit('/', 1)[1]}")

    env = Env.get(_put(_base(image), driver).id)
    again = Env.get(_put(_base(image), driver, id="@local/~/kit/again", environment_name="again").id)

    assert sorted(built_for) == ["linux/amd64", "linux/arm64"]
    assert {r["reused"] for r in again.metadata["composed"]["container"].values()} == {True}
    assert {build["dockerfile"].splitlines()[0] for build in docker.builds} == {
        "FROM local/base-1234:abcd-amd64", "FROM local/base-1234:abcd-arm64"}
    assert {r["status"] for r in env.metadata["composed"]["container"].values()} == {"unverified"}


def test_a_checkout_composes_its_own_protocol_into_the_drivers_venv(base, driver, docker, tmp_path, monkeypatch):
    source = tmp_path / "protocol"
    (source / "src" / "agentenv_protocol").mkdir(parents=True)
    (source / "src" / "agentenv_protocol" / "__init__.py").write_text("")
    (source / "pyproject.toml").write_text("[project]\nname = 'agentenv-framework-protocol'\n")
    monkeypatch.setattr(composer, "_protocol_source", lambda: source)

    _put(base, driver)

    dockerfile = docker.builds[0]["dockerfile"]
    assert "COPY protocol /tmp/agentenv-protocol" in dockerfile
    assert "pip install --no-cache-dir /tmp/agentenv-protocol -r /opt/agentenv/drivers/desk/requirements.txt" in dockerfile
    assert "protocol/pyproject.toml" in docker.builds[0]["files"]


def test_the_pairs_left_to_build_are_those_failed_or_interrupted(base, driver, docker):
    docker.fail["linux/amd64"] = "boom"
    env = Env.get(_put(base, driver).id)
    composed = env.metadata["composed"]
    composed["container"]["arm64"] = {"status": "building"}
    env.metadata = {**env.metadata, "composed": composed}

    assert composer.pending(env) == [("container", "amd64"), ("container", "arm64")]
    docker.fail.clear()
    composer.compose(env, composer.pending(env))

    assert {r["status"] for r in Env.get(env.id).metadata["composed"]["container"].values()} == {"unverified"}
    assert composer.pending(Env.get(env.id)) == []


def test_env_build_builds_what_a_put_left_and_prints_each_pair(base, driver, docker, caplog):
    caplog.set_level(logging.ERROR, logger=composer.__name__)  # live logging would put pytest's stdout back mid-command
    docker.fail["linux/amd64"] = "boom"
    env = _put(base, driver)
    runner = CliRunner()

    failed = runner.invoke(cli, ["env", "build", "--id", env.id])
    docker.fail.clear()
    built = runner.invoke(cli, ["env", "build", "--id", env.id])
    again = runner.invoke(cli, ["env", "build", "--id", env.id])

    assert failed.exit_code == 1 and "container/amd64: failed (the build failed: boom)" in failed.output
    assert built.exit_code == 0 and "container/amd64: unverified (localhost:5000/agentenv-computers:" in built.output
    assert "container/arm64: unverified (localhost:5000" in built.output and len(docker.builds) == 2
    assert again.exit_code == 0 and len(docker.builds) == 2
    assert ComputerArtifact.get(Env.get(env.id).metadata["computer"]["id"]).arch == ["amd64", "arm64"]


def test_env_build_refuses_an_env_that_isnt_a_computer_env(base, driver, docker):
    refused = CliRunner().invoke(cli, ["env", "build", "--id", driver.id])

    assert refused.exit_code == 1 and "is a computer_driver env; only a computer env is built" in refused.output


@pytest.mark.asyncio
async def test_a_computer_env_doesnt_deploy_yet_and_says_how_far_it_got(base, driver, docker):
    env = _put(base, driver)

    with pytest.raises(NotImplementedError, match=r"agent-env can't boot yet; its builds are \{'container/amd64': "
                                                  r"'unverified', 'container/arm64': 'unverified'\}"):
        await Env.get(env.id).deploy()


@pytest.mark.parametrize("changes, problem", [
    ({"apps": [{"env": "mail"}]}, "apps: computer app envs aren't supported yet"),
    ({"display": {"width": 0, "height": 800}}, "display is { width = <n>, height = <n> }, each 1 to 32767 pixels"),
    ({"display": {"width": 1280}}, "display is { width = <n>, height = <n> }"),
    ({"display": {"width": 1280, "height": 800}, "computer_settings": {"resolution": "800x600"}},
     "display and the base's resolution setting both set the screen's size; set display alone"),
    ({"network": {"mode": "allow_all", "allow_hosts": ["x.org"]}},
     "network: allow_all permits everything; entries would be silently ignored"),
    ({"network": {"mode": "allowlist", "ports": [80]}}, "network is { mode, allow_hosts, allow_cidrs }"),
    ({"computer_settings": {"wallpaper": "x"}},
     "computer_settings.wallpaper: @local/~/kit/ubuntu-24.04 declares no setting 'wallpaper'"),
    ({"computer_settings": {"timezone": 3}}, "computer_settings.timezone must be a str, like its default 'UTC', not 3"),
    ({"computer_settings": ["timezone"]}, "computer_settings must be a table of the computer's setting values"),
], ids=["apps", "display-zero", "display-side", "display-and-resolution", "network", "network-key",
        "settings-name", "settings-type", "settings-table"])
def test_what_the_env_declares_wrong_is_refused_before_anything_is_stored(base, driver, docker, changes, problem):
    with pytest.raises(ValueError, match="computer env '@local/~/kit/desktop'") as refused:
        _put(base, driver, **changes)

    assert problem in str(refused.value)
    with pytest.raises(NotFoundError):
        Env.get("@local/~/kit/desktop")
    assert docker.builds == []


@pytest.mark.parametrize("make, problem", [
    (lambda tmp: _driver(tmp, supports=[{"os": {"type": "linux", "display": ["wayland"]}, "arch": ["amd64"]}]),
     "driver: @local/~/kit/desk supports no platform @local/~/kit/ubuntu-24.04 is (linux ubuntu 24.04 x11); it "
     "supports linux wayland amd64"),
    (lambda tmp: _driver(tmp, supports=[{"os": {"type": "linux", "distribution": "debian", "display": ["x11"]},
                                         "arch": ["amd64"]}]), "supports no platform"),
    (lambda tmp: _driver(tmp, install=False), "driver: @local/~/kit/desk declares no install for linux"),
    (lambda tmp: _driver(tmp, services=[Service(name="driver", command=SERVE, scope="system")]),
     "driver: its service 'driver' runs as root, and @local/~/kit/ubuntu-24.04's supervisord runs every program as user"),
], ids=["display", "distribution", "no-install", "system-service"])
def test_a_driver_that_cant_compose_with_the_base_is_refused(base, docker, tmp_path, make, problem):
    with pytest.raises(ValueError) as refused:
        _put(base, make(tmp_path / "driver"))

    assert problem in str(refused.value)


def test_a_base_and_driver_with_no_arch_in_common_and_an_image_that_cant_be_got_are_refused(docker, tmp_path):
    image = get_artifact_store().put_document(DockerImageArtifact(id="base-image", description="",
                                                                  image_name="local/base:1"))
    driver = _driver(tmp_path / "driver", supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["arm64"]}])

    with pytest.raises(ValueError) as refused:
        _put(_base(image, arch=["amd64"]), driver)

    assert ("driver: no architecture both run: @local/~/kit/ubuntu-24.04 runs amd64, and @local/~/kit/desk arm64"
            in str(refused.value))
    assert ("computer: @local/~/kit/ubuntu-24.04's container image 'local/base:1' is in no registry and has no build "
            "context to build it from" in str(refused.value))


def test_the_platforms_this_machine_builds_are_read_from_its_builder(monkeypatch):
    inspected = ("Name:          desktop-linux\nDriver:        docker\nNodes:\nName:             desktop-linux\n"
                 "Platforms:        linux/arm64, linux/amd64, linux/amd64/v2, linux/riscv64*\nLabels:\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=inspected, returncode=0))

    assert composer._emulated_platforms() == {"linux/arm64", "linux/amd64", "linux/amd64/v2", "linux/riscv64"}


@pytest.mark.parametrize("direct_url, expected", [
    ('{{"url": "file://{folder}", "dir_info": {{"editable": true}}}}', "folder"),
    (None, None),
    ('{{"url": "https://files.pythonhosted.org/agentenv_framework_protocol-0.1.381.whl"}}', None),
], ids=["source", "index", "url"])
def test_the_protocol_is_composed_from_its_source_only_when_installed_from_one(tmp_path, monkeypatch, direct_url,
                                                                                expected):
    (tmp_path / "pyproject.toml").write_text("")
    text = direct_url.format(folder=tmp_path) if direct_url else None
    monkeypatch.setattr(composer.importlib.metadata, "distribution",
                        lambda name: SimpleNamespace(read_text=lambda file: text))

    assert composer._protocol_source() == (tmp_path if expected else None)


def test_the_packages_a_build_installed_are_read_from_its_image_without_running_it(monkeypatch):
    listed = json.dumps([{"name": "mss", "version": "10.2.0"},
                         {"name": "agentenv-framework-protocol", "version": "0.1.382"}]).encode()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        info = tarfile.TarInfo("installed.json")
        info.size = len(listed)
        tar.addfile(info, io.BytesIO(listed))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[:2])
        if argv[1] == "create":
            return SimpleNamespace(stdout="c0ffee\n", returncode=0)
        return SimpleNamespace(stdout=archive.getvalue(), returncode=0)

    monkeypatch.setattr(subprocess, "run", run)

    assert composer._installed("agentenv-computer:abc-arm64", "reference") == {
        "mss": "10.2.0", "agentenv-framework-protocol": "0.1.382"}
    assert calls == [["docker", "create"], ["docker", "cp"], ["docker", "rm"]]


def test_a_script_or_binary_in_the_drivers_files_stays_executable_in_the_image(base, docker, tmp_path, monkeypatch):
    folder = tmp_path / "scripted"
    folder.mkdir()
    (folder / "driver.py").write_text("class Desk: ...\n")
    (folder / "start").write_text("#!/bin/sh\nexec venv/bin/python -m agentenv_protocol.computer.serve driver.py:Desk\n")
    (folder / "helper").write_bytes(b"\x7fELF\x02\x01\x01")
    modes = {}
    monkeypatch.setattr(composer, "build_image", lambda dockerfile, context, tag, **kw: modes.update(
        {path.name: path.stat().st_mode & 0o111 for path in (context / "driver").iterdir()}))
    files = FileArtifactUniverse.put_bundled("@local/~/kit/scripted__files", files={
        name: folder / name for name in ("driver.py", "start", "helper")})
    driver = ComputerDriverEnv.put(
        id="@local/~/kit/scripted", module="driver.py:Desk", files=files, declaration=_driver(tmp_path / "d").declaration,
        install={"linux": Install(services=[Service(name="driver", command="./start")])}, port=18765)

    _put(base, driver)

    assert modes == {"driver.py": 0, "start": 0o111, "helper": 0o111}


def test_a_registry_base_is_signed_in_to_before_its_image_is_built_on_and_the_store_before_the_index(base, driver,
                                                                                                     docker, monkeypatch):
    signed = []
    monkeypatch.setattr(composer, "docker_login", signed.append)

    _put(base, driver)

    assert len(signed) == 3 and len(docker.builds) == 2


def test_env_build_writes_the_computer_again_when_it_isnt_over_every_pair_built(base, driver, docker, caplog):
    caplog.set_level(logging.ERROR, logger=composer.__name__)
    stored = Env.get(_put(base, driver).id)
    stale = {**stored.metadata["computer"], "images": {"arm64": stored.metadata["computer"]["images"]["arm64"]}}
    stored.update_metadata({**stored.metadata, "computer": stale})

    assert not composer.current(Env.get(stored.id))
    rebuilt = CliRunner().invoke(cli, ["env", "build", "--id", stored.id])

    assert rebuilt.exit_code == 0 and composer.current(Env.get(stored.id)) and len(docker.builds) == 2
    assert Env.get(stored.id).metadata["computer"]["version"] == stale["version"] + 1


def test_env_build_by_an_installed_alias_composes_under_the_id_of_the_env_it_names(base, driver, docker, local_stores,
                                                                                       tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=composer.__name__)
    (tmp_path / "kit").mkdir()
    (tmp_path / "kit" / ALIASES_FILE).write_text('[envs]\ndesktop = ["office"]\n')
    kit = InstalledBundle("kit", "acme-kit", "1.0", "acme_kit", tmp_path / "kit", None, None)
    monkeypatch.setattr(env_module, "installed_aliases", lambda table: installed_aliases((kit,), table))
    env = _put(base, driver, id="@local/acme-kit/kit/desktop", compose=False)
    local_stores.set_image_store(OciRegistryImageStore("registry.example"))

    built = CliRunner().invoke(cli, ["env", "build", "--id", "office"])

    stored = Env.get(env.id)
    assert built.exit_code == 0 and stored.metadata["computer"]["id"] == "@local/acme-kit/kit/desktop__computer"
    images = [r["image"] for r in stored.metadata["composed"]["container"].values()]
    assert all(ref.startswith("localhost:5000/") for ref in [*images, ComputerArtifact.get(
        "@local/acme-kit/kit/desktop__computer").image().image_name])


def test_builds_of_one_env_at_once_keep_each_others_pairs_and_a_failure_doesnt_replace_a_built_one(base, driver,
                                                                                                    docker,
                                                                                                    monkeypatch):
    env = _put(base, driver, compose=False)
    first, second, third = (Env.get(env.id) for _ in range(3))

    composer.compose(first, [("container", "amd64")])
    composer.compose(second, [("container", "arm64")])
    monkeypatch.setattr(composer, "_served", lambda ref, store: None)
    docker.fail["linux/arm64"] = "boom"
    composer.compose(third, [("container", "arm64")])

    stored = Env.get(env.id)
    composed = stored.metadata["composed"]["container"]
    assert {arch: r["status"] for arch, r in composed.items()} == {"amd64": "unverified", "arm64": "unverified"}
    assert stored.metadata["computer"]["images"] == {arch: r["image"] for arch, r in composed.items()}
