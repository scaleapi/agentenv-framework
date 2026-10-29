"""Computer envs: gate 1 refuses a base, a driver or a declaration that can't compose, naming every problem, before
anything is stored; the composer builds each form and arch on this machine's Docker, reuses an image of the same key
whichever env built it, records each result as it finishes, and writes the arches built as the env's computer. Docker
and the registry are faked."""

import asyncio
import io
import json
import logging
import re
import subprocess
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from agentenv_protocol.computer import COMPUTER_DISPLAY, COMPUTER_RECORDING
from agentenv_protocol.computer.conformance import CheckResult, ProbeReport
from agentenv_protocol.tool_access import DISABLE_TOOL_EXTENSION_URI
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
from agent_env.env.envs.computer.utils import boot, composer, gate, inside
from agent_env.env.envs.computer.utils.composer import _probe
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.app_env import ComputerAppEnv
from agent_env.env.envs.computer.utils.install import Install, Service
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
from agent_env.plugins._bundles import ALIASES_FILE, InstalledBundle, installed_aliases
from agent_env.providers.env_providers import env_computer_provider
from agent_env.providers.env_providers.env_provider import DeployRefused
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from agent_env.providers.sandbox_providers.local_docker import provider as local_sandbox
from agent_env.store.base import NotFoundError
from agent_env.task import teardown
from agent_env.task_step.context import TaskStepContext
from agent_env.store.image_store.image_store import OciRegistryImageStore
from agent_env.store.image_store.local_registry_image_store import LocalRegistryImageStore
from agent_env.store.image_store.oci_registry_credentials import RegistryAuth
from agent_env.utils.docker.build import DockerBuildError
from tst.unit.computer_home import Computer
from tst.util.oci_registry import UBUNTU_LAYER, Registry

OS = {"type": "linux", "distribution": "ubuntu", "version": "24.04", "display": "x11"}
CONFIG = {"resolution": {"default": "1280x800", "phase": "deploy"}, "timezone": {"default": "UTC", "phase": "deploy"},
          "theme": {"default": "light", "phase": "build"}}
SERVE = "venv/bin/python -m agentenv_protocol.computer.serve driver.py:Desk"
CARD_EXTENSION = "urn:agentenv:computer/v1"
CARD = {"name": "desktop", "capabilities": {"extensions": [{"uri": CARD_EXTENSION}]}}
TOOLS = ["computer_screenshot", "computer_click", "computer_type", "computer_pointer_list"]


class FakeDocker:
    """This machine's Docker for the composer, on arm64 emulating amd64: each build recorded with its Dockerfile, its
    context's files and its programs, each push landing in the fake registry as an image of its arch, and each probe
    recorded by the arch and image it booted."""

    def __init__(self, registry: Registry):
        self.registry, self.builds, self.fail, self.emulates, self.labels = registry, [], {}, {"linux/amd64"}, {}
        self.probed, self.probe_fails = [], {}

    async def probe(self, links, env, driver, image):
        arch = self._arch(self.registry.manifests[image.rpartition("@")[2]]["config"]["digest"])
        self.probed.append((arch, image))
        if arch in self.probe_fails:
            checks = [CheckResult(check="P4", name="clicks show their effect", status="failed", detail=self.probe_fails[arch])]
        else:
            checks = [CheckResult(check="P0", name="the card", status="passed", extensions=[CARD_EXTENSION])]
        return ProbeReport.of("full", "desk", checks, [CARD_EXTENSION]), CARD, "LocalSandboxProvider"

    def build_image(self, dockerfile, context, tag, *, platform, labels=None, build_args=None):
        if platform in self.fail:
            raise DockerBuildError(self.fail[platform])
        self.builds.append({"platform": platform, "tag": tag, "dockerfile": dockerfile.read_text(),
                            "files": sorted(p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()),
                            "conf": next(context.glob("agentenv-*.conf")).read_text(),
                            "confs": {conf.name: conf.read_text() for conf in context.glob("agentenv-*.conf")}})

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
    monkeypatch.setattr(local_sandbox, "host_platform", lambda: "linux/arm64")
    monkeypatch.setattr(composer, "_probe", fake.probe)
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
        "amd64": ("unverified", False, True), "arm64": ("ready", False, False)}
    assert composed["amd64"]["reason"] == "no configured sandbox provider runs linux/amd64 natively to probe it"
    assert docker.probed == [("arm64", composed["arm64"]["image"])] and composed["arm64"]["card"] == CARD
    probe = composed["arm64"]["probe"]
    assert (probe["level"], probe["passed"], probe["provider"], probe["extensions"]) == (
        "full", True, "LocalSandboxProvider", {CARD_EXTENSION: "verified"})
    assert b'"passed": true' in FileArtifact.get(probe["report"]["id"], probe["report"]["version"]).load()
    assert composed["arm64"]["installed"] == {"mss": "10.1.0"} and composed["arm64"]["key"] != composed["amd64"]["key"]
    assert composed["arm64"]["image"].startswith("localhost:5000/agentenv-computers:")
    computer = ComputerArtifact.get(stored.metadata["computer"]["id"], stored.metadata["computer"]["version"])
    assert computer.id == "@local/~/kit/desktop__computer" and computer.arch == ["amd64", "arm64"]
    assert computer.image().image_name.startswith("localhost:5000/agentenv-computers:index-")
    assert {arch: r["status"] for arch, r in computer.metadata["validated_images"]["container"].items()} == {
        "amd64": "passed", "arm64": "passed"}
    assert computer.metadata["validated_boot"] == {"container": {"arm64": probe}}
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
    assert env.metadata["composed"]["container"]["arm64"]["status"] == "ready"
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
    assert env.metadata["composed"]["container"]["arm64"]["status"] == "ready"
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
    assert {arch: r["status"] for arch, r in env.metadata["composed"]["container"].items()} == {
        "amd64": "unverified", "arm64": "ready"}


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


@pytest.mark.parametrize("left", [{"status": "building"}, {"status": "unverified", "image": "built", "key": "k"}],
                         ids=["interrupted", "unprobed"])
def test_the_pairs_left_to_compose_are_those_failed_interrupted_or_unprobed_on_a_native_arch(base, driver, docker,
                                                                                             left):
    docker.fail["linux/amd64"] = "boom"
    env = Env.get(_put(base, driver).id)
    composed = env.metadata["composed"]
    composed["container"]["arm64"] = left
    env.metadata = {**env.metadata, "composed": composed}

    assert composer.pending(env) == [("container", "amd64"), ("container", "arm64")]
    docker.fail.clear()
    composer.compose(env, composer.pending(env))

    assert {arch: r["status"] for arch, r in Env.get(env.id).metadata["composed"]["container"].items()} == {
        "amd64": "unverified", "arm64": "ready"}
    assert composer.pending(Env.get(env.id)) == []


def test_the_probe_boots_a_build_at_the_envs_settings_as_a_deploy_does(base, driver, docker, monkeypatch):
    env = _put(base, driver, compose=False, computer_settings={"timezone": "Europe/Paris"},
               display={"width": 1600, "height": 900})
    booted = []

    async def fake_boot(link, **kwargs):
        booted.append(kwargs["environment"])
        return boot.Booted(sandbox=None, url="http://computer", card=CARD)

    monkeypatch.setattr(boot, "boot", fake_boot)
    monkeypatch.setattr(boot, "discard", AsyncMock())
    monkeypatch.setattr(composer, "run_probe", AsyncMock(return_value="report"))

    assert asyncio.run(_probe([SimpleNamespace()], env, driver, "img")) == (
        "report", CARD, "SimpleNamespace")
    assert {k: v for k, v in booted[0].items() if k.startswith("AGENTENV_SETTING_")} == {
        "AGENTENV_SETTING_RESOLUTION": "1600x900", "AGENTENV_SETTING_TIMEZONE": "Europe/Paris"}


def test_a_build_whose_probe_fails_or_that_never_gets_ready_fails_at_the_probe_and_isnt_deployed(base, driver, docker,
                                                                                                monkeypatch):
    docker.probe_fails["arm64"] = "the click never landed"
    failed = Env.get(_put(base, driver).id).metadata["composed"]["container"]["arm64"]

    async def never_ready(*args):
        raise RuntimeError("LocalSandboxProvider: the computer at http://127.0.0.1:1 wasn't ready within 180 s "
                           "(not ready: display: no X)")

    async def overran(*args):
        raise TimeoutError

    monkeypatch.setattr(composer, "_probe", never_ready)
    unbooted = Env.get(_put(base, driver, id="@local/~/kit/other", environment_name="other").id).metadata[
        "composed"]["container"]["arm64"]
    monkeypatch.setattr(composer, "_probe", overran)
    stuck = Env.get(_put(base, driver, id="@local/~/kit/stuck", environment_name="stuck").id).metadata[
        "composed"]["container"]["arm64"]

    assert (failed["status"], failed["stage"], failed["reason"]) == (
        "failed", "probe", "the probe failed: P4 clicks show their effect: the click never landed")
    assert not failed["probe"]["passed"] and failed["image"]
    assert (unbooted["status"], unbooted["stage"]) == ("failed", "probe")
    assert unbooted["reason"].startswith("it didn't boot: LocalSandboxProvider: the computer at http://127.0.0.1:1 "
                                         "wasn't ready within 180 s")
    assert (stuck["status"], stuck["stage"], stuck["reason"]) == (
        "failed", "probe", f"the probe didn't finish within {composer._PROBE_TIMEOUT_S} s")


@pytest.mark.parametrize("result, stored, replaces", [
    ({"status": "unverified"}, {"status": "ready"}, False),
    ({"status": "failed", "stage": "build"}, {"status": "unverified"}, False),
    ({"status": "building"}, {"status": "unverified"}, False),
    ({"status": "failed", "stage": "probe"}, {"status": "unverified"}, True),
    ({"status": "failed", "stage": "probe"}, {"status": "ready"}, False),
    ({"status": "ready"}, {"status": "failed", "stage": "probe"}, True),
    ({"status": "building"}, {"status": "failed", "stage": "build"}, True),
], ids=["unprobed-over-ready", "build-failure-over-built", "building-over-built", "probe-failure-over-unprobed",
        "probe-failure-over-ready", "ready-over-probe-failure", "building-over-build-failure"])
def test_a_result_takes_a_pairs_place_only_when_it_got_at_least_as_far(result, stored, replaces):
    assert (composer._rank(result) >= composer._rank(stored)) is replaces


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
    assert built.exit_code == 0 and ("container/amd64: unverified (no configured sandbox provider runs linux/amd64 "
                                     "natively to probe it)") in built.output
    assert "container/arm64: ready (localhost:5000" in built.output and len(docker.builds) == 2
    assert again.exit_code == 0 and len(docker.builds) == 2
    assert ComputerArtifact.get(Env.get(env.id).metadata["computer"]["id"]).arch == ["amd64", "arm64"]


def test_env_build_refuses_an_env_that_isnt_a_computer_env_or_an_app_naming_no_computer(base, driver, docker,
                                                                                        tmp_path):
    refused = CliRunner().invoke(cli, ["env", "build", "--id", driver.id])
    alone = CliRunner().invoke(cli, ["env", "build", "--id", _app(None, None, tmp_path / "mail").id])

    assert refused.exit_code == 1 and "is a computer_driver env; only a computer env is built" in refused.output
    assert alone.exit_code == 1 and "names no computer and driver, so it has nothing to build alone" in alone.output


@pytest.mark.asyncio
async def test_a_computer_env_with_no_ready_build_is_refused_at_deploy_naming_its_builds(base, driver, docker):
    docker.probe_fails["arm64"] = "the click never landed"
    env = _put(base, driver)

    with pytest.raises(DeployRefused, match=r"has no ready build to deploy; its builds are \{'container/amd64': "
                                            r"'unverified', 'container/arm64': 'failed'\}"):
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
    assert {arch: r["status"] for arch, r in composed.items()} == {"amd64": "unverified", "arm64": "ready"}
    assert stored.metadata["computer"]["images"] == {arch: r["image"] for arch, r in composed.items()}


DRIVER_CARD = {"name": "desktop", "capabilities": {"extensions": [
    {"uri": COMPUTER_DISPLAY, "params": {"endpoint": "/computer/display", "methods": {"get": {"method": "GET"}}}},
    {"uri": DISABLE_TOOL_EXTENSION_URI, "params": {"endpoint": "/tools/disable", "methods": {"disable": {}}}},
    {"uri": COMPUTER_RECORDING, "params": {"endpoint": "/computer/recording", "methods": {"end_all": {}}}}]}}


@pytest.fixture
def boots(monkeypatch):
    """The boots a deploy asks for, each a sandbox serving the desktop driver's card at 127.0.0.1:20001; the driver
    shows a 1600x900 screen, and each extension call it gets is recorded."""
    made = SimpleNamespace(boots=[], calls=[], sandboxes=[], display=(1600, 900))

    async def fake_boot(sandbox_provider, **kwargs):
        made.boots.append({**kwargs, "provider": type(sandbox_provider).__name__})
        sandbox = SimpleNamespace(sandbox_id=f"local-{len(made.boots)}", type="local", terminated=False,
                                  ON_THIS_MACHINE=True)

        async def terminate():
            sandbox.terminated = True

        sandbox.terminate = terminate
        made.sandboxes.append(sandbox)
        return boot.Booted(sandbox, "http://127.0.0.1:20001",
                           {**DRIVER_CARD, "name": kwargs["environment"]["ENVIRONMENT_NAME"]})

    async def invoke(base_url, card, uri, params=None, timeout=30, verify=True, *, method=None):
        made.calls.append((uri, method, params))
        width, height = made.display
        return {"width": width, "height": height, "scale": 1, "modes": []} if uri == COMPUTER_DISPLAY else {}

    monkeypatch.setattr(env_computer_provider, "boot", fake_boot)
    monkeypatch.setattr(env_module.protocol_v1, "invoke_extension", invoke)
    return made


@pytest.mark.asyncio
async def test_a_computer_env_deploys_its_first_native_ready_build_with_its_settings_and_display(
        base, driver, docker, boots):
    env = _put(base, driver, display={"width": 1600, "height": 900})

    deployed = await Env.get(env.id).deploy(attribution={"run": "r1"}, env_params={
        "computer_settings": {"timezone": "Asia/Tokyo"}, "driver_settings": {"default_space": "norm"}})

    (booted,) = boots.boots
    composed = Env.get(env.id).metadata["composed"]["container"]
    assert (booted["image"], booted["port"], booted["provider"]) == (composed["arm64"]["image"], 18765,
                                                                     "LocalSandboxProvider")
    environment = booted["environment"]
    assert (environment["AGENTENV_SETTING_RESOLUTION"], environment["AGENTENV_SETTING_TIMEZONE"]) == ("1600x900", "Asia/Tokyo")
    assert json.loads(environment["AGENTENV_COMPUTER_SETTINGS"]) == {"default_space": "norm"}
    assert environment["ENVIRONMENT_NAME"] == "desktop" and "AGENTENV_SETTING_THEME" not in environment
    assert boots.calls == [(COMPUTER_DISPLAY, "get", None)]
    assert (deployed.env_provider_type, deployed.sandbox_ids) == ("computer", {"computer": {"desktop": "local-1"}})
    assert deployed.environment_card == DRIVER_CARD and deployed.mcp_url == "http://127.0.0.1:20001/mcp"
    record = deployed.metadata["computer"]
    assert {key: record[key] for key in ("form", "arch", "provider", "display", "computer_settings")} == {
        "form": "container", "arch": "arm64", "provider": "LocalSandboxProvider", "display": "1600x900",
        "computer_settings": {"resolution": "1600x900", "timezone": "Asia/Tokyo"}}
    assert record["computer"]["id"] == f"{env.id}__computer" and record["env_params"] == {
        "computer_settings": {"timezone": "Asia/Tokyo"}, "driver_settings": {"default_space": "norm"}}


@pytest.mark.asyncio
async def test_a_reset_ends_the_envs_recordings_then_boots_a_fresh_container_where_the_last_one_ran(
        base, driver, docker, boots, monkeypatch):
    env = _put(base, driver, display={"width": 1600, "height": 900})
    deployed = await Env.get(env.id).deploy(env_params={"computer_settings": {"timezone": "Asia/Tokyo"}})
    (sandbox,) = boots.sandboxes
    resets = []

    async def reset_container(self, sandbox, **kwargs):
        resets.append((sandbox, kwargs))
        boots.calls.append("reset_container")

    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "get_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "reset_container", reset_container)
    monkeypatch.setattr(env_computer_provider, "ready", AsyncMock(return_value=DRIVER_CARD))
    boots.calls.clear()

    await ResetEnvTaskStep(id="reset", version=None, env_id=env.id, fail_task_on_error=True).execute(
        TaskStepContext(deployed_envs=[deployed]))

    ((replaced, kwargs),) = resets
    assert replaced is sandbox and not sandbox.terminated
    assert (kwargs["image_name"], kwargs["port"]) == (deployed.metadata["computer"]["image"], 18765)
    assert (kwargs["env"]["AGENTENV_SETTING_TIMEZONE"], kwargs["env"]["ENVIRONMENT_NAME"]) == ("Asia/Tokyo", "desktop")
    env_computer_provider.ready.assert_awaited_once_with(deployed.environment_url, boot.READY_TIMEOUT_S)
    assert boots.calls == [(COMPUTER_RECORDING, "end_all", {"reason": "reset"}), "reset_container", (COMPUTER_DISPLAY, "get", None)]


@pytest.mark.asyncio
async def test_a_reset_a_provider_cant_carry_out_fails_the_step_rather_than_skipping_it(base, driver, docker, boots,
                                                                                       monkeypatch):
    env = _put(base, driver, display={"width": 1600, "height": 900})
    deployed = await Env.get(env.id).deploy()
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "get_sandbox", AsyncMock(return_value=boots.sandboxes[0]))
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "reset_container",
                        AsyncMock(side_effect=NotImplementedError("its sandbox is the container itself")))

    with pytest.raises(RuntimeError, match="can't be reset on LocalSandboxProvider: its sandbox is the container"):
        await ResetEnvTaskStep(id="reset", version=None, env_id=env.id, fail_task_on_error=True).execute(
            TaskStepContext(deployed_envs=[deployed]))


@pytest.mark.asyncio
async def test_teardown_closes_a_computer_env_ending_its_recordings_before_its_computer_goes(
        base, driver, docker, boots, monkeypatch):
    env = _put(base, driver, display={"width": 1600, "height": 900})
    deployed = await Env.get(env.id).deploy()
    (sandbox,) = boots.sandboxes
    terminate = sandbox.terminate

    async def terminating():
        boots.calls.append("terminate")
        await terminate()

    sandbox.terminate = terminating
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "get_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(teardown, "terminate", AsyncMock(return_value=False))
    boots.calls.clear()

    report = await teardown.teardown_run(TaskStepContext(deployed_envs=[deployed]))

    assert boots.calls == [(COMPUTER_RECORDING, "end_all", {"reason": "teardown"}), "terminate"]
    assert [s.sandbox_id for s in report.terminated] == ["local-1"] and not report.failed


@pytest.mark.asyncio
@pytest.mark.parametrize("env_params, problem", [
    ({"form": "disk"}, "env_params.form: 'disk' isn't one of"),
    ({"computer_settings": {"theme": "dark"}}, "env_params.computer_settings: theme is a build-phase setting"),
    ({"computer_settings": {"resolution": "800x600"}}, "the env's display sets the screen's size"),
    ({"computer_settings": {"timezone": 3}}, "env_params.computer_settings.timezone must be a str"),
    ({"driver_settings": {"default_space": "polar"}}, "env_params.driver_settings.default_space"),
    ({"driver_settings": {"colour": "blue"}}, "env_params.driver_settings.colour: Extra inputs are not permitted"),
    ({"colour": "blue"}, "env_params.colour"),
    ({"app_settings": {"mail": {}}}, "env_params.app_settings.mail: @local/~/kit/desktop has no app 'mail'"),
], ids=["form", "build-setting", "resolution-over-display", "setting-type", "driver-setting", "driver-unknown",
        "unknown", "app"])
async def test_env_params_a_computer_env_cant_take_are_refused_at_save_and_at_deploy(base, driver, docker, boots,
                                                                                     env_params, problem):
    env = _put(base, driver, display={"width": 1600, "height": 900})

    saved = DeployEnvTaskStep(id="d", version=None, env_id=env.id, env_params=env_params).preflight()
    with pytest.raises(DeployRefused, match=problem.replace(".", "[.]")):
        await Env.get(env.id).deploy(env_params=env_params)

    assert len(saved) == 1 and problem in saved[0] and boots.boots == []


@pytest.mark.asyncio
async def test_a_computer_that_boots_at_another_size_than_its_display_fails_the_deploy_and_goes_with_its_folder(
        base, driver, docker, boots, monkeypatch):
    boots.display = (1280, 800)
    removed = []
    monkeypatch.setattr(boot, "remove_local_work_dir", removed.append)
    env = _put(base, driver, display={"width": 1600, "height": 900})

    with pytest.raises(RuntimeError, match="booted at 1280x800, not the 1600x900 its display sets"):
        await Env.get(env.id).deploy()

    assert [sandbox.terminated for sandbox in boots.sandboxes] == [True] and removed == ["local-1"]


@pytest.mark.asyncio
async def test_a_boot_that_never_gets_ready_takes_its_sandbox_and_work_folder_down(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))
    sandbox = local_sandbox.LocalSandbox(exposed_ports=[18765], sandbox_id="local-unready")
    sandbox.terminate = AsyncMock()
    monkeypatch.setattr(boot, "ready", AsyncMock(side_effect=TimeoutError("not ready")))

    with pytest.raises(TimeoutError):
        await boot.boot(SimpleNamespace(create_container=AsyncMock(return_value=sandbox)), image="img", port=18765,
                        environment={}, network_policy=None, timeout=60, attribution=None)

    sandbox.terminate.assert_awaited_once()
    assert local_sandbox.LocalSandbox.find_work_dir("local-unready") is None


def test_a_boot_mints_a_generated_setting_resolves_a_secret_its_record_withholds_both_and_a_secret_is_a_reference(
        base, monkeypatch):
    monkeypatch.setenv("DESK_KEY", "s3cret")
    computer = base.model_copy(update={"config": {
        **base.config, "token": base.config["timezone"].model_copy(update={"default": "", "generate": True}),
        "api_key": base.config["timezone"].model_copy(update={"default": "env:DESK_KEY", "secret": True})}})

    first, second = (boot.settings_environment(computer, {}, environment_name="desktop") for _ in range(2))

    assert first["AGENTENV_SETTING_API_KEY"] == "s3cret" and first["AGENTENV_SETTING_TOKEN"]
    assert first["AGENTENV_SETTING_TOKEN"] != second["AGENTENV_SETTING_TOKEN"]
    assert boot.recorded_settings(computer, {}) == {"resolution": "1280x800", "timezone": "UTC", "token": boot.WITHHELD,
                                                    "api_key": boot.WITHHELD}
    pin = computer.model_copy(update={"config": {"pin": base.config["timezone"].model_copy(
        update={"default": 1234, "secret": True})}})
    for reference in ("env:DESK_KEY", "secret:desk/key"):
        assert gate.check_computer_settings(computer, {"api_key": reference}) == ({"api_key": reference}, [])
        assert gate.check_computer_settings(pin, {"pin": reference}) == ({"pin": reference}, [])
    for value in ("s3cret", "env:DESK_KEY?s3cret", "env:", 3):
        assert gate.check_computer_settings(computer, {"api_key": value}) == ({}, [
            "computer_settings.api_key is a secret, so it takes a secret:KEY or env:NAME reference with no ?default, "
            "not its value"])


@pytest.mark.parametrize("change, problem", [
    ("riscv64", "LocalSandboxProvider runs none of its arches (amd64, arm64) natively"),
    ("network", "LocalSandboxProvider can't enforce its network"),
])
def test_a_deploy_its_sandbox_provider_cant_carry_out_is_refused_at_save(base, driver, docker, monkeypatch, change,
                                                                       problem):
    env = _put(base, driver, network={"mode": "allowlist", "allow_hosts": ["example.com"]} if change == "network" else None)
    if change == "riscv64":
        monkeypatch.setattr(local_sandbox, "host_platform", lambda: "linux/riscv64")

    (refusal,) = DeployEnvTaskStep(id="d", version=None, env_id=env.id).preflight()

    assert problem in refusal


@pytest.fixture
def computer(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    return Computer(tmp_path / "home").install(monkeypatch)


def _deployed_env(base, driver, computer, **record):
    env = _put(base, driver, compose=False)
    env._deployed = computer.deployed(env_id=env.id, **record)
    return env


def _files(tmp_path, **contents):
    for name, data in contents.items():
        (tmp_path / "files" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "files" / name).write_bytes(data)
    return FileArtifactUniverse.put_bundled("@local/~/kit/files", files={
        name: tmp_path / "files" / name for name in contents})


@pytest.mark.asyncio
async def test_a_load_writes_each_file_into_the_computers_downloads_through_its_driver_in_batches(
        base, driver, docker, computer, tmp_path, monkeypatch):
    monkeypatch.setattr(inside, "BATCH_BYTES", 10)
    monkeypatch.setattr(inside, "issues_grants_to", lambda store, sandbox_type: False)
    env = _deployed_env(base, driver, computer)

    result = await env.load_file_artifact_universe(_files(tmp_path, **{"a.txt": b"aaaa", "b.txt": b"bbbb",
                                                                       "sub/big.bin": b"x" * 20}))

    assert result.destination_path == "Downloads"
    assert result.files == {"a.txt": "Downloads/a.txt", "b.txt": "Downloads/b.txt", "sub/big.bin": "Downloads/sub/big.bin"}
    batches = [sorted(marker["path"] for marker in markers) for kind, markers, _ in computer.calls]
    assert sorted(batches) == [["Downloads/a.txt", "Downloads/b.txt"], ["Downloads/sub/big.bin"]]
    assert (computer.home / "Downloads/sub/big.bin").read_bytes() == b"x" * 20


@pytest.mark.asyncio
@pytest.mark.parametrize("network, granted", [
    (None, True), ({"mode": "allowlist", "allow_hosts": ["example.com"]}, False), ({"mode": "allowlist"}, False),
], ids=["open", "allowlist", "closed"])
async def test_a_load_reads_through_a_grant_that_reaches_the_computer_unless_it_has_no_network(
        base, driver, docker, computer, tmp_path, monkeypatch, network, granted):
    grants = []
    monkeypatch.setattr(inside, "issues_grants_to", lambda store, sandbox_type: sandbox_type == "local")
    monkeypatch.setattr(inside, "read_object", lambda store, url: grants.append(url) or SimpleNamespace(
        size_bytes=4, model_dump=lambda mode: {"read": {"url": f"https://store.test/{len(grants)}"}}))
    env = _deployed_env(base, driver, computer, network=network)

    await env.load_file_artifact_universe(_files(tmp_path, **{"a.txt": b"aaaa"}), destination_path="Documents/in")

    ((_, (marker,), _),) = computer.calls
    assert marker["path"] == "Documents/in/a.txt"
    assert ("read" in marker, len(grants)) == ((True, 1) if granted else (False, 0))
    assert (computer.home / "Documents/in/a.txt").exists() is not granted


@pytest.mark.asyncio
@pytest.mark.parametrize("destination, problem", [
    ("/home/user/Documents", "a computer's destination_path is relative to its session user's home"),
    ("Documents/../..", "must not contain '..' segments"),
])
async def test_a_load_outside_the_home_is_refused_before_anything_is_sent(base, driver, docker, computer, tmp_path,
                                                                          destination, problem):
    env = _deployed_env(base, driver, computer)

    with pytest.raises(ValueError, match=problem):
        await env.load_file_artifact_universe(_files(tmp_path, **{"a.txt": b"aaaa"}), destination_path=destination)

    assert computer.calls == []


MAIL_SETTINGS = {"properties": {
    "signature": {"default": "", "type": "string"},
    "password": {"default": "", "type": "string", "format": "password", "writeOnly": True, "minLength": 32},
    "token": {"anyOf": [{"type": "string", "format": "password", "writeOnly": True}, {"type": "null"}], "default": None},
    "theme": {"enum": ["light", "dark"], "type": "string", "default": "light"}}, "type": "object"}


def _app(base, driver, folder, compose=True, supports=None, install=None, settings=MAIL_SETTINGS, **changes):
    """A mail app on ``base`` and ``driver``, which serves IMAP from a venv of its requirements."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "app.py").write_text("class Mail: ...\n")
    (folder / "requirements.txt").write_text("agentenv-framework-protocol\n")
    files = FileArtifactUniverse.put_bundled("@local/~/kit/mail__files", files={
        name: folder / name for name in ("app.py", "requirements.txt")})
    declaration = {
        "supports": supports or [{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}],
        "permissions": [], "launch": None, "settings": settings, "skill": None,
        "data": [], "install_steps": [], "ready_checks": [],
        "install": {"linux": install or {"python": "requirements.txt", "services": [
            {"name": "imap", "command": "venv/bin/python imap.py"}]}}}
    return ComputerAppEnv.put(**{"id": "@local/~/kit/mail", "environment_name": "mail", "module": "app.py:Mail",
                                 "files": files, "declaration": declaration, "computer": base, "driver": driver,
                                 "compose": compose, **changes})


def test_an_app_deployed_alone_is_composed_with_its_files_venv_and_programs_under_its_prefix(base, driver, docker,
                                                                                            tmp_path):
    app = _app(base, driver, tmp_path / "mail")

    assert sorted(build["platform"] for build in docker.builds) == ["linux/amd64", "linux/arm64"]
    build = docker.builds[0]
    assert "apps/mail/app.py" in build["files"] and "apps/mail/requirements.txt" in build["files"]
    dockerfile = build["dockerfile"]
    for line in ("COPY apps/mail/ /opt/agentenv/apps/mail/",
                 "RUN python3 -m venv /opt/agentenv/apps/mail/venv && /opt/agentenv/apps/mail/venv/bin/pip install "
                 "--no-cache-dir -r /opt/agentenv/apps/mail/requirements.txt",
                 "COPY agentenv-app-mail.conf /etc/supervisor/conf.d/agentenv-app-mail.conf"):
        assert line in dockerfile
    assert dockerfile.index("COPY driver/") < dockerfile.index("COPY apps/mail/") < dockerfile.index("USER user")
    assert build["confs"]["agentenv-app-mail.conf"] == (
        "[program:app-mail-imap]\ncommand=/opt/agentenv/apps/mail/venv/bin/python imap.py\n"
        "directory=/opt/agentenv/apps/mail\nenvironment=ENVIRONMENT_NAME=\"mail\"\nautorestart=true\n"
        "redirect_stderr=true\nstdout_logfile=/var/log/agentenv/app-mail-imap.log\n")
    stored = Env.get(app.id)
    assert isinstance(stored, ComputerAppEnv) and stored.build_statuses() == {"container/amd64": "unverified",
                                                                              "container/arm64": "ready"}


def test_an_app_is_composed_only_for_the_arches_it_supports(base, driver, docker, tmp_path):
    _app(base, driver, tmp_path / "mail", supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["arm64"]}])

    assert [build["platform"] for build in docker.builds] == ["linux/arm64"]


@pytest.mark.parametrize("changes, problem", [
    ({"supports": [{"os": {"type": "windows"}, "arch": ["amd64"]}]}, "app: @local/~/kit/mail supports no platform"),
    ({"supports": [{"os": {"type": "linux", "display": ["x11"]}, "arch": ["riscv64"]}]},
     "app: no architecture @local/~/kit/ubuntu-24.04, @local/~/kit/desk and @local/~/kit/mail all run"),
    ({"install": {"services": [{"name": "imap", "command": "a"}, {"name": "imap", "command": "b"}]}},
     "install.linux.services: Value error, names a service twice"),
    ({"driver": None}, "a computer app deploys alone on a computer and a driver; name both or neither"),
], ids=["platform", "arch", "install", "no-driver"])
def test_an_app_that_doesnt_fit_its_computer_and_driver_is_refused_before_anything_is_stored(base, driver, docker,
                                                                                            tmp_path, changes, problem):
    with pytest.raises(ValueError, match=problem.replace(".", "[.]")):
        _app(base, changes.pop("driver", driver), tmp_path / "mail", **changes)

    with pytest.raises(NotFoundError):
        Env.get("@local/~/kit/mail")
    assert docker.builds == []


@pytest.mark.asyncio
async def test_an_app_deployed_alone_is_named_after_it_and_handed_its_settings_a_secret_resolved(
        base, driver, docker, boots, tmp_path, monkeypatch):
    monkeypatch.setenv("MAIL_PASSWORD", "s3cret")
    monkeypatch.setenv("MAIL_TOKEN", "t0ken")
    boots.display = (1280, 800)
    app = _app(base, driver, tmp_path / "mail")
    settings = {"mail": {"signature": "-- Ada", "password": "env:MAIL_PASSWORD", "token": "env:MAIL_TOKEN"}}

    deployed = await Env.get(app.id).deploy(env_params={"app_settings": settings})

    environment = boots.boots[0]["environment"]
    assert environment["ENVIRONMENT_NAME"] == "mail" and deployed.environment_card["name"] == "mail"
    assert json.loads(environment["AGENTENV_APP_SETTINGS"]) == {"mail": {"signature": "-- Ada", "password": "s3cret",
                                                                         "token": "t0ken"}}
    assert deployed.env_id == app.id and deployed.metadata["computer"]["env_params"] == {"app_settings": settings}


def test_a_secrets_reference_meets_no_constraint_on_its_value_and_an_optional_one_may_be_null(base, driver, docker,
                                                                                            tmp_path):
    app = _app(base, driver, tmp_path / "mail")
    env_params = {"app_settings": {"mail": {"password": "env:MAIL_PASSWORD", "token": None}}}

    assert DeployEnvTaskStep(id="d", version=None, env_id=app.id, env_params=env_params).preflight() == []


def test_a_required_secret_left_out_is_named_and_a_secrets_value_never_reaches_a_message(base, driver, docker,
                                                                                         tmp_path):
    settings = {**MAIL_SETTINGS, "required": ["password", "signature"]}
    app = _app(base, driver, tmp_path / "mail", settings=settings)

    left_out = DeployEnvTaskStep(id="d", version=None, env_id=app.id,
                                 env_params={"app_settings": {"mail": {"signature": "-- Ada"}}}).preflight()
    given = DeployEnvTaskStep(id="d", version=None, env_id=app.id, env_params={"app_settings": {"mail": {
        "signature": "-- Ada", "password": "hunter2"}}}).preflight()

    assert len(left_out) == 1 and "env_params.app_settings.mail: 'password' is a required property" in left_out[0]
    assert len(given) == 1 and "password is a secret" in given[0] and "hunter2" not in given[0]


@pytest.mark.asyncio
async def test_a_load_whose_connection_to_the_driver_breaks_says_so(base, driver, docker, computer, tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(inside, "issues_grants_to", lambda store, sandbox_type: False)
    computer.add_fails = httpx.ReadError("")
    env = _deployed_env(base, driver, computer)

    with pytest.raises(RuntimeError, match=re.escape(
            f"env '{env.id}': the connection to its driver broke during data/add, so some of the files it carried "
            "may be written, each whole (ReadError)")):
        await env.load_file_artifact_universe(_files(tmp_path, **{"a.txt": b"aaaa"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("values, problem", [
    ({"colour": "blue"}, "env_params.app_settings.mail.colour: @local/~/kit/mail declares no setting 'colour'"),
    ({"signature": 3}, "env_params.app_settings.mail.signature: 3 is not of type 'string'"),
    ({"theme": "blue"}, "env_params.app_settings.mail.theme: 'blue' is not one of ['light', 'dark']"),
    ({"password": "hunter2"}, "env_params.app_settings.mail.password is a secret, so it takes a secret:KEY"),
    ({"token": "hunter2"}, "env_params.app_settings.mail.token is a secret, so it takes a secret:KEY"),
], ids=["unknown", "type", "enum", "secret-value", "optional-secret-value"])
async def test_app_settings_an_app_doesnt_declare_are_refused_at_save_and_at_deploy(base, driver, docker, boots,
                                                                                   tmp_path, values, problem):
    app = _app(base, driver, tmp_path / "mail")
    env_params = {"app_settings": {"mail": values}}

    saved = DeployEnvTaskStep(id="d", version=None, env_id=app.id, env_params=env_params).preflight()
    with pytest.raises(DeployRefused, match=re.escape(problem)):
        await Env.get(app.id).deploy(env_params=env_params)

    assert len(saved) == 1 and problem in saved[0] and boots.boots == []
