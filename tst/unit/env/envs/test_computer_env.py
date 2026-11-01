"""Computer envs: gate 1 refuses a base, a driver or a declaration that can't compose, naming every problem, before
anything is stored; the composer builds each form and arch, reuses an image of the same key whichever env built it,
records each result as it finishes, and writes the arches built as the env's computer; a deploy, a reset and a load,
of a file universe or of a snapshot of the computer's own files; and computer apps. Where and how a pair builds is test_computer_builder; a computer of several apps, test_computer_apps.
Docker and the registry are faked."""

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import re
import shutil
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from agentenv_protocol.computer import COMPUTER_DISPLAY, COMPUTER_RECORDING
from agentenv_protocol.tool_access import DISABLE_TOOL_EXTENSION_URI
from click.testing import CliRunner

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.store import get_artifact_store
from agent_env.cli import cli
from agent_env.config import get_config
from agent_env.env import Env
from agent_env.env import env as env_module
from agent_env.env.envs.computer.utils import boot, builder, composer, gate, inside, packages, resolve_in_base
from agent_env.env.envs.computer.utils.composer import Composing, _probe
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer import app_env
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
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from tst.unit.computer_home import CARD as HOME_CARD, Computer
from tst.unit.env.envs._computer import (CONFIG, SERVE, CARD_EXTENSION, CARD, MAIL_SETTINGS, MAIL_BUILD, MAIL_SERVER, _app,
                                         _base, _driver, _put, FakeVmProvider)


def test_a_computer_env_is_composed_for_each_arch_and_written_with_the_computer_its_images_make(base, driver, docker,
                                                                                                local_stores):
    local_stores.set_image_store(OciRegistryImageStore("registry.example"))
    env = _put(base, driver, computer_settings={"timezone": "Europe/Paris", "theme": "dark"},
               display={"width": 1600, "height": 900})

    stored = Env.get(env.id)
    composed = stored.metadata["composed"]["container"]
    assert {arch: (r["status"], r["reused"], r["emulated"]) for arch, r in composed.items()} == {
        "amd64": ("unverified", False, True), "arm64": ("ready", False, False)}
    assert composed["amd64"]["reason"] == "no link of the sandbox provider runs linux/amd64 natively to probe it"
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
        f"FROM {base.image().image_name}", "USER root", 'ENV AGENTENV_SETTING_THEME="dark"',
        "COPY driver/ /opt/agentenv/drivers/desk/",
        "COPY packages/wheels/driver/ /tmp/agentenv-packages/wheels/driver/",
        "RUN --network=none python3 -m venv /opt/agentenv/drivers/desk/venv && /opt/agentenv/drivers/desk/venv/bin/pip "
        "install --no-index --no-deps /tmp/agentenv-packages/wheels/driver/*.whl && /opt/agentenv/drivers/desk/venv/bin/pip list "
        "--format=json > /opt/agentenv/drivers/desk/installed.json && rm -rf /tmp/agentenv-packages/wheels/driver",
        "COPY agentenv-driver-desk.conf /etc/supervisor/conf.d/agentenv-driver-desk.conf",
        "RUN test -d /opt/agentenv/drivers/desk && test -f /etc/supervisor/conf.d/agentenv-driver-desk.conf && test -x "
        "/opt/agentenv/drivers/desk/venv/bin/python",
        "USER user", f"LABEL dev.agentenv.computer-key={composed['arm64']['key']}"]
    assert arm64["files"] == ["Dockerfile", "agentenv-driver-desk.conf", "driver/driver.py", "driver/requirements.txt",
                              "packages/wheels/driver/mss-10.1.0-py3-none-any.whl"]
    assert sorted(arch for arch, _ in docker.resolved) == ["amd64", "arm64"]
    assert docker.timeouts == {"arm64": 3600, "amd64": 10800}
    assert docker.resolved[0][1]["python"] == [{"name": "driver", "folder": "/ctx/driver",
                                                "requirements": "requirements.txt", "add": []}]
    assert composed["arm64"]["lock"].startswith("sha256:") and composed["arm64"]["lock"] != composed["amd64"]["lock"]
    assert arm64["conf"] == (
        "[program:driver-desk-driver]\ncommand=/opt/agentenv/drivers/desk/venv/bin/python -m agentenv_protocol.computer.serve "
        'driver.py:Desk\ndirectory=/opt/agentenv/drivers/desk\nenvironment=MCP_PORT="18765"\nautorestart=true\n'
        "redirect_stderr=true\nstdout_logfile=/var/log/agentenv/driver-desk-driver.log\n")
    assert ComputerEnv.from_dict(stored.to_dict()).to_dict() == stored.to_dict()


def test_a_key_covers_its_build_files_executable_bits_and_the_composers_format(base, driver, docker, tmp_path,
                                                                            monkeypatch):
    folder = tmp_path / "ctx"
    folder.mkdir()
    (folder / "run").write_bytes(b"echo\n")
    plain = composer._digest(folder)
    (folder / "run").chmod(0o755)
    runnable = composer._digest(folder)
    (folder / "run").chmod(0o640)

    first = Env.get(_put(base, driver).id).metadata["composed"]["container"]["arm64"]["key"]
    monkeypatch.setattr(composer, "FORMAT", composer.FORMAT + 1)
    raised = Env.get(_put(base, driver, id="@local/~/kit/other", environment_name="other").id).metadata["composed"][
        "container"]["arm64"]["key"]

    assert runnable != plain == composer._digest(folder) and raised != first


def test_an_env_that_differs_only_in_what_doesnt_change_the_image_reuses_the_one_another_env_built(base, driver,
                                                                                                    docker, monkeypatch):
    auth, docker_calls = RegistryAuth("localhost:5000", "user", "token"), []
    monkeypatch.setattr(LocalRegistryImageStore, "auth", lambda self, ref: auth)
    monkeypatch.setattr(builder, "docker_login", lambda signed_in: docker_calls.append(("login", signed_in)))
    monkeypatch.setattr(composer, "write_index", lambda index, images, signed_in: (
        docker_calls.append(("join", index, signed_in)), docker.joined(index, images, signed_in)))
    first_told, second_told = [], []
    first = _put(base, driver, composing=Composing(on_progress=first_told.append))
    docker_calls.clear()
    second = _put(base, driver, id="@local/~/kit/other", environment_name="other", display={"width": 640, "height": 480},
                  network={"mode": "allowlist"}, composing=Composing(on_progress=second_told.append))

    built, reused = (Env.get(env.id).metadata["composed"]["container"] for env in (first, second))
    assert len(docker.builds) == 2 and all(reused[arch]["reused"] for arch in reused)
    assert "container/arm64: building natively with this machine's Docker" in first_told
    assert ("container/amd64: building emulated, which is slow, with this machine's Docker; pass --no-emulate to skip "
            "it") in first_told
    assert not [line for line in second_told if "building" in line or "resolving" in line]
    assert second_told[0] == "composing container/amd64, container/arm64 on LocalSandboxProvider"
    assert "container/amd64: reusing the image of the same key, built earlier" in second_told
    assert any(re.fullmatch(r"container/arm64: reused, ready \(\d+ s\)", line) for line in second_told)
    assert len(docker.resolved) == 2 and {arch: r["lock"] for arch, r in reused.items()} == {
        arch: r["lock"] for arch, r in built.items()}
    assert {arch: reused[arch]["image"] for arch in reused} == {arch: built[arch]["image"] for arch in built}
    assert {arch: (r["installed"], r["emulated"]) for arch, r in reused.items()} == {
        "amd64": ({"mss": "10.1.0"}, True), "arm64": ({"mss": "10.1.0"}, False)}
    index = ComputerArtifact.get(f"{second.id}__computer").image().image_name.partition("@")[0]
    assert docker_calls == [("join", index, auth)]


@pytest.mark.parametrize("change", ["requirements", "build-setting", "service", "display-read-at-build"])
def test_what_changes_the_image_changes_its_key_and_builds_it_again(base, driver, docker, tmp_path, monkeypatch,
                                                                    change):
    staged, download = [], LocalFilesystemObjectStore.download_to_file
    monkeypatch.setattr(LocalFilesystemObjectStore, "download_to_file", lambda self, url, dest: (
        staged.append(url) if packages.PREFIX in url else None, download(self, url, dest))[1])
    if change == "display-read-at-build":
        base = _base(base.image(), config={**CONFIG, "resolution": {"default": "1280x800", "phase": "build"}})
    first = Env.get(_put(base, driver).id).metadata["composed"]["container"]["arm64"]["key"]
    assert staged == []
    if change == "requirements":
        driver = _driver(tmp_path / "changed", requirements="mss>=10,<11\n")
    elif change == "service":
        driver = _driver(tmp_path / "changed", services=[Service(name="driver", command=f"{SERVE} --verbose")])
    settings = {"theme": "dark"} if change == "build-setting" else {}
    display = {"width": 1600, "height": 900} if change == "display-read-at-build" else None

    again = Env.get(_put(base, driver, computer_settings=settings, display=display).id).metadata["composed"][
        "container"]["arm64"]

    assert again["key"] != first and again["reused"] is False and len(docker.builds) == 4
    assert bool(staged) == (change != "requirements")
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


def test_a_checkout_resolves_its_own_protocol_into_the_drivers_venv(base, driver, docker, tmp_path, monkeypatch):
    monkeypatch.setattr(composer, "_protocol_source", lambda: _protocol_checkout(tmp_path))

    _put(base, driver)

    assert {layer["name"]: layer["add"] for _, request in docker.resolved for layer in request["python"]} == {
        "driver": ["/ctx/protocol"]}
    assert "protocol/pyproject.toml" in docker.builds[0]["files"]
    assert "agentenv-protocol" not in docker.builds[0]["dockerfile"]


def _protocol_checkout(folder):
    source = folder / "protocol"
    (source / "src" / "agentenv_protocol").mkdir(parents=True, exist_ok=True)
    (source / "src" / "agentenv_protocol" / "__init__.py").write_text("")
    (source / "pyproject.toml").write_text("[project]\nname = 'agentenv-framework-protocol'\n")
    return source


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
    assert built.exit_code == 0 and ("container/amd64: unverified (no link of the sandbox provider runs linux/amd64 "
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
], ids=["display-zero", "display-side", "display-and-resolution", "network", "network-key",
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


def test_a_script_or_binary_in_the_drivers_files_stays_executable_in_the_image(base, docker, tmp_path, monkeypatch):
    folder = tmp_path / "scripted"
    folder.mkdir()
    (folder / "driver.py").write_text("class Desk: ...\n")
    (folder / "start").write_text("#!/bin/sh\nexec venv/bin/python -m agentenv_protocol.computer.serve driver.py:Desk\n")
    (folder / "helper").write_bytes(b"\x7fELF\x02\x01\x01")
    modes = {}
    monkeypatch.setattr(builder, "build_image", lambda dockerfile, context, tag, **kw: modes.update(
        {path.name: path.stat().st_mode & 0o111 for path in (context / "driver").iterdir()}) if context.name == "context"
        else None)
    files = FileArtifactUniverse.put_bundled("@local/~/kit/scripted__files", files={
        name: folder / name for name in ("driver.py", "start", "helper")})
    driver = ComputerDriverEnv.put(
        id="@local/~/kit/scripted", module="driver.py:Desk", files=files, declaration=_driver(tmp_path / "d").declaration,
        install={"linux": Install(services=[Service(name="driver", command="./start")])}, port=18765)

    _put(base, driver)

    assert modes == {"driver.py": 0, "start": 0o111, "helper": 0o111}


def test_env_build_writes_the_computer_again_when_it_isnt_over_every_pair_built(base, driver, docker, caplog):
    caplog.set_level(logging.ERROR, logger=composer.__name__)
    stored = Env.get(_put(base, driver).id)
    stale = {**stored.metadata["computer"], "images": {"arm64": stored.metadata["computer"]["images"]["arm64"]}}
    stored.update_metadata({**stored.metadata, "computer": stale})

    assert not composer.current(Env.get(stored.id))
    unknown = CliRunner().invoke(cli, ["env", "build", "--id", stored.id, "--sandbox", "nosuch"])
    rebuilt = CliRunner().invoke(cli, ["env", "build", "--id", stored.id, "--sandbox", "local"])

    assert unknown.exit_code == 1 and "Unknown sandbox backend: 'nosuch'" in unknown.output
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


@pytest.mark.asyncio
async def test_a_boot_is_ready_with_the_card_read_once_its_status_says_so_each_app_on_it(monkeypatch):
    cards = iter([{"name": "desk"}, {"name": "desk"}, {"name": "desk", "children_environments": [{"name": "mail"}]}])
    statuses = iter([{"ready": False, "checks": [{"name": "app mail", "ok": False, "detail": "starting"}]},
                     {"ready": True}])

    async def card(url, timeout):
        return next(cards)

    async def status(url, card, uri, params, timeout, method):
        return next(statuses)

    monkeypatch.setattr(boot.protocol_v1, "get_card", card)
    monkeypatch.setattr(boot.protocol_v1, "invoke_extension", status)
    monkeypatch.setattr(boot, "_POLL_S", 0)

    assert await boot.ready("http://desk", 5) == {"name": "desk", "children_environments": [{"name": "mail"}]}


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


def _deployed_env(base, driver, computer, apps=(), children=(), **record):
    """The env of ``apps``, deployed on ``computer``, its card named after it and listing ``children``."""
    env = _put(base, driver, compose=False, apps=apps)
    card = {**HOME_CARD, "name": env.environment_name,
            **({"children_environments": list(children)} if children else {})}
    env._deployed = computer.deployed(env_id=env.id, card=card, **record)
    return env


def _child(name: str, *operations: str) -> dict:
    """An app's card on its computer's, as the driver fronts it."""
    return {"name": name, "url": f"/svc/{name}/agentenv", "capabilities": {"operations": list(operations)}}


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


def _snapshot(environment_name: str, contents: dict[str, bytes]) -> EnvironmentArtifact:
    """An environment artifact of ``contents`` by path, as snapshot_computer_env keeps a computer's files."""
    files = {path: FileArtifact.put_bytes(f"snap-{environment_name}-{index}", description="snapshot",
                                          filename=path.rsplit("/", 1)[-1], content=data, content_type="text/plain")
             for index, (path, data) in enumerate(contents.items())}
    return EnvironmentArtifact.put(f"snap-{environment_name}", environment_name=environment_name, file_artifacts=files)


@pytest.mark.asyncio
async def test_a_snapshot_loads_back_its_files_then_each_apps_through_the_apps_route_leaving_out_other_environments(
        base, driver, docker, computer, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(inside, "issues_grants_to", lambda store, sandbox_type: False)
    mail = _app(base, driver, tmp_path / "mail", compose=False)
    env = _deployed_env(base, driver, computer, apps=[mail], children=[_child("mail", "data/add", "data/get")])
    contents = {"Desktop/a.txt": b"a", "Documents/r/b.txt": b"b"}
    universe = EnvironmentUniverseArtifact.put("snap", environment_artifacts=[
        _snapshot("mail", {"Mail/inbox.txt": b"m"}), _snapshot(env.environment_name, contents),
        _snapshot("notes", {"inbox/1.json": b"{}"})])

    with caplog.at_level(logging.WARNING):
        await env.load_environment_universe_artifact(universe)

    assert {path: (computer.home / path).read_bytes() for path in contents} == contents
    assert computer.added_at == ["http://127.0.0.1:20001", "http://127.0.0.1:20001/svc/mail"]
    assert [markers for _, markers, _ in computer.calls][1] == [{"path": "Mail/inbox.txt"}]
    assert not (computer.home / "inbox").exists()
    assert "leaving out ['notes']" in caplog.text


@pytest.mark.asyncio
async def test_an_environment_named_after_neither_the_computer_nor_an_app_that_loads_is_refused_before_anything_is_sent(
        base, driver, docker, computer, tmp_path):
    mail = _app(base, driver, tmp_path / "mail", compose=False)
    env = _deployed_env(base, driver, computer, apps=[mail], children=[_child("mail", "data/get")])
    notes, mailbox = _snapshot("notes", {"inbox/1.json": b"{}"}), _snapshot("mail", {"Mail/inbox.txt": b"m"})

    with pytest.raises(ValueError, match="has no environment named 'notes': it has \\['desktop', 'mail'\\]"):
        await env.load_environment_artifact(notes)
    with pytest.raises(ValueError, match="'mail' in computer env '@local/~/kit/desktop' serves no data/add"):
        await env.load_environment_universe_artifact(EnvironmentUniverseArtifact.put("both", environment_artifacts=[
            _snapshot(env.environment_name, {"Desktop/a.txt": b"a"}), mailbox]))
    with pytest.raises(RuntimeError, match="has no environment named any of \\['desktop', 'mail'\\]"):
        await env.load_environment_universe_artifact(EnvironmentUniverseArtifact.put("notes-only",
                                                                                     environment_artifacts=[notes]))
    assert computer.calls == []


PROTOCOL_VERSION = importlib.metadata.version("agentenv-framework-protocol")


def test_an_app_deployed_alone_is_composed_with_its_files_venv_and_programs_under_its_prefix(base, driver, docker,
                                                                                            tmp_path):
    app = _app(base, driver, tmp_path / "mail")

    assert sorted(build["platform"] for build in docker.builds) == ["linux/amd64", "linux/arm64"]
    build = docker.builds[0]
    assert "apps/mail/app.py" in build["files"] and "apps/mail/requirements.txt" in build["files"]
    dockerfile = build["dockerfile"]
    assert [layer for _, request in docker.resolved for layer in request["python"]][-1] == {
        "name": "apps-mail", "folder": "/ctx/apps/mail", "requirements": "requirements.txt",
        "add": [f"agentenv-framework-protocol[app]=={PROTOCOL_VERSION}"]}
    for line in ("COPY apps/mail/ /opt/agentenv/apps/mail/",
                 "RUN --network=none python3 -m venv /opt/agentenv/apps/mail/venv && /opt/agentenv/apps/mail/venv/bin/pip"
                 " install --no-index --no-deps /tmp/agentenv-packages/wheels/apps-mail/*.whl",
                 "RUN mkdir -p /opt/agentenv/apps/mail/.agentenv && chown user /opt/agentenv/apps/mail/.agentenv",
                 "COPY agentenv-app-mail.conf /etc/supervisor/conf.d/agentenv-app-mail.conf"):
        assert line in dockerfile
    assert dockerfile.index("COPY driver/") < dockerfile.index("COPY apps/mail/") < dockerfile.index("USER user")
    assert build["confs"]["agentenv-app-mail.conf"] == MAIL_SERVER + "\n" + (
        "[program:app-mail-imap]\ncommand=/opt/agentenv/apps/mail/venv/bin/python imap.py\n"
        "directory=/opt/agentenv/apps/mail\nenvironment=ENVIRONMENT_NAME=\"mail\"\nautorestart=true\n"
        "redirect_stderr=true\nstdout_logfile=/var/log/agentenv/app-mail-imap.log\n")
    stored = Env.get(app.id)
    assert isinstance(stored, ComputerAppEnv) and stored.build_statuses() == {"container/amd64": "unverified",
                                                                              "container/arm64": "ready"}


@pytest.mark.parametrize("at, checkout", [("boot", False), ("demand", True)], ids=["boot-release", "demand-checkout"])
def test_an_app_that_requires_nothing_still_gets_a_venv_for_its_own_server(base, driver, docker, tmp_path, monkeypatch,
                                                                            at, checkout):
    if checkout:
        monkeypatch.setattr(composer, "_protocol_source", lambda: _protocol_checkout(tmp_path))
    launch = {"at": at, "command": ["bin/mail"], "window": {"class": "Mail"}, "ready": {"window": True, "timeout_s": 240.0}}
    app = _app(base, driver, tmp_path / "mail", install={"services": []}, launch=launch)

    build = docker.builds[0]
    protocol = "/ctx/protocol[app]" if checkout else f"agentenv-framework-protocol[app]=={PROTOCOL_VERSION}"
    assert {layer["name"]: (layer["requirements"], layer["add"]) for _, request in docker.resolved
            for layer in request["python"]}["apps-mail"] == (None, [protocol])
    assert "/opt/agentenv/apps/mail/venv/bin/pip install --no-index --no-deps /tmp/agentenv-packages/wheels/apps-mail/*.whl" in (
        build["dockerfile"])
    assert build["confs"]["agentenv-app-mail.conf"] == MAIL_SERVER
    assert boot.ready_timeout([app], "linux") == (270.0 if at == "boot" else boot.READY_TIMEOUT_S)
    assert boot.ready_timeout([app], "windows") == boot.READY_TIMEOUT_S


def test_an_apps_packages_downloads_scripts_and_steps_are_fetched_at_put_and_installed_offline(base, driver, docker,
                                                                                               tmp_path, monkeypatch):
    tool = b"#!/bin/sh\necho tool\n"
    monkeypatch.setattr(builder.httpx, "stream", _serving({"https://example.com/tool": tool}))
    install = {"python": "requirements.txt", "packages": {"apt": ["hello", "xdotool/noble"]},
               "sources": [{"apt": "https://ppa.example/ubuntu", "suite": "noble", "components": ["main"],
                            "key": "ppa.asc"}],
               "downloads": [{"url": "https://example.com/tool", "sha256": hashlib.sha256(tool).hexdigest(),
                              "to": "bin/tool", "arch": "arm64"}],
               "run": [{"script": "set up.sh"}, {"script": "profile.sh", "scope": "session"}], "services": []}

    app = _app(base, driver, tmp_path / "mail", install=install,
               steps=[("index_mail", "session"), ("register_codecs", "system", "windows")],
               extra={"ppa.asc": "key", "set up.sh": "mkdir -p /srv/mail\n", "profile.sh": "#!/bin/sh\n"})

    requests = dict(docker.resolved)
    assert requests["arm64"]["apt"] == ["hello", "xdotool/noble"] and requests["arm64"]["sources"] == [
        {"apt": "https://ppa.example/ubuntu", "suite": "noble", "components": ["main"], "key": "/ctx/apps/mail/ppa.asc"}]
    builds = {build["platform"]: build for build in docker.builds}
    lines = builds["linux/arm64"]["dockerfile"].splitlines()
    first = lines.index("COPY packages/apt/ /tmp/agentenv-packages/apt/")
    assert lines[first:first + 3] == [
        "COPY packages/apt/ /tmp/agentenv-packages/apt/",
        "RUN --network=none apt-get install -y --no-install-recommends /tmp/agentenv-packages/apt/*.deb && rm -rf "
        "/tmp/agentenv-packages/apt", "COPY driver/ /opt/agentenv/drivers/desk/"]
    steps = lines.index("RUN mkdir -p /opt/agentenv/apps/mail/.agentenv && chown user /opt/agentenv/apps/mail/.agentenv")
    assert lines[steps + 1:steps + 8] == [
        "RUN --network=none cd /opt/agentenv/apps/mail && './set up.sh'",
        "USER user", "RUN --network=none cd /opt/agentenv/apps/mail && ./profile.sh", "USER root",
        "USER user", "RUN --network=none cd /opt/agentenv/apps/mail && venv/bin/python -m "
        "agentenv_protocol.computer.run_install_step app.py:Mail index_mail", "USER root"]
    assert "register_codecs" not in builds["linux/arm64"]["dockerfile"]
    assert {"packages/apt/hello_1.0_arm64.deb", "packages/apt/xdotool_1.0_arm64.deb", "apps/mail/bin/tool"} <= set(
        builds["linux/arm64"]["files"])
    assert "apps/mail/bin/tool" not in builds["linux/amd64"]["files"]
    composed = Env.get(app.id).metadata["composed"]["container"]
    assert composed["arm64"]["lock"] != composed["amd64"]["lock"]


@pytest.mark.parametrize("failing", ["resolver", "download", "link"])
def test_a_resolve_or_download_that_fails_fails_its_arch_at_the_resolve_stage(base, driver, docker, tmp_path,
                                                                             monkeypatch, failing):
    if failing == "resolver":
        def refuse(request, context, source, arch, work):
            raise builder.ResolveError("resolving the packages failed", "E: Unable to locate package nosuch\n")
        monkeypatch.setattr(builder.LocalBuilder, "resolve", docker.resolving(refuse))
    if failing == "link":
        (tmp_path / "credentials").write_text("secret")

        def linking(request, context, source, arch, work):
            result = docker.resolve(request, context, source, arch, work)
            wheel = work / "out" / "wheels" / "driver" / "mss-10.1.0-py3-none-any.whl"
            wheel.unlink()
            wheel.symlink_to(tmp_path / "credentials")
            return result
        monkeypatch.setattr(builder.LocalBuilder, "resolve", docker.resolving(linking))
    monkeypatch.setattr(builder.httpx, "stream", _serving({"https://example.com/tool": b"something else"}))
    install = {"python": "requirements.txt", "services": [],
               "downloads": [{"url": "https://example.com/tool", "sha256": "a" * 64, "to": "tool"}]}

    composed = Env.get(_app(base, driver, tmp_path / "mail", install=install).id).metadata["composed"]["container"]

    assert {result["stage"] for result in composed.values()} == {"resolve"} and docker.builds == []
    reason = composed["arm64"]["reason"]
    if failing == "resolver":
        assert reason == "resolving the packages failed: E: Unable to locate package nosuch" and composed["arm64"]["log"]
    elif failing == "link":
        assert reason == "the resolver's 'wheels/driver/mss-10.1.0-py3-none-any.whl' isn't a plain file it wrote"
        assert not get_config().get_object_store().list(packages.PREFIX)
    else:
        assert reason.startswith("https://example.com/tool has the sha256 ") and reason.endswith(
            f"not the {'a' * 64} its download names")


def test_a_stored_file_that_isnt_the_one_its_lock_names_fails_the_build_that_stages_it(base, driver, docker,
                                                                                        tmp_path):
    lock = packages.stored(Env.get(_put(base, driver).id).metadata["composed"]["container"]["arm64"]["lock"],
                           "@local/~/kit/desktop")
    sha256 = lock.wheels["driver"][0].sha256
    store, prefix = packages._store("@local/~/kit/desktop")
    Path(urllib.parse.urlparse(store.object_url(f"{prefix}/files/{sha256}")).path).write_bytes(b"tampered")

    again = Env.get(_put(base, driver, computer_settings={"theme": "dark"}).id).metadata["composed"]["container"]["arm64"]

    assert (again["status"], again["stage"]) == ("failed", "build")
    assert again["reason"].endswith(f"agentenv-packages/files/{sha256} isn't the file this build stages: its sha256 "
                                    f"is {hashlib.sha256(b'tampered').hexdigest()}")


def test_a_venv_whose_requirements_resolve_to_nothing_is_made_empty(base, docker, tmp_path):
    driver = _driver(tmp_path / "empty", requirements="")

    _put(base, driver)

    lines = docker.builds[0]["dockerfile"].splitlines()
    assert "RUN --network=none python3 -m venv /opt/agentenv/drivers/desk/venv && /opt/agentenv/drivers/desk/venv/bin/pip " \
           "list --format=json > /opt/agentenv/drivers/desk/installed.json" in lines
    assert not [line for line in lines if "agentenv-packages" in line]


def test_two_puts_resolving_the_same_inputs_at_once_both_get_the_first_stored_lock(docker, tmp_path, monkeypatch):
    folder = tmp_path / "ctx" / "driver"
    folder.mkdir(parents=True)
    (folder / "requirements.txt").write_text("mss\n")
    wants = [packages.Want("driver", Install(python="requirements.txt", services=[Service(name="d", command="x")]), True)]
    inputs = packages.inputs_digest(wants, tmp_path / "ctx", "base", "arm64")
    first, _ = packages.resolve(inputs, wants, _local(tmp_path / "1", tmp_path / "ctx"), "base", "arm64",
                                "@local/~/kit/desktop")

    def newer(request, context, source, arch, work):
        result = docker.resolve(request, context, source, arch, work)
        (work / "out" / "wheels" / "driver" / "mss-10.1.0-py3-none-any.whl").write_bytes(b"mss, published again")
        return result
    monkeypatch.setattr(builder.LocalBuilder, "resolve", docker.resolving(newer))
    second, _ = packages.resolve(inputs, wants, _local(tmp_path / "2", tmp_path / "ctx"), "base", "arm64",
                                 "@local/~/kit/desktop")
    monkeypatch.setattr(builder.os, "getuid", lambda: 4242)

    assert second == first == packages.stored(inputs, "@local/~/kit/desktop")
    assert packages.inputs_digest(wants, tmp_path / "ctx", "base", "arm64") == inputs


def _local(work, context):
    """A builder of this machine's Docker in ``work``, its build context ``context``'s files."""
    shutil.copytree(context, work / builder.CONTEXT)
    base = SimpleNamespace(image_name="ghcr.io/acme/base:1", pinned_name="ghcr.io/acme/base:1")
    return builder.LocalBuilder("LocalSandboxProvider", work, base, "linux/arm64")


def test_the_resolver_installs_the_apt_packages_it_downloads_before_building_wheels(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(resolve_in_base, "KEYRINGS", tmp_path / "keyrings")
    monkeypatch.setattr(resolve_in_base, "SOURCES", tmp_path)
    monkeypatch.setattr(resolve_in_base.subprocess, "run", lambda argv, **kwargs: ran.append(argv) or SimpleNamespace(
        stdout=""))
    monkeypatch.setattr(resolve_in_base.os, "chown", lambda *args: None)
    request = {"apt": ["build-essential", "libxml2-dev"], "sources": [], "owner": [0, 0], "python": [
        {"name": "driver", "folder": str(tmp_path), "requirements": "requirements.txt", "add": []}]}
    (tmp_path / "request.json").write_text(json.dumps(request))

    resolve_in_base.main(str(tmp_path / "request.json"), str(tmp_path / "out"))

    commands = [" ".join(argv[:3]) for argv in ran]
    assert commands[:3] == ["apt-get update -q", "apt-get install -y", "apt-get install -y"]
    assert "--download-only" in ran[1] and "--download-only" not in ran[2] and ran[2][-2:] == request["apt"]
    assert ran[-1][1:3] == ["wheel", "--no-cache-dir"]


def _serving(bodies):
    """``httpx.stream`` answering each URL with its body."""
    @contextlib.contextmanager
    def stream(method, url, **kwargs):
        yield httpx.Response(200, content=bodies[url], request=httpx.Request(method, url))
    return stream


def test_an_apps_build_settings_are_fixed_in_its_folder_and_a_change_builds_another_image_from_the_same_lock(
        base, driver, docker, tmp_path):
    first = _app(base, driver, tmp_path / "mail", build_settings={"locale": "fr_FR"})
    second = _app(base, driver, tmp_path / "mail", build_settings={"locale": "fr_FR", "edition": "pro"})

    assert [build["build_settings"] for build in docker.builds if build["platform"] == "linux/arm64"] == [
        {"mail": {"locale": "fr_FR"}}, {"mail": {"edition": "pro", "locale": "fr_FR"}}]
    keys = {Env.get(app.id, app.version).metadata["composed"]["container"]["arm64"]["key"] for app in (first, second)}
    assert len(keys) == 2 and len(docker.resolved) == 2
    stored = Env.get(second.id)
    assert stored.build_settings == {"locale": "fr_FR", "edition": "pro"}
    assert ComputerAppEnv.from_dict(stored.to_dict()).to_dict() == stored.to_dict()


def test_an_apps_file_in_its_servers_folder_is_refused_and_one_stored_before_is_built_over_by_its_build_settings(
        base, driver, docker, signing, tmp_path, monkeypatch):
    own = {".agentenv/build_settings.json": '{"locale": "en_GB"}'}

    with pytest.raises(ValueError, match=r"install.linux.files: '.agentenv/build_settings.json' is in .agentenv/, where "
                                         r"the app's own server keeps its socket and build settings"):
        _app(base, driver, tmp_path / "mail", extra=own, build_settings={"locale": "fr_FR"})
    monkeypatch.setattr(app_env, "_in_server_folder", lambda declaration, names: [])
    stored = _app(base, driver, tmp_path / "mail", extra=own, build_settings={"locale": "fr_FR"}, compose=False)
    composer.compose(stored, composing=Composing(FakeVmProvider(docker, tmp_path / "vms")))

    assert [build["build_settings"] for build in docker.builds] == [{"mail": {"locale": "fr_FR"}}] * 2


@pytest.mark.parametrize("build, values, problem", [
    ({**MAIL_BUILD, "required": ["locale"]}, {}, "build_settings: 'locale' is a required property"),
    (MAIL_BUILD, {"colour": "blue"}, "build_settings.colour: @local/~/kit/mail declares no build setting 'colour'"),
    (MAIL_BUILD, {"edition": "office"}, "build_settings.edition: 'office' is not one of ['home', 'pro']"),
    (None, {"edition": "pro"}, "build_settings.edition: @local/~/kit/mail declares no build setting 'edition'"),
], ids=["required", "unknown", "enum", "none-declared"])
def test_build_settings_an_app_doesnt_declare_or_its_schema_refuses_are_refused_before_anything_is_stored(
        base, driver, docker, tmp_path, build, values, problem):
    with pytest.raises(ValueError, match=re.escape(problem)):
        _app(base, driver, tmp_path / "mail", build=build, build_settings=values)

    assert docker.builds == []
    with pytest.raises(NotFoundError):
        Env.get("@local/~/kit/mail")


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
    ({"install": {"files": "linux", "services": []}},
     "install.linux.files: the app's server serves its class from the app's folder, which needs 'app.py' in 'linux'"),
], ids=["platform", "arch", "install", "no-driver", "class-file"])
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


def test_a_deploys_app_settings_are_checked_against_its_settings_on_its_computers_os(base, driver, docker, tmp_path):
    linux = {**MAIL_SETTINGS, "properties": {**MAIL_SETTINGS["properties"], "font": {"type": "string", "default": ""}}}
    app = _app(base, driver, tmp_path / "mail", linux_settings=linux)

    def saved(values):
        return DeployEnvTaskStep(id="d", version=None, env_id=app.id,
                                 env_params={"app_settings": {"mail": values}}).preflight()

    assert saved({"font": "Mono"}) == [] and (app.settings_schema("linux"), app.settings_schema("windows")) == (
        linux, MAIL_SETTINGS)
    assert "env_params.app_settings.mail.font: 3 is not of type 'string'" in saved({"font": 3})[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("values, problem", [
    ({"colour": "blue"}, "env_params.app_settings.mail.colour: @local/~/kit/mail declares no setting 'colour'"),
    ({"signature": 3}, "env_params.app_settings.mail.signature: 3 is not of type 'string'"),
    ({"theme": "blue"}, "env_params.app_settings.mail.theme: 'blue' is not one of ['light', 'dark']"),
    ({"password": "hunter2"}, "env_params.app_settings.mail.password is a secret, so it takes a secret:KEY"),
    ({"token": "hunter2"}, "env_params.app_settings.mail.token is a secret, so it takes a secret:KEY"),
    ({"edition": "pro"}, "env_params.app_settings.mail.edition is one of @local/~/kit/mail's build settings, which only "
                         "another image changes"),
], ids=["unknown", "type", "enum", "secret-value", "optional-secret-value", "build-setting"])
async def test_app_settings_an_app_doesnt_declare_are_refused_at_save_and_at_deploy(base, driver, docker, boots,
                                                                                   tmp_path, values, problem):
    app = _app(base, driver, tmp_path / "mail")
    env_params = {"app_settings": {"mail": values}}

    saved = DeployEnvTaskStep(id="d", version=None, env_id=app.id, env_params=env_params).preflight()
    with pytest.raises(DeployRefused, match=re.escape(problem)):
        await Env.get(app.id).deploy(env_params=env_params)

    assert len(saved) == 1 and problem in saved[0] and boots.boots == []
