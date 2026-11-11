"""Computer envs: gate 1 refuses a base, a driver or a declaration that can't compose, naming every problem, before
anything is stored; the composer builds each form and arch, reuses an image of the same key whichever env built it,
records each result as it finishes, and writes the arches built as the env's computer; a deploy, a reset and a load,
of a file universe or of a snapshot of the computer's own files; and computer apps. Where and how a pair builds is test_computer_builder; a computer of several apps, test_computer_apps.
Docker and the registry are faked."""

import asyncio
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import re
import shutil
import subprocess
import urllib.parse
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from agentenv_protocol import CONTROL_TOKEN_ENV
from agentenv_protocol.computer import COMPUTER_DISPLAY, COMPUTER_READY, COMPUTER_RECORDING, TOOL_PREFIX_ENV
from agentenv_protocol.computer.app import COMPUTER_APP
from agentenv_protocol.computer.app_runtime import HOLD_BOOT_LAUNCH_ENV
from agentenv_protocol.computer.conformance import CheckResult, ProbeReport
from agentenv_protocol.tool_access import DISABLE_TOOL_EXTENSION_URI, ENABLE_TOOL_EXTENSION_URI
from agentenv_protocol.website_browser import BROWSER
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
from agent_env.env.env import DeployedEnv, EnvCapabilityUnsupported
from agent_env.env.envs.computer.utils import (boot, builder, composed, composer, gate, inside, packages, params,
                                               probing, resolve_in_base)
from agent_env.env.envs.computer.utils import sites as computer_sites
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.envs.computer.utils.probing import _probe
from agent_env.env.envs.computer.env import COMPUTER_TOOLS, ComputerEnv
from agent_env.env.envs.computer import app_env
from agent_env.env.envs.computer.app_env import ComputerAppEnv
from agent_env.env.envs.computer.utils.install import Install, Service
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv, KnownFailure
from agent_env.plugins._bundles import ALIASES_FILE, InstalledBundle, installed_aliases
from agent_env import providers
from agent_env.providers.env_providers import env_computer_provider
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
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
        "RUN --network=none --mount=type=bind,source=packages/wheels/driver,target=/tmp/agentenv-packages/wheels/driver "
        "python3 -m venv /opt/agentenv/drivers/desk/venv && /opt/agentenv/drivers/desk/venv/bin/pip install --no-index "
        "--no-deps /tmp/agentenv-packages/wheels/driver/*.whl && /opt/agentenv/drivers/desk/venv/bin/pip list "
        "--format=json > /opt/agentenv/drivers/desk/installed.json",
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


def test_gate_3_probes_a_build_with_its_apps_defaults_as_a_deploy_with_no_env_params_boots_it(base, driver, docker,
                                                                                            tmp_path, monkeypatch):
    mail = _app(base, driver, tmp_path / "mail", compose=False, settings={**MAIL_SETTINGS, "required": ["signature"]},
                defaults={"signature": "-- Ada"})
    env = _put(base, driver, compose=False, apps=[mail], app_settings={"mail": {"theme": "dark"}})
    booted = []

    async def fake_boot(link, **kwargs):
        booted.append(kwargs)
        return boot.Booted(sandbox=None, url="http://computer", card=CARD)

    monkeypatch.setattr(boot, "boot", fake_boot)
    monkeypatch.setattr(boot, "discard", AsyncMock())
    monkeypatch.setattr(probing, "run_probe", AsyncMock(return_value="report"))

    asyncio.run(_probe([SimpleNamespace()], env, driver, "img"))

    assert json.loads(booted[0]["environment"]["AGENTENV_APP_SETTINGS"]) == {"mail": {"signature": "-- Ada",
                                                                                      "theme": "dark"}}



@pytest.mark.parametrize("asks", [1024, None])
def test_the_probe_boots_a_build_with_the_shared_memory_its_apps_ask_for_as_a_deploy_does(base, driver, docker,
                                                                                         tmp_path, monkeypatch, asks):
    monkeypatch.setenv("MAIL_TOKEN", "t0ken")
    install = {"python": "requirements.txt", "services": [{"name": "imap", "command": "venv/bin/python imap.py"}],
               **({"shm_mb": asks} if asks else {})}
    env = _put(base, driver, compose=False, apps=[_app(base, driver, tmp_path / "mail", compose=False, install=install,
                                                       defaults={"token": "env:MAIL_TOKEN"})])
    booted = []

    async def fake_boot(link, **kwargs):
        booted.append(kwargs)
        return boot.Booted(sandbox=None, url="http://computer", card=CARD)

    monkeypatch.setattr(boot, "boot", fake_boot)
    monkeypatch.setattr(boot, "discard", AsyncMock())
    monkeypatch.setattr(probing, "run_probe", AsyncMock(return_value="report"))

    asyncio.run(_probe([SimpleNamespace()], env, driver, "img"))

    assert booted[0]["shm_mb"] == asks


@pytest.mark.asyncio
async def test_a_boot_asks_for_shared_memory_only_where_its_provider_gives_it_and_logs_where_it_doesnt(monkeypatch,
                                                                                                    caplog):
    created, sandbox = [], SimpleNamespace(tunnel_urls={18765: "http://computer"})

    class Gives(SandboxProvider):
        SHARED_MEMORY = True

        async def create_container(self, **kwargs):
            created.append(kwargs)
            return sandbox

    class OldPlugin(SandboxProvider):
        async def create_container(self, *, image_name, port, env, network_policy=None, disk_size_gb=10, timeout=7200,
                                   attribution=None):
            created.append({"image_name": image_name})
            return sandbox

    monkeypatch.setattr(boot, "ready", AsyncMock(return_value=CARD))
    with caplog.at_level(logging.INFO, logger="agent_env.env.envs.computer.utils.boot"):
        for provider in (Gives(), OldPlugin()):
            await boot.boot(provider, image="img", port=18765, environment={}, token="t", network_policy=None,
                            timeout=60, attribution=None, shm_mb=1024)

    assert created[0]["shm_mb"] == 1024 and "shm_mb" not in created[1]
    assert "OldPlugin can't size /dev/shm: the container keeps its default, not the 1024 MiB asked for" in caplog.text
    assert "Gives can't size" not in caplog.text

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


def test_the_probe_boots_a_build_at_the_envs_settings_with_a_token_of_its_own_as_a_deploy_does(base, driver, docker,
                                                                                               monkeypatch):
    env = _put(base, driver, compose=False, computer_settings={"timezone": "Europe/Paris"},
               display={"width": 1600, "height": 900})
    booted = []

    async def fake_boot(link, **kwargs):
        booted.append(kwargs)
        return boot.Booted(sandbox=None, url="http://computer", card=CARD)

    monkeypatch.setattr(boot, "boot", fake_boot)
    monkeypatch.setattr(boot, "discard", AsyncMock())
    probed = AsyncMock(return_value="report")
    monkeypatch.setattr(probing, "run_probe", probed)

    assert asyncio.run(_probe([SimpleNamespace()], env, driver, "img")) == (
        "report", CARD, "SimpleNamespace")
    assert {k: v for k, v in booted[0]["environment"].items() if k.startswith("AGENTENV_SETTING_")} == {
        "AGENTENV_SETTING_RESOLUTION": "1600x900", "AGENTENV_SETTING_TIMEZONE": "Europe/Paris"}
    assert len(booted[0]["token"]) >= 32
    probed.assert_awaited_once_with("http://computer", "full", token=booted[0]["token"])


def test_a_build_the_probe_cant_boot_for_a_setting_that_doesnt_resolve_stays_unverified_until_a_compose_where_it_does(
        base, driver, docker, tmp_path, monkeypatch):
    monkeypatch.delenv("MAIL_TOKEN", raising=False)
    booted = _probing_for_real(monkeypatch)
    env = _put(base, driver, apps=[_app(base, driver, tmp_path / "mail", compose=False,
                                        defaults={"token": "env:MAIL_TOKEN"})])
    unprobed = Env.get(env.id).metadata["composed"]["container"]["arm64"]
    left = composer.pending(Env.get(env.id))
    monkeypatch.setenv("MAIL_TOKEN", "t0ken")
    composer.compose(Env.get(env.id), left)
    probed = Env.get(env.id).metadata["composed"]["container"]["arm64"]

    assert (unprobed["status"], unprobed["reason"]) == ("unverified", "not probed: a setting it boots with didn't "
                                                        "resolve here: Unresolved env:MAIL_TOKEN reference (no value "
                                                        "and no default)")
    assert left == [("container", "arm64")]
    assert probed["status"] == "ready" and probed["image"] == unprobed["image"]
    assert len(booted) == 1 and json.loads(booted[0]["AGENTENV_APP_SETTINGS"])["mail"]["token"] == "t0ken"


def _probing_for_real(monkeypatch) -> list[dict]:
    """Gate 3's own probe, which resolves the settings it boots with, over a boot and a probe that pass: the
    environment of each boot it makes."""
    booted = []

    async def fake_boot(link, **kwargs):
        booted.append(kwargs["environment"])
        return boot.Booted(sandbox=None, url="http://computer", card=CARD)

    monkeypatch.setattr(probing, "_probe", _probe)
    monkeypatch.setattr(boot, "boot", fake_boot)
    monkeypatch.setattr(boot, "discard", AsyncMock())
    monkeypatch.setattr(probing, "run_probe", AsyncMock(return_value=ProbeReport.of("full", "desk", [CheckResult(
        check="P0", name="the card", status="passed", extensions=[CARD_EXTENSION])], [CARD_EXTENSION])))
    return booted


def test_a_build_whose_probe_fails_or_that_never_gets_ready_fails_at_the_probe_and_isnt_deployed(base, driver, docker,
                                                                                                monkeypatch):
    docker.probe_fails["arm64"] = "the click never landed"
    failed = Env.get(_put(base, driver).id).metadata["composed"]["container"]["arm64"]

    async def never_ready(*args):
        raise RuntimeError("LocalSandboxProvider: the computer at http://127.0.0.1:1 wasn't ready within 180 s "
                           "(not ready: display: no X)")

    async def overran(*args):
        raise TimeoutError

    monkeypatch.setattr(probing, "_probe", never_ready)
    unbooted = Env.get(_put(base, driver, id="@local/~/kit/other", environment_name="other").id).metadata[
        "composed"]["container"]["arm64"]
    monkeypatch.setattr(probing, "_probe", overran)
    stuck = Env.get(_put(base, driver, id="@local/~/kit/stuck", environment_name="stuck").id).metadata[
        "composed"]["container"]["arm64"]

    assert (failed["status"], failed["stage"], failed["reason"]) == (
        "failed", "probe", "the probe failed: P4 clicks show their effect: the click never landed")
    assert not failed["probe"]["passed"] and failed["image"]
    assert (unbooted["status"], unbooted["stage"]) == ("failed", "probe")
    assert unbooted["reason"].startswith("it didn't boot: LocalSandboxProvider: the computer at http://127.0.0.1:1 "
                                         "wasn't ready within 180 s")
    assert (stuck["status"], stuck["stage"], stuck["reason"]) == (
        "failed", "probe", f"the probe didn't finish within {probing._PROBE_TIMEOUT_S} s")


def test_known_failures_for_a_distribution_are_waived_on_it_alone_and_replace_its_os_types_there(
        base, docker, tmp_path):
    docker.probe_fails["arm64"] = "the click never landed"
    clicks = KnownFailure(check="P4", name="clicks show their effect", reason="its clicks lag")
    elsewhere = _driver(tmp_path / "elsewhere", known_failures={"linux/omarchy": [clicks]})
    here = _driver(tmp_path / "here", known_failures={
        "linux": [KnownFailure(check="P0", name="the card", reason="its card was late")], "linux/ubuntu": [clicks]})

    on_elsewhere = Env.get(_put(base, elsewhere, id="@local/~/kit/elsewhere", environment_name="elsewhere").id
                           ).metadata["composed"]["container"]["arm64"]
    on_here = Env.get(_put(base, here, id="@local/~/kit/here", environment_name="here").id).metadata[
        "composed"]["container"]["arm64"]
    docker.probe_fails.clear()

    assert on_elsewhere["status"] == "failed"
    assert on_here["status"] == "ready" and on_here["probe"]["known_failures"][0]["reason"] == "its clicks lag"
    assert "unused_known_failures" not in on_here["probe"]


def test_a_check_the_driver_declares_it_fails_is_waived_and_recorded_and_a_declared_one_that_didnt_fail_is_flagged(
        base, docker, tmp_path, caplog):
    docker.probe_fails["arm64"] = "the click never landed"
    elsewhere = _driver(tmp_path / "elsewhere", known_failures={"macos": [KnownFailure(
        check="P4", name="clicks show their effect", reason="its clicks lag on macOS")]})
    still_failed = Env.get(_put(base, elsewhere).id).metadata["composed"]["container"]["arm64"]
    known = _driver(tmp_path / "known", known_failures={"linux": [
        KnownFailure(check="P4", name="clicks show their effect", reason="its clicks lag"),
        KnownFailure(check="P4", name="clicks land", reason="a name the probe doesn't have")]})

    ready = Env.get(_put(base, known, id="@local/~/kit/known", environment_name="known").id).metadata[
        "composed"]["container"]["arm64"]

    assert still_failed["status"] == "failed"
    assert ready["status"] == "ready" and not ready["probe"]["passed"]
    assert ready["probe"]["known_failures"] == [{"check": "P4", "name": "clicks show their effect",
                                                 "detail": "the click never landed", "reason": "its clicks lag"}]
    assert ready["probe"]["unused_known_failures"] == [{"check": "P4", "name": "clicks land", "status": "absent"}]
    assert Env.get(known.id, known.version).known_failures["linux"][0].reason == "its clicks lag"
    docker.probe_fails.clear()
    fixed = _driver(tmp_path / "fixed", known_failures={"linux": [KnownFailure(
        check="P0", name="the card", reason="its card was late")]})
    passed = Env.get(_put(base, fixed, id="@local/~/kit/fixed", environment_name="fixed").id).metadata[
        "composed"]["container"]["arm64"]
    assert passed["status"] == "ready" and "known_failures" not in passed["probe"]
    assert passed["probe"]["unused_known_failures"] == [{"check": "P0", "name": "the card", "status": "passed"}]
    assert ("declares P0 'the card' a known failure on linux, and its container/arm64 probe found it passed: once what "
            "it declares is fixed, take it out" in caplog.text)


def test_a_probe_ended_by_a_dropped_connection_runs_again_on_a_fresh_boot_and_one_with_a_failed_check_doesnt(
        base, driver, docker):
    """The record keeps every attempt; a build whose every probe drops its connection fails at the probe."""
    docker.probe_drops["arm64"] = 1
    retried = Env.get(_put(base, driver).id).metadata["composed"]["container"]["arm64"]
    probes = len(docker.probed)
    docker.probe_fails["arm64"] = "the click never landed"
    strict = Env.get(_put(base, driver, id="@local/~/kit/strict", environment_name="strict").id).metadata[
        "composed"]["container"]["arm64"]
    strict_probes = len(docker.probed) - probes
    docker.probe_fails.clear()
    docker.probe_drops["arm64"] = probing._PROBE_ATTEMPTS
    dropped = Env.get(_put(base, driver, id="@local/~/kit/dropped", environment_name="dropped").id).metadata[
        "composed"]["container"]["arm64"]

    assert retried["status"] == "ready" and [a["status"] for a in retried["attempts"]] == ["dropped", "ready"]
    assert retried["attempts"][0]["reason"] == "RemoteProtocolError: Server disconnected without sending a response."
    assert retried["attempts"][1]["probe"] == retried["probe"] and probes == 2
    assert (strict["status"], strict_probes, "attempts" in strict) == ("failed", 1, False)
    assert (dropped["status"], dropped["stage"]) == ("failed", "probe")
    assert [a["status"] for a in dropped["attempts"]] == ["dropped"] * probing._PROBE_ATTEMPTS
    assert dropped["reason"] == (f"each of its {probing._PROBE_ATTEMPTS} probes, each on a fresh boot, ended on a "
                                 "transport error talking to its driver, the last RemoteProtocolError: Server "
                                 "disconnected without sending a response.")


@pytest.mark.parametrize("result, stored, replaces", [
    ({"status": "unverified"}, {"status": "ready"}, False),
    ({"status": "failed", "stage": "build"}, {"status": "unverified"}, False),
    ({"status": "building"}, {"status": "unverified"}, False),
    ({"status": "failed", "stage": "probe"}, {"status": "unverified"}, True),
    ({"status": "failed", "stage": "probe"}, {"status": "ready"}, False),
    ({"status": "ready"}, {"status": "failed", "stage": "probe"}, True),
    ({"status": "building"}, {"status": "failed", "stage": "build"}, True),
    ({"status": "building"}, {"status": "ready", "composer": "sha256:another"}, True),
    ({"status": "failed", "stage": "build"}, {"status": "ready", "composer": None}, True),
], ids=["unprobed-over-ready", "build-failure-over-built", "building-over-built", "probe-failure-over-unprobed",
        "probe-failure-over-ready", "ready-over-probe-failure", "building-over-build-failure",
        "building-over-another-composers-ready", "build-failure-over-a-ready-no-composer-named"])
def test_a_result_takes_a_pairs_place_only_when_it_got_at_least_as_far(result, stored, replaces):
    mine = {"composer": composed.composition()}
    assert (composed._rank({**mine, **result}) >= composed._rank({**mine, **stored})) is replaces


def test_the_composers_digest_changes_with_a_files_bytes_and_with_its_executable_bit_alone(tmp_path, monkeypatch):
    (tmp_path / "prepare.sh").write_text("echo hi\n")
    (tmp_path / "README.md").write_text("docs")
    monkeypatch.setattr(composed, "COMPOSING", (tmp_path,))
    digests = [composed.composition.__wrapped__()]
    (tmp_path / "prepare.sh").chmod(0o755)
    digests.append(composed.composition.__wrapped__())
    (tmp_path / "prepare.sh").write_text("echo bye\n")
    digests.append(composed.composition.__wrapped__())
    (tmp_path / "README.md").write_text("other docs")
    digests.append(composed.composition.__wrapped__())

    assert len(set(digests[:3])) == 3 and digests[3] == digests[2]


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
                                            r"'unverified', 'container/arm64': 'failed'\}; compose it with: ") as refused:
        await Env.get(env.id).deploy()
    assert str(refused.value).endswith(f"agent-env env build --id {env.id}")


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
    ({"tools": "all"}, "tools is 'standard', 'driver' or 'both', not 'all'"),
], ids=["display-zero", "display-side", "display-and-resolution", "network", "network-key",
        "settings-name", "settings-type", "settings-table", "tools"])
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


@pytest.mark.parametrize("names, problem", [
    ((), "apps: @local/~/kit/desk's screen is a website browser's page, so its computer takes exactly one website "
         "browser; it has 0"),
    (("first", "second"), "it has 2: @local/~/kit/first, @local/~/kit/second"),
])
def test_a_driver_whose_screen_is_a_page_takes_exactly_one_website_browser_and_any_other_apps(base, docker, tmp_path,
                                                                                               names, problem):
    page = _driver(tmp_path / "driver", features={"screen": "page"})
    mail = _app(None, None, tmp_path / "mail", compose=False)
    browsers = [_app(None, None, tmp_path / name, compose=False, name=name, browser=True) for name in names]

    with pytest.raises(ValueError) as refused:
        _put(base, page, compose=False, apps=[*browsers, mail])
    alone = _put(base, page, compose=False, apps=[_app(None, None, tmp_path / "one", compose=False, name="one",
                                                       browser=True), mail])
    desktop = _put(base, _driver(tmp_path / "desktop-driver"), compose=False, id="@local/~/kit/two",
                   apps=[_app(None, None, tmp_path / "a", compose=False, name="a", browser=True),
                         _app(None, None, tmp_path / "b", compose=False, name="b", browser=True)])

    assert problem in str(refused.value)
    assert [app.environment_name for app in alone.apps] == ["one", "mail"] and len(desktop.apps) == 2


def test_an_app_deployed_alone_on_a_driver_whose_screen_is_a_page_is_one_website_browser(base, docker, tmp_path):
    page = _driver(tmp_path / "driver", features={"screen": "page"})

    with pytest.raises(ValueError, match="app: @local/~/kit/desk's screen is a website browser's page, so its "
                                         "computer takes exactly one website browser; it has 0"):
        _app(base, page, tmp_path / "mail", compose=False)
    browser = _app(base, page, tmp_path / "browser", compose=False, name="browser", browser=True)

    assert browser.declaration.browser is True


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
    monkeypatch.setattr(composer, "pin_if_served", lambda ref, auth: None)
    docker.fail["linux/arm64"] = "boom"
    composer.compose(third, [("container", "arm64")])

    stored = Env.get(env.id)
    composed = stored.metadata["composed"]["container"]
    assert {arch: r["status"] for arch, r in composed.items()} == {"amd64": "unverified", "arm64": "ready"}
    assert stored.metadata["computer"]["images"] == {arch: r["image"] for arch, r in composed.items()}


DRIVER_CARD = {"name": "desktop", "capabilities": {"extensions": [
    {"uri": COMPUTER_DISPLAY, "params": {"endpoint": "/computer/display", "methods": {"get": {"method": "GET"}}}},
    {"uri": DISABLE_TOOL_EXTENSION_URI, "params": {"endpoint": "/tools/disable", "methods": {"disable": {}}}},
    {"uri": ENABLE_TOOL_EXTENSION_URI, "params": {"endpoint": "/tools/enable", "methods": {"enable": {}}}},
    {"uri": COMPUTER_RECORDING, "params": {"endpoint": "/computer/recording", "methods": {"end_all": {}}}}]}}


@pytest.fixture
def boots(monkeypatch):
    """The boots a deploy asks for, each a sandbox serving the desktop driver's card at 127.0.0.1:20001; the driver
    shows a 1600x900 screen, and each extension call it gets is recorded, with the headers it carried."""
    made = SimpleNamespace(boots=[], calls=[], headers=[], sandboxes=[], display=(1600, 900))

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

    async def invoke(base_url, card, uri, params=None, timeout=30, verify=True, *, method=None, headers=None):
        made.calls.append((uri, method, params))
        made.headers.append(headers)
        width, height = made.display
        return {"width": width, "height": height, "scale": 1, "modes": []} if uri == COMPUTER_DISPLAY else {}

    monkeypatch.setattr(env_computer_provider, "boot", fake_boot)
    monkeypatch.setattr(env_module.protocol_v1, "invoke_extension", invoke)
    return made


@pytest.mark.asyncio
async def test_a_computer_env_deploys_its_first_native_ready_build_with_its_settings_display_and_a_control_token(
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
    assert len(deployed.control_token) >= 32 and booted["token"] == deployed.control_token
    assert boots.calls == [(COMPUTER_DISPLAY, "get", None)]
    assert boots.headers == [{"Authorization": f"Bearer {deployed.control_token}"}]
    assert DeployedEnv.from_dict(dataclasses.asdict(deployed)).control_token == deployed.control_token
    second = await Env.get(env.id).deploy()
    assert second.control_token != deployed.control_token
    assert (deployed.env_provider_type, deployed.sandbox_ids) == ("computer", {"computer": {"desktop": "local-1"}})
    assert deployed.environment_card == DRIVER_CARD and deployed.mcp_url == "http://127.0.0.1:20001/mcp"
    record = deployed.metadata["computer"]
    assert {key: record[key] for key in ("form", "arch", "provider", "display", "computer_settings")} == {
        "form": "container", "arch": "arm64", "provider": "LocalSandboxProvider", "display": "1600x900",
        "computer_settings": {"resolution": "1600x900", "timezone": "Asia/Tokyo"}}
    assert record["computer"]["id"] == f"{env.id}__computer" and record["env_params"] == {
        "computer_settings": {"timezone": "Asia/Tokyo"}, "driver_settings": {"default_space": "norm"}}


@pytest.mark.asyncio
async def test_a_deploy_composes_again_what_an_agent_env_that_composes_differently_built_and_boots_that(
        base, driver, docker, boots, monkeypatch):
    env = _put(base, driver, display={"width": 1600, "height": 900})
    monkeypatch.setattr(composed, "composition", lambda: "sha256:a-later-agent-env")

    stale = Env.get(env.id)
    statuses, deployable = stale.build_statuses(), stale.deployable_builds()
    await stale.deploy()

    (booted,) = boots.boots
    record = Env.get(env.id).metadata["composed"]["container"]
    assert statuses == {"container/amd64": "unverified, by another composer",
                        "container/arm64": "ready, by another composer"} and deployable == []
    assert {arch: (r["status"], r["composer"]) for arch, r in record.items()} == {
        "amd64": ("unverified", "sha256:a-later-agent-env"), "arm64": ("ready", "sha256:a-later-agent-env")}
    assert booted["image"] == record["arm64"]["image"]


@pytest.mark.asyncio
async def test_a_deploy_probes_a_container_left_unprobed_once_its_setting_resolves_and_boots_it_only_once_ready(
        base, driver, docker, boots, tmp_path, monkeypatch):
    monkeypatch.delenv("MAIL_TOKEN", raising=False)
    probed = _probing_for_real(monkeypatch)
    env = _put(base, driver, display={"width": 1600, "height": 900}, apps=[_app(
        base, driver, tmp_path / "mail", compose=False, defaults={"token": "env:MAIL_TOKEN"})])
    unprobed = Env.get(env.id).metadata["composed"]["container"]["arm64"]
    monkeypatch.setenv("MAIL_TOKEN", "t0ken")

    deployed = await Env.get(env.id).deploy()

    build = Env.get(env.id).metadata["composed"]["container"]["arm64"]
    (booted,) = boots.boots
    assert unprobed["status"] == "unverified" and build["status"] == "ready"
    assert build["image"] == unprobed["image"] == booted["image"] == deployed.metadata["computer"]["image"]
    assert len(probed) == 1 and json.loads(probed[0]["AGENTENV_APP_SETTINGS"])["mail"]["token"] == "t0ken"


@pytest.mark.asyncio
async def test_a_deploy_refuses_a_container_still_unprobed_for_a_setting_that_doesnt_resolve_saying_why(
        base, driver, docker, boots, tmp_path, monkeypatch):
    monkeypatch.delenv("MAIL_TOKEN", raising=False)
    probed = _probing_for_real(monkeypatch)
    env = _put(base, driver, apps=[_app(base, driver, tmp_path / "mail", compose=False,
                                        defaults={"token": "env:MAIL_TOKEN"})])

    with pytest.raises(DeployRefused) as refused:
        await Env.get(env.id).deploy()

    assert str(refused.value) == (
        f"{env.id} has no ready build to deploy; its builds are {{'container/amd64': 'unverified', 'container/arm64': "
        "'unverified'}; container/arm64: not probed: a setting it boots with didn't resolve here: Unresolved "
        f"env:MAIL_TOKEN reference (no value and no default); compose it with: agent-env env build --id {env.id}")
    assert probed == [] and boots.boots == []


@pytest.mark.asyncio
async def test_a_deploy_says_why_its_native_container_is_unprobed_though_another_arch_it_cant_run_is_ready(
        base, driver, docker, boots, tmp_path, monkeypatch):
    monkeypatch.delenv("MAIL_TOKEN", raising=False)
    probed = _probing_for_real(monkeypatch)
    env = _put(base, driver, apps=[_app(base, driver, tmp_path / "mail", compose=False,
                                        defaults={"token": "env:MAIL_TOKEN"})])
    stored = Env.get(env.id)
    composed = stored.metadata["composed"]
    composed["container"]["amd64"] = {**composed["container"]["amd64"], "status": "ready"}
    stored.update_metadata({**stored.metadata, "composed": composed})

    with pytest.raises(DeployRefused) as refused:
        await Env.get(env.id).deploy()

    assert str(refused.value) == (
        f"{env.id} has no ready build to deploy; its builds are {{'container/amd64': 'ready', 'container/arm64': "
        "'unverified'}; container/arm64: not probed: a setting it boots with didn't resolve here: Unresolved "
        f"env:MAIL_TOKEN reference (no value and no default); compose it with: agent-env env build --id {env.id}")
    assert probed == [] and boots.boots == []


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
    assert kwargs["env"][CONTROL_TOKEN_ENV] == deployed.control_token
    env_computer_provider.ready.assert_awaited_once_with(deployed.environment_url, boot.READY_TIMEOUT_S,
                                                         deployed.control_token, answering=False)
    assert boots.calls == [(COMPUTER_RECORDING, "end_all", {"reason": "reset"}), "reset_container", (COMPUTER_DISPLAY, "get", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("asks", [1024, None])
async def test_a_computer_boots_with_the_shared_memory_its_apps_ask_for_records_it_and_a_reset_keeps_it(
        base, driver, docker, boots, monkeypatch, tmp_path, asks):
    monkeypatch.setenv("MAIL_TOKEN", "t0ken")
    boots.display = (1280, 800)
    install = {"python": "requirements.txt", "services": [{"name": "imap", "command": "venv/bin/python imap.py"}],
               **({"shm_mb": asks} if asks else {})}
    env = _put(base, driver, apps=[_app(base, driver, tmp_path / "mail", compose=False, install=install,
                                        defaults={"token": "env:MAIL_TOKEN"})])
    deployed = await Env.get(env.id).deploy()
    (sandbox,) = boots.sandboxes
    resets = []

    async def reset_container(self, sandbox, **kwargs):
        resets.append(kwargs)

    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "get_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "reset_container", reset_container)
    monkeypatch.setattr(env_computer_provider, "ready", AsyncMock(return_value=DRIVER_CARD))

    await ResetEnvTaskStep(id="reset", version=None, env_id=env.id, fail_task_on_error=True).execute(
        TaskStepContext(deployed_envs=[deployed]))

    ((booted,), (reset,)) = boots.boots, resets
    assert booted["shm_mb"] == asks and deployed.metadata["computer"].get("shm_mb") == asks
    assert reset.get("shm_mb") == asks


@pytest.mark.asyncio
@pytest.mark.parametrize("tools, changes", [
    ("standard", [(DISABLE_TOOL_EXTENSION_URI, "disable", {"role": "default", "tools": ["*"]}),
                  (ENABLE_TOOL_EXTENSION_URI, "enable", {"role": "default", "tools": list(COMPUTER_TOOLS)})]),
    ("driver", [(DISABLE_TOOL_EXTENSION_URI, "disable", {"role": "default", "tools": list(COMPUTER_TOOLS)})]),
    ("both", []),
])
async def test_a_computer_envs_tools_are_given_its_agents_at_deploy_and_again_after_a_reset(
        base, driver, docker, boots, monkeypatch, tools, changes):
    env = _put(base, driver, display={"width": 1600, "height": 900}, tools=tools)
    deployed = await Env.get(env.id).deploy()
    (sandbox,) = boots.sandboxes
    given = [call for call in boots.calls if call[0] in (DISABLE_TOOL_EXTENSION_URI, ENABLE_TOOL_EXTENSION_URI)]
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "get_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "reset_container", AsyncMock())
    monkeypatch.setattr(env_computer_provider, "ready", AsyncMock(return_value=DRIVER_CARD))
    boots.calls.clear()

    await ResetEnvTaskStep(id="reset", version=None, env_id=env.id, fail_task_on_error=True).execute(
        TaskStepContext(deployed_envs=[deployed]))

    assert Env.get(env.id).tools == tools and given == changes
    assert [call for call in boots.calls if call[0] in (DISABLE_TOOL_EXTENSION_URI, ENABLE_TOOL_EXTENSION_URI)] == changes


@pytest.mark.asyncio
async def test_a_deploy_with_a_tool_prefix_has_its_driver_serve_its_tools_under_it_and_gives_them_by_those_names(
        base, driver, docker, boots, monkeypatch):
    env = _put(base, driver, display={"width": 1600, "height": 900}, tools="standard")
    deployed = await Env.get(env.id).deploy(tool_prefix="desktop_")
    (booted,), (sandbox,) = boots.boots, boots.sandboxes
    given = [call for call in boots.calls if call[0] in (DISABLE_TOOL_EXTENSION_URI, ENABLE_TOOL_EXTENSION_URI)]
    reset = AsyncMock()
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "get_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "reset_container", reset)
    monkeypatch.setattr(env_computer_provider, "ready", AsyncMock(return_value=DRIVER_CARD))
    boots.calls.clear()

    await ResetEnvTaskStep(id="reset", version=None, env_id=env.id, fail_task_on_error=True).execute(
        TaskStepContext(deployed_envs=[deployed]))

    changes = [(DISABLE_TOOL_EXTENSION_URI, "disable", {"role": "default", "tools": ["*"]}),
               (ENABLE_TOOL_EXTENSION_URI, "enable", {"role": "default",
                                                      "tools": [f"desktop_{tool}" for tool in COMPUTER_TOOLS]})]
    assert booted["environment"][TOOL_PREFIX_ENV] == "desktop_" == deployed.metadata["computer"]["tool_prefix"]
    assert reset.call_args.kwargs["env"][TOOL_PREFIX_ENV] == "desktop_"
    assert given == changes
    assert [call for call in boots.calls if call[0] in (DISABLE_TOOL_EXTENSION_URI, ENABLE_TOOL_EXTENSION_URI)] == changes


@pytest.mark.asyncio
async def test_a_deploy_without_a_tool_prefix_gives_its_driver_none(base, driver, docker, boots):
    deployed = await Env.get(_put(base, driver, display={"width": 1600, "height": 900}).id).deploy()
    assert TOOL_PREFIX_ENV not in boots.boots[0]["environment"] and "tool_prefix" not in deployed.metadata["computer"]


@pytest.mark.asyncio
async def test_a_tool_prefix_a_tool_name_cant_hold_is_refused_before_anything_boots(base, driver, docker, boots):
    with pytest.raises(Exception, match="tool prefix 'desk top_' holds characters a tool's name can't"):
        await Env.get(_put(base, driver, display={"width": 1600, "height": 900}).id).deploy(tool_prefix="desk top_")
    assert boots.boots == []


@pytest.mark.asyncio
@pytest.mark.parametrize("allow, reach, allowed", [
    (["pypi.org"], "https://gw.example/mcp", ("pypi.org", "gw.example")),
    (["pypi.org"], "http://localhost:18765", ("pypi.org", "host.docker.internal")),
    (["gw.example"], "https://gw.example", ("gw.example",)),
    (["pypi.org"], "", ("pypi.org",)),
], ids=["added", "as-its-sandbox-reaches-it", "already-allowed", "nothing-to-reach"])
async def test_a_deploy_given_a_url_to_reach_allows_its_host_on_a_network_that_lists_the_hosts_allowed(
        base, driver, docker, boots, monkeypatch, allow, reach, allowed):
    class Enforces(local_sandbox.LocalSandboxProvider):
        @classmethod
        def supports_network_policy(cls, policy):
            return True

    monkeypatch.setattr(providers, "build_sandbox_provider", lambda spec: Enforces())
    env = _put(base, driver, display={"width": 1600, "height": 900}, network={"mode": "allowlist", "allow_hosts": allow})

    await Env.get(env.id).deploy(sandbox_type="enforcing", reach=reach)

    assert boots.boots[0]["network_policy"].allow_hosts == allowed


@pytest.mark.asyncio
async def test_a_deploy_given_a_url_to_reach_leaves_a_network_it_doesnt_restrict_as_it_is(base, driver, docker, boots):
    await Env.get(_put(base, driver, display={"width": 1600, "height": 900}).id).deploy(reach="https://gw.example")
    assert boots.boots[0]["network_policy"] is None


@pytest.mark.asyncio
async def test_a_deploy_whose_tools_the_driver_cant_take_fails_and_tears_down_its_computer_alone(
        base, driver, docker, boots, monkeypatch):
    live = Env.get(_put(base, driver, display={"width": 1600, "height": 900}, tools="standard").id)
    first = await live.deploy()
    monkeypatch.setitem(DRIVER_CARD["capabilities"], "extensions", [
        extension for extension in DRIVER_CARD["capabilities"]["extensions"] if extension["uri"] != ENABLE_TOOL_EXTENSION_URI])

    with pytest.raises(EnvCapabilityUnsupported):
        await live.deploy()

    kept, failed = boots.sandboxes
    assert failed.terminated and not kept.terminated and live._deployed is first


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


NOTES_CARD = {"name": "notes", "capabilities": {"extensions": [
    {"uri": "urn:agentenv:website/v1", "params": {"domains": ["notes.test"]}}]}}


def _sites_gateway():
    from agent_env.env.env import DeployedGatewayEnv

    return DeployedGatewayEnv(env_id="@local/~/kit/sites", env_version=1, sandbox_id="gw", sandbox_type="local",
                              gateway_url="http://localhost:8080",
                              website_frontend_urls={"notes": "http://localhost:8080/website/notes/"})


@pytest.fixture
def routes(monkeypatch):
    """Each site routing a deploy asks for, as (sandbox, sites, options), answering what the proxy's install would; the
    gateway serving the notes site's card, its websites ready."""
    from agentenv_protocol import client as protocol_v1

    made = []

    async def route(sandbox, sites, **options):
        made.append((sandbox, sites, options))
        return {"names": {"routes": {site.name: site.domains for site in sites}}, "ca": "CA PEM"}

    async def get_card(url, timeout=10, verify=True):
        assert url == "http://localhost:8080/svc/notes"
        return NOTES_CARD

    monkeypatch.setattr(env_computer_provider.sites, "route", route)
    monkeypatch.setattr(env_computer_provider.sites, "require_ready", AsyncMock())
    monkeypatch.setattr(protocol_v1, "get_card", get_card)
    monkeypatch.setattr(env_computer_provider, "ready", AsyncMock(return_value=DRIVER_CARD))
    return made


@pytest.mark.asyncio
async def test_a_computer_with_site_routing_reaches_the_websites_the_run_deployed_before_it_by_their_names(
        base, driver, docker, boots, routes):
    env = _put(base, driver, display={"width": 1600, "height": 900})
    context = TaskStepContext(deployed_envs=[_sites_gateway()])

    step = DeployEnvTaskStep(id="d", version=None, env_id=env.id, ttl_seconds=3600, env_params={"site_routing": True})

    assert step.preflight() == []
    await step.execute(context)

    ((sandbox, sites, options),) = routes
    assert sandbox is boots.sandboxes[0]
    assert [site.model_dump() for site in sites] == [{"name": "notes", "url": "http://localhost:8080/website/notes/",
                                                      "domains": ["notes.test"]}]
    assert options == {"user": "user", "network": None, "valid_for": timedelta(seconds=3600), "label": "desktop",
                       "os_type": "linux", "display": "x11", "live_hosts": ()}
    assert boots.boots[0]["environment"][HOLD_BOOT_LAUNCH_ENV] == "1"
    record = context.deployed_envs[-1].metadata["computer"]
    assert record["sites"] == {"names": {"routes": {"notes": ["notes.test"]}}, "ca": "CA PEM", "valid_for_s": 3600}
    assert record["env_params"]["site_routing"] is True and record["env_params"]["sites"][0]["name"] == "notes"


@pytest.mark.asyncio
async def test_a_computer_deployed_to_route_its_sites_later_boots_held_and_routes_them_when_given_as_a_multi_env_does(
        base, driver, docker, boots, routes):
    env = Env.get(_put(base, driver, display={"width": 1600, "height": 900}).id)

    deployed = await env.deploy(ttl_seconds=3600, env_params={"site_routing": True}, sites_later=True)
    held, routed_before = boots.boots[0]["environment"][HOLD_BOOT_LAUNCH_ENV], list(routes)
    await env.route_sites([{"name": "notes", "url": "http://localhost:8080/website/notes/", "domains": ["notes.test"]}])

    assert held == "1" and routed_before == []
    # The trust its held apps launch with, which a boot knowing its sites would give it, before they're known.
    assert boots.boots[0]["environment"].items() >= computer_sites.TRUST_ENVIRONMENTS["linux"].items()
    ((sandbox, sites, options),) = routes
    assert sandbox is boots.sandboxes[0] and [site.name for site in sites] == ["notes"]
    assert options["valid_for"] == timedelta(seconds=3600)
    record = deployed.metadata["computer"]
    assert record["env_params"]["sites"][0]["name"] == "notes" and record["sites"]["valid_for_s"] == 3600


@pytest.mark.asyncio
async def test_a_computer_given_its_sites_later_still_refuses_sites_its_params_cant_hold(base, driver, docker, boots, routes):
    env = Env.get(_put(base, driver, display={"width": 1600, "height": 900}).id)
    await env.deploy(env_params={"site_routing": True}, sites_later=True)

    with pytest.raises(ValueError, match="sites"):
        await env.route_sites([{"name": "notes"}])
    assert routes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("routing", ["web-env", 1, ["web-env", 2], {"web-env": True}])
async def test_a_site_routing_that_is_neither_true_nor_a_list_of_env_ids_is_refused_before_it_is_read(
        base, driver, routing):
    env = _put(base, driver, display={"width": 1600, "height": 900})
    with pytest.raises(ValueError, match="site_routing is true or a list of env ids"):
        await env.run_env_params({"site_routing": routing}, [_sites_gateway()])


@pytest.mark.asyncio
async def test_a_computer_without_site_routing_routes_nothing_whatever_websites_the_run_deployed(
        base, driver, docker, boots, routes):
    env = _put(base, driver, display={"width": 1600, "height": 900})
    context = TaskStepContext(deployed_envs=[_sites_gateway()])

    await DeployEnvTaskStep(id="d", version=None, env_id=env.id).execute(context)

    assert routes == [] and "sites" not in context.deployed_envs[-1].metadata["computer"]
    assert HOLD_BOOT_LAUNCH_ENV not in boots.boots[0]["environment"] and boots.boots[0]["held"] is False
    assert not set(env_computer_provider.sites.TRUST_ENVIRONMENTS["linux"]) & set(boots.boots[0]["environment"])


@pytest.mark.asyncio
async def test_a_computer_routing_sites_boots_with_its_trust_until_its_driver_answers_and_checks_its_display(
        base, driver, docker, boots, routes, monkeypatch):
    """A page driver's screen is its browser's page, whose launch a deploy routing sites holds: the boot waits only for
    the driver to answer, and the display is checked once the sites are routed and the computer is ready."""
    env = _put(base, driver, display={"width": 1600, "height": 900})
    order = []
    routed = env_computer_provider.sites.route

    async def route(sandbox, sites, **options):
        order.append("route")
        return await routed(sandbox, sites, **options)

    async def ready(url, timeout, token, **kwargs):
        order.append(("ready", kwargs))
        return DRIVER_CARD

    async def require_display(record, resolution):
        order.append("display")

    monkeypatch.setattr(env_computer_provider.sites, "route", route)
    monkeypatch.setattr(env_computer_provider, "ready", ready)
    monkeypatch.setattr(env_computer_provider, "_require_display", require_display)
    await DeployEnvTaskStep(id="d", version=None, env_id=env.id, ttl_seconds=3600,
                            env_params={"site_routing": True}).execute(TaskStepContext(deployed_envs=[_sites_gateway()]))

    assert boots.boots[0]["held"] is True and order == ["route", ("ready", {}), "display"]
    assert env_computer_provider.sites.TRUST_ENVIRONMENTS["linux"].items() <= boots.boots[0]["environment"].items()
    assert "AGENTENV_SITE_PROXY" not in boots.boots[0]["environment"]


@pytest.mark.asyncio
async def test_a_held_boot_waits_for_the_driver_to_answer_and_still_fails_as_soon_as_it_says_it_failed(monkeypatch):
    statuses = []

    async def get_card(url, timeout=5):
        return DRIVER_CARD

    async def invoke(url, card, uri, params, timeout, *, method=None, headers=None):
        return statuses[0]

    monkeypatch.setattr(boot.protocol_v1, "get_card", get_card)
    monkeypatch.setattr(boot.protocol_v1, "invoke_extension", invoke)
    monkeypatch.setattr(boot, "_POLL_S", 0.01)
    statuses[:] = [{"ready": False, "checks": [{"name": "browser", "ok": False, "detail": "no page yet"}]}]

    assert await boot.ready("http://computer", 5, "t0ken", answering=True) == DRIVER_CARD
    with pytest.raises(TimeoutError, match="browser: no page yet"):
        await boot.ready("http://computer", 0.2, "t0ken")
    statuses[:] = [{"ready": False, "failed": "its driver exited"}]
    with pytest.raises(boot.ComputerFailed, match="its driver exited"):
        await boot.ready("http://computer", 5, "t0ken", answering=True)


@pytest.mark.asyncio
async def test_a_computer_routing_sites_launches_its_held_boot_apps_once_listed_then_keeps_its_ready_card(monkeypatch):
    order = []
    apps = [SimpleNamespace(environment_name=name, launch=lambda os_type, at=at: SimpleNamespace(at=at))
            for name, at in [("mail", "boot"), ("music", "demand"), ("notes", "boot")]]
    env = SimpleNamespace(computer=SimpleNamespace(os=SimpleNamespace(type="linux")), composed_apps=lambda: apps)

    def app_card(name):
        return {"name": name, "url": f"/svc/{name}/agentenv", "capabilities": {"extensions": [
            {"uri": COMPUTER_APP, "params": {"endpoint": f"/svc/{name}/agentenv/ext/app", "methods": {"launch": {}}}}]}}

    listed_card = {"name": "mail", "capabilities": {"extensions": [{"uri": COMPUTER_READY, "params": {}}]},
                   "children_environments": [app_card("mail"), app_card("music"), app_card("notes")]}

    async def invoke(uri, method, params=None, *, environment_name=None, child=None):
        order.append((uri, method, child.name, child.card == app_card(child.name), record.environment_card["name"]))

    async def listed(url, names, timeout):
        order.append(("listed", url, names))
        return listed_card

    async def ready(url, timeout, token):
        order.append(("ready", url, token))
        return {"name": "ready"}

    monkeypatch.setattr(env_computer_provider, "listed", listed)
    monkeypatch.setattr(env_computer_provider, "ready", ready)
    monkeypatch.setattr(env_computer_provider, "ready_timeout", lambda apps, os_type: 30.0)
    record = SimpleNamespace(invoke=invoke, control_token="t0ken", environment_card={"name": "booted"})
    await env_computer_provider._launch_held_apps(env, record, "http://computer", [])

    assert order == [("listed", "http://computer", ["mail", "notes"]), (COMPUTER_APP, "launch", "mail", True, "mail"),
                     (COMPUTER_APP, "launch", "notes", True, "mail"), ("ready", "http://computer", "t0ken")]
    assert record.environment_card == {"name": "ready"} and record.environment_card_read_at_utc
    order.clear()
    apps[:] = [apps[1]]
    await env_computer_provider._launch_held_apps(env, record, "http://computer", [])
    assert order == [("ready", "http://computer", "t0ken")]


def _app_card(name, browser_methods=None):
    """A held app's card, launched through computer-app/v1, a website browser when it serves ``browser_methods``."""
    extensions = [{"uri": COMPUTER_APP, "params": {"endpoint": f"/svc/{name}/agentenv/ext/app",
                                                   "methods": {"launch": {}}}}]
    if browser_methods is not None:
        extensions.append({"uri": BROWSER, "params": {"endpoint": f"/svc/{name}/agentenv/ext/browser",
                                                      "methods": {method: {} for method in browser_methods}}})
    return {"name": name, "url": f"/svc/{name}/agentenv", "capabilities": {"extensions": extensions}}


@pytest.mark.asyncio
async def test_each_held_browser_is_handed_the_routed_sites_before_its_launch_and_one_built_before_is_only_launched(
        monkeypatch, caplog):
    from agent_env.env.env import EnvCapabilityUnsupported
    from agent_env.env.envs.computer.utils.params import Site

    calls = []
    apps = [SimpleNamespace(environment_name=name, launch=lambda os_type: SimpleNamespace(at="boot"))
            for name in ("chromium", "older", "mail")]
    env = SimpleNamespace(computer=SimpleNamespace(os=SimpleNamespace(type="linux")), composed_apps=lambda: apps)
    listed_card = {"name": "desk", "children_environments": [
        _app_card("chromium", ["record", "sites/set"]), _app_card("older", ["record"]), _app_card("mail")]}

    async def invoke(uri, method, params=None, *, environment_name=None, child=None):
        if method == "sites/set" and child.name == "older":
            raise EnvCapabilityUnsupported(uri, method, "desk", None, child.name)
        calls.append((uri, method, child.name, params))

    monkeypatch.setattr(env_computer_provider, "listed", AsyncMock(return_value=listed_card))
    monkeypatch.setattr(env_computer_provider, "ready", AsyncMock(return_value=listed_card))
    monkeypatch.setattr(env_computer_provider, "ready_timeout", lambda apps, os_type: 30.0)
    record = SimpleNamespace(invoke=invoke, control_token="t0ken", environment_card={}, env_id="desk")
    routed = [Site(name="shop", url="http://gw/website/shop/", domains=["shop.test", "*.shop.test"]),
              Site(name="bare", url="http://gw/website/bare/")]

    with caplog.at_level(logging.WARNING):
        await env_computer_provider._launch_held_apps(env, record, "http://computer", routed)

    assert calls == [(BROWSER, "sites/set", "chromium", {"sites": [{"name": "shop", "domains": ["shop.test",
                                                                                                  "*.shop.test"]}]}),
                     (COMPUTER_APP, "launch", "chromium", None), (COMPUTER_APP, "launch", "older", None),
                     (COMPUTER_APP, "launch", "mail", None)]
    assert "older in desk serves no browser/v1 sites/set" in caplog.text


def _start_tabs_record(card, launches_by_browser):
    """A deployed computer whose card is ``card`` and whose browsers' ``record`` keeps ``launches_by_browser``, by the
    browser's name, None for the computer's own card; a browser absent from it built before start tabs."""
    from agent_env.env.env import EnvCapabilityUnsupported

    async def invoke(uri, method, params=None, *, environment_name=None, child=None):
        name = child.name if child is not None else None
        if (uri, method) != (BROWSER, "record") or name not in launches_by_browser:
            raise EnvCapabilityUnsupported(uri, method, "desk", None, name)
        return {"start_tabs": [{"at": "t", "tabs": tabs} for tabs in launches_by_browser[name]]}

    return SimpleNamespace(invoke=invoke, environment_card=card, env_id="@local/desk")


def _tab(url, site=None, outcome="loaded", **fields):
    return {"tab": "t1", "url": url, "final_url": url, "status": 200, "error": None, "load_ms": 120, "site": site,
            "outcome": outcome, **fields}


@pytest.mark.asyncio
async def test_each_browsers_start_tabs_of_its_latest_launch_are_kept_and_one_built_before_them_is_left_out():
    card = {"name": "desk", "children_environments": [_app_card("chromium", ["record"]), _app_card("older", []),
                                                      _app_card("mail")]}
    record = _start_tabs_record(card, {"chromium": [[_tab("about:blank")], [
        _tab("https://shop.test/", "shop"), _tab("https://slack.com/", outcome="timed_out", status=None)]]})
    alone = _start_tabs_record({"name": "chromium-driver", "capabilities": {"extensions": [
        {"uri": BROWSER, "params": {"endpoint": "/ext/browser", "methods": {"record": {}}}}]}},
        {None: [[_tab("https://linear.app/")]]})

    assert await env_computer_provider._start_tabs(record) == {"chromium": [
        _tab("https://shop.test/", "shop"), _tab("https://slack.com/", outcome="timed_out", status=None)]}
    assert await env_computer_provider._start_tabs(alone) == {"chromium-driver": [_tab("https://linear.app/")]}


@pytest.mark.asyncio
async def test_a_start_tab_on_one_of_the_env_sites_that_didnt_load_fails_the_boot_naming_each_once_kept(
        monkeypatch):
    card = {"name": "desk", "children_environments": [_app_card("chromium", ["record"])]}
    tabs = [_tab("https://shop.test/", "shop", "failed", status=503),
            _tab("https://mail.test/", "mail", "failed", status=None, error="net::ERR_CONNECTION_REFUSED"),
            _tab("https://notes.test/", "notes", "timed_out", status=None, load_ms=60000),
            _tab("https://slack.com/", None, "failed", status=500)]
    record = _start_tabs_record(card, {"chromium": [tabs]})
    record.metadata = {"computer": {"start_tabs": {"chromium": [_tab("https://shop.test/", "shop")]}}}
    monkeypatch.setattr(env_computer_provider, "_require_display", AsyncMock())

    with pytest.raises(RuntimeError) as failed:
        await env_computer_provider._finish_boot(SimpleNamespace(), record, None,
                                                 SimpleNamespace(site_routing=False), 0, None)

    assert record.metadata["computer"]["start_tabs"] == {"chromium": tabs}
    assert str(failed.value) == (
        "computer env '@local/desk': chromium's start tab https://shop.test/ on the env's site shop answered 503; "
        "chromium's start tab https://mail.test/ on the env's site mail didn't load: net::ERR_CONNECTION_REFUSED; "
        "chromium's start tab https://notes.test/ on the env's site notes was still loading after 60 s")


@pytest.mark.asyncio
async def test_a_held_app_is_awaited_until_the_driver_lists_it_with_its_server(monkeypatch):
    """A page driver answers before its browser's server does, and lists the browser by its name alone until then."""
    named_only = {"name": "browser", "children_environments": [{"name": "chromium", "url": "/svc/chromium/agentenv"}]}
    served = {"name": "browser", "children_environments": [{"name": "chromium", "url": "/svc/chromium/agentenv",
        "capabilities": {"extensions": [{"uri": COMPUTER_APP, "params": {"endpoint": "/svc/chromium/agentenv/ext/app",
                                                                         "methods": {"launch": {}}}}]}}]}
    cards = [named_only, named_only, served]

    async def get_card(url, timeout=5):
        return cards.pop(0) if len(cards) > 1 else cards[0]

    monkeypatch.setattr(boot.protocol_v1, "get_card", get_card)
    monkeypatch.setattr(boot, "_POLL_S", 0.01)

    assert await boot.listed("http://computer", ["chromium"], 5) == served
    cards[:] = [named_only]
    with pytest.raises(TimeoutError, match="didn't list chromium with its server"):
        await boot.listed("http://computer", ["chromium"], 0.2)


@pytest.mark.asyncio
async def test_a_reset_routes_the_fresh_computer_to_the_same_sites_under_a_ca_of_its_own(
        base, driver, docker, boots, routes, monkeypatch):
    env = _put(base, driver, display={"width": 1600, "height": 900})
    context = TaskStepContext(deployed_envs=[_sites_gateway()])
    await DeployEnvTaskStep(id="d", version=None, env_id=env.id, ttl_seconds=3600,
                            env_params={"site_routing": True}).execute(context)
    (sandbox,) = boots.sandboxes
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "get_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "reset_container", AsyncMock())
    monkeypatch.setattr(env_computer_provider, "ready", AsyncMock(return_value=DRIVER_CARD))

    await ResetEnvTaskStep(id="reset", version=None, env_id=env.id, fail_task_on_error=True).execute(context)

    first, again = routes
    assert again[0] is sandbox and again[1] == first[1] and again[2]["valid_for"] == timedelta(seconds=3600)
    assert context.deployed_envs[-1].metadata["computer"]["sites"]["valid_for_s"] == 3600


@pytest.mark.asyncio
async def test_site_routing_with_no_website_deployed_or_outside_a_run_is_refused_before_anything_boots(
        base, driver, docker, boots, routes):
    env = _put(base, driver, display={"width": 1600, "height": 900})

    with pytest.raises(ValueError, match="the run deployed none before the computer"):
        await DeployEnvTaskStep(id="d", version=None, env_id=env.id, env_params={"site_routing": True}).execute(
            TaskStepContext())
    with pytest.raises(DeployRefused, match="which a run's deploy_env names in env_params.sites"):
        await Env.get(env.id).deploy(env_params={"site_routing": True})

    assert boots.boots == [] and routes == []


@pytest.mark.asyncio
async def test_site_routing_on_a_disk_computer_is_refused_before_it_boots_and_a_container_form_is_kept(
        base, driver, docker, boots, routes, monkeypatch):
    from agent_env.env.envs.computer.env import ComputerEnv

    env = _put(base, driver, display={"width": 1600, "height": 900})
    container = ComputerEnv.deployable_builds(Env.get(env.id))
    disk = [("disk", arch, build) for _, arch, build in container]
    monkeypatch.setattr(ComputerEnv, "deployable_builds", lambda self, form=None: disk)
    params = {"site_routing": True, "sites": [{"name": "a", "url": "http://gw/website/a/", "domains": ["a.test"]}]}

    with pytest.raises(DeployRefused, match="a disk computer runs inside a launcher container"):
        await Env.get(env.id).deploy(env_params=params)

    assert boots.boots == [] and routes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("env_params, problem", [
    ({"site_routing": True, "sites": [{"name": "a", "url": "http://gw/website/a/", "domains": ["x.test"]},
                                      {"name": "b", "url": "http://gw/website/b/", "domains": ["x.test"]}]},
     "env_params.sites: names claimed twice: 'x.test' (b) overlaps 'x.test' (a)"),
    ({"sites": [{"name": "a", "url": "http://gw/website/a/"}]},
     "env_params.sites: a computer reaches sites only with site_routing"),
    ({"site_routing": "yes"}, "env_params.site_routing"),
], ids=["conflict", "sites-alone", "not-a-flag"])
async def test_sites_a_computer_cant_reach_by_their_names_are_refused_at_save_and_at_deploy(
        base, driver, docker, boots, routes, env_params, problem):
    env = _put(base, driver, display={"width": 1600, "height": 900})

    saved = DeployEnvTaskStep(id="d", version=None, env_id=env.id, env_params=env_params).preflight()
    with pytest.raises(DeployRefused, match=re.escape(problem)):
        await Env.get(env.id).deploy(env_params=env_params)

    assert len(saved) == 1 and problem in saved[0] and boots.boots == [] and routes == []


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


def test_a_drivers_own_settings_are_checked_against_the_schema_it_declares(base, driver):
    env = _put(base, driver)
    env.driver.declaration = env.driver.declaration.model_copy(update={"settings": {
        "type": "object", "additionalProperties": False,
        "properties": {"headed": {"type": "boolean"}, "default_space": {"enum": ["frame", "norm"]}}}})

    assert params.driver_problems(env, {"headed": True, "default_space": "norm"}) == []
    assert params.driver_problems(env, {"headed": "yes"}) == ["env_params.driver_settings.headed: 'yes' is not of type "
                                                              "'boolean'"]
    assert "Additional properties are not allowed ('colour' was unexpected)" in params.driver_problems(
        env, {"colour": "blue"})[0]


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
async def test_a_boot_hands_the_computer_its_token_and_one_never_ready_takes_its_sandbox_and_work_folder_down(
        monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))
    sandbox = local_sandbox.LocalSandbox(exposed_ports=[18765], sandbox_id="local-unready")
    sandbox.terminate = AsyncMock()
    monkeypatch.setattr(boot, "ready", AsyncMock(side_effect=TimeoutError("not ready")))
    create = AsyncMock(return_value=sandbox)

    with pytest.raises(TimeoutError):
        await boot.boot(SimpleNamespace(create_container=create), image="img", port=18765,
                        environment={"ENVIRONMENT_NAME": "desk"}, token="t0ken", network_policy=None, timeout=60,
                        attribution=None)

    assert create.await_args.kwargs["env"] == {"ENVIRONMENT_NAME": "desk", CONTROL_TOKEN_ENV: "t0ken"}
    assert boot.ready.await_args.args[2] == "t0ken"
    sandbox.terminate.assert_awaited_once()
    assert local_sandbox.LocalSandbox.find_work_dir("local-unready") is None


@pytest.mark.asyncio
async def test_a_boot_is_ready_with_the_card_read_once_its_status_asked_with_its_token_says_so_each_app_on_it(
        monkeypatch):
    cards = iter([{"name": "desk"}, {"name": "desk"}, {"name": "desk", "children_environments": [{"name": "mail"}]}])
    statuses = iter([{"ready": False, "checks": [{"name": "app mail", "ok": False, "detail": "starting"}]},
                     {"ready": True}])
    sent = []

    async def card(url, timeout):
        return next(cards)

    async def status(url, card, uri, params, timeout, method, headers):
        sent.append(headers)
        return next(statuses)

    monkeypatch.setattr(boot.protocol_v1, "get_card", card)
    monkeypatch.setattr(boot.protocol_v1, "invoke_extension", status)
    monkeypatch.setattr(boot, "_POLL_S", 0)

    assert await boot.ready("http://desk", 5, "t0ken") == {"name": "desk", "children_environments": [{"name": "mail"}]}
    assert sent == [{"Authorization": "Bearer t0ken"}] * 2


@pytest.mark.asyncio
async def test_a_boot_whose_status_says_the_computer_failed_raises_at_once_with_why(monkeypatch):
    asked = []

    async def card(url, timeout):
        return {"name": "desk"}

    async def status(url, card, uri, params, timeout, method, headers):
        asked.append(method)
        return {"ready": False, "failed": "browser: it ended on its own", "checks": [
            {"name": "browser", "ok": False, "detail": "it ended on its own", "final": True}]}

    monkeypatch.setattr(boot.protocol_v1, "get_card", card)
    monkeypatch.setattr(boot.protocol_v1, "invoke_extension", status)
    monkeypatch.setattr(boot, "_POLL_S", 0)

    with pytest.raises(boot.ComputerFailed, match="the computer at http://desk failed: browser: it ended on its own"):
        await boot.ready("http://desk", 5, "t0ken")
    assert asked == ["status"]


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
    ("websockets", "LocalSandboxProvider carries no WebSockets, which its website browser's CDP address needs"),
])
def test_a_deploy_its_sandbox_provider_cant_carry_out_is_refused_at_save(base, driver, docker, monkeypatch, tmp_path,
                                                                       change, problem):
    browser = [_app(None, None, tmp_path / "chromium", compose=False, name="chromium", browser=True)]
    env = _put(base, driver, network={"mode": "allowlist", "allow_hosts": ["example.com"]} if change == "network" else None,
               apps=browser if change == "websockets" else [])
    if change == "riscv64":
        monkeypatch.setattr(local_sandbox, "host_platform", lambda: "linux/riscv64")
    if change == "websockets":
        assert DeployEnvTaskStep(id="d", version=None, env_id=env.id).preflight() == []
        monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "WEBSOCKETS", False)

    (refusal,) = DeployEnvTaskStep(id="d", version=None, env_id=env.id).preflight()

    assert problem in refusal


def test_a_chain_with_no_one_link_that_both_carries_websockets_and_enforces_the_network_is_refused_at_save(
        base, driver, docker, monkeypatch, tmp_path):
    class Enforces(local_sandbox.LocalSandboxProvider):
        WEBSOCKETS = False

        @classmethod
        def supports_network_policy(cls, policy):
            return True

    class Carries(local_sandbox.LocalSandboxProvider):
        pass

    monkeypatch.setattr(providers, "build_sandbox_provider",
                        lambda spec: ChainedSandboxProvider([Enforces(), Carries()]))
    env = _put(base, driver, network={"mode": "allowlist", "allow_hosts": ["example.com"]},
               apps=[_app(None, None, tmp_path / "chromium", compose=False, name="chromium", browser=True)])

    assert Env.get(env.id).deploy_refusal(sandbox_type="chain") == f"computer env {env.id!r}: Carries can't enforce its network"


@pytest.mark.asyncio
async def test_a_deploy_with_a_website_browser_skips_a_link_that_carries_no_websockets_and_boots_on_the_next(
        base, driver, docker, boots, monkeypatch, tmp_path):
    env = _put(base, driver, display={"width": 1600, "height": 900},
               apps=[_app(None, None, tmp_path / "chromium", compose=False, name="chromium", browser=True)])

    class NoWebSockets(local_sandbox.LocalSandboxProvider):
        WEBSOCKETS = False

    chain = ChainedSandboxProvider([NoWebSockets(), local_sandbox.LocalSandboxProvider()])
    monkeypatch.setattr(providers, "build_sandbox_provider", lambda spec: chain)

    await Env.get(env.id).deploy(sandbox_type="chain")

    assert [made["provider"] for made in boots.boots] == ["LocalSandboxProvider"]


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
async def test_an_app_deployed_alone_loads_its_own_name_through_its_route_and_its_computers_through_the_driver(
        base, driver, docker, computer, tmp_path, monkeypatch):
    monkeypatch.setattr(inside, "issues_grants_to", lambda store, sandbox_type: False)
    mail = _app(base, driver, tmp_path / "mail", compose=False)
    mail._deployed = computer.deployed(env_id=mail.id, card={**HOME_CARD, "name": "mail-driver", "children_environments": [
        _child("mail", "data/add", "data/get")]})

    await mail.load_environment_universe_artifact(EnvironmentUniverseArtifact.put("alone", environment_artifacts=[
        _snapshot("mail", {"Mail/inbox.txt": b"m"}), _snapshot("mail-driver", {"Desktop/a.txt": b"a"})]))

    assert computer.added_at == ["http://127.0.0.1:20001", "http://127.0.0.1:20001/svc/mail"]
    assert (computer.home / "Desktop/a.txt").read_bytes() == b"a"
    assert [markers for _, markers, _ in computer.calls][1] == [{"path": "Mail/inbox.txt"}]


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
                 "RUN --network=none --mount=type=bind,source=packages/wheels/apps-mail,target=/tmp/agentenv-packages/"
                 "wheels/apps-mail python3 -m venv /opt/agentenv/apps/mail/venv && /opt/agentenv/apps/mail/venv/bin/pip"
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


def test_the_locks_debs_and_wheels_are_mounted_into_the_runs_that_install_them_never_copied_into_a_layer(
        base, driver, docker, tmp_path):
    """A COPY's files stay in its layer of the image though the RUN after it removes them."""
    _app(base, driver, tmp_path / "mail", install={"python": "requirements.txt", "packages": {"apt": ["hello"]},
                                                   "services": []})

    build = next(build for build in docker.builds if build["platform"] == "linux/arm64")
    lines = build["dockerfile"].splitlines()
    installs = [line for line in lines if "/tmp/agentenv-packages" in line]

    assert [line for line in lines if line.startswith("COPY packages/")] == []
    assert installs == [
        "RUN --network=none --mount=type=bind,source=packages/apt,target=/tmp/agentenv-packages/apt apt-get install -y "
        "--no-install-recommends /tmp/agentenv-packages/apt/*.deb",
        *(f"RUN --network=none --mount=type=bind,source=packages/wheels/{name},target=/tmp/agentenv-packages/wheels/"
          f"{name} python3 -m venv {prefix}/venv && {prefix}/venv/bin/pip install --no-index --no-deps "
          f"/tmp/agentenv-packages/wheels/{name}/*.whl && {prefix}/venv/bin/pip list --format=json > {prefix}/installed.json"
          for name, prefix in (("driver", "/opt/agentenv/drivers/desk"), ("apps-mail", "/opt/agentenv/apps/mail")))]
    assert {"packages/apt/hello_1.0_arm64.deb", "packages/wheels/driver/mss-10.1.0-py3-none-any.whl"} <= set(
        build["files"])


def test_an_app_that_declares_for_a_distribution_is_told_the_platform_it_was_composed_for(base, driver, docker,
                                                                                         tmp_path):
    _app(base, driver, tmp_path / "mail", steps=[("index_mail", "session", "linux/ubuntu")])

    build = {build["platform"]: build for build in docker.builds}["linux/arm64"]
    assert ("RUN --network=none cd /opt/agentenv/apps/mail && AGENTENV_PLATFORM=linux/ubuntu venv/bin/python -m "
            "agentenv_protocol.computer.run_install_step app.py:Mail index_mail") in build["dockerfile"]
    assert 'AGENTENV_PLATFORM="linux/ubuntu"' in build["confs"]["agentenv-app-mail.conf"]


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
    first = lines.index("USER root") + 1
    assert lines[first:first + 2] == [
        "RUN --network=none --mount=type=bind,source=packages/apt,target=/tmp/agentenv-packages/apt apt-get install -y "
        "--no-install-recommends /tmp/agentenv-packages/apt/*.deb", "COPY driver/ /opt/agentenv/drivers/desk/"]
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


def test_a_command_the_resolver_runs_that_fails_ends_it_with_the_commands_code_and_no_traceback(monkeypatch, capsys):
    """The composer reports the output's last line as the reason: it must be the command's error, apt's ``E:`` line,
    not a traceback naming the command."""
    def fail(*_args):
        raise subprocess.CalledProcessError(100, ["apt-get", "install"])

    monkeypatch.setattr(resolve_in_base, "main", fail)

    with pytest.raises(SystemExit) as exited:
        resolve_in_base.cli(["request.json", "out"])

    assert exited.value.code == 100
    assert capsys.readouterr().err == ""


def test_a_captured_command_that_fails_has_its_error_printed_last(monkeypatch, capsys):
    def fail(*_args):
        raise subprocess.CalledProcessError(2, ["dpkg-deb", "-f"], stderr="dpkg-deb: error: not a Debian archive\n")

    monkeypatch.setattr(resolve_in_base, "main", fail)

    with pytest.raises(SystemExit) as exited:
        resolve_in_base.cli(["request.json", "out"])

    assert exited.value.code == 2
    assert capsys.readouterr().err.strip().splitlines()[-1] == "dpkg-deb: error: not a Debian archive"


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
async def test_an_app_deployed_alone_is_a_computer_named_after_it_as_its_driver_and_handed_its_settings_a_secret_resolved(
        base, driver, docker, boots, tmp_path, monkeypatch):
    monkeypatch.setenv("MAIL_PASSWORD", "s3cret")
    monkeypatch.setenv("MAIL_TOKEN", "t0ken")
    boots.display = (1280, 800)
    app = _app(base, driver, tmp_path / "mail")
    settings = {"mail": {"signature": "-- Ada", "password": "env:MAIL_PASSWORD", "token": "env:MAIL_TOKEN"}}

    deployed = await Env.get(app.id).deploy(env_params={"app_settings": settings})

    environment = boots.boots[0]["environment"]
    assert environment["ENVIRONMENT_NAME"] == "mail-driver" and deployed.environment_card["name"] == "mail-driver"
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


@pytest.mark.asyncio
async def test_a_deploy_starts_from_its_app_envs_and_computer_envs_defaults_and_overrides_them_field_by_field(
        base, driver, docker, boots, tmp_path, monkeypatch):
    monkeypatch.setenv("MAIL_TOKEN", "t0ken")
    boots.display = (1280, 800)
    mail = _app(base, driver, tmp_path / "mail", defaults={"signature": "-- Ada", "theme": "dark"})
    env = _put(base, driver, apps=[mail], app_settings={"mail": {"theme": "light", "token": "env:MAIL_TOKEN"}})
    env_params = {"app_settings": {"mail": {"signature": "-- Bo"}}}

    deployed = await Env.get(env.id).deploy(env_params=env_params)

    assert json.loads(boots.boots[0]["environment"]["AGENTENV_APP_SETTINGS"]) == {"mail": {
        "signature": "-- Bo", "theme": "light", "token": "t0ken"}}
    assert deployed.metadata["computer"]["app_settings"] == {"mail": {
        "signature": "-- Bo", "theme": "light", "token": "<withheld>"}}
    assert deployed.metadata["computer"]["env_params"] == env_params


def test_default_settings_are_stored_with_each_env_and_left_out_when_there_are_none(base, driver, docker, tmp_path):
    mail = _app(base, driver, tmp_path / "mail", compose=False, defaults={"theme": "dark"})
    plain = _app(base, driver, tmp_path / "notes", compose=False, name="notes")
    env = _put(base, driver, compose=False, apps=[mail, plain], app_settings={"mail": {"signature": "-- Ada"}})

    stored = Env.get(env.id)
    assert (Env.get(mail.id).settings, stored.app_settings) == ({"theme": "dark"}, {"mail": {"signature": "-- Ada"}})
    assert stored.to_dict()["apps"] == [{"id": mail.id, "version": mail.version, "settings": {"signature": "-- Ada"}},
                                        {"id": plain.id, "version": plain.version}]
    assert "settings" not in plain.to_dict()
    assert stored.app_defaults() == {"mail": {"theme": "dark", "signature": "-- Ada"}, "notes": {}}


@pytest.mark.parametrize("values, problem", [
    ({"colour": "blue"}, "settings.colour: @local/~/kit/mail declares no setting 'colour'"),
    ({"theme": "blue"}, "settings.theme: 'blue' is not one of ['light', 'dark']"),
    ({"password": "hunter2"}, "settings.password is a secret, so it takes a secret:KEY"),
    ({"edition": "pro"}, "settings.edition is one of @local/~/kit/mail's build settings"),
], ids=["unknown", "enum", "secret-value", "build-setting"])
def test_default_settings_an_app_refuses_are_refused_at_put_in_its_own_env_and_in_a_computer_env(
        base, driver, docker, tmp_path, values, problem):
    with pytest.raises(ValueError, match=re.escape(problem)):
        _app(base, driver, tmp_path / "mail", compose=False, defaults=values)
    mail = _app(base, driver, tmp_path / "mail", compose=False)
    with pytest.raises(ValueError, match=re.escape(f"apps.mail.{problem}")):
        _put(base, driver, compose=False, apps=[mail], app_settings={"mail": values})
    with pytest.raises(ValueError, match=re.escape("apps.notes.settings: @local/~/kit/desktop has no app 'notes'")):
        _put(base, driver, compose=False, apps=[mail], app_settings={"notes": {}})


def test_a_required_setting_may_be_left_to_each_deploy_and_one_an_env_gives_satisfies_it(base, driver, docker,
                                                                                        tmp_path):
    settings = {**MAIL_SETTINGS, "required": ["signature"]}
    left = _app(base, driver, tmp_path / "mail", settings=settings)
    given = _app(base, driver, tmp_path / "signed", settings=settings, name="signed", defaults={"signature": "-- Ada"})

    (missing,) = DeployEnvTaskStep(id="d", version=None, env_id=left.id).preflight()
    assert "env_params.app_settings.mail: 'signature' is a required property" in missing
    assert DeployEnvTaskStep(id="d", version=None, env_id=given.id).preflight() == []
