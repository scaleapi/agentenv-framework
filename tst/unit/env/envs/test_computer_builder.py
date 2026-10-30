"""Where and how a computer env's pairs build: this machine's Docker, natively or emulated, or a VM of the put's
sandbox provider, each arch on the first link of a chain that runs it natively; what builds when emulation is off
or no link can build; a VM the provider loses; the VM's shell helper; and what a put says as it goes. Docker,
the registry and the VMs are faked."""

import hashlib
import io
import json
import logging
import re
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.cli import cli
from agent_env.env import Env
from agent_env.env.envs.computer.utils import builder, composer, layers_in_image, packages
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from tst.unit.env.envs._computer import _base, _put, FakeVm, FakeVmProvider, saved_image


def test_a_stage_that_goes_on_is_said_to_go_on_and_a_failed_pair_says_why(base, driver, docker, monkeypatch):
    docker.fail["linux/amd64"] = "#7 ERROR: pip install failed"
    build = docker.build_image
    monkeypatch.setattr(composer, "_HEARTBEAT_S", 0.1)
    monkeypatch.setattr(builder, "build_image", lambda *args, **kwargs: (time.sleep(0.5), build(*args, **kwargs))[1])
    told = []

    _put(base, driver, composing=Composing(on_progress=told.append))

    assert [line for line in told if line.startswith("container/arm64: still building the image (")]
    assert any(re.fullmatch(r"container/amd64: failed at build: the build failed: #7 ERROR: pip install failed \(\d+ s\)",
                            line) for line in told)


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

    assert amd64["reason"].startswith("the builder can't run linux/amd64 programs: its Docker has no emulation for it:")
    assert amd64["log"]


def test_a_base_built_from_its_context_is_built_here_for_each_arch_and_keyed_by_its_sources(driver, docker,
                                                                                            monkeypatch):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id="base-image", description="", image_name="local/base-1234:abcd", build_context_object_url="file:///c.tar.gz",
        dockerfile_path="Dockerfile", source_digest="sha256:" + "c" * 64))
    built_for = []
    monkeypatch.setattr(builder, "build_for", lambda image, platform: built_for.append(platform) or
                        f"{image.image_name}-{platform.rsplit('/', 1)[1]}")

    env = Env.get(_put(_base(image), driver).id)
    again = Env.get(_put(_base(image), driver, id="@local/~/kit/again", environment_name="again").id)

    assert sorted(built_for) == ["linux/amd64", "linux/arm64"]
    assert {r["reused"] for r in again.metadata["composed"]["container"].values()} == {True}
    assert {build["dockerfile"].splitlines()[0] for build in docker.builds} == {
        "FROM local/base-1234:abcd-amd64", "FROM local/base-1234:abcd-arm64"}
    assert {arch: r["status"] for arch, r in env.metadata["composed"]["container"].items()} == {
        "amd64": "unverified", "arm64": "ready"}


def test_the_platforms_this_machine_builds_are_read_from_its_builder(monkeypatch):
    inspected = ("Name:          desktop-linux\nDriver:        docker\nNodes:\nName:             desktop-linux\n"
                 "Platforms:        linux/arm64, linux/amd64, linux/amd64/v2, linux/riscv64*\nLabels:\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=inspected, returncode=0))

    assert builder._emulated_platforms() == {"linux/arm64", "linux/amd64", "linux/amd64/v2", "linux/riscv64"}


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

    assert builder.LocalBuilder.installed(None, "agentenv-computer:abc-arm64", "reference") == {
        "mss": "10.2.0", "agentenv-framework-protocol": "0.1.382"}
    assert calls == [["docker", "create"], ["docker", "cp"], ["docker", "rm"]]


def test_what_each_run_wrote_is_read_from_the_next_layer_a_shell_ran_its_command_for_in_a_saved_images_history(
        tmp_path):
    saved_image(tmp_path / "image.tar", [
        (None, {"etc/os-release": b"ubuntu"}),
        ("USER root", None),
        ("RUN /bin/sh -c cd /opt/a && ./setup.sh # buildkit", {
            "usr/local/bin/tool": b"a", "usr/bin/tool": "/usr/local/bin/tool", "etc/.wh.gone": b"", "./etc/a": b"a"}),
        ("RUN /bin/sh -c cd /opt/a && ./setup.sh.extra # buildkit", {"etc/extra": b"extra"}),
        ("COPY apps/b/ /opt/b/ # buildkit", {"opt/b/x": b"x"}),
        ("RUN /bin/sh -c cd /opt/b && ./setup.sh # buildkit", {"usr/local/bin/tool": b"b"}),
        ("RUN /bin/sh -c cd /opt/b && ./setup.sh # buildkit", {"etc/again": b"again"})])
    commands = ["cd /opt/a && ./setup.sh", "cd /opt/a && ./setup.sh.extra", "cd /opt/b && ./setup.sh",
                "cd /opt/b && ./setup.sh"]

    ran = subprocess.run([sys.executable, "-I", str(builder.LAYERS), str(tmp_path / "image.tar"), json.dumps(commands)],
                         capture_output=True, text=True)

    sha = lambda data: hashlib.sha256(data).hexdigest()
    assert json.loads(ran.stdout) == layers_in_image.written(str(tmp_path / "image.tar"), commands) == [
        {"/usr/local/bin/tool": sha(b"a"), "/usr/bin/tool": "-> /usr/local/bin/tool", "/etc/a": sha(b"a")},
        {"/etc/extra": sha(b"extra")}, {"/usr/local/bin/tool": sha(b"b")}, {"/etc/again": sha(b"again")}]
    for missing in (["cd /opt/c && ./setup.sh"], ["cd /opt/b && ./setup.sh", "cd /opt/a && ./setup.sh"],
                    [*commands, "cd /opt/b && ./setup.sh"]):
        with pytest.raises(ValueError, match="no layer of the image after the last one found was made by"):
            layers_in_image.written(str(tmp_path / "image.tar"), missing)


def test_a_registry_base_is_signed_in_to_before_its_image_is_built_on(base, driver, docker, monkeypatch):
    signed = []
    monkeypatch.setattr(builder, "docker_login", signed.append)

    _put(base, driver)

    assert len(signed) == 2 and len(docker.builds) == 2


def test_the_builder_vms_helper_lists_plain_files_marks_runnables_links_checks_and_downloads(tmp_path):
    folder = tmp_path / "out"
    (folder / "sub dir").mkdir(parents=True)
    (folder / "sub dir" / "run me").write_bytes(b"#!/bin/sh\n")
    (folder / "plain").write_bytes(b"data")
    (folder / "elf").write_bytes(b"\x7fELF\x02")
    (folder / "link").symlink_to(tmp_path / "elsewhere")
    (tmp_path / "list").write_text(f"{hashlib.sha256(b'data').hexdigest()}  plain\n{'0' * 64}  elf\n")

    def helper(*args):
        return subprocess.run(["bash", str(builder.IN_VM), *map(str, args)], capture_output=True, text=True)

    listed = helper("files", folder).stdout
    helper("runnable", folder / "sub dir" / "run me", folder / "plain", folder / "elf")
    helper("link", folder / "plain", tmp_path / "linked" / "plain")
    checked = helper("check", folder, tmp_path / "list")
    downloaded = helper("download", (folder / "plain").as_uri(), tmp_path / "fetched" / "plain")

    assert sorted(filter(None, listed.split("\0"))) == sorted(
        f"{hashlib.sha256(data).hexdigest()} {len(data)} {name}"
        for name, data in (("sub dir/run me", b"#!/bin/sh\n"), ("plain", b"data"), ("elf", b"\x7fELF\x02")))
    assert {name: (folder / name).stat().st_mode & 0o111 for name in ("sub dir/run me", "plain", "elf")} == {
        "sub dir/run me": 0o111, "plain": 0, "elf": 0o111}
    assert (tmp_path / "linked" / "plain").stat().st_ino == (folder / "plain").stat().st_ino
    assert checked.returncode == 1 and checked.stdout.startswith("elf: its sha256 is ")
    assert downloaded.stdout.strip() == hashlib.sha256(b"data").hexdigest()


def test_a_put_on_a_vm_provider_builds_each_arch_in_a_vm_of_its_own_and_its_images_are_reused_by_key_here(
        base, driver, docker, signing, tmp_path, monkeypatch):
    vms = FakeVmProvider(docker, tmp_path / "vms")
    for name in ("build_image", "push_local_image", "docker_login", "_emulated_platforms"):
        monkeypatch.setattr(builder, name, lambda *args, **kwargs: pytest.fail("this machine's Docker was used"))

    told = []
    composed = Env.get(_put(base, driver, composing=Composing(vms, on_progress=told.append)).id).metadata["composed"][
        "container"]

    assert ("container/arm64: building emulated, which is slow, in a FakeVmProvider VM; add local to --sandbox to build "
            "it natively with this machine's Docker, or pass --no-emulate to skip it") in told
    assert {arch: (r["status"], r["builder"], r["emulated"]) for arch, r in composed.items()} == {
        "amd64": ("ready", "FakeVmProvider", False), "arm64": ("unverified", "FakeVmProvider", True)}
    assert composed["arm64"]["reason"] == "no link of the sandbox provider runs linux/arm64 natively to probe it"
    assert len(vms.vms) == 2 and all(vm.terminated for vm in vms.vms) and vms.binfmt == ["arm64"]
    assert sorted(vms.sizes, key=lambda size: size["timeout"]) == [
        {"exposed_ports": [], "timeout": 7200, **builder._VM}, {"exposed_ports": [], "timeout": 21600, **builder._VM}]
    driver_files = {artifact.object_url for artifact in driver.files.get_file_artifacts().values()}
    assert sorted(vms.fetched) == sorted([*driver_files, *driver_files])
    assert vms.uploaded and all(f"{packages.PREFIX}/files/" in url for url in vms.uploaded)
    builds = {build["platform"]: build for build in docker.builds}
    assert builds["linux/amd64"]["files"] == ["Dockerfile", "agentenv-desk.conf", "driver/driver.py",
                                              "driver/requirements.txt", "packages/wheels/driver/mss-10.1.0-py3-none-any.whl"]
    assert composed["amd64"]["installed"] == {"mss": "10.1.0"}
    assert docker.labels[f"agentenv-computer:{composed['arm64']['key'][7:23]}-arm64"]["dev.agentenv.emulated"] == "true"

    again = Env.get(_put(base, driver, id="@local/~/kit/other", environment_name="other").id)

    assert {arch: (r["reused"], r["key"]) for arch, r in again.metadata["composed"]["container"].items()} == {
        arch: (True, r["key"]) for arch, r in composed.items()}


def test_a_chain_builds_each_arch_on_the_first_link_that_runs_it_natively(base, driver, docker, signing, tmp_path):
    vms = FakeVmProvider(docker, tmp_path / "vms")

    composed = Env.get(_put(base, driver, composing=Composing(ChainedSandboxProvider(
        [vms, LocalSandboxProvider()]))).id).metadata["composed"]["container"]

    assert {arch: (r["builder"], r["emulated"]) for arch, r in composed.items()} == {
        "amd64": ("FakeVmProvider", False), "arm64": ("LocalSandboxProvider", False)}
    assert len(vms.vms) == 1 and vms.binfmt == []


def test_a_builder_vm_its_provider_loses_fails_its_pair_alone_and_is_torn_down(base, driver, docker, signing, tmp_path,
                                                                              monkeypatch):
    vms, real = FakeVmProvider(docker, tmp_path / "vms"), FakeVm._docker

    class Gone(Exception):
        pass

    def gone(vm, words):
        if words[0] == "timeout" and "linux/arm64" in words:
            raise Gone("the sandbox is shutting down")
        return real(vm, words)
    monkeypatch.setattr(FakeVm, "_docker", gone)

    composed = Env.get(_put(base, driver, composing=Composing(vms)).id).metadata["composed"]["container"]

    assert (composed["arm64"]["status"], composed["arm64"]["stage"], composed["arm64"]["reason"]) == (
        "failed", "resolve", "RuntimeError: FakeVmProvider's VM failed: Gone: the sandbox is shutting down")
    assert composed["amd64"]["status"] == "ready" and all(vm.terminated for vm in vms.vms)


def test_with_emulation_off_an_arch_a_refused_link_runs_natively_fails_and_only_one_needing_emulation_is_skipped(
        base, driver, docker, tmp_path):
    vms = FakeVmProvider(docker, tmp_path / "vms")

    composed = Env.get(_put(base, driver, composing=Composing(vms, emulate=False)).id).metadata["composed"]["container"]

    assert (composed["amd64"]["status"], composed["amd64"]["reason"]) == (
        "failed", "FakeVmProvider's VMs can't fetch from the object store, which signs no URLs; build on the local "
                  "provider")
    assert (composed["arm64"]["status"], composed["arm64"]["reason"]) == (
        "skipped", "emulation is off, and FakeVmProvider doesn't run linux/arm64 natively")
    assert vms.vms == []


class Containers(SandboxProvider):
    async def create_sandbox(self, **kwargs):
        raise AssertionError("a container was created")


@pytest.mark.parametrize("links, reason", [
    (["vms"], "FakeVmProvider's VMs can't fetch from the object store, which signs no URLs; build on the local provider"),
    (["containers"], "Containers gives no VM to build in"),
    (["containers", "local"], "Containers gives no VM to build in; this machine can't build linux/amd64: its Docker has "
                              "no emulation for it"),
])
def test_an_arch_no_link_can_build_fails_saying_why_for_each_link(base, driver, docker, tmp_path, links, reason):
    docker.emulates = set()
    made = {"vms": FakeVmProvider(docker, tmp_path / "vms"), "containers": Containers(),
            "local": LocalSandboxProvider()}

    composed = Env.get(_put(base, driver, composing=Composing(ChainedSandboxProvider(
        [made[link] for link in links]))).id).metadata["composed"]["container"]

    assert (composed["amd64"]["status"], composed["amd64"]["stage"], composed["amd64"]["reason"]) == (
        "failed", "resolve", reason)
    assert made["vms"].vms == []


def test_a_put_that_wont_emulate_skips_an_arch_no_link_runs_natively_and_a_later_build_that_will_builds_it(
        base, driver, docker, caplog):
    caplog.set_level(logging.ERROR, logger=composer.__name__)
    env = _put(base, driver, composing=Composing(emulate=False))
    skipped = CliRunner().invoke(cli, ["env", "build", "--id", env.id, "--no-emulate"])
    built = CliRunner().invoke(cli, ["env", "build", "--id", env.id])

    reason = "emulation is off, and LocalSandboxProvider doesn't run linux/amd64 natively"
    assert skipped.exit_code == 0 and f"container/amd64: skipped ({reason})" in skipped.output
    assert [build["platform"] for build in docker.builds] == ["linux/arm64", "linux/amd64"]
    assert built.exit_code == 0 and "container/amd64: building emulated, which is slow, with this machine's Docker; " \
                                    "pass --no-emulate to skip it" in built.output
    composed = Env.get(env.id).metadata["composed"]["container"]
    assert {arch: r["status"] for arch, r in composed.items()} == {"amd64": "unverified", "arm64": "ready"}
