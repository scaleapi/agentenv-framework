"""The Linux disk form's recipe as the init reads what it writes: the image's start (image-env, run by the disk tools) and
the deploy's environment (the launcher's NoCloud seed, agentenv_protocol.computer.seed's), each a line per value
base64'd so any bytes survive, the init stopping the boot when the seed carries no version it reads; the launcher, with QEMU and its tools stood in for, serving each port through socat in front of QEMU's forward and booting
its disk afresh in place on a reset; and the local provider booting a disk in the launcher, with KVM and the VM's ports,
settings and size, and exporting one, compressed, once."""

import asyncio
import base64
import json
import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentenv_protocol.computer import seed as seed_protocol

from agent_env.env.published_images import DISK
from agent_env.providers.sandbox_providers.disk_launcher import CONTROL, LAUNCH, RESET, UNRESETTABLE, reset_input
from agent_env.providers.sandbox_providers.local_docker import disk as local_disk
from agent_env.providers.sandbox_providers.local_docker import provider as local_sandbox
from agent_env.providers.sandbox_providers.local_docker.cli import DockerFailed
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import SANDBOX_LABEL

TRICKY = "a value with spaces, = signs, 'quotes', $dollars and\na second line"


def _decoded(text: str) -> list[tuple[str, str]]:
    """Each line's key and value, as the disk's init decodes them."""
    return [(key, base64.b64decode(value).decode()) for line in text.splitlines() if line and not line.startswith("#")
            for key, _, value in [line.partition("=")]]


def test_image_env_writes_an_images_start_a_line_a_value_root_and_slash_when_it_names_none(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"Env": ["PATH=/usr/bin", f"NOTE={TRICKY}"], "Entrypoint": ["/usr/bin/tini", "--"],
                                  "Cmd": ["start-desktop", "init"], "User": "user", "WorkingDir": "/home/user",
                                  "Labels": {"ignored": "yes"}}))
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"Cmd": ["sh"]}))

    start, plain = (subprocess.run(["python3", str(DISK / "tools" / "image-env"), str(path)], capture_output=True,
                                   text=True, check=True).stdout for path in (config, bare))

    assert _decoded(start) == [("env", "PATH=/usr/bin"), ("env", f"NOTE={TRICKY}"), ("argv", "/usr/bin/tini"),
                               ("argv", "--"), ("argv", "start-desktop"), ("argv", "init"), ("user", "user"),
                               ("workdir", "/home/user")]
    assert _decoded(plain) == [("argv", "sh"), ("user", "root"), ("workdir", "/")]


def test_the_seed_holds_each_variable_the_launcher_is_handed_by_its_name_whatever_its_value_holds(tmp_path):
    """A value of several lines, a certificate's say, reaches it whole, and one holding a line that looks like another
    variable makes none."""
    certificate = "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n"
    forged = "first line\nVM_ENV_FAKE=1"

    _seed(tmp_path, {"AGENTENV_SETTING_RESOLUTION": "1280x800", "AGENTENV_CONTROL_TOKEN": TRICKY, "EMPTY": "",
                     "AGENTENV_SETTING_CA": certificate, "AGENTENV_SETTING_NOTE": forged})

    user_data, meta_data = (tmp_path / "user-data").read_text(), (tmp_path / "meta-data").read_text()
    assert user_data.splitlines()[0] == "#agentenv-environment"
    assert _decoded(user_data) == [("AGENTENV_CONTROL_TOKEN", TRICKY), ("AGENTENV_SETTING_CA", certificate),
                                   ("AGENTENV_SETTING_NOTE", forged), ("AGENTENV_SETTING_RESOLUTION", "1280x800"),
                                   ("EMPTY", "")]
    assert meta_data.endswith("\nlocal-hostname: computer\n") and meta_data.startswith("instance-id: ")
    assert (tmp_path / seed_protocol.OLDEST).read_text() == "1\n"


def test_the_launchers_seed_and_the_inits_reader_hold_to_the_protocols_sample_byte_for_byte(tmp_path):
    _seed(tmp_path / "written", seed_protocol.SAMPLE)
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample" / "user-data").write_text(seed_protocol.SAMPLE_USER_DATA)

    assert (tmp_path / "written" / seed_protocol.user_data_file(1)).read_text() == seed_protocol.SAMPLE_USER_DATA
    assert _seed_environment(tmp_path / "sample") == (0, seed_protocol.SAMPLE)


@pytest.mark.parametrize("files, read", [
    ({"user-data": seed_protocol.SAMPLE_USER_DATA}, (0, seed_protocol.SAMPLE)),
    ({"user-data": seed_protocol.SAMPLE_USER_DATA, "user-data.v2": "#agentenv-environment\nV2=MQ==\n",
      seed_protocol.OLDEST: "1\n"}, (0, seed_protocol.SAMPLE)),
    ({"user-data.v2": "#agentenv-environment\nV2=MQ==\n", seed_protocol.OLDEST: "2\n"},
     (1, "this disk reads seed v1; this agent-env writes v2 or later; recompose it with agent-env env build")),
    ({}, (0, {})),
], ids=["an-older-launchers-seed", "a-newer-launchers-seed", "a-seed-past-this-init", "no-user-data"])
def test_the_init_reads_the_newest_seed_version_it_knows_and_refuses_a_seed_that_carries_none(tmp_path, files, read):
    """An older launcher's seed has no oldest-version file; a newer one's carries later versions beside version 1."""
    for name, text in files.items():
        (tmp_path / name).write_text(text)

    assert _seed_environment(tmp_path) == read


def test_the_inits_reader_keeps_its_versions_and_leaves_an_images_variables_of_the_same_names_alone(tmp_path):
    """init sources it, then exports the image's environment, whose variables may share a name with the reader's."""
    (tmp_path / "user-data").write_text(seed_protocol.SAMPLE_USER_DATA)
    image = {name: "the image's" for name in ("reads", "oldest", "newest", "file", "version", "name", "line", "value",
                                             "quoted", "rest", "decoded", "refused")}

    assert _seed_environment(tmp_path, image) == (0, seed_protocol.SAMPLE)


def test_the_inits_reader_exports_a_seed_variable_named_as_one_of_its_own_and_one_holding_quotes(tmp_path):
    environment = {**seed_protocol.SAMPLE, **{name: f"{name}'s value, 'quoted'\n" for name in (
        "reads", "oldest", "file", "name", "line", "value", "quoted", "rest", "decoded", "refused", "_seed_reads")}}
    (tmp_path / "user-data").write_text(seed_protocol.user_data(environment))

    assert _seed_environment(tmp_path) == (0, environment)


def test_the_inits_decoder_keeps_every_byte_of_a_start_and_a_seed_trailing_newlines_included(tmp_path):
    """As a container runtime would: $(...) alone drops a value's trailing newlines."""
    decoder = re.search(r"^decode\(\) .*$", (DISK / "boot" / "seed-env").read_text(), re.MULTILINE).group(0)
    values = ["plain", "ends with a newline\n", "\n", f"{TRICKY}\n\n"]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"Cmd": values, "User": "user\n"}))
    start = subprocess.run(["python3", str(DISK / "tools" / "image-env"), str(config)], capture_output=True, text=True,
                           check=True).stdout
    _seed(tmp_path, {f"V{i}": value for i, value in enumerate([*values, "a\nVM_ENV_FAKE=1\n"])})
    seed = (tmp_path / "user-data").read_text()

    def decoded(text: str) -> list[str]:
        script = f"{decoder}\n" + "".join(f"decode {line.partition('=')[2]}; printf '%s\\0' \"$decoded\"\n"
                                          for line in text.splitlines() if line and not line.startswith("#"))
        return subprocess.run(["sh", "-c", script], capture_output=True, text=True, check=True).stdout.split("\0")[:-1]

    assert decoded(start) == [*values, "user\n", "/"] and decoded(seed) == [*values, "a\nVM_ENV_FAKE=1\n"]


def test_the_launcher_gets_kvm_the_vms_ports_and_size_and_its_environment_under_its_prefix():
    args = local_disk.launcher_args([18765, 5900], {"ENVIRONMENT_NAME": "desk", "A": "b c"}, 1.5, 4096)

    assert args == ["--device", "/dev/kvm", "-e", "VM_PORTS=18765 5900", "-e", "VM_CPUS=2", "-e", "VM_MEMORY_MB=4096",
                    "-e", "VM_REQUIRE_KVM=1", "-e", "VM_ENV_ENVIRONMENT_NAME=ZGVzaw==", "-e", "VM_ENV_A=YiBj"]
    assert reset_input({"ENVIRONMENT_NAME": "desk", "A": "b c"}) == (
        "VM_ENV_ENVIRONMENT_NAME=ZGVzaw==\nVM_ENV_A=YiBj\n")


#: Stand-ins for what the launcher runs, each a bash script logging its arguments as a line to $CALLS. QEMU runs until
#: it's stopped, or for $QEMU_RUNS_S seconds, logging the seed it boots with; socat exits 7 after $SOCAT_EXITS_S; zstd
#: takes $ZSTD_TAKES_S.
_STAND_INS = {
    "dpkg": "echo amd64",
    "qemu-img": 'echo "qemu-img $*" >> "$CALLS"',
    "mkfs.vfat": ":",
    "mcopy": ":",
    "cp": 'touch "${@: -1}"',
    "zstd": 'echo "zstd $*" >> "$CALLS"; sleep "${ZSTD_TAKES_S:-0}"; touch "${@: -1}"',
    "socat": 'echo "socat $*" >> "$CALLS"; sleep "${SOCAT_EXITS_S:-30}"; exit 7',
    **{qemu: f'echo "{qemu} $* seed=$(grep -vc "^#" "$VM_STATE/boot/seed/user-data")'
             f' $(sed -n 2,9p "$VM_STATE/boot/seed/user-data" | tr "\\n" " ")" >> "$CALLS"\n'
             'trap \'echo "stopped $$" >> "$CALLS"; kill $!; exit 0\' TERM\n'
             'sleep "${QEMU_RUNS_S:-30}" & wait' for qemu in ("qemu-system-x86_64", "qemu-system-aarch64")},
}


@pytest.fixture
def launcher(tmp_path):
    """Run the launcher, or ``launch reset``, with the stand-ins first on its PATH: the process, its calls' log."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in _STAND_INS.items():
        (bin_dir / name).write_text(f"#!/bin/bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    calls = tmp_path / "calls"
    base = {"PATH": f"{bin_dir}:{DISK / 'launcher'}:{os.environ['PATH']}", "CALLS": str(calls),
            "VM_STATE": str(tmp_path / "state"), "VM_ENV_ENVIRONMENT_NAME": "ZGVzaw=="}
    started = []

    def launch(*args: str, input: str | None = None, **variables: str) -> subprocess.Popen:
        process = subprocess.Popen(["bash", str(DISK / "launcher" / "launch"), *args], env={**base, **variables},
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if input is not None:
            process.stdin.write(input)
        process.stdin.close()
        started.append(process)
        return process

    yield launch, calls
    for process in started:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def _logged(calls: Path, what: str, count: int = 1) -> list[str]:
    """The lines of ``calls`` starting ``what``, once there are ``count`` of them."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        lines = [line for line in calls.read_text().splitlines() if line.startswith(what)] if calls.exists() else []
        if len(lines) >= count:
            return lines
        time.sleep(0.05)
    raise AssertionError(f"{count} {what!r} lines never came: {calls.read_text() if calls.exists() else ''}")


def test_the_launcher_serves_each_port_through_socat_in_front_of_qemus_forward_on_loopback(launcher):
    """socat's listen backlog takes a burst of connections, which QEMU's own forward, with a backlog of 1, drops, and
    socat retries its connection to that forward rather than wait out the kernel's SYN backoff."""
    launch, calls = launcher

    running = launch(VM_PORTS="18765 6080")

    retry = "connect-timeout=0.5,retry=600,interval=0.05"
    assert sorted(_logged(calls, "socat", 2)) == [
        f"socat TCP-LISTEN:18765,fork,reuseaddr,backlog=1024 TCP:127.0.0.1:28765,{retry}",
        f"socat TCP-LISTEN:6080,fork,reuseaddr,backlog=1024 TCP:127.0.0.1:16080,{retry}"]
    (qemu,) = _logged(calls, "qemu-system-x86_64")
    assert "-netdev user,id=net,hostfwd=tcp:127.0.0.1:28765-:18765,hostfwd=tcp:127.0.0.1:16080-:6080 " in qemu
    assert "-machine q35,accel=tcg" in qemu or "-machine q35,accel=kvm" in qemu
    running.terminate()
    assert running.wait(10) == 143


@pytest.mark.parametrize("ports, why", [("55536", "above 65535"), ("6080 16080", "another of VM_PORTS")])
def test_the_launcher_refuses_a_port_whose_forward_behind_the_proxy_has_no_port_of_its_own(launcher, ports, why):
    launch, calls = launcher

    refused = launch(VM_PORTS=ports)

    assert refused.wait(10) == 2 and why in refused.stderr.read()
    assert not calls.exists() or "qemu" not in calls.read_text()


@pytest.mark.parametrize("variables", [{"SOCAT_EXITS_S": "0.5"},
                                       {"SOCAT_EXITS_S": "0", "ZSTD_TAKES_S": "0.5", "VM_DISK": "/disk/disk.img.zst"}],
                         ids=["while-it-runs", "before-it-boots"])
def test_the_launcher_stops_the_computer_when_a_proxy_exits(launcher, variables, tmp_path):
    """Whenever the proxy exits: one that can't bind its port exits as the disk is expanded, before QEMU starts."""
    launch, _ = launcher

    stopped = launch(VM_PORTS="18765", **variables)

    assert stopped.wait(10) == 1
    assert "the proxy in front of a forwarded port exited (7), so the computer is stopped" in stopped.stderr.read()
    assert _gone(int((tmp_path / "state" / "control" / "qemu.pid").read_text()))


def _gone(pid: int) -> bool:
    """Whether process ``pid`` ends within 10 s."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_the_launcher_fails_rather_than_emulate_where_kvm_is_required_and_cant_run_the_disk(launcher):
    launch, calls = launcher

    refused = launch(VM_ARCH="arm64", VM_REQUIRE_KVM="1")
    emulated = launch(VM_ARCH="arm64", QEMU_RUNS_S="0.1")

    assert refused.wait(10) == 3 and "VM_REQUIRE_KVM refuses to emulate it" in refused.stderr.read()
    assert emulated.wait(10) == 0 and "-machine virt,accel=tcg" in _logged(calls, "qemu-system-aarch64")[0]


def test_the_launcher_expands_a_compressed_disk_onto_its_state_folder_and_boots_that(launcher, tmp_path):
    launch, calls = launcher

    done = launch(VM_DISK="/disk/disk.img.zst", QEMU_RUNS_S="0.1")

    assert done.wait(10) == 0
    assert _logged(calls, "zstd") == [f"zstd -q -d -f /disk/disk.img.zst -o {tmp_path}/state/disk.img"]
    assert f"-b {tmp_path}/state/disk.img -F qcow2" in _logged(calls, "qemu-img")[0]


def test_a_reset_boots_the_computer_afresh_with_its_new_environment_while_its_ports_stay_up(launcher):
    launch, calls = launcher
    running = launch(VM_PORTS="18765", VM_ENV_A="YQ==")
    (first,) = _logged(calls, "qemu-system-x86_64")

    reset = launch("reset", input="VM_ENV_ENVIRONMENT_NAME=YWdhaW4=\nnot a variable\nPATH=/nowhere\n")

    assert reset.wait(10) == 0, reset.stderr.read()
    first, second = _logged(calls, "qemu-system-x86_64", 2)
    assert "seed=2 " in first and "ENVIRONMENT_NAME=ZGVzaw==" in first and "A=YQ==" in first
    assert "seed=1 " in second and "ENVIRONMENT_NAME=YWdhaW4=" in second and "A=YQ==" not in second
    assert len(_logged(calls, "stopped")) == 1 and len(_logged(calls, "socat")) == 1 and running.poll() is None
    running.terminate()
    assert running.wait(10) == 143 and len(_logged(calls, "stopped", 2)) == 2


def test_a_reset_with_no_computer_running_says_so(launcher):
    launch, _ = launcher

    reset = launch("reset", input="")

    assert reset.wait(10) == 1 and "no computer is running here" in reset.stderr.read()


def test_a_reset_isnt_run_by_a_launcher_older_than_reset_in_place():
    """Such a launcher has no control folder, and would take ``launch reset`` for a second boot of its computer."""
    assert not Path(CONTROL).exists()

    assert subprocess.run(RESET, capture_output=True).returncode == UNRESETTABLE


def test_the_launcher_images_entrypoint_is_the_launch_script_its_providers_run():
    lines = (DISK / "launcher" / "Dockerfile").read_text().splitlines()

    assert f'ENTRYPOINT ["{LAUNCH}"]' in lines and f"COPY launch seed {PurePosixPath(LAUNCH).parent}/" in lines


def _seed(folder, env: dict[str, str]) -> None:
    """The seed of ``env`` in ``folder``, as the launcher makes it from what the local provider hands it."""
    args = local_disk.launcher_args([18765], env, 2, 4096)
    handed = dict(arg.split("=", 1) for flag, arg in zip(args, args[1:]) if flag == "-e")
    subprocess.run(["sh", str(DISK / "launcher" / "seed"), str(folder)], check=True,
                   env={"PATH": os.environ["PATH"], "HOME": "/root", **handed})


def _seed_environment(folder: Path, exported: dict[str, str] | None = None) -> tuple[int, dict[str, str] | str]:
    """The init's reading of the seed mounted at ``folder``, ``exported`` set after it sources the reader, as the image's
    environment is: 0 and the variables it exported or changed, or 1 and why it stops."""
    image = "".join(f"export {name}={shlex.quote(value)}; " for name, value in (exported or {}).items())
    script = (f'. "{DISK / "boot" / "seed-env"}"; {image}env > "$1/before"; seed_environment "$1" || '
              '{ printf %s "$refused"; exit 1; }; env -0')
    done = subprocess.run(["sh", "-c", script, "seed-env", str(folder)], capture_output=True, text=True,
                          env={"PATH": os.environ["PATH"]})
    if done.returncode:
        return done.returncode, done.stdout
    before = set((folder / "before").read_text().splitlines())
    return 0, {name: value for name, _, value in (entry.partition("=") for entry in done.stdout.split("\0")[:-1])
               if f"{name}={value}" not in before and name != "_"}


@pytest.fixture
def launched(monkeypatch):
    """The ``docker`` commands the local provider runs for a disk, with the launcher image it builds faked."""
    ran = []
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset({"linux/arm64"}))
    monkeypatch.setattr(local_sandbox, "host_platform", lambda: "linux/arm64")
    monkeypatch.setattr(local_sandbox, "machine_image", AsyncMock(return_value="agentenv-vm:0123"))
    monkeypatch.setattr(local_sandbox, "_free_host_port", lambda: 20001)
    monkeypatch.setattr(local_sandbox, "_host_ips", lambda: ("127.0.0.1",))
    monkeypatch.setattr(local_sandbox, "docker", lambda *args, input=None, timeout=None: ran.append(
        args if input is None else (*args, input, timeout)) or "")
    monkeypatch.setattr(LocalSandbox, "pull_images", AsyncMock())
    monkeypatch.setattr(LocalSandbox, "exec_script", AsyncMock(side_effect=lambda script: ran.append(script) or ""))
    return ran


@pytest.mark.asyncio
async def test_a_disk_boots_in_the_launcher_with_kvm_its_ports_published_and_a_reset_boots_it_afresh(launched):
    provider = LocalSandboxProvider()

    sandbox = await provider.create_vm(image="localhost:5000/disk@sha256:" + "a" * 64, exposed_ports=[18765],
                                       env={"ENVIRONMENT_NAME": "desk"}, cpu=2, memory=4096)
    await provider.reset_vm(sandbox, image="localhost:5000/disk@sha256:" + "a" * 64, env={"ENVIRONMENT_NAME": "again"})
    rebuilt = await provider.get_sandbox(sandbox.sandbox_id)

    first, reset = launched
    assert first == ("run", "-d", "--name", sandbox.container_name, "--label", f"{SANDBOX_LABEL}={sandbox.sandbox_id}",
                     "-p", "127.0.0.1:20001:18765", *local_disk.launcher_args([18765], {"ENVIRONMENT_NAME": "desk"},
                                                                               2, 4096), "agentenv-vm:0123")
    assert reset == ("exec", "-i", sandbox.container_name, *RESET, "VM_ENV_ENVIRONMENT_NAME=YWdhaW4=\n",
                     local_sandbox._RESET_SECONDS)
    assert local_sandbox._RESET_SECONDS > 60
    LocalSandbox.pull_images.assert_awaited_once_with(["localhost:5000/disk@sha256:" + "a" * 64])
    assert sandbox.tunnel_urls == {18765: "http://127.0.0.1:20001"} and rebuilt.owns_container


@pytest.mark.asyncio
@pytest.mark.parametrize("returncode, refused", [
    (UNRESETTABLE, "the computer in container .* was booted by an older agent-env, whose launcher can't reset it in "
                   "place; redeploy it"),
    (1, "docker exec .* failed: launch reset: no computer is running here"),
], ids=["older-launcher", "other-failure"])
async def test_a_reset_in_a_container_an_older_launcher_runs_is_refused_naming_the_fix(launched, monkeypatch,
                                                                                          returncode, refused):
    """An agent-env before reset in place started it, and a run resumed after an upgrade resets it."""
    provider = LocalSandboxProvider()
    sandbox = await provider.create_vm(image="localhost:5000/disk@sha256:" + "a" * 64, exposed_ports=[18765],
                                       env={"ENVIRONMENT_NAME": "desk"}, cpu=2, memory=4096)

    def docker(*args, **options):
        raise DockerFailed(f"docker {' '.join(args)} failed: launch reset: no computer is running here", returncode)

    monkeypatch.setattr(local_sandbox, "docker", docker)
    with pytest.raises(RuntimeError, match=refused):
        await provider.reset_vm(sandbox, image="localhost:5000/disk@sha256:" + "a" * 64, env={})


@pytest.mark.asyncio
async def test_a_boot_cancelled_while_docker_starts_the_launcher_waits_for_it_then_tears_it_down(launched, monkeypatch):
    started, release, order = threading.Event(), threading.Event(), []

    def docker(*args, **options):
        started.set()
        release.wait(5)
        order.append("run")
        return ""

    monkeypatch.setattr(local_sandbox, "docker", docker)
    monkeypatch.setattr(LocalSandbox, "terminate", AsyncMock(side_effect=lambda: order.append("terminate")))
    booting = asyncio.ensure_future(LocalSandboxProvider().create_vm(image="disk", exposed_ports=[18765], env={}))
    await asyncio.to_thread(started.wait, 5)

    booting.cancel()
    await asyncio.sleep(0.1)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await booting
    assert order == ["run", "terminate"]


@pytest.mark.asyncio
async def test_a_disk_isnt_booted_where_docker_cant_pass_kvm(launched, monkeypatch):
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset())

    with pytest.raises(ValueError, match="boots a disk only where this machine's Docker can pass it /dev/kvm"):
        await LocalSandboxProvider().create_vm(image="disk", exposed_ports=[18765], env={})

    assert launched == [] and LocalSandboxProvider.disk_image_platforms() == frozenset()


def test_a_disk_is_exported_compressed_into_the_cache_once_for_each_ref(tmp_path, monkeypatch):
    """By this machine's Docker, signed in to the image store as a boot's pull is: the disk copied out of a container of
    its image, then compressed in the launcher."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(local_disk, "_launcher", lambda: "agentenv-qemu-launcher:src-1")
    ran, compressed = [], []
    store = SimpleNamespace(auth=lambda ref: f"credentials for {ref}")
    monkeypatch.setattr(local_disk, "get_config", lambda: SimpleNamespace(get_image_store=lambda: store))
    monkeypatch.setattr(local_disk, "docker_login", lambda auth: ran.append(("login", auth)))

    def docker(*args):
        ran.append(args)
        if args[0] == "cp":
            Path(args[2]).write_bytes(b"disk of " + args[1].encode())
        return "container1" if args[0] == "create" else ""

    def run(argv, stdin=None, stdout=None, **kwargs):
        if argv[:2] == ["docker", "run"]:
            compressed.append(argv)
            stdout.write(b"zstd:" + stdin.read())
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(local_disk, "docker", docker)
    monkeypatch.setattr(local_disk.subprocess, "run", run)
    first = "localhost:5000/agentenv-computers:" + "1" * 64 + "@sha256:" + "a" * 64

    exported = [local_disk.exported_disk(image) for image in (first, first, first.replace("1", "2"))]

    assert exported[0] == exported[1] != exported[2] and exported[0].parent == tmp_path / "cache" / "agentenv" / "disks"
    assert exported[0].read_bytes() == b"zstd:disk of container1:/disk/disk.img" and exported[0].name.endswith(".img.zst")
    assert [args[0] for args in ran] == ["login", "create", "cp", "login", "create", "cp"]
    assert ran[0] == ("login", f"credentials for {first}") and ran[1][1:] == (first, "disk")
    assert compressed[0] == ["docker", "run", "--rm", "-i", "--entrypoint", "zstd", "agentenv-qemu-launcher:src-1",
                             "-q", "-3", "-T0", "-c"] and len(compressed) == 2
    assert sorted(p.name for p in exported[0].parent.iterdir()) == sorted({exported[0].name, exported[2].name})
