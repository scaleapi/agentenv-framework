"""The Linux disk form's recipe as the init reads what it writes: the image's start (image-env, run by the disk tools) and
the deploy's environment (the launcher's NoCloud seed, agentenv_protocol.computer.seed's), each a line per value
base64'd so any bytes survive, the init stopping the boot when the seed carries no version it reads; the launcher, with QEMU and its tools stood in for, serving each port through socat in front of QEMU's forward and booting
its disk afresh in place on a reset; and the local provider booting a disk in the launcher, with KVM and the VM's ports,
settings and size, and exporting one, compressed, once."""

import asyncio
import base64
import hashlib
import importlib.util
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
from agent_env.env.envs.computer.utils import windows_disk
from agent_env.providers.sandbox_providers import disk_launcher
from agent_env.providers.sandbox_providers.local_docker.provider import (LOCAL_TRUST_DIR, LocalSandbox,
                                                                         LocalSandboxProvider, trust_env)
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
    "mkfs.vfat": 'echo "mkfs.vfat $*" >> "$CALLS"',
    "mcopy": 'echo "mcopy $*" >> "$CALLS"',
    "cp": 'echo "cp $*" >> "$CALLS"; touch "${@: -1}"',
    "swtpm": 'echo "swtpm $*" >> "$CALLS"',
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


def test_the_launcher_forwards_each_host_port_from_loopback_here_to_the_host_and_seeds_the_trust_files(launcher,
                                                                                                    tmp_path):
    """The guest finds host.docker.internal at 10.0.2.2, which QEMU's network gives this container's loopback, and
    finds a local transfer CA's trust files in its seed's trust folder."""
    launch, calls = launcher
    trust = tmp_path / "trust"
    trust.mkdir()

    running = launch(VM_PORTS="18765", VM_HOST_PORTS="41234", VM_TRUST=str(trust))

    assert "socat TCP-LISTEN:41234,bind=127.0.0.1,fork,reuseaddr TCP:host.docker.internal:41234" in _logged(
        calls, "socat", 2)
    assert f"cp -r {trust} {tmp_path}/state/boot/seed/trust" in _logged(calls, "cp -r")
    assert _logged(calls, "mcopy")[0].startswith(f"mcopy -s -i {tmp_path}/state/boot/seed.img {tmp_path}/state/boot/seed/")
    running.terminate()
    assert running.wait(10) == 143


def test_a_launcher_stopped_before_its_computer_starts_exits_as_one_stopped_after_does(launcher):
    launch, calls = launcher

    running = launch(VM_DISK="/disk/disk.img.zst", ZSTD_TAKES_S="1")
    _logged(calls, "zstd")
    running.terminate()

    assert running.wait(10) == 143


@pytest.mark.parametrize("host_ports", ["18765", "28765"], ids=["a-guest-ports-proxy", "its-forward"])
def test_the_launcher_refuses_a_host_port_a_guest_ports_proxy_or_forward_takes_before_it_starts_a_proxy(
        launcher, host_ports):
    launch, calls = launcher

    refused = launch(VM_PORTS="18765", VM_HOST_PORTS=host_ports)

    assert refused.wait(10) == 2
    assert f"host port {host_ports} can't be forwarded: a guest port's proxy or its forward here takes it" in (
        refused.stderr.read())
    assert not calls.exists() or "socat" not in calls.read_text()


def test_launch_forward_serves_a_host_port_to_the_running_computer_as_the_boot_does(launcher):
    launch, calls = launcher

    forwarding = launch("forward", "52345")

    assert _logged(calls, "socat") == [
        "socat TCP-LISTEN:52345,bind=127.0.0.1,fork,reuseaddr TCP:host.docker.internal:52345"]
    forwarding.terminate()


def test_a_disks_guests_point_tls_clients_at_the_trust_files_a_containers_variables_name(tmp_path):
    """Neither guest can import the framework, so each keeps its own copy of trust_env's variables and files."""
    spec = importlib.util.spec_from_file_location("agentenv_session", windows_disk.RECIPE / "agentenv-session.py")
    session = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(session)
    init = (DISK / "boot" / "init").read_text()
    expected = trust_env(LOCAL_TRUST_DIR)

    assert all(f"{name}={value}" in init for name, value in expected.items())
    assert all(f"/seed/trust/{PurePosixPath(value).name}" in init for value in expected.values())
    (tmp_path / "ca-bundle.pem").write_text("")
    assert {name: Path(value).name for name, value in session.trust(tmp_path).items()} == {
        name: PurePosixPath(value).name for name, value in expected.items()}


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


@pytest.mark.parametrize("base", [None, "/disk/base.img.zst"], ids=["whole", "thin"])
def test_the_launcher_expands_a_compressed_disk_onto_its_state_folder_and_boots_that(launcher, tmp_path, base):
    """A thin disk's base, carried beside it, is expanded beside it, as the base.img it's backed by."""
    launch, calls = launcher

    done = launch(VM_DISK="/disk/disk.img.zst", QEMU_RUNS_S="0.1", **({"VM_BASE": base} if base else {}))

    assert done.wait(10) == 0
    assert _logged(calls, "zstd") == [f"zstd -q -d -f /disk/disk.img.zst -o {tmp_path}/state/disk.img"] + (
        [f"zstd -q -d -f {base} -o {tmp_path}/state/base.img"] if base else [])
    assert f"-b {tmp_path}/state/disk.img -F qcow2" in _logged(calls, "qemu-img")[0]


def test_a_windows_disk_boots_from_secure_boot_on_ahci_and_e1000e_with_a_fresh_tpm_each_time(launcher, tmp_path):
    """Windows has no virtio drivers of its own and installs only with secure boot and a TPM 2.0."""
    launch, calls = launcher
    disk = tmp_path / "disk" / "disk.img"
    disk.parent.mkdir()
    (disk.parent / "guest").write_text("windows\n")
    running = launch(VM_DISK=str(disk), VM_PORTS="18765")
    (qemu,) = _logged(calls, "qemu-system-x86_64")

    reset = launch("reset", input="")

    assert reset.wait(10) == 0, reset.stderr.read()
    work = tmp_path / "state" / "boot"
    assert "-machine q35,smm=on,accel=" in qemu and "file=/usr/share/OVMF/OVMF_CODE_4M.secboot.fd " in qemu
    assert "-device tpm-tis,tpmdev=tpm " in qemu and "-device ide-hd,drive=disk,bus=ide.0,bootindex=1,rotation_rate=1 " in qemu
    assert ",id=disk,discard=unmap,detect-zeroes=unmap " in qemu
    assert "-device e1000e,netdev=net " in qemu and "-device VGA,xres=1920,yres=1080 " in qemu and "virtio" not in qemu
    assert "-device usb-storage,bus=usb.0,drive=seed,removable=on " in qemu
    assert _logged(calls, "swtpm", 2) == 2 * [f"swtpm socket --tpm2 --tpmstate dir={work}/tpm --ctrl"
                                              f" type=unixio,path={work}/tpm.sock --daemon --terminate"]
    assert f"cp /usr/share/OVMF/OVMF_VARS_4M.ms.fd {work}/vars.fd" in _logged(calls, "cp")
    running.terminate()
    assert running.wait(10) == 143


@pytest.mark.parametrize("resolution, display", [(None, "xres=1920,yres=1080"), ("MTI4MHg4MDA=", "xres=1280,yres=800")],
                         ids=["default", "the-computers-own"])
def test_a_linux_desktop_guest_gets_a_virtio_gpu_at_its_resolution_trim_its_size_and_its_credentials(
        launcher, tmp_path, resolution, display):
    """A Wayland desktop's compositor needs a GPU, at the mode its monitor rule asks for."""
    launch, calls = launcher
    disk = tmp_path / "disk" / "disk.img"
    disk.parent.mkdir()
    (disk.parent / "guest").write_text("linux-desktop\n")
    running = launch(VM_DISK=str(disk), VM_PORTS="18765", VM_CREDENTIALS="tmpfiles.extra=ZCAvcnVu ssh.k=YQ==",
                     **({"VM_ENV_AGENTENV_SETTING_RESOLUTION": resolution} if resolution else {}))
    (qemu,) = _logged(calls, "qemu-system-x86_64")

    assert "-machine q35,accel=" in qemu and "file=/usr/share/OVMF/OVMF_CODE_4M.fd " in qemu and "secboot" not in qemu
    assert f"-vga none -device virtio-vga,{display} " in qemu and "-smp 4 -m 8192 " in qemu
    assert ",format=qcow2,if=virtio,discard=unmap,detect-zeroes=unmap " in qemu
    assert ("-smbios type=11,value=io.systemd.credential.binary:tmpfiles.extra=ZCAvcnVu "
            "-smbios type=11,value=io.systemd.credential.binary:ssh.k=YQ== ") in qemu
    running.terminate()
    assert running.wait(10) == 143


@pytest.mark.parametrize("guest, arch, variables, why", [
    ("windows", "arm64", {}, "a windows guest boots on amd64 alone, not arm64"),
    ("plan9", "amd64", {}, "no guest plan9"),
    ("linux-desktop", "arm64", {}, "a linux-desktop guest boots on amd64 alone, not arm64"),
    ("linux", "amd64", {"VM_CREDENTIALS": "ssh.k=YQ=="}, "VM_CREDENTIALS are for a linux-desktop guest, not linux"),
    ("linux-desktop", "amd64", {"VM_ENV_AGENTENV_SETTING_RESOLUTION": "d2lkZQ=="},
     "the computer's resolution, wide, isn't WIDTHxHEIGHT")])
def test_the_launcher_refuses_a_guest_it_cant_boot(launcher, tmp_path, guest, arch, variables, why):
    launch, calls = launcher
    (tmp_path / "guest").write_text(guest)

    refused = launch(VM_DISK=str(tmp_path / "disk.img"), VM_ARCH=arch, **variables)

    assert refused.wait(10) == 2 and why in refused.stderr.read()
    assert not calls.exists() or "qemu" not in calls.read_text()


def test_a_builder_boots_on_a_kept_overlay_made_on_the_disk_or_empty_where_theres_none_and_may_restart(
        launcher, tmp_path):
    """A base's install and a composition both keep what the guest writes, through the restarts its setup makes."""
    launch, calls = launcher
    out, disk = tmp_path / "out", tmp_path / "disk" / "disk.img"
    out.mkdir()
    disk.parent.mkdir()
    disk.touch()
    (out / "guest").write_text("windows")

    empty = launch(VM_KEEP=str(out / "base.img"), VM_DISK=str(tmp_path / "none.img"), VM_DISK_GB="80",
                   QEMU_RUNS_S="0.1")
    assert empty.wait(10) == 0, empty.stderr.read()
    backed = launch(VM_KEEP=str(out / "layer.img"), VM_DISK=str(disk), QEMU_RUNS_S="0.1")
    assert backed.wait(10) == 0, backed.stderr.read()

    assert _logged(calls, "qemu-img", 2) == [f"qemu-img create -q -f qcow2 {out}/base.img 80G",
                                             f"qemu-img create -q -f qcow2 -b {disk} -F qcow2 {out}/layer.img"]
    first, second = _logged(calls, "qemu-system-x86_64", 2)
    assert f"file={out}/base.img,format=qcow2" in first and f"file={out}/layer.img,format=qcow2" in second
    assert "-no-reboot" not in first + second and "-device tpm-tis" in first


def test_a_windows_install_boots_its_iso_first_pressing_enter_and_finds_its_answers_on_a_removable_drive(
        launcher, tmp_path):
    launch, calls = launcher
    out, answers = tmp_path / "out", tmp_path / "answers"
    out.mkdir()
    answers.mkdir()
    (out / "guest").write_text("windows")
    (answers / "autounattend.xml").write_text("<unattend/>")

    running = launch(VM_KEEP=str(out / "base.img"), VM_DISK=str(tmp_path / "none.img"), VM_MEDIA="/in/win11.iso",
                     VM_FILES=str(answers), SOCAT_EXITS_S="0")

    (qemu,) = _logged(calls, "qemu-system-x86_64")
    assert ("-drive file=/in/win11.iso,format=raw,if=none,id=media,media=cdrom,readonly=on"
            " -device ide-cd,drive=media,bus=ide.2,bootindex=0 ") in qemu
    assert "-device ide-hd,drive=disk,bus=ide.0,bootindex=1,rotation_rate=1 " in qemu
    assert "-device usb-storage,bus=usb.0,drive=files,removable=on " in qemu
    assert _logged(calls, "socat - UNIX-CONNECT:")
    assert f"mcopy -s -i {tmp_path}/state/boot/files.img {answers}/autounattend.xml ::/" in _logged(calls, "mcopy -s", 2)
    running.terminate()
    assert running.wait(10) == 143
    pressed = calls.read_text().count("socat - UNIX-CONNECT:")
    time.sleep(1.5)
    assert calls.read_text().count("socat - UNIX-CONNECT:") == pressed, "Enter is still pressed after the launcher exits"


def test_install_media_is_for_a_windows_guest(launcher):
    launch, _ = launcher

    refused = launch(VM_MEDIA="/in/installer.iso")

    assert refused.wait(10) == 2 and "VM_MEDIA and VM_FILES are for a windows guest, not linux" in refused.stderr.read()


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
    monkeypatch.setattr(local_sandbox, "machine_for", AsyncMock(return_value=local_disk.Machine("agentenv-vm:0123")))
    monkeypatch.setattr(local_sandbox, "_free_host_port", lambda: 20001)
    monkeypatch.setattr(local_sandbox, "_host_ips", lambda: ("127.0.0.1",))
    monkeypatch.setattr(local_sandbox, "local_grants", lambda: None)
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
async def test_a_disk_gets_this_processs_grant_port_and_the_trust_and_a_reconnect_its_own_but_not_a_teardown(
        launched, monkeypatch):
    monkeypatch.setattr(local_sandbox, "local_grants", lambda: (41234, Path("/state/tls/trust/ca")))

    sandbox = await LocalSandboxProvider().create_vm(image="localhost:5000/disk@sha256:" + "a" * 64,
                                                     exposed_ports=[18765], env={}, cpu=2, memory=4096)

    monkeypatch.setattr(local_sandbox, "local_grants", lambda: (52345, Path("/state/tls/trust/ca")))
    await LocalSandboxProvider().get_sandbox(sandbox.sandbox_id)

    run, forward = launched
    hosts = [arg for entry in local_sandbox._EXTRA_HOSTS for arg in ("--add-host", entry)]
    assert run == ("run", "-d", "--name", sandbox.container_name, "--label", f"{SANDBOX_LABEL}={sandbox.sandbox_id}",
                   "-p", "127.0.0.1:20001:18765", *local_disk.launcher_args([18765], {}, 2, 4096), *hosts,
                   "--mount", "type=bind,src=/state/tls/trust/ca,dst=/agentenv-trust,readonly",
                   "-e", "VM_HOST_PORTS=41234", "-e", "VM_TRUST=/agentenv-trust", "agentenv-vm:0123")
    assert forward == ("exec", "-d", sandbox.container_name, *disk_launcher.forward(52345))
    assert disk_launcher.forward(52345)[-1] == (f"grep -q '^forward()' {LAUNCH} || exit 0; "
                                                f"exec {LAUNCH} forward 52345")

    launched.clear()
    monkeypatch.setattr(LocalSandbox, "terminate", AsyncMock())
    await LocalSandboxProvider().terminate_sandbox(sandbox.sandbox_id)

    assert launched == [] and LocalSandbox.terminate.await_count == 1


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


@pytest.mark.asyncio
async def test_a_disk_boots_with_its_image_mounted_into_the_launcher_where_docker_mounts_images(launched, monkeypatch):
    mounted = local_disk.Machine("launcher:1", ("--mount", "type=image,source=disk:1,target=/disk,image-subpath=disk"))
    monkeypatch.setattr(local_sandbox, "machine_for", AsyncMock(return_value=mounted))

    await LocalSandboxProvider().create_vm(image="disk:1", exposed_ports=[18765], env={}, cpu=2, memory=4096)

    (run,) = launched
    assert run[-1] == "launcher:1" and run[run.index("--mount"):run.index("--mount") + 2] == mounted.args


@pytest.mark.asyncio
@pytest.mark.parametrize("mounts", [True, False], ids=["mounted", "copied"])
async def test_a_disk_is_mounted_into_the_launcher_where_docker_mounts_images_else_copied_into_a_machine_image(
        monkeypatch, mounts):
    """No copy of a disk, nor of a Windows disk's base, is made for each launcher where Docker can mount it."""
    monkeypatch.setattr(local_disk, "mounts_images", lambda: mounts)
    monkeypatch.setattr(local_disk, "launcher", lambda: "launcher:1")
    monkeypatch.setattr(local_disk, "machine_image", AsyncMock(return_value="agentenv-vm:1"))

    machine = await local_disk.machine_for("disk:1")

    assert machine == (local_disk.Machine("launcher:1", (
        "--mount", "type=image,source=disk:1,target=/disk,image-subpath=disk")) if mounts else local_disk.Machine(
        "agentenv-vm:1"))
    assert local_disk.machine_image.await_count == (0 if mounts else 1)


@pytest.mark.parametrize("info, mounts", [
    ('29.9.3 29.9.3 [["driver-type","io.containerd.snapshotter.v1"]]', True),
    ('28.0.0 28.0.0 [["driver-type","io.containerd.snapshotter.v1"]]', True),
    ('27.5.1 27.5.1 [["driver-type","io.containerd.snapshotter.v1"]]', False),
    ('27.5.1 29.9.3 [["driver-type","io.containerd.snapshotter.v1"]]', False),
    ('29.9.3 29.9.3 [["Backing Filesystem","extfs"],["Supports d_type","true"]]', False),
    ("4.9.3 4.9.3 null", False),
    (DockerFailed("docker info failed: template: :1: can't evaluate field ClientInfo", 1), False),
    (subprocess.TimeoutExpired(["docker", "info"], 30), False)],
    ids=["29-containerd", "28-containerd", "27", "27-client-on-a-29-server", "29-overlay2", "podman",
         "a-client-too-old-to-say", "no-answer"])
def test_docker_mounts_images_from_28_on_both_ends_on_the_containerd_image_store(monkeypatch, info, mounts):
    """A Docker that fails to say, or doesn't in time, gets the copy."""
    local_disk._docker_mounts_images.cache_clear()
    asked = []

    def docker(*args, timeout=None):
        asked.append(timeout)
        if isinstance(info, Exception):
            raise info
        return info

    monkeypatch.setattr(local_disk, "docker", docker)

    assert local_disk.mounts_images() is mounts and asked == [local_disk._DOCKER_INFO_SECONDS]
    local_disk._docker_mounts_images.cache_clear()


def test_a_docker_that_failed_to_say_is_asked_again_and_an_answer_is_kept(monkeypatch):
    """A daemon busy for a moment doesn't leave a long-lived process copying disks for good."""
    local_disk._docker_mounts_images.cache_clear()
    answers = [subprocess.TimeoutExpired(["docker", "info"], 30),
               '29.9.3 29.9.3 [["driver-type","io.containerd.snapshotter.v1"]]']
    asked = []

    def docker(*args, timeout=None):
        asked.append(args)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(local_disk, "docker", docker)

    assert [local_disk.mounts_images() for _ in range(3)] == [False, True, True] and len(asked) == 2
    local_disk._docker_mounts_images.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("layers, dockerfile", [
    ("1", "FROM launcher:1\nCOPY --from=disk:1 /disk/ /disk/\n"),
    ("2", "FROM launcher:1\nCOPY --from=disk:1 /disk/base.img /disk/base.img\n"
          "COPY --from=disk:1 /disk/disk.img /disk/guest /disk/\n")], ids=["one-layer", "thin-on-a-base"])
async def test_a_machine_copies_a_thin_disks_base_in_a_step_of_its_own_so_machines_on_one_base_share_it(
        monkeypatch, layers, dockerfile):
    built = []
    monkeypatch.setattr(local_disk, "launcher", lambda: "launcher:1")
    monkeypatch.setattr(local_disk, "present", lambda tag: False)
    monkeypatch.setattr(local_disk, "docker", lambda *args: f"sha256:1 {layers} sha256:first")
    monkeypatch.setattr(local_disk, "_build", lambda text, tag: built.append(text))

    await local_disk.machine_image("disk:1")

    assert built == [dockerfile]


@pytest.mark.asyncio
async def test_a_disk_built_again_under_its_tag_gets_a_machine_of_its_own(monkeypatch):
    """As a Windows base installed afresh is, under the tag the one it replaces had."""
    identity = "sha256:1"
    monkeypatch.setattr(local_disk, "launcher", lambda: "launcher:1")
    monkeypatch.setattr(local_disk, "present", lambda tag: True)
    monkeypatch.setattr(local_disk, "docker", lambda *args: f"{identity} 2 sha256:first")

    first = await local_disk.machine_image("disk:1")
    again = await local_disk.machine_image("disk:1")
    identity = "sha256:2"
    rebuilt = await local_disk.machine_image("disk:1")

    assert first == again != rebuilt


EXPORTED = "localhost:5000/agentenv-computers:" + "1" * 64 + "@sha256:" + "a" * 64


@pytest.fixture
def exporting(tmp_path, monkeypatch):
    """This machine's Docker as an export sees it, signed in to a store whose credentials name the ref, every call kept
    in ``log``, ``ran`` (docker's own) and ``runs`` (the processes it starts): the disk a whole one unless ``layers``
    says more, its first layer ``first``, and its guest file missing unless ``guest`` holds its text, or the error
    copying it raises; zstd as a ``zstd:`` prefix."""
    fake = SimpleNamespace(layers=1, first="sha256:linux", guest=None, log=[], ran=[], runs=[],
                           cache=tmp_path / "cache" / "agentenv" / "disks")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(local_disk, "launcher", lambda: "agentenv-qemu-launcher:src-1")
    store = SimpleNamespace(auth=lambda ref: f"credentials for {ref}")
    monkeypatch.setattr(local_disk, "get_config", lambda: SimpleNamespace(get_image_store=lambda: store))
    monkeypatch.setattr(local_disk, "docker_login", lambda auth: fake.ran.append(("login", auth)))

    def docker(*args):
        fake.ran.append(args)
        fake.log.append(args[0])
        if args[0] == "cp":
            file = args[1].partition(":")[2]
            if file.endswith("/guest") and not isinstance(fake.guest, str):
                raise fake.guest or DockerFailed(f"docker cp {args[1]} failed: Error response from daemon: Could not "
                                                 f"find the file {file} in container", 1)
            Path(args[2]).write_bytes(fake.guest.encode() if file.endswith("/guest") else f"copy of {file}".encode())
        return f"sha256:image {fake.layers} {fake.first}" if args[0] == "image" else ""

    def run(argv, stdin=None, stdout=None, **kwargs):
        fake.runs.append(argv)
        fake.log.append(argv[1])
        if argv[:2] == ["docker", "run"]:
            stdout.write(b"zstd:" + stdin.read())
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(local_disk, "docker", docker)
    monkeypatch.setattr(local_disk.subprocess, "run", run)
    return fake


def test_a_disk_is_exported_compressed_into_the_cache_once_for_each_ref(exporting):
    """By this machine's Docker, signed in to the image store as a boot's pull is: the disk copied out of a container of
    its image, then compressed in the launcher."""
    exported = [local_disk.exported_disk(image, "linux/amd64") for image in (EXPORTED, EXPORTED,
                                                                              EXPORTED.replace("1", "2"))]

    assert exported[0] == exported[1] != exported[2] and exported[0].disk.parent == exporting.cache
    assert exported[0].disk.read_bytes() == b"zstd:copy of /disk/disk.img"
    assert exported[0].disk.name.endswith(".img.zst") and exported[0].base is exported[0].guest is None
    assert [args[0] for args in exporting.ran] == ["login", "create", "cp", "image", "cp"] * 2
    assert exporting.ran[0] == ("login", f"credentials for {EXPORTED}") and exporting.ran[1][1] == "--name"
    assert exporting.ran[1][2].startswith("agentenv-export-")
    assert exporting.ran[1][3:] == ("--platform", "linux/amd64", EXPORTED, "disk")
    compressed = [argv for argv in exporting.runs if argv[:2] == ["docker", "run"]]
    assert compressed[0] == ["docker", "run", "--rm", "-i", "--entrypoint", "zstd", "agentenv-qemu-launcher:src-1",
                             "-q", "-3", "-T0", "-c"] and len(compressed) == 2
    assert sorted(p.name for p in exporting.cache.iterdir()) == sorted({exported[0].disk.name, exported[2].disk.name})


def test_an_export_killed_outright_leaves_nothing_behind_once_the_disk_is_exported_again(exporting, monkeypatch):
    """Its work folder, holding as much of a base as it had copied, and its container are named for the disk, and
    cleared under the disk's lock first."""
    left = exporting.cache / f"{hashlib.sha256(EXPORTED.encode()).hexdigest()[:32]}.img.zst.work"
    left.mkdir(parents=True)
    (left / "base.img").write_bytes(b"half a base")
    docker = local_disk.docker

    def copying_with_no_half_base(*args):
        assert args[0] != "cp" or not (left / "base.img").exists()
        return docker(*args)

    monkeypatch.setattr(local_disk, "docker", copying_with_no_half_base)

    exported = local_disk.exported_disk(EXPORTED, "linux/amd64")

    assert exporting.log[:2] == ["rm", "create"] and exporting.log[-1] == "rm"
    assert sorted(path.name for path in exporting.cache.iterdir()) == [exported.disk.name]


def test_two_users_exports_of_one_disk_through_one_docker_name_their_containers_apart(exporting, tmp_path,
                                                                                      monkeypatch):
    """Each holds a lock of its own, so neither clears the other's container."""
    for user in ("one", "two"):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / user / "cache"))
        local_disk.exported_disk(EXPORTED, "linux/amd64")

    assert len({args[2] for args in exporting.ran if args[0] == "create"}) == 2


def test_an_export_that_cant_copy_a_disks_guest_file_fails_rather_than_cache_the_disk_without_it(exporting):
    exporting.guest = DockerFailed("docker cp container1:/disk/guest failed: Error response from daemon: i/o timeout",
                                   1)

    with pytest.raises(DockerFailed, match="i/o timeout"):
        local_disk.exported_disk(EXPORTED, "linux/amd64")
    assert list(exporting.cache.iterdir()) == []


def test_a_thin_disk_is_exported_with_its_guest_file_and_its_base_once_for_every_disk_on_that_base(exporting):
    """A Windows disk is a layer on its base, whose first layer every disk composed on it shares."""
    exporting.layers, exporting.first, exporting.guest = 3, "sha256:base", "windows"

    one, two = (local_disk.exported_disk(image, "windows/amd64") for image in (EXPORTED, EXPORTED.replace("1", "2")))

    copied = [args[1].partition(":")[2] for args in exporting.ran if args[0] == "cp"]
    assert one.disk != two.disk and one.base == two.base and one.base.parent == one.disk.parent
    assert one.base.read_bytes() == b"zstd:copy of /disk/base.img"
    assert one.disk.read_bytes() == b"zstd:copy of /disk/disk.img" and one.guest.read_text() == "windows"
    assert copied == ["/disk/guest", "/disk/base.img", "/disk/disk.img", "/disk/guest", "/disk/disk.img"]
    assert local_disk.exported_disk(EXPORTED, "windows/amd64") == one
    assert len([args for args in exporting.ran if args[0] == "cp"]) == 5
