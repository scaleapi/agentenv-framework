"""The Linux disk form's recipe as the init reads what it writes: the image's start (image-env, run by the disk tools) and
the deploy's environment (the launcher's NoCloud seed), each a line per value base64'd so any bytes survive; and the
local provider booting a disk in the launcher, with KVM and the VM's ports, settings and size."""

import asyncio
import base64
import json
import os
import re
import subprocess
import threading
from unittest.mock import AsyncMock

import pytest

from agent_env.env.published_images import DISK
from agent_env.providers.sandbox_providers.local_docker import disk as local_disk
from agent_env.providers.sandbox_providers.local_docker import provider as local_sandbox
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


def test_the_seed_holds_each_vm_env_variable_by_its_name_and_nothing_else(tmp_path):
    environment = {"PATH": os.environ["PATH"], "VM_ENV_AGENTENV_SETTING_RESOLUTION": "1280x800",
                   "VM_ENV_AGENTENV_CONTROL_TOKEN": TRICKY, "VM_ENV_EMPTY": "", "VM_PORTS": "18765", "HOME": "/root"}

    subprocess.run(["sh", str(DISK / "launcher" / "seed"), str(tmp_path)], env=environment, check=True)

    user_data, meta_data = (tmp_path / "user-data").read_text(), (tmp_path / "meta-data").read_text()
    assert user_data.splitlines()[0] == "#agentenv-environment"
    assert _decoded(user_data) == [("AGENTENV_CONTROL_TOKEN", TRICKY), ("AGENTENV_SETTING_RESOLUTION", "1280x800"),
                                   ("EMPTY", "")]
    assert meta_data.endswith("\nlocal-hostname: computer\n") and meta_data.startswith("instance-id: ")


def test_the_inits_decoder_keeps_every_byte_of_a_start_and_a_seed_trailing_newlines_included(tmp_path):
    """As a container runtime would: $(...) alone drops a value's trailing newlines."""
    decoder = re.search(r"^decode\(\) .*$", (DISK / "boot" / "init").read_text(), re.MULTILINE).group(0)
    values = ["plain", "ends with a newline\n", "\n", f"{TRICKY}\n\n"]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"Cmd": values, "User": "user\n"}))
    start = subprocess.run(["python3", str(DISK / "tools" / "image-env"), str(config)], capture_output=True, text=True,
                           check=True).stdout
    subprocess.run(["sh", str(DISK / "launcher" / "seed"), str(tmp_path)], check=True, env={
        "PATH": os.environ["PATH"], **{f"VM_ENV_V{i}": value for i, value in enumerate(values)}})
    seed = (tmp_path / "user-data").read_text()

    def decoded(text: str) -> list[str]:
        script = f"{decoder}\n" + "".join(f"decode {line.partition('=')[2]}; printf '%s\\0' \"$decoded\"\n"
                                          for line in text.splitlines() if line and not line.startswith("#"))
        return subprocess.run(["sh", "-c", script], capture_output=True, text=True, check=True).stdout.split("\0")[:-1]

    assert decoded(start) == [*values, "user\n", "/"] and decoded(seed) == values


def test_the_launcher_gets_kvm_the_vms_ports_and_size_and_its_environment_under_its_prefix():
    args = local_disk.launcher_args([18765, 5900], {"ENVIRONMENT_NAME": "desk", "A": "b c"}, 1.5, 4096)

    assert args == ["--device", "/dev/kvm", "-e", "VM_PORTS=18765 5900", "-e", "VM_CPUS=2", "-e", "VM_MEMORY_MB=4096",
                    "-e", "VM_ENV_ENVIRONMENT_NAME=desk", "-e", "VM_ENV_A=b c"]


@pytest.fixture
def launched(monkeypatch):
    """The ``docker`` commands the local provider runs for a disk, with the launcher image it builds faked."""
    ran = []
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset({"linux/arm64"}))
    monkeypatch.setattr(local_sandbox, "host_platform", lambda: "linux/arm64")
    monkeypatch.setattr(local_sandbox, "machine_image", AsyncMock(return_value="agentenv-vm:0123"))
    monkeypatch.setattr(local_sandbox, "_free_host_port", lambda: 20001)
    monkeypatch.setattr(local_sandbox, "_host_ips", lambda: ("127.0.0.1",))
    monkeypatch.setattr(local_sandbox, "_docker", lambda *args: ran.append(args) or "")
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

    first, removed, second = launched
    assert first == ("run", "-d", "--name", sandbox.container_name, "--label", f"{SANDBOX_LABEL}={sandbox.sandbox_id}",
                     "-p", "127.0.0.1:20001:18765", *local_disk.launcher_args([18765], {"ENVIRONMENT_NAME": "desk"},
                                                                               2, 4096), "agentenv-vm:0123")
    assert removed == f"docker rm -f {sandbox.container_name} > /dev/null"
    assert second[-2:] == ("VM_ENV_ENVIRONMENT_NAME=again", "agentenv-vm:0123") and "VM_CPUS=2" in second
    LocalSandbox.pull_images.assert_awaited_with(["localhost:5000/disk@sha256:" + "a" * 64])
    assert sandbox.tunnel_urls == {18765: "http://127.0.0.1:20001"} and rebuilt.owns_container


@pytest.mark.asyncio
async def test_a_boot_cancelled_while_docker_starts_the_launcher_waits_for_it_then_tears_it_down(launched, monkeypatch):
    started, release, order = threading.Event(), threading.Event(), []

    def docker(*args):
        started.set()
        release.wait(5)
        order.append("run")
        return ""

    monkeypatch.setattr(local_sandbox, "_docker", docker)
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
