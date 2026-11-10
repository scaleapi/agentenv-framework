import io
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from agent_env.env.envs.computer.utils import thin_disk

GUEST = thin_disk.Guest(name="plan9", file="plan9-desktop", user="glenda", prepare="prepare", unpack="unpack",
                        provision="provision", cleanup="cleanup", compact="compact", shutdown="shutdown")


class FakeMachine:
    """A compose's launcher, its guest over ssh and the image store, as composing a disk on a base uses them: each
    command run as whom, and what each step left."""

    def __init__(self, monkeypatch, tmp_path, provisioned=0):
        self.ran, self.unpacked, self.packed, self.pushed, self.removed = [], None, None, [], []
        self.provisioned = provisioned
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

        async def machine_for(image):
            return SimpleNamespace(image="launcher:1", args=("--mount", f"type=image,source={image},target=/disk"))

        monkeypatch.setattr(thin_disk, "machine_for", machine_for)
        monkeypatch.setattr(thin_disk, "launcher", lambda: "launcher:1")
        monkeypatch.setattr(thin_disk, "launched", lambda name, variables, mounts, image: "container")
        monkeypatch.setattr(thin_disk, "ssh_port", lambda container: 2222)
        monkeypatch.setattr(thin_disk, "ssh", self.ssh)
        monkeypatch.setattr(thin_disk, "docker", lambda *args: "false" if "{{.State.Running}}" in args else "0")
        monkeypatch.setattr(thin_disk, "converted", lambda folder, image, backing=None, args=(): None)
        monkeypatch.setattr(thin_disk.subprocess, "run", self.run)
        monkeypatch.setattr(thin_disk, "pack", self.pack)

    def ssh(self, port, keys, user, command, *, check=True, timeout=120, input=None):
        self.ran.append((user, command))
        if command == GUEST.unpack:
            with tarfile.open(fileobj=io.BytesIO(input)) as tar:
                self.unpacked = sorted(tar.getnames())
        code = self.provisioned if command == GUEST.provision else 0
        return subprocess.CompletedProcess([], code, "provision.sh: line 3: no such app\n" if code else "", "")

    def run(self, argv, **kwargs):
        if argv[:3] == ["docker", "rm", "-f"]:
            self.removed.append(argv[3])
        return subprocess.CompletedProcess(argv, 0)

    def pack(self, guest, folder, image, labels, base_image):
        self.packed = (guest, (folder / "guest").read_text(), image, labels, base_image)

    def push(self, local, tag):
        self.pushed.append((local, tag))
        return f"registry.test/disks:{tag}@sha256:1"


def test_a_disk_is_composed_by_the_guests_own_commands_as_its_admin_in_order(monkeypatch, tmp_path):
    machine = FakeMachine(monkeypatch, tmp_path)
    build = tmp_path / "build"
    (build / "context" / "driver").mkdir(parents=True)
    (build / "provision.sh").write_text("#!/bin/sh\n")
    (build / "context" / "driver" / "driver.py").write_text("")
    base = thin_disk.Base("agentenv-plan9-base:1", tmp_path / "keys")
    key = "sha256:" + "ab" * 32

    pinned = thin_disk.compose(GUEST, base, build, key, {"dev.agentenv.test": "1"}, lambda line: None,
                               SimpleNamespace(push=machine.push))

    assert machine.ran == [("glenda", command) for command in ("echo ready", "prepare", "unpack", "provision",
                                                                 "cleanup", "compact", "shutdown")]
    assert machine.unpacked == ["context", "context/driver", "context/driver/driver.py", "provision.sh"]
    assert machine.packed == (GUEST, "plan9-desktop", f"agentenv-plan9:{'ab' * 8}", {"dev.agentenv.test": "1"},
                              "agentenv-plan9-base:1")
    assert machine.pushed == [(f"agentenv-plan9:{'ab' * 8}", "ab" * 32)] and pinned.endswith("@sha256:1")
    assert machine.removed == ["container"]


def test_a_provision_that_fails_stops_the_compose_with_its_output_and_removes_the_launcher(monkeypatch, tmp_path):
    machine = FakeMachine(monkeypatch, tmp_path, provisioned=2)
    build = tmp_path / "build"
    build.mkdir()

    with pytest.raises(RuntimeError, match="provisioning exited 2: provision.sh: line 3: no such app"):
        thin_disk.compose(GUEST, thin_disk.Base("agentenv-plan9-base:1", tmp_path / "keys"), build, "sha256:" + "cd" * 32,
                          {}, lambda line: None, SimpleNamespace(push=machine.push))

    assert [command for _, command in machine.ran][-1] == "provision" and machine.removed == ["container"]
    assert machine.packed is None and machine.pushed == []
