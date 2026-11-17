import io
import subprocess
import tarfile
from pathlib import Path
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


def _keygen(argv, **kwargs):
    Path(argv[-1]).write_text("key")


def test_a_base_is_built_once_then_reused_packed_from_its_image_and_key_alone_with_the_labels_its_build_gives(
        monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    built, packed, held = [], [], {}
    monkeypatch.setattr(thin_disk, "present", lambda image: bool(packed))
    monkeypatch.setattr(thin_disk, "local_labels", lambda image: packed[-1][1])
    monkeypatch.setattr(thin_disk.subprocess, "run", _keygen)
    monkeypatch.setattr(thin_disk, "pack", lambda guest, folder, image, labels, base_image: packed.append(
        (sorted(path.name for path in folder.iterdir()), labels, base_image)))
    registry = SimpleNamespace(served=held.get, push=lambda local, tag: held.setdefault(tag, f"registry.test/{tag}"))

    def build(work, keys):
        built.append(keys)
        (work / "base.img").write_text("disk")
        (work / "scratch").mkdir()
        return {"dev.agentenv.test": "1"}

    key = "sha256:" + "ef" * 32
    first = thin_disk.base(GUEST, key, build, lambda line: None, registry)
    again = thin_disk.base(GUEST, key, build, lambda line: None, registry)

    assert len(built) == 1 and packed == [(["admin.key", "base.img"], {thin_disk.KEY_LABEL: key,
                                                                       "dev.agentenv.test": "1"}, None)]
    assert first == again == thin_disk.Base(f"agentenv-plan9-base:{'ef' * 8}", built[0], packed[0][1])
    assert list(held) == [f"plan9-base-{'ef' * 8}"]


def test_a_base_the_store_holds_that_its_guest_calls_stale_is_built_afresh_saying_why(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    said, built = [], []
    monkeypatch.setattr(thin_disk, "present", lambda image: False)
    monkeypatch.setattr(thin_disk.subprocess, "run", _keygen)
    monkeypatch.setattr(thin_disk, "pack", lambda guest, folder, image, labels, base_image: None)
    registry = SimpleNamespace(served=lambda tag: "registry.test/old", labels=lambda pinned: {"age": "old"},
                               push=lambda local, tag: "registry.test/new")

    def build(work, keys):
        built.append(keys)
        (work / "base.img").write_text("disk")
        return {"age": "new"}

    base = thin_disk.base(GUEST, "sha256:" + "aa" * 32, build, said.append, registry,
                          stale=lambda labels: "it's old" if labels["age"] == "old" else None)

    assert len(built) == 1 and base.pinned is None and base.labels["age"] == "new"
    assert "it's old; building it afresh" in said
