"""A Linux computer's disk form, made by the boot recipe's Dockerfile with the disk tools as a compose builds it: a GPT
disk of an EFI system partition and an ext4 root filesystem that is a container image's, with the image's start in
/etc/agentenv/vm/image.env, in an image holding only /disk/disk.img, owned by uid 107. A disk made over another keeps
its EFI system partition byte for byte and its size, and each arch's disk is made on this machine's own platform,
without emulating the other's. Needs Docker."""

import base64
import io
import json
import subprocess
import tarfile
import uuid
from pathlib import Path

import pytest

from agent_env.env.published_images import DISK
from agent_env.providers.sandbox_providers.host_images import host_platform
from agent_env.utils.docker.build import DockerBuildError, build_image

pytestmark = pytest.mark.integration

ROOTFS = Path(__file__).parent / "disk" / "rootfs"
UKI = {"amd64": "BOOTX64.EFI", "arm64": "BOOTAA64.EFI"}
ROOT_TYPE = {"amd64": "4F68BCE3-E8CD-4DB1-96E7-FBCAF984B709", "arm64": "B921B045-1DF0-41C3-AF44-4C6F280D3FAE"}
ESP_TYPE = "C12A7328-F81F-11D2-BA4B-00A0C93EC93B"
# Each partition of a disk, written out with what the tools read of its filesystems.
_READ = """set -e
cd /work
qemu-img info --output=json disk.img > info.json
qemu-img convert -O raw disk.img disk.raw
sfdisk -J disk.raw > parts.json
python3 -c '
import json
for i, p in enumerate(json.load(open("parts.json"))["partitiontable"]["partitions"]):
    print(p["start"], p["size"], ("esp", "root")[i])
' | while read -r start size name; do dd if=disk.raw of=$name.img bs=512 skip=$start count=$size status=none; done
mtype -i esp.img ::/EFI/BOOT/$1 > uki
debugfs -R "cat /etc/agentenv/vm/image.env" root.img > image.env 2>/dev/null
debugfs -R "stat /bin/busybox" root.img 2>/dev/null | head -1 > busybox
sha256sum esp.img root.img > sums
rm disk.img disk.raw esp.img root.img
"""


@pytest.fixture(scope="module")
def tools():
    tag = f"agentenv-test-disk-tools:{uuid.uuid4().hex[:8]}"
    build_image(DISK / "tools" / "Dockerfile", DISK / "tools", tag, platform=None)
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


@pytest.fixture
def images():
    made = []
    yield made
    subprocess.run(["docker", "rmi", "-f", *made], capture_output=True)


@pytest.mark.parametrize("arch", ["amd64", "arm64"])
def test_a_disk_is_a_root_filesystem_with_its_start_beside_a_new_esp_and_one_made_over_it_keeps_that_esp(
        tools, images, tmp_path, arch):
    platform = f"linux/{arch}"
    rootfs = _built(images, ROOTFS, platform)
    uki = _built(images, _folder(tmp_path / "uki", {"Dockerfile": f"FROM scratch\nCOPY {UKI[arch]} /\n",
                                                    UKI[arch]: f"a UKI for {arch}"}), platform)
    first = _disk(images, tools, rootfs, uki, platform, tmp_path, SIZE_GB="2")
    changed = _built(images, _folder(tmp_path / "changed", {"Dockerfile": f"FROM {rootfs}\nENV EXTRA=1\n"}), platform)
    second = _disk(images, tools, changed, first, platform, tmp_path, ESP=first)

    made, over = _read(tools, first, tmp_path / "first", arch), _read(tools, second, tmp_path / "second", arch)

    for disk in (made, over):
        assert disk["file"] == ("disk.img", 107, 107)
        assert [(p["type"], p["name"]) for p in disk["partitions"]] == [(ESP_TYPE, "ESP"),
                                                                       (ROOT_TYPE[arch], "agentenv-root")]
        assert disk["virtual_size"] == 2 << 30 and disk["uki"] == f"a UKI for {arch}" and disk["busybox"]
    assert made["start"] == [("env", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
                             ("env", "GREETING=hello disk"), ("argv", "sh"), ("argv", "-c"), ("argv", "echo $GREETING"),
                             ("user", "nobody"), ("workdir", "/tmp")]
    assert ("env", "EXTRA=1") in over["start"] and over["esp"] == made["esp"] and over["root"] != made["root"]


def test_a_new_disk_without_its_size_is_refused_rather_than_made_at_some_default(tools, images, tmp_path):
    platform = host_platform()
    arch = platform.rsplit("/", 1)[1]
    rootfs = _built(images, ROOTFS, platform)
    uki = _built(images, _folder(tmp_path / "uki", {"Dockerfile": f"FROM scratch\nCOPY {UKI[arch]} /\n",
                                                    UKI[arch]: "a UKI"}), platform)

    with pytest.raises(DockerBuildError, match="a new disk needs its size, SIZE_GB"):
        _disk(images, tools, rootfs, uki, platform, tmp_path)


def _built(images: list, folder: Path, platform: str) -> str:
    tag = f"agentenv-test-disk:{uuid.uuid4().hex[:12]}"
    build_image(folder / "Dockerfile", folder, tag, platform=platform)
    images.append(tag)
    return tag


def _disk(images: list, tools: str, rootfs: str, esp: str, platform: str, tmp_path: Path, **args: str) -> str:
    """The disk the boot recipe's Dockerfile makes of ``rootfs`` for ``platform``, its ESP from ``esp``, the image a UKI
    or a disk is in."""
    config = subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config}}", rootfs], capture_output=True,
                            text=True, check=True).stdout.strip()
    tag = f"agentenv-test-disk:{uuid.uuid4().hex[:12]}"
    build_image(DISK / "boot" / "Dockerfile", DISK / "boot", tag, platform=platform, build_args={
        "ROOTFS": rootfs, "TOOLS": tools, "ROOTFS_CONFIG": base64.b64encode(config.encode()).decode(), "ESP": esp,
        **args})
    images.append(tag)
    return tag


def _read(tools: str, image: str, folder: Path, arch: str) -> dict:
    """What ``image`` holds: its file with its owner, its disk's virtual size and partitions, the UKI its EFI system
    partition holds, the start its root filesystem's image.env gives, and each partition's digest."""
    folder.mkdir()
    created = subprocess.run(["docker", "create", image, "none"], capture_output=True, text=True, check=True).stdout.strip()
    try:
        copied = subprocess.run(["docker", "cp", f"{created}:/disk/disk.img", "-"], capture_output=True, check=True)
    finally:
        subprocess.run(["docker", "rm", created], capture_output=True)
    with tarfile.open(fileobj=io.BytesIO(copied.stdout)) as tar:
        (member,) = tar.getmembers()
        (folder / "disk.img").write_bytes(tar.extractfile(member).read())
    reader = subprocess.run(["docker", "create", "--platform", host_platform(), "--entrypoint", "sh", tools, "-c", _READ,
                             "read", UKI[arch]], capture_output=True, text=True, check=True).stdout.strip()
    try:
        for command in (["cp", f"{folder}/.", f"{reader}:/work"], ["start", "--attach", reader],
                        ["cp", f"{reader}:/work/.", str(folder)]):
            subprocess.run(["docker", *command], check=True, capture_output=True)
    finally:
        subprocess.run(["docker", "rm", "--force", reader], capture_output=True)
    start = [(key, base64.b64decode(value).decode()) for line in (folder / "image.env").read_text().splitlines()
             for key, _, value in [line.partition("=")]]
    return {"file": (member.name, member.uid, member.gid), "start": start, "uki": (folder / "uki").read_text(),
            "virtual_size": json.loads((folder / "info.json").read_text())["virtual-size"],
            "partitions": json.loads((folder / "parts.json").read_text())["partitiontable"]["partitions"],
            "busybox": (folder / "busybox").read_text().startswith("Inode:"),
            **{name.removesuffix(".img"): digest for digest, name in (
                line.split() for line in (folder / "sums").read_text().splitlines())}}


def _folder(folder: Path, files: dict[str, str]) -> Path:
    folder.mkdir()
    for name, text in files.items():
        (folder / name).write_text(text)
    return folder
