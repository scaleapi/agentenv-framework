"""A disk its guest's init refuses to start, made as a compose makes a disk: a busybox root filesystem whose image runs
as a user its /etc/passwd lacks, beside the EFI system partition of a disk the boot recipe made, in the local
registry."""

import base64
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

from agent_env.env.published_images import DISK
from agent_env.providers.sandbox_providers.host_images import host_platform
from agent_env.utils.docker.build import build_image
from tst.util.published_image import pull_published_else_build

ROOTFS = Path(__file__).parent / "disk" / "rootfs"
#: What the guest's init says of it.
REFUSED = "agentenv-init: the image's user ghost isn't in /etc/passwd"


def ghost_disk(esp: str, folder: Path) -> Iterator[str]:
    """The disk, its EFI system partition ``esp``'s, the image of a disk the boot recipe made, for this machine's
    platform, pushed to the local registry under a tag of its own; every image it made is removed after."""
    platform, made = host_platform(), []
    try:
        tools = _tag(made, "agentenv-test-disk-tools")
        pull_published_else_build("disk tools", DISK / "tools" / "Dockerfile", DISK / "tools", tools, platform=platform)
        busybox = _tag(made, "agentenv-test-disk")
        build_image(ROOTFS / "Dockerfile", ROOTFS, busybox, platform=platform)
        (folder / "Dockerfile").write_text(f"FROM {busybox}\nUSER ghost\n")
        root = _tag(made, "agentenv-test-disk")
        build_image(folder / "Dockerfile", folder, root, platform=platform)
        config = subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config}}", root],
                                capture_output=True, text=True, check=True).stdout.strip()
        disk = _tag(made, "localhost:5000/agentenv-test-ghost-disk")
        build_image(DISK / "boot" / "Dockerfile", DISK / "boot", disk, platform=platform, build_args={
            "ROOTFS": root, "TOOLS": tools, "ROOTFS_CONFIG": base64.b64encode(config.encode()).decode(), "ESP": esp})
        subprocess.run(["docker", "push", "--quiet", disk], capture_output=True, check=True)
        yield disk
    finally:
        subprocess.run(["docker", "rmi", "-f", *made], capture_output=True)


def _tag(made: list[str], repository: str) -> str:
    made.append(f"{repository}:{uuid.uuid4().hex[:12]}")
    return made[-1]
