"""Booting a computer's disk form on this machine's Docker: the QEMU launcher (the published image for the sources
installed, else built here from its folder) with the disk copied in, as an image this machine keeps for the next boot,
run with /dev/kvm; and a disk exported from the image store by this machine's Docker, for a provider elsewhere to
carry."""

import asyncio
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from agent_env.artifact.artifacts.computer import DISK_FILE
from agent_env.artifact.artifacts.docker_image import docker_login
from agent_env.config import get_config
from agent_env.config.paths import cache_root
from agent_env.env.published_images import LAUNCHER_IMAGE, published_pinned
from agent_env.providers.sandbox_providers.disk_launcher import launcher_variables
from agent_env.providers.sandbox_providers.host_images import host_platform, present
from agent_env.providers.sandbox_providers.local_docker.cli import docker
from agent_env.store.local_state import holding_locks
from agent_env.utils.docker.build import build_image

KVM = "/dev/kvm"


def kvm_platforms() -> frozenset[str]:
    """This machine's platform when its Docker can pass KVM to a container: /dev/kvm here, readable and writable."""
    return frozenset({host_platform()}) if os.access(KVM, os.R_OK | os.W_OK) else frozenset()


async def machine_image(disk: str) -> str:
    """An image of the launcher holding ``disk``'s ``DISK_FILE`` for this machine's platform, built unless held."""
    launcher = await asyncio.to_thread(_launcher)
    tag = "agentenv-vm:" + hashlib.sha256(f"{launcher} {disk}".encode()).hexdigest()[:16]
    if not await asyncio.to_thread(present, tag):
        folder = f"/{PurePosixPath(DISK_FILE).parent}/"
        await asyncio.to_thread(_build, f"FROM {launcher}\nCOPY --from={disk} {folder} {folder}\n", tag)
    return tag


def launcher_args(ports: list[int], env: dict[str, str], cpu: float, memory: int) -> list[str]:
    """The ``docker run`` arguments that pass the launcher KVM and its ``launcher_variables``."""
    return ["--device", KVM, *(arg for name, value in launcher_variables(ports, env, cpu, memory).items()
                               for arg in ("-e", f"{name}={value}"))]


def exported_disk(image: str) -> Path:
    """``image``'s ``DISK_FILE`` compressed with zstd in this machine's cache, exported from its image store by this
    machine's Docker unless an export of it is there already: once for each composed disk, by its ref, whose tag is its
    composer's key."""
    path = cache_root() / "disks" / f"{hashlib.sha256(image.encode()).hexdigest()[:32]}.img.zst"
    with holding_locks([f"disk-export:{path.name}"]):
        if not path.exists():
            _export(image, path)
    return path


def _export(image: str, path: Path) -> None:
    """Signed in to the image store first, as a boot's pull is, since ``docker create`` pulls ``image`` when it isn't
    here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    docker_login(get_config().get_image_store().auth(image))
    container = docker("create", image, "disk")
    try:
        with tempfile.TemporaryDirectory(prefix="export-", dir=path.parent) as work:
            raw, compressed = Path(work) / "disk.img", Path(work) / path.name
            docker("cp", f"{container}:/{DISK_FILE}", str(raw))
            with raw.open("rb") as source, compressed.open("wb") as out:
                done = subprocess.run(["docker", "run", "--rm", "-i", "--entrypoint", "zstd", _launcher(), "-q", "-3",
                                       "-T0", "-c"], stdin=source, stdout=out, stderr=subprocess.PIPE, text=False)
            if done.returncode:
                raise RuntimeError(f"compressing {image}'s disk failed: {done.stderr.decode().strip()}")
            os.replace(compressed, path)
    finally:
        subprocess.run(["docker", "rm", container], capture_output=True)


def _launcher() -> str:
    if (pinned := published_pinned(LAUNCHER_IMAGE)) is not None:
        if not present(pinned):
            subprocess.run(["docker", "pull", "-q", pinned], check=True, capture_output=True, text=True)
        return pinned
    tag = f"agentenv-qemu-launcher:{LAUNCHER_IMAGE.tag()}"
    if not present(tag):
        build_image(LAUNCHER_IMAGE.dockerfile, LAUNCHER_IMAGE.dockerfile.parent, tag, platform=None)
    return tag


def _build(dockerfile: str, tag: str) -> None:
    with tempfile.TemporaryDirectory(prefix="agentenv-vm-") as context:
        (Path(context) / "Dockerfile").write_text(dockerfile)
        build_image(Path(context) / "Dockerfile", Path(context), tag, platform=None)
