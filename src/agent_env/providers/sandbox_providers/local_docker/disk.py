"""Booting a computer's disk form on this machine's Docker: the QEMU launcher (the published image for the sources
installed, else built here from its folder) run with /dev/kvm, the disk image's files mounted into it where this
machine's Docker mounts images, else copied into an image of it this machine keeps for the next boot; and a disk
exported from the image store by this machine's Docker, for a provider elsewhere to carry."""

import asyncio
import functools
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from agent_env.artifact.artifacts.computer import DISK_BASE_FILE, DISK_FILE, DISK_GUEST_FILE
from agent_env.artifact.artifacts.docker_image import docker_login
from agent_env.config import get_config
from agent_env.config.paths import cache_root
from agent_env.env.published_images import LAUNCHER_IMAGE, published_pinned
from agent_env.providers.sandbox_providers.disk_launcher import launcher_variables
from agent_env.providers.sandbox_providers.host_images import host_platform, present
from agent_env.providers.sandbox_providers.local_docker.cli import DockerFailed, docker
from agent_env.store.local_state import holding_locks
from agent_env.utils.docker.build import build_image

KVM = "/dev/kvm"
#: How long asking this machine's Docker whether it mounts images may take, before a boot copies the disk instead.
_DOCKER_INFO_SECONDS = 30
#: What ``docker cp`` says of a path its container lacks: Docker 23 and later, then earlier.
_NO_SUCH_PATH = ("Could not find the file", "No such container:path")


def kvm_platforms() -> frozenset[str]:
    """This machine's platform when its Docker can pass KVM to a container: /dev/kvm here, readable and writable; and
    windows/amd64 on an amd64 machine, whose launcher boots a Windows guest too."""
    if not os.access(KVM, os.R_OK | os.W_OK):
        return frozenset()
    here = host_platform()
    return frozenset({here, "windows/amd64"} if here == "linux/amd64" else {here})


@dataclass(frozen=True)
class Machine:
    """How the launcher boots a disk on this machine: ``image`` run with ``args``. Where Docker mounts images (28 and
    later, on the containerd image store), the launcher itself with the disk image's folder of files mounted read-only
    where the launcher finds them, so no copy of a disk, nor of its base, is made for each launcher; else a machine
    image of the launcher with the disk copied in."""

    image: str
    args: tuple[str, ...] = ()


async def machine_for(disk: str) -> Machine:
    if await asyncio.to_thread(mounts_images):
        folder = PurePosixPath(DISK_FILE).parent
        return Machine(await asyncio.to_thread(launcher),
                       ("--mount", f"type=image,source={disk},target=/{folder},image-subpath={folder}"))
    return Machine(await machine_image(disk))


def mounts_images() -> bool:
    """Whether this machine's Docker mounts an image's files into a container (``--mount type=image``, with its
    ``image-subpath``): its client, which parses the option, and its server both 28 or later, on the containerd image
    store. A Docker that fails to say, or doesn't within ``_DOCKER_INFO_SECONDS``, gets the copy this time and is asked
    again next time; an answer is kept for the process."""
    try:
        return _docker_mounts_images()
    except (DockerFailed, subprocess.TimeoutExpired):
        return False


@functools.cache
def _docker_mounts_images() -> bool:
    client, server, status = docker("info", "--format",
                                    "{{.ClientInfo.Version}} {{.ServerVersion}} {{json .DriverStatus}}",
                                    timeout=_DOCKER_INFO_SECONDS).split(" ", 2)
    return _major(client) >= 28 and _major(server) >= 28 and "io.containerd.snapshotter.v1" in status


def _major(version: str) -> int:
    head = version.split(".", 1)[0]
    return int(head) if head.isdigit() else 0


async def machine_image(disk: str) -> str:
    """An image of the launcher holding ``disk``'s files for this machine's platform, built unless held: one for each
    image ``disk`` names, by its id, so a disk built again under its tag gets a machine of its own. A thin disk's base
    gets a step of its own, so every machine on that base shares one copy of it."""
    source = await asyncio.to_thread(launcher)
    layout = await asyncio.to_thread(_layout, disk)
    tag = "agentenv-vm:" + hashlib.sha256(f"{source} {disk} {layout.identity}".encode()).hexdigest()[:16]
    if not await asyncio.to_thread(present, tag):
        await asyncio.to_thread(_build, _machine_dockerfile(source, disk, thin=layout.base is not None), tag)
    return tag


def _machine_dockerfile(launcher: str, disk: str, thin: bool) -> str:
    folder = f"/{PurePosixPath(DISK_FILE).parent}/"
    if not thin:
        return f"FROM {launcher}\nCOPY --from={disk} {folder} {folder}\n"
    return (f"FROM {launcher}\nCOPY --from={disk} /{DISK_BASE_FILE} /{DISK_BASE_FILE}\n"
            f"COPY --from={disk} /{DISK_FILE} /{DISK_GUEST_FILE} {folder}\n")


@dataclass(frozen=True)
class _Layout:
    """A disk image's id and, for a thin disk, one of more than one layer on ``DISK_BASE_FILE``, its ``base``: the
    digest of its first layer, the base's own, which every disk composed on that base shares; None for a whole one."""

    identity: str
    base: str | None


def _layout(disk: str) -> _Layout:
    identity, layers, first = docker("image", "inspect", "-f",
                                     "{{.Id}} {{len .RootFS.Layers}} {{index .RootFS.Layers 0}}", disk).split()
    return _Layout(identity, first if int(layers) > 1 else None)


def launcher_args(ports: list[int], env: dict[str, str], cpu: float, memory: int) -> list[str]:
    """The ``docker run`` arguments that pass the launcher KVM and its ``launcher_variables``."""
    return ["--device", KVM, *(arg for name, value in launcher_variables(ports, env, cpu, memory).items()
                               for arg in ("-e", f"{name}={value}"))]


@dataclass(frozen=True)
class Exported:
    """A disk exported for a provider elsewhere to carry: its ``DISK_FILE``, compressed; for a thin disk, its base,
    compressed, exported once for each base; and its guest file, when it names one."""

    disk: Path
    base: Path | None = None
    guest: Path | None = None


def exported_disk(image: str, platform: str) -> Exported:
    """``image``'s disk compressed with zstd in this machine's cache, exported from its image store by this machine's
    Docker unless an export of it is there already: once for each composed disk, by its ref, whose tag is its composer's
    key; a thin disk's base once for each base, by its layer, which every disk composed on it shares. ``platform`` is
    the disk's own, which this machine's Docker may not be."""
    cached = _Cached(cache_root() / "disks" / f"{hashlib.sha256(image.encode()).hexdigest()[:32]}.img.zst")
    with holding_locks([f"disk-export:{cached.disk.name}"]):
        if not cached.disk.exists():
            _export(image, platform, cached)
    return cached.exported()


@dataclass(frozen=True)
class _Cached:
    """A disk's export in this machine's cache, named for its image's ref: the ``disk``, compressed, and beside it its
    guest file, a file naming its base's export, which every disk on that base shares, and the folder an export of it
    works in."""

    disk: Path

    @property
    def guest(self) -> Path:
        return self._beside("guest")

    @property
    def base_link(self) -> Path:
        return self._beside("base")

    @property
    def work(self) -> Path:
        return self._beside("work")

    def base_of(self, layer: str) -> Path:
        """The export of the base whose first layer is ``layer``."""
        return self.disk.with_name(f"base-{hashlib.sha256(layer.encode()).hexdigest()[:32]}.img.zst")

    def exported(self) -> Exported:
        base = self.disk.with_name(self.base_link.read_text()) if self.base_link.exists() else None
        return Exported(self.disk, base, self.guest if self.guest.exists() else None)

    def _beside(self, what: str) -> Path:
        return self.disk.with_name(f"{self.disk.name}.{what}")


def _export(image: str, platform: str, cached: _Cached) -> None:
    """Signed in to the image store first, as a boot's pull is, since ``docker create`` pulls ``image`` when it isn't
    here; for ``platform``, as the image's index holds that one alone, which a pull for this machine's wouldn't find.
    The disk is placed last, so one that's there was exported whole, with its base and guest file. Its work folder and
    container are named for the export and cleared first, under its lock, so an export killed outright leaves no copy of
    a base behind once the disk is exported again; the container by the folder's whole path, as another user's export
    of the same disk through this machine's Docker holds a lock of its own."""
    container = f"agentenv-export-{hashlib.sha256(str(cached.work).encode()).hexdigest()[:24]}"
    _cleared(cached.work, container)
    cached.work.mkdir(parents=True)
    docker_login(get_config().get_image_store().auth(image))
    docker("create", "--name", container, "--platform", platform, image, "disk")
    try:
        if (guest := _copied(container, DISK_GUEST_FILE, cached.work / "guest")) is not None:
            guest.replace(cached.guest)
        if (layer := _layout(image).base) is not None:
            base = cached.base_of(layer)
            with holding_locks([f"disk-export:{base.name}"]):
                if not base.exists():
                    _compressed(container, DISK_BASE_FILE, base, cached.work)
            cached.base_link.write_text(base.name)
        _compressed(container, DISK_FILE, cached.disk, cached.work)
    finally:
        _cleared(cached.work, container)


def _cleared(work: Path, container: str) -> None:
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    shutil.rmtree(work, ignore_errors=True)


def _copied(container: str, file: str, to: Path) -> Path | None:
    """``file`` copied out of ``container`` to ``to``; None when the container has no such file. Any other failure is
    raised, as an export cached without a guest file it has would boot it as the wrong OS."""
    try:
        docker("cp", f"{container}:/{file}", str(to))
    except DockerFailed as e:
        if not any(missing in str(e) for missing in _NO_SUCH_PATH):
            raise
        return None
    return to


def _compressed(container: str, file: str, path: Path, work: Path) -> None:
    """``file`` of ``container`` compressed with zstd to ``path``, through ``work``."""
    raw, compressed = work / PurePosixPath(file).name, work / path.name
    docker("cp", f"{container}:/{file}", str(raw))
    with raw.open("rb") as source, compressed.open("wb") as out:
        done = subprocess.run(["docker", "run", "--rm", "-i", "--entrypoint", "zstd", launcher(), "-q", "-3", "-T0",
                               "-c"], stdin=source, stdout=out, stderr=subprocess.PIPE, text=False)
    if done.returncode:
        raise RuntimeError(f"compressing a disk's {file} failed: {done.stderr.decode().strip()}")
    raw.unlink()
    os.replace(compressed, path)


def launcher() -> str:
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
