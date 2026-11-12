"""Thin disks, made with this machine's Docker and KVM: a base packed as ``disk/base.img`` in a layer of its own, an
empty overlay of it as ``disk/disk.img`` and the guest file beside, and a composed disk a layer on its base. A compose
boots the base on an overlay kept on this machine, gives it its parts over ssh as the base's admin, by the key agent-env
made for that base, shuts it down and packs the overlay onto the base's image, so every disk composed on a base shares
one copy of it. A base is found or built by ``base``, the OS's own build making its image. What's its guest OS's own,
how its admin is reached and what its shell runs, is a ``Guest``; Windows' is in ``windows_disk``, Omarchy's in
``omarchy_disk``."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import shutil
import subprocess
import tarfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from agent_env.artifact.artifacts.computer import DISK_BASE_FILE, DISK_FILE, DISK_OWNER
from agent_env.artifact.artifacts.docker_image import docker_login, push_local_image
from agent_env.config.paths import cache_root
from agent_env.providers.sandbox_providers.host_images import present
from agent_env.providers.sandbox_providers.local_docker.cli import docker
from agent_env.providers.sandbox_providers.local_docker.disk import KVM, launcher, machine_for
from agent_env.store.image_store.image_contents import labels as image_labels
from agent_env.store.image_store.image_store import ImageStore
from agent_env.store.image_store.registry_api import pin_digest, pin_if_served
from agent_env.store.local_state import holding_locks
from agent_env.utils.docker.build import build_image

KEY_LABEL = "dev.agentenv.key"
#: Where a base's image keeps the key to its admin, which composing a layer on it takes over ssh. The image is in the
#: user's own image store, and the guest's ssh is reached only on this machine's loopback while a layer is composed.
ADMIN_KEY = PurePosixPath("/disk/admin.key")
#: How long a compose may take.
COMPOSE_TIMEOUT_S = 3600
#: How long a guest may take to power off once told.
SHUTDOWN_TIMEOUT_S = 300
#: How long trimming a disk's free space may take.
COMPACT_TIMEOUT_S = 1800
#: How often a compose looks for the base's ssh, and a shutdown for the guest's end.
BOOT_POLL_S, SHUTDOWN_POLL_S = 5, 2
CPUS, MEMORY_MB = 4, 8192


@dataclass(frozen=True)
class Guest:
    """A thin disk's guest OS as composing on it needs it: its ``name``, which names its images, locks and work folders
    and starts its errors; the launcher's guest file it boots as (``file``); the admin ``user`` its ssh takes; and what
    its shell runs for a compose: ``prepare`` an empty folder for the build, ``unpack`` a tar on stdin into it,
    ``provision`` from it, ``cleanup`` after, ``compact`` its free space, which may fail, and ``shutdown``."""

    name: str
    file: str
    user: str
    prepare: str
    unpack: str
    provision: str
    cleanup: str
    compact: str
    shutdown: str


@dataclass(frozen=True)
class Registry:
    """The env's image store as thin disks are kept in it: by tag in ``repository``, pinned by digest, as Linux
    disks are."""

    store: ImageStore
    repository: str

    def served(self, tag: str) -> str | None:
        """``tag``'s ref, pinned to the digest the store serves for it now; None when it serves none."""
        ref = self.store.image_ref(self.repository, tag)
        return pin_if_served(ref, self.store.auth(ref))

    def labels(self, pinned: str) -> dict[str, str]:
        return image_labels(pinned, "amd64", self.store.auth(pinned))

    def push(self, local: str, tag: str) -> str:
        """``local`` pushed as ``tag``; its ref there, pinned."""
        ref = self.store.image_ref(self.repository, tag)
        push_local_image(local, ref, self.store)
        return pin_digest(ref, self.store.auth(ref))

    def pull(self, pinned: str) -> None:
        docker_login(self.store.auth(pinned))
        docker("pull", "-q", pinned)


class Names(NamedTuple):
    """What a base or a disk is called: ``local`` in this machine's Docker, ``tag`` in the image store."""

    local: str
    tag: str


def base_names(guest: Guest, key: str) -> Names:
    return Names(f"agentenv-{guest.name}-base:{short(key)}", f"{guest.name}-base-{short(key)}")


def disk_names(guest: Guest, key: str) -> Names:
    """A disk's names, its tag in the image store being its key, as a Linux disk's is."""
    return Names(f"agentenv-{guest.name}:{short(key)}", key.removeprefix("sha256:"))


def short(key: str) -> str:
    return key.removeprefix("sha256:")[:16]


@dataclass(frozen=True)
class Base:
    """A base: its image's tag on this machine, the folder of its admin's key, the labels its image carries, and, for one
    the image store holds that this machine doesn't yet, its ref there, pulled when a layer is composed on it."""

    image: str
    keys: Path
    labels: dict[str, str] = field(default_factory=dict)
    pinned: str | None = None


def base_lock(guest: Guest, image: str) -> str:
    return f"{guest.name}-base:{image}"


def base(guest: Guest, key: str, build: Callable[[Path, Path], dict[str, str]], say: Callable[[str], None],
         registry: Registry, stale: Callable[[dict[str, str]], str | None] = lambda labels: None) -> Base:
    """``guest``'s base of ``key``: this machine's, pushed to the image store unless it holds one of ``key``; else the
    one the store holds, pulled with the key to its admin it carries when a layer is composed on it; else built and
    pushed there. One whose labels ``stale`` gives a reason against is built afresh. ``build`` makes ``base.img`` in the
    work folder it's given, its admin reached by the new key in the folder it's given, and returns the labels the base
    records beside its key. One build of a key runs at once on a machine, and another waits for it."""
    (image, tag), keys = base_names(guest, key), cache_root() / guest.name / "bases" / short(key)
    with holding_locks([base_lock(guest, image)], lambda: say("waiting for the build of the same base under way")):
        if present(image) and (keys / "id").exists():
            if (why := stale(labels := local_labels(image))) is None:
                if registry.served(tag) is None:
                    say("pushing the base to the image store, which doesn't hold it")
                    registry.push(image, tag)
                return Base(image, keys, labels)
            say(f"{why}; building it afresh")
        elif (pinned := registry.served(tag)) is not None:
            if (why := stale(labels := registry.labels(pinned))) is None:
                return Base(image, keys, labels, pinned)
            say(f"{why}; building it afresh")
        shutil.rmtree(keys, ignore_errors=True)
        keys.mkdir(parents=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"agentenv {image}", "-f",
                        str(keys / "id")], check=True)
        with working(guest, f"base-{short(key)}") as work:
            labels = {KEY_LABEL: key, **build(work, keys)}
            say("packing the base")
            packed = work / "packed"
            packed.mkdir()
            (work / "base.img").replace(packed / "base.img")
            shutil.copy(keys / "id", packed / ADMIN_KEY.name)
            pack(guest, packed, image, labels, base_image=None)
        say("pushing the base to the image store")
        registry.push(image, tag)
        return Base(image, keys, labels)


def here(guest: Guest, base_: Base, registry: Registry, say: Callable[[str], None]) -> None:
    """``base_`` on this machine, with the key to its admin: pulled from the image store when it's the store's, under
    the base's lock, so a compose of another disk on it waits, then finds it."""
    if base_.pinned is None:
        return
    with holding_locks([base_lock(guest, base_.image)], lambda: say("waiting for the pull of the same base under way")):
        if present(base_.image) and (base_.keys / "id").exists():
            return
        say(f"pulling its base from the image store, {base_.pinned}")
        registry.pull(base_.pinned)
        docker("tag", base_.pinned, base_.image)
        take_admin_key(base_.image, base_.keys)


def take_admin_key(image: str, keys: Path) -> None:
    """The key to the admin of the base ``image`` holds, written to ``keys``, as composing a layer on it needs."""
    shutil.rmtree(keys, ignore_errors=True)
    keys.mkdir(parents=True)
    container = docker("create", image, "none")
    try:
        docker("cp", f"{container}:{ADMIN_KEY}", str(keys / "id"))
    finally:
        subprocess.run(["docker", "rm", container], capture_output=True)
    (keys / "id").chmod(0o600)


def compose(guest: Guest, base_: Base, build: Path, key: str, labels: dict[str, str], say: Callable[[str], None],
            registry: Registry) -> str:
    """The disk of ``key`` composed on ``base_``: the base booted on an overlay kept in a folder of this machine,
    ``build`` copied in and provisioned from as ``guest``'s admin over ssh, then shut down, packed onto the base's image
    with ``labels`` and pushed to the image store; its ref there, pinned. A base the store holds is pulled first."""
    image, tag = disk_names(guest, key)
    here(guest, base_, registry, say)
    with working(guest, f"disk-{short(key)}") as work:
        out = work / "out"
        out.mkdir()
        name = f"agentenv-{guest.name}-compose-{short(key)}"
        say(f"booting its base in {name}")
        machine = asyncio.run(machine_for(base_.image))
        with running(name, {"VM_KEEP": "/out/disk.img"}, ["-v", f"{out}:/out", *machine.args],
                     machine.image) as (container, port):
            answering(guest, container, port, base_.keys, time.monotonic() + COMPOSE_TIMEOUT_S)
            say("installing its layers over ssh")
            ssh(port, base_.keys, guest.user, guest.prepare)
            ssh(port, base_.keys, guest.user, guest.unpack, input=tarred(build), timeout=600)
            done = ssh(port, base_.keys, guest.user, guest.provision, check=False, timeout=COMPOSE_TIMEOUT_S)
            if done.returncode != 0:
                raise RuntimeError(f"provisioning exited {done.returncode}: {(done.stderr + done.stdout).strip()[-3000:]}")
            ssh(port, base_.keys, guest.user, guest.cleanup)
            compacted(guest, port, base_.keys, say)
            shut_down(guest, container, port, base_.keys)
        converted(out, machine.image, backing=f"/{DISK_BASE_FILE}", args=machine.args)
        subprocess.run(["docker", "run", "--rm", "-v", f"{out}:/out", "--entrypoint", "qemu-img", launcher(), "rebase",
                        "-u", "-b", Path(DISK_BASE_FILE).name, "-F", "qcow2", "/out/disk.img"],
                       check=True, capture_output=True)
        (out / "guest").write_text(guest.file)
        say("packing the disk")
        pack(guest, out, image, labels, base_image=base_.image)
    say("pushing the disk to the image store")
    return registry.push(image, tag)


def ssh(port: int, keys: Path, user: str, command: str, *, check: bool = True, timeout: float = 120,
        input: bytes | None = None) -> subprocess.CompletedProcess:
    """``command`` run on the guest as ``user``, over ssh to ``port`` here with the base's key, its output decoded;
    refused when it fails unless not ``check``, as one that took over ``timeout`` has."""
    argv = ["ssh", "-p", str(port), "-i", str(keys / "id"), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o",
            "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o",
            "ConnectTimeout=10", f"{user}@127.0.0.1", command]
    try:
        done = subprocess.run(argv, input=input, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        done = subprocess.CompletedProcess(argv, 255, b"", f"it took over {timeout:.0f} s".encode())
    done = subprocess.CompletedProcess(argv, done.returncode, done.stdout.decode(errors="replace"),
                                       done.stderr.decode(errors="replace"))
    if check and done.returncode != 0:
        raise RuntimeError(f"ssh {command[:200]!r} exited {done.returncode}: "
                           f"{(done.stderr + done.stdout).strip()[-2000:]}")
    return done


def compacted(guest: Guest, port: int, keys: Path, say: Callable[[str], None]) -> None:
    """What the guest keeps that a computer doesn't need removed and its free space trimmed (``guest.compact``), so its
    disk's image, rewritten after it shuts down, keeps only what it uses."""
    say("trimming its free space")
    done = ssh(port, keys, guest.user, guest.compact, check=False, timeout=COMPACT_TIMEOUT_S)
    if done.returncode != 0:
        say(f"trimming exited {done.returncode}, so its image keeps what it held: {(done.stderr + done.stdout)[-300:]}")


def converted(folder: Path, image: str, backing: str | None = None, args: tuple[str, ...] = ()) -> None:
    """``folder``'s ``disk.img`` rewritten by ``qemu-img convert`` in ``image`` run with ``args``, which give it its
    backing file, keeping only the clusters it uses: a base's whole, a layer's over ``backing``."""
    over = ["-B", backing, "-F", "qcow2"] if backing else []
    subprocess.run(["docker", "run", "--rm", "-v", f"{folder}:/out", *args, "--entrypoint", "qemu-img", image, "convert",
                    "-O", "qcow2", *over, "/out/disk.img", "/out/compact.img"], check=True, capture_output=True)
    (folder / "compact.img").replace(folder / "disk.img")


def launched(name: str, variables: dict[str, str], mounts: list[str], image: str) -> str:
    """A launcher container of ``image``, named ``name``, booting with ``variables`` and KVM, its guest's ssh published
    on a port of this machine's loopback; any earlier one of the name removed first."""
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    environment = {"VM_PORTS": "22", "VM_CPUS": str(CPUS), "VM_MEMORY_MB": str(MEMORY_MB), "VM_REQUIRE_KVM": "1",
                   **variables}
    return docker("run", "-d", "--name", name, "--device", KVM, "-p", "127.0.0.1::22", *mounts,
                  *(arg for key, value in environment.items() for arg in ("-e", f"{key}={value}")), image)


def ssh_port(container: str) -> int:
    return int(docker("port", container, "22/tcp").splitlines()[0].rpartition(":")[2])


@contextlib.contextmanager
def running(name: str, variables: dict[str, str], mounts: list[str], image: str) -> Iterator[tuple[str, int]]:
    """A launcher container ``launched`` starts, with the port here its guest's ssh is on; removed when its use ends,
    however it ends."""
    container = launched(name, variables, mounts, image)
    try:
        yield container, ssh_port(container)
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)


def answering(guest: Guest, container: str, port: int, keys: Path, deadline: float) -> None:
    """Return once ``guest``'s admin answers over ssh; raise as ``alive`` does when its QEMU exits or ``deadline`` passes
    first."""
    while ssh(port, keys, guest.user, "echo ready", check=False, timeout=60).returncode != 0:
        alive(guest, container, "booting", deadline)
        time.sleep(BOOT_POLL_S)


def alive(guest: Guest, container: str, doing: str, deadline: float) -> None:
    """Raise when ``container`` has stopped or ``deadline`` has passed, with the end of its output."""
    running = docker("inspect", "-f", "{{.State.Running}}", container) == "true"
    if running and time.monotonic() < deadline:
        return
    tail = subprocess.run(["docker", "logs", "--tail", "20", container], capture_output=True, text=True)
    why = "its QEMU exited" if not running else f"it was still {doing} after the time it's given"
    raise RuntimeError(f"{guest.name}: {why}: {(tail.stdout + tail.stderr).strip()[-2000:]}")


def shut_down(guest: Guest, container: str, port: int, keys: Path) -> None:
    """The guest powered off, and so its launcher's container exited, which keeps its overlay."""
    ssh(port, keys, guest.user, guest.shutdown, check=False, timeout=30)
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    while docker("inspect", "-f", "{{.State.Running}}", container) == "true":
        if time.monotonic() > deadline:
            raise RuntimeError(f"{guest.name}: the guest didn't power off within {SHUTDOWN_TIMEOUT_S} s")
        time.sleep(SHUTDOWN_POLL_S)
    if (code := docker("inspect", "-f", "{{.State.ExitCode}}", container)) != "0":
        raise RuntimeError(f"{guest.name}: its launcher exited {code} as the guest powered off")


def pack(guest: Guest, folder: Path, image: str, labels: dict[str, str], base_image: str | None) -> None:
    """``image`` of ``folder``'s disk: a base's, ``base.img`` in a layer of its own under an empty overlay of it, with
    the key to its admin, or a composed one's ``disk.img`` over ``base_image``; the guest file beside it."""
    owner = f"--chown={DISK_OWNER}:{DISK_OWNER}"
    if base_image is None:
        subprocess.run(["docker", "run", "--rm", "-v", f"{folder}:/out", "--entrypoint", "qemu-img", launcher(), "create",
                        "-q", "-f", "qcow2", "-b", Path(DISK_BASE_FILE).name, "-F", "qcow2", "/out/disk.img"],
                       check=True, capture_output=True)
        (folder / "guest").write_text(guest.file)
        dockerfile = (f"FROM scratch\nCOPY {owner} base.img /{DISK_BASE_FILE}\n"
                      f"COPY --chmod=600 {ADMIN_KEY.name} {ADMIN_KEY}\n"
                      f"COPY {owner} disk.img guest /{Path(DISK_FILE).parent}/\n")
    else:
        dockerfile = f"FROM {base_image}\nCOPY {owner} disk.img guest /{Path(DISK_FILE).parent}/\n"
    (folder / "Dockerfile").write_text(dockerfile)
    build_image(folder / "Dockerfile", folder, image, platform="linux/amd64", labels=labels)


@contextlib.contextmanager
def working(guest: Guest, name: str) -> Iterator[Path]:
    """A work folder of ``name`` in this machine's cache, emptied first of what a build killed in it left, and removed
    after; its caller holds the lock of its build, so one build works in it at a time."""
    work = cache_root() / guest.name / "work" / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


def tarred(folder: Path) -> bytes:
    """``folder``'s files as a tar, which the guest's own tar unpacks."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted(folder.rglob("*")):
            tar.add(path, arcname=path.relative_to(folder).as_posix(), recursive=False)
    return buffer.getvalue()


def local_labels(image: str) -> dict[str, str]:
    """The labels of ``image`` in this machine's Docker."""
    return json.loads(docker("image", "inspect", "-f", "{{json .Config.Labels}}", image) or "null") or {}
