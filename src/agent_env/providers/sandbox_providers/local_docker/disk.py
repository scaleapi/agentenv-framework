"""Booting a computer's disk form on this machine's Docker: the QEMU launcher (the published image for the sources
installed, else built here from its folder) with the disk copied in, as an image this machine keeps for the next boot,
run with /dev/kvm."""

import asyncio
import base64
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from agent_env.artifact.artifacts.computer import DISK_FILE
from agent_env.env.published_images import LAUNCHER_IMAGE, published_pinned
from agent_env.providers.sandbox_providers.host_images import host_platform, present
from agent_env.utils.docker.build import build_image

KVM = "/dev/kvm"
#: How the launcher takes the computer's environment, each variable under this prefix, base64'd (see its launch script).
ENV_PREFIX = "VM_ENV_"


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
    """The ``docker run`` arguments that size the launcher's VM, forward ``ports`` and hand it ``env``, each value
    base64'd, so one of several lines reaches the computer whole."""
    variables = {"VM_PORTS": " ".join(map(str, ports)), "VM_CPUS": str(max(1, math.ceil(cpu))),
                 "VM_MEMORY_MB": str(memory),
                 **{f"{ENV_PREFIX}{name}": base64.b64encode(value.encode()).decode() for name, value in env.items()}}
    return ["--device", KVM, *(arg for name, value in variables.items() for arg in ("-e", f"{name}={value}"))]


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
