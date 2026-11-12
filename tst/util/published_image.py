"""Tag an image on this machine's Docker the way the gateway tier's and the base computer's fixtures do: pulled when the
registry serves the image agent-env published from the same Dockerfile and build context, else built here."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from agent_env.env.bootstrap import PUBLISHED_IMAGES
from agent_env.env.published_images import published_pinned
from agent_env.store.image_store.registry_api import read_manifest

logger = logging.getLogger(__name__)

_PUBLISHED_IMAGES = {(image.dockerfile, image.dockerfile.parent): image for image in PUBLISHED_IMAGES}


def pull_published_else_build(name: str, dockerfile: Path, context: Path, tag: str,
                              platform: str = "linux/amd64") -> None:
    """``tag``, pulled for ``platform`` when the registry serves the published image of these sources, whose tag is a
    digest of them, so a change to them is built; also built when the lookup or pull fails. A failed build raises."""
    published = _PUBLISHED_IMAGES.get((dockerfile, context))
    try:
        if pinned := published and _platform_image(published_pinned(published), platform):
            logger.info(f"Pulling {name}'s published image {pinned}...")
            _docker("pull", "--platform", platform, pinned)
            _docker("tag", pinned, tag)
            return
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Pulling {name}'s published image failed, so it's built: {e}")
    logger.info(f"Building {name} Docker image...")
    _docker("build", "--platform", platform, "-f", str(dockerfile), "-t", tag, str(context))


def _platform_image(pinned: str | None, platform: str) -> str | None:
    """``pinned``'s image for ``platform``, pinned by its own digest: the containerd image store keeps a pulled index
    whole, and ``docker save`` of one it holds only some platforms of fails."""
    if pinned is None:
        return None
    manifest = read_manifest(pinned, None)
    if "manifests" not in manifest:
        return pinned
    os, _, architecture = platform.partition("/")
    digest = next((m["digest"] for m in manifest["manifests"]
                   if m.get("platform", {}).get("os") == os and m["platform"].get("architecture") == architecture), None)
    return digest and f"{pinned.partition('@')[0]}@{digest}"


def _docker(*args: str) -> None:
    result = subprocess.run(["docker", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{shlex.join(['docker', *args])} failed: {result.stderr}")
