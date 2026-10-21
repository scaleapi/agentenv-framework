"""Images built from their build context on this machine, for the sandboxes that run here.

An image that is only its build context is built where it runs. Here that is this machine's Docker: built natively,
under the image's name, labeled with the sources it was built from, and built again only when those change, so a later
deploy, a parallel one or another run finds it. A VM on this machine starts with no build cache, so it loads a saved
copy of that build, kept once per sources. Neither is a store version: the image is its build context.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import logging
import platform as host
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.config import get_config
from agent_env.store.local_state import holding_locks
from agent_env.utils.docker.build import build_image, normalize_platform

logger = logging.getLogger(__name__)

_SOURCES_LABEL = "dev.agentenv.sources"
_SAVED_PREFIX = "host-images"


def host_platform() -> str:
    """The platform this machine's Docker builds natively."""
    return normalize_platform(f"linux/{host.machine()}")


def held(image: DockerImageArtifact) -> bool:
    """Whether this machine's Docker holds ``image``, built from its sources."""
    return holds(image.image_name, sources(image))


def holds(image_name: str, image_sources: str) -> bool:
    """Whether this machine's Docker holds ``image_name`` built from ``image_sources``."""
    return _built_from(image_name) == image_sources


def present(image_name: str) -> bool:
    """Whether this machine's Docker holds an image named ``image_name``."""
    try:
        return subprocess.run(["docker", "image", "inspect", image_name], capture_output=True).returncode == 0
    except FileNotFoundError:  # no docker on PATH, so no image here
        return False


def built_here(image_name: str) -> bool:
    """Whether this machine's Docker holds ``image_name`` built here from a build context, which no registry has."""
    return _built_from(image_name) is not None


def sources(image: DockerImageArtifact) -> str:
    """What an image built from ``image``'s build context is labeled with."""
    return image.source_digest or image.build_context_object_url


def build(image: DockerImageArtifact, before: Callable[[], None] | None = None) -> None:
    """Have this machine's Docker hold ``image``, built natively from its stored build context unless it holds one built
    from the same sources, calling ``before`` first. Builds of one image wait for each other. Raises DockerBuildError."""
    with holding_locks([f"image-build:{sources(image)}"]):
        if held(image):
            return
        if before is not None:
            before()
        with tempfile.TemporaryDirectory(prefix="agent-env-build-") as work:
            context = Path(work)
            data = get_config().get_object_store_at(image.build_context_object_url).get(image.build_context_object_url)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                archive.extractall(context, filter="data")
            logger.info(f"Building {image.image_name} from its build context on this machine...")
            _build(context, image.dockerfile_path, image.image_name, sources(image))


def build_staged(context: Path, dockerfile: str, image_name: str, image_sources: str,
                 before: Callable[[], None] | None = None) -> None:
    """``build`` from the build context at ``context``, before it's stored, as ``image_name`` from ``image_sources``,
    calling ``before`` first. Raises DockerBuildError."""
    with holding_locks([f"image-build:{image_sources}"]):
        if holds(image_name, image_sources):
            return
        if before is not None:
            before()
        _build(context, dockerfile, image_name, image_sources)


def _build(context: Path, dockerfile: str, image_name: str, image_sources: str) -> None:
    build_image(context / dockerfile, context, image_name, platform=None, labels={_SOURCES_LABEL: image_sources})


async def build_here(image: DockerImageArtifact) -> None:
    """``build`` off the event loop."""
    await asyncio.to_thread(build, image)


async def saved_here(image: DockerImageArtifact) -> str:
    """The object url of ``image`` built here and saved as a tar.gz, which a VM on this machine loads; saved once per
    sources."""
    return await asyncio.to_thread(_saved_here, image)


def _built_from(image_name: str) -> str | None:
    """The sources this machine's image ``image_name`` was built from, or None when there is none."""
    try:
        result = subprocess.run(["docker", "image", "inspect", "--format",
                                 f'{{{{ index .Config.Labels "{_SOURCES_LABEL}" }}}}', image_name],
                                capture_output=True, text=True, errors="replace")
    except FileNotFoundError:  # no docker on PATH, so no image here; a build says so
        return None
    return (result.stdout.strip() or None) if result.returncode == 0 else None


def _saved_here(image: DockerImageArtifact) -> str:
    build(image)
    objects = get_config().get_object_store()
    # Per name too: images of equal sources under other names share a build, but a save loads under its own name.
    name = hashlib.sha256(image.image_name.encode()).hexdigest()[:16]
    key = f"{_SAVED_PREFIX}/{sources(image).removeprefix('sha256:')}-{host_platform().replace('/', '-')}-{name}.tar.gz"
    with holding_locks([f"image-save:{key}"]):
        if objects.exists(key):
            return objects.object_url(key)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as saved:
            save = subprocess.Popen(["docker", "save", image.image_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with gzip.open(saved.name, "wb", compresslevel=1) as out:
                while chunk := save.stdout.read(1 << 20):
                    out.write(chunk)
            if save.wait() != 0:
                raise RuntimeError(f"docker save {image.image_name} failed: {save.stderr.read().decode(errors='replace')}")
            return objects.put_file(key, saved.name, "application/gzip")
