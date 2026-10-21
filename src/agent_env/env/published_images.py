"""The images agent-env publishes for the infra envs it ships, for linux/amd64 and linux/arm64: one per Dockerfile, built
from the Dockerfile's folder. Each is tagged by its build context's content, so an install holding the same sources
finds the image built from them; a release wheel also records the digest its release published under each tag."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_env.store.image_store.registry_api import pin_digest
from agent_env.utils.docker.context import BuildContext

logger = logging.getLogger(__name__)

REGISTRY = "ghcr.io/scaleapi"
# The release workflow writes this into the wheels it publishes; a checkout, or a wheel built elsewhere, has none.
RELEASE_FILE = Path(__file__).with_name("published_images.json")
_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
# Short, so a network that drops requests doesn't hold up a run's checks for long.
_LOOKUP_TIMEOUT_SECONDS = 10
_SERVED: dict[str, str] = {}  # the digest the registry served each ref under, which never changes: no tag is moved
# The agents bundle agent-env ships, whose agents a run deploys like a bundle's own.
_AGENTS = Path(__file__).parents[1] / "agents" / "agents"


@dataclass(frozen=True)
class PublishedImage:
    """An image agent-env publishes, under ``name`` in the registry, built from ``dockerfile`` with its folder as the
    build context."""

    name: str
    dockerfile: Path

    @property
    def repository(self) -> str:
        return f"{REGISTRY}/{self.name}"

    def tag(self) -> str:
        """``src-`` and the hex of the build context's source digest, the same for the same sources on any platform."""
        digest = BuildContext.of(self.dockerfile.parent, self.dockerfile).source_digest(None)
        return f"src-{digest.removeprefix('sha256:')}"

    def ref(self) -> str:
        """The repository and the tag of the image built from these sources."""
        return f"{self.repository}:{self.tag()}"


# The images of the agents agent-env ships that its releases publish, Claude Code's CLI included: the default agent's.
AGENT_IMAGES = (PublishedImage("agentenv-claude-code", _AGENTS / "claude-code" / "Dockerfile"),)


def published_digest(image: PublishedImage) -> str | None:
    """The digest ``image`` is published under for the sources installed: the one this install's release recorded, else
    the one the registry serves under its tag now; None when it serves none or can't be asked."""
    ref = image.ref()
    return release_digests(RELEASE_FILE).get(ref) or _registry_digest(ref)


def _registry_digest(ref: str) -> str | None:
    if ref not in _SERVED:
        try:
            _SERVED[ref] = pin_digest(ref, None, timeout=_LOOKUP_TIMEOUT_SECONDS).rpartition("@")[2]
        except ValueError as e:
            logger.info(f"No published image for {ref}: {e}")
            return None
    return _SERVED[ref]


def release_digests(path: Path = RELEASE_FILE) -> dict[str, str]:
    """The digest the release published under each ref (``repository:tag``), as its wheel records them; empty when the
    install records none. Raises ValueError when the file isn't such a record."""
    if not path.is_file():
        return {}
    try:
        images = json.loads(path.read_text())["images"]
        digests = {f"{image['repository']}:{image['tag']}": image["digest"] for image in images}
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path} isn't a record of published images: {type(e).__name__}: {e}") from e
    if bad := [ref for ref, digest in digests.items() if not (isinstance(digest, str) and _DIGEST.match(digest))]:
        raise ValueError(f"{path} records no sha256 digest for {', '.join(bad)}")
    return digests


def write_release_digests(digests: Mapping[str, str], path: Path = RELEASE_FILE) -> None:
    """Record ``digests`` (``repository:tag`` to digest) as ``release_digests`` reads them."""
    images = []
    for ref, digest in sorted(digests.items()):
        repository, _, tag = ref.rpartition(":")
        images.append({"repository": repository, "tag": tag, "digest": digest})
    path.write_text(json.dumps({"images": images}, indent=2) + "\n")
