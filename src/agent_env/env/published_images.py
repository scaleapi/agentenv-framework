"""The images agent-env publishes for the infra envs it ships, for linux/amd64 and linux/arm64: one per Dockerfile, built
from the Dockerfile's folder. Each is tagged by its build context's content, so an install holding the same sources
finds the image built from them; a release wheel also records the digest its release published under each tag."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_env.utils.docker.context import BuildContext

REGISTRY = "ghcr.io/scaleapi"
# The release workflow writes this into the wheels it publishes; a checkout, or a wheel built elsewhere, has none.
RELEASE_FILE = Path(__file__).with_name("published_images.json")
_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")


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
