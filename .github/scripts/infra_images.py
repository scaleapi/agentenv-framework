"""The infra images agent-env publishes (``agent_env.env.published_images``), for the release workflow's jobs that only
read: this imports agent-env, so it runs where agent-env and its dependencies are installed, and no job that holds a
token to push images or publish to PyPI runs it (those run ``infra_build.py``).

``plan`` prints each image's name, repository, tag, Dockerfile and context as JSON, ``--missing`` only those the
registry hasn't published: it has no image under the tag, or refuses to show one, as GHCR does for a package that
doesn't exist yet, so the first publish can start. ``write PATH`` records the digest the registry serves under every
image's tag as a release wheel carries them, and fails if any is missing. ``public`` warns about each image a user, who
pulls anonymously, can't. The registry is read with ``GHCR_TOKEN`` (as ``GITHUB_ACTOR``) when it's set, anonymously
otherwise, and always anonymously by ``public``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agent_env.env.bootstrap import PUBLISHED_IMAGES
from agent_env.env.published_images import REGISTRY, PublishedImage, write_release_digests
from agent_env.store.image_store.oci_registry_credentials import RegistryAuth
from agent_env.store.image_store.registry_api import NoSuchImageError, RegistryRefusedError, pin_digest

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan").add_argument("--missing", action="store_true")
    commands.add_parser("write").add_argument("path", type=Path)
    commands.add_parser("public")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps([_described(image) for image in PUBLISHED_IMAGES
                          if not args.missing or _published_digest(image) is None]))
        return 0
    if args.command == "public":
        _warn_unless_public()
        return 0
    digests = {image.ref(): _digest(image) for image in PUBLISHED_IMAGES}
    if missing := sorted(ref for ref, digest in digests.items() if digest is None):
        print(f"::error::the registry has no image under {', '.join(missing)}")
        return 1
    write_release_digests(digests, args.path)
    print(f"recorded {len(digests)} published images in {args.path}")
    return 0


def _described(image: PublishedImage) -> dict[str, str]:
    return {"name": image.name, "repository": image.repository, "tag": image.tag(),
            "dockerfile": str(image.dockerfile.relative_to(REPO)), "context": str(image.dockerfile.parent.relative_to(REPO))}


def _digest(image: PublishedImage) -> str | None:
    """The digest the registry serves under the image's tag, or None when it has none."""
    try:
        return pin_digest(image.ref(), _auth()).rpartition("@")[2]
    except NoSuchImageError:
        return None


def _published_digest(image: PublishedImage) -> str | None:
    """As ``_digest``, with a refusal taken as unpublished too: GHCR may refuse a token a package that doesn't exist yet,
    and if the token really can't reach the package, pushing to it fails instead."""
    try:
        return _digest(image)
    except RegistryRefusedError:
        return None


def _warn_unless_public() -> None:
    """A warning for each image an anonymous pull doesn't get, which a new package's first push leaves it as: GHCR makes
    a package private until its admin makes it public, and an install that can't pull it builds the image instead."""
    for image in PUBLISHED_IMAGES:
        try:
            pin_digest(image.ref(), None)
        except ValueError as e:
            why = str(e).split(": ", 1)[-1].split(";")[0]  # what the registry answered, without the ref
            settings = f"https://github.com/orgs/{REGISTRY.split('/')[1]}/packages/container/{image.name}/settings"
            print(f"::warning title={image.name} isn't public::{image.ref()}: {why}. Make the package public at "
                  f"{settings}; until then an install can't pull it and builds the image instead.")


def _auth() -> RegistryAuth | None:
    token = os.environ.get("GHCR_TOKEN")
    return RegistryAuth(REGISTRY.split("/")[0], os.environ.get("GITHUB_ACTOR", "token"), token) if token else None


if __name__ == "__main__":
    sys.exit(main())
