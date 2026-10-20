"""The infra images agent-env publishes (``agent_env.env.published_images``), for the release workflow.

``plan`` prints each image's name, repository, tag, Dockerfile and context as JSON, ``--missing`` only those whose tag
the registry doesn't have; ``smoke NAME IMAGE`` checks that IMAGE, a local build of NAME, runs; ``write PATH`` records
the digest the registry serves under every image's tag as a release wheel carries them, and fails if any is missing;
``notes PATH`` prints that record as the release notes list it; ``public`` warns about each image a user, who pulls
anonymously, can't. The registry is read with ``GHCR_TOKEN`` (as ``GITHUB_ACTOR``) when it's set, anonymously otherwise,
and always anonymously by ``public``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from agent_env.env.bootstrap import PUBLISHED_IMAGES
from agent_env.env.published_images import REGISTRY, PublishedImage, release_digests, write_release_digests
from agent_env.store.image_store.oci_registry_credentials import RegistryAuth
from agent_env.store.image_store.registry_api import NoSuchImageError, pin_digest

REPO = Path(__file__).resolve().parents[2]
_HEALTHY_WITHIN_SECONDS = 120
_HEALTH_POLL_SECONDS = 2
_LOG_TAIL_LINES = 30  # of a container that never turned healthy, in the error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan").add_argument("--missing", action="store_true")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("name", choices=[image.name for image in PUBLISHED_IMAGES])
    smoke.add_argument("image")
    commands.add_parser("write").add_argument("path", type=Path)
    commands.add_parser("notes").add_argument("path", type=Path)
    commands.add_parser("public")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps([_described(image) for image in PUBLISHED_IMAGES
                          if not args.missing or _digest(image) is None]))
        return 0
    if args.command == "smoke":
        return _smoke(args.name, args.image)
    if args.command == "public":
        _warn_unless_public()
        return 0
    if args.command == "notes":
        print("Infra images this release pulls, by digest:")
        print("\n".join(f"- `{ref}@{digest}`" for ref, digest in release_digests(args.path).items()))
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


# Each image's check, beyond its build: the gateway imports its server, Postgres turns healthy, and the sidecars and the
# browser answer their own commands.
_SMOKES = {
    "agentenv-gateway": ["docker", "run", "--rm", "--entrypoint", "python", "{image}", "-c", "import gateway.entrypoint"],
    "agentenv-db-web": ["docker", "run", "--rm", "--entrypoint", "/usr/bin/pgweb", "{image}", "--version"],
    "agentenv-db-mcp": ["docker", "run", "--rm", "{image}", "--help"],
    "agentenv-website-browser": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                                 'npx --no-install @playwright/mcp@"$PLAYWRIGHT_MCP_VERSION" --help >/dev/null'
                                 " && ls -d /root/.cache/ms-playwright/chromium-*"],
}


def _smoke(name: str, image: str) -> int:
    if name == "agentenv-service-db":
        return _healthy(image)
    command = [part.replace("{image}", image) for part in _SMOKES[name]]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"::error::{name}: `{' '.join(command)}` exited {result.returncode}:\n{result.stdout}{result.stderr}")
    return result.returncode


def _healthy(image: str) -> int:
    """Start ``image`` and wait for its health check to pass, as compose does before starting what depends on it."""
    container = subprocess.run(["docker", "run", "-d", image], capture_output=True, text=True, check=True).stdout.strip()
    try:
        deadline = time.monotonic() + _HEALTHY_WITHIN_SECONDS
        while time.monotonic() < deadline:
            status = subprocess.run(["docker", "inspect", "-f", "{{.State.Health.Status}}", container],
                                    capture_output=True, text=True).stdout.strip()
            if status == "healthy":
                return 0
            time.sleep(_HEALTH_POLL_SECONDS)
        logs = subprocess.run(["docker", "logs", "--tail", str(_LOG_TAIL_LINES), container], capture_output=True,
                              text=True)
        print(f"::error::{image} wasn't healthy within {_HEALTHY_WITHIN_SECONDS}s (last: {status}):\n"
              f"{logs.stdout}{logs.stderr}")
        return 1
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)


if __name__ == "__main__":
    sys.exit(main())
