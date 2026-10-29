"""The image local Docker runs beside a container it closes off from every network, to publish that container's port:
built here from this folder the first time it's needed, under a tag of its sources."""

import asyncio
import hashlib
from pathlib import Path

from agent_env.providers.sandbox_providers.host_images import present
from agent_env.utils.docker.build import build_image

_FOLDER = Path(__file__).parent
_SOURCES = ("Dockerfile", "relay.py")


async def relay_image() -> str:
    """The relay image's tag, built here unless this machine's Docker already holds it."""
    tag = f"agentenv-relay:{_sources_digest()}"
    if not await asyncio.to_thread(present, tag):
        await asyncio.to_thread(build_image, _FOLDER / "Dockerfile", _FOLDER, tag, platform=None)
    return tag


def _sources_digest() -> str:
    hashed = hashlib.sha256()
    for name in _SOURCES:
        hashed.update(name.encode() + b"\0" + (_FOLDER / name).read_bytes())
    return hashed.hexdigest()[:16]
