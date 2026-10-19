"""Fixtures the bundle tests share."""


import pytest

from agent_env.artifact.artifacts import docker_image as docker_image_module
from agent_env.bundle import materialize as materialize_module
from agent_env.providers.sandbox_providers import host_images
from tst.unit.bundle._support import DIGEST, listing


@pytest.fixture
def docker_on_path(monkeypatch):
    monkeypatch.setattr(materialize_module.shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.fixture
def builds(monkeypatch, docker_on_path):
    """Stands in for this machine's Docker: each build is recorded with its Dockerfile, the files of its context, a
    link marked with a trailing ``@``, its tag and its platform, and Docker then holds the tag, labeled with its
    sources."""
    calls, held = [], {}

    def build(dockerfile, context, tag, *, platform, labels=None):
        calls.append({"build": (dockerfile.relative_to(context).as_posix(), listing(context), tag, platform)})
        held[tag] = (labels or {}).get(host_images._SOURCES_LABEL)

    monkeypatch.setattr(host_images, "build_image", build)
    monkeypatch.setattr(host_images, "_built_from", held.get)
    return calls


@pytest.fixture
def registry(monkeypatch):
    """Each ref the registry is asked to pin; every tag pins to DIGEST."""
    asked = []

    def pin(ref, auth):
        asked.append(ref)
        return ref if "@" in ref else f"{ref}@{DIGEST}"

    monkeypatch.setattr(docker_image_module, "pin_digest", pin)
    return asked
