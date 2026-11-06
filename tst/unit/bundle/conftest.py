"""Fixtures the bundle tests share."""


import shutil

import pytest

from agent_env.artifact.artifacts import docker_image as docker_image_module
from agent_env.env.envs.computer.utils import composer
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.published_images import AGENT_IMAGES
from agent_env.providers.sandbox_providers import host_images
from tst.unit.bundle._support import CLAUDE_CODE_IMAGE, DIGEST, listing


@pytest.fixture
def docker_on_path(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


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


@pytest.fixture
def agents_published(monkeypatch):
    """The registry serves each shipped agent's image, under DIGEST, and nothing else."""
    refs = {image.ref() for image in AGENT_IMAGES}
    monkeypatch.setattr("agent_env.env.published_images._registry_digest", lambda ref: DIGEST if ref in refs else None)
    return CLAUDE_CODE_IMAGE.ref()


@pytest.fixture
def composed(tmp_path, monkeypatch, local_stores, cli_routing, builds):
    """Stands in for the composer: each computer env it's asked to compose is recorded with its computer, driver, provider
    and, when given, the pairs, and nothing is built."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("agent_env.env.published_images._registry_digest", lambda ref: None)
    calls = []

    def compose(env, pairs=None, composing=Composing()):
        if composing.on_progress is not None:
            composing.on_progress("composing")
        calls.append((env.id, env.computer.id, env.driver.id, composing.sandbox_provider)
                     + ((pairs,) if pairs is not None else ()))

    monkeypatch.setattr(composer, "compose", compose)
    return calls
