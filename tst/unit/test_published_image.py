"""The gateway tier's fixtures pull a published image of unchanged sources, and build it when there's none or the pull
fails."""

from __future__ import annotations

import subprocess

import pytest

from agent_env.env.bootstrap import PUBLISHED_IMAGES
from tst.util import published_image

IMAGE = PUBLISHED_IMAGES[0]
INDEX = f"{IMAGE.ref()}@sha256:{'1' * 64}"
AMD64 = f"sha256:{'a' * 64}"


@pytest.fixture
def docker(monkeypatch):
    """The ``docker`` commands run, each failing with ``fails[command]`` when set; the registry serves ``INDEX``, an
    index of an amd64 and an arm64 image, unless ``served`` is emptied."""
    state = {"calls": [], "fails": {}, "served": INDEX}

    def run(argv, **kwargs):
        state["calls"].append(argv[1:])
        return subprocess.CompletedProcess(argv, 1 if argv[1] in state["fails"] else 0, "", state["fails"].get(argv[1], ""))

    monkeypatch.setattr(published_image.subprocess, "run", run)
    monkeypatch.setattr(published_image, "published_pinned", lambda image: state["served"])
    monkeypatch.setattr(published_image, "read_manifest", lambda ref, auth: {"manifests": [
        {"digest": f"sha256:{'b' * 64}", "platform": {"os": "linux", "architecture": "arm64"}},
        {"digest": AMD64, "platform": {"os": "linux", "architecture": "amd64"}}]})
    return state


def _make(tag="env-gateway"):
    published_image.pull_published_else_build("gateway", IMAGE.dockerfile, IMAGE.dockerfile.parent, tag)


def test_a_published_image_of_the_same_sources_is_pulled_by_its_amd64_digest_and_tagged_not_built(docker):
    _make()

    pinned = f"{IMAGE.ref()}@{AMD64}"
    assert docker["calls"] == [["pull", "--platform", "linux/amd64", pinned], ["tag", pinned, "env-gateway"]]


def test_an_image_for_another_platform_is_pulled_by_that_platforms_digest(docker):
    published_image.pull_published_else_build("gateway", IMAGE.dockerfile, IMAGE.dockerfile.parent, "base",
                                               platform="linux/arm64")

    pinned = f"{IMAGE.ref()}@sha256:{'b' * 64}"
    assert docker["calls"] == [["pull", "--platform", "linux/arm64", pinned], ["tag", pinned, "base"]]


@pytest.mark.parametrize("failing", ["pull", "tag"])
def test_a_failed_pull_or_tag_builds_the_image_instead(docker, failing, caplog):
    docker["fails"][failing] = "failed to copy: httpReadSeeker: failed open"

    _make()

    assert docker["calls"][-1][0] == "build" and "so it's built" in caplog.text


def test_a_registry_lookup_that_fails_builds_the_image(docker, monkeypatch):
    def unreachable(image):
        raise ValueError("ghcr.io: couldn't read its manifest")

    monkeypatch.setattr(published_image, "published_pinned", unreachable)

    _make()

    assert [call[0] for call in docker["calls"]] == ["build"]


def test_sources_nothing_has_published_and_ones_agent_env_doesnt_publish_are_built(docker, tmp_path):
    docker["served"] = None
    _make()
    published_image.pull_published_else_build("slack", tmp_path / "Dockerfile", tmp_path, "mcp-slack")

    assert [call[0] for call in docker["calls"]] == ["build", "build"]
    assert docker["calls"][1] == ["build", "--platform", "linux/amd64", "-f", str(tmp_path / "Dockerfile"), "-t",
                                  "mcp-slack", str(tmp_path)]


def test_a_failed_build_raises_with_dockers_output(docker):
    docker["served"] = None
    docker["fails"]["build"] = "no such file: Dockerfile"

    with pytest.raises(RuntimeError, match="docker build .* failed: no such file: Dockerfile"):
        _make()
