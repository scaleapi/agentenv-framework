"""DockerImageArtifact.put's ``docker save`` streaming: bounded, drained, and reaped.

Regression guard: put used to read the save process's stdout to EOF before waiting
on it, with stderr never drained. A save that filled its stderr pipe deadlocked, and
a stalled save blocked the read itself, so the timeout in the wait never fired.
The docker CLI is a PATH stub; the image store and the push are stubbed at the same
boundaries the other docker image tests use. No Docker daemon is involved.
"""

import gzip
import os
import time
from types import SimpleNamespace

import pytest

import agent_env.artifact.artifacts.docker_image as docker_image
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.config import get_config

_STUB = """#!/bin/sh
if [ "$1" = "save" ]; then
  case "$FAKE_DOCKER_MODE" in
    flood) python3 -c 'import sys; sys.stderr.write("x" * 300000); sys.stdout.write("FAKETAR")' ;;
    stall) exec sleep 60 ;;
    fail) echo "boom: no such image" >&2; exit 3 ;;
    *) printf 'FAKETAR' ;;
  esac
  exit 0
fi
exit 0
"""


@pytest.fixture
def stubbed_put(local_stores, tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "docker"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setattr(docker_image, "push_local_image", lambda *a, **k: None)
    image_store = SimpleNamespace(
        image_ref=lambda repo, tag: f"localhost:5000/{repo}:{tag}",
        ensure_repository=lambda repo: None,
    )
    monkeypatch.setattr(get_config(), "get_image_store_for", lambda id: image_store)
    monkeypatch.setattr(DockerImageArtifact, "DOCKER_SAVE_TIMEOUT_SECONDS", 2)
    return monkeypatch


def _put(monkeypatch, mode, artifact_id):
    monkeypatch.setenv("FAKE_DOCKER_MODE", mode)
    return DockerImageArtifact.put(artifact_id, description="d", image_name="img:latest")


def test_save_with_saturated_stderr_completes(stubbed_put):
    art = _put(stubbed_put, "flood", "img-flood")

    assert gzip.decompress(art.load()) == b"FAKETAR"


def test_stalled_save_is_killed_at_the_timeout(stubbed_put):
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out after 2 seconds"):
        _put(stubbed_put, "stall", "img-stall")
    assert time.monotonic() - start < 20


def test_failed_save_reports_its_stderr(stubbed_put):
    with pytest.raises(RuntimeError, match="boom: no such image"):
        _put(stubbed_put, "fail", "img-fail")


def test_save_streams_the_archive_to_the_object_store(stubbed_put):
    art = _put(stubbed_put, "ok", "img-ok")

    assert art.version == 1
    assert gzip.decompress(art.load()) == b"FAKETAR"
