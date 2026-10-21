"""The images agent-env publishes for its infra envs: one per shipped Dockerfile, tagged by its sources, and the record
of their digests a release wheel carries."""

import hashlib
import json
import shutil

import pytest

from agent_env.env import bootstrap, published_images
from agent_env.env.published_images import (
    REGISTRY,
    PublishedImage,
    _registry_digest,
    published_digest,
    release_digests,
    write_release_digests,
)
from agent_env.store.image_store.registry_api import NoSuchImageError
from agent_env.utils.docker.context import BuildContext

_DIGEST = "sha256:" + "b" * 64


def test_every_shipped_dockerfile_is_published_once_under_a_name_of_its_own():
    shipped = [dockerfile for dockerfiles in bootstrap._DOCKERFILES.values() for dockerfile in dockerfiles]

    assert sorted(image.dockerfile for image in bootstrap.PUBLISHED_IMAGES) == sorted(shipped)
    assert len({image.name for image in bootstrap.PUBLISHED_IMAGES}) == len(bootstrap.PUBLISHED_IMAGES)
    assert all(image.repository == f"{REGISTRY}/{image.name}" for image in bootstrap.PUBLISHED_IMAGES)


@pytest.mark.parametrize("kind", sorted(bootstrap._DOCKERFILES))
def test_a_kinds_tags_are_the_digests_its_build_inputs_record(kind):
    """So an install whose recorded build inputs are this release's holds the sources each tag names."""
    tags = {image.dockerfile: image.tag() for image in bootstrap.PUBLISHED_IMAGES}
    digest = hashlib.sha256()
    for dockerfile in bootstrap._DOCKERFILES[kind]:
        digest.update(f"sha256:{tags[dockerfile].removeprefix('src-')}".encode() + b"\0")

    assert digest.hexdigest() == bootstrap.build_inputs_digest(kind)


def test_a_tag_is_a_valid_docker_tag_named_by_the_sources(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    (context / "app.py").write_text("print('v1')\n")
    image = PublishedImage("agentenv-thing", context / "Dockerfile")
    copy = tmp_path / "copy"
    shutil.copytree(context, copy)
    (copy / "__pycache__").mkdir()
    (copy / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"\0")

    tag = image.tag()

    assert tag == f"src-{BuildContext.of(context, context / 'Dockerfile').source_digest(None).removeprefix('sha256:')}"
    assert len(tag) <= 128 and all(c.isalnum() or c in "_.-" for c in tag)
    assert PublishedImage("agentenv-thing", copy / "Dockerfile").tag() == tag
    (context / "app.py").write_text("print('v2')\n")
    assert image.tag() != tag
    assert image.ref() == f"{REGISTRY}/agentenv-thing:{image.tag()}"


def test_an_install_without_a_record_has_no_release_digests(tmp_path):
    assert release_digests(tmp_path / "published_images.json") == {}


def test_a_record_reads_back_as_it_was_written(tmp_path):
    path = tmp_path / "published_images.json"
    digests = {f"{REGISTRY}/agentenv-gateway:src-{'a' * 64}": _DIGEST,
               f"{REGISTRY}/agentenv-db-web:src-{'c' * 64}": "sha256:" + "d" * 64}

    write_release_digests(digests, path)

    assert release_digests(path) == digests


@pytest.mark.parametrize("content, message", [
    ("not json", "isn't a record of published images"),
    (json.dumps({"images": [{"repository": "r", "tag": "t"}]}), "isn't a record of published images: KeyError"),
    (json.dumps({"images": [{"repository": "r", "tag": "t", "digest": "sha256:short"}]}), "records no sha256 digest for r:t"),
    (json.dumps([]), "isn't a record of published images: TypeError"),
])
def test_a_file_that_isnt_a_record_is_refused(tmp_path, content, message):
    path = tmp_path / "published_images.json"
    path.write_text(content)

    with pytest.raises(ValueError, match=message):
        release_digests(path)


def test_an_install_whose_release_recorded_an_images_digest_pulls_it_without_asking_the_registry(tmp_path, monkeypatch):
    image = bootstrap.PUBLISHED_IMAGES[0]
    record = tmp_path / "published_images.json"
    write_release_digests({image.ref(): _DIGEST}, record)
    monkeypatch.setattr(published_images, "RELEASE_FILE", record)
    monkeypatch.setattr(published_images, "_registry_digest", lambda ref: pytest.fail(f"asked the registry for {ref}"))

    assert published_digest(image) == _DIGEST


def test_an_image_its_release_didnt_record_is_looked_up_and_only_what_was_served_is_remembered(monkeypatch):
    """A published tag is never moved, so the digest it was served under holds; one not served yet may be later."""
    monkeypatch.setattr(published_images, "_registry_digest", _registry_digest)
    monkeypatch.setattr(published_images, "_SERVED", {})
    served, missing = bootstrap.PUBLISHED_IMAGES[:2]
    asked = []

    def pin(ref, auth, *, timeout):
        asked.append((ref, auth, timeout))
        if ref == missing.ref():
            raise NoSuchImageError(f"{ref}: ghcr.io has no such image")
        return f"{ref}@{_DIGEST}"

    monkeypatch.setattr(published_images, "pin_digest", pin)

    assert [published_digest(image) for image in (served, missing, served, missing)] == [_DIGEST, None, _DIGEST, None]
    assert asked == [(served.ref(), None, 10), (missing.ref(), None, 10), (missing.ref(), None, 10)]
