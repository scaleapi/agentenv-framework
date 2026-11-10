"""The images agent-env publishes for its infra envs, its agents, an example's env, its base computers and their disk
forms: one per shipped Dockerfile, tagged by its sources (a disk form's by its base's too), and the record of their
digests a release wheel carries."""

import dataclasses
import hashlib
import json
import shutil
import tomllib

import pytest

from agent_env.env import bootstrap, published_images
from agent_env.env.published_images import (
    AGENT_IMAGES,
    DISK,
    DISK_IMAGES,
    DISK_TOOLS_IMAGE,
    EXAMPLE_IMAGES,
    REGISTRY,
    PublishedImage,
    _registry_digest,
    published_bundle_image,
    published_digest,
    published_disk,
    release_digests,
    write_release_digests,
)
from agent_env.store.image_store.registry_api import NoSuchImageError
from agent_env.utils.docker.context import BuildContext

_DIGEST = "sha256:" + "b" * 64


def test_every_shipped_dockerfile_is_published_once_under_a_name_of_its_own():
    """The infra envs', every shipped agent's, the env `a2a-agent validate` deploys, every shipped base computer's, and
    the Linux disk form's recipe: its disk tools, its QEMU launcher, and each base's disk, from the boot recipe."""
    shipped = [dockerfile for dockerfiles in bootstrap._DOCKERFILES.values() for dockerfile in dockerfiles]
    shipped += published_images._AGENTS.glob("*/Dockerfile")
    shipped.append(published_images._EXAMPLES / "mail-mcp" / "envs" / "mail" / "Dockerfile")
    shipped += published_images._COMPUTERS.glob("*/Dockerfile")
    shipped += DISK.glob("*/Dockerfile")
    shipped += [DISK / "boot" / "Dockerfile"] * (len(DISK_IMAGES) - 1)

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


def test_a_shipped_agents_sources_find_its_published_image_and_other_sources_none(monkeypatch):
    claude_code = next(image for image in AGENT_IMAGES if image.name == "agentenv-claude-code")
    sources = BuildContext.of(claude_code.dockerfile.parent, claude_code.dockerfile).source_digest(None)
    monkeypatch.setattr(published_images, "_registry_digest", lambda ref: _DIGEST if ref == claude_code.ref() else None)

    assert published_bundle_image(sources) == (claude_code.ref(), _DIGEST)
    assert published_bundle_image("sha256:" + "0" * 64) is None


def test_the_mail_examples_env_sources_find_its_published_image(monkeypatch):
    mail = EXAMPLE_IMAGES[0]
    sources = BuildContext.of(mail.dockerfile.parent, mail.dockerfile).source_digest(None)
    monkeypatch.setattr(published_images, "_registry_digest", lambda ref: _DIGEST if ref == mail.ref() else None)

    assert mail.name == "agentenv-mail-mcp" and published_bundle_image(sources) == (mail.ref(), _DIGEST)


def test_a_recorded_image_the_registry_doesnt_serve_anonymously_isnt_published_yet(monkeypatch, tmp_path):
    """A new package stays private until a package admin makes it public, so a sandbox pulling it by name would fail."""
    mail = EXAMPLE_IMAGES[0]
    sources = BuildContext.of(mail.dockerfile.parent, mail.dockerfile).source_digest(None)
    write_release_digests({mail.ref(): _DIGEST}, tmp_path / "published_images.json")
    monkeypatch.setattr(published_images, "RELEASE_FILE", tmp_path / "published_images.json")
    served = {}
    monkeypatch.setattr(published_images, "_registry_digest", lambda ref: served.get(ref))

    assert published_bundle_image(sources) is None
    served[mail.ref()] = _DIGEST
    assert published_bundle_image(sources) == (mail.ref(), _DIGEST)


def test_a_shipped_agents_sources_whose_image_isnt_published_find_none(monkeypatch):
    claude_code = next(image for image in AGENT_IMAGES if image.name == "agentenv-claude-code")
    sources = BuildContext.of(claude_code.dockerfile.parent, claude_code.dockerfile).source_digest(None)

    assert published_bundle_image(sources) is None



def test_a_disk_forms_tag_is_of_its_boot_recipe_its_bases_sources_and_the_disk_tools_and_it_is_found_by_its_bases(
        tmp_path, monkeypatch):
    (disk,) = DISK_IMAGES
    base = dict(disk.args)["ROOTFS"]
    recipe = BuildContext.of(disk.dockerfile.parent, disk.dockerfile).source_digest(None).removeprefix("sha256:")
    served = {}
    monkeypatch.setattr(published_images, "_registry_digest", served.get)
    other = tmp_path / "base"
    shutil.copytree(base.dockerfile.parent, other)
    (other / "start-desktop").write_text("#!/bin/sh\n")
    moved = PublishedImage(disk.name, disk.dockerfile, (("ROOTFS", PublishedImage(base.name, other / "Dockerfile")),
                                                        ("TOOLS", DISK_TOOLS_IMAGE)))
    sources = BuildContext.of(base.dockerfile.parent, base.dockerfile).source_digest(None)

    size = tomllib.loads(base.dockerfile.with_name("artifact.toml").read_text())["forms"]["disk"]["min_disk_gb"]
    assert disk.name == "agentenv-ubuntu-24.04-disk" and dict(disk.args) == {"ROOTFS": base, "TOOLS": DISK_TOOLS_IMAGE}
    assert disk.build_args == (("SIZE_GB", str(size)),)
    assert disk.tag() == "src-" + hashlib.sha256("\0".join(
        [recipe, f"ROOTFS={base.tag()}", f"TOOLS={DISK_TOOLS_IMAGE.tag()}", f"SIZE_GB={size}"]).encode()).hexdigest()
    assert moved.tag() != disk.tag() != dataclasses.replace(disk, build_args=(("SIZE_GB", "32"),)).tag()
    assert published_disk(sources) is None
    served[disk.ref()] = _DIGEST
    assert published_disk(sources) == f"{disk.ref()}@{_DIGEST}" and published_disk("sha256:" + "0" * 64) is None
