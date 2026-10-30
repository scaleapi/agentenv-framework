"""Saved images: an image a bundle carries as a ``docker save`` archive is stored as its tar.gz naming no image, under a
name its digest gives, never built, with the build context its Dockerfile names kept beside it, and everything naming it
is written over it."""

import gzip
import hashlib
import io
import json
import tarfile

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.bundle import BundleError
from agent_env.bundle.materialize import materialize
from agent_env.config import get_config
from agent_env.env import Env
from agent_env.store.ids import local_image_repository
from tst.unit.bundle._support import layout, local_builds, plan_of
from tst.util.docker_save import saved_image

ROOT = "@local/~/triage"
IMAGE = f"{ROOT}/crm-image"
WEB = f"{ROOT}/web"
SAVED = saved_image([{"app/hello.txt": "hi\n"}], oci=True, tags=["acme/crm:1"])
AGAIN = saved_image([{"app/hello.txt": "hi again\n"}], oci=True, tags=["acme/crm:1"])
USERS = {
    "envs/crm/env.toml": 'image = "crm-image"\nenvironment_name = "crm"\n',
    "agents/solver/agent.toml": 'image = "crm-image"\n',
    "tasks/t.json": json.dumps([
        {"id": "crm", "type": "deploy_env", "env_id": "crm"},
        {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"},
    ]),
}
FOLDER = {"artifacts/crm-image/artifact.toml": 'type = "docker_image"\narchive = "image.tar"\nplatform = "linux/arm64"\n',
          **USERS}
WEB_FILES = {
    "envs/web/Dockerfile": "FROM scratch\nCOPY server.py /app/\n",
    "envs/web/server.py": "print('serving')\n",
    "envs/web/env.toml": 'environment_name = "web"\nimage = { archive = "./image.tar", dockerfile = "Dockerfile" }\n',
    "tasks/t.json": json.dumps([{"id": "web", "type": "deploy_env", "env_id": "web"}]),
}


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "triage"


def _run(root):
    plan = plan_of(root)
    return materialize(plan, build_here=local_builds(plan))


def _summary(materialization):
    return {done.write.id: (done.version, done.reused, done.reasons) for done in materialization.writes}


def _saved(root, data, name="artifacts/crm-image/image.tar"):
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(data)


def _stored(url):
    with get_config().get_object_store_at(url).open(url) as stream:
        return stream.read()


def _tags(stored):
    """The names the stored tar.gz gives its image: in its manifest.json, and on its index.json descriptors."""
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(stored))) as tar:
        manifest, index = (json.load(tar.extractfile(name)) for name in ("manifest.json", "index.json"))
    return [entry["RepoTags"] for entry in manifest] + [d.get("annotations") for d in index.get("manifests", [])]


def _named(data):
    return f"{local_image_repository(IMAGE)}:{hashlib.sha256(data).hexdigest()[:16]}"


def _images_named(done):
    env = Env.get(f"{ROOT}/crm", done.version_of("env", f"{ROOT}/crm")).docker_image_artifact
    agent = A2AAgent.get(f"{ROOT}/solver", done.version_of("agent", f"{ROOT}/solver")).docker_image_artifact
    return [(image.id, image.version) for image in (env, agent)]


def test_a_saved_image_is_stored_naming_none_under_its_digest_unbuilt_and_everything_naming_it_is_written_over_it(
        bundle_dir, builds):
    layout(bundle_dir, FOLDER)
    _saved(bundle_dir, SAVED)

    done = _run(bundle_dir)

    image = DockerImageArtifact.get(IMAGE)
    assert (image.image_name, image.description, image.platform) == (
        _named(SAVED), "loaded from artifacts/crm-image/image.tar", "linux/arm64")
    assert (image.stored_under_build_tag, image.build_context_object_url) == (True, None)
    assert _tags(_stored(image.tar_gz_object_url)) == [None, {"org.example.kept": "yes"}]
    assert _images_named(done) == [(IMAGE, 1)] * 2
    assert _summary(_run(bundle_dir))[IMAGE] == (1, True, ())
    assert builds == []


def test_a_new_saved_image_is_a_new_version_under_a_new_name_and_everything_naming_it_is_rewritten(bundle_dir):
    layout(bundle_dir, FOLDER)
    _saved(bundle_dir, SAVED)
    _run(bundle_dir)
    _saved(bundle_dir, AGAIN)

    done = _run(bundle_dir)

    assert _summary(done)[IMAGE][:2] == (2, False)
    assert DockerImageArtifact.get(IMAGE).image_name == _named(AGAIN)
    assert _images_named(done) == [(IMAGE, 2)] * 2


def test_a_saved_image_already_gzipped_is_named_by_its_digest_as_given(bundle_dir):
    gzipped = gzip.compress(SAVED)
    layout(bundle_dir, {**FOLDER, "artifacts/crm-image/artifact.toml": 'type = "docker_image"\narchive = "image.tar.gz"\n'})
    _saved(bundle_dir, gzipped, "artifacts/crm-image/image.tar.gz")

    _run(bundle_dir)

    image = DockerImageArtifact.get(IMAGE)
    assert (image.image_name, image.platform) == (_named(gzipped), None)
    assert _tags(_stored(image.tar_gz_object_url))[0] is None


def test_a_file_docker_save_didnt_write_is_refused_naming_its_folder(bundle_dir):
    layout(bundle_dir, FOLDER)
    _saved(bundle_dir, b"not an image")

    with pytest.raises(BundleError) as caught:
        _run(bundle_dir)

    assert caught.value.problems == ("artifacts/crm-image: it isn't a tar docker save writes: truncated header",)


def test_a_saved_image_naming_its_dockerfile_keeps_its_folder_but_the_archive_as_its_build_context(bundle_dir, builds):
    layout(bundle_dir, WEB_FILES)
    _saved(bundle_dir, SAVED, "envs/web/image.tar")

    done = _run(bundle_dir)

    image = Env.get(WEB, done.version_of("env", WEB)).docker_image_artifact
    with tarfile.open(fileobj=io.BytesIO(_stored(image.build_context_object_url))) as context:
        files = sorted(member.name.removeprefix("./") for member in context.getmembers() if member.isfile())
    assert (image.id, image.dockerfile_path, files) == (f"{WEB}__env_image", "Dockerfile",
                                                        ["Dockerfile", "env.toml", "server.py"])
    assert image.stored_under_build_tag and image.source_digest
    assert builds == []


def test_an_edit_to_a_saved_images_build_context_rewrites_it_under_the_same_name(bundle_dir):
    layout(bundle_dir, WEB_FILES)
    _saved(bundle_dir, SAVED, "envs/web/image.tar")
    first = Env.get(WEB, _run(bundle_dir).version_of("env", WEB)).docker_image_artifact
    (bundle_dir / "envs/web/server.py").write_text("print('serving v2')\n")

    done = _run(bundle_dir)

    image = Env.get(WEB, done.version_of("env", WEB)).docker_image_artifact
    assert (image.version, image.image_name) == (2, first.image_name)
    assert image.source_digest != first.source_digest


def test_a_saved_image_is_in_no_build_context_of_its_folder_and_a_new_one_rebuilds_nothing_else(bundle_dir, builds):
    layout(bundle_dir, {
        "envs/shop/Dockerfile": "FROM scratch\n",
        "envs/shop/env.toml": 'type = "website"\nenvironment_name = "shop"\n'
                              'backend_image = { dockerfile = "Dockerfile" }\n'
                              'frontend_image = { archive = "front.tar", dockerfile = "Dockerfile" }\n',
        "tasks/t.json": json.dumps([{"id": "shop", "type": "deploy_env", "env_id": "shop"}]),
    })
    _saved(bundle_dir, SAVED, "envs/shop/front.tar")
    shop = f"{ROOT}/shop"

    first = Env.get(shop, _run(bundle_dir).version_of("env", shop))
    _saved(bundle_dir, AGAIN, "envs/shop/front.tar")
    done = _run(bundle_dir)

    context = _stored(first.frontend_docker_image_artifact.build_context_object_url)
    with tarfile.open(fileobj=io.BytesIO(context)) as tar:
        kept = sorted(member.name.removeprefix("./") for member in tar.getmembers() if member.isfile())
    assert ([call["build"][1] for call in builds], kept) == ([["Dockerfile", "env.toml"]], ["Dockerfile", "env.toml"])
    shop_now = Env.get(shop, done.version_of("env", shop))
    assert (shop_now.backend_docker_image_artifact.version, shop_now.frontend_docker_image_artifact.version) == (1, 2)


def test_an_archive_key_that_isnt_an_images_leaves_its_file_in_the_build_context(bundle_dir, builds):
    layout(bundle_dir, {
        "agents/solver/Dockerfile": "FROM scratch\nCOPY assets.zip /\n",
        "agents/solver/assets.zip": "zipped",
        "agents/solver/agent.toml": 'default_env_vars = { archive = "assets.zip" }\n',
        "tasks/t.json": json.dumps([{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}]),
    })

    _run(bundle_dir)

    assert builds[0]["build"][1] == ["Dockerfile", "agent.toml", "assets.zip"]
