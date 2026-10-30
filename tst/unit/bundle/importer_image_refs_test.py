"""``bundle init --image-ref``: an image given a registry reference is written as that reference, its tag pinned to the
digest its registry serves now, which a run pulls, with the part of its stored build context its Dockerfile reads kept
beside it."""

import json
import tomllib

import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError
from agent_env.bundle import importer as importer_module
from agent_env.bundle.importer import import_task, task_images
from agent_env.bundle.materialize import materialize
from agent_env.bundle.resolve import build_step
from agent_env.cli import cli
from agent_env.config import get_config
from agent_env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from agent_env.task import Task
from tst.unit.bundle._support import context_image, local_builds, plan_of

REF = "123456789012.dkr.ecr.us-west-2.amazonaws.com/delivery/crm:t1@sha256:" + "c" * 64
TAG = REF.split("@")[0]
DEPLOY = {"id": "env", "type": "deploy_env", "env_id": "crm"}


@pytest.fixture
def home(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def pins(monkeypatch):
    """Each ref the import asks its registry to pin; a tag pins to REF's digest."""
    asked = []

    def pin(ref, auth):
        asked.append(ref)
        return f"{ref}@{REF.split('@')[1]}"

    monkeypatch.setattr(importer_module, "pin_digest", pin)
    return asked


def _stored(image):
    MCPServerEnv.put(id="crm", docker_image_artifact=image, environment_name="crm")
    Task.put(id="t", steps=[build_step(dict(DEPLOY))])


def _toml(root, path):
    return tomllib.loads((root / path).read_text())


def test_an_image_given_a_ref_is_written_as_it_with_its_pruned_context_beside_it(home):
    _stored(context_image(home, "crm-image"))

    imported = import_task("t", home / "t", image_refs={"crm-image": REF})

    assert _toml(imported.root, "envs/crm/env.toml")["image"] == {"ref": REF, "dockerfile": "Dockerfile"}
    assert sorted(path.name for path in (imported.root / "envs/crm").iterdir()) == ["Dockerfile", "app.py", "env.toml"]
    (image,) = imported.images
    assert (image.ref, image.locked, image.unlocked, imported.pulled) == (REF, None, None, ())
    assert dict(imported.report())["Images"] == [
        f"envs/crm: image pulled as {REF}, its build context beside it from Dockerfile, 2 file(s), 61 B; largest: "
        "Dockerfile (42 B), app.py (19 B)"]
    readme = (imported.root / "README.md").read_text()
    assert "An image *Images* says is pulled comes from its registry and is never built here." in readme
    assert "Its images are built from the build contexts" not in readme
    assert "`123456789012.dkr.ecr.us-west-2.amazonaws.com`" in readme
    assert json.loads((imported.root / "provenance.json").read_text())["images"][0]["ref"] == REF


def test_an_imported_ref_is_written_by_a_run_as_that_image_pinned_with_its_context_kept(home, registry, builds,
                                                                                         cli_routing):
    _stored(context_image(home, "crm-image"))
    imported = import_task("t", home / "t", image_refs={"crm-image": REF})

    plan = plan_of(imported.root)
    done = materialize(plan, build_here=local_builds(plan))

    env = "@local/~/t/crm"
    image = Env.get(env, done.version_of("env", env)).docker_image_artifact
    assert (image.pinned_name, image.dockerfile_path, bool(image.build_context_object_url)) == (REF, "Dockerfile", True)
    assert (registry, builds) == ([REF], [])


def test_an_image_with_no_context_given_a_ref_is_written_as_the_ref_alone(home):
    _stored(get_artifact_store().put_document(DockerImageArtifact(
        id="crm-image", description="d", image_name="crm-image:v1", tar_gz_s3_url="file:///nowhere/crm.tar.gz")))

    imported = import_task("t", home / "t", image_refs={"crm-image": REF})

    assert _toml(imported.root, "envs/crm/env.toml")["image"] == {"ref": REF}
    assert [(pull.folder, pull.image) for pull in imported.pulled] == [("envs/crm", REF)]


def test_a_tag_given_for_an_image_is_pinned_to_the_digest_its_registry_serves_once_per_image(home, pins):
    image = context_image(home, "crm-image")
    MCPServerEnv.put(id="crm", docker_image_artifact=image, environment_name="crm")
    MCPServerEnv.put(id="crm-eu", docker_image_artifact=image, environment_name="crm")
    Task.put(id="t", steps=[build_step(dict(DEPLOY)), build_step({**DEPLOY, "id": "env2", "env_id": "crm-eu"})])

    imported = import_task("t", home / "t", image_refs={"crm-image": TAG})

    assert [_toml(imported.root, f"envs/{name}/env.toml")["image"]["ref"] for name in ("crm", "crm-eu")] == [REF] * 2
    assert pins == [TAG]


def test_a_digest_given_for_an_image_is_written_as_given_without_asking_its_registry(home, pins):
    _stored(context_image(home, "crm-image"))

    import_task("t", home / "t", image_refs={"crm-image": REF})

    assert pins == []


def test_a_tag_its_registry_cant_pin_fails_the_import_naming_the_image(home, monkeypatch):
    _stored(context_image(home, "crm-image"))

    def refuse(ref, auth):
        raise ValueError(f"{ref}: the registry has no such tag")

    monkeypatch.setattr(importer_module, "pin_digest", refuse)

    with pytest.raises(BundleError) as caught:
        import_task("t", home / "t", image_refs={"crm-image": TAG})

    assert caught.value.problems == (f"envs/crm: image: the reference given for crm-image, {TAG!r}, can't be pinned to "
                                     f"a digest: {TAG}: the registry has no such tag",)


@pytest.mark.parametrize("refs, problem", [
    ({"crm-image": TAG + "@sha256:"},
     "envs/crm: image: the reference given for crm-image, '" + TAG + "@sha256:', doesn't give a whole digest, as "
     "<registry>/<repo>:<tag>@sha256:<64 hex digits> does"),
    ({"crm-image": REF[:-1]}, "doesn't give a whole digest"),
    ({"crm-image": REF[:-1] + "X"}, "doesn't give a whole digest"),
    ({"crm-image": "crm@sha256:" + "c" * 64}, "doesn't name its registry"),
    ({"crm-image": REF, "other-image": REF},
     "a reference was given for other-image, which the import writes no image of"),
], ids=["empty digest", "short digest", "not hex", "no registry", "unused"])
def test_a_ref_that_cant_be_written_fails_the_import_and_writes_nothing(home, refs, problem):
    _stored(context_image(home, "crm-image"))

    with pytest.raises(BundleError) as caught:
        import_task("t", home / "t", image_refs=refs)

    assert any(problem in found for found in caught.value.problems)
    assert not (home / "t").exists()


def test_an_image_whose_context_no_configured_store_can_read_is_written_as_its_ref_alone(home, monkeypatch):
    _stored(get_artifact_store().put_document(DockerImageArtifact(
        id="crm-image", description="d", image_name="crm-image:v1",
        build_context_s3_url="ftp://elsewhere/crm/build-context.tar.gz", dockerfile_path="Dockerfile")))
    config_type, store_at = type(get_config()), type(get_config()).get_object_store_at

    def refuse_ftp(self, url):
        if url.startswith("ftp://"):
            raise ValueError(f"no configured object store reads {url}")
        return store_at(self, url)

    monkeypatch.setattr(config_type, "get_object_store_at", refuse_ftp)

    imported = import_task("t", home / "t", image_refs={"crm-image": REF})

    assert _toml(imported.root, "envs/crm/env.toml")["image"] == {"ref": REF}
    assert [(pull.image, pull.why) for pull in imported.pulled] == [
        (REF, "its build context can't be downloaded: the configured stores can't read it")]
    readme = (imported.root / "README.md").read_text()
    assert "comes from its registry and is never built here" in readme
    assert "Its images are built from the build contexts" not in readme


def test_the_cli_takes_each_ref_as_id_equals_ref(home):
    _stored(context_image(home, "crm-image"))
    runner = CliRunner()

    written = runner.invoke(cli, ["bundle", "init", "--from-task", "t", "--image-ref", f"crm-image={REF}",
                                  str(home / "t")])
    refused = runner.invoke(cli, ["bundle", "init", "--from-task", "t", "--image-ref", REF, str(home / "u")])

    assert written.exit_code == 0, written.output
    assert _toml(home / "t", "envs/crm/env.toml")["image"]["ref"] == REF
    assert refused.exit_code == 2 and "--image-ref takes ID=REF" in refused.output


def test_the_images_a_task_writes_are_listed_once_in_order_without_downloading_and_take_refs(home, monkeypatch):
    shared, registry_image = context_image(home, "crm-image"), get_artifact_store().put_document(DockerImageArtifact(
        id="mail-image", description="d", image_name="ghcr.io/acme/mail:1"))
    MCPServerEnv.put(id="crm", docker_image_artifact=shared, environment_name="crm")
    MCPServerEnv.put(id="crm-eu", docker_image_artifact=shared, environment_name="crm")
    MCPServerEnv.put(id="inbox", docker_image_artifact=registry_image, environment_name="inbox")
    Task.put(id="t", steps=[build_step({**DEPLOY, "id": f"env-{env}", "env_id": env})
                            for env in ("crm", "crm-eu", "inbox")])
    downloads = []
    real_download = LocalFilesystemObjectStore.download_to_file
    monkeypatch.setattr(LocalFilesystemObjectStore, "download_to_file",
                        lambda self, url, dest: downloads.append(url) or real_download(self, url, dest))

    images = task_images("t")

    assert ([(image.id, image.version) for image in images], downloads) == ([("crm-image", 1), ("mail-image", 1)], [])
    refs = {image.id: f"ghcr.io/acme/delivery/{image.id}:1@sha256:" + "e" * 64 for image in images}
    imported = import_task("t", home / "t", image_refs=refs)
    assert [_toml(imported.root, f"envs/{env}/env.toml")["image"]["ref"] for env in ("crm", "crm-eu", "inbox")] == [
        refs["crm-image"], refs["crm-image"], refs["mail-image"]]


def test_listing_a_task_the_import_cant_write_fails_before_any_push_would_start(home):
    _stored(get_artifact_store().put_document(DockerImageArtifact(
        id="crm-image", description="d", image_name="crm-image:v1", tar_gz_s3_url="file:///nowhere/crm.tar.gz")))

    with pytest.raises(BundleError) as caught:
        task_images("t")

    assert any("stored only as a docker save tarball" in problem for problem in caught.value.problems)
