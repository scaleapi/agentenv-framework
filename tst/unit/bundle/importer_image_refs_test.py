"""``bundle init --image-ref``: an image given a registry reference pinned by digest is written as that reference,
which a run pulls, with the part of its stored build context its Dockerfile reads kept beside it."""

import json
import tomllib

import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError
from agent_env.bundle.importer import import_task
from agent_env.bundle.materialize import materialize
from agent_env.bundle.resolve import build_step
from agent_env.cli import cli
from agent_env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.task import Task
from tst.unit.bundle._support import context_image, local_builds, plan_of

REF = "123456789012.dkr.ecr.us-west-2.amazonaws.com/delivery/crm:t1@sha256:" + "c" * 64
DEPLOY = {"id": "env", "type": "deploy_env", "env_id": "crm"}


@pytest.fixture
def home(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


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
    assert "An image *Images* says is pulled is pulled by its digest, never built" in readme
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


@pytest.mark.parametrize("refs, problem", [
    ({"crm-image": REF.split("@")[0]},
     "envs/crm: image: the reference given for crm-image, '" + REF.split("@")[0] + "', doesn't pin a digest, as "
     "<registry>/<repo>:<tag>@sha256:<digest> does"),
    ({"crm-image": "crm@sha256:" + "c" * 64}, "doesn't name its registry"),
    ({"crm-image": REF, "other-image": REF},
     "a reference was given for other-image, which the import writes no image of"),
], ids=["no digest", "no registry", "unused"])
def test_a_ref_that_cant_be_written_fails_the_import_and_writes_nothing(home, refs, problem):
    _stored(context_image(home, "crm-image"))

    with pytest.raises(BundleError) as caught:
        import_task("t", home / "t", image_refs=refs)

    assert any(problem in found for found in caught.value.problems)
    assert not (home / "t").exists()


def test_the_cli_takes_each_ref_as_id_equals_ref(home):
    _stored(context_image(home, "crm-image"))
    runner = CliRunner()

    written = runner.invoke(cli, ["bundle", "init", "--from-task", "t", "--image-ref", f"crm-image={REF}",
                                  str(home / "t")])
    refused = runner.invoke(cli, ["bundle", "init", "--from-task", "t", "--image-ref", REF, str(home / "u")])

    assert written.exit_code == 0, written.output
    assert _toml(home / "t", "envs/crm/env.toml")["image"]["ref"] == REF
    assert refused.exit_code == 2 and "--image-ref takes ID=REF" in refused.output
