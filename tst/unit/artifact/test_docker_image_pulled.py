"""An image with no tar.gz: read, written back, and loadable only when its name names the registry to pull it from."""

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact

REF = "ghcr.io/example/image@sha256:" + "0" * 64


@pytest.mark.parametrize("document", [
    {"id": "img", "version": 1, "type": "docker_image", "description": "d", "image_name": REF},
    {"id": "img", "version": 1, "type": "docker_image", "description": "d", "image_name": REF,
     "tar_gz_s3_url": None, "tar_gz_object_url": None},
])
def test_a_document_with_no_tar_gz_is_read(document):
    image = DockerImageArtifact.model_validate(document)

    assert image.tar_gz_object_url is None
    assert image.load_problem() is None


def test_one_is_written_with_both_spellings_of_its_tar_gz_empty_and_reads_back_the_same():
    image = DockerImageArtifact(id="img", version=1, description="d", image_name=REF)

    dumped = image.model_dump()

    assert dumped["tar_gz_s3_url"] is None and dumped["tar_gz_object_url"] is None
    assert DockerImageArtifact.model_validate(dumped) == image


@pytest.mark.parametrize("image_name, tar_gz, problem", [
    ("img:v1", "s3://bucket/img.tar.gz", None),
    (REF, None, None),
    ("localhost:5000/img:v1", None, None),
    ("img:v1", None, "'img' v1 has no tar.gz, and its image name 'img:v1' doesn't name a registry to pull it from"),
    ("team/img:v1", None, "'img' v1 has no tar.gz, and its image name 'team/img:v1' doesn't name a registry to pull it "
                          "from"),
])
def test_an_image_loads_from_its_tar_gz_or_by_pulling_a_name_that_names_its_registry(image_name, tar_gz, problem):
    image = DockerImageArtifact(id="img", version=1, description="d", image_name=image_name, tar_gz_object_url=tar_gz)

    assert image.load_problem() == problem


@pytest.mark.parametrize("image_name", ["localhost:5000/img:v1", "127.0.0.1:5000/img:v1"])
def test_a_sandbox_elsewhere_cant_pull_from_this_machines_registry(image_name):
    image = DockerImageArtifact(id="img", version=1, description="d", image_name=image_name)
    here = f"{image_name} is in a registry on this machine, which a sandbox elsewhere can't pull from"

    assert (image.load_problem(), image.by_name_problem()) == (None, None)
    assert image.load_problem(on_this_machine=False) == image.by_name_problem(on_this_machine=False) == here


def test_a_vm_elsewhere_loads_the_tar_gz_of_an_image_named_for_this_machines_registry():
    image = DockerImageArtifact(id="img", version=1, description="d", image_name="localhost:5000/img:v1",
                                tar_gz_object_url="file:///state/img.tar.gz")

    assert image.load_problem(on_this_machine=False) is None


def test_one_with_no_tar_gz_has_no_bytes_to_load():
    with pytest.raises(ValueError, match="has no tar.gz; its image is pulled from ghcr.io/example/image@sha256:"):
        DockerImageArtifact(id="img", version=1, description="d", image_name=REF).load()


def test_put_tar_refuses_an_empty_tar_gz():
    with pytest.raises(ValueError, match="tar_gz_object_url is empty"):
        DockerImageArtifact.put_tar("img", description="d", image_name="img:1", tar_gz_object_url="")
