"""An artifact version's metadata: written with it, then changed by compare-and-swap without touching its versioned
fields, merging with a concurrent writer's keys rather than overwriting them. An environment universe, whose stored
metadata key holds its legacy metadata file ids, keeps none."""

import pytest

from agent_env.artifact import Artifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.store import ConcurrentModificationError, NotFoundError


def test_an_environment_universe_reads_its_metadata_key_as_its_legacy_file_ids_and_keeps_no_artifact_metadata(
        local_stores):
    legacy = get_artifact_store().put_document(EnvironmentUniverseArtifact.model_validate(
        {"id": "universe", "description": "", "service_artifact_ids": ["env"], "metadata": {"readme": "file-1"}}))

    stored = Artifact.get("universe", legacy.version)

    assert (stored.legacy_metadata, stored.metadata) == ({"readme": "file-1"}, {})
    with pytest.raises(TypeError, match="keeps no artifact metadata"):
        stored.merge_metadata({"validated": {"x": 1}})
    assert Artifact.get("universe", legacy.version).legacy_metadata == {"readme": "file-1"}


def _image(**metadata) -> DockerImageArtifact:
    return get_artifact_store().put_document(DockerImageArtifact(
        id="desktop-image", description="a desktop", image_name="docker.io/library/ubuntu:24.04", metadata=metadata))


def test_metadata_is_stored_with_the_artifact_and_merged_without_a_new_version(local_stores):
    image = _image(source="put")
    image.merge_metadata({"validated_images": {"container": {"amd64": {"status": "passed"}}}})
    stored = Artifact.get("desktop-image", image.version)
    assert stored.version == image.version == 1 and stored.image_name == image.image_name
    assert stored.metadata["source"] == "put" and stored.metadata["validated_images"]["container"]["amd64"] == {
        "status": "passed"}
    assert stored.metadata == image.metadata and "updated_at" in stored.metadata
    assert get_artifact_store().next_version("desktop-image") == 2


def test_a_merge_after_another_writer_keeps_both_keys(local_stores):
    mine = _image()
    theirs = Artifact.get("desktop-image", mine.version)
    theirs.merge_metadata({"validated_boot": {"container": "passed"}})
    mine.merge_metadata({"validated_images": {"container": "passed"}})
    stored = Artifact.get("desktop-image", mine.version).metadata
    assert stored["validated_boot"] == {"container": "passed"} and stored["validated_images"] == {"container": "passed"}


def test_a_stale_update_and_a_missing_version_are_refused(local_stores):
    image = _image()
    store = get_artifact_store()
    fresh = store.update_metadata(image.id, image.version, image.metadata, {"a": 1})
    store.update_metadata(image.id, image.version, fresh, {"a": 2})
    with pytest.raises(ConcurrentModificationError):
        store.update_metadata(image.id, image.version, fresh, {"a": 3})
    with pytest.raises(NotFoundError):
        store.update_metadata(image.id, 9, {}, {"a": 1})
    with pytest.raises(ValueError, match="isn't stored yet"):
        DockerImageArtifact(id="unsaved", description="", image_name="docker.io/library/x:1").merge_metadata({"a": 1})
