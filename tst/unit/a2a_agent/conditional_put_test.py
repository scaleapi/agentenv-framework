"""A2AAgent.put with expected_version: it lands only while that version is still the agent's latest."""

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.store.base import ConcurrentModificationError


@pytest.fixture
def image(local_stores):
    return get_artifact_store().put_document(DockerImageArtifact(
        id="agent__agent_image", description="d", image_name="ghcr.io/acme/agent:1", tar_gz_s3_url="s3://b/x"))


def test_a_put_naming_the_latest_version_lands_on_the_next(image):
    assert A2AAgent.put(id="agent", docker_image_artifact=image, expected_version=0).version == 1
    assert A2AAgent.put(id="agent", docker_image_artifact=image, expected_version=1).version == 2


def test_a_put_naming_an_older_version_writes_nothing(image):
    A2AAgent.put(id="agent", docker_image_artifact=image)
    A2AAgent.put(id="agent", docker_image_artifact=image, metadata={"by": "other"})

    with pytest.raises(ConcurrentModificationError, match="already has a version 2"):
        A2AAgent.put(id="agent", docker_image_artifact=image, expected_version=1, metadata={"by": "stale"})

    latest = A2AAgent.get("agent")
    assert latest.version == 2 and latest.metadata["by"] == "other"
