"""A2AAgent.put with a version: it lands on exactly that version, which must be the agent's next."""

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.store.base import ConcurrentModificationError


@pytest.fixture
def image(local_stores):
    return get_artifact_store().put_document(DockerImageArtifact(
        id="agent__agent_image", description="d", image_name="ghcr.io/acme/agent:1", tar_gz_s3_url="s3://b/x"))


def test_a_put_naming_the_next_version_lands_on_it(image):
    assert A2AAgent.put(id="agent", docker_image_artifact=image, version=1).version == 1
    assert A2AAgent.put(id="agent", docker_image_artifact=image, version=2).version == 2


@pytest.mark.parametrize("version", [2, 5])
def test_a_put_naming_another_version_writes_nothing(image, version):
    A2AAgent.put(id="agent", docker_image_artifact=image)
    A2AAgent.put(id="agent", docker_image_artifact=image, metadata={"by": "other"})

    with pytest.raises(ConcurrentModificationError, match=f"at version 2, so its next version is 3, not {version}"):
        A2AAgent.put(id="agent", docker_image_artifact=image, version=version, metadata={"by": "stale"})

    latest = A2AAgent.get("agent")
    assert latest.version == 2 and latest.metadata["by"] == "other"
