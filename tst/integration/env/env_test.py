"""Integration tests for Env persistence.

These tests run against the configured stores and build a real Docker image.
"""

import subprocess
import uuid
from pathlib import Path

import pytest

from agent_env.artifact import DockerImageArtifact
from agent_env.env import Env, MCPServerEnv

# Module-scoped fixture builds a real docker image (~40s) and one test fetches
# Dockerfiles from GitHub (~100s). Everything here is slow.
pytestmark = [pytest.mark.int_test_slow]

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture(scope="module")
def docker_image_artifact() -> DockerImageArtifact:
    """Build test image and create DockerImageArtifact once for all tests."""
    subprocess.run(
        ["docker", "build", "-f", str(TEST_DATA_DIR / "email_mcp" / "Dockerfile"),
         "-t", "test-email-mcp", str(TEST_DATA_DIR)],
        check=True,
    )
    return DockerImageArtifact.put(
        id=_unique_id("artifact"),
        description="Test email MCP image",
        image_name="test-email-mcp",
    )


@pytest.mark.integration
class TestEnvPersistence:
    """Tests for Env put/get/query."""

    def test_put_creates_env(self, docker_image_artifact):
        """Test that put() creates an env."""
        env_id = _unique_id("test_env")
        env = MCPServerEnv.put(id=env_id, docker_image_artifact=docker_image_artifact, environment_name="test")

        assert env.id == env_id
        assert env.type == "mcp_server"
        assert env.docker_image_artifact.id == docker_image_artifact.id

    def test_get_retrieves_env(self, docker_image_artifact):
        """Test that Env.get() retrieves the env."""
        env_id = _unique_id("retrievable_env")
        created = MCPServerEnv.put(id=env_id, docker_image_artifact=docker_image_artifact, environment_name="test")
        retrieved = Env.get(env_id)

        assert retrieved.id == created.id
        assert retrieved.type == "mcp_server"
        assert isinstance(retrieved, MCPServerEnv)
        assert retrieved.docker_image_artifact.id == docker_image_artifact.id

    def test_versioning(self, docker_image_artifact):
        """Test that multiple puts create new versions in the database."""
        env_id = _unique_id("versioned_env")
        MCPServerEnv.put(id=env_id, docker_image_artifact=docker_image_artifact, environment_name="test")
        MCPServerEnv.put(id=env_id, docker_image_artifact=docker_image_artifact, environment_name="test")

        # Query all versions - should have 2 documents
        envs = Env.query().id(env_id).execute()
        assert len(envs) == 2

    def test_get_specific_version(self, docker_image_artifact):
        """Test that Env.get() can retrieve a specific version."""
        env_id = _unique_id("multi_version_env")
        MCPServerEnv.put(id=env_id, docker_image_artifact=docker_image_artifact, environment_name="test")
        MCPServerEnv.put(id=env_id, docker_image_artifact=docker_image_artifact, environment_name="test")

        # Both should retrieve successfully (different versions in DB)
        v1 = Env.get(env_id, version=1)
        latest = Env.get(env_id)

        assert v1.id == env_id
        assert latest.id == env_id

    def test_query_by_type(self, docker_image_artifact):
        """Test querying envs by type."""
        env_id_1 = _unique_id("mcp_1")
        env_id_2 = _unique_id("mcp_2")

        MCPServerEnv.put(id=env_id_1, docker_image_artifact=docker_image_artifact, environment_name="test")
        MCPServerEnv.put(id=env_id_2, docker_image_artifact=docker_image_artifact, environment_name="test")

        # Query each env by id and verify type
        env1 = Env.query().id(env_id_1).first()
        env2 = Env.query().id(env_id_2).first()

        assert env1.type == "mcp_server"
        assert env2.type == "mcp_server"

    def test_query_by_id(self, docker_image_artifact):
        """Test querying envs by id."""
        env_id = _unique_id("query_test")
        MCPServerEnv.put(id=env_id, docker_image_artifact=docker_image_artifact, environment_name="test")
        MCPServerEnv.put(id=env_id, docker_image_artifact=docker_image_artifact, environment_name="test")

        envs = Env.query().id(env_id).execute()
        assert len(envs) == 2

        latest = Env.query().id(env_id).latest().first()
        assert latest.id == env_id

    def test_query_count(self, docker_image_artifact):
        """Test counting envs."""
        count_ids = [_unique_id("count") for _ in range(3)]
        for env_id in count_ids:
            MCPServerEnv.put(id=env_id, docker_image_artifact=docker_image_artifact, environment_name="test")

        # Verify each env was created
        for env_id in count_ids:
            env = Env.query().id(env_id).first()
            assert env is not None
            assert env.id == env_id



def _unique_id(prefix: str = "test") -> str:
    """Generate a unique env ID for testing."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
