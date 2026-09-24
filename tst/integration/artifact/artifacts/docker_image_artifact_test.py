"""Integration tests for DockerImageArtifact.

These tests run against the configured object, image and document stores.
"""

import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

from agent_env.artifact import DockerImageArtifact
from agent_env.config import get_config

# Builds a real Docker image, uploads ~70 MB to the object store, pushes to the image store (~80s each).
pytestmark = [pytest.mark.int_test_slow]


# Path to test data
TEST_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _unique_id(prefix: str = "test") -> str:
    """Generate a unique artifact ID for testing."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.mark.integration
class TestDockerImageArtifact:
    """Tests for DockerImageArtifact."""

    def test_put_creates_artifact(self):
        """Test that put() creates an artifact from a local Docker image."""
        # Build the test image
        subprocess.run(
            ["docker", "build", "-f", str(TEST_DATA_DIR / "email_mcp" / "Dockerfile"),
             "-t", "test-email-mcp", str(TEST_DATA_DIR)],
            check=True,
        )

        artifact_id = _unique_id("docker_image")
        artifact = DockerImageArtifact.put(
            id=artifact_id,
            description="Test email MCP image",
            image_name="test-email-mcp",
        )

        assert artifact.id == artifact_id
        assert artifact.version == 1
        assert artifact.type == "docker_image"
        # put() retags + pushes to the image store; image_name is the registry URI, not the local tag.
        assert artifact.image_name.endswith(f":v{artifact.version}")
        assert artifact_id in artifact.image_name
        assert get_config().get_object_store().get_object_metadata_at(artifact.tar_gz_object_url) is not None

    def test_load_and_import_to_docker(self):
        """Test that load() returns tar.gz that can be imported back into Docker."""
        # Build and put with a unique local tag (put() retags it for the image store).
        local_tag = f"test-email-mcp-load-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            ["docker", "build", "-f", str(TEST_DATA_DIR / "email_mcp" / "Dockerfile"),
             "-t", local_tag, str(TEST_DATA_DIR)],
            check=True,
        )
        artifact = DockerImageArtifact.put(
            id=_unique_id("docker_load"),
            description="Test load",
            image_name=local_tag,
        )

        # The artifact's recorded name is the registry URI; the tar.gz contains that tag.
        loaded_tag = artifact.image_name

        # Remove both local copies so docker load is the only way to recover the image.
        subprocess.run(["docker", "rmi", local_tag], capture_output=True)
        subprocess.run(["docker", "rmi", loaded_tag], capture_output=True)

        # Download the tar.gz
        data = artifact.load()
        assert data[:2] == b'\x1f\x8b'  # Verify gzip magic bytes

        # Load it back into Docker
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            # gunzip and docker load
            result = subprocess.run(
                f"gunzip -c {tmp_path} | docker load",
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            assert "Loaded image" in result.stdout

            # Verify the image exists under the registry tag that put() recorded.
            result = subprocess.run(
                ["docker", "images", "-q", loaded_tag],
                capture_output=True,
                text=True,
                check=True,
            )
            assert result.stdout.strip() != "", f"Image {loaded_tag} not found after docker load"
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            subprocess.run(["docker", "rmi", loaded_tag], capture_output=True)
