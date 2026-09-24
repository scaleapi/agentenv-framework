"""Integration tests for FileArtifact.

These tests run against the configured object and document stores.
"""

import json
import uuid
from pathlib import Path

import pytest

from agent_env.artifact import Artifact, FileArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.config import get_config
from agent_env.store import ObjectAlreadyExistsError


# Path to test data
TEST_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
EMAIL_ARTIFACT_PATH = TEST_DATA_DIR / "email_artifact.json"


def _unique_id(prefix: str = "test") -> str:
    """Generate a unique artifact ID for testing."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.mark.integration
class TestFileArtifact:
    """Tests for FileArtifact."""

    def test_put_creates_artifact(self):
        """Test that put() creates an artifact from a file."""
        artifact_id = _unique_id("test_emails")
        artifact = FileArtifact.put(
            id=artifact_id,
            description="Test email data",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )

        assert artifact.id == artifact_id
        assert artifact.version == 1
        assert artifact.type == "file"
        assert artifact.description == "Test email data"
        assert artifact.filename == "email_artifact.json"
        assert artifact.content_type == "application/json"
        assert get_config().get_object_store().get_object_metadata_at(artifact.object_url) is not None

    def test_load_returns_file_contents(self):
        """Test that load() returns the original file contents."""
        artifact_id = _unique_id("test_emails")
        artifact = FileArtifact.put(
            id=artifact_id,
            description="Test email data",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        loaded_data = artifact.load()

        # Verify we can parse it as JSON
        parsed = json.loads(loaded_data)
        assert parsed["user_email"] == "agent@skyhaven-support.example"
        assert len(parsed["emails"]) == 3
        assert len(parsed["contacts"]) == 15

    def test_versioning(self):
        """Test that multiple puts create new versions."""
        artifact_id = _unique_id("versioned_artifact")
        artifact_v1 = FileArtifact.put(
            id=artifact_id,
            description="Version 1",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        artifact_v2 = FileArtifact.put(
            id=artifact_id,
            description="Version 2",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )

        assert artifact_v1.version == 1
        assert artifact_v2.version == 2
        assert artifact_v1.description == "Version 1"
        assert artifact_v2.description == "Version 2"

    def test_get_retrieves_artifact(self):
        """Test that Artifact.get() retrieves the artifact."""
        artifact_id = _unique_id("retrievable")
        created = FileArtifact.put(
            id=artifact_id,
            description="Test retrieval",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        retrieved = Artifact.get(artifact_id)

        assert retrieved.id == created.id
        assert retrieved.version == created.version
        assert isinstance(retrieved, FileArtifact)

    def test_get_specific_version(self):
        """Test that Artifact.get() can retrieve a specific version."""
        artifact_id = _unique_id("multi_version")
        FileArtifact.put(
            id=artifact_id,
            description="Version 1",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        FileArtifact.put(
            id=artifact_id,
            description="Version 2",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )

        v1 = Artifact.get(artifact_id, version=1)
        latest = Artifact.get(artifact_id)

        assert v1.version == 1
        assert v1.description == "Version 1"
        assert latest.version == 2
        assert latest.description == "Version 2"

    def test_put_object_prevents_overwrite(self):
        """Test that put_object raises ObjectAlreadyExistsError if object already exists.

        This prevents the attack where a user could:
        1. Upload artifact (object put succeeds, document put succeeds)
        2. Modify file locally
        3. Upload again with same id/version (object store overwrites, document store fails on duplicate)
        Result: object corrupted but the document still references the "old" version.
        """
        artifact_id = _unique_id("overwrite_test")
        store = get_artifact_store()

        # First upload should succeed
        store.put_object(
            artifact_type="file",
            id=artifact_id,
            version=1,
            object_name="test.json",
            data=b'{"original": true}',
            content_type="application/json",
        )

        # Second upload to same key should fail
        with pytest.raises(ObjectAlreadyExistsError) as exc_info:
            store.put_object(
                artifact_type="file",
                id=artifact_id,
                version=1,
                object_name="test.json",
                data=b'{"modified": true}',
                content_type="application/json",
            )

        assert "already exists" in str(exc_info.value)
        assert artifact_id in str(exc_info.value)
