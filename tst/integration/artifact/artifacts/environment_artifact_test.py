"""Integration tests for EnvironmentArtifact.

These tests run against the configured object and document stores.
Requires AWS credentials with access to secrets manager and S3.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_env.artifact import Artifact, ArtifactRef, FileArtifact, EnvironmentArtifact
from agent_env.artifact.store import ARTIFACTS_COLLECTION
from agent_env.config import get_config
from agent_env.store import Filter


TEST_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
EMAIL_ARTIFACT_PATH = TEST_DATA_DIR / "email_artifact.json"
EMAIL_SAMPLE_PATH = TEST_DATA_DIR / "email_mcp" / "sample_data.json"


def _unique_id(prefix: str = "test") -> str:
    """Generate a unique artifact ID for testing."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _artifact_docs():
    """Document-store handle for hand-crafting legacy/malformed docs the model won't write."""
    return get_config().get_document_store()


def _insert_legacy_service_doc(*, service_id: str, service_name: str,
                               service_version: int, file_artifact_id: str) -> None:
    """Hand-craft a pre-pinning EnvironmentArtifact document, bypassing the model."""
    doc = {
        "id": service_id,
        "version": 1,
        "type": "environment",
        "service_name": service_name,
        "service_version": service_version,
        "file_artifact_id": file_artifact_id,  # legacy unpinned shape
        "created_at_utc": datetime.now(timezone.utc),
    }
    _artifact_docs().insert(ARTIFACTS_COLLECTION, doc)


@pytest.mark.integration
class TestServiceArtifact:

    def test_put_creates_service_artifact(self):
        file_artifact = FileArtifact.put(
            id=_unique_id("svc_file"),
            description="Email data for service test",
            file_path=str(EMAIL_SAMPLE_PATH),
        )

        service_id = _unique_id("svc_put")
        environment_artifact = EnvironmentArtifact.put(
            id=service_id,
            environment_name="email",
            file_artifact=file_artifact,
        )

        assert environment_artifact.id == service_id
        assert environment_artifact.version == 1
        assert environment_artifact.type == "environment"
        assert environment_artifact.environment_name == "email"
        assert not hasattr(environment_artifact, "service_version")
        assert environment_artifact.file_artifact_ref == ArtifactRef(
            id=file_artifact.id, version=file_artifact.version
        )

    def test_get_retrieves_service_artifact(self):
        file_artifact = FileArtifact.put(
            id=_unique_id("svc_get_file"),
            description="Email data",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        service_id = _unique_id("svc_get")
        created = EnvironmentArtifact.put(
            id=service_id,
            environment_name="slack",
            file_artifact=file_artifact,
        )

        retrieved = Artifact.get(service_id)

        assert isinstance(retrieved, EnvironmentArtifact)
        assert retrieved.id == created.id
        assert retrieved.version == created.version
        assert retrieved.environment_name == "slack"
        assert retrieved.file_artifact_ref == created.file_artifact_ref

    def test_versioning(self):
        service_id = _unique_id("svc_versioned")
        file1 = FileArtifact.put(
            id=_unique_id("svc_ver_file1"),
            description="Email data v1",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        file2 = FileArtifact.put(
            id=_unique_id("svc_ver_file2"),
            description="Email data v2",
            file_path=str(EMAIL_SAMPLE_PATH),
        )

        v1 = EnvironmentArtifact.put(
            id=service_id,
            environment_name="email",
            file_artifact=file1,
        )
        v2 = EnvironmentArtifact.put(
            id=service_id,
            environment_name="email",
            file_artifact=file2,
        )

        assert v1.version == 1
        assert v2.version == 2
        assert v1.file_artifact_ref.id == file1.id
        assert v2.file_artifact_ref.id == file2.id

    def test_get_specific_version(self):
        service_id = _unique_id("svc_multi_ver")
        file1 = FileArtifact.put(
            id=_unique_id("svc_mver_file1"),
            description="Data v1",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        file2 = FileArtifact.put(
            id=_unique_id("svc_mver_file2"),
            description="Data v2",
            file_path=str(EMAIL_SAMPLE_PATH),
        )

        EnvironmentArtifact.put(
            id=service_id,
            environment_name="email",
            file_artifact=file1,
        )
        EnvironmentArtifact.put(
            id=service_id,
            environment_name="email",
            file_artifact=file2,
        )

        v1 = Artifact.get(service_id, version=1)
        latest = Artifact.get(service_id)

        assert v1.version == 1
        assert v1.file_artifact_ref.id == file1.id
        assert latest.version == 2
        assert latest.file_artifact_ref.id == file2.id

    def test_get_file_artifact_loads_data(self):
        file_artifact = FileArtifact.put(
            id=_unique_id("svc_load_file"),
            description="Email data",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        environment_artifact = EnvironmentArtifact.put(
            id=_unique_id("svc_load"),
            environment_name="email",
            file_artifact=file_artifact,
        )

        result = environment_artifact.get_file_artifact()
        assert isinstance(result, FileArtifact)
        data = json.loads(result.load())
        assert data["user_email"] == "agent@skyhaven-support.example"

    # ------------------------------------------------------------------
    # Pinning tests
    # ------------------------------------------------------------------

    def test_put_pins_file_artifact_version(self):
        """The ref must record the FileArtifact's exact version at put time."""
        file_artifact = FileArtifact.put(
            id=_unique_id("svc_pin_file"),
            description="data",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        environment_artifact = EnvironmentArtifact.put(
            id=_unique_id("svc_pin"),
            environment_name="email",
            file_artifact=file_artifact,
        )
        assert environment_artifact.file_artifact_ref is not None
        assert environment_artifact.file_artifact_ref.id == file_artifact.id
        assert environment_artifact.file_artifact_ref.version == file_artifact.version

    def test_get_file_artifact_returns_pinned_version_after_republish(self):
        """A EnvironmentArtifact pinned to FileArtifact v1 must still load v1 after a republish."""
        file_id = _unique_id("svc_drift_file")
        file_v1 = FileArtifact.put(
            id=file_id,
            description="v1",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        service_artifact_v1 = EnvironmentArtifact.put(
            id=_unique_id("svc_drift"),
            environment_name="email",
            file_artifact=file_v1,
        )
        pinned_file_version = file_v1.version

        # Republish the same FileArtifact id with different bytes.
        file_v2 = FileArtifact.put(
            id=file_id,
            description="v2 (mutated)",
            file_path=str(EMAIL_SAMPLE_PATH),
        )
        assert file_v2.version == pinned_file_version + 1

        # Reloading the Day-1 EnvironmentArtifact must still get v1's FileArtifact.
        retrieved = Artifact.get(service_artifact_v1.id, version=service_artifact_v1.version)
        loaded_file = retrieved.get_file_artifact()
        assert loaded_file.version == pinned_file_version

    def test_legacy_doc_falls_back_to_latest(self, caplog):
        """Pre-pinning EnvironmentArtifact (only file_artifact_id) must still load."""
        file_id = _unique_id("svc_legacy_file")
        file_v1 = FileArtifact.put(
            id=file_id,
            description="v1",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        service_id = _unique_id("svc_legacy")
        _insert_legacy_service_doc(
            service_id=service_id,
            service_name="email",
            service_version=1,
            file_artifact_id=file_id,
        )

        retrieved = Artifact.get(service_id)
        assert isinstance(retrieved, EnvironmentArtifact)
        assert retrieved.legacy_file_artifact_id == file_id
        assert retrieved.file_artifact_ref is None

        # Republish FileArtifact -> legacy EnvironmentArtifact resolves to latest.
        file_v2 = FileArtifact.put(
            id=file_id,
            description="v2",
            file_path=str(EMAIL_SAMPLE_PATH),
        )

        with caplog.at_level(logging.WARNING, logger=EnvironmentArtifact.__module__):
            loaded = retrieved.get_file_artifact()
        assert loaded.version == file_v2.version  # latest, not pinned
        assert any(
            "falling back to latest" in r.message and "file_artifact_id" in r.message
            for r in caplog.records
        )

    def test_get_file_artifact_raises_on_malformed_doc(self):
        """A EnvironmentArtifact doc with neither ref nor legacy id must raise."""
        service_id = _unique_id("svc_malformed")
        doc = {
            "id": service_id,
            "version": 1,
            "type": "environment",
            "service_name": "email",
            "service_version": 1,
            "created_at_utc": datetime.now(timezone.utc),
        }
        _artifact_docs().insert(ARTIFACTS_COLLECTION, doc)

        retrieved = Artifact.get(service_id)
        with pytest.raises(ValueError, match="malformed EnvironmentArtifact"):
            retrieved.get_file_artifact()

    def test_new_doc_omits_legacy_file_artifact_id(self):
        """New docs must not write the legacy file_artifact_id key."""
        file_artifact = FileArtifact.put(
            id=_unique_id("svc_clean_file"),
            description="data",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        environment_artifact = EnvironmentArtifact.put(
            id=_unique_id("svc_clean"),
            environment_name="email",
            file_artifact=file_artifact,
        )

        raw = _artifact_docs().find_one(
            ARTIFACTS_COLLECTION, Filter.of(id=environment_artifact.id, version=environment_artifact.version)
        )
        assert raw is not None
        assert "file_artifact_id" not in raw
        assert "file_artifact_ref" in raw
        assert raw["file_artifact_ref"]["id"] == file_artifact.id
        assert raw["file_artifact_ref"]["version"] == file_artifact.version

    def test_legacy_dump_preserves_legacy_key(self):
        """model_dump on a legacy-loaded EnvironmentArtifact must keep file_artifact_id."""
        file_id = _unique_id("svc_legacy_dump_file")
        FileArtifact.put(
            id=file_id,
            description="v1",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        service_id = _unique_id("svc_legacy_dump")
        _insert_legacy_service_doc(
            service_id=service_id,
            service_name="email",
            service_version=1,
            file_artifact_id=file_id,
        )

        retrieved = Artifact.get(service_id)
        dumped = retrieved.model_dump()
        assert dumped["file_artifact_id"] == file_id
        assert "legacy_file_artifact_id" not in dumped

        json_dumped = json.loads(retrieved.model_dump_json())
        assert json_dumped["file_artifact_id"] == file_id
        assert "legacy_file_artifact_id" not in json_dumped

    def test_mixed_doc_prefers_ref(self):
        """If both file_artifact_ref and legacy file_artifact_id are present, ref wins."""
        file_id = _unique_id("svc_mixed_file")
        file_v1 = FileArtifact.put(
            id=file_id,
            description="v1",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        service_id = _unique_id("svc_mixed")
        doc = {
            "id": service_id,
            "version": 1,
            "type": "environment",
            "service_name": "email",
            "service_version": 1,
            "file_artifact_ref": {"id": file_v1.id, "version": file_v1.version},
            "file_artifact_id": file_v1.id,  # stale, should be ignored
            "created_at_utc": datetime.now(timezone.utc),
        }
        _artifact_docs().insert(ARTIFACTS_COLLECTION, doc)

        retrieved = Artifact.get(service_id)
        # Publish a newer FileArtifact under the same id; refs should still pin v1.
        FileArtifact.put(
            id=file_id,
            description="v2",
            file_path=str(EMAIL_SAMPLE_PATH),
        )

        loaded = retrieved.get_file_artifact()
        assert loaded.version == file_v1.version, (
            "ref-and-legacy mixed doc must resolve via the ref, not the legacy id"
        )

    def test_new_doc_json_serialization_omits_legacy_key(self):
        """model_dump_json (Rust path) must also drop the legacy key."""
        file_artifact = FileArtifact.put(
            id=_unique_id("svc_json_file"),
            description="data",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        environment_artifact = EnvironmentArtifact.put(
            id=_unique_id("svc_json"),
            environment_name="email",
            file_artifact=file_artifact,
        )

        data = json.loads(environment_artifact.model_dump_json())
        assert "file_artifact_id" not in data
        assert data["file_artifact_ref"]["id"] == file_artifact.id
        assert data["file_artifact_ref"]["version"] == file_artifact.version
