"""Integration tests for FileArtifactUniverse version pinning (F2, additive).

These tests run against the configured object and document stores.

They mirror the EnvironmentUniverseArtifact pinning suite, scoped to the additive
change: new docs carry BOTH file_artifact_refs (pinned) and file_artifact_ids
(legacy), get_file_artifacts() prefers refs, and pre-pinning docs still resolve
to latest with a warning.
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_env.artifact import Artifact, FileArtifact, FileArtifactUniverse
from agent_env.artifact.store import ARTIFACTS_COLLECTION
from agent_env.config import get_config
from agent_env.store.document_store import Filter


TEST_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
EMAIL_ARTIFACT_PATH = TEST_DATA_DIR / "email_artifact.json"
SLACK_SAMPLE_PATH = TEST_DATA_DIR / "slack_mcp" / "sample_data.json"
EMAIL_SAMPLE_PATH = TEST_DATA_DIR / "email_mcp" / "sample_data.json"


def _unique_id(prefix: str = "test") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _doc_store():
    return get_config().get_document_store()


def _make_file_artifact(file_path: str, prefix: str) -> FileArtifact:
    return FileArtifact.put(
        id=_unique_id(f"{prefix}_file"),
        description=f"{prefix} data",
        file_path=file_path,
    )


def _insert_legacy_doc(*, universe_id: str, file_artifact_ids: dict[str, str]) -> None:
    """Hand-craft a pre-pinning document (only file_artifact_ids)."""
    _doc_store().insert(
        ARTIFACTS_COLLECTION,
        {
            "id": universe_id,
            "version": 1,
            "type": "file_artifact_universe",
            "file_artifact_ids": file_artifact_ids,
            "created_at_utc": datetime.now(timezone.utc),
        },
    )


@pytest.mark.integration
class TestFileArtifactUniversePinning:

    def test_put_records_both_refs_and_ids(self):
        """New docs carry pinned refs AND legacy ids (additive)."""
        fa1 = _make_file_artifact(str(SLACK_SAMPLE_PATH), "fau_slack")
        fa2 = _make_file_artifact(str(EMAIL_SAMPLE_PATH), "fau_email")

        universe = FileArtifactUniverse.put(
            id=_unique_id("fau_both"),
            file_artifacts={"slack.json": fa1, "email.json": fa2},
        )

        assert universe.file_artifact_refs is not None
        assert universe.file_artifact_refs["slack.json"].id == fa1.id
        assert universe.file_artifact_refs["slack.json"].version == fa1.version
        assert universe.file_artifact_refs["email.json"].id == fa2.id
        assert universe.file_artifact_refs["email.json"].version == fa2.version
        # Legacy ids stay populated for raw-doc readers.
        assert universe.file_artifact_ids == {"slack.json": fa1.id, "email.json": fa2.id}

    def test_put_with_empty_raises(self):
        with pytest.raises(ValueError, match="file_artifacts must be non-empty"):
            FileArtifactUniverse.put(id=_unique_id("fau_empty"), file_artifacts={})

    def test_raw_doc_has_both_keys(self):
        """On-disk doc must contain both file_artifact_refs and file_artifact_ids."""
        fa = _make_file_artifact(str(EMAIL_ARTIFACT_PATH), "fau_raw")
        universe = FileArtifactUniverse.put(
            id=_unique_id("fau_raw"),
            file_artifacts={"data.json": fa},
        )

        raw = _doc_store().find_one(
            ARTIFACTS_COLLECTION, Filter.of(id=universe.id, version=universe.version)
        )
        assert raw is not None
        assert raw["file_artifact_refs"]["data.json"]["id"] == fa.id
        assert raw["file_artifact_refs"]["data.json"]["version"] == fa.version
        assert raw["file_artifact_ids"] == {"data.json": fa.id}

    def test_get_file_artifacts_returns_pinned_after_child_republish(self):
        """After the inner FileArtifact id is bumped, the universe still loads the pinned bytes."""
        fa_v1 = _make_file_artifact(str(SLACK_SAMPLE_PATH), "fau_drift")
        universe = FileArtifactUniverse.put(
            id=_unique_id("fau_drift"),
            file_artifacts={"f.json": fa_v1},
        )
        pinned_version = fa_v1.version
        pinned_bytes = fa_v1.load()

        # Republish the SAME FileArtifact id with different bytes.
        fa_v2 = FileArtifact.put(
            id=fa_v1.id,
            description="v2 mutated",
            file_path=str(EMAIL_SAMPLE_PATH),
        )
        assert fa_v2.version == pinned_version + 1
        assert fa_v2.load() != pinned_bytes

        retrieved = Artifact.get(universe.id, version=universe.version)
        resolved = retrieved.get_file_artifacts()
        assert resolved["f.json"].version == pinned_version
        assert resolved["f.json"].load() == pinned_bytes

    def test_legacy_doc_falls_back_to_latest(self, caplog):
        """A pre-pinning doc (only file_artifact_ids) still loads — via latest, with a warning."""
        fa_v1 = _make_file_artifact(str(SLACK_SAMPLE_PATH), "fau_legacy")

        legacy_id = _unique_id("fau_legacy")
        _insert_legacy_doc(universe_id=legacy_id, file_artifact_ids={"f.json": fa_v1.id})

        retrieved = Artifact.get(legacy_id)
        assert isinstance(retrieved, FileArtifactUniverse)
        assert retrieved.file_artifact_refs is None
        assert retrieved.file_artifact_ids == {"f.json": fa_v1.id}

        fa_v2 = FileArtifact.put(
            id=fa_v1.id,
            description="v2",
            file_path=str(EMAIL_SAMPLE_PATH),
        )

        with caplog.at_level(
            logging.WARNING,
            logger="agent_env.artifact.artifacts.file_artifact_universe",
        ):
            resolved = retrieved.get_file_artifacts()
        assert resolved["f.json"].version == fa_v2.version  # latest, not pinned
        assert any(
            "falling back to latest" in r.message and "file_artifact_ids" in r.message
            for r in caplog.records
        ), "expected legacy-fallback warning to be logged"

    def test_mixed_doc_prefers_refs(self):
        """If both keys are present, refs win over stale ids."""
        fa_v1 = _make_file_artifact(str(SLACK_SAMPLE_PATH), "fau_mixed")

        universe_id = _unique_id("fau_mixed")
        _doc_store().insert(
            ARTIFACTS_COLLECTION,
            {
                "id": universe_id,
                "version": 1,
                "type": "file_artifact_universe",
                "file_artifact_refs": {"f.json": {"id": fa_v1.id, "version": fa_v1.version}},
                "file_artifact_ids": {"f.json": fa_v1.id},  # stale, should be ignored
                "created_at_utc": datetime.now(timezone.utc),
            },
        )

        retrieved = Artifact.get(universe_id)
        FileArtifact.put(
            id=fa_v1.id,
            description="v2",
            file_path=str(EMAIL_SAMPLE_PATH),
        )

        resolved = retrieved.get_file_artifacts()
        assert resolved["f.json"].version == fa_v1.version  # pinned via refs, not latest

    def test_empty_refs_falls_back_to_ids(self, caplog):
        """A degenerate empty-{} refs dict must fall back to ids, not return zero files."""
        fa = _make_file_artifact(str(SLACK_SAMPLE_PATH), "fau_emptyrefs")

        universe_id = _unique_id("fau_emptyrefs")
        _doc_store().insert(
            ARTIFACTS_COLLECTION,
            {
                "id": universe_id,
                "version": 1,
                "type": "file_artifact_universe",
                "file_artifact_refs": {},  # degenerate: present but empty
                "file_artifact_ids": {"f.json": fa.id},
                "created_at_utc": datetime.now(timezone.utc),
            },
        )

        retrieved = Artifact.get(universe_id)
        with caplog.at_level(
            logging.WARNING,
            logger="agent_env.artifact.artifacts.file_artifact_universe",
        ):
            resolved = retrieved.get_file_artifacts()
        assert set(resolved) == {"f.json"}
        assert any("falling back to latest" in r.message for r in caplog.records)

    def test_get_file_artifacts_raises_on_malformed_doc(self):
        """A doc with neither refs nor ids must raise."""
        universe_id = _unique_id("fau_malformed")
        _doc_store().insert(
            ARTIFACTS_COLLECTION,
            {
                "id": universe_id,
                "version": 1,
                "type": "file_artifact_universe",
                "created_at_utc": datetime.now(timezone.utc),
            },
        )

        retrieved = Artifact.get(universe_id)
        with pytest.raises(ValueError, match="malformed FileArtifactUniverse"):
            retrieved.get_file_artifacts()

    def test_put_bundled_pins_versions(self):
        """put_bundled builds versioned FileArtifacts and must pin them via refs."""
        universe = FileArtifactUniverse.put_bundled(
            id=_unique_id("fau_bundled"),
            files={"nested/a.json": EMAIL_ARTIFACT_PATH},
            s3_url=get_config().get_object_store().object_url(f"test/{uuid.uuid4().hex}/"),
        )
        assert universe.file_artifact_refs is not None
        ref = universe.file_artifact_refs["nested/a.json"]
        assert ref.version >= 1
        assert universe.file_artifact_ids["nested/a.json"] == ref.id

    def test_put_existing_pins_versions(self):
        """put_existing wraps files already under an object-store prefix; it must pin too."""
        store = get_config().get_object_store()
        prefix = f"test/{uuid.uuid4().hex}/"
        store.put(f"{prefix}sub/a.json", b'{"a": 1}')
        store.put(f"{prefix}b.json", b'{"b": 2}')

        universe = FileArtifactUniverse.put_existing(
            id=_unique_id("fau_existing"),
            s3_url=store.object_url(prefix),
        )

        assert universe.file_artifact_refs is not None
        assert set(universe.file_artifact_refs) == {"sub/a.json", "b.json"}
        for fname, ref in universe.file_artifact_refs.items():
            assert ref.version >= 1
            assert universe.file_artifact_ids[fname] == ref.id
