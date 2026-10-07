"""Integration tests for EnvironmentUniverseArtifact.

These tests run against the configured object and document stores.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_env.artifact import Artifact, ArtifactRef, FileArtifact, EnvironmentArtifact, EnvironmentUniverseArtifact
from agent_env.artifact.store import ARTIFACTS_COLLECTION
from agent_env.config import get_config
from agent_env.store import Filter


def _artifact_docs():
    """Document-store handle for hand-crafting legacy/malformed docs the model won't write."""
    return get_config().get_document_store()


# Path to test data
TEST_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
EMAIL_ARTIFACT_PATH = TEST_DATA_DIR / "email_artifact.json"
SLACK_SAMPLE_PATH = TEST_DATA_DIR / "slack_mcp" / "sample_data.json"
EMAIL_SAMPLE_PATH = TEST_DATA_DIR / "email_mcp" / "sample_data.json"


def _unique_id(prefix: str = "test") -> str:
    """Generate a unique artifact ID for testing."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _make_service_artifact(file_path: str, service_name: str, prefix: str) -> EnvironmentArtifact:
    file_artifact = FileArtifact.put(
        id=_unique_id(f"{prefix}_file"),
        description=f"{service_name} data",
        file_path=file_path,
    )
    return EnvironmentArtifact.put(
        id=_unique_id(prefix),
        environment_name=service_name,
        file_artifact=file_artifact,
    )


def _refs_to_id_set(refs: list[ArtifactRef]) -> set[str]:
    return {r.id for r in refs}


def _insert_legacy_doc(*, universe_id: str, service_artifact_ids: list[str],
                       metadata: dict[str, str] | None = None) -> None:
    """Hand-craft a pre-pinning document, bypassing the model."""
    doc = {
        "id": universe_id,
        "version": 1,
        "type": "environment_universe",
        "service_artifact_ids": service_artifact_ids,
        "created_at_utc": datetime.now(timezone.utc),
    }
    if metadata is not None:
        doc["metadata"] = metadata
    _artifact_docs().insert(ARTIFACTS_COLLECTION, doc)


@pytest.mark.integration
class TestServiceUniverseArtifact:

    def test_put_with_multiple_artifacts(self):
        slack = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_slack")
        email = _make_service_artifact(str(EMAIL_SAMPLE_PATH), "email", "su_email")

        universe_id = _unique_id("su_all")
        universe = EnvironmentUniverseArtifact.put(
            id=universe_id,
            environment_artifacts=[slack, email],
        )

        assert universe.id == universe_id
        assert universe.version == 1
        assert universe.type == "environment_universe"
        assert slack.id in _refs_to_id_set(universe.environment_artifact_refs)
        assert email.id in _refs_to_id_set(universe.environment_artifact_refs)
        assert len(universe.environment_artifact_refs) == 2

    def test_put_with_empty_list_raises(self):
        with pytest.raises(ValueError, match="environment_artifacts must be non-empty"):
            EnvironmentUniverseArtifact.put(
                id=_unique_id("su_empty"),
                environment_artifacts=[],
            )



    def test_versioning(self):
        universe_id = _unique_id("su_versioned")

        email_v1 = _make_service_artifact(str(EMAIL_ARTIFACT_PATH), "email", "su_v1_email")
        v1 = EnvironmentUniverseArtifact.put(id=universe_id, environment_artifacts=[email_v1])

        slack_v2 = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_v2_slack")
        v2 = EnvironmentUniverseArtifact.put(
            id=universe_id,
            environment_artifacts=[email_v1, slack_v2],
        )

        assert v1.version == 1
        assert v2.version == 2
        assert len(v1.environment_artifact_refs) == 1
        assert len(v2.environment_artifact_refs) == 2

    def test_get_retrieves_universe(self):
        email = _make_service_artifact(str(EMAIL_ARTIFACT_PATH), "email", "su_get_email")
        universe_id = _unique_id("su_get")
        created = EnvironmentUniverseArtifact.put(
            id=universe_id,
            environment_artifacts=[email],
        )

        retrieved = Artifact.get(universe_id)

        assert isinstance(retrieved, EnvironmentUniverseArtifact)
        assert retrieved.id == created.id
        assert retrieved.version == created.version
        assert [r.id for r in retrieved.environment_artifact_refs] == [email.id]

    def test_get_specific_version(self):
        universe_id = _unique_id("su_multi_ver")
        email = _make_service_artifact(str(EMAIL_ARTIFACT_PATH), "email", "su_ver_email")
        EnvironmentUniverseArtifact.put(id=universe_id, environment_artifacts=[email])

        slack = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_ver_slack")
        EnvironmentUniverseArtifact.put(
            id=universe_id,
            environment_artifacts=[email, slack],
        )

        v1 = Artifact.get(universe_id, version=1)
        latest = Artifact.get(universe_id)

        assert v1.version == 1
        assert len(v1.environment_artifact_refs) == 1
        assert latest.version == 2
        assert len(latest.environment_artifact_refs) == 2

    def test_get_service_artifacts_loads_data(self):
        slack = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_load_slack")
        email = _make_service_artifact(str(EMAIL_SAMPLE_PATH), "email", "su_load_email")
        universe = EnvironmentUniverseArtifact.put(
            id=_unique_id("su_load"),
            environment_artifacts=[slack, email],
        )

        results = universe.get_environment_artifacts()
        assert len(results) == 2
        assert all(isinstance(r, EnvironmentArtifact) for r in results)

        service_names = {r.environment_name for r in results}
        assert service_names == {"slack", "email"}

    # -----------------------------------------------------------------
    # Version pinning tests
    # -----------------------------------------------------------------

    def test_put_pins_child_versions(self):
        """Refs must record the exact version of each child at put time."""
        slack_v1 = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_pin_slack")
        email_v1 = _make_service_artifact(str(EMAIL_SAMPLE_PATH), "email", "su_pin_email")

        universe = EnvironmentUniverseArtifact.put(
            id=_unique_id("su_pin"),
            environment_artifacts=[slack_v1, email_v1],
        )

        refs_by_id = {r.id: r.version for r in universe.environment_artifact_refs}
        assert refs_by_id[slack_v1.id] == slack_v1.version
        assert refs_by_id[email_v1.id] == email_v1.version

    def test_get_service_artifacts_returns_pinned_after_child_republish(self):
        """Even after a newer EnvironmentArtifact is published, the universe loads the pinned one."""
        slack_v1 = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_drift_slack")
        universe = EnvironmentUniverseArtifact.put(
            id=_unique_id("su_drift"),
            environment_artifacts=[slack_v1],
        )
        pinned_version = slack_v1.version

        # Publish a new version of the same EnvironmentArtifact id.
        new_file = FileArtifact.put(
            id=_unique_id("su_drift_file_v2"),
            description="v2 data",
            file_path=str(SLACK_SAMPLE_PATH),
        )
        slack_v2 = EnvironmentArtifact.put(
            id=slack_v1.id,  # same id -> new version
            environment_name="slack",
            file_artifact=new_file,
        )
        assert slack_v2.version == pinned_version + 1

        # Reloading the universe must yield the originally-pinned version.
        retrieved = Artifact.get(universe.id, version=universe.version)
        children = retrieved.get_environment_artifacts()
        assert len(children) == 1
        assert children[0].version == pinned_version

    def test_new_doc_omits_legacy_environment_artifact_ids(self):
        """New universes must not contain the legacy `service_artifact_ids` key."""
        email = _make_service_artifact(str(EMAIL_ARTIFACT_PATH), "email", "su_clean_email")
        universe = EnvironmentUniverseArtifact.put(
            id=_unique_id("su_clean"),
            environment_artifacts=[email],
        )

        raw = _artifact_docs().find_one(
            ARTIFACTS_COLLECTION, Filter.of(id=universe.id, version=universe.version)
        )
        assert raw is not None
        assert "service_artifact_ids" not in raw, (
            "new docs must not write the legacy service_artifact_ids field"
        )
        assert "service_artifact_refs" in raw
        assert len(raw["service_artifact_refs"]) == 1
        assert raw["service_artifact_refs"][0]["id"] == email.id
        assert raw["service_artifact_refs"][0]["version"] == email.version

    def test_legacy_doc_falls_back_to_latest(self, caplog):
        """A pre-pinning doc (only service_artifact_ids) must still load — via latest resolution."""
        slack_v1 = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_legacy_slack")

        legacy_universe_id = _unique_id("su_legacy")
        _insert_legacy_doc(
            universe_id=legacy_universe_id,
            service_artifact_ids=[slack_v1.id],
        )

        retrieved = Artifact.get(legacy_universe_id)
        assert isinstance(retrieved, EnvironmentUniverseArtifact)
        assert retrieved.legacy_environment_artifact_ids == [slack_v1.id]
        assert retrieved.environment_artifact_refs is None

        new_file = FileArtifact.put(
            id=_unique_id("su_legacy_file_v2"),
            description="v2 data",
            file_path=str(SLACK_SAMPLE_PATH),
        )
        slack_v2 = EnvironmentArtifact.put(
            id=slack_v1.id,
            environment_name="slack",
            file_artifact=new_file,
        )

        with caplog.at_level(logging.WARNING, logger=EnvironmentUniverseArtifact.__module__):
            children = retrieved.get_environment_artifacts()
        assert len(children) == 1
        assert children[0].version == slack_v2.version  # latest, not pinned
        assert any(
            "falling back to latest" in r.message and "service_artifact_ids" in r.message
            for r in caplog.records
        ), "expected legacy-fallback warning to be logged"

    def test_legacy_doc_metadata_falls_back_to_latest(self, caplog):
        """A pre-pinning doc with only `metadata: dict[str, str]` must still resolve to latest."""
        slack = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_legacy_md_slack")
        md_file_v1 = FileArtifact.put(
            id=_unique_id("su_legacy_md_file"),
            description="md v1",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )

        legacy_universe_id = _unique_id("su_legacy_md")
        _insert_legacy_doc(
            universe_id=legacy_universe_id,
            service_artifact_ids=[slack.id],
            metadata={"config": md_file_v1.id},
        )

        retrieved = Artifact.get(legacy_universe_id)
        assert retrieved.legacy_metadata == {"config": md_file_v1.id}
        assert retrieved.metadata_refs is None

        md_v2 = FileArtifact.put(
            id=md_file_v1.id,
            description="md v2",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )

        with caplog.at_level(logging.WARNING, logger=EnvironmentUniverseArtifact.__module__):
            loaded = retrieved.get_metadata()
        assert loaded["config"].version == md_v2.version  # latest, not pinned
        assert any(
            "falling back to latest" in r.message and "metadata" in r.message
            for r in caplog.records
        ), "expected legacy-fallback warning for metadata"

    def test_legacy_dump_preserves_legacy_keys(self):
        """model_dump on a legacy-loaded universe must keep legacy keys."""
        slack = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_legacy_dump_slack")
        md_file = FileArtifact.put(
            id=_unique_id("su_legacy_dump_md"),
            description="md",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        universe_id = _unique_id("su_legacy_dump")
        _insert_legacy_doc(
            universe_id=universe_id,
            service_artifact_ids=[slack.id],
            metadata={"config": md_file.id},
        )

        retrieved = Artifact.get(universe_id)
        dumped = retrieved.model_dump()
        assert dumped["service_artifact_ids"] == [slack.id]
        assert dumped["metadata"] == {"config": md_file.id}

        # Same invariant on the JSON path.
        json_dumped = json.loads(retrieved.model_dump_json())
        assert json_dumped["service_artifact_ids"] == [slack.id]
        assert json_dumped["metadata"] == {"config": md_file.id}

    def test_new_doc_dump_omits_legacy_keys(self):
        """model_dump on a newly-put universe must drop legacy keys (Python path)."""
        email = _make_service_artifact(str(EMAIL_ARTIFACT_PATH), "email", "su_new_dump_email")
        universe = EnvironmentUniverseArtifact.put(
            id=_unique_id("su_new_dump"),
            environment_artifacts=[email],
        )

        dumped = universe.model_dump()
        assert "service_artifact_ids" not in dumped
        assert "metadata" not in dumped
        assert len(dumped["service_artifact_refs"]) == 1

    def test_new_doc_json_serialization_omits_legacy_keys(self):
        """model_dump_json (Rust path) must also drop legacy keys."""
        email = _make_service_artifact(str(EMAIL_ARTIFACT_PATH), "email", "su_json_email")
        universe = EnvironmentUniverseArtifact.put(
            id=_unique_id("su_json"),
            environment_artifacts=[email],
        )

        data = json.loads(universe.model_dump_json())
        assert "service_artifact_ids" not in data, (
            "model_dump_json must drop legacy service_artifact_ids"
        )
        assert "metadata" not in data, "model_dump_json must drop legacy metadata"
        assert data["service_artifact_refs"][0]["id"] == email.id
        assert data["service_artifact_refs"][0]["version"] == email.version

    def test_get_service_artifacts_raises_on_malformed_doc(self):
        """A doc with neither service_artifact_refs nor service_artifact_ids must raise."""
        universe_id = _unique_id("su_malformed")
        doc = {
            "id": universe_id,
            "version": 1,
            "type": "environment_universe",
            "created_at_utc": datetime.now(timezone.utc),
        }
        _artifact_docs().insert(ARTIFACTS_COLLECTION, doc)

        retrieved = Artifact.get(universe_id)
        with pytest.raises(ValueError, match="malformed EnvironmentUniverseArtifact"):
            retrieved.get_environment_artifacts()

    def test_legacy_alias_round_trips_on_disk_key(self):
        """Pydantic alias keeps the on-disk key as `metadata` (not `legacy_metadata`)."""
        slack = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_alias_slack")
        md_file = FileArtifact.put(
            id=_unique_id("su_alias_md_file"),
            description="md",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        universe_id = _unique_id("su_alias")
        _insert_legacy_doc(
            universe_id=universe_id,
            service_artifact_ids=[slack.id],
            metadata={"config": md_file.id},
        )

        # Raw document read: keys are `metadata` and `service_artifact_ids`.
        raw = _artifact_docs().find_one(ARTIFACTS_COLLECTION, Filter.of(id=universe_id, version=1))
        assert "metadata" in raw
        assert "service_artifact_ids" in raw
        assert "legacy_metadata" not in raw
        assert "legacy_environment_artifact_ids" not in raw

        # Python attribute access goes via the legacy_-prefixed names.
        retrieved = Artifact.get(universe_id)
        assert retrieved.legacy_metadata == {"config": md_file.id}
        assert retrieved.legacy_environment_artifact_ids == [slack.id]

        # model_dump round-trips via the alias key, not the field name.
        dumped = retrieved.model_dump()
        assert "metadata" in dumped
        assert "service_artifact_ids" in dumped
        assert "legacy_metadata" not in dumped
        assert "legacy_environment_artifact_ids" not in dumped

    def test_end_to_end_pinning_through_service_to_file(self):
        """Republishing FileArtifact and EnvironmentArtifact under the same id must not change what an old universe loads."""
        file_id = _unique_id("e2e_file")
        file_v1 = FileArtifact.put(
            id=file_id,
            description="v1",
            file_path=str(SLACK_SAMPLE_PATH),
        )
        sa_id = _unique_id("e2e_svc")
        sa_v1 = EnvironmentArtifact.put(
            id=sa_id,
            environment_name="slack",
            file_artifact=file_v1,
        )
        universe = EnvironmentUniverseArtifact.put(
            id=_unique_id("e2e_universe"),
            environment_artifacts=[sa_v1],
        )
        pinned_file_version = file_v1.version
        pinned_sa_version = sa_v1.version

        # Republish the SAME service id with new bytes — mirrors `agent-env
        # artifact service put --id <same> <new-file>` after a fix. This creates
        # FileArtifact v2 AND EnvironmentArtifact v2.
        file_v2 = FileArtifact.put(
            id=file_id,
            description="v2 mutated",
            file_path=str(EMAIL_SAMPLE_PATH),
        )
        sa_v2 = EnvironmentArtifact.put(
            id=sa_id,
            environment_name="slack",
            file_artifact=file_v2,
        )
        assert file_v2.version == pinned_file_version + 1
        assert sa_v2.version == pinned_sa_version + 1

        # Reload the universe; walk the entire chain and assert nothing drifted.
        retrieved = Artifact.get(universe.id, version=universe.version)
        services = retrieved.get_environment_artifacts()
        assert len(services) == 1
        assert services[0].version == pinned_sa_version
        loaded_file = services[0].get_file_artifact()
        assert loaded_file.version == pinned_file_version

    def test_mixed_doc_prefers_refs(self):
        """If both legacy and new fields are present, refs win."""
        slack_v1 = _make_service_artifact(str(SLACK_SAMPLE_PATH), "slack", "su_mixed_slack")

        # Hand-craft a doc that has BOTH service_artifact_refs (pinned to v1) AND
        # a stale service_artifact_ids field (would point at the same id).
        universe_id = _unique_id("su_mixed")
        doc = {
            "id": universe_id,
            "version": 1,
            "type": "environment_universe",
            "service_artifact_refs": [{"id": slack_v1.id, "version": slack_v1.version}],
            "service_artifact_ids": [slack_v1.id],  # stale, should be ignored
            "created_at_utc": datetime.now(timezone.utc),
        }
        _artifact_docs().insert(ARTIFACTS_COLLECTION, doc)

        retrieved = Artifact.get(universe_id)
        # Publish a newer child; refs should still pin v1.
        new_file = FileArtifact.put(
            id=_unique_id("su_mixed_file_v2"),
            description="v2",
            file_path=str(SLACK_SAMPLE_PATH),
        )
        EnvironmentArtifact.put(
            id=slack_v1.id,
            environment_name="slack",
            file_artifact=new_file,
        )

        children = retrieved.get_environment_artifacts()
        assert len(children) == 1
        assert children[0].version == slack_v1.version  # pinned via refs, not latest

    def test_metadata_pinning(self):
        """Metadata FileArtifacts must also be pinned on put and resolved at the pinned version."""
        email = _make_service_artifact(str(EMAIL_ARTIFACT_PATH), "email", "su_md_email")
        md_file_v1 = FileArtifact.put(
            id=_unique_id("su_md_file"),
            description="md v1",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )

        universe = EnvironmentUniverseArtifact.put(
            id=_unique_id("su_md"),
            environment_artifacts=[email],
            metadata={"config": md_file_v1},
        )

        assert universe.metadata_refs is not None
        assert universe.metadata_refs["config"].id == md_file_v1.id
        assert universe.metadata_refs["config"].version == md_file_v1.version

        # Bump the metadata FileArtifact -> get_metadata must still return v1.
        FileArtifact.put(
            id=md_file_v1.id,
            description="md v2",
            file_path=str(EMAIL_ARTIFACT_PATH),
        )
        retrieved = Artifact.get(universe.id, version=universe.version)
        loaded = retrieved.get_metadata()
        assert loaded["config"].version == md_file_v1.version


class TestParseArtifactRef:
    """Pure unit test for the CLI ref parser — no integration needed."""

    def test_no_version(self):
        from agent_env.cli.utils import parse_artifact_ref
        assert parse_artifact_ref("slack") == ("slack", None)

    def test_with_version(self):
        from agent_env.cli.utils import parse_artifact_ref
        assert parse_artifact_ref("slack:3") == ("slack", 3)

    def test_trailing_colon_rejected(self):
        import click
        from agent_env.cli.utils import parse_artifact_ref
        with pytest.raises(click.BadParameter):
            parse_artifact_ref("slack:")

    def test_non_int_version_rejected(self):
        import click
        from agent_env.cli.utils import parse_artifact_ref
        with pytest.raises(click.BadParameter):
            parse_artifact_ref("slack:abc")

    def test_double_colon_rejected(self):
        import click
        from agent_env.cli.utils import parse_artifact_ref
        with pytest.raises(click.BadParameter):
            parse_artifact_ref("slack:1:2")

    def test_zero_version_rejected(self):
        import click
        from agent_env.cli.utils import parse_artifact_ref
        with pytest.raises(click.BadParameter):
            parse_artifact_ref("slack:0")

    def test_empty_id_rejected(self):
        import click
        from agent_env.cli.utils import parse_artifact_ref
        with pytest.raises(click.BadParameter):
            parse_artifact_ref(":3")


class TestArtifactRef:

    def test_str(self):
        assert str(ArtifactRef(id="slack-data", version=3)) == "slack-data:3"

    def test_str_round_trips_through_parser(self):
        """str(ref) must be a valid --environment-artifact value."""
        from agent_env.cli.utils import parse_artifact_ref
        ref = ArtifactRef(id="slack-data", version=3)
        assert parse_artifact_ref(str(ref)) == (ref.id, ref.version)
