"""Environment universe artifact for bundling related environment artifacts together."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import AliasChoices, ConfigDict, Field, model_serializer

from agent_env.artifact.ref import ArtifactRef
from agent_env.artifact.universe import Universe

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.file import FileArtifact
    from agent_env.artifact.artifacts.environment import EnvironmentArtifact


class EnvironmentUniverseArtifact(Universe):
    """A universe artifact that bundles EnvironmentArtifacts together."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["environment_universe"] = "environment_universe"
    # serialization_alias pins the emitted key: the attribute is renamed, the document is not.
    environment_artifact_refs: Optional[list[ArtifactRef]] = Field(
        default=None,
        serialization_alias="service_artifact_refs",
        validation_alias=AliasChoices("service_artifact_refs", "environment_artifact_refs"),
        description="List of pinned (id, version) refs to EnvironmentArtifacts",
    )
    metadata_refs: Optional[dict[str, ArtifactRef]] = Field(
        default=None,
        description="Optional pinned (id, version) refs to metadata FileArtifacts",
    )
    # Pre-pinning docs only; aliases keep on-disk keys as `service_artifact_ids` / `metadata`.
    legacy_environment_artifact_ids: Optional[list[str]] = Field(
        default=None,
        alias="service_artifact_ids",
        description="Legacy unpinned EnvironmentArtifact ids",
    )
    legacy_metadata: Optional[dict[str, str]] = Field(
        default=None,
        alias="metadata",
        description="Legacy unpinned metadata FileArtifact ids",
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        # `@model_serializer(mode="wrap")` is the only hook invoked by both
        # `model_dump` and the Rust-implemented `model_dump_json`.
        data = handler(self)
        if "legacy_environment_artifact_ids" in data:
            data["service_artifact_ids"] = data.pop("legacy_environment_artifact_ids")
        if "legacy_metadata" in data:
            data["metadata"] = data.pop("legacy_metadata")
        if self.environment_artifact_refs is not None:  # new doc
            data.pop("service_artifact_ids", None)
            data.pop("metadata", None)
            # Dual-write: the pinned refs get a twin, and only them.
            # The scope line also asked for a `service_artifact_ids` twin on these
            # modern docs — declined. That key means "unpinned, resolve at latest"
            # to every reader, and v0.9.1 (pinned permanently by
            # synthetic-artifacts-pipeline) acts on it, so writing it onto a doc
            # that IS pinned tells that worker to ignore the pins. Nothing legacy
            # is dropped here: unpinned docs never reach this branch and still
            # round-trip `service_artifact_ids` / `metadata` above, and
            # get_environment_artifacts still falls back to them.
            # `by_alias=False` dumps under the field name, `by_alias=True` under the
            # serialization alias; emit both keys either way, as this always has.
            refs = data.pop("environment_artifact_refs", None)
            if refs is None:
                refs = data["service_artifact_refs"]
            data["service_artifact_refs"] = refs
            data["environment_artifact_refs"] = refs
        return data

    @classmethod
    def put(
        cls,
        id: str,
        *,
        environment_artifacts: list[EnvironmentArtifact],
        metadata: Optional[dict[str, FileArtifact]] = None,
    ) -> EnvironmentUniverseArtifact:
        from agent_env.artifact.store import get_artifact_store

        if not environment_artifacts:
            raise ValueError("environment_artifacts must be non-empty")
        instance = cls(
            id=id,
            environment_artifact_refs=[ea.as_ref() for ea in environment_artifacts],
            metadata_refs=(
                {k: fa.as_ref() for k, fa in metadata.items()} if metadata else None
            ),
        )
        return get_artifact_store().put_document(instance)

    def get_environment_artifacts(self) -> list[EnvironmentArtifact]:
        from agent_env.artifact.artifacts.environment import EnvironmentArtifact

        if self.environment_artifact_refs is not None:
            return [
                EnvironmentArtifact.get(ref.id, version=ref.version)
                for ref in self.environment_artifact_refs
            ]
        if self.legacy_environment_artifact_ids:
            logger.warning(
                "EnvironmentUniverseArtifact %s v%s has unpinned service_artifact_ids; falling back to latest. Republish to pin.",
                self.id, self.version,
            )
            return [
                EnvironmentArtifact.get(sa_id) for sa_id in self.legacy_environment_artifact_ids
            ]
        raise ValueError(
            f"malformed EnvironmentUniverseArtifact {self.id} v{self.version}: "
            "neither service_artifact_refs nor service_artifact_ids present"
        )

    def get_file_artifacts(self) -> dict[str, FileArtifact]:
        """``{relative filename: FileArtifact}`` — a plain-file view of this universe.

        Matching ``FileArtifactUniverse.get_file_artifacts()`` is what lets the same
        loaders stage a service universe as a file tree (grading a frozen snapshot)
        rather than restoring it into an env. Returns the pinned artifacts as-is —
        nothing is copied or re-registered.
        """
        out: dict[str, FileArtifact] = {}
        for sa in self.get_environment_artifacts():
            fa = sa.get_file_artifact()
            name = f"{sa.environment_name}/{fa.filename}"
            if name in out:
                raise ValueError(
                    f"EnvironmentUniverseArtifact {self.id} v{self.version} has duplicate "
                    f"environment '{sa.environment_name}'; cannot stage it as a file tree"
                )
            out[name] = fa
        return out

    def get_metadata(self) -> dict[str, FileArtifact]:
        from agent_env.artifact.artifacts.file import FileArtifact

        if self.metadata_refs:
            return {
                k: FileArtifact.get(ref.id, version=ref.version)
                for k, ref in self.metadata_refs.items()
            }
        if self.legacy_metadata:
            logger.warning(
                "EnvironmentUniverseArtifact %s v%s has unpinned metadata; falling back to latest. Republish to pin.",
                self.id, self.version,
            )
            return {k: FileArtifact.get(fa_id) for k, fa_id in self.legacy_metadata.items()}
        # Metadata is optional; absence is not a malformed doc (unlike service refs).
        return {}
