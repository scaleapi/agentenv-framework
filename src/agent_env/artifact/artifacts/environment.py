"""Environment artifact for wrapping a FileArtifact with environment identity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import ConfigDict, Field, model_serializer

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.ref import ArtifactRef

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.file import FileArtifact


class EnvironmentArtifact(Artifact):
    """A FileArtifact wrapped with an environment name + schema version."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["environment"] = "environment"
    environment_name: str = Field(alias="service_name", description="Name of the environment this artifact belongs to")
    file_artifact_ref: Optional[ArtifactRef] = Field(
        default=None,
        description="Pinned (id, version) ref to the wrapped FileArtifact",
    )
    # Pre-pinning docs only; alias keeps the on-disk key as `file_artifact_id`.
    legacy_file_artifact_id: Optional[str] = Field(
        default=None,
        alias="file_artifact_id",
        description="Legacy unpinned wrapped-FileArtifact id",
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        # `@model_serializer(mode="wrap")` is the only hook invoked by both
        # `model_dump` and the Rust-implemented `model_dump_json`.
        data = handler(self)
        if "legacy_file_artifact_id" in data:
            data["file_artifact_id"] = data.pop("legacy_file_artifact_id")
        if self.file_artifact_ref is not None:  # new doc
            data.pop("file_artifact_id", None)
        # Dual-write. Sourced from the attribute, not from `data`,
        # because `alias=` emits exactly one spelling and which one depends on the
        # caller's `by_alias` — the store persists with by_alias=True
        # (artifact/store.py), while in-process callers dump either way.
        data["service_name"] = self.environment_name
        data["environment_name"] = self.environment_name
        return data

    @classmethod
    def put(
        cls,
        id: str,
        *,
        environment_name: Optional[str] = None,
        file_artifact: FileArtifact,
    ) -> EnvironmentArtifact:
        from agent_env.artifact.store import get_artifact_store

        if not environment_name:
            raise ValueError("environment_name cannot be empty")
        instance = cls(
            id=id,
            environment_name=environment_name,
            file_artifact_ref=file_artifact.as_ref(),
        )
        return get_artifact_store().put_document(instance)

    def get_file_artifact(self) -> FileArtifact:
        from agent_env.artifact.artifacts.file import FileArtifact

        if self.file_artifact_ref is not None:
            return FileArtifact.get(
                self.file_artifact_ref.id, version=self.file_artifact_ref.version
            )
        if self.legacy_file_artifact_id is not None:
            logger.warning(
                "EnvironmentArtifact %s v%s has unpinned file_artifact_id; falling back to latest. Republish to pin.",
                self.id, self.version,
            )
            return FileArtifact.get(self.legacy_file_artifact_id)
        raise ValueError(
            f"malformed EnvironmentArtifact {self.id} v{self.version}: "
            "neither file_artifact_ref nor file_artifact_id present"
        )
