"""Environment artifact for wrapping a FileArtifact with environment identity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, model_serializer

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.ref import ArtifactRef
from agent_env.entity_refs import EntityRef
from agent_env.store.ids import derive_id

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext


class EnvironmentArtifact(Artifact):
    """A FileArtifact wrapped with the name of the environment it seeds."""

    model_config = ConfigDict(populate_by_name=True)

    # An artifact.toml names the environment and either the artifact to wrap (file) or nothing, to wrap the folder's
    # one file, whose description it may set. The stored document's own names for these aren't taken.
    toml_keys: ClassVar[dict[str, type]] = {"environment_name": str, "file": object, "description": str}
    toml_refs: ClassVar[tuple[EntityRef, ...]] = (EntityRef.artifact("file", artifact_type="file"),)
    toml_stored_names: ClassVar[dict[str, str]] = {"service_name": "environment_name", "file_artifact_id": "file",
                                                   "file_artifact_ref": "file"}

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

    # No return annotation: pydantic builds the serialization schema from one, and a dict drops the fields.
    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any):
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

    @staticmethod
    def derived_file_id(id: str) -> str:
        """The id an environment's file is written under when it's written with the environment: ``<id>__file``."""
        return derive_id(id, "file")

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> EnvironmentArtifact:
        """Write the environment authored as ``data`` (its artifact.toml, with ``file`` resolved to a file artifact's
        id) under ``ctx.id`` and return it: over the artifact ``file`` names, or over the folder's one file, written
        as ``<id>__file`` with ``description`` defaulting to its name."""
        fields = cls.accept_toml(data, ctx)
        if "file" in fields:
            file = ctx.artifact(fields["file"], FileArtifact)
        else:
            filename, path = ctx.file()
            file = FileArtifact.put_attempt(cls.derived_file_id(ctx.id), file_path=str(path), filename=filename,
                                            description=fields.get("description", filename))
        return cls.put(ctx.id, environment_name=fields["environment_name"], file_artifact=file)

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an artifact.toml, an environment takes. Raises BundleError listing every problem."""
        data, problems = ctx.renamed(data, cls.toml_stored_names)
        fields, more = ctx.accepted(data, **cls.toml_keys)
        problems += more
        if "environment_name" not in data:
            problems.append(ctx.config_problem("environment_name is required: the environment the file seeds"))
        elif fields.get("environment_name") == "":
            problems.append(ctx.config_problem("environment_name can't be empty"))
        if "file" in fields and "description" in fields:
            problems.append(ctx.config_problem("description describes the folder's own file, and file names an "
                                               "artifact instead; drop one"))
        if problems:
            ctx.refuse(problems)
        return fields

    def get_file_artifact(self) -> FileArtifact:
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
