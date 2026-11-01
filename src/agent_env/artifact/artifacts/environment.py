"""Environment artifact: the files of the environment it seeds, by path, with its name."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, model_serializer

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.ref import ArtifactRef
from agent_env.entity.export import Unwritable
from agent_env.entity.refs import EntityRef
from agent_env.store.ids import derive_id, derive_path_id
from agent_env.utils.paths import validate_file_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.entity.export import ExportContext


class SeedsEnvironments(ABC):
    """An artifact type whose write seeds environments: a bundle checks each against the servers of the env a task
    loads it into, and that one artifact's environments have names of their own."""

    @classmethod
    @abstractmethod
    def seeded_environments(cls, ctx: AuthoringContext) -> tuple[tuple[str, str], ...]:
        """``(path in the folder, environment_name)`` of each environment writing it from ``ctx``'s folder seeds, the
        path ``""`` for the folder itself. Raises BundleError when the folder can't be read."""


class EnvironmentArtifact(Artifact, SeedsEnvironments):
    """The files of the environment it seeds, by path, with its name: one wrapped FileArtifact, as every environment
    artifact was before it could hold several, or ``file_artifact_refs`` when it holds several or one inside a folder."""

    model_config = ConfigDict(populate_by_name=True)

    # An artifact.toml names the environment and either the artifact to wrap (file) or nothing, to wrap the folder's
    # one file, whose description it may set, or keep its several. The stored document's own names for these aren't
    # taken.
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
    file_artifact_refs: Optional[dict[str, ArtifactRef]] = Field(
        default=None,
        description="Pinned (id, version) refs to the environment's FileArtifacts by path, in place of "
                    "file_artifact_ref when it has several, or one inside a folder",
    )

    # No return annotation: pydantic builds the serialization schema from one, and a dict drops the fields.
    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any):
        # `@model_serializer(mode="wrap")` is the only hook invoked by both
        # `model_dump` and the Rust-implemented `model_dump_json`.
        data = handler(self)
        if "legacy_file_artifact_id" in data:
            data["file_artifact_id"] = data.pop("legacy_file_artifact_id")
        if self.file_artifact_ref is not None or self.file_artifact_refs is not None:  # new doc
            data.pop("file_artifact_id", None)
        if self.file_artifact_refs is None:
            data.pop("file_artifact_refs", None)
        else:
            data.pop("file_artifact_ref", None)
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
        file_artifact: Optional[FileArtifact] = None,
        file_artifacts: Optional[dict[str, FileArtifact]] = None,
    ) -> EnvironmentArtifact:
        """Write the environment of ``environment_name`` over ``file_artifact``, or ``file_artifacts`` by path. One file
        whose path is its filename is written as ``file_artifact``, the document every environment artifact had."""
        from agent_env.artifact.store import get_artifact_store

        if not environment_name:
            raise ValueError("environment_name cannot be empty")
        if (file_artifact is None) == (file_artifacts is None):
            raise ValueError("an environment artifact holds a file_artifact or file_artifacts, one of the two")
        if file_artifacts is not None:
            if not file_artifacts:
                raise ValueError("file_artifacts can't be empty")
            for path in file_artifacts:
                validate_file_path(path)
            if len(file_artifacts) == 1 and (path := next(iter(file_artifacts))) == file_artifacts[path].filename:
                file_artifact, file_artifacts = file_artifacts[path], None
        instance = cls(
            id=id,
            environment_name=environment_name,
            file_artifact_ref=file_artifact.as_ref() if file_artifact is not None else None,
            file_artifact_refs=({path: file.as_ref() for path, file in file_artifacts.items()}
                                if file_artifacts is not None else None),
        )
        return get_artifact_store().put_document(instance)

    @staticmethod
    def derived_file_id(id: str) -> str:
        """The id an environment's file is written under when it's written with the environment: ``<id>__file``."""
        return derive_id(id, "file")

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> EnvironmentArtifact:
        """Write the environment authored as ``data`` (its artifact.toml, with ``file`` resolved to a file artifact's
        id) under ``ctx.id`` and return it: over the artifact ``file`` names; over the folder's one file, written as
        ``<id>__file`` with ``description`` defaulting to its name; or over its several files by path, each written
        under the id a file universe's is (``derive_path_id``)."""
        fields = cls.accept_toml(data, ctx)
        if "file" in fields:
            file = ctx.artifact(fields["file"], FileArtifact)
        elif len(files := ctx.files()) == 1:
            filename, path = ctx.file()
            file = FileArtifact.put_attempt(cls.derived_file_id(ctx.id), file_path=str(path), filename=filename,
                                            description=fields.get("description", filename))
        else:
            return cls.put(ctx.id, environment_name=fields["environment_name"], file_artifacts={
                path: FileArtifact.put_attempt(derive_path_id(ctx.id, path), file_path=str(local),
                                               filename=PurePosixPath(path).name, description=path)
                for path, local in files.items()})
        return cls.put(ctx.id, environment_name=fields["environment_name"], file_artifact=file)

    def to_toml(self, out: ExportContext) -> dict:
        """The artifact.toml of a folder holding the environment's files at their paths, which ``from_toml`` reads back
        as this. One file kept at a path other than its filename is refused: a folder's one file reads back at its
        filename."""
        if self.file_artifact_refs is not None:
            if len(self.file_artifact_refs) == 1:
                raise Unwritable(f"its one file is kept at {next(iter(self.file_artifact_refs))!r}, not at its filename, "
                                 "and a folder's one file reads back kept at its filename")
            for path, file in self.get_file_artifacts().items():
                out.file(path, file.object_url)
            return {"type": self.type, "environment_name": self.environment_name}
        file = self.get_file_artifact()
        out.file(file.filename, file.object_url)
        described = {} if file.description == file.filename else {"description": file.description}
        return {"type": self.type, "environment_name": self.environment_name, **described}

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an artifact.toml, an environment takes, and its folder: its files, or only
        artifact.toml when ``file`` names the artifact to wrap. Raises BundleError listing every problem."""
        problems = []
        if (files := ctx.try_files(empty_ok=True)) is not None:  # None when the plan's walk reports why
            if any(key == "file" or cls.toml_stored_names.get(key) == "file" for key in data):
                if files:
                    problems.append(f"its file is the artifact file names, so the folder holds only artifact.toml, and "
                                    f"it also has {', '.join(repr(name) for name in files)}; give the file an artifact "
                                    "folder of its own, or leave out file to keep the folder's files")
            elif not files:
                problems.append(ctx.no_files("file for the environment to wrap",
                                             "put its files here, or name the artifact to wrap with file"))
            elif "description" in data and len(files) > 1:
                problems.append(ctx.config_problem(f"description describes the folder's one file, and it holds "
                                                   f"{', '.join(repr(name) for name in files)}; drop description"))
        data, renamed = ctx.renamed(data, cls.toml_stored_names)
        problems += renamed
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

    @classmethod
    def seeded_environments(cls, ctx: AuthoringContext) -> tuple[tuple[str, str], ...]:
        """Its own environment, seeded from the folder itself."""
        name = ctx.entry.config.get("environment_name") if isinstance(ctx.entry.config, dict) else None
        return (("", name),) if isinstance(name, str) and name else ()

    def get_file_artifacts(self) -> dict[str, FileArtifact]:
        """The environment's files by path: ``{filename: file}`` for the one file it wraps."""
        if self.file_artifact_refs is not None:
            return {path: FileArtifact.get(ref.id, version=ref.version) for path, ref in self.file_artifact_refs.items()}
        file = self.get_file_artifact()
        return {file.filename: file}

    def get_file_artifact(self) -> FileArtifact:
        """The one file the environment wraps; refused for one that holds several, whose files
        ``get_file_artifacts`` reads."""
        if self.file_artifact_refs is not None:
            raise ValueError(f"EnvironmentArtifact {self.id} v{self.version} holds {len(self.file_artifact_refs)} files "
                             f"by path, not one; read them with get_file_artifacts()")
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
