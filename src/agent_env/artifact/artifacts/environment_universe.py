"""Environment universe artifact for bundling related environment artifacts together."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

from pydantic import AliasChoices, ConfigDict, Field, model_serializer, model_validator

from agent_env.artifact.artifact import FolderChild
from agent_env.artifact.artifacts.environment import EnvironmentArtifact, SeedsEnvironments
from agent_env.artifact.artifacts.environment_universe_folder import (
    METADATA,
    UniverseFolder,
    UniverseFolderError,
    environment_id,
    environment_path,
    metadata_id,
    metadata_path,
    read_folder,
)
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.ref import ArtifactRef
from agent_env.artifact.universe import Universe
from agent_env.entity.export import Unwritable
from agent_env.entity.refs import EntityRef

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.entity.export import ExportContext


class EnvironmentUniverseArtifact(Universe, SeedsEnvironments):
    """A universe artifact that bundles EnvironmentArtifacts together."""

    model_config = ConfigDict(populate_by_name=True)

    # An artifact.toml may name environment artifacts to add to the ones its folder holds (see from_toml). The stored
    # document's own names aren't taken, nor is metadata, which a metadata/<key>/ folder holds.
    toml_keys: ClassVar[dict[str, type]] = {"environment_artifacts": list}
    toml_refs: ClassVar[tuple[EntityRef, ...]] = (
        EntityRef.artifact("environment_artifacts[]", artifact_type="environment"),)
    toml_stored_names: ClassVar[dict[str, str]] = {
        "service_artifact_refs": "environment_artifacts", "environment_artifact_refs": "environment_artifacts",
        "service_artifact_ids": "environment_artifacts", "metadata": "a metadata/<key>/ folder holding its file",
        "metadata_refs": "a metadata/<key>/ folder holding its file"}
    # What a universe's metadata is called: the folder `environment-universe get --output-dir` writes its files to, and
    # the segment of the ids they're written under.
    metadata_name: ClassVar[str] = METADATA

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

    @model_validator(mode="before")
    @classmethod
    def _metadata_is_legacy(cls, data: Any) -> Any:
        """A stored universe's ``metadata`` key holds its legacy metadata file ids, never artifact metadata."""
        if isinstance(data, dict) and "metadata" in data:
            data = {**data, "legacy_metadata": data["metadata"]}
            del data["metadata"]
        return data

    # No return annotation: pydantic builds the serialization schema from one, and a dict drops the fields.
    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any):
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
            # No `service_artifact_ids` twin on these modern docs: that key means
            # "unpinned, resolve at latest" to every reader, older releases
            # included, so writing it onto a doc that IS pinned tells those
            # readers to ignore the pins. Nothing legacy
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

    def merge_metadata(self, updates: dict[str, Any], retries: int = 3) -> None:
        raise TypeError(f"artifact {self.id!r} is an environment universe, whose metadata key holds its legacy metadata "
                        f"file ids, so it keeps no artifact metadata")

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

    @staticmethod
    def derived_environment_id(id: str, environment_name: str) -> str:
        """The id an environment is written under when it's written with the universe: ``<id>__<environment_name>``."""
        return environment_id(id, environment_name)

    @staticmethod
    def derived_metadata_id(id: str, key: str) -> str:
        """The id a metadata file is written under when it's written with the universe: ``<id>__metadata__<key>``."""
        return metadata_id(id, key)

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> EnvironmentUniverseArtifact:
        """Write the universe authored as ``data`` (its artifact.toml, with ``environment_artifacts`` resolved to
        environment artifacts' ids) and its folder under ``ctx.id``, and return it. The folder is laid out as
        ``environment-universe get --output-dir`` writes one: each ``<environment_name>/`` folder's file is written as
        an environment of that name, ``<id>__<name>`` over ``<id>__<name>__file``, and each ``metadata/<key>/``
        folder's file as ``<id>__metadata__<key>``: a name or key holding ``/`` is nested folders, and one an id can't
        hold is written under a hash of it (``environment_universe_folder``). Its environments are the folders', in
        name order, then the ones ``environment_artifacts`` names. Nothing is written when it has none, or two of one
        name."""
        fields = cls.accept_toml(data, ctx)
        layout = cls._read_folder(ctx)
        named = [ctx.artifact(ref, EnvironmentArtifact) for ref in fields.get("environment_artifacts", [])]
        names = Counter([laid.name for laid in layout.environments] + [env.environment_name for env in named])
        if repeated := sorted(name for name, count in names.items() if count > 1):
            ctx.refuse([ctx.config_problem(f"two of its environments are named {name!r}; a universe's environments "
                                           "need names of their own") for name in repeated])
        environments = []
        for laid in layout.environments:
            file = FileArtifact.put_attempt(laid.file_id, description=laid.filename, file_path=str(laid.path),
                                            filename=laid.filename)
            environments.append(EnvironmentArtifact.put(laid.environment_id, environment_name=laid.name,
                                                        file_artifact=file))
        metadata = {laid.name: FileArtifact.put_attempt(laid.file_id, description=laid.filename,
                                                        file_path=str(laid.path), filename=laid.filename)
                    for laid in layout.metadata}
        return cls.put(ctx.id, environment_artifacts=environments + named, metadata=metadata or None)

    def to_toml(self, out: ExportContext) -> dict:
        """The artifact.toml of a folder laid out as ``environment-universe get --output-dir`` writes one."""
        for environment in self.get_environment_artifacts():
            name = environment.environment_name
            if name.split("/", 1)[0] == METADATA:
                raise Unwritable(f"its environment {name!r} would read back as metadata, from its {METADATA}/ folder")
            if environment.file_artifact_refs is not None:
                raise Unwritable(f"its environment {name!r} holds several files, and a universe's folder holds one per "
                                 f"environment; write environment artifact {environment.id} to a folder of its own")
            file = environment.get_file_artifact()
            out.file(environment_path(Path(), name, file.filename).as_posix(), file.object_url)
        for key, file in (self.get_metadata() or {}).items():
            out.file(metadata_path(Path(), key, file.filename).as_posix(), file.object_url)
        return {"type": self.type}

    @classmethod
    def written_besides(cls, ctx: AuthoringContext) -> tuple[FolderChild, ...]:
        """What ``from_toml`` writes besides the universe, as ``read_folder`` names it: each environment its folder
        holds, that environment's file, and each metadata file."""
        layout = cls._read_folder(ctx)
        return (*(FolderChild(id, laid.name) for laid in layout.environments for id in (laid.environment_id,
                                                                                         laid.file_id)),
                *(FolderChild(laid.file_id, f"{METADATA}/{laid.name}") for laid in layout.metadata))

    @classmethod
    def seeded_environments(cls, ctx: AuthoringContext) -> tuple[tuple[str, str], ...]:
        """Each environment its folder holds, at the path naming it."""
        return tuple((laid.name, laid.name) for laid in cls._read_folder(ctx).environments)

    @classmethod
    def _read_folder(cls, ctx: AuthoringContext) -> UniverseFolder:
        """The universe's folder, read as ``environment-universe get --output-dir`` writes one (``read_folder``).
        Raises BundleError listing every problem."""
        files, folders = ctx.listing()
        try:
            return read_folder(ctx.id, files, folders)
        except UniverseFolderError as e:
            ctx.refuse([], within=e.problems)

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an artifact.toml, a universe takes, and its folder, read as ``environment-universe
        get --output-dir`` writes one, which needs an environment of its own unless ``environment_artifacts`` names
        one. Raises BundleError listing every problem."""
        data, problems = ctx.renamed(data, cls.toml_stored_names)
        fields, more = ctx.accepted(data, **cls.toml_keys)
        problems, within = problems + more, []
        if (listing := ctx.try_listing()) is not None:  # None when the plan's walk reports why
            try:
                layout = read_folder(ctx.id, *listing)
            except UniverseFolderError as e:
                within = e.problems
            else:
                if not layout.environments and not data.get("environment_artifacts"):
                    problems.append("a universe needs an environment: a folder of its own, named after it and holding "
                                    "its file, or one environment_artifacts names")
        if problems or within:
            ctx.refuse(problems, within=within)
        return fields

    def get_environment_artifacts(self) -> list[EnvironmentArtifact]:
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
        """``{<environment_name>/<path>: FileArtifact}`` — a plain-file view of this universe, each environment's
        files at their paths (one file's at its filename).

        Matching ``FileArtifactUniverse.get_file_artifacts()`` is what lets the same
        loaders stage a service universe as a file tree (grading a frozen snapshot)
        rather than restoring it into an env. Returns the pinned artifacts as-is —
        nothing is copied or re-registered.
        """
        out: dict[str, FileArtifact] = {}
        for sa in self.get_environment_artifacts():
            for path, fa in sa.get_file_artifacts().items():
                name = f"{sa.environment_name}/{path}"
                if name in out:
                    raise ValueError(
                        f"EnvironmentUniverseArtifact {self.id} v{self.version} has duplicate "
                        f"environment '{sa.environment_name}'; cannot stage it as a file tree"
                    )
                out[name] = fa
        return out

    def get_metadata(self) -> dict[str, FileArtifact]:
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
