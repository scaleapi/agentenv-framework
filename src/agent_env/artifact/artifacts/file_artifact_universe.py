"""FileArtifactUniverse — bundles FileArtifacts produced by collect_artifacts."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, model_serializer

from agent_env.artifact.artifact import _write_twin
from agent_env.artifact.ref import ArtifactRef
from agent_env.artifact.universe import Universe
from agent_env.config import get_config
from agent_env.store.ids import derive_id
from agent_env.utils.deprecation import renamed_keyword

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.file import FileArtifact
    from agent_env.bundle.authoring import AuthoringContext


class FileArtifactUniverse(Universe):
    """A universe that bundles FileArtifacts together.

    Typically auto-created by the ``collect_artifacts`` task step, which
    turns each collected file into a ``FileArtifact`` and wraps them in one
    of these universes. The universe preserves the original filename of
    each file so downstream browsers can display them without needing to
    load every ``FileArtifact`` document.
    """

    model_config = ConfigDict(populate_by_name=True)

    toml_keys: ClassVar[dict[str, type]] = {}  # what an artifact.toml may set
    type: Literal["file_artifact_universe"] = "file_artifact_universe"
    file_artifact_refs: Optional[dict[str, ArtifactRef]] = Field(
        default=None,
        description=(
            "Mapping of original filename -> pinned (id, version) ref to its "
            "FileArtifact. Preferred over file_artifact_ids on read; absence "
            "means a pre-pinning doc that resolves to latest."
        ),
    )
    # Kept alongside file_artifact_refs (additive) so raw-doc readers and CLI
    # attribute access keep working; new docs carry both keys.
    file_artifact_ids: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of original filename -> FileArtifact id. The filename "
            "is a denormalized convenience for fast catalog rendering "
            "(avoiding N FileArtifact lookups per card); canonical file "
            "metadata lives on the referenced FileArtifact document."
        ),
    )
    bundle_object_url: str | None = Field(
        default=None,
        alias="bundle_s3_url",
        description=(
            "Optional object-store prefix under which all bundled files "
            "reside. Set when the universe represents a contiguous directory "
            "(e.g. an agent snapshot). Consumers can use this directly without "
            "resolving individual FileArtifact documents."
        ),
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        # Dual-write, from the attribute: `alias=` emits one spelling, which one depending on the caller's `by_alias`.
        data = handler(self)
        _write_twin(data, "bundle_s3_url", "bundle_object_url", self.bundle_object_url)
        return data

    @classmethod
    def put(
        cls,
        id: str,
        *,
        file_artifacts: dict[str, "FileArtifact"],
        bundle_object_url: str | None = None,
        bundle_s3_url: str | None = None,
    ) -> "FileArtifactUniverse":
        from agent_env.artifact.store import get_artifact_store

        bundle_object_url = renamed_keyword(
            "FileArtifactUniverse.put", "bundle_object_url", bundle_object_url, "bundle_s3_url", bundle_s3_url
        )
        if not file_artifacts:
            raise ValueError("file_artifacts must be non-empty")
        if bundle_object_url:
            get_config().check_object_url(id, bundle_object_url)

        store = get_artifact_store()
        version = store.next_version(id)
        instance = cls(
            id=id,
            version=version,
            file_artifact_refs={
                fname: fa.as_ref() for fname, fa in file_artifacts.items()
            },
            file_artifact_ids={fname: fa.id for fname, fa in file_artifacts.items()},
            bundle_object_url=bundle_object_url,
        )
        return store.put_document(instance)

    def get_file_artifacts(self) -> dict[str, "FileArtifact"]:
        """Resolve each referenced FileArtifact. Returns {filename: FileArtifact}."""
        from agent_env.artifact.artifacts.file import FileArtifact

        # Truthiness (not `is not None`): a degenerate empty-`{}` refs dict
        # (never produced by put(), only by a hand-crafted/migrated doc) falls
        # through to the ids fallback instead of silently returning zero files.
        # Also keeps this consistent with the hub download branch, which guards
        # on `doc.get("file_artifact_refs")`.
        if self.file_artifact_refs:
            return {
                fname: FileArtifact.get(ref.id, version=ref.version)
                for fname, ref in self.file_artifact_refs.items()
            }
        if self.file_artifact_ids:
            logger.warning(
                "FileArtifactUniverse %s v%s has unpinned file_artifact_ids; "
                "falling back to latest. Republish to pin.",
                self.id,
                self.version,
            )
            return {
                fname: FileArtifact.get(fa_id)
                for fname, fa_id in self.file_artifact_ids.items()
            }
        raise ValueError(
            f"malformed FileArtifactUniverse {self.id} v{self.version}: "
            "neither file_artifact_refs nor file_artifact_ids present"
        )

    @classmethod
    def put_bundled(
        cls,
        id: str,
        *,
        files: dict[str, Path],
        prefix_url: str | None = None,
        s3_url: str | None = None,
    ) -> "FileArtifactUniverse":
        """Upload ``files`` (bundle key -> local path) under ``prefix_url`` and register them as one universe.
        Without ``prefix_url``, each call writes under a prefix of its own (``ArtifactStore.attempt_prefix``)."""
        from agent_env.artifact.artifacts.file import FileArtifact
        from agent_env.artifact.store import get_artifact_store

        prefix_url = renamed_keyword("FileArtifactUniverse.put_bundled", "prefix_url", prefix_url, "s3_url", s3_url)
        if not files:
            raise ValueError("files must be non-empty")
        if prefix_url is None:
            prefix_url = get_artifact_store().attempt_prefix(cls.model_fields["type"].default, id)
        if not prefix_url.endswith("/"):
            prefix_url += "/"

        file_artifacts: dict[str, FileArtifact] = {}
        for rel_path, local_path in files.items():
            if rel_path.startswith("/"):
                raise ValueError(f"Bundle key {rel_path!r} must be a relative path")
            if ".." in rel_path.split("/"):
                raise ValueError(f"Bundle key {rel_path!r} must not contain '..' segments")
            path_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]
            fa = FileArtifact.put_at(
                id=derive_id(id, path_hash),
                description=f"Bundled file '{rel_path}' of FileArtifactUniverse '{id}'",
                file_path=str(local_path),
                object_url=prefix_url + rel_path,
                filename=rel_path.rsplit("/", 1)[-1],
            )
            file_artifacts[rel_path] = fa

        return cls.put(id=id, file_artifacts=file_artifacts, bundle_object_url=prefix_url)

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> "FileArtifactUniverse":
        """Write the folder's files, each a file artifact keyed by its path in the folder."""
        ctx.accept(data, **cls.toml_keys)
        return cls.put_bundled(ctx.id, files=ctx.files())

    @classmethod
    def put_existing(
        cls,
        id: str,
        *,
        prefix_url: str | None = None,
        s3_url: str | None = None,
    ) -> "FileArtifactUniverse":
        """Wrap files that already exist under an object-store prefix as a universe.

        Unlike ``put_bundled`` (which uploads local files), this lists the
        objects already present under *prefix_url* and registers each one as a
        ``FileArtifact`` via ``FileArtifact.put_existing``. No bytes are
        downloaded or re-uploaded. Useful when another process (e.g. an
        agent's snapshot extension) has already written the files.
        """
        from agent_env.artifact.artifacts.file import FileArtifact

        prefix_url = renamed_keyword("FileArtifactUniverse.put_existing", "prefix_url", prefix_url, "s3_url", s3_url)
        if prefix_url is None:
            raise TypeError("FileArtifactUniverse.put_existing() missing required keyword argument: 'prefix_url'")
        if not prefix_url.endswith("/"):
            prefix_url += "/"

        store = get_config().get_object_store_at(prefix_url)

        file_artifacts: dict[str, FileArtifact] = {}
        for object_url in store.list_at(prefix_url):
            rel_path = object_url[len(prefix_url):]
            if not rel_path:
                continue
            path_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]
            fa = FileArtifact.put_existing(
                id=derive_id(id, path_hash),
                description=f"Bundled file '{rel_path}' of FileArtifactUniverse '{id}'",
                object_url=object_url,
            )
            file_artifacts[rel_path] = fa

        if not file_artifacts:
            raise ValueError(f"No files found under prefix {prefix_url!r}")

        return cls.put(id=id, file_artifacts=file_artifacts, bundle_object_url=prefix_url)
