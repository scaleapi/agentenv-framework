"""Generic file artifact for storing arbitrary files in the object store."""

from __future__ import annotations

import mimetypes
import os
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import ConfigDict, Field, model_serializer

from agent_env.artifact.artifact import Artifact, _write_twin
from agent_env.config import get_config

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext


class FileArtifact(Artifact):
    """A generic file artifact that stores any file in the object store.

    Clients define their own schemas and upload files (JSON, images, etc.).
    When loaded, returns raw bytes for the client to parse as needed.

    Example:
        # Create a file artifact
        artifact = FileArtifact.put(
            id="customer_data",
            description="Customer email data in JSON format",
            file_path="/path/to/emails.json",
        )

        # Load the file bytes
        data = artifact.load()
        emails = json.loads(data)  # Client parses as needed
    """

    model_config = ConfigDict(populate_by_name=True)

    toml_keys: ClassVar[dict[str, type]] = {"description": str}  # what an artifact.toml may set
    type: Literal["file"] = "file"
    description: str = Field(description="Human-readable description of the artifact contents")
    filename: str = Field(description="Original filename (preserved for reference)")
    content_type: str = Field(description="MIME type of the file")
    object_url: str = Field(alias="s3_url", description="Object-store locator where the file is stored")

    # No return annotation: pydantic builds the serialization schema from one, and a dict drops the fields.
    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any):
        # Dual-write, from the attribute: `alias=` emits one spelling, which one depending on the caller's `by_alias`.
        data = handler(self)
        _write_twin(data, "s3_url", "object_url", self.object_url)
        return data

    @classmethod
    def put(
        cls,
        id: str,
        *,
        description: str,
        file_path: str,
    ) -> Self:
        from agent_env.artifact.store import get_artifact_store

        store = get_artifact_store()

        filename = os.path.basename(file_path)
        prefix = store.attempt_prefix("file", id)
        return cls.put_at(
            id, description=description, file_path=file_path,
            object_url=f"{prefix}{filename}", filename=filename,
        )
    @classmethod
    def put_bytes(
        cls,
        id: str,
        *,
        description: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> Self:
        """Like ``put``, but takes raw bytes instead of a file path."""
        from agent_env.artifact.store import get_artifact_store

        store = get_artifact_store()
        prefix = store.attempt_prefix("file", id)
        object_url = f"{prefix}{filename}"
        object_store = get_config().get_object_store_to_write(object_url, id)
        stored_url = object_store.put(
            object_store.get_object_key(object_url), content, content_type
        )

        instance = cls(
            id=id,
            version=store.next_version(id),
            description=description,
            filename=filename,
            content_type=content_type,
            object_url=stored_url,
        )
        return store.put_document(instance)

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> Self:
        """Write the folder's one file (``put_attempt``); ``description`` defaults to the file's name."""
        fields = ctx.accept(data, **cls.toml_keys)
        filename, path = ctx.file()
        return cls.put_attempt(ctx.id, description=fields.get("description", filename), file_path=str(path),
                               filename=filename)

    @classmethod
    def put_attempt(cls, id: str, *, description: str, file_path: str, filename: str) -> Self:
        """``put_at`` under a prefix of this attempt's own (``ArtifactStore.attempt_prefix``), so a write that
        fails partway never blocks the next one."""
        from agent_env.artifact.store import get_artifact_store

        prefix = get_artifact_store().attempt_prefix(cls.model_fields["type"].default, id)
        return cls.put_at(id, description=description, file_path=file_path, object_url=prefix + filename,
                          filename=filename)

    def load(self) -> bytes:
        from agent_env.artifact.store import get_artifact_store

        return get_artifact_store().get_object(self.object_url)

    def get_file_artifacts(self) -> dict[str, "FileArtifact"]:
        return {self.filename: self}

    @classmethod
    def put_at(
        cls,
        id: str,
        *,
        description: str,
        file_path: str,
        object_url: str,
        filename: str | None = None,
    ) -> Self:
        """Upload ``file_path`` to an explicit object_url (write-once) and register it."""
        from agent_env.artifact.store import get_artifact_store

        store = get_config().get_object_store_to_write(object_url, id)

        if filename is None:
            filename = os.path.basename(file_path)
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = "application/octet-stream"

        artifact_store = get_artifact_store()
        version = artifact_store.next_version(id)
        stored_url = store.put_file_at(object_url, file_path, content_type)
        instance = cls(
            id=id,
            version=version,
            description=description,
            filename=filename,
            content_type=content_type,
            object_url=stored_url,
        )
        return artifact_store.put_document(instance)

    @classmethod
    def put_existing(
        cls,
        id: str,
        *,
        description: str,
        object_url: str,
    ) -> Self:
        """Register an already-uploaded object as a FileArtifact.

        Unlike ``put`` and ``put_at``, this does NOT upload bytes; it asserts
        the object exists and records artifact metadata. Use when the file was
        already written by some other process (e.g. an agent's snapshot
        extension) and you want to wrap it without round-tripping through disk.
        """
        from agent_env.artifact.store import get_artifact_store

        store = get_config().get_object_store_to_write(object_url, id)
        if object_url.endswith("/"):
            raise ValueError(f"object_url must point at an object, not a prefix: {object_url!r}")
        metadata = store.get_object_metadata_at(object_url)
        if metadata is None:
            raise ValueError(f"Object does not exist at {object_url!r}")

        # The last segment, not urlparse's path, which ends at a "#" or "?" that keys may hold.
        filename = object_url.rsplit("/", 1)[-1]
        content_type = metadata.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

        artifact_store = get_artifact_store()
        version = artifact_store.next_version(id)
        instance = cls(
            id=id,
            version=version,
            description=description,
            filename=filename,
            content_type=content_type,
            object_url=object_url,
        )
        return artifact_store.put_document(instance)
