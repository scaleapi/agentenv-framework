"""Generic file artifact for storing arbitrary files in S3."""

from __future__ import annotations

import mimetypes
import os
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field

from agent_env.artifact.artifact import Artifact
from agent_env.config import get_config


class FileArtifact(Artifact):
    """A generic file artifact that stores any file in S3.

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

    type: Literal["file"] = "file"
    description: str = Field(description="Human-readable description of the artifact contents")
    filename: str = Field(description="Original filename (preserved for reference)")
    content_type: str = Field(description="MIME type of the file")
    object_url: str = Field(alias="s3_url", description="Object-store locator where the file is stored")

    @classmethod
    def put(
        cls,
        id: str,
        *,
        description: str,
        file_path: str,
    ) -> "FileArtifact":
        from agent_env.artifact.store import get_artifact_store

        store = get_artifact_store()

        filename = os.path.basename(file_path)
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = "application/octet-stream"

        version = store.next_version(id)

        # Use put_object_file (boto3 managed multipart upload) instead of
        # put_object (s3.put_object — hard 5 GB single-object limit). Streams
        # directly from disk so large files don't get loaded into RAM either.
        s3_url = store.put_object_file(
            artifact_type="file",
            id=id,
            version=version,
            object_name=filename,
            file_path=file_path,
            content_type=content_type,
        )

        # Store artifact document in MongoDB
        instance = cls(
            id=id,
            version=version,
            description=description,
            filename=filename,
            content_type=content_type,
            s3_url=s3_url,
        )
        return store.put_document(instance)

    @classmethod
    def put_bytes(
        cls,
        id: str,
        *,
        description: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> "FileArtifact":
        """Like ``put``, but takes raw bytes instead of a file path."""
        from agent_env.artifact.store import get_artifact_store

        store = get_artifact_store()
        version = store.next_version(id)

        s3_url = store.put_object(
            artifact_type="file",
            id=id,
            version=version,
            object_name=filename,
            data=content,
            content_type=content_type,
        )

        instance = cls(
            id=id,
            version=version,
            description=description,
            filename=filename,
            content_type=content_type,
            s3_url=s3_url,
        )
        return store.put_document(instance)

    def load(self) -> bytes:
        from agent_env.artifact.store import get_artifact_store

        return get_artifact_store().get_object(self.object_url)

    @classmethod
    def put_at(
        cls,
        id: str,
        *,
        description: str,
        file_path: str,
        object_url: str,
    ) -> "FileArtifact":
        """Upload ``file_path`` to an explicit object_url (write-once) and register it."""
        from agent_env.artifact.store import get_artifact_store

        store = get_config().get_object_store()

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
            s3_url=stored_url,
        )
        return artifact_store.put_document(instance)

    @classmethod
    def put_existing(
        cls,
        id: str,
        *,
        description: str,
        object_url: str,
    ) -> "FileArtifact":
        """Register an already-uploaded object as a FileArtifact.

        Unlike ``put`` and ``put_at``, this does NOT upload bytes; it asserts
        the object exists and records artifact metadata. Use when the file was
        already written by some other process (e.g. an agent's snapshot
        extension) and you want to wrap it without round-tripping through disk.
        """
        from agent_env.artifact.store import get_artifact_store

        store = get_config().get_object_store()
        path = urlparse(object_url).path
        if not path or path.endswith("/"):
            raise ValueError(f"object_url must point at an object, not a prefix: {object_url!r}")
        metadata = store.get_object_metadata_at(object_url)
        if metadata is None:
            raise ValueError(f"Object does not exist at {object_url!r}")

        filename = os.path.basename(path)
        content_type = metadata.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

        artifact_store = get_artifact_store()
        version = artifact_store.next_version(id)
        instance = cls(
            id=id,
            version=version,
            description=description,
            filename=filename,
            content_type=content_type,
            s3_url=object_url,
        )
        return artifact_store.put_document(instance)
