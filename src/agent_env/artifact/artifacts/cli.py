"""CliArtifact — packages a CLI tool (entrypoint + supporting files) as a versioned artifact.

A CliArtifact references one FileArtifactUniverse bundling every file in the CLI
directory tree, colocated under a single object-store prefix (`artifacts/cli/<key_segment(id)>/<version>/`).
The `entrypoint` field is a relative path within that prefix that consumers
(e.g. an agent gateway) chmod +x and symlink onto PATH.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import ConfigDict, Field, model_serializer

from agent_env.artifact.artifact import Artifact, _write_twin
from agent_env.config import get_config
from agent_env.store.ids import derive_id, key_segment

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse


class CliArtifact(Artifact):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["cli"] = "cli"

    cli_files_id: str = Field(description="ID of the FileArtifactUniverse bundling every file in the CLI tree")
    cli_object_url: str = Field(alias="cli_s3_url", description="Object-store prefix where the CLI bundle lives")
    entrypoint: str = Field(description="Relative path within the bundle to the executable file (e.g. 'bin/slack')")
    command_name: str = Field(description="Command name installed on PATH (e.g. 'slack' -> /usr/local/bin/slack)")

    env_id: Optional[str] = Field(default=None, description="Source env id this CLI was generated from, if any")
    env_version: Optional[int] = Field(default=None, description="Source env version this CLI was generated from, if any")

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        # Dual-write, from the attribute: `alias=` emits one spelling, which one depending on the caller's `by_alias`.
        data = handler(self)
        _write_twin(data, "cli_s3_url", "cli_object_url", self.cli_object_url)
        return data

    @classmethod
    def put(
        cls,
        id: str,
        *,
        command_name: str,
        entrypoint: str,
        cli_dir: Path,
        env_id: Optional[str] = None,
        env_version: Optional[int] = None,
    ) -> "CliArtifact":
        from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
        from agent_env.artifact.store import get_artifact_store

        if not cli_dir.is_dir():
            raise ValueError(f"cli_dir {cli_dir} is not a directory")
        entrypoint_path = cli_dir / entrypoint
        if not entrypoint_path.is_file():
            raise ValueError(f"entrypoint {entrypoint!r} not found at {entrypoint_path}")

        files: dict[str, Path] = {}
        for p in sorted(cli_dir.rglob("*")):
            if p.is_file():
                files[p.relative_to(cli_dir).as_posix()] = p
        if not files:
            raise ValueError(f"cli_dir {cli_dir} contains no files")

        store = get_artifact_store()
        version = store.next_version(id)
        key = f"{get_config().get_artifact_key_prefix()}artifacts/cli/{key_segment(id)}/{version}/"
        cli_object_url = get_config().get_object_store_for(id).object_url(key)

        universe = FileArtifactUniverse.put_bundled(id=derive_id(id, "files"), files=files, prefix_url=cli_object_url)

        instance = cls(
            id=id,
            version=version,
            cli_files_id=universe.id,
            cli_object_url=cli_object_url,
            entrypoint=entrypoint,
            command_name=command_name,
            env_id=env_id,
            env_version=env_version,
        )
        return store.put_document(instance)

    def get_cli_files(self) -> "FileArtifactUniverse":
        from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
        return FileArtifactUniverse.get(self.cli_files_id)
