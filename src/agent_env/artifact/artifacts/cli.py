"""CliArtifact — packages a CLI tool (entrypoint + supporting files) as a versioned artifact.

A CliArtifact references one FileArtifactUniverse bundling every file in the CLI
directory tree, colocated under one object-store prefix per write (`ArtifactStore.attempt_prefix`).
The `entrypoint` field is a relative path within that prefix that consumers
(e.g. an agent gateway) chmod +x and symlink onto PATH.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Mapping, Optional

from pydantic import ConfigDict, Field, model_serializer

from agent_env.artifact.artifact import Artifact, _write_twin
from agent_env.store.ids import derive_id

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
    from agent_env.bundle.authoring import AuthoringContext

# A command a bundle's CLI is installed as, at /opt/cli/<command_name>/.
_COMMAND_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
# The file installing a CLI writes the gateway URL to, beside its entrypoint.
GATEWAY_ENV_FILE = ".env"


class CliArtifact(Artifact):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["cli"] = "cli"

    cli_files_id: str = Field(description="ID of the FileArtifactUniverse bundling every file in the CLI tree")
    cli_object_url: str = Field(alias="cli_s3_url", description="Object-store prefix where the CLI bundle lives")
    entrypoint: str = Field(description="Relative path within the bundle to the executable file (e.g. 'bin/slack')")
    command_name: str = Field(description="Command name installed on PATH (e.g. 'slack' -> /usr/local/bin/slack)")

    env_id: Optional[str] = Field(default=None, description="Source env id this CLI was generated from, if any")
    env_version: Optional[int] = Field(default=None, description="Source env version this CLI was generated from, if any")

    toml_keys: ClassVar[dict[str, type]] = {"command_name": str, "entrypoint": str}  # what an artifact.toml sets

    # No return annotation: pydantic builds the serialization schema from one, and a dict drops the fields.
    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any):
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
        cli_dir: Optional[Path] = None,
        files: Optional[Mapping[str, Path]] = None,
        env_id: Optional[str] = None,
        env_version: Optional[int] = None,
    ) -> "CliArtifact":
        """Write the CLI in ``cli_dir``, or the ``files`` given by their paths in the CLI, ``entrypoint`` among them."""
        from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
        from agent_env.artifact.store import get_artifact_store

        if (cli_dir is None) == (files is None):
            raise ValueError("specify exactly one of cli_dir or files")
        if cli_dir is not None:
            if not cli_dir.is_dir():
                raise ValueError(f"cli_dir {cli_dir} is not a directory")
            if not (cli_dir / entrypoint).is_file():
                raise ValueError(f"entrypoint {entrypoint!r} not found at {cli_dir / entrypoint}")
            files = {p.relative_to(cli_dir).as_posix(): p for p in sorted(cli_dir.rglob("*")) if p.is_file()}
        elif entrypoint not in files:
            raise ValueError(f"entrypoint {entrypoint!r} isn't among the CLI's files")

        store = get_artifact_store()
        version = store.next_version(id)
        cli_object_url = store.attempt_prefix(cls.model_fields["type"].default, id)

        universe = FileArtifactUniverse.put_bundled(id=derive_id(id, "files"), files=dict(files),
                                                    prefix_url=cli_object_url)

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

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an artifact.toml, a CLI takes, checked against the folder's files the way installing
        it reads them. Raises BundleError listing every problem."""
        files = ctx.try_files(empty_ok=True)  # None when the plan's walk reports why
        problems = [ctx.no_files("files to write")] if files == {} else []
        fields, more = ctx.accepted(data, **cls.toml_keys)
        problems += more
        for key in cls.toml_keys:
            if key not in data:
                problems.append(ctx.config_problem(f"{key} is required"))
        command_name, entrypoint = fields.get("command_name"), fields.get("entrypoint")
        if command_name is not None and not _COMMAND_NAME.fullmatch(command_name):
            problems.append(ctx.config_problem(
                f"command_name must be a plain name, of letters, digits, '.', '_' and '-' and not starting with '.' or "
                f"'-', since the CLI is installed at /opt/cli/<command_name>; not {command_name!r}"))
        if entrypoint is not None and files:
            gateway_env = posixpath.join(posixpath.dirname(entrypoint), GATEWAY_ENV_FILE)
            if entrypoint not in files:
                problems.append(ctx.config_problem(f"entrypoint {entrypoint!r} isn't one of this folder's files"))
            elif any(name == gateway_env or name.startswith(f"{gateway_env}/") for name in files):
                problems.append(f"{gateway_env}: installing the CLI writes the gateway URL to this file, beside its "
                                "entrypoint; rename or remove the folder's own")
        if problems:
            ctx.refuse(problems)
        return fields

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> "CliArtifact":
        """Write the CLI in the bundle folder: every file its folder walk lists."""
        fields = cls.accept_toml(data, ctx)
        return cls.put(ctx.id, command_name=fields["command_name"], entrypoint=fields["entrypoint"],
                       files=ctx.files())

    def get_cli_files(self) -> "FileArtifactUniverse":
        from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
        return FileArtifactUniverse.get(self.cli_files_id)
