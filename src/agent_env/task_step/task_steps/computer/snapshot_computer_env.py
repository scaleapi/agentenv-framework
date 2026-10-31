"""snapshot_computer_env: a computer env's files, kept as a FileArtifactUniverse by their paths, which load_artifact
loads back into a computer."""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import PurePosixPath
from typing import ClassVar, Optional

from agentenv_protocol import uploaded_object_path
from agentenv_protocol.types import FileWithBytes
from agentenv_protocol.transfers import RelativePath
from pydantic import TypeAdapter, ValidationError

from agent_env.a2a_agent.object_transfer import ObjectLimits
from agent_env.artifact import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.entity.refs import EntityRef, RefRole
from agent_env.env import legacy_protocol
from agent_env.env.env import Env
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.store.base import NotFoundError
from agent_env.store.ids import derive_id, is_local_id, validate_local_id
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.snapshot_env import (
    DEFAULT_EXPORT_TIMEOUT_SECONDS,
    SnapshotEnvTaskStep,
    _snapshot_upload,
    get_data,
    registered_upload,
)

logger = logging.getLogger(__name__)

_PATHS = TypeAdapter(list[RelativePath])
#: What a computer may upload through its snapshot grant: each of its files, each as large as one upload to the store
#: can be, up to this in all.
FILE_TREE_LIMITS = ObjectLimits(max_objects=100_000, max_object_bytes=50 * 1024**3, max_total_bytes=50 * 1024**3)
#: How many of its files are written to the store at once.
_WRITES_IN_FLIGHT = 8


class SnapshotComputerEnvTaskStep(SnapshotEnvTaskStep):
    """Snapshot a deployed computer env's files into a new ``FileArtifactUniverse``, each under its path relative to
    the session user's home: those under ``file_paths``, by default the folders its card declares. The computer
    uploads each file through a grant when the store's grants reach it, and returns it inline otherwise.
    ``load_artifact`` loads the universe back into a computer, through ``data/add``.

    Picks the deployment and names the snapshot as ``snapshot_env`` does. Output:
    ``context.metadata["env_snapshotted_universes"][self.id]`` = ``{"id", "version"}``."""

    type: ClassVar[str] = "snapshot_computer_env"
    entity_refs = (
        EntityRef.env("env_id"),
        EntityRef.artifact("snapshot_id", role=RefRole.OUTPUT, artifact_type="file_artifact_universe"),
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: Optional[str] = None,
        env_instance_id: Optional[str] = None,
        env_step_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        file_paths: Optional[list[str]] = None,
        export_timeout_seconds: int = DEFAULT_EXPORT_TIMEOUT_SECONDS,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, env_id=env_id, env_instance_id=env_instance_id, env_step_id=env_step_id,
                         snapshot_id=snapshot_id, export_timeout_seconds=export_timeout_seconds,
                         depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.file_paths = file_paths

    def to_dict(self) -> dict:
        return {**TaskStep.to_dict(self), "env_id": self.env_id, "env_instance_id": self.env_instance_id,
                "env_step_id": self.env_step_id, "snapshot_id": self.snapshot_id, "file_paths": self.file_paths,
                "export_timeout_seconds": self.export_timeout_seconds}

    @classmethod
    def from_dict(cls, data: dict) -> SnapshotComputerEnvTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data.get("env_id"), env_instance_id=data.get("env_instance_id"),
                   env_step_id=data.get("env_step_id"), snapshot_id=data.get("snapshot_id"),
                   file_paths=data.get("file_paths"),
                   export_timeout_seconds=data.get("export_timeout_seconds", DEFAULT_EXPORT_TIMEOUT_SECONDS))

    def config_problems(self) -> list[str]:
        problems = super().config_problems()
        if self.file_paths is not None:
            try:
                _PATHS.validate_python(self.file_paths)
            except ValidationError:
                problems.append(f"file_paths are normalized paths relative to the computer's session user's home, "
                                f"not {self.file_paths!r}")
        return problems

    def preflight(self) -> list[str]:
        """The env this step names when it isn't a computer env."""
        problems = super().preflight()
        if self.env_id is None:
            return problems
        try:
            env = Env.get(self.env_id)
        except NotFoundError:  # what the run reports
            return problems
        return problems if isinstance(env, ComputerEnv) else [*problems, _not_a_computer(env)]

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        snapshot_id = self._derive_snapshot_id(context)
        if is_local_id(snapshot_id):
            validate_local_id(snapshot_id)
        deployed = self._resolve_deployed_env(context)
        env = await asyncio.to_thread(Env.get, deployed.env_id, deployed.env_version)
        if not isinstance(env, ComputerEnv):
            raise RuntimeError(_not_a_computer(env))
        files = await _files(deployed, env, snapshot_id, self.file_paths, self.export_timeout_seconds)
        if not files:
            under = ", ".join(self.file_paths) if self.file_paths is not None else "the folders its card declares"
            raise RuntimeError(f"computer env {env.id} has no files under {under} to snapshot")
        universe = await asyncio.to_thread(FileArtifactUniverse.put, snapshot_id, file_artifacts=files)
        logger.info(f"snapshot_computer_env: {env.id} -> {universe.id} v{universe.version} ({len(files)} files)")
        context.metadata.setdefault("env_snapshotted_universes", {})[self.id] = {"id": universe.id,
                                                                                 "version": universe.version}
        return context


async def _files(deployed, env: ComputerEnv, snapshot_id: str, paths: Optional[list[str]],
                 timeout_seconds: float) -> dict[str, FileArtifact]:
    """Each file ``env``'s own data/get returns under ``paths``, by its path, written to the store as ``<snapshot
    id>__file-<index>``: an upload under the grant in place, without a download, else its inline bytes."""
    name = env.environment_name
    base_url = await legacy_protocol.v1_base_url(deployed, None, name)
    _, card = await legacy_protocol.child_env_card(deployed, None, name)
    if base_url is None:
        raise RuntimeError(f"computer env {env.id}'s recorded card names no environment {name!r}")
    upload = await asyncio.to_thread(_snapshot_upload, card or {}, timeout_seconds, deployed.sandbox_type,
                                     FILE_TREE_LIMITS)
    resp, upload = await get_data(base_url, name, timeout_seconds, upload,
                                  **({} if paths is None else {"paths": paths}))
    writing = asyncio.Semaphore(_WRITES_IN_FLIGHT)

    async def written(index: int, part) -> tuple[str, FileArtifact]:
        path = uploaded_object_path(part)
        if path is None:
            raise RuntimeError(f"computer env {env.id}'s data/get returned a {part.kind} part naming no path")
        file_id, filename = derive_id(snapshot_id, f"file-{index}"), PurePosixPath(path).name
        content_type, description = part.file.mimeType or "application/octet-stream", f"{path} of computer env {env.id}"
        async with writing:
            if isinstance(part.file, FileWithBytes):
                return path, await asyncio.to_thread(FileArtifact.put_bytes, file_id, description=description,
                                                     filename=filename, content=base64.b64decode(part.file.bytes),
                                                     content_type=content_type)
            if upload is None:
                raise RuntimeError(f"computer env {env.id}'s data/get returned {path} neither inline nor uploaded")
            return path, await asyncio.to_thread(registered_upload, upload.object_url(path), file_id, filename,
                                                 content_type, description)

    return dict(await asyncio.gather(*(written(index, part) for index, part in enumerate(resp.parts))))


def _not_a_computer(env: Env) -> str:
    return f"snapshot_computer_env snapshots a computer env, and {env.id} is a {env.type} env"
