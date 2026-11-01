"""snapshot_computer_env: a computer env's files, kept by their paths in its environment artifact, which load_artifact
loads back into a computer."""

from __future__ import annotations

import asyncio
from typing import ClassVar, Optional

from agentenv_protocol.transfers import RelativePath
from pydantic import TypeAdapter, ValidationError

from agent_env.entity.refs import EntityRef, RefRole
from agent_env.env.env import Env
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.store.base import NotFoundError
from agent_env.store.ids import is_local_id, validate_local_id
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.snapshot_env import DEFAULT_EXPORT_TIMEOUT_SECONDS, SnapshotEnvTaskStep

_PATHS = TypeAdapter(list[RelativePath])


class SnapshotComputerEnvTaskStep(SnapshotEnvTaskStep):
    """Snapshot a deployed computer env's files into a new environment universe, as ``snapshot_env`` snapshots an env,
    each file under its path relative to the session user's home in the computer's environment artifact: those under
    ``file_paths``, by default the folders its card declares. ``load_artifact`` loads the snapshot back into a
    computer, through ``data/add``.

    Output: ``context.metadata["env_snapshotted_universes"][self.id]`` = ``{"id", "version"}``."""

    type: ClassVar[str] = "snapshot_computer_env"
    entity_refs = (
        EntityRef.env("env_id"),
        EntityRef.artifact("snapshot_id", role=RefRole.OUTPUT, artifact_type="environment_universe"),
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
        result = await self.snapshot_env_state(
            env=env, gateway_url=None, snapshot_id=snapshot_id, deployed=deployed,
            export_timeout_seconds=self.export_timeout_seconds,
            data_get_params=None if self.file_paths is None else {"paths": self.file_paths})
        context.metadata.setdefault("env_snapshotted_universes", {})[self.id] = {
            "id": result.environment_universe_artifact_id, "version": result.environment_universe_artifact_version}
        return context


def _not_a_computer(env: Env) -> str:
    return f"snapshot_computer_env snapshots a computer env, and {env.id} is a {env.type} env"
