"""snapshot_computer_env: snapshot_env for a computer env, the files its snapshot keeps named by their paths."""

from __future__ import annotations

from typing import ClassVar, Optional

from agentenv_protocol.transfers import RelativePath
from pydantic import TypeAdapter, ValidationError

from agent_env.env.env import Env
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.store.base import NotFoundError
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.task_step.task_steps.snapshot_env import DEFAULT_EXPORT_TIMEOUT_SECONDS, SnapshotEnvTaskStep

_PATHS = TypeAdapter(list[RelativePath])


class SnapshotComputerEnvTaskStep(SnapshotEnvTaskStep):
    """``snapshot_env`` for a computer env: ``file_paths``, each relative to its session user's home, names the files
    and folders its own data/get returns, by default the folders its card declares. Any other environment it serves,
    an app say, is snapshotted as ``snapshot_env`` snapshots it."""

    type: ClassVar[str] = "snapshot_computer_env"
    entity_refs = SnapshotEnvTaskStep.entity_refs

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: Optional[str] = None,
        env_instance_id: Optional[str] = None,
        env_step_id: Optional[str] = None,
        gateway_url: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        original_universe_artifact_id: Optional[str] = None,
        export_timeout_seconds: int = DEFAULT_EXPORT_TIMEOUT_SECONDS,
        include_env_trajectory: bool = False,
        include_env_changes: bool = False,
        file_paths: Optional[list[str]] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, env_id=env_id, env_instance_id=env_instance_id, env_step_id=env_step_id,
                         gateway_url=gateway_url, snapshot_id=snapshot_id,
                         original_universe_artifact_id=original_universe_artifact_id,
                         export_timeout_seconds=export_timeout_seconds, include_env_trajectory=include_env_trajectory,
                         include_env_changes=include_env_changes, depends_on=depends_on,
                         fail_task_on_error=fail_task_on_error)
        self.file_paths = file_paths

    def to_dict(self) -> dict:
        return {**super().to_dict(), "file_paths": self.file_paths}

    @classmethod
    def from_dict(cls, data: dict) -> SnapshotComputerEnvTaskStep:
        step = super().from_dict(data)
        step.file_paths = data.get("file_paths")
        return step

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

    def _data_get_params(self, env) -> dict[str, dict]:
        if not isinstance(env, ComputerEnv):
            raise RuntimeError(_not_a_computer(env))
        return {env.environment_name: {"paths": self.file_paths}} if self.file_paths is not None else {}


def _not_a_computer(env: Env) -> str:
    return f"snapshot_computer_env snapshots a computer env, and {env.id} is a {env.type} env"
