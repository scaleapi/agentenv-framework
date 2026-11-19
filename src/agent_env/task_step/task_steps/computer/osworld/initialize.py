"""``osworld_initialize``: a classic OSWorld task set up on a computer that runs the ``osworld-server`` app: its
``init_config``, OSWorld's ``[{type, parameters}]``, run by upstream OSWorld's ``SetupController`` in the worker
(:mod:`.runtime`) against the computer's guest server, with the env's control token, which the agent doesn't hold. An
OSWorld-V2 task (``osworld_v2_task_path``), whose ``TASK_CLASS.setup()`` is the task's own code, isn't run here: the
step fails saying so, rather than leave the computer unset."""
from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import StepContextError, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.computer.osworld import guest, runtime

logger = logging.getLogger(__name__)

#: How much of a setup error an exception or a log line carries: the rest can be large and hold the task's content.
SETUP_ERROR_CHARS = 300
V2_UNSUPPORTED = ("OSWorld-V2 tasks (osworld_v2_task_path), whose setup and evaluate are the task's own code, aren't run "
                  "by the osworld_* steps")


class OSWorldInitializeTaskStep(TaskStep):
    type: ClassVar[str] = "osworld_initialize"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, init_config: Optional[list[dict]] = None,
                 osworld_v2_task_path: Optional[str] = None, osworld_v2_task_url: Optional[str] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        """``osworld_v2_task_url`` is the field the hub's CUA importer writes an OSWorld-V2 task's URL to; either one
        fails the step (``V2_UNSUPPORTED``)."""
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.init_config = init_config or []
        self.osworld_v2_task_path = osworld_v2_task_path or osworld_v2_task_url

    def to_dict(self) -> dict:
        base = {**super().to_dict(), "env_id": self.env_id, "init_config": self.init_config}
        if self.osworld_v2_task_path is not None:
            base["osworld_v2_task_path"] = self.osworld_v2_task_path
        return base

    @classmethod
    def from_dict(cls, data: dict) -> OSWorldInitializeTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], init_config=data.get("init_config"),
                   osworld_v2_task_path=data.get("osworld_v2_task_path"),
                   osworld_v2_task_url=data.get("osworld_v2_task_url"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise StepContextError(f"Env '{self.env_id}' not found in context.deployed_envs")
        if self.osworld_v2_task_path:
            raise ValueError(f"osworld_initialize '{self.id}': {V2_UNSUPPORTED}")
        if self.init_config:
            guest_url, headers = await guest.guest_access(deployed)
            await run_setup(guest_url, headers, self.init_config, what=f"osworld_initialize '{self.id}'")
        return context


async def run_setup(guest_url: str, headers: dict[str, str], config: list[dict], *, what: str) -> None:
    """``config`` run on the guest (``runtime.setup``); raises with what failed, bounded."""
    if not isinstance(config, list):
        raise ValueError(f"{what}: a setup config must be a list of {{type, parameters}}, got {type(config).__name__}")
    try:
        await runtime.run_setup(guest_url, headers, config)
    except Exception as e:
        raise RuntimeError(f"{what}: setup failed: {type(e).__name__}: {str(e)[:SETUP_ERROR_CHARS]}") from e
