"""Reset a deployed env to a clean between-tasks state via ``Env.reset()``.

Env-agnostic: looks up the deployed env in context and calls its ``reset()``.
Best-effort by default — a reset failure (including an env that doesn't support
it) is logged and the run continues. Place right before prompt_agent in a
warm-reuse task.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep

logger = logging.getLogger(__name__)


class ResetEnvTaskStep(TaskStep):
    """Return a deployed env to a clean state via its ``Env.reset()``."""

    type = "reset_env"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        depends_on: Optional[list] = None,
        fail_task_on_error: bool = False,
    ):
        if not env_id:
            raise ValueError("reset_env requires an env_id (the env to reset)")
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "ResetEnvTaskStep":
        if not data.get("env_id"):
            raise ValueError("reset_env step requires 'env_id'")
        base = cls._base_from_dict(data)
        # Stay best-effort when unset: _base_from_dict would otherwise default it to True.
        base["fail_task_on_error"] = data.get("fail_task_on_error", False)
        return cls(**base, env_id=data["env_id"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.env.env import Env

        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            logger.warning("reset_env: no deployed env '%s' in context; skipping", self.env_id)
            return context
        try:
            env = Env.get(deployed.env_id, deployed.env_version)
            await env.reset(deployed)
            logger.info("reset_env: reset env '%s'", self.env_id)
        except NotImplementedError:
            # Env doesn't support reset — not an error.
            logger.info("reset_env: env '%s' does not support reset; skipping", self.env_id)
        except Exception as e:
            logger.warning("reset_env: reset failed (%s); continuing", repr(e)[:160])
            if self.fail_task_on_error:
                raise
        return context
