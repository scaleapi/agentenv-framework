"""Task step that turns on the checks the roles it names can schedule for themselves on a deployed env's gateway
(notifications/v1).

With it each named role gets schedule_check, list_scheduled_checks and cancel_scheduled_check, on the env's virtual
clock; a check comes due for the role that scheduled it, and no other role sees the tools. Place it after
sync_env_clock (a check needs the clock) and before prompt_agent (the agent needs the tools on its first turn).
prompt_agent puts what has come due for its agent's role at the start of the agent's next message.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.env.gateway.constants import EXT_NOTIFICATIONS_URI
from agent_env.env.gateway.notifications import DEFAULT_MAX_ACTIVE, DEFAULT_MIN_INTERVAL
from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class ConfigureScheduledChecksTaskStep(TaskStep):
    type: ClassVar[str] = "configure_scheduled_checks"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        roles: list[str],
        max_active: int = DEFAULT_MAX_ACTIVE,
        min_interval: str = DEFAULT_MIN_INTERVAL,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        timeout_seconds: int = 30,
    ):
        """roles: the AgentEnv-Roles that get the tools ("default" is the solver's). max_active caps each role's checks
        still to come; min_interval is the shortest `every` it may ask for."""
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if (not isinstance(roles, list) or not roles
                or not all(isinstance(r, str) and r.strip() for r in roles)):
            raise ValueError('roles must name the roles that get the tools, e.g. ["default"] for the solver')
        if isinstance(max_active, bool) or not isinstance(max_active, int) or max_active < 1:
            raise ValueError("max_active must be a positive integer")
        self.env_id = env_id
        self.roles = roles
        self.max_active = max_active
        self.min_interval = min_interval
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["roles"] = self.roles
        base["max_active"] = self.max_active
        base["min_interval"] = self.min_interval
        base["timeout_seconds"] = self.timeout_seconds
        return base

    @classmethod
    def from_dict(cls, data: dict) -> ConfigureScheduledChecksTaskStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            roles=data.get("roles"),
            max_active=data.get("max_active", DEFAULT_MAX_ACTIVE),
            min_interval=data.get("min_interval", DEFAULT_MIN_INTERVAL),
            timeout_seconds=data.get("timeout_seconds", 30),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        if not deployed.supports(EXT_NOTIFICATIONS_URI, "config"):
            raise RuntimeError(f"Env '{self.env_id}' runs a gateway without {EXT_NOTIFICATIONS_URI}; redeploy it on a "
                               "gateway image that has it.")
        body = {"roles": self.roles, "max_active": self.max_active, "min_interval": self.min_interval}
        try:
            state = await deployed.invoke(EXT_NOTIFICATIONS_URI, "config", body, timeout=self.timeout_seconds)
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"notifications config failed (HTTP {e.response.status_code}): {e.response.text}") from e
        logger.info(f"configure_scheduled_checks completed for env={self.env_id} roles={self.roles}")
        context.metadata.setdefault("scheduled_check_configurations", []).append({
            "step_id": self.id, "env_id": self.env_id, **body,
            "applied": {k: state.get(k) for k in ("roles", "max_active", "min_interval")},
        })
        return context
