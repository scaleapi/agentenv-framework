"""Task step that turns on the checks an agent can schedule for itself on a deployed env's gateway (wakeups/v1).

With it the agent gets schedule_check, list_scheduled_checks and cancel_scheduled_check, on the env's virtual clock.
Place it after sync_env_clock (a check needs the clock) and before prompt_agent (the agent needs the tools on its
first turn). A post-turn hook reads the gateway's wakeups state to wake the agent when a check comes due.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.env.gateway.constants import EXT_WAKEUPS_URI
from agent_env.env.gateway.wakeups import DEFAULT_MAX_ACTIVE, DEFAULT_MIN_INTERVAL
from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class ConfigureWakeupsTaskStep(TaskStep):
    type: ClassVar[str] = "configure_wakeups"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        scheduled_checks: bool = True,
        max_active: int = DEFAULT_MAX_ACTIVE,
        min_interval: str = DEFAULT_MIN_INTERVAL,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        timeout_seconds: int = 30,
    ):
        """max_active caps the agent's checks still to come; min_interval is the shortest `every` it may ask for."""
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not isinstance(scheduled_checks, bool):
            raise ValueError("scheduled_checks must be a boolean")
        if isinstance(max_active, bool) or not isinstance(max_active, int) or max_active < 1:
            raise ValueError("max_active must be a positive integer")
        self.env_id = env_id
        self.scheduled_checks = scheduled_checks
        self.max_active = max_active
        self.min_interval = min_interval
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["scheduled_checks"] = self.scheduled_checks
        base["max_active"] = self.max_active
        base["min_interval"] = self.min_interval
        base["timeout_seconds"] = self.timeout_seconds
        return base

    @classmethod
    def from_dict(cls, data: dict) -> ConfigureWakeupsTaskStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            scheduled_checks=data.get("scheduled_checks", True),
            max_active=data.get("max_active", DEFAULT_MAX_ACTIVE),
            min_interval=data.get("min_interval", DEFAULT_MIN_INTERVAL),
            timeout_seconds=data.get("timeout_seconds", 30),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        if not deployed.supports(EXT_WAKEUPS_URI, "config"):
            raise RuntimeError(f"Env '{self.env_id}' runs a gateway without {EXT_WAKEUPS_URI}; redeploy it on a gateway "
                               "image that has it.")
        body = {"scheduled_checks": self.scheduled_checks, "max_active": self.max_active,
                "min_interval": self.min_interval}
        try:
            state = await deployed.invoke(EXT_WAKEUPS_URI, "config", body, timeout=self.timeout_seconds)
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"wakeups config failed (HTTP {e.response.status_code}): {e.response.text}") from e
        logger.info(f"configure_wakeups env={self.env_id}: {body}")
        context.metadata.setdefault("wakeup_configurations", []).append({
            "step_id": self.id, "env_id": self.env_id, **body,
            "applied": {k: state.get(k) for k in ("scheduled_checks", "max_active", "min_interval")},
        })
        return context
