"""Change an armed env gateway clock's rate in place (clock/v1 set_rate), keeping its time and its time triggers' marks.

A rate of 0 stops the clock, and with it every time trigger and virtual wait. Arm the clock at rate 0 during setup and
set the task's rate right before prompt_agent, so the agent starts at the armed time; set 0 after the agent's last
prompt_agent, so verifiers read a world that no longer moves on its clock.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.entity.refs import EntityRef
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task_step.context import StepContextError, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class SetEnvClockRateTaskStep(TaskStep):
    type: ClassVar[str] = "set_env_clock_rate"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        virtual_seconds_per_real_second: float,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        timeout_seconds: int = 30,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        rate = virtual_seconds_per_real_second
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate < 0:
            raise ValueError(f"virtual_seconds_per_real_second must be a non-negative number, got {rate!r}")
        self.env_id = env_id
        self.virtual_seconds_per_real_second = rate
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["virtual_seconds_per_real_second"] = self.virtual_seconds_per_real_second
        base["timeout_seconds"] = self.timeout_seconds
        return base

    @classmethod
    def from_dict(cls, data: dict) -> SetEnvClockRateTaskStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            virtual_seconds_per_real_second=data["virtual_seconds_per_real_second"],
            timeout_seconds=data.get("timeout_seconds", 30),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise StepContextError(f"Env '{self.env_id}' not found in context.deployed_envs")
        try:
            state = await deployed.invoke(EXT_CLOCK_URI, "set_rate",
                                          {"virtual_seconds_per_real_second": self.virtual_seconds_per_real_second},
                                          timeout=self.timeout_seconds)
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"clock rate change failed (HTTP {e.response.status_code}): {e.response.text}") from e
        context.metadata.setdefault("clock_configurations", []).append({
            "step_id": self.id, "env_id": self.env_id,
            "virtual_seconds_per_real_second": self.virtual_seconds_per_real_second,
            "virtual_time": state.get("virtual_time"),
        })
        logger.info(f"set_env_clock_rate env={self.env_id}: rate={self.virtual_seconds_per_real_second} "
                    f"at {state.get('virtual_time')}")
        return context
