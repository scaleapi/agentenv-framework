"""Task step that registers env triggers (urn:agentenv:triggers/v1) on a deployed env's gateway."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class RegisterEnvTriggersStep(TaskStep):
    type: ClassVar[str] = "register_env_triggers"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        triggers: list[dict],
        watch_roles: Optional[list[str]] = None,
        executor_agent_name: Optional[str] = None,
        executor_timeout_seconds: float = 120,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not isinstance(triggers, list) or not all(isinstance(t, dict) for t in triggers):
            raise ValueError("triggers must be a list of objects")
        self.env_id = env_id
        self.triggers = triggers
        self.watch_roles = watch_roles
        self.executor_agent_name = executor_agent_name
        self.executor_timeout_seconds = executor_timeout_seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["triggers"] = self.triggers
        base["watch_roles"] = self.watch_roles
        base["executor_agent_name"] = self.executor_agent_name
        base["executor_timeout_seconds"] = self.executor_timeout_seconds
        return base

    @classmethod
    def from_dict(cls, data: dict) -> RegisterEnvTriggersStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            triggers=data["triggers"],
            watch_roles=data.get("watch_roles"),
            executor_agent_name=data.get("executor_agent_name"),
            executor_timeout_seconds=data.get("executor_timeout_seconds", 120),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        body: dict = {"triggers": self.triggers}
        if self.watch_roles is not None:
            body["watch_roles"] = self.watch_roles
        if self.executor_agent_name is not None:
            agent = next((a for a in context.deployed_agents if a.agent_name == self.executor_agent_name), None)
            if agent is None:
                raise RuntimeError(f"Executor agent '{self.executor_agent_name}' not found in context.deployed_agents")
            if not agent.role:
                raise RuntimeError(
                    f"Executor agent '{self.executor_agent_name}' was deployed without an explicit role; "
                    f"it would run under the default role and its own tool calls would fire triggers. "
                    f"Deploy it with a role outside watch_roles.")
            if self.watch_roles is not None and agent.role in self.watch_roles:
                raise RuntimeError(
                    f"Executor agent role '{agent.role}' is in watch_roles {self.watch_roles} — "
                    f"the executor's own tool calls would fire triggers (cascade).")
            body["executor"] = {"a2a_url": agent.a2a_url or agent.api_url,
                                "timeout_seconds": self.executor_timeout_seconds,
                                "role": agent.role}
        url = f"{deployed.gateway_url}/triggers/register"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, timeout=30)
            if resp.status_code >= 400:
                raise RuntimeError(f"trigger registration failed (HTTP {resp.status_code}): {resp.text}")
            result = resp.json()
        logger.info(f"registered triggers on env={self.env_id}: {result}")
        context.metadata.setdefault("env_trigger_registrations", []).append({
            "step_id": self.id,
            "env_id": self.env_id,
            "added": result.get("added", []),
            "executor_agent_name": self.executor_agent_name,
        })
        return context
