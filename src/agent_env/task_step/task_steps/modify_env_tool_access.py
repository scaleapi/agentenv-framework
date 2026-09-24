"""Task step that toggles role-based tool access on a deployed env's gateway."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.env.gateway import TOOL_DISABLE_ACTION, TOOL_ENABLE_ACTION
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class ModifyEnvToolAccessStep(TaskStep):
    type: ClassVar[str] = "modify_env_tool_access"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        action: str,
        role: str,
        tools: list[str],
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if action not in (TOOL_DISABLE_ACTION, TOOL_ENABLE_ACTION):
            raise ValueError(f"action must be {TOOL_DISABLE_ACTION!r} or {TOOL_ENABLE_ACTION!r}, got {action!r}")
        if not isinstance(role, str) or not role:
            raise ValueError("role must be a non-empty string")
        if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
            raise ValueError("tools must be a list of strings")
        self.env_id = env_id
        self.action = action
        self.role = role
        self.tools = tools

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["action"] = self.action
        base["role"] = self.role
        base["tools"] = self.tools
        return base

    @classmethod
    def from_dict(cls, data: dict) -> ModifyEnvToolAccessStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            action=data["action"],
            role=data["role"],
            tools=data["tools"],
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        url = f"{deployed.gateway_url}/tools/{self.action}"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={"role": self.role, "tools": self.tools}, timeout=30)
            resp.raise_for_status()
            body = resp.json()
        logger.info(f"{self.action} role={self.role!r} tools={self.tools} on env={self.env_id}: {body}")
        context.metadata.setdefault("tool_access_changes", []).append({
            "step_id": self.id,
            "env_id": self.env_id,
            "action": self.action,
            "role": self.role,
            "tools_requested": self.tools,
            "role_state_after": {"disabled": body.get("disabled", []), "allowed": body.get("allowed", [])},
        })
        return context
