"""Verify the agent's extension routes take its control token: a call without it is refused with a 401, one with it is
not."""
from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.a2a_agent import A2AAgent
from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

# Reads the SDK serves for every agent that enables the extension, so the check changes nothing on the agent.
_READS = ((A2AAgent.EXT_MCP_CONFIG, "list"), (A2AAgent.EXT_AGENT_CONFIG, "get"))


class VerifyA2AControlTokenStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_control_token"
    entity_refs = (EntityRef.agent("a2a_agent_id", version_field="a2a_agent_version"),)

    def __init__(self, id: str, version: Optional[int], a2a_agent_id: str, a2a_agent_version: Optional[int] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.a2a_agent_version = a2a_agent_version

    def to_dict(self) -> dict:
        return {**super().to_dict(), "a2a_agent_id": self.a2a_agent_id, "a2a_agent_version": self.a2a_agent_version}

    @classmethod
    def from_dict(cls, data: dict) -> "VerifyA2AControlTokenStep":
        return cls(**cls._base_from_dict(data), a2a_agent_id=data["a2a_agent_id"],
                   a2a_agent_version=data.get("a2a_agent_version"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((a for a in context.deployed_agents if a.agent_name == TaskStep.DEFAULT_AGENT_NAME), None)
        if deployed is None:
            raise RuntimeError("VerifyA2AControlTokenStep: no deployed default-agent in context")
        card = deployed.a2a_card or {}
        route = next((path for uri, name in _READS
                      if (path := A2AAgent.operation(A2AAgent.find_extension(card, uri), name)[1]) is not None), None)
        entry: dict = {"route": route, "refuses_without_token": False, "accepts_with_token": False}
        if route is not None and deployed.control_token:
            url = (deployed.a2a_url or deployed.api_url) + route
            try:
                async with httpx.AsyncClient() as client:
                    without = await client.get(url, timeout=30)
                    with_token = await client.get(url, headers=deployed.control_headers, timeout=30)
                entry["refuses_without_token"] = without.status_code == 401
                entry["accepts_with_token"] = with_token.status_code != 401
            except httpx.HTTPError as e:
                logger.warning(f"VerifyA2AControlTokenStep: {url} unreachable: {e}")
        entry["passed"] = entry["refuses_without_token"] and entry["accepts_with_token"]
        context.metadata.setdefault("verifications", {})["a2a_control_token"] = entry
        logger.info(f"control token validation: {entry}")
        return context
