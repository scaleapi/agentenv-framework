"""Verify core A2A protocol support (message/send + tasks/get)."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class VerifyCoreA2AProtocolStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_core_protocol"
    entity_refs = (EntityRef.agent("a2a_agent_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        prompt_id: str,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.prompt_id = prompt_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        base["prompt_id"] = self.prompt_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyCoreA2AProtocolStep:
        return cls(**cls._base_from_dict(data), a2a_agent_id=data["a2a_agent_id"], prompt_id=data["prompt_id"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent

        prompt_response = next((pr for pr in context.prompt_responses if pr.prompt_id == self.prompt_id), None)

        # message/send succeeded if we have a prompt response at all
        message_send_ok = prompt_response is not None
        # tasks/get succeeded if the response has no error_type
        tasks_get_ok = message_send_ok and prompt_response.error_type is None

        logger.info(f"Core A2A protocol: message/send={message_send_ok} tasks/get={tasks_get_ok}")
        if not message_send_ok:
            logger.error("message/send failed: no prompt response found")
        elif not tasks_get_ok:
            logger.warning(f"tasks/get returned error: {prompt_response.error_type}")

        protocol = {
            "message/send": {"supported": message_send_ok},
            "tasks/get": {"supported": tasks_get_ok},
        }

        agent = A2AAgent.get(self.a2a_agent_id)
        agent.update_metadata({**agent.metadata, "validated_a2a_protocol": protocol})

        context.metadata.setdefault("verifications", {})["a2a_core_protocol"] = protocol
        return context
