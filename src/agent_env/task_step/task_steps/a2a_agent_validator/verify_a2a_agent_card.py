"""Verify A2A agent card structure and persist to agent metadata."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

REQUIRED_CARD_FIELDS = ["name", "description", "url", "version", "capabilities", "defaultInputModes", "defaultOutputModes", "skills"]
OPTIONAL_CARD_FIELDS = ["authentication", "provider", "documentationUrl"]


class VerifyA2AAgentCardStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_agent_card"
    entity_refs = (EntityRef.agent("a2a_agent_id", version_field="a2a_agent_version"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        a2a_agent_version: Optional[int] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.a2a_agent_version = a2a_agent_version

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        base["a2a_agent_version"] = self.a2a_agent_version
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyA2AAgentCardStep:
        return cls(**cls._base_from_dict(data), a2a_agent_id=data["a2a_agent_id"], a2a_agent_version=data.get("a2a_agent_version"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent

        deployed = next((a for a in context.deployed_agents), None)
        if deployed is None:
            raise RuntimeError("No deployed agent found in context")

        card = deployed.a2a_card
        agent = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)

        # Validate card is accessible
        if not card:
            logger.error("Agent card is inaccessible or empty")
            validated = {"accessible": False, "error": "Inaccessible Agent Card"}
            agent.update_metadata({**agent.metadata, "validated_agent_card": validated})
            raise RuntimeError("Inaccessible Agent Card: agent did not return a valid agent card at /.well-known/agent.json")

        logger.info(f"Agent card for '{self.a2a_agent_id}': name={card.get('name')}")

        # Check required fields
        required_fields = {}
        missing_required = []
        for field in REQUIRED_CARD_FIELDS:
            value = card.get(field)
            present = field in card and (
                bool(value) or (field == "skills" and value == [])
            )
            required_fields[field] = {"present": present}
            if not present:
                missing_required.append(field)

        if missing_required:
            logger.warning(f"Agent card missing required fields: {missing_required}")
        else:
            logger.info("Agent card has all required fields")

        # Check optional fields
        optional_fields = {}
        missing_optional = []
        for field in OPTIONAL_CARD_FIELDS:
            present = field in card and bool(card[field])
            optional_fields[field] = {"present": present}
            if not present:
                missing_optional.append(field)

        if missing_optional:
            logger.info(f"Agent card missing optional fields: {missing_optional}")

        # Log which extensions the card advertises
        for ext_uri in [A2AAgent.EXT_MCP_CONFIG, A2AAgent.EXT_SKILL_CONFIG, A2AAgent.EXT_TRAJECTORY, A2AAgent.EXT_AGENT_CONFIG, A2AAgent.EXT_TOOLS]:
            ext = A2AAgent.find_extension(card, ext_uri)
            logger.info(f"  {ext_uri}: {'advertised' if ext else 'not advertised'}")

        # Persist agent_card (raw) and validated_agent_card (our validation)
        validated = {"accessible": True, "required_fields": required_fields, "optional_fields": optional_fields}
        agent.update_metadata({**agent.metadata, "agent_card": card, "validated_agent_card": validated})

        context.metadata.setdefault("verifications", {})["a2a_agent_card"] = {"agent_card": card, "validated_agent_card": validated}
        return context
