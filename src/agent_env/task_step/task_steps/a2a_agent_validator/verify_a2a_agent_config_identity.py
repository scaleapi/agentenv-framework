"""Verify the A2A agent-config extension actually mutates the served card."""
from __future__ import annotations

import logging
import uuid
from typing import ClassVar, Optional

import httpx

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class VerifyA2AAgentConfigIdentityStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_agent_config_identity"
    entity_refs = (EntityRef.agent("a2a_agent_id", version_field="a2a_agent_version"),)

    def __init__(self, id: str, version: Optional[int], a2a_agent_id: str, a2a_agent_version: Optional[int] = None, depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.a2a_agent_version = a2a_agent_version

    def to_dict(self) -> dict:
        return {**super().to_dict(), "a2a_agent_id": self.a2a_agent_id, "a2a_agent_version": self.a2a_agent_version}

    @classmethod
    def from_dict(cls, data: dict) -> "VerifyA2AAgentConfigIdentityStep":
        return cls(**cls._base_from_dict(data), a2a_agent_id=data["a2a_agent_id"], a2a_agent_version=data.get("a2a_agent_version"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((a for a in context.deployed_agents if a.agent_name == TaskStep.DEFAULT_AGENT_NAME), None)
        if deployed is None:
            raise RuntimeError("VerifyA2AAgentConfigIdentityStep: no deployed default-agent in context")
        a2a_url = deployed.a2a_url or deployed.api_url
        ext = A2AAgent.find_extension(deployed.a2a_card or {}, A2AAgent.EXT_AGENT_CONFIG)
        supported = ((ext or {}).get("params") or {}).get("methods", {}).get("set", {}).get("request", {}).get("supported", [])
        supports_identity = "name" in supported and "description" in supported

        identity_round_trip_ok = False
        if ext is not None and supports_identity:
            name, desc = f"IdentityName-{uuid.uuid4().hex[:12]}", f"IdentityDesc-{uuid.uuid4().hex[:12]}"
            endpoint = a2a_url + (ext.get("params") or {}).get("endpoint", "/ext/agent-config")
            try:
                async with httpx.AsyncClient() as client:
                    (await client.post(endpoint, json={"name": name, "description": desc}, timeout=30)).raise_for_status()
                    card_resp = await client.get(f"{a2a_url}/.well-known/agent.json", timeout=30)
                    card_resp.raise_for_status()
                    card = card_resp.json() or {}
                identity_round_trip_ok = card.get("name") == name and card.get("description") == desc
            except httpx.HTTPError as e:
                logger.warning(f"VerifyA2AAgentConfigIdentityStep: round-trip failed: {e}")

        entry = {
            "supported": ext is not None and supports_identity and identity_round_trip_ok,
            "extension_advertised": ext is not None,
            "supports_identity": supports_identity,
            "identity_round_trip_ok": identity_round_trip_ok,
        }
        agent = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)
        validated_ext = agent.metadata.get("validated_a2a_extensions", {})
        validated_ext[A2AAgent.EXT_AGENT_CONFIG] = entry
        agent.update_metadata({**agent.metadata, "validated_a2a_extensions": validated_ext})
        context.metadata.setdefault("verifications", {})["a2a_agent_config_identity"] = entry
        logger.info(f"agent-config identity validation: {entry}")
        return context
