"""Verify a deployed environment's composed EnvironmentCard and persist it to env metadata."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

REQUIRED_CARD_FIELDS = ["name", "protocolVersion", "url", "preferredTransport", "capabilities"]

# Env-doc metadata keys the validator writes (read back by the CLIs / hub consumers).
ENVIRONMENT_CARD_KEY = "environment_card"
VALIDATED_ENVIRONMENT_CARD_KEY = "validated_environment_card"


class VerifyEnvironmentCardStep(TaskStep):
    type: ClassVar[str] = "verify_env_card"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyEnvironmentCardStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agentenv_protocol import client as protocol_v1
        from agent_env.env.env import Env

        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")

        card = None
        error = None
        try:
            card = await protocol_v1.get_card(deployed.gateway_url)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.warning(f"env card for '{self.env_id}' inaccessible at {deployed.gateway_url}: {error}")

        env = Env.get(self.env_id, deployed.env_version)

        if not card:
            validated = {"accessible": False, "error": error or "empty card"}
            env.update_metadata({**env.metadata, VALIDATED_ENVIRONMENT_CARD_KEY: validated})
            context.metadata.setdefault("verifications", {})["environment_card"] = {VALIDATED_ENVIRONMENT_CARD_KEY: validated}
            return context

        required_fields = {}
        missing_required = []
        for field in REQUIRED_CARD_FIELDS:
            present = field in card and bool(card[field])
            required_fields[field] = {"present": present}
            if not present:
                missing_required.append(field)
        if missing_required:
            logger.warning(f"env card '{self.env_id}' missing required fields: {missing_required}")

        children = card.get("children_environments") or []
        capabilities = card.get("capabilities") or {}
        extensions = [uri for e in capabilities.get("extensions") or [] if (uri := e.get("uri"))]
        tools = [name for t in capabilities.get("tools") or [] if (name := t.get("name"))]
        logger.info(f"env card '{self.env_id}': name={card.get('name')} children={len(children)} extensions={extensions} tools={tools}")

        registered_name = getattr(env, "environment_name", None)
        served_names = [c.get("name") for c in children if c.get("name")]
        name_check = None
        if registered_name and served_names:
            matches = registered_name in served_names
            name_check = {"registered_name": registered_name, "served_card_names": served_names, "matches": matches}
            if not matches:
                logger.warning(
                    f"env card name drift for '{self.env_id}': registered environment_name={registered_name!r} is not "
                    f"among the served card names {served_names!r} — the server likely declares a different "
                    f"@environment_card(name=...); registration should match the card."
                )

        validated = {
            "accessible": True,
            "required_fields": required_fields,
            "children_count": len(children),
            "extensions": extensions,
            "tools": tools,
            "name_check": name_check,
        }
        env.update_metadata({**env.metadata, ENVIRONMENT_CARD_KEY: card, VALIDATED_ENVIRONMENT_CARD_KEY: validated})

        context.metadata.setdefault("verifications", {})["environment_card"] = {ENVIRONMENT_CARD_KEY: card, VALIDATED_ENVIRONMENT_CARD_KEY: validated}
        return context
