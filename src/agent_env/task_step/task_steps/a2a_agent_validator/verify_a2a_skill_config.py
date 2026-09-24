"""Verify A2A agent skill-config extension by reading rubric results and listing skills."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class VerifyA2ASkillConfigStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_skill_config"
    entity_refs = (EntityRef.agent("a2a_agent_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        rubric_verifier_id: str,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.rubric_verifier_id = rubric_verifier_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        base["rubric_verifier_id"] = self.rubric_verifier_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyA2ASkillConfigStep:
        return cls(**cls._base_from_dict(data), a2a_agent_id=data["a2a_agent_id"], rubric_verifier_id=data["rubric_verifier_id"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent

        deployed_agent = next((a for a in context.deployed_agents), None)
        if deployed_agent is None:
            raise RuntimeError("No deployed agent found in context")

        card = deployed_agent.a2a_card or {}
        skill_ext = A2AAgent.find_extension(card, A2AAgent.EXT_SKILL_CONFIG)

        if not skill_ext:
            logger.warning("Skill-config extension not advertised in agent card")
            agent = A2AAgent.get(self.a2a_agent_id)
            validated_ext = agent.metadata.get("validated_a2a_extensions", {})
            validated_ext[A2AAgent.EXT_SKILL_CONFIG] = {"supported": False}
            agent.update_metadata({**agent.metadata, "validated_a2a_extensions": validated_ext})
            context.metadata.setdefault("verifications", {})["a2a_skill_config"] = {"supported": False}
            return context

        # Read per-criterion rubric results
        rubric_result = context.metadata.get("verifications", {}).get(self.rubric_verifier_id, {})
        results = rubric_result.get("results", [])
        inline_passed = any(r.get("id") == "secret_code_inline" and r.get("score") == 1.0 for r in results)
        s3_passed = any(r.get("id") == "secret_code_s3" and r.get("score") == 1.0 for r in results)
        logger.info(f"Rubric verifier '{self.rubric_verifier_id}': inline={inline_passed} s3={s3_passed}")

        # List skills via GET endpoint
        a2a_url = deployed_agent.a2a_url or deployed_agent.api_url
        ext_config = skill_ext.get("config") or skill_ext.get("params") or {}
        endpoint = a2a_url + ext_config.get("endpoint", "/ext/skill-config")

        list_ok = False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(endpoint, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                skills = data.get("skills", {})
                list_ok = isinstance(skills, dict) and len(skills) > 0
                logger.info(f"Listed {len(skills)} skill(s): {list(skills.keys())}")
        except Exception as e:
            logger.warning(f"Skill list failed: {e}")

        # Build validated entry — map per-criterion results to skill vs skill_s3_url
        add_supported = inline_passed or s3_passed
        skill_entry = {
            "supported": add_supported or list_ok,
            "methods": {
                "add": {
                    "supported": add_supported,
                    "options": {
                        "name": {"supported": add_supported},
                        "description": {"supported": add_supported},
                        "skill": {"supported": inline_passed},
                        "skill_s3_url": {"supported": s3_passed},
                    },
                },
                "list": {"supported": list_ok},
            },
        }

        agent = A2AAgent.get(self.a2a_agent_id)
        validated_ext = agent.metadata.get("validated_a2a_extensions", {})
        validated_ext[A2AAgent.EXT_SKILL_CONFIG] = skill_entry
        agent.update_metadata({**agent.metadata, "validated_a2a_extensions": validated_ext})

        context.metadata.setdefault("verifications", {})["a2a_skill_config"] = {"inline": inline_passed, "s3": s3_passed, "list": list_ok}
        return context
