"""Verify A2A agent skill-config extension by reading rubric results and listing skills."""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar, Optional

import httpx
from agentenv_protocol.a2a_agent import SkillAddResponse

from agent_env.task_step.context import TaskStepContext
from agent_env.a2a_agent.object_transfer import (
    TRANSFER_TIMEOUT_SECONDS,
    TransferMode,
    invoke_transfer,
    parse_response,
    skill_add_call,
)
from agent_env.a2a_agent.staging import transfer_store
from agent_env.config import get_config
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
        skill_bundle_object_url: Optional[str] = None,
        skill_s3_url: Optional[str] = None,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.rubric_verifier_id = rubric_verifier_id
        self.skill_bundle_object_url = skill_bundle_object_url
        self.skill_s3_url = skill_s3_url

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        base["rubric_verifier_id"] = self.rubric_verifier_id
        base["skill_bundle_object_url"] = self.skill_bundle_object_url
        base["skill_s3_url"] = self.skill_s3_url
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyA2ASkillConfigStep:
        return cls(
            **cls._base_from_dict(data),
            a2a_agent_id=data["a2a_agent_id"],
            rubric_verifier_id=data["rubric_verifier_id"],
            skill_bundle_object_url=data.get("skill_bundle_object_url"),
            skill_s3_url=data.get("skill_s3_url"),
        )

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
        add_method, add_path = A2AAgent.operation(skill_ext, "add")
        a2a_url = deployed_agent.a2a_url or deployed_agent.api_url

        async def probe_add(form: TransferMode, name: str, description: str, object_url: str | None) -> bool:
            """Whether the agent registers an object-backed skill sent in exactly ``form``."""
            if object_url is None:
                return False
            store = transfer_store(
                get_config().get_object_store(), a2a_url, deployed_agent.a2a_card,
                sandbox_type=deployed_agent.sandbox_type,
            )
            try:
                call = await asyncio.to_thread(
                    skill_add_call,
                    add_method,
                    store,
                    name=name,
                    description=description,
                    object_url=object_url,
                    forms=(form,),
                    sandbox_type=deployed_agent.sandbox_type,
                    agent_name=deployed_agent.agent_name,
                )
            except RuntimeError as exc:
                logger.info("Skill %s request: not sent (%s)", form, exc)
                return False
            except Exception as exc:
                logger.warning("Skill %s request: failed (%s)", form, exc)
                return False
            try:
                answer = await invoke_transfer(
                    a2a_url + add_path,
                    call,
                    verb="POST",
                    operation=f"skill add ({form})",
                    timeout=TRANSFER_TIMEOUT_SECONDS,
                    store=store,
                )
                result = parse_response(SkillAddResponse, answer, operation=f"skill add ({form})")
                if result.name != name:
                    raise ValueError("response name does not match the requested skill")
                logger.info("Skill %s request: OK", form)
                return True
            except Exception as exc:
                logger.warning("Skill %s request: failed (%s)", form, exc)
                return False

        bundle_passed = await probe_add(
            "objects",
            "validator-probe-bundle",
            "Validator probe for the portable skill bundle variant.",
            self.skill_bundle_object_url,
        )
        s3_passed = await probe_add(
            "legacy",
            "validator-probe-s3",
            "Validator probe for the legacy S3 skill variant.",
            self.skill_s3_url,
        )
        logger.info(
            "Skill validation '%s': inline=%s bundle=%s s3=%s",
            self.rubric_verifier_id,
            inline_passed,
            bundle_passed,
            s3_passed,
        )

        # List skills via GET endpoint
        list_ok = False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(a2a_url + A2AAgent.operation(skill_ext, "list")[1], timeout=30)
                resp.raise_for_status()
                data = resp.json()
                skills = data.get("skills", {})
                list_ok = isinstance(skills, dict) and len(skills) > 0
                logger.info(f"Listed {len(skills)} skill(s): {list(skills.keys())}")
        except Exception as e:
            logger.warning(f"Skill list failed: {e}")

        # Build validated entry from observed behavior for each request variant.
        add_supported = inline_passed or bundle_passed or s3_passed
        skill_entry = {
            "supported": add_supported or list_ok,
            "methods": {
                "add": {
                    "supported": add_supported,
                    "options": {
                        "name": {"supported": add_supported},
                        "description": {"supported": add_supported},
                        "skill": {"supported": inline_passed},
                        "skill_md": {"supported": inline_passed},
                        "skill_bundle": {"supported": bundle_passed},
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

        context.metadata.setdefault("verifications", {})["a2a_skill_config"] = {
            "inline": inline_passed,
            "bundle": bundle_passed,
            "s3": s3_passed,
            "list": list_ok,
        }
        return context
