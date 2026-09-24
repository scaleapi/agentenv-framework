"""Verify A2A agent trajectory extension supports inline and S3 retrieval."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class VerifyA2ATrajectoryStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_trajectory"
    entity_refs = (EntityRef.agent("a2a_agent_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyA2ATrajectoryStep:
        return cls(**cls._base_from_dict(data), a2a_agent_id=data["a2a_agent_id"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent
        from agent_env.config import get_config

        deployed_agent = next((a for a in context.deployed_agents), None)
        if deployed_agent is None:
            raise RuntimeError("No deployed agent found in context")

        card = deployed_agent.a2a_card or {}
        traj_ext = A2AAgent.find_extension(card, A2AAgent.EXT_TRAJECTORY)
        if not traj_ext:
            logger.warning("Trajectory extension not advertised in agent card")
            agent = A2AAgent.get(self.a2a_agent_id)
            validated_ext = agent.metadata.get("validated_a2a_extensions", {})
            validated_ext[A2AAgent.EXT_TRAJECTORY] = {"supported": False}
            agent.update_metadata({**agent.metadata, "validated_a2a_extensions": validated_ext})
            context.metadata.setdefault("verifications", {})["a2a_trajectory"] = {"supported": False}
            return context

        a2a_url = deployed_agent.a2a_url or deployed_agent.api_url
        ext_config = traj_ext.get("config") or traj_ext.get("params") or {}
        endpoint = a2a_url + ext_config.get("endpoint", "/ext/trajectory")

        # Get task_id from the MCP step
        mcp_result = context.metadata.get("verifications", {}).get("a2a_agent_mcp", {})
        task_id = mcp_result.get("task_id")
        if not task_id:
            raise RuntimeError("No task_id from MCP verification step — trajectory validation requires a completed task")

        # Test inline trajectory retrieval
        inline_ok = False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(endpoint, json={"task_id": task_id}, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                if "trajectory" in data and data["trajectory"]:
                    inline_ok = True
                    logger.info(f"Inline trajectory: OK ({len(data['trajectory'])} events)")
                else:
                    logger.warning("Inline trajectory: response missing 'trajectory' key or empty")
        except Exception as e:
            logger.warning(f"Inline trajectory: failed ({e})")

        # Test S3 prefix trajectory retrieval
        s3_ok = False
        bucket = get_config().get_s3_bucket()
        s3_prefix = f"s3://{bucket}/a2a_validator_trajectories/{self.a2a_agent_id}/"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(endpoint, json={"task_id": task_id, "trajectory_s3_prefix": s3_prefix}, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                if "trajectory_s3_prefix" in data and data["trajectory_s3_prefix"]:
                    s3_ok = True
                    logger.info(f"S3 trajectory: OK ({data['trajectory_s3_prefix']})")
                else:
                    logger.warning("S3 trajectory: response missing 'trajectory_s3_prefix' key or empty")
        except Exception as e:
            logger.warning(f"S3 trajectory: failed ({e})")

        logger.info(f"Trajectory validation: inline={inline_ok} s3={s3_ok}")

        # Build validated_a2a_extensions entry
        traj_entry = {
            "supported": inline_ok or s3_ok,
            "methods": {
                "get": {
                    "supported": inline_ok or s3_ok,
                    "options": {
                        "task_id": {"supported": inline_ok or s3_ok},
                        "trajectory_s3_prefix": {"supported": s3_ok},
                    },
                },
            },
        }

        agent = A2AAgent.get(self.a2a_agent_id)
        validated_ext = agent.metadata.get("validated_a2a_extensions", {})
        validated_ext[A2AAgent.EXT_TRAJECTORY] = traj_entry
        agent.update_metadata({**agent.metadata, "validated_a2a_extensions": validated_ext})

        context.metadata.setdefault("verifications", {})["a2a_trajectory"] = {"inline": inline_ok, "s3": s3_ok}
        return context
