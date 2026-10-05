"""Verify the advertised A2A trajectory retrieval variants."""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.a2a_agent.object_transfer import (
    TrajectoryUpload,
    fetch_trajectory,
    trajectory_mode,
)
from agent_env.task_step.snapshot_utils.agent_state_capture import trajectory_object_url
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
        get_method, get_path = A2AAgent.operation(traj_ext, "get")
        endpoint = a2a_url + get_path

        # Get task_id from the MCP step
        mcp_result = context.metadata.get("verifications", {}).get("a2a_agent_mcp", {})
        task_id = mcp_result.get("task_id")
        if not task_id:
            raise RuntimeError("No task_id from MCP verification step — trajectory validation requires a completed task")

        # Test inline trajectory retrieval
        inline_ok = False
        try:
            fetched = await fetch_trajectory(endpoint, {"task_id": task_id})
            if fetched.inline:
                inline_ok = True
                logger.info(f"Inline trajectory: OK ({len(fetched.inline)} events)")
            else:
                logger.warning("Inline trajectory: response missing 'trajectory' key or empty")
        except Exception as e:
            logger.warning(f"Inline trajectory: failed ({e})")

        # The object variant is probed only on a store that can issue its grant.
        objects_ok = False
        config = get_config()
        store = config.get_object_store()
        probe_prefix = f"{config.get_artifact_key_prefix()}a2a_validator_trajectories/{self.a2a_agent_id}/"
        if trajectory_mode(get_method, store, by="task_id") == "objects":
            try:
                prefix = store.object_url(probe_prefix)
                upload = await asyncio.to_thread(
                    TrajectoryUpload.to, store, trajectory_object_url(prefix, name=task_id, store=store)
                )
                await fetch_trajectory(endpoint, {"task_id": task_id}, upload=upload)
                objects_ok = True
                logger.info("Object trajectory: OK")
            except Exception as e:
                logger.warning(f"Object trajectory: failed ({e})")

        logger.info(f"Trajectory validation: inline={inline_ok} objects={objects_ok}")

        # Build validated_a2a_extensions entry
        traj_entry = {
            "supported": inline_ok or objects_ok,
            "methods": {
                "get": {
                    "supported": inline_ok or objects_ok,
                    "options": {
                        "task_id": {"supported": inline_ok or objects_ok},
                        "objects": {"supported": objects_ok},
                    },
                },
            },
        }

        agent = A2AAgent.get(self.a2a_agent_id)
        validated_ext = agent.metadata.get("validated_a2a_extensions", {})
        validated_ext[A2AAgent.EXT_TRAJECTORY] = traj_entry
        agent.update_metadata({**agent.metadata, "validated_a2a_extensions": validated_ext})

        context.metadata.setdefault("verifications", {})["a2a_trajectory"] = {
            "inline": inline_ok,
            "objects": objects_ok,
        }
        return context
