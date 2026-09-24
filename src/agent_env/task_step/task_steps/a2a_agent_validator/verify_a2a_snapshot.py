"""Verify A2A agent snapshot extension by reading rubric results and recording extension support."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class VerifyA2ASnapshotStep(TaskStep):
    """Record snapshot extension validation results on the agent's metadata.

    Assumes upstream steps have:
      1. Planted state on agent A via a prompt step,
      2. Captured a snapshot via SnapshotAgentStateTaskStep,
      3. Deployed agent B with the snapshot_files_artifact_id,
      4. Sent a recall prompt to agent B,
      5. Run a RubricsVerifierTaskStep that scored whether the recall worked.

    This step reads that rubric verifier's result, checks the snapshot
    extension is advertised in both deployed agents, and writes the
    consolidated outcome to agent.metadata['validated_a2a_extensions'][EXT_SNAPSHOT].
    """

    type: ClassVar[str] = "verify_a2a_snapshot"
    entity_refs = (EntityRef.agent("a2a_agent_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        rubric_verifier_id: str,
        rubric_criterion_id: str,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.rubric_verifier_id = rubric_verifier_id
        self.rubric_criterion_id = rubric_criterion_id

    def to_dict(self) -> dict:
        return {**super().to_dict(), "a2a_agent_id": self.a2a_agent_id, "rubric_verifier_id": self.rubric_verifier_id, "rubric_criterion_id": self.rubric_criterion_id}

    @classmethod
    def from_dict(cls, data: dict) -> VerifyA2ASnapshotStep:
        return cls(
            **cls._base_from_dict(data),
            a2a_agent_id=data["a2a_agent_id"],
            rubric_verifier_id=data["rubric_verifier_id"],
            rubric_criterion_id=data["rubric_criterion_id"],
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent

        # Extension advertised on at least one deployed agent
        extension_advertised = False
        for deployed in context.deployed_agents:
            card = deployed.a2a_card or {}
            if A2AAgent.find_extension(card, A2AAgent.EXT_SNAPSHOT) is not None:
                extension_advertised = True
                break

        # Rubric outcome — did recall on the loaded agent contain the planted token?
        rubric_result = context.metadata.get("verifications", {}).get(self.rubric_verifier_id, {})
        results = rubric_result.get("results", []) or []
        recall_passed = any(
            r.get("id") == self.rubric_criterion_id and r.get("score") == 1.0 for r in results
        )

        # The save+load round-trip succeeded if a snapshot was actually written *and*
        # the recall rubric passed. The snapshot universe id is recorded by the
        # snapshot step in context.metadata['agent_snapshots'].
        snapshots_recorded = bool(context.metadata.get("agent_snapshots") or [])
        save_ok = snapshots_recorded
        load_ok = recall_passed  # if recall worked, load (and resume) must have worked
        recall_ok = recall_passed

        snap_entry = {
            "supported": extension_advertised and save_ok and load_ok and recall_ok,
            "extension_advertised": extension_advertised,
            "methods": {
                "save": {"supported": save_ok},
                "load": {"supported": load_ok},
            },
            "recall_after_load": recall_ok,
        }

        agent = A2AAgent.get(self.a2a_agent_id)
        validated_ext = agent.metadata.get("validated_a2a_extensions", {})
        validated_ext[A2AAgent.EXT_SNAPSHOT] = snap_entry
        agent.update_metadata({**agent.metadata, "validated_a2a_extensions": validated_ext})

        context.metadata.setdefault("verifications", {})["a2a_snapshot"] = {
            "extension_advertised": extension_advertised,
            "save": save_ok,
            "load": load_ok,
            "recall": recall_ok,
        }
        logger.info(
            f"Snapshot validation: extension_advertised={extension_advertised} "
            f"save={save_ok} load={load_ok} recall={recall_ok}"
        )
        return context
