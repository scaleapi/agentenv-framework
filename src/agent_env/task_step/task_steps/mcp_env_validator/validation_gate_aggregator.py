"""Rolls the per-verifier verdicts under ``context.metadata["verifications"]``
(``mcp_tool_schema``, ``mcp_tool_correctness``) into one
``validation_gate`` verdict, persisted to context + env metadata so ``mcp-server put``
reads a single field.

Record-only: never raises, so ``validate()`` runs to completion and enforcement stays
in one place (``put``). A required gate with no verdict is a failure (fail-closed); a
gate that reported itself ``skipped`` (nothing to check) is advisory and does not block.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

# Live-server gates required to publish. Unit tests are checked separately at `put`.
# (build-time artifact); the environment card is advisory and not required here.
DEFAULT_REQUIRED_GATES = ("mcp_tool_schema", "mcp_tool_correctness")


def aggregate_gate(verifications: dict, required_gates) -> dict:
    """Roll up per-gate verdicts (gate-key → ``{"passed": bool}``) into
    ``{passed, required_gates, failed_gates, missing_gates, skipped_gates}``. A gate
    absent from ``verifications`` fails the aggregate; one whose verdict is ``skipped``
    (verifier ran, nothing to check) is advisory and doesn't block.
    """
    verifications = verifications or {}
    failed, missing, skipped = [], [], []
    for gate in required_gates:
        v = verifications.get(gate)
        if v is None:
            missing.append(gate)
        elif v.get("skipped"):
            skipped.append(gate)
        elif not v.get("passed", False):
            failed.append(gate)
    return {
        "passed": not failed and not missing,
        "required_gates": list(required_gates),
        "failed_gates": failed,
        "missing_gates": missing,
        "skipped_gates": skipped,
    }


class ValidationGateAggregatorStep(TaskStep):
    type: ClassVar[str] = "validation_gate_aggregator"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        required_gates: Optional[list[str]] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.required_gates = list(required_gates) if required_gates else list(DEFAULT_REQUIRED_GATES)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["required_gates"] = self.required_gates
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationGateAggregatorStep":
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            required_gates=data.get("required_gates"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.env.env import Env

        summary = aggregate_gate(context.metadata.get("verifications", {}), self.required_gates)
        logger.info(
            f"Validation gate for '{self.env_id}': passed={summary['passed']} "
            f"failed={summary['failed_gates']} missing={summary['missing_gates']} "
            f"skipped={summary['skipped_gates']}"
        )
        context.metadata["validation_gate"] = summary

        # Persist to env metadata so `put` / the CLI can read the verdict directly.
        try:
            deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
            env = Env.get(self.env_id, deployed.env_version if deployed else None)
            env.merge_metadata({"validation_gate": summary})
        except Exception as e:  # pragma: no cover - metadata persistence is best-effort
            logger.warning(f"Could not persist validation_gate to env metadata: {e}")

        return context
