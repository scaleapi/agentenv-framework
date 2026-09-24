"""Combine the programmatic + agent-judge universe-compatibility verdicts.

This is the single authoritative writer of the ``UNIVERSE_COMPATIBILITY`` artifact. The round-trip
step computes the programmatic verdict into ``context.metadata`` (without persisting it), the rubrics
verifier records the agent-judge verdict, and this step — running last — merges them and persists the
result once, atomically. Keeping the write here (rather than in the round-trip step) means a crash or
judge failure before this step leaves NO half-written, ungated result masquerading as final.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class CombineUniverseVerdictsStep(TaskStep):
    type: ClassVar[str] = "combine_universe_verdicts"
    entity_refs = (
        EntityRef.env("env_id"),
        EntityRef.artifact("universe_artifact_id", version_field="universe_artifact_version"),
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        universe_artifact_id: str,
        universe_artifact_version: int,
        judge_verifier_id: str,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.universe_artifact_id = universe_artifact_id
        self.universe_artifact_version = universe_artifact_version
        self.judge_verifier_id = judge_verifier_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["universe_artifact_id"] = self.universe_artifact_id
        base["universe_artifact_version"] = self.universe_artifact_version
        base["judge_verifier_id"] = self.judge_verifier_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> CombineUniverseVerdictsStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            universe_artifact_id=data["universe_artifact_id"],
            universe_artifact_version=data["universe_artifact_version"],
            judge_verifier_id=data["judge_verifier_id"],
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.env.env_artifact_store import EnvArtifactType, get_env_artifact_store
        from .verify_universe_agent_judge import apply_judge_verdict

        verifs = context.metadata.get("verifications", {})
        compat = verifs.get(EnvArtifactType.UNIVERSE_COMPATIBILITY)
        if compat is None:
            logger.warning("combine: no programmatic UNIVERSE_COMPATIBILITY result in context; nothing to persist")
            return context

        judge = verifs.get(self.judge_verifier_id)
        if judge is not None:
            apply_judge_verdict(compat, judge)  # mutates compat in place; gates `compatible`
            logger.info(f"combine: merged agent-judge verdict; compatible={compat.get('compatible')}")
        else:
            # Defensive: with a load-bearing judge (fail_task_on_error=True on the judge steps) we
            # only reach here when the judge succeeded, so this is unexpected. Persist the
            # programmatic result rather than drop it, but MARK it so downstream/UI can tell the
            # verdict was NOT gated by the judge (don't mistake an ungated result for a complete one).
            compat["agent_judge_skipped"] = True
            logger.warning("combine: agent-judge verdict missing; persisting programmatic result UNGATED (agent_judge_skipped=True)")

        # Re-point the dual-written "environments" twin at the merged "services" dict. The two are
        # aliased when built, but a Temporal heartbeat round-trip re-materializes them as separate
        # dicts, and apply_judge_verdict only writes "services" — so without this the twin can
        # persist stale.
        if "services" in compat:
            compat["environments"] = compat["services"]

        # env_version must match what the round-trip step used so this upserts the same row.
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            logger.warning(f"combine: env '{self.env_id}' not in deployed_envs; skipping persist")
            return context

        get_env_artifact_store().put(
            env_id=self.env_id,
            env_version=deployed.env_version,
            artifact_id=self.universe_artifact_id,
            artifact_version=self.universe_artifact_version,
            type=EnvArtifactType.UNIVERSE_COMPATIBILITY,
            data=compat,
        )
        return context
