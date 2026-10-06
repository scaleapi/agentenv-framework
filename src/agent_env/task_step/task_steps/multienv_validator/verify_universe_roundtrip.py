"""Verify universe load/export roundtrip for MultiEnv compatibility."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from agent_env.env.env_artifact_store import EnvArtifactType, get_env_artifact_store
from agent_env.artifact.store import artifact_write_lock
from agent_env.store.ids import derive_id, fs_safe, is_local_id
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

from .universe_comparison import classify_issues, compare_dicts, normalize

if TYPE_CHECKING:
    from agent_env.env.env import DeployedEnv

logger = logging.getLogger(__name__)


class VerifyUniverseLoadExportRoundtripStep(TaskStep):
    type: ClassVar[str] = "verify_universe_load_export_roundtrip"
    entity_refs = (
        EntityRef.env("env_id"),
        EntityRef.artifact("universe_artifact_id", version_field="universe_artifact_version", artifact_type="environment_universe"),
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        universe_artifact_id: str,
        universe_artifact_version: Optional[int] = None,
        emit_file_artifact_universe: bool = False,
        persist_result: bool = True,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.universe_artifact_id = universe_artifact_id
        self.universe_artifact_version = universe_artifact_version
        # When set, also bundle the original/export1/export2 JSON into a FileArtifactUniverse so a
        # downstream agent-judge can diff them on its filesystem. Off by default — the
        # programmatic verdict does not need it. The bundle's id is deterministic (see
        # file_artifact_universe_id) so the callsite can reference it statically.
        self.emit_file_artifact_universe = emit_file_artifact_universe
        # When False, compute the programmatic verdict into context but DON'T persist it to the
        # env-artifact store — a downstream step (CombineUniverseVerdictsStep) becomes the single
        # authoritative writer, so a crash before the judge merges can't leave an ungated half-write
        # masquerading as final. Defaults True to preserve the step's standalone contract.
        self.persist_result = persist_result

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["universe_artifact_id"] = self.universe_artifact_id
        base["universe_artifact_version"] = self.universe_artifact_version
        base["emit_file_artifact_universe"] = self.emit_file_artifact_universe
        base["persist_result"] = self.persist_result
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyUniverseLoadExportRoundtripStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            universe_artifact_id=data["universe_artifact_id"],
            universe_artifact_version=data.get("universe_artifact_version"),
            emit_file_artifact_universe=data.get("emit_file_artifact_universe", False),
            persist_result=data.get("persist_result", True),
        )

    @staticmethod
    def validation_id(kind: str, env_id: str, env_version: int, universe_id: str, universe_version: int) -> str:
        """The id of a compatibility validation's ``kind`` of task or artifact for ``universe_id`` in ``env_id``.
        The env owns it, or the universe when only the universe is ``@local``, and the other id goes into its
        suffix, through ``fs_safe`` when it is ``@local``."""
        if is_local_id(universe_id) and not is_local_id(env_id):
            return derive_id(universe_id, f"{kind}-v{universe_version}-{env_id}-v{env_version}")
        return derive_id(env_id, f"{kind}-v{env_version}-{fs_safe(universe_id)}-v{universe_version}")

    @staticmethod
    def file_artifact_universe_id(env_id: str, env_version: int, universe_id: str, universe_version: int) -> str:
        """Deterministic id of the emitted FileArtifactUniverse, so the callsite can reference it
        statically without a runtime context lookup (mirrors the export1 artifact id scheme)."""
        prefix = VerifyUniverseLoadExportRoundtripStep.validation_id("validate", env_id, env_version, universe_id, universe_version)
        return f"{prefix}-fau"

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.artifact import EnvironmentUniverseArtifact
        from agent_env.env.env import Env

        # Phase 1: Resolve deployed env and universe artifact
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        env = Env.get(deployed.env_id, deployed.env_version)
        env = await type(env).from_deployed_env(deployed)
        universe = EnvironmentUniverseArtifact.get(self.universe_artifact_id, self.universe_artifact_version)
        environment_artifacts = universe.get_environment_artifacts()
        environment_names = [sa.environment_name for sa in environment_artifacts]

        # Phase 2: Load original universe + read original data for comparison
        logger.info(f"Loading universe {self.universe_artifact_id} into env {self.env_id}...")
        await env.load_environment_universe_artifact(universe)
        originals = {}
        for sa in environment_artifacts:
            try:
                data = await asyncio.to_thread(lambda sa=sa: sa.get_file_artifact().load())
                originals[sa.environment_name] = normalize(json.loads(data))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.info(f"{sa.environment_name}: non-JSON artifact, will only compare export1 vs export2")

        # Phase 3: Export #1
        logger.info("Exporting state (round 1)...")
        export1_raw = await self._export_all(deployed, environment_names)
        export1 = {name: normalize(data) for name, data in export1_raw.items()}

        # Phase 4: Create artifacts from export #1
        logger.info("Creating artifacts from export #1...")
        exported_universe = await asyncio.to_thread(
            self._create_universe_artifact, environment_artifacts, export1_raw, deployed.env_version, universe.version
        )

        # Phase 5: Reload from exported artifacts
        logger.info("Reloading from exported artifacts...")
        await env.load_environment_universe_artifact(exported_universe)

        # Phase 6: Export #2
        logger.info("Exporting state (round 2)...")
        export2_raw = await self._export_all(deployed, environment_names)
        export2 = {name: normalize(data) for name, data in export2_raw.items()}

        # Phase 7: Compare
        logger.info("Comparing results...")
        # "environments" is the SAME dict object as "services" (dual-write), so every later
        # mutation — the loop below, and apply_judge_verdict's in-place merge — lands in both.
        per_environment: dict[str, Any] = {}
        result: dict[str, Any] = {"compatible": True, "services": per_environment, "environments": per_environment, "exported_universe_artifact_id": exported_universe.id}
        for name in environment_names:
            load_issues = compare_dicts(originals[name], export1[name], "universe", "export1") if name in originals else []
            idempotency_issues = compare_dicts(export1[name], export2[name], "export1", "export2")
            annotated, is_compatible = classify_issues(load_issues, idempotency_issues)
            if not is_compatible:
                result["compatible"] = False
            result["services"][name] = {"compatible": is_compatible, "issues": annotated}
            critical_count = sum(1 for i in annotated if i["critical"])
            logger.info(f"  {name}: {'COMPATIBLE' if is_compatible else 'INCOMPATIBLE'} ({len(annotated)} issues, {critical_count} critical)")

        # Phase 7b: Optionally bundle original/export1/export2 JSON into a FileArtifactUniverse so a
        # downstream agent-judge can diff them on its filesystem. Additive — does not touch
        # the programmatic verdict above.
        if self.emit_file_artifact_universe:
            fau = await asyncio.to_thread(
                self._create_file_artifact_universe,
                environment_artifacts, exported_universe, export2_raw, deployed.env_version, universe.version,
            )
            result["file_artifact_universe_id"] = fau.id
            logger.info(f"Emitted FileArtifactUniverse for agent-judge: {fau.id}")

        # Phase 8: Always expose the programmatic result in context (downstream steps read it);
        # only persist to the env-artifact store when this step owns the write (persist_result).
        # When a CombineUniverseVerdictsStep follows, it is the single authoritative writer.
        context.metadata.setdefault("verifications", {})[EnvArtifactType.UNIVERSE_COMPATIBILITY] = result
        if self.persist_result:
            get_env_artifact_store().put(env_id=self.env_id, env_version=deployed.env_version, artifact_id=universe.id, artifact_version=universe.version, type=EnvArtifactType.UNIVERSE_COMPATIBILITY, data=result)

        return context

    def _create_file_artifact_universe(self, original_environment_artifacts: list, exported_universe: Any, export2_raw: dict[str, dict], env_version: int, universe_version: int) -> Any:
        """Bundle the three universe snapshots into one FileArtifactUniverse for the agent-judge.

        Faithfulness (this validator measures lossiness, so the bundle must not transform data):
        - ``original/<svc>.json`` references the universe's EXISTING per-service FileArtifacts
          (verbatim bytes — what compare_dicts loaded), no re-serialization.
        - ``export_1/<svc>.json`` references the FileArtifacts the export1 universe was built from
          (the exact bytes the programmatic verdict used).
        - ``export_2/<svc>.json`` is materialized fresh (export2 is not otherwise persisted), dumped
          from the same in-memory raw dict, with the same serialization as export1
          (``default=str`` for parity; ``ensure_ascii=False`` so the judge reads real unicode).
        """
        from agent_env.artifact import FileArtifact, FileArtifactUniverse

        prefix = self.validation_id("validate", self.env_id, env_version, self.universe_artifact_id, universe_version)
        with artifact_write_lock(prefix):
            file_artifacts: dict[str, Any] = {}
            for sa in original_environment_artifacts:
                file_artifacts[f"original/{sa.environment_name}.json"] = sa.get_file_artifact()
            for sa in exported_universe.get_environment_artifacts():
                file_artifacts[f"export_1/{sa.environment_name}.json"] = sa.get_file_artifact()
            for name, data in export2_raw.items():
                # Cleanup wraps both the dump and the put, so a failure in either can't leak the temp
                # file; mkstemp yields the path up front so `finally` always has a valid path to unlink.
                fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix=f"export2-{name}-")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(data, f, default=str, ensure_ascii=False)
                    fa = FileArtifact.put(id=f"{prefix}-export2-{name}", description=f"Export #2 of {name} for universe-judge", file_path=tmp_path)
                finally:
                    os.unlink(tmp_path)
                file_artifacts[f"export_2/{name}.json"] = fa

            return FileArtifactUniverse.put(
                id=self.file_artifact_universe_id(self.env_id, env_version, self.universe_artifact_id, universe_version),
                file_artifacts=file_artifacts,
            )

    @staticmethod
    async def _export_all(deployed: DeployedEnv, environment_names: list[str]) -> dict[str, dict]:
        """Export each service's state as JSON (see ``legacy_protocol.service_state``)."""
        from agent_env.env import legacy_protocol
        from agent_env.env.env import gateway_url_of
        return {name: await legacy_protocol.service_state(deployed, gateway_url_of(deployed), name) for name in environment_names}

    def _create_universe_artifact(self, original_environment_artifacts: list, export_data: dict[str, dict], env_version: int, universe_version: int) -> Any:
        """Create FileArtifact + EnvironmentArtifact per service, bundle into EnvironmentUniverseArtifact."""
        from agent_env.artifact import FileArtifact, EnvironmentArtifact, EnvironmentUniverseArtifact
        prefix = self.validation_id("validate", self.env_id, env_version, self.universe_artifact_id, universe_version)
        with artifact_write_lock(prefix):
            export_environment_artifacts = []
            for sa in original_environment_artifacts:
                name = sa.environment_name
                if name not in export_data:
                    continue
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix=f"export1-{name}-") as f:
                    json.dump(export_data[name], f, default=str)
                    tmp_path = f.name
                try:
                    file_artifact = FileArtifact.put(id=f"{prefix}-export1-{name}", description=f"Export #1 of {name} for universe compat validation", file_path=tmp_path)
                finally:
                    os.unlink(tmp_path)
                svc_artifact = EnvironmentArtifact.put(id=f"{prefix}-export1-svc-{name}", environment_name=sa.environment_name, file_artifact=file_artifact)
                export_environment_artifacts.append(svc_artifact)

            return EnvironmentUniverseArtifact.put(id=f"{prefix}-export1", environment_artifacts=export_environment_artifacts)
