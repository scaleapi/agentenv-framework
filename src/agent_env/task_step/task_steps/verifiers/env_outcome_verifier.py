"""Verify environment outcome using a dynamically loaded Python verifier script."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import os
import tempfile
import uuid
from typing import ClassVar, Optional

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.entity.refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.verifiers.run_context import load_run_context
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score

__all__ = ["EnvOutcomeVerifierTaskStep", "ScoreAggregator", "aggregate_score"]


logger = logging.getLogger(__name__)


class EnvOutcomeVerifierTaskStep(TaskStep):
    """Runs the artifact's ``verify(mcp_url)``; a ``run`` parameter also gets the ``Run`` of ``prompt_id``, and an
    ``env_url`` parameter the env's address, which the protocol client takes as it is: like every step, ``verify``
    runs under ``Task.run``'s grant of each deployed env's control token, so its calls reach the env's framework-only
    routes as written."""

    type: ClassVar[str] = "env_outcome_verifier"
    entity_refs = (
        EntityRef.env("env_id"),
        EntityRef.artifact("file_artifact_id", version_field="file_artifact_version", artifact_type="file"),
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        file_artifact_id: str,
        file_artifact_version: Optional[int] = None,
        score_aggregator: Optional[ScoreAggregator] = None,
        verifier_id: Optional[str] = None,
        prompt_id: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.verifier_id = verifier_id or uuid.uuid4().hex
        self.prompt_id = prompt_id
        self.env_id = env_id
        self.file_artifact_id = file_artifact_id
        self.file_artifact_version = file_artifact_version
        if isinstance(score_aggregator, str):
            score_aggregator = ScoreAggregator(score_aggregator)
        self.score_aggregator = score_aggregator or ScoreAggregator.ALL_PASS

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["verifier_id"] = self.verifier_id
        base["env_id"] = self.env_id
        base["file_artifact_id"] = self.file_artifact_id
        base["file_artifact_version"] = self.file_artifact_version
        base["score_aggregator"] = self.score_aggregator.value
        if self.prompt_id is not None:
            base["prompt_id"] = self.prompt_id
        return base

    @classmethod
    def put(cls, *, verify_script_file_path: Optional[str] = None, **kwargs) -> "EnvOutcomeVerifierTaskStep":
        """Store the step. With ``verify_script_file_path`` the script is uploaded as
        ``<id>-verifier-script`` first; otherwise the kwargs name an existing ``file_artifact_id``
        (the shape ``agent-env task create`` builds from JSON)."""
        from agent_env.task_step.store import get_task_step_store

        if verify_script_file_path is not None:
            from agent_env.artifact import FileArtifact

            artifact = FileArtifact.put(
                id=f"{kwargs['id']}-verifier-script",
                description=f"Verifier script for {kwargs['id']}",
                file_path=verify_script_file_path,
            )
            kwargs.update(file_artifact_id=artifact.id, file_artifact_version=artifact.version)
        kwargs.setdefault("version", None)
        return get_task_step_store().put_document(cls(**kwargs))

    @classmethod
    def from_dict(cls, data: dict) -> EnvOutcomeVerifierTaskStep:
        raw = data.get("score_aggregator")
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            file_artifact_id=data["file_artifact_id"],
            file_artifact_version=data.get("file_artifact_version"),
            score_aggregator=ScoreAggregator(raw) if raw else None,
            verifier_id=data.get("verifier_id"),
            prompt_id=data.get("prompt_id"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.artifact import FileArtifact

        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")

        artifact = FileArtifact.get(self.file_artifact_id, self.file_artifact_version)
        verifier_bytes = await asyncio.to_thread(artifact.load)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py")
        try:
            os.write(tmp_fd, verifier_bytes)
            os.close(tmp_fd)

            module_name = f"outcome_verifier_{self.id}"
            spec = importlib.util.spec_from_file_location(module_name, tmp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            logger.info(f"Running verify() from artifact '{self.file_artifact_id}' against {deployed.mcp_url}")
            parameters = inspect.signature(module.verify).parameters
            keywords = {"env_url": deployed.environment_url} if "env_url" in parameters else {}
            if "run" in parameters:
                keywords["run"] = await load_run_context(context, self._graded_prompt(context))
            results = await module.verify(deployed.mcp_url, **keywords)

            score = aggregate_score(results, self.score_aggregator)
            if "verifications" not in context.metadata:
                context.metadata["verifications"] = {}
            context.metadata["verifications"][self.verifier_id] = {
                "results": results,
                "score": score,
            }
            logger.info(f"Verification '{self.verifier_id}' complete: {len(results)} criteria, score={score}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return context

    def _graded_prompt(self, context: TaskStepContext) -> Optional[PromptResponse]:
        if self.prompt_id is None:
            return context.prompt_responses[-1] if context.prompt_responses else None
        prompt_response = next((pr for pr in context.prompt_responses if pr.prompt_id == self.prompt_id), None)
        if prompt_response is None:
            raise RuntimeError(f"PromptResponse with prompt_id='{self.prompt_id}' not found in context")
        return prompt_response
