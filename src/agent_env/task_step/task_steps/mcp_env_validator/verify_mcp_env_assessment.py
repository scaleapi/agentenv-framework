"""Assess MCP tool correctness from an agent's structured response."""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class VerifyMCPEnvAssessmentStep(TaskStep):
    type: ClassVar[str] = "verify_mcp_env_assessment"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        prompt_id: str,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.prompt_id = prompt_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["prompt_id"] = self.prompt_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyMCPEnvAssessmentStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], prompt_id=data["prompt_id"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.env.env import Env

        prompt_response = next((pr for pr in context.prompt_responses if pr.prompt_id == self.prompt_id), None)
        if prompt_response is None:
            raise RuntimeError(f"PromptResponse with prompt_id='{self.prompt_id}' not found in context")

        if prompt_response.error_type:
            logger.warning(f"Skipping assessment: prompt had error_type={prompt_response.error_type}")
            result: dict[str, Any] = {"passed": False, "error": f"Agent error: {prompt_response.error_type}", "results": []}
        else:
            result = self._parse_response(prompt_response.response)

        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        env = Env.get(self.env_id, deployed.env_version)
        env.update_metadata({**env.metadata, "mcp_tool_correctness_validation": result})

        if "verifications" not in context.metadata:
            context.metadata["verifications"] = {}
        context.metadata["verifications"]["mcp_tool_correctness"] = result

        return context

    @staticmethod
    def _parse_response(response_text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse agent response as JSON: {e}")
            return {"passed": False, "error": f"Invalid JSON response: {e}", "results": []}

        results = parsed.get("results", []) if isinstance(parsed, dict) else []
        all_passed = all(r.get("passed", False) for r in results) and len(results) > 0
        return {"passed": all_passed, "total_tools": len(results), "results": results}
