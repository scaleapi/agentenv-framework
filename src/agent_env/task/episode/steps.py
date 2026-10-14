from __future__ import annotations

from typing import Any, ClassVar, Optional

from agent_env.entity_refs import EntityRef
from agent_env.task.episode.run import _episode_of
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.deploy_agent import bound_deployment, live_mcp_url


class _EpisodeAgentStep(TaskStep):
    """An outside agent's ``deploy_agent``: deploys nothing, and binds the agent to its envs where it stood."""

    type: ClassVar[str] = "episode_agent"
    entity_refs = (EntityRef.env("env_ids[]"),)

    def __init__(
        self, id: str, version: Optional[int], *, agent_name: str, env_ids: list[str], env_step_id: str | None,
        role: str | None, depends_on: Optional[list] = None, fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.agent_name = agent_name
        self.env_ids = list(env_ids)
        self.env_step_id = env_step_id
        self.role = role

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "agent_name": self.agent_name, "env_ids": self.env_ids,
                "env_step_id": self.env_step_id, "role": self.role}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _EpisodeAgentStep:
        return cls(**cls._base_from_dict(data), agent_name=data["agent_name"], env_ids=data["env_ids"],
                   env_step_id=data["env_step_id"], role=data["role"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        mcp_urls = {}
        for env_id in self.env_ids:
            deployed = bound_deployment(context, env_id, env_step_id=self.env_step_id, step_id=self.id)
            mcp_urls[env_id] = deployed.mcp_url if deployed is not None else live_mcp_url(env_id)
        _episode_of(context)._bind(self.agent_name, self.role, mcp_urls)
        return context


class _EpisodeTurnStep(TaskStep):
    """An outside agent's ``prompt_agent``: hands the turn to the caller and records the reply it gets back."""

    type: ClassVar[str] = "episode_turn"
    entity_refs = ()

    def __init__(
        self, id: str, version: Optional[int], *, agent_name: str, prompt_id: str, prompt: str,
        system_prompt: str | None, timeout_seconds: float, depends_on: Optional[list] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.agent_name = agent_name
        self.prompt_id = prompt_id
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "agent_name": self.agent_name, "prompt_id": self.prompt_id,
                "prompt": self.prompt, "system_prompt": self.system_prompt, "timeout_seconds": self.timeout_seconds}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _EpisodeTurnStep:
        return cls(**cls._base_from_dict(data), agent_name=data["agent_name"], prompt_id=data["prompt_id"],
                   prompt=data["prompt"], system_prompt=data["system_prompt"],
                   timeout_seconds=data["timeout_seconds"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        reply = await _episode_of(context)._play(self)
        context.prompt_responses.append(PromptResponse(
            prompt_id=self.prompt_id, response=reply, prompt_text=self.prompt, agent_name=self.agent_name,
            step_id=self.id,
        ))
        return context
