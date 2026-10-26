from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Optional

from agent_env.a2a_agent import protocol
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.bundle.installed import write_installed
from agent_env.task.episode.run import _episode_of
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.deploy_agent import bound_deployment, live_mcp_url
from agent_env.task_step.task_steps.install_agent import PROXY_PARAM

# The agents bundle's proxy, deployed in the place of an agent an episode plays.
PROXY_AGENT_ID = "@local/agentenv-framework/agents/proxy"


class _EpisodeAgentStep(TaskStep):
    """An outside agent's ``deploy_agent`` or ``install_agent``. In a task that uses the agent's sandbox it runs that
    step with the proxy agent in the agent's place; otherwise it serves a proxy in this process. Either way the episode
    then plays the proxy, which the agent's ``prompt_agent`` steps message as they would the agent."""

    type: ClassVar[str] = "episode_agent"
    entity_refs = ()

    def __init__(
        self, id: str, version: Optional[int], *, step: TaskStep, sandboxed: bool,
        depends_on: Optional[list] = None, fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.step = step
        self.sandboxed = sandboxed

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "step": self.step.to_dict(), "sandboxed": self.sandboxed}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _EpisodeAgentStep:
        step = get_task_step_registry()[data["step"]["type"]].from_dict(data["step"])
        return cls(**cls._base_from_dict(data), step=step, sandboxed=data["sandboxed"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        episode = _episode_of(context)
        agent = self.step.agent_name
        if not self.sandboxed:
            mcp_urls = {}
            for env_id in getattr(self.step, "env_ids", None) or []:
                deployed = bound_deployment(context, env_id, env_step_id=self.step.env_step_id, step_id=self.id)
                mcp_urls[env_id] = deployed.mcp_url if deployed is not None else live_mcp_url(env_id)
            deployed = await episode._serve_in_process(agent, getattr(self.step, "role", None), mcp_urls)
            context.deployed_agents.append(deployed)
            if system_prompt := getattr(self.step, "system_prompt", None):
                config = A2AAgent.negotiate_agent_config(deployed.a2a_card, {"system_prompt": system_prompt})
                await protocol.post_agent_config(deployed.a2a_url + config.endpoint, config.fields)
            return context
        await asyncio.to_thread(write_installed, PROXY_AGENT_ID)
        override = {"a2a_agent_id": PROXY_AGENT_ID} if self.step.type == "deploy_agent" else {PROXY_PARAM: PROXY_AGENT_ID}
        step_params = context.metadata.setdefault("user_overrides", {}).setdefault("step_params", {})
        step_params[self.step.id] = {**step_params.get(self.step.id, {}), **override}
        context = await self.step.execute(context)
        deployed = next(deployed for deployed in context.deployed_agents if deployed.agent_name == agent)
        await episode._attach(agent, deployed)
        return context
