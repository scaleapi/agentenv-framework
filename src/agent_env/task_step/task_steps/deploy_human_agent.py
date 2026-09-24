"""Register an external human A2A endpoint as a peerable DeployedAgent, so a
single-turn solver can consult a human via the peer_send_message tool. A peer
needs only a name + URL, so the card is best-effort.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

_CARD_FETCH_TIMEOUT_SECONDS = 5


class DeployHumanAgentTaskStep(TaskStep):
    type: ClassVar[str] = "deploy_human_agent"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: Optional[int],
        agent_name: str = "human_agent",
        a2a_url: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.agent_name = agent_name
        self.a2a_url = a2a_url

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["agent_name"] = self.agent_name
        base["a2a_url"] = self.a2a_url
        return base

    @classmethod
    def from_dict(cls, data: dict) -> DeployHumanAgentTaskStep:
        return cls(
            **cls._base_from_dict(data),
            agent_name=data.get("agent_name", "human_agent"),
            a2a_url=data.get("a2a_url"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.config import get_config

        if context.metadata.get("human_agents"):
            raise RuntimeError(
                "DeployHumanAgentTaskStep: a human agent is already registered; "
                "only one human agent is allowed per task"
            )
        if any(a.agent_name == self.agent_name for a in context.deployed_agents):
            raise RuntimeError(f"Agent with name '{self.agent_name}' is already deployed")

        if self.a2a_url:
            url = self.a2a_url
        else:
            base = get_config().get_default_human_a2a_url()
            # Run-scope the URL so the human endpoint can attribute the ask to this
            # task instance from the request path.
            url = f"{base.rstrip('/')}/instance/{context.instance_id}" if context.instance_id else base
        card = await self._resolve_card(url)

        context.deployed_agents.append(
            DeployedAgent(agent_name=self.agent_name, api_url=url, a2a_url=url, a2a_card=card)
        )
        context.metadata.setdefault("human_agents", []).append(
            {"step_id": self.id, "agent_name": self.agent_name, "a2a_url": url}
        )
        logger.info(f"Registered human peer '{self.agent_name}' at {url} (card name={card.get('name')})")
        return context

    async def _resolve_card(self, url: str) -> dict:
        """Best-effort card fetch — a human endpoint may not serve one, and a peer needs no card, so fall back to a synthesized card."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{url}/.well-known/agent.json", timeout=_CARD_FETCH_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Human agent card at {url} returned {resp.status_code}; using a synthesized card")
        except Exception as e:
            logger.warning(f"Could not fetch human agent card from {url} ({type(e).__name__}: {e}); using a synthesized card")
        return self._synthesized_card()

    def _synthesized_card(self) -> dict:
        return {
            "name": self.agent_name,
            "description": "A human operator available to answer questions",
            "capabilities": {},
            "default_input_modes": ["text"],
            "default_output_modes": ["text"],
        }
