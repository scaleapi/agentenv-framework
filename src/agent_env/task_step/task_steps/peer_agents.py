"""Push routing tables to deployed A2A agents' /ext/peer-agents extension."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from typing import ClassVar, Optional

import httpx

from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)
_POST_TIMEOUT_SECONDS = 30


@dataclass
class AgentPeers:
    source_agent_name: str
    peer_agent_names: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "AgentPeers":
        return cls(source_agent_name=data["source_agent_name"], peer_agent_names=list(data["peer_agent_names"]))


class PeerAgentsTaskStep(TaskStep):
    type: ClassVar[str] = "peer_agents"
    entity_refs = ()

    def __init__(self, id: str, version: Optional[int], peerings: list, depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.peerings = [p if isinstance(p, AgentPeers) else AgentPeers.from_dict(p) for p in peerings]

    def to_dict(self) -> dict:
        return {**super().to_dict(), "peerings": [dataclasses.asdict(p) for p in self.peerings]}

    @classmethod
    def from_dict(cls, data: dict) -> "PeerAgentsTaskStep":
        return cls(**cls._base_from_dict(data), peerings=[AgentPeers.from_dict(p) for p in data["peerings"]])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent
        if not self.peerings:
            raise RuntimeError("PeerAgentsTaskStep: no peerings supplied")
        deployed_by_name = {a.agent_name: a for a in context.deployed_agents}
        self._validate(deployed_by_name, A2AAgent)
        await asyncio.gather(*[self._post(deployed_by_name, p, A2AAgent) for p in self.peerings])
        recorded = context.metadata.setdefault("agent_peerings", [])
        for p in self.peerings:
            recorded.append({"source_agent_name": p.source_agent_name, "peer_agent_names": list(p.peer_agent_names)})
        logger.info("PeerAgentsTaskStep: peered %d source(s)", len(self.peerings))
        return context

    def _validate(self, deployed_by_name: dict[str, DeployedAgent], a2a_agent_cls) -> None:
        seen: set[str] = set()
        for p in self.peerings:
            src = p.source_agent_name
            if src in seen:
                raise RuntimeError(f"PeerAgentsTaskStep: source '{src}' listed twice")
            seen.add(src)
            agent = deployed_by_name.get(src)
            if agent is None:
                raise RuntimeError(f"PeerAgentsTaskStep: source '{src}' not in deployed_agents (available: {sorted(deployed_by_name)})")
            if not (agent.a2a_url or agent.api_url):
                raise RuntimeError(f"PeerAgentsTaskStep: source '{src}' has no a2a_url")
            if a2a_agent_cls.find_extension(agent.a2a_card or {}, a2a_agent_cls.EXT_PEER_AGENTS) is None:
                raise RuntimeError(f"PeerAgentsTaskStep: source '{src}' does not advertise the peer-agents extension")
            if src in p.peer_agent_names:
                raise RuntimeError(f"PeerAgentsTaskStep: source '{src}' lists itself as a peer")
            missing = [n for n in p.peer_agent_names if n not in deployed_by_name]
            if missing:
                raise RuntimeError(f"PeerAgentsTaskStep: peers {missing} for source '{src}' not in deployed_agents")

    async def _post(self, deployed_by_name: dict[str, DeployedAgent], peering: AgentPeers, a2a_agent_cls) -> None:
        src = deployed_by_name[peering.source_agent_name]
        a2a_url = src.a2a_url or src.api_url
        ext = a2a_agent_cls.find_extension(src.a2a_card or {}, a2a_agent_cls.EXT_PEER_AGENTS)
        endpoint = (ext.get("params") or {}).get("endpoint", "/ext/peer-agents")
        peers = [self._record(deployed_by_name[n]) for n in peering.peer_agent_names]
        async with httpx.AsyncClient() as client:
            resp = await client.post(a2a_url + endpoint, json={"peers": peers}, timeout=_POST_TIMEOUT_SECONDS)
        if resp.status_code >= 400:
            raise RuntimeError(f"PeerAgentsTaskStep: POST {a2a_url + endpoint} failed for '{peering.source_agent_name}': {resp.status_code} {resp.text}")
        logger.info("PeerAgentsTaskStep: peered '%s' with %s", peering.source_agent_name, peering.peer_agent_names)

    @staticmethod
    def _record(peer: DeployedAgent) -> dict:
        card = peer.a2a_card or {}
        return {"name": peer.agent_name, "url": peer.a2a_url or peer.api_url, "card": card, "description": card.get("description")}
