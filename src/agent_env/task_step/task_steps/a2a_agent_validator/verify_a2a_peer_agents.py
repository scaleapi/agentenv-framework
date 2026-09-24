"""Verify the A2A peer-agents extension end-to-end."""
from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class VerifyA2APeerAgentsStep(TaskStep):
    """Consolidate four checks of the peer-agents extension and stamp results on agent.metadata.

    Assumes upstream steps have deployed the agent under test plus a peer-target
    agent, peered them via PeerAgentsTaskStep, and prompted the agent under test
    to use peer_send_message to relay an `expected_token` from the peer."""

    type: ClassVar[str] = "verify_a2a_peer_agents"
    entity_refs = (EntityRef.agent("a2a_agent_id", version_field="a2a_agent_version"),)

    def __init__(self, id: str, version: Optional[int], a2a_agent_id: str, a2a_agent_version: Optional[int], peer_prompt_id: str, expected_token: str, depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.a2a_agent_version = a2a_agent_version
        self.peer_prompt_id = peer_prompt_id
        self.expected_token = expected_token

    def to_dict(self) -> dict:
        return {**super().to_dict(), "a2a_agent_id": self.a2a_agent_id, "a2a_agent_version": self.a2a_agent_version, "peer_prompt_id": self.peer_prompt_id, "expected_token": self.expected_token}

    @classmethod
    def from_dict(cls, data: dict) -> "VerifyA2APeerAgentsStep":
        return cls(**cls._base_from_dict(data), a2a_agent_id=data["a2a_agent_id"], a2a_agent_version=data.get("a2a_agent_version"), peer_prompt_id=data["peer_prompt_id"], expected_token=data["expected_token"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((a for a in context.deployed_agents if a.agent_name == TaskStep.DEFAULT_AGENT_NAME), None)
        if deployed is None:
            raise RuntimeError(f"VerifyA2APeerAgentsStep: no deployed agent named '{TaskStep.DEFAULT_AGENT_NAME}' in context")
        a2a_url = deployed.a2a_url or deployed.api_url
        if not a2a_url:
            raise RuntimeError(f"VerifyA2APeerAgentsStep: deployed agent has no a2a_url")

        extension = A2AAgent.find_extension(deployed.a2a_card or {}, A2AAgent.EXT_PEER_AGENTS)
        extension_advertised = extension is not None
        endpoint = (extension.get("params") or {}).get("endpoint", "/ext/peer-agents") if extension else "/ext/peer-agents"
        post_ok, list_ok = await self._check_round_trip(a2a_url, endpoint)
        sidecar_registered = await self._check_sidecar_registered(a2a_url)
        peer_call_ok, response_excerpt = self._check_peer_call(context)

        peer_entry = {
            "supported": extension_advertised and post_ok and list_ok and sidecar_registered and peer_call_ok,
            "extension_advertised": extension_advertised,
            "methods": {"set": {"supported": post_ok}, "list": {"supported": list_ok}},
            "sidecar_self_registered": sidecar_registered,
            "peer_call_succeeded": peer_call_ok,
            "peer_call_response_excerpt": response_excerpt,
        }

        agent = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)
        validated_ext = agent.metadata.get("validated_a2a_extensions", {})
        validated_ext[A2AAgent.EXT_PEER_AGENTS] = peer_entry
        agent.update_metadata({**agent.metadata, "validated_a2a_extensions": validated_ext})

        context.metadata.setdefault("verifications", {})["a2a_peer_agents"] = peer_entry
        logger.info(f"peer-agents validation: {peer_entry}")
        return context

    @staticmethod
    async def _check_round_trip(a2a_url: str, endpoint: str) -> tuple[bool, bool]:
        probe = {"peers": [{"name": "probe", "url": "http://example.invalid"}]}
        post_ok = list_ok = False
        async with httpx.AsyncClient() as client:
            try:
                post_resp = await client.post(f"{a2a_url}{endpoint}", json=probe, timeout=30)
                post_ok = post_resp.status_code < 400
                get_resp = await client.get(f"{a2a_url}{endpoint}", timeout=30)
                list_ok = get_resp.status_code < 400 and any(p.get("name") == "probe" for p in (get_resp.json().get("peers") or []))
            finally:
                await client.post(f"{a2a_url}{endpoint}", json={"peers": []}, timeout=30)
        return post_ok, list_ok

    @staticmethod
    async def _check_sidecar_registered(a2a_url: str) -> bool:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{a2a_url}/ext/mcp-config", timeout=30)
        if resp.status_code >= 400:
            return False
        servers = (resp.json() or {}).get("mcp_servers") or {}
        return any("127.0.0.1" in (info.get("url") or "") and (info.get("url") or "").endswith("/mcp") for info in servers.values())

    def _check_peer_call(self, context: TaskStepContext) -> tuple[bool, str]:
        prompt_resp = next((p for p in context.prompt_responses if p.prompt_id == self.peer_prompt_id), None)
        if prompt_resp is None:
            return False, ""
        response = prompt_resp.response or ""
        return self.expected_token in response, response[:200]
