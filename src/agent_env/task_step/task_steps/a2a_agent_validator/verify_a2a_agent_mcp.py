"""Verify A2A agent can invoke MCP tools by sending a test prompt and reading the env trajectory."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import ClassVar, Optional

import httpx

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

TEST_PROMPT = (
    "Use the `search_emails` MCP tool (from the registered MCP server) to search for "
    "emails containing the word 'invoice'. Do not answer from memory and do not call "
    "any other tool — actually invoke `search_emails`. Report the tool's raw output."
)


class VerifyA2AAgentMCPStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_agent_mcp"
    entity_refs = (EntityRef.agent("a2a_agent_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 10,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        base["timeout_seconds"] = self.timeout_seconds
        base["poll_interval_seconds"] = self.poll_interval_seconds
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyA2AAgentMCPStep:
        return cls(**cls._base_from_dict(data), a2a_agent_id=data["a2a_agent_id"], timeout_seconds=data.get("timeout_seconds", 300), poll_interval_seconds=data.get("poll_interval_seconds", 10))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent, protocol

        deployed_agent = next((a for a in context.deployed_agents), None)
        if deployed_agent is None:
            raise RuntimeError("No deployed agent found in context")
        deployed_env = next((e for e in context.deployed_envs), None)
        if deployed_env is None:
            raise RuntimeError("No deployed env found in context")

        a2a_url = deployed_agent.a2a_url or deployed_agent.api_url
        message_id = uuid.uuid4().hex

        # Send test prompt via message/send (A2A spec 0.3)
        logger.info(f"Sending test prompt to agent at {a2a_url}...")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{a2a_url}/a2a", json={
                "jsonrpc": "2.0", "id": "1", "method": "message/send",
                "params": {"message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": [{"kind": "text", "text": TEST_PROMPT}],
                }},
            }, timeout=self.timeout_seconds)
            resp.raise_for_status()
            send_body = resp.json()
            if "error" in send_body:
                raise RuntimeError(f"A2A message/send failed: {send_body['error']}")
            send_result = send_body.get("result") or {}
            task_id = send_result["id"]
            logger.info(f"message/send returned task_id={task_id}")

        # Poll tasks/get until terminal state
        logger.info("Polling for task completion...")
        a2a_result = await self._poll_task(a2a_url, task_id)
        task_status = a2a_result["status"]["state"]

        # Parse the terminal message through the shared compatibility layer so
        # both legacy top-level telemetry and typed usage telemetry are handled.
        status_msg = (a2a_result.get("status") or {}).get("message") or {}
        terminal = protocol.TerminalResponse.from_message(status_msg)
        agent_response = terminal.response_text
        tool_call_count_reported = terminal.tool_call_count is not None

        # Read trajectory from env sandbox
        tools_invoked = await self._count_tool_calls(deployed_env.sandbox_id)
        passed = task_status == "completed" and tools_invoked > 0
        logger.info(f"MCP validation: status={task_status} tools_invoked={tools_invoked} passed={passed}")

        mcp_validation = {
            "passed": passed,
            "tools_invoked": tools_invoked,
            "task_status": task_status,
            "task_id": task_id,
            "agent_response": agent_response[:500],
        }
        context.metadata.setdefault("verifications", {})["a2a_agent_mcp"] = mcp_validation

        # Build validated mcp-config entry from what we actually tested
        card = deployed_agent.a2a_card or {}
        mcp_ext = A2AAgent.find_extension(card, A2AAgent.EXT_MCP_CONFIG)
        mcp_entry: dict = {"supported": mcp_ext is not None and passed}
        if mcp_ext:
            ext_config = mcp_ext.get("config") or mcp_ext.get("params") or {}
            methods: dict = {}
            for method_name, method_def in ext_config.get("methods", {}).items():
                method_entry: dict = {"supported": passed}
                request = method_def.get("request", {})
                options = []
                for key in ("required", "optional", "supported"):
                    options.extend(request.get(key, []))
                for one_of in request.get("oneOf", []):
                    options.extend(one_of.get("required", []))
                if options:
                    method_entry["options"] = {name: {"supported": passed} for name in options}
                methods[method_name] = method_entry
            mcp_entry["methods"] = methods

        agent = A2AAgent.get(self.a2a_agent_id)
        validated_ext = agent.metadata.get("validated_a2a_extensions", {})
        validated_ext[A2AAgent.EXT_MCP_CONFIG] = mcp_entry
        validated_data_ext = agent.metadata.get("validated_data_extensions", {})
        validated_data_ext["tool_call_count"] = {"supported": tool_call_count_reported}
        agent.update_metadata({**agent.metadata, "validated_a2a_extensions": validated_ext, "validated_data_extensions": validated_data_ext})

        return context

    async def _poll_task(self, a2a_url: str, task_id: str) -> dict:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll_interval_seconds)
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(f"{a2a_url}/a2a", json={
                        "jsonrpc": "2.0", "id": "poll", "method": "tasks/get",
                        "params": {"id": task_id},
                    }, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    if "error" in data:
                        logger.warning(f"A2A poll error (will retry): {data['error']}")
                        continue
                    result = data["result"]
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                logger.warning(f"A2A poll failed (will retry): {e}")
                continue
            state = result["status"]["state"]
            if state in ("completed", "failed"):
                return result
        raise TimeoutError(f"A2A task {task_id} did not complete within {self.timeout_seconds}s")

    @staticmethod
    async def _count_tool_calls(sandbox_id: str) -> int:
        """Read the env gateway trajectory from the gateway container and count tool_call events."""
        from agent_env.providers import get_env_sandbox_provider
        from agent_env.providers.gateway_provider import GatewayProvider
        sandbox = await get_env_sandbox_provider().get_sandbox(sandbox_id)
        events = await GatewayProvider().read_trajectory(sandbox)
        count = sum(1 for e in events if e.get("event_type") == "tool_call")
        return count
