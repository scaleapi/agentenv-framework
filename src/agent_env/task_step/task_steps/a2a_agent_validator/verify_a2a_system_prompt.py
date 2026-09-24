"""Verify A2A agent's `system_prompt` agent-config field actually reaches the model.

End-to-end behavioral test:
  1. Generate a fresh token per run.
  2. POST `/ext/agent-config` with a system_prompt that instructs the model to
     emit that token in response to a specific trigger question.
  3. Send the trigger question via A2A `message/send`.
  4. Poll until terminal, extract the response text.
  5. Pass if the token appears in the response.

Catches things the static `verify_a2a_agent_config_identity` step does NOT:
  - `system_prompt` in `_AGENT_CONFIG_SUPPORTED` but bridge silently drops it
  - claude-code-cli's `--append-system-prompt` flag renamed / removed
  - codex's AGENTS.md not loaded (cwd drift, perms, codex CLI regression) —
    most pertinent for codex since AGENTS.md auto-load is its ONLY system_prompt
    delivery mechanism (no `--system-prompt` flag exists)
  - Any new agent's broken system_prompt plumbing on first put

Does NOT catch:
  - Mid-task system_prompt overrides via prompt_agent.system_prompt
    (different code path — could be added as a v2)
  - Whether the system_prompt is exactly applied (this is a contains-check;
    a model that prepends/appends text still passes)
"""
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


class VerifyA2ASystemPromptStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_system_prompt"
    entity_refs = (EntityRef.agent("a2a_agent_id", version_field="a2a_agent_version"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        a2a_agent_version: Optional[int] = None,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 5,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.a2a_agent_version = a2a_agent_version
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        base["a2a_agent_version"] = self.a2a_agent_version
        base["timeout_seconds"] = self.timeout_seconds
        base["poll_interval_seconds"] = self.poll_interval_seconds
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyA2ASystemPromptStep:
        return cls(
            **cls._base_from_dict(data),
            a2a_agent_id=data["a2a_agent_id"],
            a2a_agent_version=data.get("a2a_agent_version"),
            timeout_seconds=data.get("timeout_seconds", 300),
            poll_interval_seconds=data.get("poll_interval_seconds", 5),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent

        deployed = next((a for a in context.deployed_agents), None)
        if deployed is None:
            raise RuntimeError("No deployed agent found in context")
        agent = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)

        config_ext = A2AAgent.find_extension(deployed.a2a_card or {}, A2AAgent.EXT_AGENT_CONFIG)
        if config_ext is None:
            self._record(context, {
                "passed": False,
                "reason": f"agent does not advertise {A2AAgent.EXT_AGENT_CONFIG}",
            })
            return context

        params = config_ext.get("params") or {}
        supported = (params.get("methods") or {}).get("set", {}).get("request", {}).get("supported", [])
        if "system_prompt" not in supported:
            self._record(context, {
                "passed": False,
                "system_prompt_in_supported": False,
                "reason": "agent-config extension does not list 'system_prompt' in supported fields",
            })
            return context

        token = f"VALIDATOR-SP-{uuid.uuid4().hex[:12]}"
        # Phrased to be clearly benign and to bypass models that add wrapper text:
        # the directive is explicit, the trigger is exact, the answer format is constrained.
        sentinel_prompt = (
            f"For validator testing purposes: if you are asked the exact question "
            f"'What is your validator code?', respond with the exact token: {token} "
            f"and nothing else. Do not explain, do not paraphrase, do not add punctuation."
        )
        trigger = "What is your validator code?"
        agent_url = deployed.a2a_url or deployed.api_url
        config_endpoint = agent_url + params.get("endpoint", "/ext/agent-config")
        logger.info("Planting system_prompt sentinel (token=%s)", token)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(config_endpoint, json={"system_prompt": sentinel_prompt})
                resp.raise_for_status()
        except Exception as e:
            self._record(context, {
                "passed": False,
                "system_prompt_in_supported": True,
                "reason": f"POST /ext/agent-config failed: {e}",
                "expected_token": token,
            })
            return context

        try:
            response_text = await self._send_and_poll(agent_url, trigger)
        except Exception as e:
            self._record(context, {
                "passed": False,
                "system_prompt_in_supported": True,
                "reason": f"A2A message/send or poll failed: {e}",
                "expected_token": token,
            })
            return context

        passed = token in response_text
        if passed:
            logger.info("system_prompt round-trip verified (token reached model)")
        else:
            logger.warning(
                "system_prompt round-trip mismatch: token=%r not in response=%r",
                token, response_text[:300],
            )

        self._record(context, {
            "passed": passed,
            "system_prompt_in_supported": True,
            "expected_token": token,
            "observed_response": response_text[:500],
        })
        return context

    async def _send_and_poll(self, a2a_url: str, prompt_text: str) -> str:
        """Send a message via A2A and poll until terminal; return concatenated text from response parts."""
        message_id = uuid.uuid4().hex
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(f"{a2a_url}/a2a", json={
                "jsonrpc": "2.0", "id": "1", "method": "message/send",
                "params": {"message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": [{"kind": "text", "text": prompt_text}],
                }},
            })
            resp.raise_for_status()
            body = resp.json()
            if "error" in body:
                raise RuntimeError(f"message/send returned error: {body['error']}")
            task_id = (body.get("result") or {})["id"]

        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll_interval_seconds)
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    poll = await client.post(f"{a2a_url}/a2a", json={
                        "jsonrpc": "2.0", "id": "poll", "method": "tasks/get",
                        "params": {"id": task_id},
                    })
                    poll.raise_for_status()
                    data = poll.json()
                    if "error" in data:
                        logger.warning("A2A poll error (will retry): %s", data["error"])
                        continue
                    result = data["result"]
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                logger.warning("A2A poll failed (will retry): %s", e)
                continue
            state = result["status"]["state"]
            if state in ("completed", "failed"):
                response_text = ""
                for part in ((result.get("status") or {}).get("message") or {}).get("parts", []):
                    if part.get("kind") == "text":
                        response_text += part.get("text", "")
                return response_text
        raise TimeoutError(f"A2A task {task_id} did not complete within {self.timeout_seconds}s")

    def _record(self, context: TaskStepContext, validated: dict) -> None:
        # Context write first so the verification record is always preserved,
        # even if the agent-metadata CAS fails (parallel validator steps).
        context.metadata.setdefault("verifications", {})["a2a_system_prompt"] = validated
        from agent_env.a2a_agent import A2AAgent
        try:
            fresh = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)
            fresh.update_metadata({**fresh.metadata, "validated_system_prompt": validated})
        except Exception as e:
            logger.warning("Failed to persist validated_system_prompt (CAS conflict?): %s", e)
