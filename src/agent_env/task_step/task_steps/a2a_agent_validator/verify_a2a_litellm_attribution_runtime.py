"""Verify A2A agent actually emits LiteLLM attribution on outbound LLM calls.

Optional behavioral counterpart to the static
`VerifyA2ALitellmAttributionStep`. When a runtime advertises its private probe,
this step asks whether it actually forwards project_id/task_id to the LLM call.
The probe is not part of the general AgentEnv SDK contract.

How it works:
  1. The validator plants unique probe values in `initial_context.metadata`
     before any prompt fires.
  2. The standard `prompt_agent` flow POSTs those values via
     `/ext/agent-config` and sends a prompt — the agent processes it,
     makes an outbound LLM call, and records the attribution it claims
     to have emitted in module-level state.
  3. This step queries the agent's `/ext/attribution-probe` extension to
     read back that recorded state.
  4. Passes when the recorded values match the planted probe values.

Catches:
  - Bridge ignores project_id/task_id from agent-config
  - Bridge sets the wrong env var / header names
  - Agent advertises support but never wires it through

Does NOT catch:
  - CLI eats the env var and never emits the actual HTTP header (probe
    trusts the bridge's claim about what it sent)
  - Proxy strips the header on receive

For full end-to-end coverage of the latter two, a spend-log probe would
be needed (requires proxy admin access).
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

EXT_ATTRIBUTION_PROBE_URI = "urn:agentenv:attribution-probe/v1"
EXT_ATTRIBUTION_PROBE_PATH = "/ext/attribution-probe"


class VerifyA2ALitellmAttributionRuntimeStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_litellm_attribution_runtime"

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        expected_project_id: str,
        expected_task_id: str,
        a2a_agent_version: Optional[int] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.a2a_agent_version = a2a_agent_version
        self.expected_project_id = expected_project_id
        self.expected_task_id = expected_task_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        base["a2a_agent_version"] = self.a2a_agent_version
        base["expected_project_id"] = self.expected_project_id
        base["expected_task_id"] = self.expected_task_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "VerifyA2ALitellmAttributionRuntimeStep":
        return cls(
            **cls._base_from_dict(data),
            a2a_agent_id=data["a2a_agent_id"],
            a2a_agent_version=data.get("a2a_agent_version"),
            expected_project_id=data["expected_project_id"],
            expected_task_id=data["expected_task_id"],
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent

        deployed = next((a for a in context.deployed_agents), None)
        if deployed is None:
            raise RuntimeError("No deployed agent found in context")

        agent = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)

        # This is a runtime-specific diagnostic, not a general SDK capability.
        # Its absence is therefore not a conformance failure.
        probe_ext = A2AAgent.find_extension(deployed.a2a_card or {}, EXT_ATTRIBUTION_PROBE_URI)
        if probe_ext is None:
            logger.info(
                "Agent '%s' does not advertise the attribution-probe extension "
                "(%s) — skipping optional runtime attribution verification.",
                self.a2a_agent_id, EXT_ATTRIBUTION_PROBE_URI,
            )
            validated = {
                "passed": True,
                "skipped": True,
                "extension_present": False,
                "reason": "optional runtime attribution probe not advertised",
                "expected_project_id": self.expected_project_id,
                "expected_task_id": self.expected_task_id,
            }
            agent.update_metadata({**agent.metadata, "validated_litellm_attribution_runtime": validated})
            context.metadata.setdefault("verifications", {})["a2a_litellm_attribution_runtime"] = validated
            return context

        endpoint = (probe_ext.get("params") or {}).get("endpoint", EXT_ATTRIBUTION_PROBE_PATH)
        url = (deployed.a2a_url or deployed.api_url) + endpoint
        logger.info("Querying attribution probe at %s", url)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json={})
                resp.raise_for_status()
                probe_data = resp.json()
        except Exception as e:
            logger.warning("attribution-probe request failed: %s", e)
            validated = {
                "passed": False,
                "extension_present": True,
                "reason": f"probe request failed: {e}",
                "expected_project_id": self.expected_project_id,
                "expected_task_id": self.expected_task_id,
            }
            agent.update_metadata({**agent.metadata, "validated_litellm_attribution_runtime": validated})
            context.metadata.setdefault("verifications", {})["a2a_litellm_attribution_runtime"] = validated
            return context

        last_seen = probe_data.get("last_seen_attribution") or {}
        observed_project_id = last_seen.get("project_id")
        observed_task_id = last_seen.get("task_id")

        project_matches = observed_project_id == self.expected_project_id
        task_matches = observed_task_id == self.expected_task_id
        passed = project_matches and task_matches

        if passed:
            logger.info(
                "Runtime attribution verified for '%s': observed projectId=%s taskId=%s",
                self.a2a_agent_id, observed_project_id, observed_task_id,
            )
        else:
            logger.warning(
                "Runtime attribution mismatch for '%s': expected projectId=%s taskId=%s, "
                "observed projectId=%s taskId=%s",
                self.a2a_agent_id, self.expected_project_id, self.expected_task_id,
                observed_project_id, observed_task_id,
            )

        validated = {
            "passed": passed,
            "extension_present": True,
            "expected_project_id": self.expected_project_id,
            "expected_task_id": self.expected_task_id,
            "observed_project_id": observed_project_id,
            "observed_task_id": observed_task_id,
            "project_matches": project_matches,
            "task_matches": task_matches,
            "last_seen_at_utc": probe_data.get("last_seen_at_utc"),
        }
        agent.update_metadata({**agent.metadata, "validated_litellm_attribution_runtime": validated})
        context.metadata.setdefault("verifications", {})["a2a_litellm_attribution_runtime"] = validated
        return context
