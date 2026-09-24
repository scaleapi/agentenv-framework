"""Verify A2A agent accepts a `role` in /ext/agent-config and round-trips it via GET.

Confirms two coupled pieces of the per-agent role-filtering plumbing:
  1. The agent advertises `role` in agent-config's supported set (the field the
     env gateway's role filter ultimately keys off via the AgentEnv-Role header).
  2. The agent advertises a `get` method on agent-config and that GET returns
     the in-memory config dict so callers (and this validator) can confirm what
     was actually accepted.

Catches:
  - Agent silently drops `role` from agent-config POST
  - Agent stores `role` somewhere unreadable / no GET method to confirm
  - Agent regresses on adding new agent-config fields

Does NOT catch:
  - Bridge ignores the stored role when building the MCP config (i.e. role is
    accepted but never threaded to the CLI subprocess). End-to-end coverage of
    that lives in the partition experiment task itself — runs that depend on
    role-based filtering will hard-fail if the header never reaches the gateway.
"""
from __future__ import annotations

import logging
import uuid
from typing import ClassVar, Optional

import httpx

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class VerifyA2ARoleStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_role"
    entity_refs = (EntityRef.agent("a2a_agent_id", version_field="a2a_agent_version"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        a2a_agent_version: Optional[int] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.a2a_agent_version = a2a_agent_version

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        base["a2a_agent_version"] = self.a2a_agent_version
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyA2ARoleStep:
        return cls(
            **cls._base_from_dict(data),
            a2a_agent_id=data["a2a_agent_id"],
            a2a_agent_version=data.get("a2a_agent_version"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent

        deployed = next((a for a in context.deployed_agents), None)
        if deployed is None:
            raise RuntimeError("No deployed agent found in context")
        agent = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)

        config_ext = A2AAgent.find_extension(deployed.a2a_card or {}, A2AAgent.EXT_AGENT_CONFIG)
        if config_ext is None:
            self._record(agent, context, {
                "passed": False,
                "reason": f"agent does not advertise {A2AAgent.EXT_AGENT_CONFIG}",
            })
            return context

        params = config_ext.get("params") or {}
        methods = params.get("methods") or {}
        supported = (methods.get("set") or {}).get("request", {}).get("supported", [])
        role_in_supported = "role" in supported
        get_method_advertised = "get" in methods

        if not role_in_supported or not get_method_advertised:
            self._record(agent, context, {
                "passed": False,
                "role_in_supported": role_in_supported,
                "get_method_advertised": get_method_advertised,
                "reason": "agent-config extension missing 'role' in supported and/or 'get' method",
            })
            return context

        endpoint = params.get("endpoint", "/ext/agent-config")
        url = (deployed.a2a_url or deployed.api_url) + endpoint
        test_role = f"validator-role-{uuid.uuid4().hex[:8]}"
        logger.info("Round-tripping role=%r through %s", test_role, url)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                post_resp = await client.post(url, json={"role": test_role})
                post_resp.raise_for_status()
                get_resp = await client.get(url)
                get_resp.raise_for_status()
                config = (get_resp.json() or {}).get("config") or {}
        except Exception as e:
            self._record(agent, context, {
                "passed": False,
                "role_in_supported": True,
                "get_method_advertised": True,
                "reason": f"agent-config POST/GET round-trip failed: {e}",
            })
            return context

        observed_role = config.get("role")
        passed = observed_role == test_role
        if passed:
            logger.info("Role round-trip verified for '%s'", self.a2a_agent_id)
        else:
            logger.warning(
                "Role round-trip mismatch for '%s': expected=%r observed=%r",
                self.a2a_agent_id, test_role, observed_role,
            )

        self._record(agent, context, {
            "passed": passed,
            "role_in_supported": True,
            "get_method_advertised": True,
            "expected_role": test_role,
            "observed_role": observed_role,
        })
        return context

    def _record(self, agent, context: TaskStepContext, validated: dict) -> None:
        # Write to context FIRST so the verification record is always preserved,
        # even if the agent-metadata CAS fails (other validator steps run in
        # parallel and may have updated metadata between our read and write).
        context.metadata.setdefault("verifications", {})["a2a_role"] = validated
        # Re-fetch fresh to minimize the CAS window. Swallow conflicts — the
        # record in context.metadata is the source of truth; metadata is just
        # a denormalized view for `agent get` inspection.
        from agent_env.a2a_agent import A2AAgent
        try:
            fresh = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)
            fresh.update_metadata({**fresh.metadata, "validated_role_via_agent_config": validated})
        except Exception as e:
            logger.warning("Failed to persist validated_role_via_agent_config (CAS conflict?): %s", e)
