"""Verify A2A agent declares LiteLLM cost-attribution support on its card.

Static check: reads the agent's `/.well-known/agent.json` (already fetched
at deploy time and stored on `deployed.a2a_card`) and inspects the
`urn:agentenv:agent-config/v1` extension. An agent "supports LiteLLM
attribution" when its agent-config extension advertises **both**
`project_id` and `task_id` in its `params.methods.set.request.supported`
list.

The check tells consumers (e.g. claude-code's auto-integration flow,
the hub UI) which agents will honor per-prompt attribution delivered
through `/ext/agent-config` from `PromptAgentTaskStep`. Agents that
don't advertise it will have those fields filtered out at the worker
side with a warning, and spend logs from those agents will not carry
the `projectId:<id>` / `taskId:<id>` tags.

This is a card-level static check only. It does NOT verify the agent
actually threads attribution into its outbound LLM calls — that's an
agent-implementation concern (e.g. an agent bridge's own
`_apply_litellm_attribution` helper). A future runtime probe could be added
if a self-reporting endpoint is exposed; for now, the static check
mirrors what an external integrator can discover via the AgentCard.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

# Body fields the worker (PromptAgentTaskStep) sends via /ext/agent-config
# for LiteLLM cost attribution. Both must be in the agent-config extension's
# `supported` list for the agent to be considered attribution-capable.
ATTRIBUTION_FIELDS = ("project_id", "task_id")


class VerifyA2ALitellmAttributionStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_litellm_attribution"

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
    def from_dict(cls, data: dict) -> "VerifyA2ALitellmAttributionStep":
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

        card = deployed.a2a_card or {}
        agent = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)

        ext = A2AAgent.find_extension(card, A2AAgent.EXT_AGENT_CONFIG)
        supported: list[str] = []
        if ext:
            supported = (
                ((ext.get("params") or {}).get("methods") or {})
                .get("set", {})
                .get("request", {})
                .get("supported")
                or []
            )

        advertises = {field: field in supported for field in ATTRIBUTION_FIELDS}
        missing = [field for field, present in advertises.items() if not present]
        passed = ext is not None and not missing

        if not ext:
            logger.warning(
                "Agent '%s' does not advertise the agent-config extension — "
                "LiteLLM attribution cannot flow.", self.a2a_agent_id,
            )
        elif missing:
            logger.warning(
                "Agent '%s' advertises agent-config but is missing attribution "
                "fields in its `supported` set: %s. Spend logs from this agent "
                "will not carry projectId/taskId tags. See an A2A agent bridge's "
                "`_apply_litellm_attribution` helper for a reference implementation.",
                self.a2a_agent_id, missing,
            )
        else:
            logger.info(
                "Agent '%s' supports LiteLLM attribution: advertises %s in "
                "agent-config.supported", self.a2a_agent_id, list(ATTRIBUTION_FIELDS),
            )

        validated = {
            "passed": passed,
            "extension_present": ext is not None,
            "supported_fields": supported,
            "advertises": advertises,
            "missing": missing,
        }
        agent.update_metadata({**agent.metadata, "validated_litellm_attribution": validated})
        context.metadata.setdefault("verifications", {})["a2a_litellm_attribution"] = validated
        return context
