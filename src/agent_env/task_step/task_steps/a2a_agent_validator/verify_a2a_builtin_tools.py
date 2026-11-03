"""Verify an agent whose agent-config takes ``builtin_tools`` honours it. A second agent is deployed with
``{"disable": "*"}`` and no envs, so it has no MCP tools and any tool call it makes is one of its own: asked to run a
shell command, it must make none. On that agent, a tool its card doesn't name is refused, and agent-config reads the
selection back. The validation agent itself is never limited, since the other checks need its tools.

An agent whose agent-config doesn't take ``builtin_tools`` is recorded as not supporting it, which isn't a failure.

Does NOT catch: a model that wouldn't have used a tool it has, which reads as honoured.
"""
from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx

from agent_env.a2a_agent import A2AAgent
from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

logger = logging.getLogger(__name__)

AGENT_NAME = "builtin-tools-agent"
NO_SUCH_TOOL = "agentenv-validator-no-such-tool"
NO_TOOLS = "NO-TOOLS"
PROMPT = ("Run the shell command `uname -a` with a tool you have and reply with its exact output. If you have no tool "
          f"that can run a shell command, reply with exactly {NO_TOOLS} and nothing else.")


class VerifyA2ABuiltinToolsStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_builtin_tools"
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
    def from_dict(cls, data: dict) -> VerifyA2ABuiltinToolsStep:
        return cls(
            **cls._base_from_dict(data),
            a2a_agent_id=data["a2a_agent_id"],
            a2a_agent_version=data.get("a2a_agent_version"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        main = next((a for a in context.deployed_agents), None)
        if main is None:
            raise RuntimeError("No deployed agent found in context")
        params = (A2AAgent.find_extension(main.a2a_card or {}, A2AAgent.EXT_AGENT_CONFIG) or {}).get("params") or {}
        methods = params.get("methods") or {}
        if "builtin_tools" not in (methods.get("set") or {}).get("request", {}).get("supported", []):
            self._record(context, {"supported": False, "reason": "agent-config takes no builtin_tools"})
            return context

        prompt_id = f"{self.id}-prompt"
        try:
            await DeployAgentTaskStep(id=f"{self.id}-deploy", version=None, a2a_agent_id=self.a2a_agent_id,
                                      a2a_agent_version=self.a2a_agent_version, agent_name=AGENT_NAME,
                                      builtin_tools={"disable": "*"}).execute(context)
            limited = next(a for a in context.deployed_agents if a.agent_name == AGENT_NAME)
            url = (limited.a2a_url or limited.api_url) + params.get("endpoint", "/ext/agent-config")
            async with httpx.AsyncClient(timeout=30.0) as client:
                refused = (await client.post(url, json={"builtin_tools": {"only": [NO_SUCH_TOOL]}})).status_code
                config = (((await client.get(url)).json() or {}).get("config") or {}) if "get" in methods else None
            await PromptAgentTaskStep(id=prompt_id, version=None, prompt=PROMPT, prompt_id=prompt_id,
                                      agent_name=AGENT_NAME, timeout_seconds=180).execute(context)
        except Exception as e:
            self._record(context, {"supported": True, "passed": False, "reason": f"{type(e).__name__}: {e}"[:500]})
            return context

        response = next(r for r in reversed(context.prompt_responses) if r.prompt_id == prompt_id)
        count = response.tool_call_count
        checks = {
            "unknown_tool_refused": refused == 400,
            "read_back": None if config is None else _set(config.get("builtin_tools")) == {"disable": "*"},
            "no_tool_calls": count == 0 if count is not None else response.response.strip() == NO_TOOLS,
        }
        passed = all(check is not False for check in checks.values())
        if not passed:
            logger.warning("builtin_tools not honoured by '%s': %s", self.a2a_agent_id, checks)
        self._record(context, {"supported": True, "passed": passed, **checks, "tool_call_count": count,
                               "response": response.response[:500]})
        return context

    def _record(self, context: TaskStepContext, validated: dict) -> None:
        # As the role check: the context's record first, then the agent's metadata, whose CAS can lose to the other
        # validator steps running beside this one.
        context.metadata.setdefault("verifications", {})["a2a_builtin_tools"] = validated
        try:
            fresh = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)
            fresh.update_metadata({**fresh.metadata, "validated_builtin_tools_via_agent_config": validated})
        except Exception as e:
            logger.warning("Failed to persist validated_builtin_tools_via_agent_config (CAS conflict?): %s", e)


def _set(selection: dict | None) -> dict:
    """A selection as read back, without the fields it leaves unset, which an agent may return as null."""
    return {key: value for key, value in (selection or {}).items() if value is not None}
