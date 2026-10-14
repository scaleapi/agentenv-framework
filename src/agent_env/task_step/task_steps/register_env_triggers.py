"""Task step that registers env triggers (urn:agentenv:triggers/v1) on a deployed env's gateway."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx
from agentenv_protocol import client as protocol_v1

from agent_env.env.env import DeployedEnv, DeployedSandboxEnv
from agent_env.env.gateway.constants import (
    EXT_TRIGGERS_URI,
    TRIGGER_ACTION_DELAY,
    TRIGGER_EXECUTOR_HOURS,
    TRIGGER_NAMED_EXECUTORS,
    TRIGGER_PROVOKING_ROLES,
    TRIGGER_UNLESS,
)
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class RegisterEnvTriggersStep(TaskStep):
    """An nl action may name its `executor`, and a trigger's `when.roles` may list executor roles so personas react to
    each other. Such a chain ends when a persona's turn makes no matching write, so give conversational nl actions no
    `verify` (its retry forces a write), give persona prompts a stop condition, and prefer a `where` off-ramp. An
    action's `delay` and `executor_hours` pace replies in virtual time, and a time trigger's `unless` makes it a deadline."""

    type: ClassVar[str] = "register_env_triggers"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        triggers: list[dict],
        watch_roles: Optional[list[str]] = None,
        executor_agent_name: Optional[str] = None,
        executor_timeout_seconds: float = 120,
        executor_hours: Optional[dict[str, dict]] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not isinstance(triggers, list) or not all(isinstance(t, dict) for t in triggers):
            raise ValueError("triggers must be a list of objects")
        self.env_id = env_id
        self.triggers = triggers
        self.watch_roles = watch_roles
        self.executor_agent_name = executor_agent_name
        self.executor_timeout_seconds = executor_timeout_seconds
        self.executor_hours = executor_hours

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["triggers"] = self.triggers
        base["watch_roles"] = self.watch_roles
        base["executor_agent_name"] = self.executor_agent_name
        base["executor_timeout_seconds"] = self.executor_timeout_seconds
        base["executor_hours"] = self.executor_hours
        return base

    @classmethod
    def from_dict(cls, data: dict) -> RegisterEnvTriggersStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            triggers=data["triggers"],
            watch_roles=data.get("watch_roles"),
            executor_agent_name=data.get("executor_agent_name"),
            executor_timeout_seconds=data.get("executor_timeout_seconds", 120),
            executor_hours=data.get("executor_hours"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        body: dict = {"triggers": self.triggers}
        if self.watch_roles is not None:
            body["watch_roles"] = self.watch_roles
        if self.executor_agent_name is not None:
            body["executor"] = self._executor_config(context, deployed, self.executor_agent_name)
        named = _named_executors(self.triggers)
        unused_hours = sorted(set(self.executor_hours or {}) - {self.executor_agent_name, *named})
        if unused_hours:
            raise RuntimeError(f"executor_hours names {unused_hours}, which no nl action or executor_agent_name uses")
        # An older gateway ignores what it does not declare: named actions would go to the default executor, a
        # trigger's when.roles would leave it to the watched roles, delays and hours would not wait, and a deadline
        # would fire even after its `unless` call.
        used = {TRIGGER_NAMED_EXECUTORS: bool(named), TRIGGER_PROVOKING_ROLES: _when_has(self.triggers, "roles"),
                TRIGGER_ACTION_DELAY: _delays(self.triggers), TRIGGER_EXECUTOR_HOURS: bool(self.executor_hours),
                TRIGGER_UNLESS: _when_has(self.triggers, "unless")}
        missing = [feature for feature, needed in used.items() if needed and feature not in _trigger_features(deployed)]
        if missing:
            raise RuntimeError(
                f"Env '{self.env_id}' runs a gateway whose triggers/v1 does not declare {missing}, which these triggers "
                f"use; redeploy it on a gateway image that does.")
        if named:
            body["executors"] = {name: self._executor_config(context, deployed, name) for name in named}
        try:
            result = await deployed.invoke(EXT_TRIGGERS_URI, "register", body)
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"trigger registration failed (HTTP {e.response.status_code}): {e.response.text}") from e
        logger.info(f"registered triggers on env={self.env_id}: {result}")
        context.metadata.setdefault("env_trigger_registrations", []).append({
            "step_id": self.id,
            "env_id": self.env_id,
            "added": result.get("added", []),
            "executor_agent_name": self.executor_agent_name,
            "executor_agent_names": named,
        })
        return context

    def _executor_config(self, context: TaskStepContext, deployed: DeployedEnv, agent_name: str) -> dict:
        agent = next((a for a in context.deployed_agents if a.agent_name == agent_name), None)
        if agent is None:
            raise RuntimeError(f"Executor agent '{agent_name}' not found in context.deployed_agents")
        if not agent.role:
            raise RuntimeError(
                f"Executor agent '{agent_name}' was deployed without an explicit role; "
                f"it would run under the default role and its own tool calls would fire triggers. "
                f"Deploy it with a role outside watch_roles.")
        if self.watch_roles is not None and agent.role in self.watch_roles:
            raise RuntimeError(
                f"Executor agent role '{agent.role}' is in watch_roles {self.watch_roles} — "
                f"the executor's own tool calls would fire triggers (cascade).")
        # The env's gateway calls the executor; an env outside our sandboxes gets its URL as it is.
        executor_url = (agent.url_for(deployed.sandbox_type) if isinstance(deployed, DeployedSandboxEnv)
                        else agent.a2a_url or agent.api_url)
        config = {"a2a_url": executor_url, "timeout_seconds": self.executor_timeout_seconds, "role": agent.role}
        if agent_name in (self.executor_hours or {}):
            config["hours"] = self.executor_hours[agent_name]
        return config


def _named_executors(triggers: list[dict]) -> list[str]:
    """The executor agent names the triggers' nl actions name, in first-use order."""
    names = (action.get("executor") for trigger in triggers for action in trigger.get("actions") or []
             if isinstance(action, dict) and action.get("type") == "nl")
    return list(dict.fromkeys(name for name in names if isinstance(name, str) and name))


def _when_has(triggers: list[dict], key: str) -> bool:
    return any(isinstance(trigger.get("when"), dict) and trigger["when"].get(key) for trigger in triggers)


def _delays(triggers: list[dict]) -> bool:
    return any(isinstance(action, dict) and action.get("delay") is not None
               for trigger in triggers for action in trigger.get("actions") or [])


def _trigger_features(deployed: DeployedEnv) -> list:
    """The triggers/v1 features the env's gateway declares on its card; one that predates them declares none."""
    return protocol_v1.extension_params(deployed.environment_card or {}, EXT_TRIGGERS_URI).get("features") or []
