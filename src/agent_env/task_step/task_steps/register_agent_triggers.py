"""Task step that registers agent triggers (urn:agentenv:triggers/v1) on a deployed stakeholder A2A agent."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, ClassVar, Optional

import httpx

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


def _referenced_env_triggers(when: Any) -> set[tuple[str, str]]:
    """(env_id, trigger_id) pairs named by env_trigger conditions anywhere in a (possibly all/any-nested) when."""
    refs: set[tuple[str, str]] = set()
    if not isinstance(when, dict):
        return refs
    if when.get("type") == "env_trigger":
        env_id, trigger_id = when.get("env_id"), when.get("trigger_id")
        if isinstance(env_id, str) and env_id and isinstance(trigger_id, str) and trigger_id:
            refs.add((env_id, trigger_id))
    for sub in when.get("of") or []:
        refs |= _referenced_env_triggers(sub)
    return refs


def _env_trigger_conditions(when: Any, trail: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict]]:
    """Every env_trigger condition in a (possibly all/any-nested) when, with its path parts under it."""
    if not isinstance(when, dict):
        return
    if when.get("type") == "env_trigger":
        yield trail, when
    for i, sub in enumerate(when.get("of") or []):
        yield from _env_trigger_conditions(sub, (*trail, f"of[{i}]"))


class RegisterAgentTriggersStep(TaskStep):
    type: ClassVar[str] = "register_agent_triggers"
    entity_refs = (EntityRef.env("triggers[].when.env_id", walk=_env_trigger_conditions),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        agent_name: str,
        triggers: list[dict],
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not isinstance(triggers, list) or not all(isinstance(t, dict) for t in triggers):
            raise ValueError("triggers must be a list of objects")
        self.agent_name = agent_name
        self.triggers = triggers

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["agent_name"] = self.agent_name
        base["triggers"] = self.triggers
        return base

    @classmethod
    def from_dict(cls, data: dict) -> RegisterAgentTriggersStep:
        return cls(
            **cls._base_from_dict(data),
            agent_name=data["agent_name"],
            triggers=data["triggers"],
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        agent = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
        if agent is None:
            raise RuntimeError(f"Agent '{self.agent_name}' not found in context.deployed_agents")
        card = agent.a2a_card or {}
        ext = A2AAgent.find_extension(card, A2AAgent.EXT_TRIGGERS)
        if ext is None:
            raise RuntimeError(f"Agent '{self.agent_name}' does not advertise {A2AAgent.EXT_TRIGGERS}")
        deployed_env_ids = {d.env_id for d in context.deployed_envs}
        registered_env_triggers: dict[str, set] = {}
        for reg in context.metadata.get("env_trigger_registrations", []):
            registered_env_triggers.setdefault(reg["env_id"], set()).update(reg.get("added", []))
        for t in self.triggers:
            for env_id, trigger_id in _referenced_env_triggers(t.get("when")):
                if env_id not in deployed_env_ids:
                    raise RuntimeError(
                        f"trigger '{t.get('id', '?')}' references env_trigger env_id '{env_id}' "
                        f"not in context.deployed_envs {sorted(deployed_env_ids)}")
                if trigger_id not in registered_env_triggers.get(env_id, set()):
                    raise RuntimeError(
                        f"trigger '{t.get('id', '?')}' references env_trigger '{trigger_id}' not registered "
                        f"on env '{env_id}' by a register_env_triggers step (registered there: "
                        f"{sorted(registered_env_triggers.get(env_id, set()))})")
        endpoint = (ext.get("params") or {}).get("endpoint", "/ext/triggers")
        body: dict = {"triggers": self.triggers}
        url = f"{agent.a2a_url or agent.api_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=body, timeout=30)
            if resp.status_code >= 400:
                raise RuntimeError(f"agent trigger registration failed (HTTP {resp.status_code}): {resp.text}")
            result = resp.json()
        logger.info(f"registered agent triggers on agent={self.agent_name}: {result}")
        context.metadata.setdefault("agent_trigger_registrations", []).append({
            "step_id": self.id,
            "agent_name": self.agent_name,
            "added": result.get("added", []),
        })
        return context
