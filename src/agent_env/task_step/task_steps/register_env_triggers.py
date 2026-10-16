"""Task step that registers env triggers (urn:agentenv:triggers/v1) on a deployed env's gateway."""

from __future__ import annotations

import asyncio
import logging
import time
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

# The await reads the gateway's trigger state once per poll interval until the step's deadline; a read gets at most the
# read timeout, never starts without the minimum window left, and when it fails is simply retried on the next poll. A
# trigger that reads one of the terminal statuses can no longer fire, so the wait stops there. A failure message quotes
# at most this many gateway events.
_AWAIT_TARGET_STATUS = "fired"
_AWAIT_TERMINAL_STATUSES = ("failed", "cancelled")
_AWAIT_POLL_INTERVAL_S = 1.0
_STATE_READ_TIMEOUT_S = 15
_STATE_READ_MIN_TIMEOUT_S = 1
_AWAIT_FAILURE_EVENTS_SHOWN = 6


class RegisterEnvTriggersStep(TaskStep):
    """An nl action may name its `executor`, and a trigger's `when.roles` may list executor roles so personas react to
    each other. Such a chain ends when a persona's turn makes no matching write, so give conversational nl actions no
    `verify` (its retry forces a write), give persona prompts a stop condition, and prefer a `where` off-ramp. An
    action's `delay` and `executor_hours` pace replies in virtual time, and a time trigger's `unless` makes it a deadline.
    `await_fired` holds the step until the named one-shot triggers have fired (a recurrence re-arms after each fire), so
    a dependent step starts only once start-of-run time triggers have landed in the world."""

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
        await_fired: Optional[list[str]] = None,
        await_timeout_seconds: float = 120,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not isinstance(triggers, list) or not all(isinstance(t, dict) for t in triggers):
            raise ValueError("triggers must be a list of objects")
        if await_fired is not None and (not isinstance(await_fired, list)
                                        or not all(isinstance(i, str) and i for i in await_fired)):
            raise ValueError("await_fired must be a list of trigger ids")
        if (isinstance(await_timeout_seconds, bool) or not isinstance(await_timeout_seconds, (int, float))
                or await_timeout_seconds <= 0):
            raise ValueError("await_timeout_seconds must be a positive number")
        self.env_id = env_id
        self.triggers = triggers
        self.watch_roles = watch_roles
        self.executor_agent_name = executor_agent_name
        self.executor_timeout_seconds = executor_timeout_seconds
        self.executor_hours = executor_hours
        self.await_fired = list(dict.fromkeys(await_fired)) if await_fired else None
        self.await_timeout_seconds = float(await_timeout_seconds)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["triggers"] = self.triggers
        base["watch_roles"] = self.watch_roles
        base["executor_agent_name"] = self.executor_agent_name
        base["executor_timeout_seconds"] = self.executor_timeout_seconds
        base["executor_hours"] = self.executor_hours
        base["await_fired"] = self.await_fired
        base["await_timeout_seconds"] = self.await_timeout_seconds
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
            await_fired=data.get("await_fired"),
            await_timeout_seconds=data.get("await_timeout_seconds", 120),
        )

    def preflight(self) -> list[str]:
        unknown = self._unknown_awaited()
        if unknown:
            return [f"register_env_triggers '{self.id}': await_fired names {unknown}, which this step does not register"]
        return []

    def _unknown_awaited(self) -> list[str]:
        registered = {t.get("id") for t in self.triggers}
        return [tid for tid in (self.await_fired or []) if tid not in registered]

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        problems = self.preflight()
        if problems:
            raise RuntimeError("; ".join(problems))
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
        if self.await_fired and not deployed.supports(EXT_TRIGGERS_URI, "state"):
            raise RuntimeError(
                f"Env '{self.env_id}' runs a gateway whose triggers/v1 does not offer 'state', which await_fired needs; "
                f"redeploy it on a gateway image that does.")
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
        entry = {
            "step_id": self.id,
            "env_id": self.env_id,
            "added": result.get("added", []),
            "executor_agent_name": self.executor_agent_name,
            "executor_agent_names": named,
        }
        # Recorded before the wait: the triggers are on the gateway whether or not the wait succeeds, and later steps
        # look them up here. `awaited` then carries the last statuses seen, complete on success.
        context.metadata.setdefault("env_trigger_registrations", []).append(entry)
        if self.await_fired:
            entry["awaited"] = {tid: None for tid in self.await_fired}
            await self._await_statuses(deployed, entry["awaited"])
        return context

    async def _await_statuses(self, deployed: DeployedEnv, seen: dict[str, Optional[str]]) -> None:
        """Poll the gateway, updating `seen` in place, until every awaited trigger has fired; raise when one reaches a
        terminal status first or when the deadline passes. A read starts only with a full minimum read window left
        before the deadline and is cut to that remainder, so the step never waits past `await_timeout_seconds`; a read
        that fails is retried on the next poll, and the last failure is reported if the deadline passes."""
        deadline = time.monotonic() + self.await_timeout_seconds
        state: dict = {}
        read_error: Optional[Exception] = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining < _STATE_READ_MIN_TIMEOUT_S:
                raise RuntimeError(self._await_failure(seen, state, stuck=[], read_error=read_error))
            try:
                state = await deployed.invoke(EXT_TRIGGERS_URI, "state", timeout=min(_STATE_READ_TIMEOUT_S, remaining))
                read_error = None
            except Exception as e:  # noqa: BLE001 - any read failure is retried on the next poll
                read_error = e
            else:
                statuses = {t.get("id"): t.get("status") for t in state.get("triggers", []) if isinstance(t, dict)}
                seen.update({tid: statuses.get(tid) for tid in self.await_fired})
                if all(status == _AWAIT_TARGET_STATUS for status in seen.values()):
                    logger.info(f"awaited triggers on env={self.env_id} fired: {sorted(seen)}")
                    return
                stuck = sorted(tid for tid, status in seen.items() if status in _AWAIT_TERMINAL_STATUSES)
                if stuck:
                    raise RuntimeError(self._await_failure(seen, state, stuck))
            await asyncio.sleep(max(0.0, min(_AWAIT_POLL_INTERVAL_S, deadline - time.monotonic())))

    def _await_failure(self, seen: dict[str, Optional[str]], state: dict, stuck: list[str],
                       read_error: Optional[Exception] = None) -> str:
        if stuck:
            why = f"awaited triggers reached a terminal status before firing: {stuck}"
        elif read_error is not None:
            why = f"could not read /triggers/state within the {self.await_timeout_seconds:g}s await budget: {read_error}"
        else:
            why = f"awaited triggers did not all fire within {self.await_timeout_seconds:g}s"
        return f"Env '{self.env_id}': {why}; statuses {dict(seen)}" + _failure_detail(state, set(seen))

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


def _failure_detail(state: dict, trigger_ids: set) -> str:
    """The gateway's failure events for `trigger_ids`, newest last, without their sequence and clock stamps."""
    events = [e for e in state.get("events", []) if isinstance(e, dict) and e.get("trigger_id") in trigger_ids
              and e.get("kind") in ("failed", "eval_error", "action_failed")]
    if not events:
        return ""
    shown = [{k: v for k, v in e.items() if k not in ("seq", "ts", "virtual_time")}
             for e in events[-_AWAIT_FAILURE_EVENTS_SHOWN:]]
    return f"; failures: {shown}"


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
