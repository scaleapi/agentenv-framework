"""Terminate the sandboxes behind named agents, envs and bare sandboxes, mid-run.

A long pipeline keeps every sandbox it deployed until the run ends or its TTL expires. Place this
step after the last step that uses a target (via ``depends_on``) to stop paying for it early.
Strictly best-effort: a sandbox that is already gone or fails to terminate is logged and the run
continues. The step never raises on a terminate, so a retry never re-runs it over partly torn-down
state, and ``fail_task_on_error`` must stay false. The ids it terminated are appended to ``metadata["torn_down_sandbox_ids"]``; the
``deployed_*`` records stay, since context list fields persist additions only.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from agent_env.a2a_agent.staging import LAST_DRAIN_SECONDS, drain_all, staged_changelogs
from agent_env.entity_refs import EntityRef
from agent_env.env.env import DeployedSandboxEnv
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxProvider,
    build_sandbox_provider,
    get_agent_sandbox_provider,
    get_env_sandbox_provider,
    get_sandbox_provider,
)
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

TORN_DOWN_KEY = "torn_down_sandbox_ids"


class TeardownSandboxesTaskStep(TaskStep):
    """Terminate every sandbox behind the named deployed agents, envs and bare sandboxes."""

    type = "teardown_sandboxes"
    entity_refs = (EntityRef.env("env_ids[]"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        agent_names: Optional[list[str]] = None,
        env_ids: Optional[list[str]] = None,
        sandbox_names: Optional[list[str]] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = False,
    ):
        if fail_task_on_error:
            raise ValueError(
                "teardown_sandboxes is best-effort (a failed terminate is logged, never raised); "
                "fail_task_on_error must be false"
            )
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.agent_names = list(agent_names or [])
        self.env_ids = list(env_ids or [])
        self.sandbox_names = list(sandbox_names or [])
        if not (self.agent_names or self.env_ids or self.sandbox_names):
            raise ValueError("teardown_sandboxes needs at least one of agent_names, env_ids, sandbox_names")

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["agent_names"] = self.agent_names
        base["env_ids"] = self.env_ids
        base["sandbox_names"] = self.sandbox_names
        return base

    @classmethod
    def from_dict(cls, data: dict) -> TeardownSandboxesTaskStep:
        base = cls._base_from_dict(data)
        # Stay best-effort when unset: _base_from_dict would otherwise default it to True.
        base["fail_task_on_error"] = data.get("fail_task_on_error", False)
        return cls(
            **base,
            agent_names=data.get("agent_names"),
            env_ids=data.get("env_ids"),
            sandbox_names=data.get("sandbox_names"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        targets, missing = self._resolve(context)
        if missing:
            logger.warning(f"teardown_sandboxes: not deployed in this run, skipping: {', '.join(missing)}")
        already = set(context.metadata.get(TORN_DOWN_KEY) or [])
        pending = {sid: (stype, default) for sid, (stype, default) in targets.items() if sid not in already}
        for name in self.agent_names:  # a changelog staged on an agent goes into the store before the agent does
            await drain_all(staged_changelogs(context.metadata, agent_name=name), within=LAST_DRAIN_SECONDS)
        results = await asyncio.gather(
            *(_terminate(sid, stype, default) for sid, (stype, default) in pending.items()),
            return_exceptions=True,
        )
        terminated, failed = [], []
        for sandbox_id, result in zip(pending, results):
            if isinstance(result, BaseException):
                failed.append(sandbox_id)
                logger.warning(f"teardown_sandboxes: terminate {sandbox_id} failed: {result!r}"[:300])
            else:
                terminated.append(sandbox_id)
        if terminated:
            context.metadata[TORN_DOWN_KEY] = [*context.metadata.get(TORN_DOWN_KEY, []), *terminated]
        logger.info(
            f"teardown_sandboxes: terminated={len(terminated)} failed={len(failed)} "
            f"already_torn_down={len(targets) - len(pending)} missing={len(missing)}"
        )
        return context

    def _resolve(
        self, context: TaskStepContext
    ) -> tuple[dict[str, tuple[Optional[str], str]], list[str]]:
        """``{sandbox_id: (sandbox_type, default_slot)}`` for every target, plus unmatched names."""
        targets: dict[str, tuple[Optional[str], str]] = {}
        missing: list[str] = []
        for name in self.agent_names:
            agents = [a for a in context.deployed_agents if a.agent_name == name]
            if not agents:
                missing.append(f"agent {name!r}")
            for agent in agents:
                if agent.sandbox_id:
                    targets[agent.sandbox_id] = (agent.sandbox_type, "agent")
        for name in self.sandbox_names:
            sandboxes = [s for s in context.deployed_sandboxes if s.sandbox_name == name]
            if not sandboxes:
                missing.append(f"sandbox {name!r}")
            for sandbox in sandboxes:
                targets[sandbox.sandbox_id] = (sandbox.sandbox_type, "sandbox")
        for env_id in self.env_ids:
            envs = [e for e in context.deployed_envs if e.env_id == env_id]
            if not envs:
                missing.append(f"env {env_id!r}")
            for env in envs:
                targets.update({sid: (stype, "env") for sid, stype in _env_sandbox_ids(env)})
        return targets, missing


def _env_sandbox_ids(env) -> list[tuple[str, Optional[str]]]:
    """Every (sandbox_id, sandbox_type) of a deployed env: its primary sandbox plus ``sandbox_ids``.

    ``sandbox_ids`` values are an id or a ``{service: id}`` dict and inherit the env's type, unless
    the key itself names a registered backend (a sandbox on another backend than the gateway's).
    """
    if not isinstance(env, DeployedSandboxEnv):  # an env outside our sandboxes owns its own lifetime
        return []
    ids = [(env.sandbox_id, env.sandbox_type)] if env.sandbox_id else []
    for key, value in (env.sandbox_ids or {}).items():
        entry_type = key if _is_backend(key) else env.sandbox_type
        values = [value] if isinstance(value, str) else [v for v in value.values() if isinstance(v, str)]
        ids.extend((v, entry_type) for v in values)
    return ids


def _is_backend(name: str) -> bool:
    try:
        build_sandbox_provider(name)
    except Exception:
        return False
    return True


_DEFAULT_PROVIDERS = {
    "agent": get_agent_sandbox_provider,
    "env": get_env_sandbox_provider,
    "sandbox": get_sandbox_provider,
}


async def _terminate(sandbox_id: str, sandbox_type: Optional[str], default_slot: str) -> None:
    provider: SandboxProvider = (
        build_sandbox_provider(sandbox_type) if sandbox_type else _DEFAULT_PROVIDERS[default_slot]()
    )
    sandbox = await provider.get_sandbox(sandbox_id)
    await sandbox.terminate()
