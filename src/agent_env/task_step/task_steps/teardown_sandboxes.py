"""Take down the sandboxes behind named agents, envs and bare sandboxes, mid-run.

A long pipeline keeps every sandbox it deployed until the run ends or its TTL expires. Place this
step after the last step that uses a target (via ``depends_on``) to stop paying for it early. It takes
them down as the end of a run does (``agent_env.task.teardown.tear_down``): the named agents' staged
changelogs drain, each named env closes, then every sandbox behind them is terminated by id.
Strictly best-effort: a sandbox that is already gone or fails to terminate is logged and the run
continues. The step never raises on a terminate, so a retry never re-runs it over partly torn-down
state, and ``fail_task_on_error`` must stay false. The ids it terminated are appended to
``metadata["torn_down_sandbox_ids"]``; the ``deployed_*`` records stay, since context list fields persist additions only.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent_env.a2a_agent.staging import LAST_DRAIN_SECONDS
from agent_env.entity.refs import EntityRef
from agent_env.env.env import DeployedEnv
from agent_env.task.teardown import (
    TERMINATE_TIMEOUT_SECONDS,
    TORN_DOWN_KEY,
    RecordedSandbox,
    before_teardown,
    env_sandbox_ids,
    tear_down,
)
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class TeardownSandboxesTaskStep(TaskStep):
    """Take down every sandbox behind the named deployed agents, envs and bare sandboxes."""

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
        sandboxes, envs, missing = self._resolve(context)
        if missing:
            logger.warning(f"teardown_sandboxes: not deployed in this run, skipping: {', '.join(missing)}")
        already = set(context.metadata.get(TORN_DOWN_KEY) or [])
        await before_teardown(context, LAST_DRAIN_SECONDS, agent_names=self.agent_names)
        report = await tear_down(context, sandboxes, envs, timeout=TERMINATE_TIMEOUT_SECONDS)
        for sandbox, why in report.failed:
            logger.warning(f"teardown_sandboxes: {sandbox.sandbox_id}: {why}"[:300])
        logger.info(
            f"teardown_sandboxes: terminated={len(report.terminated)} failed={len(report.failed)} "
            f"already_torn_down={sum(sandbox.sandbox_id in already for sandbox in sandboxes)} missing={len(missing)}"
        )
        return context

    def _resolve(self, context: TaskStepContext) -> tuple[list[RecordedSandbox], list[DeployedEnv], list[str]]:
        """Every sandbox behind the targets, once each, the env records named, and the names that matched nothing."""
        sandboxes: dict[str, RecordedSandbox] = {}
        envs: list[DeployedEnv] = []
        missing: list[str] = []
        for name in self.agent_names:
            agents = [a for a in context.deployed_agents if a.agent_name == name]
            if not agents:
                missing.append(f"agent {name!r}")
            for agent in agents:
                if agent.sandbox_id:
                    sandboxes[agent.sandbox_id] = RecordedSandbox(agent.sandbox_id, agent.sandbox_type, "agent")
        for name in self.sandbox_names:
            found = [s for s in context.deployed_sandboxes if s.sandbox_name == name]
            if not found:
                missing.append(f"sandbox {name!r}")
            for sandbox in found:
                sandboxes[sandbox.sandbox_id] = RecordedSandbox(sandbox.sandbox_id, sandbox.sandbox_type, "sandbox")
        for env_id in self.env_ids:
            named = [e for e in context.deployed_envs if e.env_id == env_id]
            if not named:
                missing.append(f"env {env_id!r}")
            envs.extend(named)
            for env in named:
                sandboxes.update({sid: RecordedSandbox(sid, stype, "env") for sid, stype in env_sandbox_ids(env)})
        return list(sandboxes.values()), envs, missing
