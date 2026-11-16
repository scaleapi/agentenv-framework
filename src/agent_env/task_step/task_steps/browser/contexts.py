"""browser_lease_context and browser_release_context: a context of its own leased in an env's website browser by a
label, which the other browser steps act in when they name it as their ``context``, and ended."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agentenv_protocol.website_browser.contexts import label_problem

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.task_step.task_steps.browser.base import BrowserStep

logger = logging.getLogger(__name__)


class BrowserLeaseContextTaskStep(BrowserStep):
    """Lease a context of its own in the browser as ``label``: its own cookies, local storage, tabs and downloads
    folder, recorded as granted to each agent ``grants`` names, which nothing enforces yet. A label leased before finds
    that lease, so a retried step is harmless; a reset, or the browser's relaunch, ends every lease."""

    type: ClassVar[str] = "browser_lease_context"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, label: str, grants: Optional[list[str]] = None,
                 browser: Optional[str] = None, depends_on: Optional[list[TaskStepDependency]] = None,
                 fail_task_on_error: bool = True):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.label, self.grants = label, grants if grants is not None else []

    def config_problems(self) -> list[str]:
        problems = super().config_problems()
        if problem := label_problem(self.label):
            problems.append(f"step {self.id!r}: label: {problem}")
        if not isinstance(self.grants, list) or not all(isinstance(name, str) and name for name in self.grants):
            problems.append(f"step {self.id!r}: grants, when given, must be a list of agent names")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "label": self.label, "grants": self.grants}

    @classmethod
    def from_dict(cls, data: dict) -> BrowserLeaseContextTaskStep:
        return cls(**cls._browser_from_dict(data), label=data.get("label"), grants=data.get("grants"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        leased = await self._call(context, "contexts/lease", {"label": self.label})
        for name in self.grants:
            await self._call(context, "contexts/grant", {"context": self.label, "user": name})
        logger.info(f"{self.id}: leased the context {self.label!r} ({leased['id']}) in the browser of env "
                    f"{self.env_id!r}" + (f", granted to {self.grants}" if self.grants else ""))
        self._record(context, label=leased["context"], id=leased["id"], grants=list(self.grants))
        return context


class BrowserReleaseContextTaskStep(BrowserStep):
    """End the lease ``label`` in the browser: its tabs closed and its folder deleted, downloads and all. A label that
    isn't leased is left as it is, which is recorded, so a retried step is harmless."""

    type: ClassVar[str] = "browser_release_context"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, label: str, browser: Optional[str] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.label = label

    def config_problems(self) -> list[str]:
        problems = super().config_problems()
        if problem := label_problem(self.label, by_task=False):
            problems.append(f"step {self.id!r}: label: {problem}")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "label": self.label}

    @classmethod
    def from_dict(cls, data: dict) -> BrowserReleaseContextTaskStep:
        return cls(**cls._browser_from_dict(data), label=data.get("label"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        released = (await self._call(context, "contexts/release", {"context": self.label}))["released"]
        logger.info(f"{self.id}: {'released' if released else 'found no lease'} {self.label!r} in the browser of env "
                    f"{self.env_id!r}")
        self._record(context, label=self.label, released=released)
        return context
