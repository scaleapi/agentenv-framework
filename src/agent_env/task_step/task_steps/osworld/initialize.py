"""``osworld_initialize``: an OSWorld task set up on a computer through the controller ``osworld_deploy_controller``
deployed beside it. An OSWorld-V2 task (``osworld_v2_task_path``, a ``task_NNN.py`` whose ``TASK_CLASS.setup()`` the
controller runs) resets the controller and initializes the task; a classic one runs its ``init_config``, OSWorld's
``[{type, parameters}]``, through the controller's ``/setup``, which serves every setup type."""
from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.entity.refs import EntityRef
from agent_env.osworld import controller, guest
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class OSWorldInitializeTaskStep(TaskStep):
    type: ClassVar[str] = "osworld_initialize"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, init_config: Optional[list[dict]] = None,
                 osworld_v2_task_path: Optional[str] = None, osworld_v2_task_url: Optional[str] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        """``osworld_v2_task_path`` is an http(s) URL or an object of the object store under its artifacts, which the
        step signs for the controller to fetch; ``osworld_v2_task_url`` is its older name, which it wins over."""
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.init_config = init_config or []
        self.osworld_v2_task_path = osworld_v2_task_path or osworld_v2_task_url

    def to_dict(self) -> dict:
        base = {**super().to_dict(), "env_id": self.env_id, "init_config": self.init_config}
        if self.osworld_v2_task_path is not None:
            base["osworld_v2_task_path"] = self.osworld_v2_task_path
        return base

    @classmethod
    def from_dict(cls, data: dict) -> OSWorldInitializeTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], init_config=data.get("init_config"),
                   osworld_v2_task_path=data.get("osworld_v2_task_path"),
                   osworld_v2_task_url=data.get("osworld_v2_task_url"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        url = guest.controller_url(deployed)
        if self.osworld_v2_task_path:
            logger.info(f"OSWorld-V2 initialize '{self.id}': task_ref={self.osworld_v2_task_path}")
            # /initialize_task refuses to run before a reset.
            await controller.post(url, "/reset", {}, expect_json=False)
            answer = await controller.post(url, "/initialize_task", {
                "id": self.id, "osworld_v2_task_url": controller.resolve_task_ref(self.osworld_v2_task_path)})
            if answer.get("success") is False or answer.get("error"):
                raise RuntimeError(f"OSWorld-V2 initialize failed: {answer.get('error') or answer.get('message')}")
            return context
        if self.init_config:
            answer = await run_setup(url, self.init_config, what=f"osworld_initialize '{self.id}'")
            if sites := answer.get("results"):
                context.metadata.setdefault("osworld_seeded_sites", []).extend(sites)
        return context


async def run_setup(controller_url: str, config: list[dict], *, what: str) -> dict:
    """``config`` through the controller's ``/setup``; raises when it reports failure or a site it seeded failed, since
    a seed that answers 200 with every site failed leaves the apps on their own fixtures."""
    answer = await controller.post(controller_url, "/setup", {"config": config})
    if not answer.get("success", True):
        raise RuntimeError(f"{what}: setup failed: {answer.get('error') or answer}")
    if failed := [site for site in answer.get("results") or [] if isinstance(site, dict) and not site.get("ok", True)]:
        detail = "; ".join(f"{site.get('app')}: {site.get('error')}" for site in failed)
        raise RuntimeError(f"{what}: setup failed for {len(failed)} site(s): {detail}")
    return answer
