"""browser_set_local_storage and browser_clear_local_storage: an origin's local storage set or cleared in an env's
website browser."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agentenv_protocol.website_browser import origin_of

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.task_step.task_steps.browser.base import BrowserStep

logger = logging.getLogger(__name__)


class BrowserSetLocalStorageTaskStep(BrowserStep):
    """Set each of ``items``, a map of keys to values, in ``origin``'s local storage, leaving the origin's other keys
    as they are."""

    type: ClassVar[str] = "browser_set_local_storage"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, origin: str, items: dict[str, str],
                 browser: Optional[str] = None, depends_on: Optional[list[TaskStepDependency]] = None,
                 fail_task_on_error: bool = True, context: Optional[str] = None):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         context=context)
        self.origin, self.items = origin, items

    def config_problems(self) -> list[str]:
        problems = super().config_problems() + _origin_problems(self.id, self.origin, required=True)
        if (not isinstance(self.items, dict) or not self.items
                or not all(isinstance(key, str) and isinstance(value, str) for key, value in self.items.items())):
            problems.append(f"step {self.id!r}: items must be a non-empty map of keys to string values")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "origin": self.origin, "items": self.items}

    @classmethod
    def from_dict(cls, data: dict) -> BrowserSetLocalStorageTaskStep:
        return cls(**cls._in_context_from_dict(data), origin=data["origin"], items=data["items"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        await self._call(context, "local_storage/set", {"origins": [{
            "origin": self.origin, "localStorage": [{"name": key, "value": value} for key, value in self.items.items()]}]})
        logger.info(f"{self.id}: set {sorted(self.items)} in {self.origin}'s local storage in the browser of env "
                    f"{self.env_id!r}")
        return context


class BrowserClearLocalStorageTaskStep(BrowserStep):
    """Clear ``origin``'s local storage in the browser, or every origin's it has seen when it names none."""

    type: ClassVar[str] = "browser_clear_local_storage"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, origin: Optional[str] = None,
                 browser: Optional[str] = None, depends_on: Optional[list[TaskStepDependency]] = None,
                 fail_task_on_error: bool = True, context: Optional[str] = None):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error,
                         context=context)
        self.origin = origin

    def config_problems(self) -> list[str]:
        return super().config_problems() + _origin_problems(self.id, self.origin, required=False)

    def to_dict(self) -> dict:
        return {**super().to_dict(), "origin": self.origin}

    @classmethod
    def from_dict(cls, data: dict) -> BrowserClearLocalStorageTaskStep:
        return cls(**cls._in_context_from_dict(data), origin=data.get("origin"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        cleared = (await self._call(context, "local_storage/clear",
                                    {} if self.origin is None else {"origins": [self.origin]}))["origins"]
        logger.info(f"{self.id}: cleared the local storage of {cleared} in the browser of env {self.env_id!r}")
        return context


def _origin_problems(step_id: str, origin: object, *, required: bool) -> list[str]:
    if origin is None and not required:
        return []
    if not isinstance(origin, str):
        return [f"step {step_id!r}: origin must be an http or https origin such as https://shop.test"]
    try:
        origin_of(origin, strict=True)
    except ValueError as e:
        return [f"step {step_id!r}: origin: {e}"]
    return []
