"""browser_set_cookies and browser_delete_cookies: cookies set in, or deleted from, an env's website browser."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agentenv_protocol.website_browser import CookieToSet
from pydantic import ValidationError

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.task_step.task_steps.browser.base import BrowserStep

logger = logging.getLogger(__name__)


class BrowserSetCookiesTaskStep(BrowserStep):
    """Set ``cookies`` in the browser, all or none, each in the shape cookies/set takes: a ``url``, or a ``domain`` and
    ``path``, and ``max_age`` in place of ``expires`` for a lifetime from when it's set."""

    type: ClassVar[str] = "browser_set_cookies"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, cookies: list[dict],
                 browser: Optional[str] = None, depends_on: Optional[list[TaskStepDependency]] = None,
                 fail_task_on_error: bool = True):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.cookies = cookies

    def config_problems(self) -> list[str]:
        problems = super().config_problems()
        if not isinstance(self.cookies, list) or not self.cookies:
            return [*problems, f"step {self.id!r}: cookies must be a non-empty list"]
        for index, cookie in enumerate(self.cookies):
            try:
                CookieToSet.model_validate(cookie)
            except ValidationError as e:
                problems.append(f"step {self.id!r}: cookies[{index}]: " + "; ".join(
                    f"{'.'.join(map(str, error['loc'])) or 'it'}: {error['msg']}"
                    for error in e.errors(include_input=False, include_url=False)))
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "cookies": self.cookies}

    @classmethod
    def from_dict(cls, data: dict) -> BrowserSetCookiesTaskStep:
        return cls(**cls._browser_from_dict(data), cookies=data["cookies"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        await self._call(context, "cookies/set", {"cookies": self.cookies})
        logger.info(f"{self.id}: set {len(self.cookies)} cookie(s) in the browser of env {self.env_id!r}")
        return context


class BrowserDeleteCookiesTaskStep(BrowserStep):
    """Delete every cookie in the browser matching all of ``name``, ``domain`` (exactly as the cookie stores it, a
    domain cookie's with its leading dot) and ``path`` given, or every cookie when none is; deleting none is fine,
    and the count is recorded."""

    type: ClassVar[str] = "browser_delete_cookies"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, name: Optional[str] = None,
                 domain: Optional[str] = None, path: Optional[str] = None, browser: Optional[str] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        super().__init__(id, version, env_id, browser, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.name, self.domain, self.path = name, domain, path

    def config_problems(self) -> list[str]:
        problems = super().config_problems()
        for field in ("name", "domain", "path"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                problems.append(f"step {self.id!r}: {field}, when given, must be a non-empty string")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "name": self.name, "domain": self.domain, "path": self.path}

    @classmethod
    def from_dict(cls, data: dict) -> BrowserDeleteCookiesTaskStep:
        return cls(**cls._browser_from_dict(data), name=data.get("name"), domain=data.get("domain"),
                   path=data.get("path"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        matching = {field: value for field in ("name", "domain", "path") if (value := getattr(self, field)) is not None}
        deleted = (await self._call(context, "cookies/delete", matching))["deleted"]
        logger.info(f"{self.id}: deleted {deleted} cookie(s) matching {matching or 'anything'} in the browser of env "
                    f"{self.env_id!r}")
        self._record(context, deleted=deleted)
        return context
