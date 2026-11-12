"""BrowserStep: a task step on the website browser in a deployed env, found among the env's children by its card, or
among those of a multi env's computers by each computer's own record, and called through browser/v1 with the control
token of the record that serves it."""

from __future__ import annotations

from typing import Any, Optional

from agentenv_protocol.client import find_extension, find_extension_method, invoke_extension
from agentenv_protocol.website_browser import BROWSER

from agent_env.env.env import DeployedEnv, Env
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.preflight import stored_env
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

#: How long a browser/v1 operation may take to answer.
CALL_TIMEOUT_S = 60
#: Where the steps record what each found or did, by step id.
RECORDS_KEY = "browser_steps"


class BrowserStep(TaskStep):
    """A step on the website browser in the env ``env_id``: the one ``browser`` names, else the env's only one.
    Preflight refuses an env that has no browser, or none of that name, or several when it names none."""

    def __init__(self, id: str, version: Optional[int], env_id: str, browser: Optional[str] = None,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.browser = browser

    def config_problems(self) -> list[str]:
        problems = []
        if not isinstance(self.env_id, str) or not self.env_id:
            problems.append(f"step {self.id!r}: env_id must be a non-empty string")
        if self.browser is not None and (not isinstance(self.browser, str) or not self.browser):
            problems.append(f"step {self.id!r}: browser, when given, must be a non-empty string")
        return problems

    def preflight(self) -> list[str]:
        problems = super().preflight()
        if not isinstance(self.env_id, str) or not self.env_id:
            return problems
        env = stored_env(self.env_id)
        if env is None:
            return problems
        names = stored_browsers(env)
        if not names:
            problems.append(f"step {self.id!r}: env {self.env_id!r} has no website browser: browser steps act on a "
                            "computer env's browser app, or a browser deployed alone")
        elif self.browser is not None and self.browser not in names:
            problems.append(f"step {self.id!r}: env {self.env_id!r} has no website browser named {self.browser!r} "
                            f"(it has {names})")
        elif self.browser is None and len(names) > 1:
            problems.append(f"step {self.id!r}: env {self.env_id!r} has {len(names)} website browsers, {names}: "
                            "name one as browser")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "env_id": self.env_id, "browser": self.browser}

    @classmethod
    def _browser_from_dict(cls, data: dict) -> dict:
        return {**cls._base_from_dict(data), "env_id": data["env_id"], "browser": data.get("browser")}

    def _find(self, context: TaskStepContext) -> tuple[DeployedEnv, Optional[str], dict]:
        """The record that serves the browser, the deployed env's or a multi env's computer's, and the browser's name,
        None for that record's env itself, and card."""
        return find_browser(deployed_env(context, self.env_id), self.browser)

    async def _call(self, context: TaskStepContext, op: str, params: dict,
                    timeout_s: float = CALL_TIMEOUT_S) -> Any:
        """``op``'s answer from the browser, refused when its card serves no such operation."""
        deployed, name, card = self._find(context)
        if find_extension_method(card, BROWSER, op) is None:
            raise RuntimeError(f"the website browser {name or deployed.env_id!r} in env {deployed.env_id!r} serves no "
                               f"browser/v1 {op}: rebuild it on a newer agentenv-framework-protocol")
        return await invoke_extension(deployed.environment_url, card, BROWSER, params, timeout_s, method=op,
                                      headers=deployed.control_headers)

    def _record(self, context: TaskStepContext, **fields: Any) -> None:
        context.metadata.setdefault(RECORDS_KEY, {})[self.id] = {"env_id": self.env_id, "browser": self.browser,
                                                                 **fields}


def stored_browsers(env: Env) -> list[str]:
    """The names of the website browsers a stored env deploys: a computer env's browser apps, the app itself for one
    deployed alone, a multi env's computers' browser apps and its browser computer, the one its browser_envs names or
    the run's; none for any other env."""
    if isinstance(env, MultiEnv):
        computers = [*env.computer_envs, *([browser] if (browser := env.website_browser().computer) else [])]
        return [name for computer in computers for name in stored_browsers(computer)]
    if not isinstance(env, ComputerEnv):
        return []
    return [app.environment_name for app in env.composed_apps() if app.declaration.browser]


def deployed_env(context: TaskStepContext, env_id: str) -> DeployedEnv:
    deployed = next((d for d in context.deployed_envs if d.env_id == env_id), None)
    if deployed is None:
        raise RuntimeError(f"env {env_id!r} isn't in context.deployed_envs")
    return deployed


def find_browser(deployed: DeployedEnv, named: Optional[str]) -> tuple[DeployedEnv, Optional[str], dict]:
    """The website browser in ``deployed``, the one ``named`` else its only one: the record that serves it, its name
    among that record's children, or None for the record's env itself, and its card. A multi env's gateway lists its
    computers but not their apps, so a browser in one is found by the computer's own record, kept in the multi env's
    (``DeployedGatewayEnv.child_envs``), whose control token reaches it."""
    browsers = _browsers(deployed)
    for data in (getattr(deployed, "child_envs", None) or {}).values():
        browsers += _browsers(DeployedEnv.from_dict(data))
    if named is not None:
        found = next((browser for browser in browsers if browser[1] == named), None)
        if found is None:
            raise RuntimeError(f"env {deployed.env_id!r} has no website browser named {named!r}")
        return found
    if len(browsers) != 1:
        raise RuntimeError(f"env {deployed.env_id!r} has {len(browsers) or 'no'} website browsers"
                           + (": name one as browser" if browsers else ""))
    return browsers[0]


def _browsers(record: DeployedEnv) -> list[tuple[DeployedEnv, Optional[str], dict]]:
    """The website browsers ``record``'s card offers: its env itself, and each of its children that is one."""
    card = record.environment_card or {}
    found = [(record, None, card)] if find_extension(card, BROWSER) else []
    return found + [(record, child.get("name"), child) for child in card.get("children_environments") or []
                    if find_extension(child, BROWSER)]
