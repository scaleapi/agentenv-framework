"""The toy plugins the installer tier installs: small, but real agent-env plugins."""

from __future__ import annotations

from pathlib import Path

from tst.installer.conftest import build_wheel

GRADER = "agentenv-toy-grader"
BROWSER = "agentenv-toy-browser"
REQUIRES = ("agentenv-framework",)

GRADE_STEP = '''from typing import ClassVar

from agent_env.task_step.task_step import TaskStep


class GradeTaskStep(TaskStep):
    type: ClassVar[str] = "toy_grade"
    entity_refs = ()

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        context.metadata.setdefault("verifications", {})[self.id] = {"score": 1.0, "results": []}
        return context
'''

NAVIGATE_STEP = '''from typing import ClassVar

from agent_env.task_step.task_step import TaskStep


class NavigateTaskStep(TaskStep):
    type: ClassVar[str] = "toy_navigate"
    entity_refs = ()

    def __init__(self, id, version, url="about:blank", **base):
        super().__init__(id, version, **base)
        self.url = url

    def to_dict(self):
        return {**super().to_dict(), "url": self.url}

    @classmethod
    def from_dict(cls, data):
        return cls(url=data.get("url", "about:blank"), **cls._base_from_dict(data))

    async def execute(self, context):
        context.metadata.setdefault("visited", []).append(self.url)
        return context
'''

BROWSER_ENV = '''from agent_env.env.env import Env


class BrowserEnv(Env):
    type = "toy_browser"

    @classmethod
    def from_dict(cls, data):
        return cls(data["id"], data.get("version"))
'''


def grader_wheel(dest: Path, version: str = "0.3.0") -> Path:
    modules = {"agentenv_toy_grader/__init__.py": "", "agentenv_toy_grader/steps.py": GRADE_STEP}
    steps = {"agent_env.task_steps": {"toy_grade": "agentenv_toy_grader.steps:GradeTaskStep"}}
    return build_wheel(dest, GRADER, version, modules, steps, REQUIRES)


def browser_wheel(dest: Path) -> Path:
    modules = {"agentenv_toy_browser/__init__.py": "", "agentenv_toy_browser/env.py": BROWSER_ENV,
               "agentenv_toy_browser/steps.py": NAVIGATE_STEP}
    points = {"agent_env.envs": {"toy_browser": "agentenv_toy_browser.env:BrowserEnv"},
              "agent_env.task_steps": {"toy_navigate": "agentenv_toy_browser.steps:NavigateTaskStep"}}
    return build_wheel(dest, BROWSER, "1.0.0", modules, points, REQUIRES)
