"""A stand-in for the Antigravity SDK, ``google.antigravity``, as much of it as the agent uses, so its tests run offline
without the SDK's wheel: an ``Agent`` whose turns play a script of streamed chunks, tool calls through the hooks the
config names, results and errors, recording the config each turn started with. The names, values and shapes are the
pinned SDK's; tst/integration/agent/antigravity_env_date_test.py runs the real one."""

from __future__ import annotations

import asyncio
import enum
import sys
import types
from dataclasses import dataclass, field
from typing import Any


class _Model:
    """A pydantic-like record: its keyword arguments, and their dump."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def model_dump(self, mode: str | None = None, exclude: set[str] | None = None, warnings: bool = True) -> dict:
        return {key: value for key, value in self.__dict__.items() if key not in (exclude or set())}

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.__dict__ == self.__dict__

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__dict__})"


class BuiltinTools(str, enum.Enum):
    LIST_DIR = "list_directory"
    SEARCH_DIR = "search_directory"
    FIND_FILE = "find_file"
    VIEW_FILE = "view_file"
    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    RUN_COMMAND = "run_command"
    ASK_QUESTION = "ask_question"
    INVOKE_SUBAGENT = "invoke_subagent"
    RUN_WORKFLOW = "run_workflow"
    GENERATE_IMAGE = "generate_image"
    SEARCH_WEB = "search_web"
    READ_URL_CONTENT = "read_url_content"
    SCHEDULE = "schedule"
    FINISH = "finish"


class AgentBehavior(enum.Enum):
    AUTONOMOUS = "autonomous"
    INTERACTIVE = "interactive"
    MINIMAL = "minimal"


class StopReason(enum.Enum):
    UNSPECIFIED = 0
    MAX_MODEL_CALLS_EXCEEDED = 1
    MAX_TOOL_CALLS_EXCEEDED = 2
    QUOTA_EXHAUSTED = 6


class BudgetScope(enum.Enum):
    LIFETIME = 0
    FORWARD_LOOKING = 1


class AntigravityExecutionError(Exception):
    pass


class CapabilitiesConfig(_Model): ...
class McpStreamableHttpServer(_Model): ...
class RetryConfig(_Model): ...
class ModelAPIRetryConfig(_Model): ...
class BudgetConfig(_Model): ...
class LocalOpenAIAgentConfig(_Model): ...
class HookResult(_Model): ...
class Text(_Model): ...
class Thought(_Model): ...
class ToolCall(_Model): ...
class ToolResult(_Model): ...


@dataclass
class Turn:
    """One scripted turn: its steps, each ``("chunk", obj)``, ``("call", hooked ToolCall)``, ``("result", ToolResult)``,
    ``("error", tool error)``, ``("sleep", seconds)`` or ``("raise", exception)``; then the conversation id and stop
    reason the turn ends with."""

    steps: list[tuple[str, Any]] = field(default_factory=list)
    conversation_id: str = "conv-1"
    stop_reason: StopReason = StopReason.UNSPECIFIED


class Agent:
    turns: list[Turn] = []
    configs: list[LocalOpenAIAgentConfig] = []
    prompts: list[Any] = []

    def __init__(self, config: LocalOpenAIAgentConfig) -> None:
        self.config = config
        self.conversation_id: str | None = None

    async def __aenter__(self) -> Agent:
        Agent.configs.append(self.config)
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def chat(self, prompt: Any) -> ChatResponse:
        Agent.prompts.append(prompt)
        return ChatResponse(self, Agent.turns.pop(0))


class ChatResponse:
    def __init__(self, agent: Agent, turn: Turn) -> None:
        self._agent, self._turn = agent, turn
        self.stop_reason = StopReason.UNSPECIFIED

    @property
    def chunks(self):
        return self._play()

    async def _play(self):
        hooks = getattr(self._agent.config, "hooks", None) or []
        for kind, value in self._turn.steps:
            if kind == "chunk":
                # As the SDK does: the conversation has its id, and its store, from the turn's first streamed step.
                self._agent.conversation_id = self._turn.conversation_id
                yield value
            elif kind == "call":
                for hook in hooks:
                    if hook.kind == "pre":
                        await hook(value)
            elif kind == "result":
                for hook in hooks:
                    if hook.kind == "post":
                        await hook(value)
            elif kind == "error":
                for hook in hooks:
                    if hook.kind == "error":
                        await hook(value)
            elif kind == "sleep":
                await asyncio.sleep(value)
            elif kind == "raise":
                raise value
        self._agent.conversation_id = self._turn.conversation_id
        self.stop_reason = self._turn.stop_reason


def _hook(kind: str):
    def decorate(function):
        function.kind = kind
        return function
    return decorate


def install() -> None:
    """Register the stand-in as ``google.antigravity`` and its submodules, beside whatever else lives under ``google``."""
    package = types.ModuleType("google.antigravity")
    sdk_types = types.ModuleType("google.antigravity.types")
    hooks_package = types.ModuleType("google.antigravity.hooks")
    hooks = types.ModuleType("google.antigravity.hooks.hooks")
    policy = types.ModuleType("google.antigravity.hooks.policy")
    for name in ("BuiltinTools", "AgentBehavior", "StopReason", "BudgetScope", "AntigravityExecutionError",
                 "CapabilitiesConfig", "McpStreamableHttpServer", "RetryConfig", "ModelAPIRetryConfig", "BudgetConfig",
                 "HookResult", "Text", "Thought", "ToolCall", "ToolResult"):
        setattr(sdk_types, name, globals()[name])
    package.Agent, package.LocalOpenAIAgentConfig, package.CapabilitiesConfig = (
        Agent, LocalOpenAIAgentConfig, CapabilitiesConfig)
    package.types, package.hooks = sdk_types, hooks_package
    hooks.pre_tool_call_decide, hooks.post_tool_call, hooks.on_tool_error = _hook("pre"), _hook("post"), _hook("error")
    policy.allow_all = lambda: "allow_all"
    hooks_package.hooks, hooks_package.policy = hooks, policy
    sys.modules.update({"google.antigravity": package, "google.antigravity.types": sdk_types,
                        "google.antigravity.hooks": hooks_package, "google.antigravity.hooks.hooks": hooks,
                        "google.antigravity.hooks.policy": policy})
