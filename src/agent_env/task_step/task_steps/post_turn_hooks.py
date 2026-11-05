"""prompt_agent's post-turn hooks: task-referenced Python run after each agent turn, trusted as a verifier script is."""
from __future__ import annotations

import asyncio
import copy
import functools
import importlib.util
import inspect
import os
import sys
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

import httpx
from a2a.types import Part
from agentenv_protocol import client as protocol_v1

from agent_env.artifact import FileArtifact
from agent_env.env.env import DeployedEnv, EnvCapabilityUnsupported
from agent_env.env.gateway.constants import (
    AGENT_ENV_ROLE_HEADER,
    EXT_CLOCK_URI,
    EXT_NOTIFICATIONS_URI,
    EXT_STEP_URI,
    EXT_TRIGGERS_URI,
    STEP_SOURCE,
    STEP_SOURCE_POST_TURN_HOOK,
)
from agent_env.task_step.context import TaskStepContext, _strip_redacted_keys

_SPEC_KEYS = ("file_artifact_id", "file_artifact_version", "params")
_RECORD_TAGS = ("hook", "turn")
_INVOKABLE = (EXT_CLOCK_URI, EXT_TRIGGERS_URI, EXT_NOTIFICATIONS_URI)
_TOOL_CALL_TIMEOUT_S = 600

Answer = dict[str, Any]


def validate_hook_specs(specs: Any) -> None:
    """Refuse a ``post_turn_hooks`` value that names no hook artifact the way the step loads one."""
    if not isinstance(specs, list):
        raise ValueError("post_turn_hooks must be a list of {file_artifact_id, file_artifact_version?, params?}")
    for i, spec in enumerate(specs):
        where = f"post_turn_hooks[{i}]"
        if not isinstance(spec, dict):
            raise ValueError(f"{where} must be an object, got {type(spec).__name__}")
        unknown = sorted(set(spec) - set(_SPEC_KEYS))
        if unknown:
            raise ValueError(f"{where} has unknown key(s) {unknown}; expected {list(_SPEC_KEYS)}")
        if not isinstance(spec.get("file_artifact_id"), str) or not spec["file_artifact_id"]:
            raise ValueError(f"{where}.file_artifact_id must be a non-empty string")
        version = spec.get("file_artifact_version")
        if version is not None and (isinstance(version, bool) or not isinstance(version, int) or version < 1):
            raise ValueError(f"{where}.file_artifact_version must be a positive integer, got {version!r}")
        if spec.get("params") is not None and not isinstance(spec["params"], dict):
            raise ValueError(f"{where}.params must be an object")


@dataclass(frozen=True)
class PostTurnHook:
    artifact_id: str
    version: Optional[int]
    params: dict
    post_turn: Callable[..., Awaitable[Optional[dict]]]


class PostTurnHooks:
    """The hooks of one step run, each loaded once; ``close`` removes their scripts' temporary files."""

    def __init__(self) -> None:
        self.hooks: list[PostTurnHook] = []
        self._paths: list[str] = []
        self._modules: list[str] = []

    @classmethod
    async def load(cls, specs: list[dict], step_id: str) -> PostTurnHooks:
        loaded = cls()
        try:
            for index, spec in enumerate(specs):
                loaded.hooks.append(await loaded._load(spec, f"post_turn_hook_{step_id}_{index}"))
        except BaseException:
            loaded.close()
            raise
        return loaded

    async def _load(self, spec: dict, module_name: str) -> PostTurnHook:
        artifact = FileArtifact.get(spec["file_artifact_id"], spec.get("file_artifact_version"))
        source = await asyncio.to_thread(artifact.load)
        fd, path = tempfile.mkstemp(suffix=".py")
        self._paths.append(path)
        os.write(fd, source)
        os.close(fd)
        module_name = f"{module_name}_{uuid.uuid4().hex[:8]}"  # concurrent runs may share a step id
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(module_spec)
        # registered while it runs, as an import would be: dataclasses and typing look a module up by name
        sys.modules[module_name] = module
        self._modules.append(module_name)
        module_spec.loader.exec_module(module)
        post_turn = getattr(module, "post_turn", None)
        if not inspect.iscoroutinefunction(post_turn):
            raise ValueError(f"post-turn hook '{spec['file_artifact_id']}' must define "
                             "`async def post_turn(turn, params)`: steps and runs share one event loop")
        return PostTurnHook(spec["file_artifact_id"], spec.get("file_artifact_version"), spec.get("params") or {},
                            post_turn)

    def close(self) -> None:
        for path in self._paths:
            if os.path.exists(path):
                os.unlink(path)
        self._paths.clear()
        for name in self._modules:
            sys.modules.pop(name, None)
        self._modules.clear()


class HookEnv:
    """A deployed env as a hook reaches it: clock, triggers and notifications calls, and tool calls recorded as the
    hook's."""

    def __init__(self, deployed: DeployedEnv, hook_artifact_id: str) -> None:
        self.env_id = deployed.env_id
        self.env_step_id = (deployed.metadata or {}).get("deploy_step_id")
        self.instance_id = deployed.instance_id
        self._deployed = deployed
        self._hook_artifact_id = hook_artifact_id

    def supports(self, extension: str, method: str) -> bool:
        return self._deployed.supports(extension, method)

    async def invoke(self, extension: str, method: str, params: Optional[dict] = None) -> Any:
        if extension not in _INVOKABLE:
            raise ValueError(f"a post-turn hook invokes {list(_INVOKABLE)} only; call tools with call_tool")
        return await self._deployed.invoke(extension, method, params)

    async def call_tool(self, tool: str, args: Optional[dict], role: str) -> dict:
        """Call ``tool`` as ``role`` through /step, so role rules, triggers watching the role and barriers apply."""
        if not isinstance(role, str) or not role:
            raise ValueError("call_tool needs the AgentEnv-Role to call as")
        card = self._deployed.environment_card or {}
        op = protocol_v1.find_extension_method(card, EXT_STEP_URI, "step")
        if op is None or not self._deployed.environment_url:
            raise EnvCapabilityUnsupported(EXT_STEP_URI, "step", self.env_id, self.instance_id)
        if STEP_SOURCE not in (protocol_v1.extension_params(card, EXT_STEP_URI).get("features") or []):
            raise RuntimeError(f"env '{self.env_id}' can't record a post-turn hook's tool calls as the hook's; "
                               "rebuild its gateway image")
        body = {"action": "call_tool", "tool_name": tool, "arguments": args or {},
                "source": STEP_SOURCE_POST_TURN_HOOK, "hook_artifact_id": self._hook_artifact_id}
        async with httpx.AsyncClient() as client:
            resp = await client.request(op["method"], f"{self._deployed.environment_url}{op['endpoint']}", json=body,
                                        headers={AGENT_ENV_ROLE_HEADER: role, **self._deployed.control_headers},
                                        timeout=_TOOL_CALL_TIMEOUT_S)
        if resp.status_code >= 400:
            raise RuntimeError(f"env '{self.env_id}': /step call_tool {tool!r} failed (HTTP {resp.status_code}): "
                               f"{resp.text[:300]}")
        return resp.json()


class _Answer:
    """The user side's answer this turn: asked at most once, then the same for every later call."""

    def __init__(self, ask_user: Callable[[], Awaitable[Answer]],
                 user_triggers: Callable[[], Awaitable[Optional[Answer]]]) -> None:
        self.value: Optional[Answer] = None
        self._ask_user = ask_user
        self._user_triggers = user_triggers
        self._asking = asyncio.Lock()

    async def ask_user(self) -> Answer:
        async with self._asking:
            if self.value is None:
                self.value = await self._ask_user()
            return self.value

    async def user_triggers(self) -> Optional[Answer]:
        async with self._asking:
            if self.value is None:
                self.value = await self._user_triggers()
            return self.value


class Turn:
    """What a hook is handed after an agent turn; ``last`` after the final one, when a returned message is ignored."""

    def __init__(self, *, context: TaskStepContext, step_id: str, hook: PostTurnHook, answer: _Answer, number: int,
                 last: bool, agent_reply: str, agent_parts: list[dict]) -> None:
        self.number = number
        self.last = last
        self.agent_reply = agent_reply
        self.agent_parts = copy.deepcopy(agent_parts)
        self.seed = copy.deepcopy(context.metadata.get("seed", {}))
        self.envs = [HookEnv(d, hook.artifact_id) for d in context.deployed_envs]
        self._context = context
        self._step_id = step_id
        self._hook = hook
        self._answer = answer

    @functools.cached_property
    def metadata(self) -> Mapping[str, Any]:
        """A read-only copy of the run's metadata, its secret keys stripped."""
        return MappingProxyType(_strip_redacted_keys(self._context.metadata))

    async def ask_user(self) -> Answer:
        """Today's user side: agent triggers, then a human or the LLM user sim, which sees only replies sent here."""
        return await self._answer.ask_user()

    async def user_triggers(self) -> Optional[Answer]:
        """Agent triggers alone; None, leaving no trace, when none fires."""
        return await self._answer.user_triggers()

    def record(self, entry: dict) -> None:
        """Append ``entry`` to ``metadata["post_turn_hooks"][step_id]``, tagged with this hook and turn."""
        if not isinstance(entry, dict) or set(entry) & set(_RECORD_TAGS):
            raise ValueError(f"a hook records an object, without the keys {list(_RECORD_TAGS)} it is tagged with")
        (self._context.metadata.setdefault("post_turn_hooks", {}).setdefault(self._step_id, [])
            .append({"hook": self._hook.artifact_id, "turn": self.number, **copy.deepcopy(entry)}))

    def records(self) -> list[dict]:
        """This hook's own records in this conversation, oldest first: what ``record`` appended for this step, as
        read-only copies (another step's conversation, or another hook in this one, keeps its own)."""
        entries = (self._context.metadata.get("post_turn_hooks") or {}).get(self._step_id, [])
        return [copy.deepcopy(e) for e in entries if e.get("hook") == self._hook.artifact_id]

    def heartbeat(self) -> None:
        self._context.heartbeat()


def _decision(value: Any, hook: PostTurnHook) -> Optional[Answer]:
    """A hook's return as an answer: ``{"message": parts}`` or ``{"end": reason}`` with an optional closing message.
    ``{"message": []}`` says nothing: the agent's next message is then only what came due for it on its envs'
    notifications/v1, and the conversation ends (``nothing_to_send``) when nothing has."""
    if value is None:
        return None
    shape = "None, {'message': text or parts ([] for none)}, or {'end': reason} with an optional closing 'message'"
    if not isinstance(value, dict) or not value or set(value) - {"message", "end"}:
        raise RuntimeError(f"post-turn hook '{hook.artifact_id}' returned {value!r}; expected {shape}")
    if "end" in value and (not isinstance(value["end"], str) or not value["end"]):
        raise RuntimeError(f"post-turn hook '{hook.artifact_id}' ended with {value['end']!r}; the reason is a string")
    answer: Answer = {"end": value["end"]} if "end" in value else {}
    if "message" in value:
        parts = _parts(value["message"], hook)
        if parts or "end" not in answer:  # an empty closing message records nothing
            answer["message"] = parts
    return answer


def _parts(message: Any, hook: PostTurnHook) -> list[dict]:
    if isinstance(message, str):
        return [{"kind": "text", "text": message}]
    if not isinstance(message, list):
        raise RuntimeError(f"post-turn hook '{hook.artifact_id}': a message is text or a list of A2A parts")
    for i, part in enumerate(message):
        try:
            Part.model_validate(part)
        except Exception as e:
            raise RuntimeError(f"post-turn hook '{hook.artifact_id}': message part {i} is not an A2A Part: {e}") from e
    return copy.deepcopy(message)


async def run_post_turn_hooks(hooks: PostTurnHooks, *, context: TaskStepContext, step_id: str, number: int, last: bool,
                              agent_reply: str, agent_parts: list[dict],
                              ask_user: Callable[[], Awaitable[Answer]],
                              user_triggers: Callable[[], Awaitable[Optional[Answer]]]) -> Optional[Answer]:
    """Run the hooks in order; the first to return a message or an end decides, and one that asked returns that."""
    answer = _Answer(ask_user, user_triggers)
    for hook in hooks.hooks:
        turn = Turn(context=context, step_id=step_id, hook=hook, answer=answer, number=number, last=last,
                    agent_reply=agent_reply, agent_parts=agent_parts)
        returned = await hook.post_turn(turn, copy.deepcopy(hook.params))
        if answer.value is not None:
            if returned != answer.value:
                raise RuntimeError(f"post-turn hook '{hook.artifact_id}' asked the user side at turn {number} and "
                                   "must return its answer")
            return answer.value
        decision = _decision(returned, hook)
        if decision is not None:
            return decision
    return None
