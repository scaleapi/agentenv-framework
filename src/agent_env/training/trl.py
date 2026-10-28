"""TRL's trainers on a bundle served as an OpenEnv env (``agent_env.openenv.build_app``, or the package ``openenv import``
writes): ``AgentEnv`` reads the server's tasks into a dataset, and makes the ``environment_factory`` whose environments
each play one episode of a task over OpenEnv's WebSocket, offer its turn's tools as methods and own its reward.
``GRPOTrainer`` and AsyncGRPO take both as they are. Every session runs on one event-loop thread of this module's own,
since TRL calls an environment synchronously, from its own threads."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import keyword
import logging
import operator
import re
import threading
import time
import weakref
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from functools import cached_property, reduce
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx
import uvicorn

try:
    from datasets import Dataset
    from openenv.core.generic_client import GenericEnvClient
except ModuleNotFoundError as exc:
    if exc.name not in ("datasets", "openenv"):
        raise
    raise ImportError("agent_env.training.trl requires pip install 'agentenv-framework[trl]'") from exc

from agent_env.openenv import build_app

logger = logging.getLogger(__name__)

FAILED = "The environment failed to answer, so this episode won't be scored."
ENDED = "The episode has ended."
END_TURN = "end_turn"
CONNECT_SECONDS = 60.0
CLOSE_SECONDS = 300.0

_CAPACITY = re.compile(r"Server at capacity: \d+/(\d+)")
_TYPES: dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool, "object": dict,
                          "null": type(None)}
_RESERVED = {"reset", "get_reward"}

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()
_environments: weakref.WeakSet[_Environment] = weakref.WeakSet()
_servers: set[_Server] = set()


class AgentEnv:
    """A server's tasks for TRL. ``dataset`` holds one row per task of ``split``, and ``factory`` is the
    ``environment_factory`` whose environments play them: each holds one session on the server, ``reset`` starts the
    row's task, its methods call the turn's tools, ``end_turn`` hands over the reply and answers with the next turn's
    prompt, and ``get_reward`` is the episode's reward, ``None`` when it ended in an error. A task's turn not ended by
    the model ends with an empty reply. Tasks whose agents get different tools take TRL's multi-environment mode:
    ``factory`` is then a dict, and each row names its ``environment``.

    Args:
        url: The server's base URL.
        split: The split ``dataset`` holds.
        tool_timeout: One tool call's deadline in seconds. A call that misses it ends its episode unscored.
        reset_timeout: How long an episode may take to deploy and open its first turn, and each turn to be ended,
            in seconds.

    Example:

    ```python
    env = AgentEnv("http://localhost:8000")
    trainer = GRPOTrainer(
        model="Qwen/Qwen3-4B",
        args=GRPOConfig(max_tool_calling_iterations=12),
        train_dataset=env.dataset,
        environment_factory=env.factory,
    )
    ```
    """

    def __init__(self, url: str, split: str = "all", *, tool_timeout: float = 60.0, reset_timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.split = split
        self._tool_timeout, self._reset_timeout = tool_timeout, reset_timeout
        self._server: _Server | None = None
        self._shared: list[_Shared] = []
        self._name, self._tasks = _read_tasks(self.url)
        self._toolsets = _toolsets(self._name, self._tasks)
        self._split(split)

    @classmethod
    def local(cls, bundle: str | Path | Mapping[str, str | Path], split: str = "all", *,
              reward: str | Callable[[dict[str, float]], float] | None = None, sandbox: str | None = None,
              max_concurrent_envs: int = 8, tool_timeout: float = 60.0, reset_timeout: float = 600.0) -> AgentEnv:
        """``bundle`` served by ``build_app`` in this process on a loopback port, for a trainer on a machine that can
        deploy its envs. ``max_concurrent_envs`` must cover the rollouts the trainer generates at once. The server
        stops at ``close``, or when the process exits, ending its episodes."""
        server = _Server(build_app(bundle, reward=reward, sandbox=sandbox, max_concurrent_envs=max_concurrent_envs))
        try:
            env = cls(server.url, split, tool_timeout=tool_timeout, reset_timeout=reset_timeout)
        except BaseException:
            server.stop()
            raise
        env._server = server
        return env

    @cached_property
    def dataset(self) -> Dataset:
        return self.dataset_for(self.split)

    def dataset_for(self, split: str) -> Dataset:
        """``split``'s tasks as TRL's rows: the ``prompt`` as chat messages, and the ``split`` and ``index`` that
        ``reset`` plays, with the ``environment`` in multi-environment mode."""
        several = len(self._toolsets) > 1
        return Dataset.from_list([{
            "prompt": _messages(task["system_prompt"], task["prompt"]), "split": split, "index": task["index"],
            **({"environment": self._toolsets[_key(task)].name} if several else {}),
        } for task in self._split(split)])

    @cached_property
    def factory(self) -> type[_Environment] | dict[str, type[_Environment]]:
        """TRL's ``environment_factory``: an environment class per toolset of the server's tasks, or the one class when
        every task has the same. Building it plays one task of each toolset in turn to list its tools, since TRL reads
        them from the class."""
        classes = on_loop(self._environments(), len(self._toolsets) * (self._reset_timeout + CLOSE_SECONDS))
        return classes[0] if len(classes) == 1 else {each.__name__: each for each in classes}

    def close(self) -> None:
        """Ends the episodes this env's environments hold, closing their sessions, and stops the server ``local``
        started."""
        mine = [environment for environment in list(_environments) if environment._shared in self._shared]
        try:
            on_loop(_gathered(environment._close() for environment in mine), CLOSE_SECONDS)
        finally:
            if self._server is not None:
                self._server.stop()

    async def _environments(self) -> list[type[_Environment]]:
        return [await self._environment(toolset) for toolset in self._toolsets.values()]

    async def _environment(self, toolset: _Toolset) -> type[_Environment]:
        split, index = toolset.first
        client = await _opened(self.url, self._reset_timeout)
        try:
            reset = await client.reset(split=split, index=index)
            listed = reset if reset.done else await client.step({"type": "list_tools"})
        finally:
            with contextlib.suppress(Exception):
                await client.close()
        if listed.done:
            raise RuntimeError(f"task {index} of split {split!r} ended before its tools could be listed: "
                               f"{_error_of(listed.observation)}")
        methods = {tool["name"]: tool_method(tool) for tool in listed.observation["tools"]}
        shared = _Shared(url=self.url, offered=frozenset(methods), tool_timeout=self._tool_timeout,
                         reset_timeout=self._reset_timeout)
        self._shared.append(shared)
        environment = type(toolset.name, (_Environment,), {**methods, "_shared": shared})
        for method in methods.values():
            method.__qualname__ = f"{toolset.name}.{method.__name__}"
        return environment

    def _split(self, split: str) -> list[dict[str, Any]]:
        if split not in self._tasks:
            raise ValueError(f"the server at {self.url} has no split {split!r}; it serves {', '.join(self._tasks)}")
        return self._tasks[split]


@dataclass(frozen=True)
class _Toolset:
    name: str
    first: tuple[str, int]


@dataclass(frozen=True, eq=False)
class _Shared:
    """What an environment class's instances share: the server, the tools every episode must offer, and the
    deadlines."""

    url: str
    offered: frozenset[str]
    tool_timeout: float
    reset_timeout: float


class _Failed(Exception):
    pass


@dataclass
class _Rollout:
    started: asyncio.Task[None]
    done: bool = False
    reward: float | None = None
    failure: str | None = None


class _Environment:
    """One session on the server, playing one episode at a time. Its public methods other than ``reset`` and
    ``get_reward`` are the tools, which ``AgentEnv.factory`` binds per toolset; every call on the session waits its
    turn, as AsyncGRPO can call two tools of a completion at once."""

    _shared: ClassVar[_Shared]

    def __init__(self) -> None:
        self._client: GenericEnvClient | None = None
        self._rollout: _Rollout | None = None
        self._lock = asyncio.Lock()
        _environments.add(self)

    def reset(self, *, split: str, index: int, prompt: list[dict[str, Any]] | None = None, **_: Any) -> None:
        """Starts the row's episode and returns before it is deployed, since ``GRPOTrainer`` resets its environments
        one at a time: the first tool call or ``get_reward`` waits for it. Raises when the server refuses the session,
        as one at capacity does."""
        on_loop(self._reset(split, index, _prompt_of(prompt)), CONNECT_SECONDS + CLOSE_SECONDS)

    def get_reward(self) -> float | None:
        rollout, self._rollout = self._rollout, None
        if rollout is None:
            return None
        try:
            on_loop(self._finish(rollout), None)
        except Exception as e:
            rollout.failure = rollout.failure or _why(e)
        if rollout.failure is not None:
            logger.warning("rollout left unscored: %s", rollout.failure)
            return None
        return rollout.reward

    def _call(self, name: str, arguments: dict[str, Any]) -> str:
        return on_loop(self._step(name, arguments),
                       self._shared.reset_timeout + self._shared.tool_timeout + CLOSE_SECONDS)

    async def _reset(self, split: str, index: int, expected: tuple[str | None, str] | None) -> None:
        async with self._lock:
            if (rollout := self._rollout) is not None:
                rollout.started.cancel()
                await asyncio.wait([rollout.started])
            self._rollout = None
            if self._client is None:
                self._client = await _opened(self._shared.url, self._shared.reset_timeout)
            self._rollout = _Rollout(asyncio.ensure_future(self._begin(split, index, expected)))

    async def _begin(self, split: str, index: int, expected: tuple[str | None, str] | None) -> None:
        reset = await self._client.reset(split=split, index=index)
        if reset.done:
            raise _Failed(_error_of(reset.observation))
        metadata = reset.observation.get("metadata") or {}
        if expected is not None and (metadata.get("system_prompt") or None, metadata.get("prompt")) != expected:
            raise _Failed("the turn's prompt differs from the dataset row's")
        listed = await self._client.step({"type": "list_tools"})
        if listed.done:
            raise _Failed(_error_of(listed.observation))
        if (offered := {tool["name"] for tool in listed.observation["tools"]}) != self._shared.offered:
            raise _Failed(f"the episode offers the tools {sorted(offered)}, not {sorted(self._shared.offered)}")

    async def _step(self, name: str, arguments: dict[str, Any]) -> str:
        async with self._lock:
            rollout = self._rollout
            if rollout is None or rollout.failure is not None:
                return FAILED
            if rollout.done:
                return ENDED
            try:
                await rollout.started
                result = await asyncio.wait_for(self._client.step(
                    {"type": "call_tool", "tool_name": name, "arguments": arguments}), self._deadline(name))
            except _Failed as e:
                await self._fail(rollout, str(e))
                return FAILED
            except TimeoutError:
                await self._fail(rollout, f"{name} didn't answer within {self._shared.tool_timeout:g}s")
                return FAILED
            except Exception as e:
                await self._fail(rollout, f"{name}: {_why(e)}")
                return FAILED
            return self._answer(rollout, name, result)

    def _deadline(self, name: str) -> float | None:
        """A tool's deadline; ``end_turn`` has the session's, the reset deadline, since the server answers it once the
        task's next turn opens or its verifiers have scored the episode."""
        return None if name == END_TURN else self._shared.tool_timeout

    async def _finish(self, rollout: _Rollout) -> None:
        """Ends the turns the model left open with an empty reply, until the episode is scored; each call on the
        session has the reset's deadline."""
        async with self._lock:
            try:
                await rollout.started
                while not (rollout.done or rollout.failure):
                    result = await self._client.step(
                        {"type": "call_tool", "tool_name": END_TURN, "arguments": {"response": ""}})
                    if error := result.observation.get("error"):
                        rollout.failure = f"{END_TURN}: {error.get('message')}"
                    self._answer(rollout, END_TURN, result)
            except Exception as e:
                await self._fail(rollout, str(e) if isinstance(e, _Failed) else _why(e))

    def _answer(self, rollout: _Rollout, name: str, result: Any) -> str:
        """What the model sees of a tool call: the tool's text or error, the next turn's prompt for ``end_turn``,
        or that the episode ended, which records its reward or why it has none."""
        observation = result.observation
        if error := observation.get("error"):
            return error.get("message") or json.dumps(error)
        if result.done:
            rollout.done = True
            if (observation.get("metadata") or {}).get("status") == "error":
                rollout.failure = _error_of(observation)
                return FAILED
            rollout.reward = result.reward
            return ENDED
        answer = observation.get("result") or {}
        if name == END_TURN:
            return answer.get("prompt") or ""
        return answer["data"] if isinstance(answer.get("data"), str) else json.dumps(answer)

    async def _fail(self, rollout: _Rollout, why: str) -> None:
        """Records why the rollout won't be scored and drops the session, which may be out of step with the server; the
        server ends a dropped session's episode, and the next reset opens another."""
        rollout.failure = rollout.failure or why
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()

    async def _close(self) -> None:
        rollout, self._rollout = self._rollout, None
        if rollout is not None:
            rollout.started.cancel()
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()


class _Server:
    """An app under uvicorn on a thread of its own, its lifespan included, on a free loopback port."""

    def __init__(self, app: Any):
        self._uvicorn = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        self._thread = threading.Thread(target=self._uvicorn.run, name="agentenv-openenv-server", daemon=True)
        self._thread.start()
        while not self._uvicorn.started:
            if not self._thread.is_alive():
                raise RuntimeError("the bundle's server didn't start; its log says why")
            time.sleep(0.05)
        self.url = f"http://127.0.0.1:{self._uvicorn.servers[0].sockets[0].getsockname()[1]}"
        _servers.add(self)

    def stop(self) -> None:
        _servers.discard(self)
        self._uvicorn.should_exit = True
        self._thread.join(CLOSE_SECONDS)


def episode_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="agentenv-trl-sessions", daemon=True).start()
        return _loop


def on_loop(coroutine: Coroutine[Any, Any, Any], timeout: float | None) -> Any:
    future = asyncio.run_coroutine_threadsafe(coroutine, episode_loop())
    try:
        return future.result(timeout)
    except TimeoutError:
        future.cancel()
        raise


async def _gathered(coroutines: Any) -> list[Any]:
    return list(await asyncio.gather(*coroutines))


def _close_all() -> None:
    """A trainer that stops or crashes ends the episodes it left running, since TRL never closes its environments,
    then stops the servers ``AgentEnv.local`` started. It runs before threading's shutdown, while the thread pools an
    episode's teardown uses still take work."""
    try:
        on_loop(_gathered(environment._close() for environment in list(_environments)), CLOSE_SECONDS)
    except Exception as e:
        logger.warning("couldn't end the episodes left running: %s", _why(e))
    for server in list(_servers):
        server.stop()


threading._register_atexit(_close_all)


async def _opened(url: str, timeout: float) -> GenericEnvClient:
    """A session on the server, opened now rather than at the first reset, so a server at capacity refuses it while
    the trainer sets up."""
    client = GenericEnvClient(base_url=url, message_timeout_s=timeout)
    try:
        await client.state()
    except Exception as e:
        with contextlib.suppress(Exception):
            await client.close()
        if match := _CAPACITY.search(str(e)):
            raise RuntimeError(
                f"the server at {url} is at its limit of {match[1]} sessions at once, and the trainer holds one per "
                "rollout it generates at once: serve the bundle with MAX_CONCURRENT_ENVS (or max_concurrent_envs=) "
                "at least that many, or generate fewer rollouts at once") from e
        raise ConnectionError(f"couldn't open a session on the server at {url}: {_why(e)}") from e
    return client


def _read_tasks(url: str) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """The server's env name, and each split's tasks as its Task API describes them."""
    with httpx.Client(base_url=url, timeout=CONNECT_SECONDS) as http:
        try:
            [name] = http.get("/list_environments").raise_for_status().json()
        except httpx.TransportError as e:
            raise ConnectionError(f"no OpenEnv server answers at {url}: serve the bundle first (`uv run server` in the "
                                  "package `openenv import` wrote), or train on AgentEnv.local(bundle)") from e
        splits = [split["name"] for split in http.get(f"/{name}/splits").raise_for_status().json()]
        tasks = {split: http.post(f"/{name}/tasks", json={"split": split}).raise_for_status().json()["tasks"]
                 for split in splits}
    if not all("toolset" in task for described in tasks.values() for task in described):
        raise ValueError(f"the server at {url} doesn't describe its tasks' toolsets, so it isn't a bundle served by "
                         "agent_env.openenv.build_app from this version of agentenv-framework or later")
    return name, tasks


def _toolsets(name: str, tasks: dict[str, list[dict[str, Any]]]) -> dict[str, _Toolset]:
    """Each toolset the tasks have, with the first task that has it, named after the server when it's the only one
    and otherwise after its envs, and ``bash`` when its turns offer it."""
    firsts: dict[str, tuple[str, int]] = {}
    for split, described in tasks.items():
        for task in described:
            firsts.setdefault(_key(task), (split, task["index"]))
    if len(firsts) == 1:
        return {key: _Toolset(name, first) for key, first in firsts.items()}
    named: dict[str, _Toolset] = {}
    for key, first in firsts.items():
        toolset = json.loads(key)
        base = "+".join([*(env.rsplit("/", 1)[-1] for env in toolset["envs"]), *(["bash"] if toolset["bash"] else [])])
        base = base or "agent"
        taken = {toolset.name for toolset in named.values()}
        name, n = base, 1
        while name in taken:
            n += 1
            name = f"{base}-{n}"
        named[key] = _Toolset(name, first)
    return named


def _key(task: Mapping[str, Any]) -> str:
    return json.dumps(task["toolset"], sort_keys=True)


def _messages(system: str | None, prompt: str) -> list[dict[str, str]]:
    return [*([{"role": "system", "content": system}] if system else []), {"role": "user", "content": prompt}]


def _prompt_of(messages: list[dict[str, Any]] | None) -> tuple[str | None, str] | None:
    """The system prompt and prompt a row of ``dataset_for`` carries."""
    if not messages:
        return None
    system = next((message["content"] for message in messages if message["role"] == "system"), None)
    return system or None, messages[-1]["content"]


def _error_of(observation: Mapping[str, Any]) -> str:
    return (observation.get("metadata") or {}).get("error") or "the episode ended in an error"


def _why(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def tool_method(tool: Mapping[str, Any]) -> Callable[..., str]:
    """A method calling ``tool``, typed and documented from its JSON Schema the way transformers' ``get_json_schema``
    reads a function, so TRL shows the model the tool as the env describes it. Built without generating source."""
    name = tool["name"]
    if not _callable_name(name) or name in _RESERVED:
        raise ValueError(f"the env's tool {name!r} can't be a TRL tool: TRL calls an environment's tools as its "
                         "methods, so a tool's name must be a public Python identifier other than reset and "
                         "get_reward. Rename the tool in the env")
    schema = tool.get("input_schema") or {}
    required = set(schema.get("required") or [])
    parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    annotations: dict[str, Any] = {}
    described = []
    for argument, spec in (schema.get("properties") or {}).items():
        if not _callable_name(argument):
            raise ValueError(f"the env's tool {name!r} takes an argument {argument!r}, which TRL can't pass: an "
                             "argument's name must be a public Python identifier. Rename it in the env")
        hint = _hint(spec, f"{name}.{argument}")
        default = inspect.Parameter.empty
        if argument not in required:
            hint, default = hint | None, spec.get("default")
        parameters.append(inspect.Parameter(argument, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=hint))
        annotations[argument] = hint
        described.append(f"    {argument}: {_describe(argument, spec)}")
    annotations["return"] = str

    def method(self: _Environment, **arguments: Any) -> str:
        return self._call(name, arguments)

    method.__name__ = method.__qualname__ = name
    method.__doc__ = (tool.get("description") or f"Call the {name} tool.") + (
        "\n\nArgs:\n" + "\n".join(described) if described else "")
    method.__annotations__ = annotations
    method.__signature__ = inspect.Signature(parameters, return_annotation=str)
    return method


def _callable_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name) and not name.startswith("_")


def _hint(schema: Mapping[str, Any], where: str) -> Any:
    if options := schema.get("anyOf") or schema.get("oneOf"):
        return _union([_hint(option, where) for option in options])
    if "enum" in schema:
        return Literal[tuple(schema["enum"])]
    kind = schema.get("type")
    if isinstance(kind, list):
        return _union([_hint({**schema, "type": each}, where) for each in kind])
    if kind == "array":
        items = schema.get("items")
        return list[_hint(items, where)] if items else list
    if kind is None:
        return Any
    if kind not in _TYPES:
        raise ValueError(f"{where}: JSON Schema type {kind!r} has no Python type TRL can show the model")
    return _TYPES[kind]


def _union(hints: list[Any]) -> Any:
    return reduce(operator.or_, list(dict.fromkeys(hints)))


def _describe(name: str, spec: Mapping[str, Any]) -> str:
    """One line, since transformers ends an argument's description at the next line that looks like another."""
    text = spec.get("description") or spec.get("title") or name
    return " ".join(f"{text} {_fields(spec)}".split())


def _fields(spec: Mapping[str, Any]) -> str:
    """The fields of an object argument, or of an array's entries, which a Python type hint can't carry."""
    for option in [spec, *(spec.get("anyOf") or spec.get("oneOf") or [])]:
        if option.get("type") == "array" and (items := option.get("items") or {}).get("properties"):
            return "Each entry: " + _field_list(items)
        if option.get("type") == "object" and option.get("properties"):
            return "Fields: " + _field_list(option)
    return ""


def _field_list(schema: Mapping[str, Any]) -> str:
    required = set(schema.get("required") or [])
    parts = []
    for name, spec in schema["properties"].items():
        kind = _kind(spec) + (", required" if name in required else "")
        parts.append(f"{name} ({kind})" + (f": {spec['description']}" if spec.get("description") else ""))
    return "; ".join(parts) + "."


def _kind(spec: Mapping[str, Any]) -> str:
    if options := spec.get("anyOf") or spec.get("oneOf"):
        return " or ".join(_kind(option) for option in options)
    kind = spec.get("type", "any")
    return " or ".join(kind) if isinstance(kind, list) else kind


__all__ = ["AgentEnv"]
