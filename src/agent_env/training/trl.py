"""TRL's trainers on a bundle served as an OpenEnv env (``agent_env.openenv.build_app``, or the package ``openenv import``
writes): ``AgentEnv`` reads the server's tasks into a dataset, and makes the ``environment_factory`` whose environments
each play one episode of a task over OpenEnv's WebSocket, offer its turn's tools as methods and own its reward.
``GRPOTrainer`` takes both as they are, verified with TRL 1.16, and AsyncGRPO takes the same ``environment_factory``;
either needs TRL 1.15 or later, for an environment that owns its reward. Every session runs on one event-loop thread of
this module's own, since TRL calls an environment synchronously, from its own threads."""

from __future__ import annotations

import asyncio
import contextlib
import copyreg
import json
import logging
import re
import threading
import time
import weakref
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

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
from agent_env.openenv.environment import END_TURN as END_TURN_TOOL
from agent_env.openenv.environment import TIMEOUT_METADATA, Toolset
from agent_env.task.teardown import TERMINATE_TIMEOUT_SECONDS
from agent_env.training._tools import tool_method

logger = logging.getLogger(__name__)

FAILED = "The environment failed to answer, so this episode won't be scored."
ENDED = "The episode has ended."
END_TURN = END_TURN_TOOL.name
CONNECT_SECONDS = 60.0
CLOSE_SECONDS = 300.0
#: How much longer than a tool's deadline the client waits for the server to end the call, before dropping the session.
GRACE_SECONDS = 30.0
#: How long a server ``AgentEnv.local`` started is given to stop, at ``close`` and at exit: its episodes' teardowns run
#: at once, and each saves the agents' changelogs, ends open recordings and terminates the sandboxes, every part within
#: ``TERMINATE_TIMEOUT_SECONDS``.
STOP_SECONDS = 3 * TERMINATE_TIMEOUT_SECONDS + 60.0
#: How long the exit hook waits for the sessions to close.
EXIT_SECONDS = 30.0
#: How often, and how far apart in seconds, an environment tries to open a session in place of a dropped one while the
#: server is at its limit, which it stays at until it has seen the dropped one close.
REOPEN_ATTEMPTS, REOPEN_SECONDS = 5, 1.0

_CAPACITY = re.compile(r"Server at capacity: \d+/(\d+)")

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()
_environments: weakref.WeakSet[_Environment] = weakref.WeakSet()
_servers: set[_Server] = set()


class AgentEnv:
    """A server's tasks for TRL. ``dataset`` holds one row per task of ``split``, and ``factory`` is the
    ``environment_factory`` whose environments play them: each holds one session on the server, ``reset`` starts the
    row's task, its methods call the turn's tools, ``end_turn`` hands over the reply and answers with the next turn's
    prompt, and ``get_reward`` is the episode's reward, ``None`` when it ended in an error. TRL hands an environment
    no completion, so the model's reply reaches the task only through ``end_turn(response=...)``: a turn the model
    leaves open ends with an empty reply, which a verifier judging the reply scores as such. Tasks whose agents get
    different tools take TRL's multi-environment mode: ``factory`` is then a dict, and each row names its
    ``environment``.

    Args:
        url: The server's base URL.
        split: The split ``dataset`` holds.
        tool_timeout: One tool call's deadline in seconds. The server ends a call that misses it, and the episode with
            it, since what the call does may still be under way: the rollout is left unscored, and the session kept.
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
        """``split``'s tasks as TRL's rows: the ``prompt`` as chat messages, the ``split`` and ``index`` that
        ``reset`` plays, the ``task_id`` it checks the episode against, and the ``environment`` in multi-environment
        mode. The prompt is the model's to change."""
        several = len(self._toolsets) > 1
        return Dataset.from_list([{
            "prompt": _messages(task["system_prompt"], task["prompt"]), "split": split, "index": task["index"],
            "task_id": task["id"],
            **({"environment": self._toolsets[_key(task)].name} if several else {}),
        } for task in self._split(split)])

    @cached_property
    def factory(self) -> type[_Environment] | dict[str, type[_Environment]]:
        """TRL's ``environment_factory``: an environment class per toolset of the server's tasks, or the one class when
        every task has the same. Building it walks one task of each toolset through its turns, in turn, ending each but
        the last empty and listing its tools, then ends the episode unscored, since TRL reads the tools from the
        class: its methods are every tool a turn offers, and one a turn doesn't offer answers so. Each call on a
        probe's session has the reset deadline."""
        classes = _on_loop(self._environments(), None)
        return classes[0] if len(classes) == 1 else {each.__name__: each for each in classes}

    def close(self) -> None:
        """Ends the episodes this env's environments hold, closing their sessions, and stops the server ``local``
        started."""
        mine = [environment for environment in list(_environments) if environment._shared in self._shared]
        try:
            _on_loop(_gathered(environment._close() for environment in mine), CLOSE_SECONDS)
        finally:
            if self._server is not None:
                self._server.stop()

    async def _environments(self) -> list[type[_Environment]]:
        return [await self._environment(toolset) for toolset in self._toolsets.values()]

    async def _environment(self, toolset: _Toolset) -> type[_Environment]:
        split, index, turns = toolset.first
        client = await _opened(self.url, self._reset_timeout)
        first, tools = None, {}
        try:
            turn = await client.reset(split=split, index=index)
            for number in range(turns):
                if turn.done:
                    break
                listed = await _list_tools(client)
                if listed.done:
                    turn = listed
                    break
                first = first or frozenset(tool["name"] for tool in listed.observation["tools"])
                tools |= {tool["name"]: tool for tool in listed.observation["tools"] if tool["name"] not in tools}
                if number + 1 < turns:
                    turn = await _end_turn(client, "")
                    if error := turn.observation.get("error"):
                        raise RuntimeError(f"task {index} of split {split!r} refused an empty {END_TURN}: "
                                           f"{error.get('message')}")
        finally:
            with contextlib.suppress(Exception):
                await client.close()
        if first is None:
            raise RuntimeError(f"task {index} of split {split!r} ended before its tools could be listed: "
                               f"{_error_of(turn.observation)}")
        shared = _Shared(url=self.url, offered=first, tool_timeout=self._tool_timeout,
                         reset_timeout=self._reset_timeout)
        self._shared.append(shared)
        return _environment_class(toolset.name, tuple(tools.values()), shared)

    def _split(self, split: str) -> list[dict[str, Any]]:
        if split not in self._tasks:
            raise ValueError(f"the server at {self.url} has no split {split!r}; it serves {', '.join(self._tasks)}")
        return self._tasks[split]


@dataclass(frozen=True)
class _Toolset:
    name: str
    first: tuple[str, int, int]  # the split, index and turn count of its first task


@dataclass(frozen=True, eq=False)
class _Shared:
    """What an environment class's instances share: the server, the tools every episode's first turn must offer, and
    the deadlines."""

    url: str
    offered: frozenset[str]
    tool_timeout: float
    reset_timeout: float


class _Failed(Exception):
    pass


class _AtCapacity(RuntimeError):
    def __init__(self, url: str, limit: str):
        super().__init__(f"the server at {url} is at its limit of {limit} sessions at once, and the trainer holds one "
                         "per rollout it generates at once: serve the bundle with MAX_CONCURRENT_ENVS (or "
                         "max_concurrent_envs=) at least that many, or generate fewer rollouts at once")
        self.limit = limit


class _Misgrouped(_Failed):
    """An episode offering other tools than its class has: a toolset that misses what tells tasks apart, which no
    rollout fixes."""


@dataclass
class _Rollout:
    started: asyncio.Task[None] = field(init=False)
    system: str | None = None
    done: bool = False
    reward: float | None = None
    failure: str | None = None
    misgrouped: bool = False


class _Environment:
    """One session on the server, playing one episode at a time. Its public methods other than ``reset`` and
    ``get_reward`` are the tools, which ``AgentEnv.factory`` binds per toolset; every call on the session waits its
    turn, as AsyncGRPO can call two tools of a completion at once."""

    _shared: ClassVar[_Shared]
    _tools: ClassVar[tuple[dict[str, Any], ...]]

    def __init__(self) -> None:
        self._client: GenericEnvClient | None = None
        self._opened = False
        self._rollout: _Rollout | None = None
        self._lock = asyncio.Lock()
        _environments.add(self)

    def reset(self, *, split: str, index: int, task_id: str | None = None, **_: Any) -> None:
        """Starts the row's episode and returns before it is deployed, since ``GRPOTrainer`` resets its environments
        one at a time: the first tool call or ``get_reward`` waits for it. Raises when the server refuses this
        environment's first session, as one at capacity does; a later session it refuses, after one was dropped,
        leaves only that rollout unscored."""
        _on_loop(self._reset(split, index, task_id), CONNECT_SECONDS + CLOSE_SECONDS)

    def get_reward(self) -> float | None:
        """The episode's reward, or ``None`` when it ended in an error. Raises when the episode offered other tools
        than this class has, which no rollout fixes."""
        rollout, self._rollout = self._rollout, None
        if rollout is None:
            return None
        try:
            _on_loop(self._finish(rollout), None)
        except Exception as e:
            rollout.failure = rollout.failure or _why(e)
        if rollout.misgrouped:
            raise RuntimeError(f"{rollout.failure}. The tasks of one toolset must offer the same tools, since TRL shows "
                               "the model a class's: their toolset misses what tells them apart")
        if rollout.failure is not None:
            logger.warning("rollout left unscored: %s", rollout.failure)
            return None
        return rollout.reward

    def _call(self, name: str, arguments: dict[str, Any]) -> str:
        return _on_loop(self._step(name, arguments),
                        self._shared.reset_timeout + self._shared.tool_timeout + GRACE_SECONDS + CLOSE_SECONDS)

    async def _reset(self, split: str, index: int, expected: str | None) -> None:
        async with self._lock:
            if (rollout := self._rollout) is not None:
                rollout.started.cancel()
                await asyncio.wait([rollout.started])
            self._rollout = None
            if not self._opened:
                self._client, self._opened = await _opened(self._shared.url, self._shared.reset_timeout), True
            rollout = _Rollout()
            rollout.started = asyncio.ensure_future(self._begin(rollout, split, index, expected))
            self._rollout = rollout

    async def _begin(self, rollout: _Rollout, split: str, index: int, expected: str | None) -> None:
        if self._client is None:
            self._client = await _reopened(self._shared.url, self._shared.reset_timeout)
        reset = await self._client.reset(split=split, index=index)
        if reset.done:
            raise _Failed(_error_of(reset.observation))
        metadata = reset.observation.get("metadata") or {}
        rollout.system, played = metadata.get("system_prompt") or None, (metadata.get("task") or {}).get("id")
        if expected is not None and played != expected:
            raise _Failed(f"the episode plays task {played!r}, not the row's {expected!r}")
        listed = await _list_tools(self._client)
        if listed.done:
            raise _Failed(_error_of(listed.observation))
        if (offered := {tool["name"] for tool in listed.observation["tools"]}) != self._shared.offered:
            raise _Misgrouped(f"task {played!r} offers the tools {sorted(offered)}, not the "
                              f"{sorted(self._shared.offered)} its toolset's first task does")

    async def _step(self, name: str, arguments: dict[str, Any]) -> str:
        async with self._lock:
            rollout = self._rollout
            if rollout is None or rollout.failure is not None:
                return FAILED
            if rollout.done:
                return ENDED
            try:
                await rollout.started
                deadline = None if name == END_TURN else self._shared.tool_timeout
                result = await asyncio.wait_for(_call_tool(self._client, name, arguments, deadline),
                                                None if deadline is None else deadline + GRACE_SECONDS)
            except _Failed as e:
                rollout.misgrouped = isinstance(e, _Misgrouped)
                await self._fail(rollout, str(e))
                return FAILED
            except TimeoutError:
                await self._fail(rollout, f"{name} didn't answer within {self._shared.tool_timeout:g}s, and the "
                                          f"server didn't end the call in the {GRACE_SECONDS:g}s after")
                return FAILED
            except Exception as e:
                await self._fail(rollout, f"{name}: {_why(e)}")
                return FAILED
            return self._answer(rollout, name, result)

    async def _finish(self, rollout: _Rollout) -> None:
        """Ends the turns the model left open with an empty reply, until the episode is scored; each call on the
        session has the reset's deadline, as ``end_turn`` does, since the server answers it once the next turn opens
        or the verifiers have scored the episode."""
        async with self._lock:
            try:
                await rollout.started
                while not (rollout.done or rollout.failure):
                    result = await _end_turn(self._client, "")
                    if error := result.observation.get("error"):
                        rollout.failure = f"{END_TURN}: {error.get('message')}"
                    self._answer(rollout, END_TURN, result)
            except Exception as e:
                rollout.misgrouped = isinstance(e, _Misgrouped)
                await self._fail(rollout, str(e) if isinstance(e, _Failed) else _why(e))

    def _answer(self, rollout: _Rollout, name: str, result: Any) -> str:
        """What the model sees of a tool call: the tool's text or error, the next turn's prompt for ``end_turn``, after
        its system prompt when that changed, since TRL can't change the conversation's own; or that the episode ended,
        which records its reward or why it has none."""
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
            prompt, system = answer.get("prompt") or "", answer.get("system_prompt") or None
            if system is None or system == rollout.system:
                return prompt
            rollout.system = system
            return f"{system}\n\n{prompt}"
        return answer["data"] if isinstance(answer.get("data"), str) else json.dumps(answer)

    async def _fail(self, rollout: _Rollout, why: str) -> None:
        """Records why the rollout won't be scored and drops the session, which may be out of step with the server; the
        server ends a dropped session's episode, and the next rollout opens another."""
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


class _EnvironmentClass(type):
    """The type of ``AgentEnv.factory``'s classes, which pickle as the toolset and server they're built from, since
    AsyncGRPO builds its environments in a process of its own."""


def _environment_class(name: str, tools: tuple[dict[str, Any], ...], shared: _Shared) -> type[_Environment]:
    methods = {tool["name"]: tool_method(tool) for tool in tools}
    for method in methods.values():
        method.__qualname__ = f"{name}.{method.__name__}"
    return _EnvironmentClass(name, (_Environment,), {**methods, "_shared": shared, "_tools": tools})


def _reduce_environment_class(environment: _EnvironmentClass) -> tuple[Any, ...]:
    return _environment_class, (environment.__name__, environment._tools, environment._shared)


copyreg.pickle(_EnvironmentClass, _reduce_environment_class)


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

    def stop(self, timeout: float = STOP_SECONDS) -> None:
        _servers.discard(self)
        self._uvicorn.should_exit = True
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("the bundle's server at %s was still tearing down its episodes after %gs; their sandboxes "
                           "may be left running", self.url, timeout)


def _episode_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="agentenv-trl-sessions", daemon=True).start()
        return _loop


def _on_loop(coroutine: Coroutine[Any, Any, Any], timeout: float | None) -> Any:
    future = asyncio.run_coroutine_threadsafe(coroutine, _episode_loop())
    try:
        return future.result(timeout)
    except TimeoutError:
        future.cancel()
        raise


async def _gathered(coroutines: Any) -> list[Any]:
    return list(await asyncio.gather(*coroutines))


def _close_all() -> None:
    """A trainer that stops or crashes ends the episodes it left running, since TRL never closes its environments,
    then stops the servers ``AgentEnv.local`` started, which tear down their episodes, waiting at most
    ``EXIT_SECONDS`` for the sessions and ``STOP_SECONDS`` for each server. It runs before threading's shutdown, while
    the thread pools an episode's teardown uses still take work."""
    try:
        _on_loop(_gathered(environment._close() for environment in list(_environments)), EXIT_SECONDS)
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
            raise _AtCapacity(url, match[1]) from e
        raise ConnectionError(f"couldn't open a session on the server at {url}: {_why(e)}") from e
    return client


async def _reopened(url: str, timeout: float) -> GenericEnvClient:
    """A session in place of a dropped one, tried again while the server is at its limit; raises _Failed, which leaves
    the rollout unscored, when it can't open one."""
    for attempt in range(REOPEN_ATTEMPTS):
        try:
            return await _opened(url, timeout)
        except _AtCapacity as e:
            if attempt + 1 == REOPEN_ATTEMPTS:
                raise _Failed(f"its session was dropped, and the server stayed at its limit of {e.limit} sessions for "
                              f"{(REOPEN_ATTEMPTS - 1) * REOPEN_SECONDS:g}s after") from e
            await asyncio.sleep(REOPEN_SECONDS)
        except ConnectionError as e:
            raise _Failed(f"its session was dropped, and {e}") from e


async def _list_tools(client: GenericEnvClient) -> Any:
    return await client.step({"type": "list_tools"})


async def _call_tool(client: GenericEnvClient, name: str, arguments: dict[str, Any],
                     deadline: float | None = None) -> Any:
    """The call, with its ``deadline`` in seconds for the server to end it at."""
    return await client.step({"type": "call_tool", "tool_name": name, "arguments": arguments,
                              **({"metadata": {TIMEOUT_METADATA: deadline}} if deadline is not None else {})})


async def _end_turn(client: GenericEnvClient, response: str) -> Any:
    return await _call_tool(client, END_TURN, {"response": response})


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
    if not all("toolset" in task and "turns" in task for described in tasks.values() for task in described):
        raise ValueError(f"the server at {url} doesn't describe its tasks' toolsets and turns, so it isn't a bundle "
                         "served by agent_env.openenv.build_app from this version of agentenv-framework or later")
    return name, tasks


def _toolsets(name: str, tasks: dict[str, list[dict[str, Any]]]) -> dict[str, _Toolset]:
    """Each toolset the tasks have, with the first task that has it, named after the server when it's the only one
    and otherwise after its envs, and ``bash`` when its turns offer it."""
    firsts: dict[str, tuple[str, int, int]] = {}
    for split, described in tasks.items():
        for task in described:
            firsts.setdefault(_key(task), (split, task["index"], task["turns"]))
    if len(firsts) == 1:
        return {key: _Toolset(name, first) for key, first in firsts.items()}
    named: dict[str, _Toolset] = {}
    for key, first in firsts.items():
        toolset: Toolset = json.loads(key)
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


def _error_of(observation: Mapping[str, Any]) -> str:
    return (observation.get("metadata") or {}).get("error") or "the episode ended in an error"


def _why(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


__all__ = ["AgentEnv"]
