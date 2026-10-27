"""A bundle's tasks as an OpenEnv env: each session plays one episode of a task at a time (``EpisodeSpec``), as
OpenEnv's own tool envs do. Reset deploys and opens the first turn, steps list and call the env's tools, and the
``end_turn`` tool hands the reply back. An episode that can't go on ends in an observation, never an exception; a step
before the session's first reset raises, as OpenEnv's imported envs do, so a tool call never starts an episode."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import httpx
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import (
    RESERVED_TOOL_NAMES,
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
    ToolError,
    ToolErrorType,
)
from openenv.core.env_server.types import Observation, State

from agent_env.task import EpisodeError, EpisodeSpec, Task
from agent_env.task.episode import Episode, Turn
from agent_env.task.episode.run import TurnTools

END_TURN = Tool(
    name="end_turn",
    description="End your turn with your reply. The episode goes on to your next turn, or ends and is scored.",
    input_schema={"type": "object", "properties": {"response": {"type": "string", "description": "Your reply."}}},
)
ENDED = "the episode has ended; call reset"
NOT_RESET = "no episode has started; call reset before stepping"


@dataclass(frozen=True)
class ServedTask:
    """A task served as an episode, and its reward as a function of the scores it records, by step id."""

    task: Task
    name: str
    spec: EpisodeSpec
    reward: Callable[[dict[str, float]], float | None]


@dataclass
class Served:
    """What one server serves, shared by every session: each split's tasks, where envs deploy, and the sessions
    holding an episode, which the server's shutdown tears down."""

    splits: dict[str, tuple[ServedTask, ...]]
    sandbox: str | None
    live: set[AgentEnvEnvironment] = field(default_factory=set)
    ending: set[asyncio.Task] = field(default_factory=set)

    async def close(self, reason: str) -> None:
        await asyncio.gather(*(env._abandon_when_free(reason) for env in list(self.live)), *self.ending)

    def end_later(self, ending: Coroutine[Any, Any, None]) -> asyncio.Task:
        task = asyncio.ensure_future(ending)
        self.ending.add(task)
        task.add_done_callback(self.ending.discard)
        return task


class AgentEnvEnvironment(Environment):
    """Reset, step and close take turns: OpenEnv's HTTP MCP sessions and its web playground can call one env
    concurrently."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, served: Served):
        super().__init__()
        self._served = served
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._served_task: ServedTask | None = None
        self._episode: Episode | None = None
        self._turn: Turn | None = None
        self._tools: TurnTools | None = None
        self._stack: AsyncExitStack | None = None
        self._ending: asyncio.Task | None = None
        self._state = State(episode_id=None, step_count=0, done=False)

    # The Task API

    def list_splits(self) -> list[str]:
        return list(self._served.splits)

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return [self._describe(split, index) for index in range(self.num_tasks(split))]

    def num_tasks(self, split: str) -> int:
        return len(self._split(split))

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        return self._describe(split, index)

    def get_task_range(self, split: str, start: int | None = None, stop: int | None = None) -> list[dict[str, Any]]:
        return [self._describe(split, index) for index in range(self.num_tasks(split))[start:stop]]

    # The episode

    async def reset_async(self, seed: int | None = None, episode_id: str | None = None, *, split: str | None = None,
                          index: int | None = None, **kwargs: Any) -> Observation:
        """Plays task ``index`` of ``split``, or without either the first task served, as OpenEnv's ORS and verifiers
        importers do."""
        async with self._lock:
            return await self._reset(split, index)

    async def _reset(self, split: str | None, index: int | None) -> Observation:
        await self._abandon("reset")
        self._loop = asyncio.get_running_loop()
        self._served_task, self._state = None, State(episode_id=None, step_count=0, done=False)
        if (split is None) != (index is None):
            return self._error_observation("split and index go together: give both, or neither for the first task served")
        if split is None:
            split, index = next(name for name, tasks in self._served.splits.items() if tasks), 0
        try:
            self._served_task = self._task_at(split, index)
        except IndexError as e:
            return self._error_observation(str(e))
        self._state = State(episode_id=None, step_count=0, task_id=self._served_task.task.id,
                            task_name=self._served_task.name, split=split, index=index, turn=None, done=False)
        self._served.live.add(self)
        try:
            self._episode = await self._served_task.spec.start(sandbox=self._served.sandbox)
            self._state.episode_id = self._episode.id
            turn = await self._episode.next_turn()
            if turn is None:
                raise EpisodeError("the run ended before the agent's first turn")
            await self._open(turn)
        except Exception as e:
            return await self._end_with_error(e)
        return Observation(done=False, reward=None, metadata={
            "prompt": turn.prompt, "system_prompt": turn.system_prompt, "task": self._task_metadata()})

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> Observation:
        raise NotImplementedError("an AgentEnv episode runs on the server's event loop: reset_async")

    async def step_async(self, action: Any, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        async with self._lock:
            return await self._step(action, timeout_s)

    async def _step(self, action: Any, timeout_s: float | None) -> Observation:
        if self._loop is None:
            raise RuntimeError(NOT_RESET)
        if isinstance(action, ListToolsAction):
            if (stopped := await self._why_stopped()) is not None:
                return ListToolsObservation(tools=[], done=True, reward=None, metadata=stopped)
            return ListToolsObservation(tools=[*map(_tool, self._tools.list()), END_TURN])
        if not isinstance(action, CallToolAction):
            raise TypeError(f"unsupported action {type(action).__name__}; use ListToolsAction or CallToolAction")
        if (stopped := await self._why_stopped()) is not None:
            return CallToolObservation(tool_name=action.tool_name, done=True, reward=None, metadata=stopped)
        self._state.step_count += 1
        if action.tool_name == END_TURN.name:
            return await self._end_turn(action.tool_name, action.arguments.get("response") or "")
        return await self._call(action, timeout_s)

    def step(self, action: Any, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        """OpenEnv's web playground steps from a worker thread; the episode lives on the loop its reset ran on."""
        if self._loop is None:
            raise RuntimeError(NOT_RESET)
        return asyncio.run_coroutine_threadsafe(self.step_async(action, timeout_s), self._loop).result()

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        """OpenEnv closes a session's env from a worker thread, and an HTTP request's from its own event loop, which
        can't wait here; there the teardown runs on, and the server's shutdown waits for it."""
        if self._loop is None or self._loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._served.end_later(self._abandon_when_free("the session closed"))
        else:
            asyncio.run_coroutine_threadsafe(self._abandon_when_free("the session closed"), self._loop).result()

    async def _call(self, action: CallToolAction, timeout_s: float | None) -> CallToolObservation:
        name = action.tool_name
        try:
            result = await asyncio.wait_for(self._tools.call(name, action.arguments), timeout_s)
        except EpisodeError as e:
            return _tool_error(name, ToolErrorType.TOOL_NOT_FOUND, e)
        except TimeoutError as e:
            return _tool_error(name, ToolErrorType.TIMEOUT, e)
        except httpx.HTTPError as e:
            return _tool_error(name, ToolErrorType.TRANSPORT_ERROR, e)
        if result.isError:
            return _tool_error(name, ToolErrorType.EXECUTION_ERROR, result.text)
        return CallToolObservation(tool_name=name, result={
            "content": result.content, "structured_content": None, "meta": None, "data": result.text, "is_error": False})

    async def _end_turn(self, name: str, response: str) -> CallToolObservation:
        try:
            await self._turn.end(response)
            await self._close_tools()
            turn = await self._episode.next_turn()
            if turn is not None:
                await self._open(turn)
                return CallToolObservation(tool_name=name, done=False, reward=None, result={
                    "prompt": turn.prompt, "system_prompt": turn.system_prompt})
            scores = await self._episode.result()
        except Exception as e:
            observation = await self._end_with_error(e)
            return CallToolObservation(tool_name=name, done=True, reward=None, metadata=observation.metadata)
        await self._abandon("the episode ended")
        return CallToolObservation(tool_name=name, done=True, **self._scored(scores))

    def _scored(self, scores: dict[str, float]) -> dict[str, Any]:
        self._state.done = True
        metadata: dict[str, Any] = {"scores": scores, "task": self._task_metadata()}
        try:
            reward = self._served_task.reward(scores)
        except Exception as e:
            reward, metadata["status"], metadata["error"] = None, "error", f"no reward: {_described(e)}"
        return {"reward": reward, "metadata": metadata}

    async def _open(self, turn: Turn) -> None:
        self._turn, self._stack = turn, AsyncExitStack()
        self._tools = await self._stack.enter_async_context(turn.tools())
        self._state.turn = turn.step_id
        if clash := sorted({tool.name for tool in self._tools.list()} & {*RESERVED_TOOL_NAMES, END_TURN.name}):
            raise EpisodeError(f"the env offers tools OpenEnv reserves: {', '.join(clash)}")

    async def _why_stopped(self) -> dict[str, Any] | None:
        """The error metadata a step answers with once the episode can't go on: it has ended, its run has, or the call
        opening its turn was cancelled."""
        if self._episode is None:
            return {"status": "error", "error": ENDED, "task": self._task_metadata()}
        if self._tools is None:
            return (await self._end_with_error("the call opening the episode's turn was cancelled")).metadata
        if not self._episode.finished:
            return None
        try:
            await self._episode.result()
            error = "the run ended while the agent's turn was open"
        except Exception as e:
            error = _described(e)
        return (await self._end_with_error(error)).metadata

    async def _end_with_error(self, error: Exception | str) -> Observation:
        message = error if isinstance(error, str) else _described(error)
        await self._abandon(message)
        return self._error_observation(message)

    def _error_observation(self, error: str) -> Observation:
        self._state.done = True
        return Observation(done=True, reward=None, metadata={
            "status": "error", "error": error, "task": self._task_metadata()})

    async def _abandon_when_free(self, reason: str) -> None:
        async with self._lock:
            await self._abandon(reason)

    async def _abandon(self, reason: str) -> None:
        """Abandon the episode and close the turn's tools, waiting for the teardown; the caller holds the lock. The
        abandon is tracked before anything is awaited, so a server shutting down meanwhile waits for it too, and a
        caller cancelled while it waits leaves it running for the next reset to wait for."""
        episode, self._episode = self._episode, None
        if episode is not None:
            self._ending = self._served.end_later(episode.abandon(reason))
        self._served.live.discard(self)
        await self._close_tools()
        self._turn = None
        if (ending := self._ending) is not None:
            try:
                await asyncio.shield(ending)
            finally:
                if ending.done():
                    self._ending = None

    async def _close_tools(self) -> None:
        stack, self._stack, self._tools = self._stack, None, None
        if stack is not None:
            await stack.aclose()

    def _split(self, split: str) -> tuple[ServedTask, ...]:
        if split not in self._served.splits:
            raise IndexError(f"there is no split {split!r}; the splits are {', '.join(self._served.splits)}")
        return self._served.splits[split]

    def _task_at(self, split: str, index: int) -> ServedTask:
        tasks = self._split(split)
        if not 0 <= index < len(tasks):
            raise IndexError(f"split {split!r} has {len(tasks)} tasks, so it has no task {index}")
        return tasks[index]

    def _describe(self, split: str, index: int) -> dict[str, Any]:
        served = self._task_at(split, index)
        prompt, system_prompt = served.spec.prompts[0]
        return {"id": served.task.id, "name": served.name, "index": index, "prompt": prompt,
                "system_prompt": system_prompt, "agent": served.spec.trainer_agents[0]}

    def _task_metadata(self) -> dict[str, Any] | None:
        if self._served_task is None:
            return None
        return {"id": self._served_task.task.id, "name": self._served_task.name, "split": self._state.split,
                "index": self._state.index}


def _tool(tool: Any) -> Tool:
    return Tool(name=tool.name, description=tool.description or "", input_schema=tool.inputSchema or {"type": "object"})


def _tool_error(name: str, kind: ToolErrorType, error: Exception | str) -> CallToolObservation:
    return CallToolObservation(tool_name=name, error=ToolError(
        error_type=kind, message=error if isinstance(error, str) else _described(error)))


def _described(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
