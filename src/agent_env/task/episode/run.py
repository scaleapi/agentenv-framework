from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

from agentenv_protocol import EnvironmentTool, ToolResult, ToolSession

from agent_env.providers.sandbox_providers.sandbox_provider import sandbox_request_headers_for_url
from agent_env.task.store import record_task_cancelled
from agent_env.task.task import Task, recorded_scores, substitute_seed_in_steps
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext

if TYPE_CHECKING:
    from agent_env.task.episode.steps import _EpisodeTurnStep

logger = logging.getLogger(__name__)

_live: dict[str, Episode] = {}


class EpisodeError(RuntimeError):
    """A task that can't run as an episode, or an episode asked for what it can't give."""


class Turn:
    """An outside agent's turn. Its model reads ``system_prompt`` and ``prompt``, acts through ``tools()`` (or an MCP
    client of its own on ``mcp_urls``, sending ``headers`` and ``role``), and hands its reply back with ``end()``."""

    def __init__(
        self, *, agent: str, step_id: str, prompt: str, system_prompt: str | None, role: str | None,
        mcp_urls: dict[str, str],
    ):
        self.agent = agent
        self.step_id = step_id
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.role = role
        self.mcp_urls = mcp_urls
        self.headers = {env_id: sandbox_request_headers_for_url(url) for env_id, url in mcp_urls.items()}
        self._reply: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    def tools(self, *, timeout: float = 600.0) -> TurnTools:
        """A client for the tools of the envs this agent acts in, calling them as its role with the headers each URL
        needs; use it with ``async with``. ``timeout`` bounds each call, in seconds."""
        return TurnTools(self, timeout)

    async def end(self, response: str | None = None) -> None:
        """Hand the turn back, and the run carries on. No reply records an empty one, as an agent that answers with
        nothing leaves. Once the turn is over, ended or not, this does nothing."""
        if self._open:
            self._reply.set_result(response or "")

    @property
    def _open(self) -> bool:
        return not self._reply.done()

    async def _replied_within(self, seconds: float) -> str:
        return await asyncio.wait_for(self._reply, seconds)

    def _fail(self, error: Exception) -> None:
        if self._open:
            self._reply.set_exception(error)


class TurnTools:
    """The tools of a turn's envs, listed once when it opens: ``list()`` names every one, and ``call()`` routes a
    call to the env that offers it."""

    def __init__(self, turn: Turn, timeout: float):
        self._step_id = turn.step_id
        self._sessions = {
            env_id: ToolSession(mcp_url=url, role=turn.role, headers=turn.headers[env_id], timeout=timeout)
            for env_id, url in turn.mcp_urls.items()
        }
        self._tools: list[EnvironmentTool] = []
        self._offered_by: dict[str, str] = {}

    async def __aenter__(self) -> TurnTools:
        try:
            for env_id, session in self._sessions.items():
                await session.open()
                for tool in await session.list_tools():
                    if (other := self._offered_by.get(tool.name)) is not None:
                        raise EpisodeError(f"tool {tool.name!r} is offered by both env {other!r} and env {env_id!r}")
                    self._offered_by[tool.name] = env_id
                    self._tools.append(tool)
        except BaseException:
            await self._close()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._close()

    def list(self) -> list[EnvironmentTool]:
        return list(self._tools)

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        if name not in self._offered_by:
            raise EpisodeError(
                f"no env of turn {self._step_id!r} offers tool {name!r}; they offer {sorted(self._offered_by)}")
        return await self._sessions[self._offered_by[name]].call_tool(name, arguments)

    async def _close(self) -> None:
        await asyncio.gather(*(session.close() for session in self._sessions.values()))


class Episode:
    """One run of an ``EpisodeSpec``'s task, which waits at each outside turn for the caller to play it.

    Cancelling a caller of ``next_turn()``, ``result()`` or ``abandon()`` cancels that call, never the run. Use an
    episode on the event loop that started it."""

    def __init__(self, id: str, task: Task, context: TaskStepContext):
        self.id = id
        self._task = task
        self._context = context
        self._run: asyncio.Task[None]
        self._bindings: dict[str, tuple[str | None, dict[str, str]]] = {}
        self._waiting: deque[Turn] = deque()
        self._turns: set[Turn] = set()
        self._changed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._tearing_down = False
        self._abandoned: str | None = None

    async def next_turn(self) -> Turn | None:
        """The next outside turn the run reaches, or None once the run has finished. Raises the run's error if it
        failed first. End the turns you hold before waiting on one that depends on them."""
        while True:
            changed = self._changed
            while self._waiting and self._abandoned is None:
                turn = self._waiting.popleft()
                if turn._open:
                    return turn
            if self._run.done():
                self._raise_unless_completed()
                return None
            await self._wait(changed)

    async def result(self) -> dict[str, float]:
        """Every score the run recorded, by the id of the step that recorded it, once the run and its teardown are
        over: what ``agent-env run`` reports. Making a reward of them is the caller's."""
        while not self._run.done():
            changed = self._changed
            if open_turn := next((turn for turn in self._turns if turn._open), None):
                raise EpisodeError(
                    f"turn {open_turn.step_id!r} of agent {open_turn.agent!r} isn't ended, so the run can't finish"
                )
            await self._wait(changed)
        self._raise_unless_completed()
        seeded = substitute_seed_in_steps(self._task.steps, self._context.metadata.get("seed") or {})
        return recorded_scores(seeded, self._context)

    async def abandon(self, reason: str) -> None:
        """Stop the run and wait for its teardown; it is recorded cancelled with ``reason``. A run a step has
        already failed is only finishing what's in flight, so this ends its open turns and the run's error stands.
        Once the run has ended, or while it tears down, this only waits."""
        if not (self._run.done() or self._tearing_down or self._abandoned is not None):
            if any(failure.get("is_fatal") for failure in self._context.metadata.get("failed_steps") or []):
                for turn in self._turns:
                    turn._fail(EpisodeError(f"episode abandoned: {reason}"))
            else:
                self._abandoned = reason
                self._run.cancel(msg=reason)
        await asyncio.wait((self._run,))

    async def _launch(self) -> None:
        """Start the run and return once its record exists. Cancelled, it takes the run down and waits for teardown."""
        _live[self.id] = self
        running = asyncio.Event()
        self._run = asyncio.create_task(self._drive(running))
        try:
            await running.wait()
        except asyncio.CancelledError:
            self._abandoned = "start() was cancelled"
            self._run.cancel()
            await asyncio.wait((self._run,))
            raise

    async def _drive(self, running: asyncio.Event) -> None:
        running.set()
        try:
            await self._task.run(context=self._context, instance_id=self.id)
        finally:
            self._tearing_down = True
            try:
                report = await teardown_run(self._context)
                if report.still_up:
                    logger.warning("episode %s: %s may still be up", self.id, [s.sandbox_id for s in report.still_up])
                if self._abandoned is not None:
                    await asyncio.to_thread(record_task_cancelled, self.id, self._abandoned, Task._utc_now_str())
            finally:
                del _live[self.id]

    async def _wait(self, changed: asyncio.Future[None]) -> None:
        """Until the run ends or what was ``changed`` when the caller last looked changes again."""
        await asyncio.wait((self._run, changed), return_when=asyncio.FIRST_COMPLETED)

    def _raise_unless_completed(self) -> None:
        if self._abandoned is not None:
            raise EpisodeError(f"episode abandoned: {self._abandoned}")
        if (error := self._run.exception()) is not None:
            raise error

    def _bind(self, agent: str, role: str | None, mcp_urls: dict[str, str]) -> None:
        self._bindings[agent] = (role, mcp_urls)

    async def _play(self, step: _EpisodeTurnStep) -> str:
        if step.agent_name not in self._bindings:
            raise RuntimeError(
                f"agent {step.agent_name!r} isn't deployed yet; turn {step.id!r} must depend on its deploy_agent step"
            )
        role, mcp_urls = self._bindings[step.agent_name]
        turn = Turn(agent=step.agent_name, step_id=step.id, prompt=step.prompt, system_prompt=step.system_prompt,
                    role=role, mcp_urls=dict(mcp_urls))
        self._waiting.append(turn)
        self._turns.add(turn)
        self._changed.set_result(None)
        self._changed = asyncio.get_running_loop().create_future()
        try:
            return await turn._replied_within(step.timeout_seconds)
        except TimeoutError:
            raise TimeoutError(
                f"turn {step.id!r} of agent {step.agent_name!r} wasn't ended within {step.timeout_seconds}s"
            ) from None
        finally:
            self._turns.discard(turn)


def _episode_of(context: TaskStepContext) -> Episode:
    return _live[context.metadata["episode"]["id"]]
