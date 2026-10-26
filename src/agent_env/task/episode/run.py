from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

import httpx
from agentenv_protocol import EnvironmentTool, ToolResult
from agentenv_protocol.a2a_agent import (
    ProxyBashAction,
    ProxyBashResult,
    ProxySkill,
    ProxyTextPart,
    ProxyToolAction,
)

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.providers.sandbox_providers.sandbox_provider import sandbox_request_headers_for_url
from agent_env.store.routing import namespace_routing, run_scope
from agent_env.task.episode.proxy import InProcessProxy, ProxyClient, ProxyRefused
from agent_env.task.store import record_task_cancelled
from agent_env.task.task import Task, recorded_scores, substitute_seed_in_steps
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import _mcp_add_body

logger = logging.getLogger(__name__)

_live: dict[str, Episode] = {}
_DRAIN_RETRY_SECONDS = 1.0
_CARD_TIMEOUT_SECONDS = 10.0


class EpisodeError(RuntimeError):
    """A task that can't run as an episode, or an episode asked for what it can't give."""


# A skill the task gave the agent: its instructions are ``path``'s ``SKILL.md``, in the agent's sandbox.
Skill = ProxySkill


class Turn:
    """An outside agent's turn. Its model reads ``system_prompt``, ``prompt`` and ``skills``, acts through ``tools()``,
    each of its calls recorded with ``record_model_call()``, and hands its reply back with ``end()``. Every action runs
    in the agent's proxy, which records the turn."""

    def __init__(
        self, *, agent: str, step_id: str, prompt: str, system_prompt: str | None, skills: list[Skill],
        proxy: ProxyClient, task_id: str,
    ):
        self.agent = agent
        self.step_id = step_id
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.skills = skills
        self._proxy = proxy
        self._task_id = task_id
        self._open = True

    def tools(self, *, timeout: float = 600.0) -> TurnTools:
        """A client for what this turn can call: the tools of the envs its agent acts in, and ``bash`` when it works
        in a sandbox, each run where the agent works; use it with ``async with``. ``timeout`` bounds each env tool's
        call, in seconds."""
        return TurnTools(self, timeout)

    async def record_model_call(self, message: str | dict[str, Any], *, model: str | None = None,
                                usage: dict[str, Any] | None = None) -> None:
        """Record one of the model's calls, once it has returned: an OpenAI chat message, an Anthropic message or a
        string. From then on each call to the turn's tools must be one the last recorded call made, and all must run
        before the next is recorded or the turn ends; else ``EpisodeError``. Calls to tools the turn doesn't offer
        aren't checked."""
        await self._checked(self._proxy.record(self._task_id, message, model, usage))

    async def end(self, response: str | None = None) -> None:
        """Hand the turn back, and the run carries on. No reply records an empty one, as an agent that answers with
        nothing leaves. Once the turn is over, ended or not, this does nothing. Raises ``EpisodeError``, leaving the
        turn open, while calls a recorded model call made haven't run; a proxy that can't be reached fails the run,
        which ``next_turn()`` and ``result()`` report."""
        if not self._open:
            return
        try:
            await self._checked(self._proxy.reply(self._task_id, parts=[ProxyTextPart(text=response or "")]))
        except httpx.HTTPError as e:
            logger.warning("turn %r of agent %r: its reply didn't reach the proxy: %s", self.step_id, self.agent, e)
        self._open = False

    async def _fail(self, error: Exception) -> None:
        if self._open:
            self._open = False
            try:
                await self._proxy.reply(self._task_id, error=str(error))
            except httpx.HTTPError as e:
                logger.warning("turn %r of agent %r: its failure didn't reach the proxy: %s", self.step_id, self.agent,
                               e)

    async def _checked(self, call: Any) -> Any:
        """``call``'s answer, with the proxy's refusal, or a turn it no longer has, as an ``EpisodeError``."""
        if not self._open:
            call.close()
            raise EpisodeError(f"turn {self.step_id!r} of agent {self.agent!r} is over")
        try:
            return await call
        except ProxyRefused as e:
            raise EpisodeError(f"turn {self.step_id!r} of agent {self.agent!r}: {e}") from None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self._open = False
                raise EpisodeError(f"turn {self.step_id!r} of agent {self.agent!r} is over") from None
            raise


class TurnTools:
    """What a turn can call, listed once when it opens: its envs' tools, and ``bash`` when its agent works in a
    sandbox. ``list()`` names every one, and ``call()`` has the agent's proxy run a call where the agent works."""

    def __init__(self, turn: Turn, timeout: float):
        self._turn = turn
        self._timeout = timeout
        self._tools: list[EnvironmentTool] = []
        self._servers: dict[str, str | None] = {}

    async def __aenter__(self) -> TurnTools:
        for tool in await self._turn._checked(self._turn._proxy.tools(self._turn._task_id)):
            self._tools.append(EnvironmentTool(name=tool.name, description=tool.description, inputSchema=tool.input_schema))
            self._servers[tool.name] = tool.server
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass

    def list(self) -> list[EnvironmentTool]:
        return list(self._tools)

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        if name not in self._servers:
            raise EpisodeError(f"turn {self._turn.step_id!r} offers no tool {name!r}; it offers {sorted(self._servers)}")
        if (server := self._servers[name]) is None:
            if not isinstance(command := (arguments or {}).get("command"), str):
                raise EpisodeError(f"tool {name!r} takes a command string, not {command!r}")
            action: ProxyBashAction | ProxyToolAction = ProxyBashAction(kind="bash", command=command)
        else:
            action = ProxyToolAction(kind="tool", server=server, name=name, arguments=arguments or {},
                                     timeout_seconds=self._timeout)
        result = await self._turn._checked(self._turn._proxy.act(self._turn._task_id, action))
        if isinstance(result, ProxyBashResult):
            if result.timed_out:
                return ToolResult(content=[{"type": "text", "text": "error: the command timed out"}], isError=True)
            text = result.stdout + result.stderr
            if failed := result.exit_code != 0:
                text = "\n".join(filter(None, (text.removesuffix("\n"), f"exit code {result.exit_code}")))
            return ToolResult(content=[{"type": "text", "text": text}], isError=failed)
        return ToolResult(content=result.content, isError=result.is_error)


class Episode:
    """One run of an ``EpisodeSpec``'s task, which waits at each outside turn for the caller to play it.

    Cancelling a caller of ``next_turn()``, ``result()`` or ``abandon()`` cancels that call, never the run. Use an
    episode on the event loop that started it."""

    def __init__(self, id: str, task: Task, context: TaskStepContext):
        self.id = id
        self._task = task
        self._context = context
        self._run: asyncio.Task[None]
        self._waiting: deque[Turn] = deque()
        self._turns: set[Turn] = set()
        self._changed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._tearing_down = False
        self._abandoned: str | None = None
        self._proxies: list[ProxyClient] = []
        self._drains: list[asyncio.Task[None]] = []
        self._played: set[str] = set()
        self._in_process: list[InProcessProxy] = []

    @property
    def finished(self) -> bool:
        """Whether the run has ended, its teardown with it; ``result()`` then returns or raises at once."""
        return self._run.done()

    async def next_turn(self) -> Turn | None:
        """The next outside turn the run reaches, or None once the run has finished. Raises the run's error if it
        failed first. End the turns you hold before waiting on one that depends on them."""
        while True:
            changed = self._changed
            self._close_turns_of_failed_steps()
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
            self._close_turns_of_failed_steps()
            if open_turn := next((turn for turn in self._turns if turn._open), None):
                raise EpisodeError(
                    f"turn {open_turn.step_id!r} of agent {open_turn.agent!r} isn't ended, so the run can't finish"
                )
            await self._wait(changed)
        self._raise_unless_completed()
        return recorded_scores(self._seeded_steps(), self._context)

    async def abandon(self, reason: str) -> None:
        """Stop the run and wait for its teardown; it is recorded cancelled with ``reason``. A run a step has
        already failed is only finishing what's in flight, so this ends its open turns and the run's error stands.
        Once the run has ended, or while it tears down, this only waits."""
        if not (self._run.done() or self._tearing_down or self._abandoned is not None):
            if any(failure.get("is_fatal") for failure in self._context.metadata.get("failed_steps") or []):
                for turn in list(self._turns):
                    await turn._fail(EpisodeError(f"episode abandoned: {reason}"))
            else:
                self._abandoned = reason
                for turn in self._turns:
                    turn._open = False
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
        """Run the task with namespace routing on, as ``agent-env run`` does, so the proxy, an ``@local`` agent,
        resolves."""
        running.set()
        with namespace_routing():
            try:
                await self._task.run(context=self._context, instance_id=self.id)
            finally:
                self._tearing_down = True
                try:
                    await self._release_proxies()
                    report = await teardown_run(self._context)
                    if report.still_up:
                        logger.warning("episode %s: %s may still be up", self.id,
                                       [s.sandbox_id for s in report.still_up])
                    if self._abandoned is not None:
                        await asyncio.to_thread(_record_cancelled, self._task.id, self.id, self._abandoned)
                finally:
                    del _live[self.id]

    async def _wait(self, changed: asyncio.Future[None]) -> None:
        """Until the run ends or what was ``changed`` when the caller last looked changes again."""
        await asyncio.wait((self._run, changed), return_when=asyncio.FIRST_COMPLETED)

    def _close_turns_of_failed_steps(self) -> None:
        """A step that failed has given up on its turns, which its proxy drops only a long-poll later."""
        failed = {failure.get("step_id") for failure in self._context.metadata.get("failed_steps") or []}
        for turn in self._turns:
            if turn.step_id in failed:
                turn._open = False

    def _raise_unless_completed(self) -> None:
        if self._abandoned is not None:
            raise EpisodeError(f"episode abandoned: {self._abandoned}")
        if (error := self._run.exception()) is not None:
            raise error

    async def _serve_in_process(self, agent: str, role: str | None, mcp_urls: dict[str, str]) -> DeployedAgent:
        """A proxy served in this process for ``agent``, which works in no sandbox, with each env server registered as
        ``deploy_agent`` registers them with any agent, at the URL this process reaches: played as the others are."""
        server = InProcessProxy()
        self._in_process.append(server)
        url = await server.start()
        async with httpx.AsyncClient(timeout=_CARD_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{url}/.well-known/agent.json")
            response.raise_for_status()
            card = response.json()
            mcp_config = A2AAgent.find_extension(card, A2AAgent.EXT_MCP_CONFIG)
            for env_id, mcp_url in mcp_urls.items():
                body = _mcp_add_body(mcp_config, mcp_url, sandbox_request_headers_for_url(mcp_url), env_id, role)
                (await client.post(url + mcp_config["params"]["endpoint"], json=body)).raise_for_status()
        deployed = DeployedAgent(agent_name=agent, api_url=url, a2a_url=url, a2a_card=card, role=role)
        await self._attach(agent, deployed)
        return deployed

    async def _attach(self, agent: str, deployed: DeployedAgent) -> None:
        """Claim ``agent``'s proxy and hand out the turns it opens."""
        proxy = ProxyClient(deployed.a2a_url, deployed.a2a_card or {})
        self._proxies.append(proxy)
        await proxy.claim()
        self._drains.append(asyncio.create_task(self._drain(agent, proxy)))

    async def _drain(self, agent: str, proxy: ProxyClient) -> None:
        """Hand out each turn the proxy opens, and close each one it drops, until the run ends."""
        after, open_turns = 0, {}
        while True:
            try:
                answer = await proxy.next(after)
            except httpx.HTTPError as e:
                logger.warning("episode %s: reading agent %r's turns failed, retrying: %s", self.id, agent, e)
                await asyncio.sleep(_DRAIN_RETRY_SECONDS)
                continue
            after = answer.version
            listed = {item.task_id: item for item in answer.turns}
            for task_id in [task_id for task_id in open_turns if task_id not in listed]:
                turn = open_turns.pop(task_id)
                turn._open = False
                self._turns.discard(turn)
            for task_id, item in listed.items():
                if task_id not in open_turns:
                    open_turns[task_id] = turn = Turn(
                        agent=agent, step_id=self._step_of(agent, item.context_id),
                        prompt="\n".join(part.text for part in item.parts), system_prompt=item.system_prompt,
                        skills=list(item.skills), proxy=proxy, task_id=task_id,
                    )
                    self._waiting.append(turn)
                    self._turns.add(turn)
            self._changed.set_result(None)
            self._changed = asyncio.get_running_loop().create_future()

    def _step_of(self, agent: str, context_id: str) -> str:
        """The ``prompt_agent`` step a message to ``agent`` on ``context_id`` comes from: of its prompts in that
        conversation, the first not yet played, since a prompt continuing a conversation runs after the one it
        continues."""
        conversations = self._context.metadata.get("a2a_conversations", {})
        steps = [step.id for step in self._seeded_steps() if step.type == "prompt_agent" and step.agent_name == agent
                 and (step.context_id or conversations.get(step.id)) == context_id]
        step_id = next((step_id for step_id in steps if step_id not in self._played), steps[-1] if steps else context_id)
        self._played.add(step_id)
        return step_id

    def _seeded_steps(self) -> list:
        return substitute_seed_in_steps(self._task.steps, self._context.metadata.get("seed") or {})

    async def _release_proxies(self) -> None:
        for drain in self._drains:
            drain.cancel()
        await asyncio.gather(*self._drains, return_exceptions=True)
        for turn in self._turns:
            turn._open = False
        for server in self._in_process:
            server.stop()
        await asyncio.gather(*(proxy.close() for proxy in self._proxies), return_exceptions=True)


def _episode_of(context: TaskStepContext) -> Episode:
    return _live[context.metadata["episode"]["id"]]


def _record_cancelled(task_id: str, instance_id: str, reason: str) -> None:
    """Marked where the run recorded its instance, as ``Task.run`` scopes it: an ``@local`` task's in the local
    stores."""
    with run_scope(task_id):
        record_task_cancelled(instance_id, reason, Task._utc_now_str())
