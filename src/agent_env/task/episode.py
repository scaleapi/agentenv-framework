"""Run a task with some of its agents played from outside AgentEnv, by a training loop's model or any other caller.

AgentEnv runs everything else exactly as ``Task.run`` does, so the reward is the score ``agent-env run`` would give.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from agent_env.entity_refs import EntityRef
from agent_env.providers.sandbox_providers.sandbox_provider import sandbox_request_headers_for_url
from agent_env.task.episode_rules import problems
from agent_env.task.store import record_task_cancelled
from agent_env.task.task import Task, substitute_seed_in_steps
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.deploy_agent import bound_deployment, live_mcp_url
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

logger = logging.getLogger(__name__)

_live: dict[str, Episode] = {}


class EpisodeError(RuntimeError):
    """A task that can't run as an episode, or an episode asked for what it can't give."""


@dataclass(frozen=True)
class TurnSpec:
    step_id: str
    prompt: str
    system_prompt: str | None


@dataclass(frozen=True)
class AgentSpec:
    role: str | None
    turns: tuple[TurnSpec, ...]


class EpisodeSpec:
    """A task read as an episode: the agents played from outside, their turns, and the copy of the task that runs."""

    def __init__(
        self, *, copy: Task, agents: dict[str, AgentSpec], reward_verifier_id: str, seed: dict[str, str] | None,
    ):
        self._copy = copy
        self.agents = agents
        self._reward_verifier_id = reward_verifier_id
        self._seed = seed

    @classmethod
    def from_task(
        cls, task: Task, *, agents: Sequence[str] | None = None, seed: Mapping[str, str] | None = None,
        turn_timeout: float | None = None,
    ) -> EpisodeSpec:
        """``turn_timeout`` is every outside turn's deadline in seconds, in place of its prompt's own
        ``timeout_seconds``, which budgets an agent's run rather than the loop playing it."""
        seed = dict(seed) if seed else None
        seeded = substitute_seed_in_steps(task.steps, seed or {})
        outside = _outside_agents(task, seeded, agents)
        found = problems(seeded, outside)
        verifiers = _top_level_verifiers(seeded)
        if len(verifiers) != 1:
            found.append(
                f"the reward is the score of the one verifier no aggregate_verifiers step lists, and the task has "
                f"{len(verifiers)}" + (f": {[step.id for step in verifiers]}" if verifiers else "")
            )
        if found:
            raise EpisodeError(f"task {task.id!r} can't run as an episode: " + "; ".join(found))
        deploys = {
            as_run.agent_name: step for step, as_run in zip(task.steps, seeded)
            if step.type == "deploy_agent" and as_run.agent_name in outside
        }
        steps = [_stand_in(step, as_run, deploys, turn_timeout) for step, as_run in zip(task.steps, seeded)]
        return cls(
            copy=Task(id=task.id, version=task.version, steps=steps),
            agents={name: _agent_spec(seeded, name) for name in outside},
            reward_verifier_id=verifiers[0].verifier_id,
            seed=seed,
        )

    async def start(self, *, run_group_id: str | None = None) -> Episode:
        """Start one run of the task, a new episode each call. It returns once the run's record exists, before
        anything is deployed; ``next_turn()`` is where the deploys and setup are waited for."""
        episode_id = uuid.uuid4().hex
        metadata: dict[str, Any] = {"episode": {"id": episode_id, "agents": list(self.agents)}}
        if self._seed:
            metadata["seed"] = dict(self._seed)
        if run_group_id is not None:
            metadata["run_group_id"] = run_group_id
        episode = Episode(episode_id, self._copy, self._reward_verifier_id, TaskStepContext(metadata=metadata))
        await episode._launch()
        return episode


class Turn:
    """An outside agent's turn: what it is asked, the envs it acts in and the headers each needs, until ``end()``."""

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


class Episode:
    """One run of an ``EpisodeSpec``'s task, which waits at each outside turn for the caller to play it.

    Cancelling a caller of ``next_turn()``, ``result()`` or ``abandon()`` cancels that call, never the run. Use an
    episode on the event loop that started it."""

    def __init__(self, id: str, task: Task, reward_verifier_id: str, context: TaskStepContext):
        self.id = id
        self._task = task
        self._reward_verifier_id = reward_verifier_id
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

    async def result(self) -> float:
        """The reward, once the run and its teardown are over: the score of the task's one top-level verifier."""
        while not self._run.done():
            changed = self._changed
            if open_turn := next((turn for turn in self._turns if turn._open), None):
                raise EpisodeError(
                    f"turn {open_turn.step_id!r} of agent {open_turn.agent!r} isn't ended, so the run can't finish"
                )
            await self._wait(changed)
        self._raise_unless_completed()
        verifications = self._context.metadata.get("verifications") or {}
        score = (verifications.get(self._reward_verifier_id) or {}).get("score")
        if not isinstance(score, (int, float)):
            raise EpisodeError(f"verifier {self._reward_verifier_id!r} left no score")
        return float(score)

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


def _outside_agents(task: Task, steps: list[TaskStep], agents: Sequence[str] | None) -> list[str]:
    prompted = list(dict.fromkeys(step.agent_name for step in steps if step.type == "prompt_agent"))
    if agents is None:
        if len(prompted) != 1:
            raise EpisodeError(
                f"task {task.id!r} prompts {prompted}; name the ones to play with agents=" if prompted
                else f"task {task.id!r} prompts no agent"
            )
        return prompted
    outside = list(dict.fromkeys(agents))
    if not outside:
        raise EpisodeError(f"agents= names no agent; task {task.id!r} prompts {prompted}")
    if unprompted := [name for name in outside if name not in prompted]:
        raise EpisodeError(f"task {task.id!r} doesn't prompt {unprompted}; it prompts {prompted}")
    return outside


def _top_level_verifiers(steps: list[TaskStep]) -> list[TaskStep]:
    aggregated = {vid for step in steps if step.type == "aggregate_verifiers" for vid in step.verifier_ids}
    return [step for step in steps if getattr(step, "verifier_id", None) and step.verifier_id not in aggregated]


def _agent_spec(steps: list[TaskStep], name: str) -> AgentSpec:
    deploy = next(step for step in steps if step.type == "deploy_agent" and step.agent_name == name)
    return AgentSpec(role=deploy.role, turns=tuple(
        TurnSpec(step_id=step.id, prompt=step.prompt, system_prompt=step.system_prompt or deploy.system_prompt)
        for step in steps if step.type == "prompt_agent" and step.agent_name == name
    ))


def _stand_in(step: TaskStep, as_run: TaskStep, deploys: dict[str, TaskStep], turn_timeout: float | None) -> TaskStep:
    """The step the episode's copy of the task runs in place of ``step``, judged by ``as_run``, the step as the
    seed makes it; ``deploys`` are the outside agents', by name. A stand-in keeps the step's own fields, which the
    run seeds."""
    if as_run.type == "deploy_agent" and as_run.agent_name in deploys:
        return _EpisodeAgentStep(
            id=step.id, version=step.version, depends_on=step.depends_on, fail_task_on_error=step.fail_task_on_error,
            agent_name=step.agent_name, env_ids=step.env_ids, env_step_id=step.env_step_id, role=step.role,
        )
    if as_run.type == "prompt_agent" and as_run.agent_name in deploys:
        timeout = step.timeout_seconds if turn_timeout is None else turn_timeout
        return _EpisodeTurnStep(
            id=step.id, version=step.version, depends_on=step.depends_on, fail_task_on_error=step.fail_task_on_error,
            agent_name=step.agent_name, prompt_id=step.prompt_id, prompt=step.prompt,
            system_prompt=step.system_prompt or deploys[as_run.agent_name].system_prompt,
            timeout_seconds=PromptAgentTaskStep.DEFAULT_PROMPT_TIMEOUT_SECONDS if timeout is None else timeout,
        )
    return step


class _EpisodeAgentStep(TaskStep):
    """An outside agent's ``deploy_agent``: deploys nothing, and binds the agent to its envs where it stood."""

    type: ClassVar[str] = "episode_agent"
    entity_refs = (EntityRef.env("env_ids[]"),)

    def __init__(
        self, id: str, version: Optional[int], *, agent_name: str, env_ids: list[str], env_step_id: str | None,
        role: str | None, depends_on: Optional[list] = None, fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.agent_name = agent_name
        self.env_ids = list(env_ids)
        self.env_step_id = env_step_id
        self.role = role

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "agent_name": self.agent_name, "env_ids": self.env_ids,
                "env_step_id": self.env_step_id, "role": self.role}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _EpisodeAgentStep:
        return cls(**cls._base_from_dict(data), agent_name=data["agent_name"], env_ids=data["env_ids"],
                   env_step_id=data["env_step_id"], role=data["role"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        mcp_urls = {}
        for env_id in self.env_ids:
            deployed = bound_deployment(context, env_id, env_step_id=self.env_step_id, step_id=self.id)
            mcp_urls[env_id] = deployed.mcp_url if deployed is not None else live_mcp_url(env_id)
        _episode_of(context)._bind(self.agent_name, self.role, mcp_urls)
        return context


class _EpisodeTurnStep(TaskStep):
    """An outside agent's ``prompt_agent``: hands the turn to the caller and records the reply it gets back."""

    type: ClassVar[str] = "episode_turn"
    entity_refs = ()

    def __init__(
        self, id: str, version: Optional[int], *, agent_name: str, prompt_id: str, prompt: str,
        system_prompt: str | None, timeout_seconds: float, depends_on: Optional[list] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.agent_name = agent_name
        self.prompt_id = prompt_id
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "agent_name": self.agent_name, "prompt_id": self.prompt_id,
                "prompt": self.prompt, "system_prompt": self.system_prompt, "timeout_seconds": self.timeout_seconds}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _EpisodeTurnStep:
        return cls(**cls._base_from_dict(data), agent_name=data["agent_name"], prompt_id=data["prompt_id"],
                   prompt=data["prompt"], system_prompt=data["system_prompt"],
                   timeout_seconds=data["timeout_seconds"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        reply = await _episode_of(context)._play(self)
        context.prompt_responses.append(PromptResponse(
            prompt_id=self.prompt_id, response=reply, prompt_text=self.prompt, agent_name=self.agent_name,
            step_id=self.id,
        ))
        return context
