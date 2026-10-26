from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from agent_env.task.episode.rules import DEPLOYS, problems, works_in_sandbox
from agent_env.task.episode.run import Episode, EpisodeError
from agent_env.task.episode.steps import _EpisodeAgentStep
from agent_env.task.task import Task, substitute_seed_in_steps
from agent_env.task_step.context import TaskStepContext, sandbox_overrides
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


class EpisodeSpec:
    """A task read as an episode: the agents a loop plays, and the copy of the task that runs. Built by
    ``from_task``."""

    def __init__(self, *, copy: Task, trainer_agents: tuple[str, ...], seed: dict[str, str] | None):
        self._copy = copy
        self.trainer_agents = trainer_agents
        self._seed = seed

    @classmethod
    def from_task(
        cls, task: Task, *, trainer_agents: Sequence[str] | None = None, seed: Mapping[str, str] | None = None,
        turn_timeout: float | None = None,
    ) -> EpisodeSpec:
        """``trainer_agents`` are the agents the loop plays, by default the task's only prompted agent.
        ``turn_timeout`` is every turn's deadline in seconds, rounded up to a whole one, in place of its prompt's own
        ``timeout_seconds``, which budgets an agent's run rather than the loop playing it."""
        seed = dict(seed) if seed else None
        seeded = substitute_seed_in_steps(task.steps, seed or {})
        outside = _trainer_agents(task, seeded, trainer_agents)
        if found := problems(seeded, outside):
            raise EpisodeError(f"task {task.id!r} can't run as an episode: " + "; ".join(found))
        sandboxed = {name: works_in_sandbox(seeded, name) for name in outside}
        steps = [_in_episode(step, as_run, sandboxed, turn_timeout) for step, as_run in zip(task.steps, seeded)]
        return cls(copy=Task(id=task.id, version=task.version, steps=steps), trainer_agents=tuple(outside), seed=seed)

    @property
    def prompts(self) -> tuple[tuple[str, str | None], ...]:
        """Each outside turn's prompt and system prompt as the run hands them over, in step order: what a dataset
        row can carry before anything is deployed. A prompt that sets no system prompt keeps the one its agent last
        had, from its deploy or an earlier prompt."""
        seeded = substitute_seed_in_steps(self._copy.steps, self._seed or {})
        system_prompts: dict[str, str | None] = {}
        prompts = []
        for step in seeded:
            if isinstance(step, _EpisodeAgentStep):
                system_prompts[step.step.agent_name] = getattr(step.step, "system_prompt", None)
            elif step.type == "prompt_agent" and step.agent_name in self.trainer_agents:
                system_prompts[step.agent_name] = step.system_prompt or system_prompts.get(step.agent_name)
                prompts.append((step.prompt, system_prompts[step.agent_name]))
        return tuple(prompts)

    async def start(self, *, sandbox: str | None = None) -> Episode:
        """Start one run of the task, a new episode each call. It returns once the run's record exists, before
        anything is deployed; ``next_turn()`` is where the deploys and setup are waited for. ``sandbox`` is the
        provider its envs, agents, sandboxes and judge deploy on, as ``agent-env run --sandbox`` sets it."""
        episode_id = uuid.uuid4().hex
        metadata: dict[str, Any] = {"episode": {"id": episode_id, "trainer_agents": list(self.trainer_agents)}}
        if self._seed:
            metadata["seed"] = dict(self._seed)
        if sandbox:
            metadata["user_overrides"] = sandbox_overrides(sandbox)
        episode = Episode(episode_id, self._copy, TaskStepContext(metadata=metadata))
        await episode._launch()
        return episode


def _trainer_agents(task: Task, steps: list[TaskStep], named: Sequence[str] | None) -> list[str]:
    prompted = list(dict.fromkeys(step.agent_name for step in steps if step.type == "prompt_agent"))
    if named is None:
        if len(prompted) != 1:
            raise EpisodeError(
                f"task {task.id!r} prompts {prompted}; name the ones to play with trainer_agents=" if prompted
                else f"task {task.id!r} prompts no agent"
            )
        return prompted
    outside = list(dict.fromkeys(named))
    if not outside:
        raise EpisodeError(f"trainer_agents= names no agent; task {task.id!r} prompts {prompted}")
    if unprompted := [name for name in outside if name not in prompted]:
        raise EpisodeError(f"task {task.id!r} doesn't prompt {unprompted}; it prompts {prompted}")
    return outside


def _in_episode(step: TaskStep, as_run: TaskStep, sandboxed: dict[str, bool], turn_timeout: float | None) -> TaskStep:
    """The step the episode's copy of the task runs in place of ``step``, judged by ``as_run``, the step as the
    seed makes it; ``sandboxed`` says, for each outside agent by name, whether its proxy is deployed in its
    sandbox. The copy keeps every step's own fields, which the run seeds."""
    if as_run.type in DEPLOYS and as_run.agent_name in sandboxed:
        return _EpisodeAgentStep(
            id=step.id, version=step.version, step=step, sandboxed=sandboxed[as_run.agent_name],
            depends_on=step.depends_on, fail_task_on_error=step.fail_task_on_error,
        )
    if as_run.type == "prompt_agent" and as_run.agent_name in sandboxed and turn_timeout is not None:
        return PromptAgentTaskStep.from_dict({**step.to_dict(), "timeout_seconds": math.ceil(turn_timeout)})
    return step
