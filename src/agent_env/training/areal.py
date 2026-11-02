"""Train a bundle's own agents with AReaL's agent workflows (``rollout.agent.mode: inline``): ``AgentEnvWorkflow`` is
the agent AReaL's ``OpenAIProxyWorkflow`` runs for each sample, which plays the row's task as ``agent-env run`` does,
its solver's model calls going through AReaL's proxy, which records the tokens AReaL trains on, and returns the reward
the task's verifiers score. Needs no extra: AReaL calls it by its methods alone."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agent_env.store.routing import namespace_routing
from agent_env.task.task import Task
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.training._loop import session_loop
from agent_env.training._solver import first_prompt, rewarded, run_scores, solver_overrides, trained

#: The model the solver names: AReaL's proxy serves its policy whatever a request names, and its own agents name this.
_MODEL = "default"


def task_rows(bundle: str | Path, *, tasks: Sequence[str] = (), evals: Sequence[str] = (),
              agent: str | None = None) -> list[dict[str, Any]]:
    """The trainer's dataset, a row per task ``AgentEnvWorkflow`` plays with the same arguments: the task's id, which
    is how the workflow finds it, and its first prompt as ``messages``, for AReaL's logs. The model never sees this
    prompt: the task's own agent builds its own."""
    return [{"task_id": task.id, "messages": [{"role": "user", "content": first_prompt(task)}]}
            for task, _ in trained(bundle, tasks, evals, agent)]


class AgentEnvWorkflow:
    """The agent AReaL runs in inline mode, named as ``workflow: agent_env.training.areal.AgentEnvWorkflow`` with the
    arguments as ``workflow_kwargs``. A process prepares the bundle to run on local Docker once, however often AReaL
    builds the workflow; ``reward`` is as ``build_app``'s, and ``agent`` the agent the solver deploys, as
    ``agent-env run --agent``'s."""

    def __init__(self, bundle: str | Path, *, tasks: Sequence[str] = (), evals: Sequence[str] = (),
                 reward: str | Callable[[dict[str, float]], float] | None = None, agent: str | None = None):
        self._played = {task.id: (task, solver, rewarded(task, reward))
                        for task, solver in _prepared(str(bundle), tuple(tasks), tuple(evals), agent)}

    async def run(self, data: dict[str, Any], *, base_url: str, api_key: str, **_: Any) -> float:
        """Plays ``data``'s task, its solver calling the model at ``base_url`` with ``api_key``, the key of the session
        AReaL records the calls in, and returns its reward. Raises, which has AReaL leave the sample out, when the run
        fails or records no score its reward reads. Cancelled, it stops the run and passes the cancel on once the run
        is torn down."""
        if (played := self._played.get(data.get("task_id"))) is None:
            raise KeyError(f"the row names no task this workflow plays: {data.get('task_id')!r}; build the dataset "
                           "with task_rows, from the same bundle, tasks and evals")
        task, solver, reward = played
        run = _Run(task, solver_overrides(solver, {"litellm_base_url": base_url, "litellm_api_key": api_key}))
        done = asyncio.wrap_future(asyncio.run_coroutine_threadsafe(run.scores(), session_loop()))
        try:
            scores = await asyncio.shield(done)
        except asyncio.CancelledError:
            session_loop().call_soon_threadsafe(run.stop)
            await asyncio.wait([done])
            raise
        if (score := reward(scores)) is None:
            raise ValueError(f"task {task.id!r} records no score to train on")
        return float(score)

    def should_reject_failed_sample(self, error: Exception, disposition: str | None) -> bool:
        """AReaL's question, after 2.1.0, for a sample whose ``run`` raised: it is left out of its group, the group's
        other samples still training, unless AReaL found the model at fault, which it trains on as a zero reward."""
        return disposition != "model_failure_zero"


class _Run:
    """One run of a task on the session loop, which ``stop``, called there, cuts short; its teardown always runs to the
    end."""

    def __init__(self, task: Task, overrides: dict[str, Any]):
        self._task, self._overrides = task, overrides
        self._running: asyncio.Task | None = None
        self._stopped = False

    async def scores(self) -> dict[str, float]:
        context = TaskStepContext(metadata={"task_id": self._task.id, "user_overrides": self._overrides})
        with namespace_routing():
            try:
                if self._stopped:
                    raise asyncio.CancelledError
                self._running = asyncio.ensure_future(self._task.run(agent_model=_MODEL, context=context))
                await self._running
                return run_scores(self._task, context)
            finally:
                await teardown_run(context)

    def stop(self) -> None:
        self._stopped = True
        if self._running is not None:
            self._running.cancel()


@functools.cache
def _prepared(bundle: str, tasks: tuple[str, ...], evals: tuple[str, ...],
              agent: str | None) -> list[tuple[Task, dict[str, dict[str, Any]]]]:
    return trained(bundle, tasks, evals, agent)
