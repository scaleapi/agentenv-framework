"""What training a bundle's own agents shares, the harness and AReaL's helpers: the tasks whose solver, the agent a step
prompts, can train, prepared as ``agent-env run`` prepares them to run on local Docker, which alone reaches the
trainer's endpoint on this machine, a remote sandbox needing a tunnel; and the overrides that point the solver alone at
that endpoint, its other agents and judges keeping the configured one."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agent_env.bundle.plan import solver_steps
from agent_env.bundle.resolve import build_step
from agent_env.bundle.run import planned_tasks, prepare_bundle, solver_params
from agent_env.store.routing import namespace_routing
from agent_env.task.task import Task
from agent_env.task_step.context import sandbox_overrides
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.training.reward import reward_of

logger = logging.getLogger(__name__)
SANDBOX = "local"


def trained(bundle: str | Path, tasks: Sequence[str], evals: Sequence[str],
            agent: str | None) -> list[tuple[Task, dict[str, dict[str, Any]]]]:
    """The selected tasks that can train their solver, prepared, each with its solver's deploys and their step params,
    ``agent``'s if it swaps the solver. The others are left out before they're checked, so what they alone need can't
    stop the rest."""
    with namespace_routing():
        kept = []
        for entry in planned_tasks(bundle, tasks=tasks, evals=evals, agent=agent):
            if why := _untrainable(entry.config):
                logger.warning("%s is left out: %s", entry.entry.name, why)
            else:
                kept.append(entry.entry.name)
        if not kept:
            raise ValueError("no task of the bundle has a solver to train: an agent a step prompts, deployed by a "
                             "deploy_agent step, that isn't also its judge")
        prepared = prepare_bundle(bundle, tasks=kept, sandbox=SANDBOX, agent=agent, solver_endpoint=True)
    swapped = solver_params(prepared.plan, prepared.materialization)
    return [(task, {step: swapped.get(entry.id, {}).get(step, {})
                    for step in solver_steps([step.to_dict() for step in task.steps])})
            for entry, task in prepared.tasks]


def solver_overrides(solver: dict[str, dict[str, Any]], endpoint: dict[str, str]) -> dict[str, Any]:
    """A run's ``user_overrides``: every deploy on local Docker, and ``endpoint``, a ``litellm_base_url`` and
    ``litellm_api_key``, on the solver's deploys alone."""
    return {**sandbox_overrides(SANDBOX),
            "step_params": {step: {**params, **endpoint} for step, params in solver.items()}}


def rewarded(task: Task, reward: str | Callable[[dict[str, float]], float] | None
             ) -> Callable[[dict[str, float]], float | None]:
    of, why = reward_of(task, reward)
    if of is None:
        raise ValueError(why)
    return of


def first_prompt(task: Task) -> str:
    return next((step.prompt for step in task.steps if step.type == "prompt_agent"), "")


def _untrainable(config: list[dict[str, Any]]) -> str | None:
    """Why a task's rollouts can't train its solver alone, or None."""
    solver = set(solver_steps(config))
    steps = [build_step(step) for step in config]
    deploys = [step for step in steps if step.id in solver]
    if not deploys or not all(isinstance(step, DeployAgentTaskStep) for step in deploys):
        return "training needs the agent a step prompts deployed by a deploy_agent step"
    agents = {step.agent_name for step in deploys}
    if any(isinstance(step, RubricsVerifierTaskStep) and step.use_agent_judge and step.agent_name in agents
           for step in steps):
        return "its rubric judge is its solver, whose grading would be trained on"
    return None
