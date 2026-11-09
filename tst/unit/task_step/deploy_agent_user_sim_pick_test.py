"""A run-wide agent pick (``user_overrides.a2a_agent_id``: ``task run --a2a-agent-id``, a hub run's agent pick) swaps the
tested agent, not the one the task names as its user simulator: deploy_agent and install_agent read the run's stored
task and leave that agent (a user_sim step's agent_name, else its name; a prompt_agent's user_agent_name) as it is
stored. The pick keeps today's version rule wherever it applies."""

from __future__ import annotations

import pytest

from agent_env.task import Task
from agent_env.task.store import register_task_instance
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import deploy_agent
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep, plays_a_user_sim
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep

pytestmark = pytest.mark.usefixtures("local_stores")

SOLVER = {"id": "deploy-solver", "type": "deploy_agent", "env_ids": [], "agent_name": "solver",
          "a2a_agent_id": "theirs", "a2a_agent_version": 4}
LINDA = {"id": "deploy-linda", "type": "deploy_agent", "env_ids": [], "agent_name": "linda",
         "a2a_agent_id": "sim", "a2a_agent_version": 2}
WHO = {"id": "who", "type": "user_sim", "name": "linda", "model": "m"}
LOOP = {"id": "loop", "type": "always_on", "prompt": "Help Linda.", "user_sim": "linda", "agent_name": "solver"}
HUMAN = {"id": "deploy-human", "type": "deploy_agent", "env_ids": [], "agent_name": "human_agent",
         "a2a_agent_id": "sim", "a2a_agent_version": 2}
ASK = {"id": "ask", "type": "prompt_agent", "prompt_id": "ask", "prompt": "hi", "agent_name": "solver",
       "max_conversation_turns": 3}


def _run_of(*steps: dict, overrides: dict | None = None) -> tuple[Task, TaskStepContext]:
    task = Task.from_dict({"id": "pick-task", "version": None, "steps": list(steps)})
    stored = Task.put(id=task.id, steps=task.steps)
    instance = register_task_instance(stored.id, stored.version, len(steps))
    ctx = TaskStepContext(instance_id=instance.instance_id,
                          metadata={"user_overrides": {"a2a_agent_id": "mine", **(overrides or {})}})
    return stored, ctx


def _ref(task: Task, step_id: str, ctx: TaskStepContext):
    """The agent ref of ``step_id``, a step of the stored ``task`` (read back from the store by the run)."""
    return next(s for s in task.steps if s.id == step_id)._resolve_agent_ref(ctx)


def _user_sim_task(monkeypatch, *steps: dict) -> tuple[list, TaskStepContext]:
    """A run of a task with user_sim and always_on steps (registered on the step's branch): its deploy steps, and the
    task's config as the run's stored task would give it."""
    monkeypatch.setattr(deploy_agent, "_run_task_config", lambda context: list(steps))
    built = [DeployAgentTaskStep.from_dict(s) if s["type"] == "deploy_agent" else
             InstallAgentTaskStep.from_dict(s) for s in steps if s["type"] in ("deploy_agent", "install_agent")]
    return built, TaskStepContext(instance_id="run-1", metadata={"user_overrides": {"a2a_agent_id": "mine"}})


def test_the_pick_swaps_the_tested_agent_and_the_user_sim_step_s_agent_keeps_its_own(monkeypatch):
    (solver, linda), ctx = _user_sim_task(monkeypatch, SOLVER, LINDA, WHO, LOOP)
    assert solver._resolve_agent_ref(ctx) == ("mine", 4)  # today's rule: the step's stored version
    assert linda._resolve_agent_ref(ctx) == ("sim", 2)


def test_a_prompt_agent_s_user_agent_keeps_its_own():
    task, ctx = _run_of(SOLVER, HUMAN, ASK, overrides={"a2a_agent_version": 9})
    assert _ref(task, "deploy-solver", ctx) == ("mine", 9)
    assert _ref(task, "deploy-human", ctx) == ("sim", 2)


def test_a_user_sim_agent_s_own_step_params_still_win(monkeypatch):
    (_, linda), ctx = _user_sim_task(monkeypatch, SOLVER, LINDA, WHO, LOOP)
    ctx.metadata["user_overrides"]["step_params"] = {"deploy-linda": {"a2a_agent_id": "other-sim"}}
    assert linda._resolve_agent_ref(ctx) == ("other-sim", None)


def test_a_user_sim_named_but_not_deployed_leaves_the_pick_on_the_rest(monkeypatch):
    (solver,), ctx = _user_sim_task(monkeypatch, SOLVER, WHO, LOOP)
    assert solver._resolve_agent_ref(ctx) == ("mine", 4)


def test_a_task_without_a_user_sim_keeps_the_pick_everywhere():
    judge = {**LINDA, "id": "deploy-judge", "agent_name": "judge"}
    task, ctx = _run_of(SOLVER, judge, {**ASK, "max_conversation_turns": 1})
    assert _ref(task, "deploy-solver", ctx) == ("mine", 4)
    assert _ref(task, "deploy-judge", ctx) == ("mine", 2)


def test_a_run_with_no_recorded_instance_keeps_the_pick_as_before():
    step = DeployAgentTaskStep.from_dict(LINDA)
    ctx = TaskStepContext(metadata={"user_overrides": {"a2a_agent_id": "mine"}})
    assert step._resolve_agent_ref(ctx) == ("mine", 2) and not plays_a_user_sim("linda", ctx)


def test_install_agent_follows_the_same_rule(monkeypatch):
    installed = {"id": "install-linda", "type": "install_agent", "agent_name": "linda", "a2a_agent_id": "sim",
                 "a2a_agent_version": 2, "sandbox_name": "world"}
    (solver, linda), ctx = _user_sim_task(monkeypatch, SOLVER, installed, WHO, LOOP)
    assert linda._agent_ref(ctx) == ("sim", 2) and solver._resolve_agent_ref(ctx) == ("mine", 4)
