"""Finding a computer in an env, for the steps that reach inside one: its deploy record in a run, by its name in a multi
env or as the only computer there, and whether a stored env has it."""

from __future__ import annotations

from typing import Optional

from agentenv_protocol.computer import COMPUTER_EXEC

from agent_env.env.env import DeployedEnv, DeployedSandboxEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.task_step.context import TaskStepContext


def find_computer(context: TaskStepContext, env_id: str, computer: Optional[str]) -> DeployedSandboxEnv:
    """The deploy record of the computer in ``env_id`` that serves computer-exec/v1 ``run``: the env's own, or one of a
    multi env's computers' (``child_computer``)."""
    deployed = next((d for d in context.deployed_envs if d.env_id == env_id), None)
    if deployed is None:
        raise RuntimeError(f"env {env_id!r} isn't in context.deployed_envs")
    if computer is None and deployed.supports(COMPUTER_EXEC, "run"):
        return deployed
    return child_computer(deployed, computer)[1]


def child_computer(deployed: DeployedEnv, computer: Optional[str]) -> tuple[str, DeployedSandboxEnv]:
    """A multi env's computer in its deployment (``DeployedGatewayEnv.child_envs``), by its name and with its deploy
    record: the one ``computer`` names, else its only one serving computer-exec/v1 ``run``."""
    children = {name: DeployedEnv.from_dict(data)
                for name, data in (getattr(deployed, "child_envs", None) or {}).items()}
    if computer is not None:
        found = children.get(computer)
        if found is None:
            raise RuntimeError(f"env {deployed.env_id!r} has no computer named {computer!r} "
                               f"(it has {sorted(children)})")
        return computer, found
    computers = [(name, record) for name, record in children.items() if record.supports(COMPUTER_EXEC, "run")]
    if len(computers) != 1:
        raise RuntimeError(f"env {deployed.env_id!r} has {len(computers) or 'no'} computers serving {COMPUTER_EXEC} run"
                           + (f": name one as computer, of {sorted(children)}" if computers else ""))
    return computers[0]


def computer_problem(env: object, computer: Optional[str]) -> Optional[str]:
    if isinstance(env, ComputerEnv):
        return None if computer is None else f"env {env.id!r} is a computer env, so it takes no computer name"
    if not isinstance(env, MultiEnv):
        return f"env {getattr(env, 'id', '?')!r} is no computer env, nor a multi env with one"
    browser = env.website_browser().computer
    names = [computer_env.environment_name for computer_env in (*env.computer_envs, *([browser] if browser else []))]
    if computer is not None and computer not in names:
        return f"env {env.id!r} has no computer named {computer!r} (it has {names})"
    if computer is None and len(names) != 1:
        return (f"env {env.id!r} has {len(names) or 'no'} computers" +
                (f": name one as computer, of {names}" if names else ""))
    return None
