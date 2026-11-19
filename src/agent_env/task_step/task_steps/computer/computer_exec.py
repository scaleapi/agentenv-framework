"""computer_exec: a command run inside a deployed computer as its session user, through its driver's computer-exec/v1
``run``, a control route the agent can't call, as a task sets a computer up: a program run to its end, its exit status
and output recorded, or with ``background`` an app started and left running."""

from __future__ import annotations

import logging
import shlex
from typing import ClassVar, Optional, Union

from agentenv_protocol.computer import COMPUTER_EXEC

from agent_env.entity.refs import EntityRef
from agent_env.env.env import DeployedEnv, DeployedSandboxEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.utils import inside
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.preflight import stored_env
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

#: Where the step records what each run did, by step id.
RECORDS_KEY = "computer_exec"
#: How long a command may run when the step doesn't say.
DEFAULT_TIMEOUT_S = 120.0
#: The most of each stream the record keeps, and the most of stderr a failure says: the end of it.
RECORDED_CHARS = 4000
FAILURE_CHARS = 300
#: How a ``background`` command starts: its program found and executable, else it fails as the shell does with 127,
#: then detached from the run's pipes, so the run returns as the shell does.
_BACKGROUND = ('case "$1" in */*) [ -f "$1" ] && [ -x "$1" ] ;; *) command -v "$1" >/dev/null ;; esac'
               ' || { echo "computer_exec: $1: not found or not executable" >&2; exit 127; }; '
               '"$@" </dev/null >/dev/null 2>&1 &')

Command = Union[list[str], str]


class ComputerExecTaskStep(TaskStep):
    """Run ``command`` inside the computer of the env ``env_id`` as its session user, in ``cwd`` (relative to the home,
    the home by default), with ``env`` added to its environment, for at most ``timeout_s``. ``command`` is a program and
    its arguments, or a string run by ``sh -c``. For a multi env, ``computer`` names the computer, by its environment
    name, and may be left out when the env has one.

    The run is recorded in ``context.metadata["computer_exec"][step_id]``: its exit code, the end of its stdout and
    stderr, and whether it timed out or was cut. A run that exits non-zero or times out fails the step, unless
    ``check`` is false. With ``background``, the command is started and left running, its output discarded, as an app
    launched for the agent is: the step waits for the start alone, and fails as a run does when its program isn't
    there to start."""

    type: ClassVar[str] = "computer_exec"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, command: Command,
                 computer: Optional[str] = None, cwd: Optional[str] = None, env: Optional[dict[str, str]] = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S, background: bool = False, check: bool = True,
                 depends_on: Optional[list[TaskStepDependency]] = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.command = command
        self.computer = computer
        self.cwd = cwd
        self.env = env
        self.timeout_s = timeout_s
        self.background = background
        self.check = check

    def config_problems(self) -> list[str]:
        problems = []
        if not isinstance(self.env_id, str) or not self.env_id:
            problems.append(f"step {self.id!r}: env_id must be a non-empty string")
        if isinstance(self.command, str):
            if not self.command.strip():
                problems.append(f"step {self.id!r}: command must not be blank")
        elif not (isinstance(self.command, list) and self.command and all(isinstance(a, str) for a in self.command)):
            problems.append(f"step {self.id!r}: command must be a non-empty list of strings, or a string run by sh -c")
        if self.computer is not None and (not isinstance(self.computer, str) or not self.computer):
            problems.append(f"step {self.id!r}: computer, when given, must be a non-empty string")
        if self.cwd is not None and (not isinstance(self.cwd, str) or self.cwd.startswith("/")):
            problems.append(f"step {self.id!r}: cwd is a folder relative to the session user's home, not {self.cwd!r}")
        if self.env is not None and not (isinstance(self.env, dict) and all(
                isinstance(k, str) and k and isinstance(v, str) for k, v in self.env.items())):
            problems.append(f"step {self.id!r}: env maps variable names to string values")
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or self.timeout_s <= 0:
            problems.append(f"step {self.id!r}: timeout_s must be a number of seconds above 0")
        for name in ("background", "check"):
            if not isinstance(getattr(self, name), bool):
                problems.append(f"step {self.id!r}: {name} must be true or false")
        return problems

    def preflight(self) -> list[str]:
        """The env this step names when it has no computer, or none of that name, or several when it names none."""
        problems = super().preflight()
        if not isinstance(self.env_id, str) or not self.env_id or (env := stored_env(self.env_id)) is None:
            return problems
        if problem := computer_problem(env, self.computer):
            problems.append(f"step {self.id!r}: {problem}")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "env_id": self.env_id, "command": self.command, "computer": self.computer,
                "cwd": self.cwd, "env": self.env, "timeout_s": self.timeout_s, "background": self.background,
                "check": self.check}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerExecTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], command=data["command"],
                   computer=data.get("computer"), cwd=data.get("cwd"), env=data.get("env"),
                   timeout_s=data.get("timeout_s", DEFAULT_TIMEOUT_S), background=data.get("background", False),
                   check=data.get("check", True))

    def argv(self) -> list[str]:
        """What the driver runs: the command, through ``sh -c`` when it's a string, started detached with
        ``background``."""
        argv = ["sh", "-c", self.command] if isinstance(self.command, str) else list(self.command)
        return ["sh", "-c", _BACKGROUND, "computer_exec", *argv] if self.background else argv

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = find_computer(context, self.env_id, self.computer)
        if missing := [m for m in inside.missing(deployed) if m.startswith(COMPUTER_EXEC)]:
            raise RuntimeError(f"{self.type} {self.id!r}: env {self.env_id!r}'s computer serves no {missing[0]}")
        if self.timeout_s > (limit := inside.max_timeout_s(deployed)):
            raise ValueError(f"{self.type} {self.id!r}: timeout_s {self.timeout_s} is above the {limit} s the computer "
                             f"of env {self.env_id!r} lets a command run")
        argv = self.argv()
        answer = await inside.run(deployed, argv, timeout_s=self.timeout_s, env=self.env, cwd=self.cwd)
        record = {"env_id": self.env_id, "computer": self.computer, "argv": argv, "background": self.background,
                  "exit_code": answer.get("exit_code"), "timed_out": bool(answer.get("timed_out")),
                  "truncated": bool(answer.get("truncated")),
                  "stdout": _tail(answer.get("stdout") or "", RECORDED_CHARS),
                  "stderr": _tail(answer.get("stderr") or "", RECORDED_CHARS)}
        context.metadata.setdefault(RECORDS_KEY, {})[self.id] = record
        shown = shlex.join(argv)
        if record["timed_out"]:
            outcome = f"timed out after {self.timeout_s} s"
        elif record["exit_code"] != 0:
            outcome = f"exited {record['exit_code']}"
        else:
            logger.info(f"{self.id}: {'started' if self.background else 'ran'} {shown} in env {self.env_id!r}")
            return context
        message = f"{self.type} {self.id!r}: {shown} {outcome} in env {self.env_id!r}"
        if stderr := _tail(record["stderr"].strip(), FAILURE_CHARS):
            message += f": {stderr}"
        if self.check:
            raise RuntimeError(message)
        logger.warning(f"{message} (check is off)")
        return context


def find_computer(context: TaskStepContext, env_id: str, computer: Optional[str]) -> DeployedSandboxEnv:
    """The deploy record of the computer in ``env_id`` that serves computer-exec/v1 ``run``: the env's own, or one of a
    multi env's computers' (``DeployedGatewayEnv.child_envs``), the one ``computer`` names else its only one."""
    deployed = next((d for d in context.deployed_envs if d.env_id == env_id), None)
    if deployed is None:
        raise RuntimeError(f"env {env_id!r} isn't in context.deployed_envs")
    children = {name: DeployedEnv.from_dict(data) for name, data in (getattr(deployed, "child_envs", None) or {}).items()}
    if computer is not None:
        found = children.get(computer)
        if found is None:
            raise RuntimeError(f"env {env_id!r} has no computer named {computer!r} (it has {sorted(children)})")
        return found
    if deployed.supports(COMPUTER_EXEC, "run"):
        return deployed
    computers = [record for record in children.values() if record.supports(COMPUTER_EXEC, "run")]
    if len(computers) != 1:
        raise RuntimeError(f"env {env_id!r} has {len(computers) or 'no'} computers serving {COMPUTER_EXEC} run"
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


def _tail(text: str, chars: int) -> str:
    return text if len(text) <= chars else "…" + text[-chars:]
