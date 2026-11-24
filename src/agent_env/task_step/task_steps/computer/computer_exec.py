"""computer_exec: a command run inside a deployed computer as its session user, through its driver's computer-exec/v1
``run``, a control route the agent can't call, or with ``scope: "system"`` as root, through the computer's sandbox, as a
task sets a computer up: a program run to its end, its exit status and output recorded, or with ``background`` an app
started and left running."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import posixpath
import shlex
import time
from typing import ClassVar, Optional, Union

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import COMPUTER, COMPUTER_EXEC

from agent_env.entity.refs import EntityRef
from agent_env.env.env import DeployedSandboxEnv
from agent_env.env.envs.computer.utils import inside
from agent_env.env.envs.computer.utils.sites import RootChannel
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.preflight import stored_env
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.computer.lookup import computer_problem, find_computer

logger = logging.getLogger(__name__)

#: Where the step records what each run did, by step id.
RECORDS_KEY = "computer_exec"
#: How long a command may run when the step doesn't say.
DEFAULT_TIMEOUT_S = 120.0
#: The most of each stream the record keeps, and the most of stderr a failure says: the end of it.
RECORDED_CHARS = 4000
FAILURE_CHARS = 300
#: How long a ``background`` app is watched after it starts: one that exits non-zero within it fails the step.
START_WINDOW_S = 2.0
#: How much longer than its start window a ``background`` start may take, for the shell and the stderr it reads.
START_MARGIN_S = 3.0
#: The most of a failed start's stderr the launcher keeps: the end of it.
START_STDERR_BYTES = 4096
#: How often the launcher checks a ``background`` start, and how long it waits for a failed start's stderr.
START_POLL_S = 0.1
START_STDERR_WAIT_S = 1.0
#: Who a command runs as: the session user, through the driver, or for ``system``, root, through the sandbox.
SCOPES = ("session", "system")
#: How long the session user's home takes to find, which a ``system`` command runs in.
HOME_TIMEOUT_S = 30.0
#: How much longer than ``timeout_s`` a ``system`` command's exec may take before the step stops waiting for it.
SYSTEM_MARGIN_S = 30.0
#: How long ``timeout`` gives a ``system`` command after its stop signal before it kills it.
SYSTEM_KILL_AFTER_S = 5
#: What ``timeout`` exits with when it stopped the command, and when it had to kill it (128 + SIGKILL).
TIMED_OUT_EXIT = 124
KILLED_EXIT = 137
#: The most of each stream a ``system`` command keeps in the computer, the end of it, so a command that prints without
#: end can't fill the task runner's memory; the record keeps less still.
SYSTEM_OUTPUT_BYTES = 65536
#: How a ``system`` command runs, given that cap and then the command: the end of each stream kept and base64'd, so
#: whatever bytes it printed decode, and the command's exit status passed on as its own.
_CAPPED = """cap=$1; shift
d=$(mktemp -d) || exit 125
{ { "$@"; echo $? >"$d/rc"; } 2>&1 >&3 3>&- | tail -c "$cap" >"$d/err"; } 3>&1 | tail -c "$cap" | base64 -w0
base64 -w0 "$d/err" >&2
rc=$(cat "$d/rc" 2>/dev/null)
rm -rf "$d"
exit "${rc:-125}"
"""


def background_script() -> str:
    """How a ``background`` command starts, as ``sh -c`` runs it with the command as its arguments: its program found
    and executable, else it fails as the shell does with 127; then started detached from the run's pipes, its stdout
    discarded and the end of its stderr kept, and watched for ``START_WINDOW_S``. One that exits non-zero in that window
    fails with its exit status and its stderr, which also catches a string command (whose program is ``sh``) naming a
    program that isn't there; one that exits 0, as ``xdg-open`` and single-instance apps do once they hand off, or is
    still running, has started. Its folder goes either way: an app still running keeps the stderr file it writes open,
    so removing the folder takes nothing from it."""
    window = max(1, round(START_WINDOW_S / START_POLL_S))
    stderr_wait = max(1, round(START_STDERR_WAIT_S / START_POLL_S))
    return (
        'case "$1" in */*) [ -f "$1" ] && [ -x "$1" ] ;; *) command -v "$1" >/dev/null ;; esac'
        ' || { echo "computer_exec: $1: not found or not executable" >&2; exit 127; }; '
        'd=$(mktemp -d) || exit 1; '
        f'( {{ "$@" 2>&1 >/dev/null; echo $? > "$d/rc"; }} | tail -c {START_STDERR_BYTES} > "$d/err" ) '
        '</dev/null >/dev/null 2>&1 & '
        f'n=0; while [ ! -s "$d/rc" ] && [ $n -lt {window} ]; do sleep {START_POLL_S:g}; n=$((n+1)); done; '
        '[ -s "$d/rc" ] || { rm -rf "$d"; exit 0; }; '
        'rc=$(cat "$d/rc"); [ "$rc" = 0 ] && { rm -rf "$d"; exit 0; }; '
        f'n=0; while [ ! -s "$d/err" ] && [ $n -lt {stderr_wait} ]; do sleep {START_POLL_S:g}; n=$((n+1)); done; '
        'echo "computer_exec: $* exited $rc right after starting" >&2; cat "$d/err" >&2; rm -rf "$d"; exit "$rc"')


Command = Union[list[str], str]


class ComputerExecTaskStep(TaskStep):
    """Run ``command`` inside the computer of the env ``env_id`` as its session user, in ``cwd`` (relative to the home,
    the home by default), with ``env`` added to its environment, for at most ``timeout_s``. ``command`` is a program and
    its arguments, or a string run by ``sh -c``. For a multi env, ``computer_name`` names the computer by its
    environment name there (its key in the deploy's ``child_envs``, not its env id), and may be left out when the env
    has one.

    With ``scope: "system"`` it runs as root instead, through the computer's sandbox rather than its driver, in the
    same folder and with the session user's ``HOME``; a computer booted from a disk has no such route.

    The run is recorded in ``context.metadata["computer_exec"][step_id]``: its exit code, the end of its stdout and
    stderr, and whether it timed out or was cut. A run that exits non-zero or times out fails the step, unless
    ``check`` is false. With ``background``, the command is started and left running, its output discarded, as an app
    launched for the agent is: the step watches it for ``START_WINDOW_S`` and fails as a run does when its program
    isn't there to start, or when it exits non-zero in that window, with the end of its stderr; an exit of 0 counts as
    started, since ``xdg-open`` and single-instance apps hand off to another process and exit."""

    type: ClassVar[str] = "computer_exec"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(self, id: str, version: Optional[int], env_id: str, command: Command,
                 computer_name: Optional[str] = None, cwd: Optional[str] = None, env: Optional[dict[str, str]] = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S, background: bool = False, check: bool = True,
                 scope: str = "session", depends_on: Optional[list[TaskStepDependency]] = None,
                 fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id
        self.command = command
        self.computer_name = computer_name
        self.cwd = cwd
        self.env = env
        self.timeout_s = timeout_s
        self.background = background
        self.check = check
        self.scope = scope

    def config_problems(self) -> list[str]:
        problems = []
        if not isinstance(self.env_id, str) or not self.env_id:
            problems.append(f"step {self.id!r}: env_id must be a non-empty string")
        if isinstance(self.command, str):
            if not self.command.strip():
                problems.append(f"step {self.id!r}: command must not be blank")
        elif not (isinstance(self.command, list) and self.command and all(isinstance(a, str) for a in self.command)):
            problems.append(f"step {self.id!r}: command must be a non-empty list of strings, or a string run by sh -c")
        if self.computer_name is not None and (not isinstance(self.computer_name, str) or not self.computer_name):
            problems.append(f"step {self.id!r}: computer_name, when given, must be a non-empty string")
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
        if (self.background is True and isinstance(self.timeout_s, (int, float))
                and self.timeout_s < (least := START_WINDOW_S + START_MARGIN_S)):
            problems.append(f"step {self.id!r}: a background start is watched for {START_WINDOW_S:g} s, so its "
                            f"timeout_s must be at least {least:g}")
        if self.scope not in SCOPES:
            problems.append(f"step {self.id!r}: scope is 'session', the session user's, or 'system', root's, "
                            f"not {self.scope!r}")
        elif self.scope == "system" and self.background is True:
            problems.append(f"step {self.id!r}: background runs as the session user; a 'system' command runs to its "
                            "end")
        return problems

    def preflight(self) -> list[str]:
        """The env this step names when it has no computer, or none of that name, or several when it names none."""
        problems = super().preflight()
        if not isinstance(self.env_id, str) or not self.env_id or (env := stored_env(self.env_id)) is None:
            return problems
        if problem := computer_problem(env, self.computer_name):
            problems.append(f"step {self.id!r}: {problem}")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "env_id": self.env_id, "command": self.command,
                "computer_name": self.computer_name, "cwd": self.cwd, "env": self.env, "timeout_s": self.timeout_s,
                "background": self.background, "check": self.check, "scope": self.scope}

    @classmethod
    def from_dict(cls, data: dict) -> ComputerExecTaskStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], command=data["command"],
                   computer_name=data.get("computer_name", data.get("computer")), cwd=data.get("cwd"),
                   env=data.get("env"), timeout_s=data.get("timeout_s", DEFAULT_TIMEOUT_S),
                   background=data.get("background", False), check=data.get("check", True),
                   scope=data.get("scope", "session"))

    def command_argv(self) -> list[str]:
        """The command as a program and its arguments: through ``sh -c`` when it's a string."""
        return ["sh", "-c", self.command] if isinstance(self.command, str) else list(self.command)

    def argv(self) -> list[str]:
        """What the driver runs: the command, started by ``background_script`` with ``background``."""
        argv = self.command_argv()
        return ["sh", "-c", background_script(), "computer_exec", *argv] if self.background else argv

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = find_computer(context, self.env_id, self.computer_name)
        if missing := [m for m in inside.missing(deployed) if m.startswith(COMPUTER_EXEC)]:
            raise RuntimeError(f"{self.type} {self.id!r}: env {self.env_id!r}'s computer serves no {missing[0]}")
        if self.timeout_s > (limit := inside.max_timeout_s(deployed)):
            raise ValueError(f"{self.type} {self.id!r}: timeout_s {self.timeout_s} is above the {limit} s the computer "
                             f"of env {self.env_id!r} lets a command run")
        if (self.background or isinstance(self.command, str)) and _os(deployed) == "windows":
            raise RuntimeError(f"{self.type} {self.id!r}: env {self.env_id!r}'s computer runs Windows, which has no sh "
                               "to run a string or start an app in the background: give the command as a program and "
                               "its arguments, without background")
        argv = self.argv()
        if self.scope == "system":
            answer = await self._as_root(deployed, argv)
        else:
            answer = await inside.run(deployed, argv, timeout_s=self.timeout_s, env=self.env, cwd=self.cwd)
        record = {"env_id": self.env_id, "computer_name": self.computer_name, "argv": argv, "scope": self.scope,
                  "background": self.background,
                  "exit_code": answer.get("exit_code"), "timed_out": bool(answer.get("timed_out")),
                  "truncated": bool(answer.get("truncated")),
                  "stdout": _tail(answer.get("stdout") or "", RECORDED_CHARS),
                  "stderr": _tail(answer.get("stderr") or "", RECORDED_CHARS)}
        context.metadata.setdefault(RECORDS_KEY, {})[self.id] = record
        shown = shlex.join(self.command_argv()) + (" as root" if self.scope == "system" else "")
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

    async def _as_root(self, deployed: DeployedSandboxEnv, argv: list[str]) -> dict:
        """``argv`` run as root through the computer's sandbox (``RootChannel``), in the folder and with the ``HOME`` a
        session run has, so ``cwd`` and ``~`` name the same files, under ``timeout``, its output capped in the computer
        (``_CAPPED``); answered as ``run`` answers. It timed out when ``timeout`` stopped or killed it past the
        deadline, which a command exiting with that status of its own, earlier, didn't."""
        if (deployed.metadata.get("computer") or {}).get("form") == "disk":
            raise RuntimeError(f"{self.type} {self.id!r}: a 'system' command runs through the computer's container, "
                               f"and env {self.env_id!r}'s computer boots from a disk, which core can't reach as root")
        home = (await inside.run(deployed, ["pwd"], timeout_s=HOME_TIMEOUT_S))["stdout"].strip()
        variables = [f"{name}={value}" for name, value in (self.env or {}).items()]
        command = ["sh", "-c", _CAPPED, "computer_exec", str(SYSTEM_OUTPUT_BYTES),
                   "timeout", "-k", str(SYSTEM_KILL_AFTER_S), f"{self.timeout_s:g}", "env", "-C",
                   posixpath.join(home, self.cwd) if self.cwd else home, f"HOME={home}", *variables, *argv]
        sandbox = await build_sandbox_provider(deployed.sandbox_type).get_sandbox(deployed.sandbox_id)
        started = time.monotonic()
        code, stdout, stderr = await asyncio.wait_for(RootChannel(sandbox).exec(command),
                                                      self.timeout_s + SYSTEM_MARGIN_S)
        timed_out = code in (TIMED_OUT_EXIT, KILLED_EXIT) and time.monotonic() - started >= self.timeout_s
        out, err = _unpacked(stdout), _unpacked(stderr)
        return {"exit_code": None if timed_out else code, "timed_out": timed_out,
                "truncated": max(len(out), len(err)) >= SYSTEM_OUTPUT_BYTES,
                "stdout": out.decode("utf-8", errors="replace"), "stderr": err.decode("utf-8", errors="replace")}


def _os(deployed: DeployedSandboxEnv) -> Optional[str]:
    """The computer's OS, as its card's computer/v1 platform says."""
    return (protocol_v1.extension_params(deployed.environment_card or {}, COMPUTER).get("platform") or {}).get("os")


def _unpacked(text: str) -> bytes:
    """A stream ``_CAPPED`` base64'd, as bytes; what the sandbox said in its place, as it said it."""
    try:
        return base64.b64decode(text.strip(), validate=True)
    except (binascii.Error, ValueError):
        return text.encode()


def _tail(text: str, chars: int) -> str:
    return text if len(text) <= chars else "…" + text[-chars:]
