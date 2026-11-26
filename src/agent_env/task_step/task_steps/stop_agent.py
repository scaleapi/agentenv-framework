"""Stop an agent before the steps that read what it left, such as a verifier that runs tests in its container.

A task author places it after the agent's last turn, and has those steps depend on it (``agent-env task lint`` warns of
a verifier that reads an agent's container without one). In order, it:

1. cancels the A2A task of each turn the run sent the agent, so its server works on none;
2. ends the processes the agent started where it works, its container, or the sandbox when that is its container: those
   running as its server's user that started after the server did and descend from it, or were orphaned, as a shell's
   background job is when the shell exits; the server is what listens on the agent's A2A port there. Each takes
   SIGTERM, and those left after ``grace_s`` SIGKILL. The server keeps running, so a later step can still prompt it, as
   does everything that started before it and what that starts: a container's init, a task container's own services,
   the clock and egress programs of an image that runs them. So do the servers of the other agents in its container,
   and what they started; an orphan one of them left can't be told from this agent's, and ends too;
3. removes what is at each of ``clear_paths``, a result file the agent may have written, say.

Its report is ``context.metadata["stop_agent"][step_id]``: each task and whether it was cancelled, the server, each
process it ended with its command line and the signals it took, and each path as it was before it was removed (its
kind, and a file's size, mtime and sha256). A process still running after SIGKILL, or a path left in place, fails the
step once the report is written. An agent installed on its sandbox's host is refused, since the processes and files
there aren't its alone.
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
from datetime import UTC, datetime
from typing import Any, ClassVar, Optional

from agent_env.a2a_agent import protocol
from agent_env.a2a_agent.a2a_agent import DEFAULT_A2A_PORT
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_VM,
    build_sandbox_provider,
    get_agent_sandbox_provider,
)
from agent_env.task_step.context import DeployedAgent, StepContextError, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.sandbox_utils.sandbox_utils import find_agent_container

logger = logging.getLogger(__name__)

REPORT_KEY = "stop_agent"
DEFAULT_GRACE_S = 5
MAX_GRACE_S = 300
# How long each script may take beyond its grace period.
_EXEC_MARGIN_S = 60
_COMMAND_CHARS = 500
# How much of a script's stderr a failure quotes, and how many of the processes still running it names.
_STDERR_TAIL_CHARS = 500
_LEFT_NAMED = 5
_LEFT_COMMAND_CHARS = 80
# The processes a report lists, so an agent that left thousands can't swell the run's record; it counts them all.
_REPORTED_PROCESSES = 200

# How often the script looks for what SIGTERM left, how many rounds of SIGKILL it sends, and the pause after each.
_POLL_S = 0.1
_KILL_ROUNDS = 5
_KILL_PAUSE_S = 0.2

# Args: the server's port, how many polls to wait after SIGTERM, _POLL_S, _KILL_ROUNDS, _KILL_PAUSE_S, then the ports of
# the other agents in the container. Prints the server, then a line per signal sent and per process left, each its pid
# and command line. A port's server is the oldest process holding the socket listening on it, or, when those processes'
# descriptors can't be read (root without CAP_SYS_PTRACE reads only its own user's), the oldest process of the socket's
# user. A process is the agent's when it runs as the server's user, started after it, and is the server's descendant or
# an orphan whose ancestors below the init all started after it too (`ours`): a child of anything older, a service of
# the container's own say, isn't, nor is another agent's server or what that started, nor this script or what runs it.
# A record is read whole, as a process name may hold a newline, and a scan that fails fails the script rather than
# reading as no process left. PATH is the system's alone, so no folder the agent can write to is searched for the
# programs it runs.
_END_PROCESSES = r"""
PATH=/usr/sbin:/usr/bin:/sbin:/bin
port=$1 polls=$2 poll_s=$3 kill_rounds=$4 kill_pause_s=$5 me=$$
shift 5
info() {
  pid=$1 stat=
  while IFS= read -r line; do stat="$stat$line "; done 2>/dev/null < "/proc/$pid/stat" || return 1
  set -- ${stat##*") "}
  [ $# -ge 20 ] || return 1
  state=$1 parent=$2
  shift 19
  start=$1 uid=
  while read -r key value _; do
    if [ "$key" = Uid: ]; then uid=$value; break; fi
  done 2>/dev/null < "/proc/$pid/status"
  [ -n "$uid" ]
}
command_of() {
  tr '\000\011\012' '   ' 2>/dev/null < "/proc/$1/cmdline"
}
server_on() {
  hex=$(printf '%04X' "$1") inode= found= found_owner= why=
  for table in /proc/net/tcp /proc/net/tcp6; do
    [ -r "$table" ] || continue
    while read -r _ local _ st _ _ _ user _ ino _; do
      if [ "$st" = 0A ] && [ "${local##*:}" = "$hex" ]; then inode=$ino found_owner=$user; break; fi
    done < "$table"
    [ -n "$inode" ] && break
  done
  [ -n "$inode" ] || { why="nothing listens on port $1, so the agent's server isn't running"; return 1; }
  holders=$(ls -l /proc/[0-9]*/fd 2>/dev/null | while read -r line; do
    case $line in
      /proc/*/fd:) dir=${line#/proc/}; dir=${dir%/fd:} ;;
      *" -> socket:[$inode]") echo "$dir" ;;
    esac
  done)
  [ -n "$holders" ] || holders=$(for dir in /proc/[0-9]*; do
    info "${dir#/proc/}" && [ "$uid" = "$found_owner" ] && [ "$state" != Z ] && echo "${dir#/proc/}"
  done)
  for p in $holders; do
    info "$p" || continue
    if [ -z "$found" ] || [ "$start" -lt "$found_start" ]; then found=$p found_start=$start; fi
  done
  [ -n "$found" ] || { why="no process holds the socket listening on port $1"; return 1; }
}
chain=" $me "
q=$me
while info "$q" && [ "$parent" -gt 0 ]; do chain="$chain$parent "; q=$parent; done
server_on "$port" || { echo "$why" >&2; exit 3; }
server=$found since=$found_start owner=$found_owner
kept=" "
for other in "$@"; do
  server_on "$other" && kept="$kept$found "
done
ours() {
  case $chain$kept in *" $1 "*) return 1 ;; esac
  info "$1" || return 1
  while :; do
    q=$parent
    case $q in "$server") return 0 ;; "$me") return 1 ;; esac
    case $kept in *" $q "*) return 1 ;; esac
    [ "$q" -gt 1 ] || return 0
    info "$q" && [ "$start" -gt "$since" ] || return 1
  done
}
printf 'SERVER\t%s\t%s\n' "$server" "$(command_of "$server")"
candidates() {
  for dir in /proc/[0-9]*; do
    p=${dir#/proc/}
    info "$p" || continue
    [ "$state" != Z ] && [ "$uid" = "$owner" ] && [ "$start" -gt "$since" ] || continue
    ours "$p" && printf '%s\n' "$p"
  done
  return 0
}
scan() {
  listed=$(candidates) || { echo "the scan of the container's processes failed" >&2; exit 4; }
}
signal() {
  scan
  for p in $listed; do
    c=$(command_of "$p")
    kill -s "$1" "$p" 2>/dev/null && printf '%s\t%s\t%s\n' "$1" "$p" "$c"
  done
}
signal TERM
n=0
while [ "$n" -lt "$polls" ] && scan && [ -n "$listed" ]; do sleep "$poll_s"; n=$((n + 1)); done
n=0
while [ "$n" -lt "$kill_rounds" ] && scan && [ -n "$listed" ]; do signal KILL; sleep "$kill_pause_s"; n=$((n + 1)); done
scan
for p in $listed; do printf 'LEFT\t%s\t%s\n' "$p" "$(command_of "$p")"; done
"""

# Args: the paths. Prints a line per path, in order: its kind, whether it was removed, and a file's size, mtime and
# sha256 or a link's target.
_CLEAR_PATHS = r"""
PATH=/usr/sbin:/usr/bin:/sbin:/bin
for p in "$@"; do
  kind=missing size= mtime= sha= target= removed=
  if [ -L "$p" ]; then kind=link target=$(readlink -- "$p" | tr '\011\012' '  ')
  elif [ -f "$p" ]; then
    kind=file size=$(wc -c < "$p") mtime=$(stat -c %Y -- "$p" 2>/dev/null) sha=$(sha256sum < "$p" 2>/dev/null)
    sha=${sha%% *}
  elif [ -d "$p" ]; then kind=directory
  elif [ -e "$p" ]; then kind=other
  fi
  if [ "$kind" != missing ] && rm -rf -- "$p" && [ ! -e "$p" ] && [ ! -L "$p" ]; then removed=1; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$kind" "$removed" "$size" "$mtime" "$sha" "$target"
done
"""


class StopAgentTaskStep(TaskStep):
    """Cancel ``agent_name``'s turns, end the processes it started and remove ``clear_paths``."""

    type: ClassVar[str] = "stop_agent"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: Optional[int],
        agent_name: Optional[str] = None,
        clear_paths: Optional[list[str]] = None,
        grace_s: float = DEFAULT_GRACE_S,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
        self.clear_paths = list(clear_paths or [])
        self.grace_s = grace_s

    def config_problems(self) -> list[str]:
        problems = []
        if not isinstance(self.agent_name, str):
            problems.append(f"agent_name must be a string, not {self.agent_name!r}")
        for path in self.clear_paths:
            if not isinstance(path, str) or not posixpath.isabs(path):
                problems.append(f"clear_paths entry {path!r} isn't an absolute path")
            elif posixpath.normpath(path) == "/":
                problems.append(f"clear_paths entry {path!r} would remove the container's whole filesystem")
        if (isinstance(self.grace_s, bool) or not isinstance(self.grace_s, (int, float))
                or not 0 <= self.grace_s <= MAX_GRACE_S):
            problems.append(f"grace_s must be a number of seconds from 0 to {MAX_GRACE_S}, not {self.grace_s!r}")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "agent_name": self.agent_name, "clear_paths": self.clear_paths,
                "grace_s": self.grace_s}

    @classmethod
    def from_dict(cls, data: dict) -> StopAgentTaskStep:
        return cls(**cls._base_from_dict(data), agent_name=data.get("agent_name"),
                   clear_paths=data.get("clear_paths"), grace_s=data.get("grace_s", DEFAULT_GRACE_S))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        agent = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
        if agent is None:
            raise StepContextError(f"Agent '{self.agent_name}' not found in context.deployed_agents")
        if agent.on_host:
            raise RuntimeError(f"stop_agent: agent '{self.agent_name}' is installed on its sandbox's host, whose "
                               "processes and files aren't its alone; install it into a container to stop it")
        if not agent.sandbox_id:
            raise RuntimeError(f"stop_agent: agent '{self.agent_name}' runs outside agent-env's sandboxes, so its "
                               "processes can't be reached")
        report: dict[str, Any] = {"agent_name": self.agent_name,
                                  "stopped_at": datetime.now(UTC).isoformat(timespec="seconds")}
        context.metadata.setdefault(REPORT_KEY, {})[self.id] = report
        report["cancelled"] = await self._cancel(context, agent)
        sandbox, container = await self._place(agent)
        report["container"] = container

        def run(script: str, *args: str):
            prefix = ("sudo", "docker", "exec", "-u", "0", "-w", "/", container) if container else ()
            return asyncio.wait_for(sandbox.exec_with_output(*prefix, "/bin/sh", "-c", script, "stop_agent", *args),
                                    timeout=self.grace_s + _EXEC_MARGIN_S)

        exit_code, stdout, stderr = await run(_END_PROCESSES, str(agent.a2a_port or DEFAULT_A2A_PORT),
                                              str(round(self.grace_s / _POLL_S)), str(_POLL_S), str(_KILL_ROUNDS),
                                              str(_KILL_PAUSE_S), *_neighbours(context, agent))
        server, ended, left = _read_processes(stdout)
        report.update(server=server, processes=ended[:_REPORTED_PROCESSES], processes_ended=len(ended),
                      processes_left=left)
        if exit_code != 0:
            raise RuntimeError(f"stop_agent: couldn't end agent '{self.agent_name}''s processes (exit {exit_code}): "
                               f"{stderr.strip()[-_STDERR_TAIL_CHARS:]}")
        problems = [f"{len(left)} process(es) still ran after SIGKILL: " + ", ".join(
            f"{p['pid']} ({p['command'][:_LEFT_COMMAND_CHARS]})" for p in left[:_LEFT_NAMED])] if left else []
        if self.clear_paths:
            exit_code, stdout, stderr = await run(_CLEAR_PATHS, *self.clear_paths)
            report["cleared"] = cleared = _read_paths(self.clear_paths, stdout)
            if exit_code != 0 or len(cleared) != len(self.clear_paths):
                problems.append(f"couldn't clear the paths (exit {exit_code}): "
                                f"{stderr.strip()[-_STDERR_TAIL_CHARS:]}")
            if kept := [entry["path"] for entry in cleared if entry["kind"] != "missing" and not entry["removed"]]:
                problems.append(f"couldn't remove {kept}")
        logger.info(f"stop_agent: agent '{self.agent_name}': cancelled {len(report['cancelled'])} task(s), ended "
                    f"{len(ended)} process(es), cleared {len(self.clear_paths)} path(s)")
        if problems:
            raise RuntimeError(f"stop_agent: agent '{self.agent_name}': " + "; ".join(problems))
        return context

    async def _cancel(self, context: TaskStepContext, agent: DeployedAgent) -> list[dict[str, Any]]:
        """Each A2A task of the agent's turns in this run, cancelled, with the agent's answer when it didn't cancel:
        a turn that ended already is past cancelling."""
        tasks = list(dict.fromkeys(response.agent_session_id for response in context.prompt_responses
                                   if response.agent_name == self.agent_name and response.agent_session_id))
        url = agent.a2a_url or agent.api_url
        answers = await asyncio.gather(*(protocol.cancel_a2a_task(url, task) for task in tasks))
        return [{"task_id": task, "cancelled": why is None, **({"why_not": why} if why else {})}
                for task, why in zip(tasks, answers)]

    @staticmethod
    async def _place(agent: DeployedAgent):
        """The agent's sandbox, and the container in it the agent works in; None when the sandbox is its container."""
        provider = build_sandbox_provider(agent.sandbox_type) if agent.sandbox_type else get_agent_sandbox_provider()
        sandbox = await provider.get_sandbox(agent.sandbox_id)
        if sandbox.mode != SANDBOX_MODE_VM:
            return sandbox, None
        return sandbox, agent.installed_container or await find_agent_container(sandbox)


def _neighbours(context: TaskStepContext, agent: DeployedAgent) -> list[str]:
    """The ports of the other agents in ``agent``'s container, whose servers, and what those start, the stop spares."""
    return [str(other.a2a_port or DEFAULT_A2A_PORT) for other in context.deployed_agents
            if other is not agent and other.sandbox_id == agent.sandbox_id and not other.on_host
            and other.installed_container == agent.installed_container]


def _read_processes(stdout: str) -> tuple[Optional[dict], list[dict], list[dict]]:
    """The server, each process signalled with the signals it took, in order, and each left running."""
    server, ended, left = None, {}, []
    for line in stdout.splitlines():
        what, _, rest = line.partition("\t")
        pid, _, command = rest.partition("\t")
        if not pid.isdigit():
            continue
        entry = {"pid": int(pid), "command": command.strip()[:_COMMAND_CHARS]}
        if what == "SERVER":
            server = entry
        elif what in ("TERM", "KILL"):
            ended.setdefault(entry["pid"], {**entry, "signals": []})["signals"].append(what)
        elif what == "LEFT":
            left.append(entry)
    return server, list(ended.values()), left


def _read_paths(paths: list[str], stdout: str) -> list[dict[str, Any]]:
    """Each path's state before it was removed, and whether it was."""
    cleared = []
    for path, line in zip(paths, stdout.splitlines()):
        kind, removed, size, mtime, sha256, target = (line.split("\t") + [""] * 6)[:6]
        entry: dict[str, Any] = {"path": path, "kind": kind, "removed": removed == "1"}
        if kind == "file":
            entry.update(size=int(size) if size.strip().isdigit() else None, sha256=sha256 or None,
                         mtime=(datetime.fromtimestamp(int(mtime), UTC).isoformat() if mtime.isdigit() else None))
        elif kind == "link":
            entry["target"] = target.strip()
        cleared.append(entry)
    return cleared
