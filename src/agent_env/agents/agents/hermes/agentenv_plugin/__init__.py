"""A Hermes plugin, loaded by Hermes itself, that the agent's bridge drives each turn through.

`hermes agentenv turn` runs one turn in this process, the way `hermes -z` once did, rather than handing it to a Hermes
gateway: so the turn ends with this process, and a timeout or cancel that kills it stops the turn. Hermes' plugin
hooks write the turn's events, each as one JSON line, to the pipe AGENTENV_EVENTS_FD names, as they happen; the bridge
reads them as the turn's trajectory. Each event is a hook's name as ``type`` and the hook's fields the bridge keeps.

On an armed env clock the bridge runs this process on real time and passes ``--env-clock``: everything the process
starts follows the env's clock, and the dates Hermes shows the model are read from one of those (``_EnvDatetime``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import hermes_state_ids
import hermes_time

_EVENTS_FD = "AGENTENV_EVENTS_FD"
_write_lock = threading.Lock()
# The Hermes release whose names --env-clock replaces (hermes_time.datetime, hermes_state_ids.new_session_id and the
# copy agent.agent_init imports); a test holds it to the Dockerfile's HERMES_COMMIT, so a bump checks them again.
ENV_CLOCK_HERMES_COMMIT = "bf0203a794a07aac3ff764972ab9087baa4e4f4e"
# The env's time, in whole seconds since the epoch, as a command the model runs reads it.
ENV_TIME_COMMAND = ("date", "+%s")

# Each hook this plugin records, and the fields of its payload an event keeps.
_HOOKS = {
    "on_session_start": ("session_id", "model"),
    "pre_tool_call": ("session_id", "turn_id", "tool_call_id", "tool_name", "args"),
    "post_tool_call": ("session_id", "turn_id", "tool_call_id", "tool_name", "result", "status", "duration_ms",
                       "error_type", "error_message"),
    # ``response`` is Hermes' sanitized record of the model's answer: its assistant message, tool calls and usage.
    "post_api_request": ("session_id", "turn_id", "api_call_count", "model", "finish_reason", "response"),
    "api_request_error": ("session_id", "turn_id", "api_call_count", "model", "status_code", "reason", "retryable",
                          "error"),
    "post_llm_call": ("session_id", "turn_id", "assistant_response"),
    "on_session_end": ("session_id", "turn_id", "completed", "failed", "interrupted", "turn_exit_reason"),
}


class _EnvDatetime(datetime):
    """``datetime`` whose ``now`` is the env's time, read from a command this process starts: this process stays on
    real time, and what it starts follows the env's clock."""

    @classmethod
    def now(cls, tz=None) -> datetime:
        seconds = int(subprocess.run(ENV_TIME_COMMAND, capture_output=True, text=True, check=True).stdout)
        return datetime.fromtimestamp(seconds, tz)


def _show_the_envs_date() -> None:
    """Hermes' date line on the env's clock: the date it shows (``hermes_time.now``) and the start date it reads from
    the session id's stamp (``new_session_id``, and ``agent.agent_init``'s copy if that module has loaded; it isn't
    imported here, so Hermes loads its modules in its own order)."""
    hermes_time.datetime = _EnvDatetime
    minted = hermes_state_ids.new_session_id

    def new_session_id(now: datetime | None = None, **kwargs) -> str:
        return minted(_EnvDatetime.now(), **kwargs)

    hermes_state_ids.new_session_id = new_session_id
    if (agent_init := sys.modules.get("agent.agent_init")) is not None:
        agent_init.new_session_id = new_session_id


def _write(event: dict) -> None:
    descriptor = os.environ.get(_EVENTS_FD)
    if not descriptor:
        return
    line = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode()
    with _write_lock:
        os.write(int(descriptor), line)


def _recorder(hook: str, fields: tuple[str, ...]):
    def record(**payload) -> None:
        event = {"type": hook, "timestamp": int(time.time() * 1000)}
        event.update({name: payload[name] for name in fields if payload.get(name) is not None})
        _write(event)
    return record


def _setup(parser) -> None:
    commands = parser.add_subparsers(dest="agentenv_command", required=True)
    turn = commands.add_parser("turn", help="Run one turn and exit")
    turn.add_argument("--prompt-file", required=True, help="The turn's message, read from this file")
    turn.add_argument("--model")
    turn.add_argument("--provider")
    turn.add_argument("--resume", help="The session the turn continues")
    turn.add_argument("--reasoning", help="The reasoning effort")
    turn.add_argument("--usage-file", help="Where to write the turn's usage report")
    turn.add_argument("--env-clock", action="store_true",
                      help="This process is on real time while the env's clock is armed: show the model the env's date")


def _handle(args) -> None:
    # The one-shot runner: the turn in this process, then Hermes' own cleanup and a hard exit.
    from hermes_cli.main import _run_and_exit_oneshot

    if args.env_clock:
        _show_the_envs_date()
    with open(args.prompt_file, encoding="utf-8") as prompt:
        message = prompt.read()
    _run_and_exit_oneshot(message, model=args.model, provider=args.provider, resume=args.resume,
                          reasoning=args.reasoning, usage_file=args.usage_file)


def register(ctx) -> None:
    ctx.register_cli_command("agentenv", "Run one agent-env turn in this process", _setup, _handle)
    for hook, fields in _HOOKS.items():
        ctx.register_hook(hook, _recorder(hook, fields))
