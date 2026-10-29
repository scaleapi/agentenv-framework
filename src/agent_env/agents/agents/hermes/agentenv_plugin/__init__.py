"""A Hermes plugin, loaded by Hermes itself, that the agent's bridge drives each turn through.

`hermes agentenv turn` runs one turn in this process, the way `hermes -z` once did, rather than handing it to a Hermes
gateway: so the turn ends with this process, and a timeout or cancel that kills it stops the turn. Hermes' plugin
hooks write the turn's events, each as one JSON line, to the pipe AGENTENV_EVENTS_FD names, as they happen; the bridge
reads them as the turn's trajectory. Each event is a hook's name as ``type`` and the hook's fields the bridge keeps.
"""

from __future__ import annotations

import json
import os
import threading
import time

_EVENTS_FD = "AGENTENV_EVENTS_FD"
_write_lock = threading.Lock()

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


def _handle(args) -> None:
    # The one-shot runner: the turn in this process, then Hermes' own cleanup and a hard exit.
    from hermes_cli.main import _run_and_exit_oneshot

    with open(args.prompt_file, encoding="utf-8") as prompt:
        message = prompt.read()
    _run_and_exit_oneshot(message, model=args.model, provider=args.provider, resume=args.resume,
                          reasoning=args.reasoning, usage_file=args.usage_file)


def register(ctx) -> None:
    ctx.register_cli_command("agentenv", "Run one agent-env turn in this process", _setup, _handle)
    for hook, fields in _HOOKS.items():
        ctx.register_hook(hook, _recorder(hook, fields))
