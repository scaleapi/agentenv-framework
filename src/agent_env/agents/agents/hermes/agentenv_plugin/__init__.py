"""A Hermes plugin, loaded by Hermes itself, that the agent's bridge drives each turn through.

`hermes agentenv turn` runs one turn in this process, the way `hermes -z` once did, rather than handing it to a Hermes
gateway: so the turn ends with this process, and a timeout or cancel that kills it stops the turn. Hermes' plugin
hooks write the turn's events, each as one JSON line, to the pipe AGENTENV_EVENTS_FD names, as they happen; the bridge
reads them as the turn's trajectory. Each event is a hook's name as ``type`` and the hook's fields the bridge keeps.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import threading
import time
from pathlib import Path

_EVENTS_FD = "AGENTENV_EVENTS_FD"
_write_lock = threading.Lock()
_MEDIA_TAG = re.compile(r"MEDIA:(/[^\s\"'\\]+)")


def _message(message) -> dict:
    """The assistant message Hermes normalized from the model's answer, as plain data and whole: its text, its
    reasoning and its tool calls, each call's arguments the JSON string the model wrote."""
    calls = [{"id": call.id, "name": call.name, "arguments": call.arguments}
             for call in getattr(message, "tool_calls", None) or ()]
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    shaped = {"content": getattr(message, "content", None), "reasoning": reasoning, "tool_calls": calls}
    return {name: value for name, value in shaped.items() if value}


def _answer(response) -> str | None:
    """An auxiliary call's answer text, from Hermes' sanitized record of its response."""
    message = response.get("assistant_message") if isinstance(response, dict) else None
    return message.get("content") if isinstance(message, dict) else None


def _media(payload: dict) -> list[dict] | None:
    """The images a tool's result names by ``MEDIA:<path>`` in Hermes' media cache, as a screenshot an MCP tool
    returned, read while the file is there: a route that takes no images hands the model that path alone."""
    result, home = payload.get("result"), os.environ.get("HERMES_HOME")
    if not isinstance(result, str) or not home:
        return None
    cache, media = Path(home, "cache").resolve(), []
    for tagged in dict.fromkeys(_MEDIA_TAG.findall(result)):
        path = Path(tagged).resolve()
        kind = mimetypes.guess_type(path.name)[0] or ""
        if not (kind.startswith("image/") and path.is_relative_to(cache)):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        media.append({"path": tagged, "media_type": kind, "data": base64.b64encode(data).decode()})
    return media or None


# Each hook this plugin records, and the fields of its payload an event keeps.
_HOOKS = {
    "on_session_start": ("session_id", "model"),
    "pre_tool_call": ("session_id", "turn_id", "api_request_id", "tool_call_id", "tool_name", "args"),
    # With ``media``, the images its result names in Hermes' media cache (``_ADDED``).
    "post_tool_call": ("session_id", "turn_id", "api_request_id", "tool_call_id", "tool_name", "result", "status",
                       "duration_ms", "error_type", "error_message"),
    # ``response`` is Hermes' sanitized record of the model's answer, its strings cut at 8000 characters; ``message``
    # the answer whole (``assistant_message``). ``usage`` is the call's token counts, normalized (``prompt_tokens``
    # counts the cache), never truncated; ``started_at`` when the request went out, in epoch seconds.
    "post_api_request": ("session_id", "turn_id", "api_request_id", "api_call_count", "model", "response_model",
                         "started_at", "finish_reason", "response", "usage", "assistant_message"),
    "api_request_error": ("session_id", "turn_id", "api_request_id", "api_call_count", "retry_count", "model",
                          "started_at", "status_code", "reason", "retryable", "error"),
    # A title, compression or vision call, each attempt: its counts as post_api_request's, and its answer's text
    # (``response``).
    "post_auxiliary_call": ("session_id", "turn_id", "api_request_id", "aux_task", "model", "response_model",
                            "started_at", "usage", "response", "error"),
    "subagent_start": ("parent_session_id", "child_session_id", "child_subagent_id", "child_role", "child_goal"),
    "subagent_stop": ("parent_session_id", "child_session_id", "child_status", "child_summary"),
    "post_compaction": ("session_id", "old_session_id", "kind", "reason", "before_tokens", "after_tokens"),
    "post_llm_call": ("session_id", "turn_id", "assistant_response"),
    "on_session_end": ("session_id", "turn_id", "completed", "failed", "interrupted", "turn_exit_reason"),
}
# Payload fields an event keeps under a name and in a form of their own, by hook.
_SHAPED = {
    "post_api_request": {"assistant_message": ("message", _message)},
    "post_auxiliary_call": {"response": ("content", _answer)},
}
# Fields an event adds, each made from the hook's whole payload, by hook.
_ADDED = {
    "post_tool_call": {"media": _media},
}


def _write(event: dict) -> None:
    descriptor = os.environ.get(_EVENTS_FD)
    if not descriptor:
        return
    line = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode()
    with _write_lock:
        os.write(int(descriptor), line)


def _recorder(hook: str, fields: tuple[str, ...]):
    shaped, added = _SHAPED.get(hook, {}), _ADDED.get(hook, {})

    def record(**payload) -> None:
        event = {"type": hook, "timestamp": int(time.time() * 1000)}
        for name in fields:
            value = payload.get(name)
            if name in shaped and value is not None:
                name, value = shaped[name][0], shaped[name][1](value)
            if value is not None:
                event[name] = value
        for name, make in added.items():
            if (value := make(payload)) is not None:
                event[name] = value
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
