"""The env's time in Antigravity's model calls. The SDK's harness, `localharness`, is a static Go binary, which
agent-env's libfaketime can't reach, so it runs on real time wherever the image follows the env's clock, while the
commands it starts read the env's time. The model sees the harness's clock in two places: the line ``The current local
time is: <RFC 3339>.`` in each turn's user message, and the ``Created At:`` and ``Completed At:`` stamps that head a
command's result. While this agent follows an armed clock the turn's model relay puts the env's time in the line and
shifts the stamps by the env's offset from real time when it first sees them, so a command keeps its duration. When the
env's date moves on during a turn, past the date the model was last told, the next command result gets a note saying so
(``DATE_CHANGED``), as Claude Code tells its model. Each keeps what it was first given, recorded in ``Records`` beside
the conversations by the time and the message it is in, which the harness sends again as it stored it, so the earlier
turns a resumed conversation sends again keep theirs.
tst/integration/agent/antigravity_env_date_test.py checks the SDK's model calls hold the machine's date nowhere else."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_egress import clock_armed
from model_relay import Refusal

_STAMP = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+(?:Z|[+-][0-9]{2}:[0-9]{2})"
LOCAL_TIME = re.compile(rf"(The current local time is: )({_STAMP})(\.)")
# The harness's head of a command's result: when the command started and, once it has, when it ended.
RUN_TIMES = re.compile(rf"^((?:Created|Completed) At: )({_STAMP})$", re.MULTILINE)
# What a command's result gets when the env's date has changed since the model was last told it, in the harness's own
# metadata block, and worded so LOCAL_TIME never reads it as a turn's line.
DATE_CHANGED = ("\n\n<ADDITIONAL_METADATA>\nThe date has changed. The current local time is now: {}.\n"
                "</ADDITIONAL_METADATA>")
_DATE_TIMEOUT_S = 5
# The hex digits of a message's digest a key holds.
_KEY_DIGEST_CHARS = 16
_SHOWN_CHARS = 200


class Records:
    """A JSON object of records beside the conversations, in ``path``, written whole by a rename: the env time each
    turn's local-time line was given and each command result's env and real time when the relay first saw it, keyed by
    ``line_key``, or the images each turn's message carries."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        return self._read().get(key)

    def all(self) -> dict[str, Any]:
        return self._read()

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            times = self._read()
            times[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            staged = self.path.with_suffix(".tmp")
            staged.write_text(json.dumps(times))
            staged.replace(self.path)

    def merge(self, other: dict[str, Any]) -> None:
        """Add a snapshot's times, its own winning over none."""
        with self._lock:
            times = {**other, **self._read()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(times))

    def _read(self) -> dict[str, Any]:
        try:
            times = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return times if isinstance(times, dict) else {}


class Tracked:
    """A turn's view of ``Records``: the keys it read a record by, or wrote, so they can be filed under the turn's
    conversation."""

    def __init__(self, records: Records) -> None:
        self.records = records
        self.keys: set[str] = set()

    def get(self, key: str) -> Any:
        value = self.records.get(key)
        if value is not None:
            self.keys.add(key)
        return value

    def put(self, key: str, value: Any) -> None:
        self.records.put(key, value)
        self.keys.add(key)


def env_time_rewrite(times: Records | Tracked):
    """The relay's rewrite for a turn: while the agent follows an armed clock, every local-time line of a user message
    gets the env time recorded for it, and the last one, the turn's own, the env's time when the turn first sends it;
    every command result's stamps are shifted by the offset recorded for it, a result of the turn's own recording the
    env's offset now; and the turn's latest command result, sent last, gets a ``DATE_CHANGED`` note once the env's
    date is past the one the turn's line or an earlier note told the model, every recorded note going with its result
    again. A line or result of an earlier turn with none recorded is left as written: the agent followed no clock
    then. Unchanged otherwise; a call that isn't JSON, or whose env time can't be read, is refused, so the model is
    never told the machine's time instead."""

    given: set[str] = set()

    async def rewrite(body: bytes) -> bytes:
        if not clock_armed():
            return body
        try:
            request = json.loads(body)
        except ValueError as exc:
            raise Refusal("the model call isn't JSON this agent can read, so the model can't be told the env's time") \
                from exc
        if not isinstance(request, dict):
            return body
        messages = [message for message in request.get("messages") or [] if isinstance(message, dict)]
        turn = max((index for index, message in enumerate(messages) if message.get("role") == "user"), default=-1)
        lines = [(holder, key) for message in messages if message.get("role") == "user"
                 for holder, key in _texts(message) if LOCAL_TIME.search(holder[key])]
        tools = [(index, holder, key, holder[key], str(message.get("tool_call_id"))) for index, message
                 in enumerate(messages) if message.get("role") == "tool" for holder, key in _texts(message)]
        runs = [(holder, key, index > turn) for index, holder, key, text, _ in tools if RUN_TIMES.search(text)]
        if not lines and not runs:
            return body
        now = _Now()
        if lines:
            latest = lines[-1][0][lines[-1][1]]
            current = line_key(latest, LOCAL_TIME.findall(latest)[-1][1])
            if current not in given:
                times.put(current, (await now())[0])
                given.add(current)
        for holder, key in lines:
            text = holder[key]
            holder[key] = LOCAL_TIME.sub(lambda match: _placed(match, line_key(text, match.group(2)), times), text)
        for holder, key, own in runs:
            text = holder[key]
            run = line_key(text, "run")
            if times.get(run) is None and own:
                times.put(run, " ".join(await now()))
            if (recorded := times.get(run)) is not None:
                holder[key] = RUN_TIMES.sub(lambda match: match.group(1) + _shifted(match.group(2), recorded), text)
        if lines:
            await _note_date_changes(tools, turn, len(messages), times.get(current), times, now)
        # Escaped, so a lone surrogate an MCP server's tool sent, which UTF-8 can't encode, goes as it came.
        return json.dumps(request, separators=(",", ":")).encode()

    return rewrite


async def _note_date_changes(tools: list, turn: int, count: int, told: str | None, times: Records | Tracked,
                             now: _Now) -> None:
    """Every command result with a recorded ``DATE_CHANGED`` note gets it again, and the turn's own result sent last a
    new one when the env's date is past ``told``, the turn's env time, or the last note of the turn's. A note is the
    result's by the model's id for its call, which the harness sends again as it stored it, and its text, so a later
    result with the same text never takes an earlier one's."""
    for index, holder, key, original, call in tools:
        if (noted := times.get(_note_key(call, original))) is not None:
            holder[key] += DATE_CHANGED.format(noted)
            told = noted if index > turn else told
    if not tools or told is None:
        return
    index, holder, key, original, call = tools[-1]
    if index != count - 1 or index <= turn or times.get(_note_key(call, original)) is not None:
        return
    env = (await now())[0]
    if env[:10] != told[:10]:
        times.put(_note_key(call, original), env)
        holder[key] += DATE_CHANGED.format(env)


def _note_key(call: str, text: str) -> str:
    return f"note#{call}#{message_digest(text)}"


def line_key(text: str, real: str) -> str:
    """A local-time line's or a command result's key: its time and the message text it is in, so two turns that start
    in one second keep their own env times."""
    return f"{real}#{message_digest(text)}"


def message_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:_KEY_DIGEST_CHARS]


def _placed(match: re.Match, line: str, times: Records | Tracked) -> str:
    env = times.get(line)
    return match.group(0) if env is None else f"{match.group(1)}{env}{match.group(3)}"


def _shifted(stamp: str, recorded: str) -> str:
    """``stamp``, a real time, at the env's offset from real time that ``recorded`` holds, in the env's zone."""
    env, real = recorded.split(" ")
    shifted = datetime.fromisoformat(env) + (datetime.fromisoformat(stamp) - datetime.fromisoformat(real))
    text = shifted.isoformat(timespec="seconds")
    return text[:-6] + "Z" if text.endswith("+00:00") else text


class _Now:
    """The env's time and the real time, read once a call needs them and kept for the rest of it."""

    def __init__(self) -> None:
        self._read: tuple[str, str] | None = None

    async def __call__(self) -> tuple[str, str]:
        if self._read is None:
            env = await _env_now()
            self._read = (env, _real_now())
        return self._read


def _real_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _texts(message: dict):
    """Each text of a message, as (holder, key), so it can be replaced in place."""
    content = message.get("content")
    if isinstance(content, str):
        yield message, "content"
    elif isinstance(content, list):
        yield from ((part, "text") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))


async def _env_now() -> str:
    """The env's time now in its zone, as the harness writes its own: RFC 3339 to the second, UTC as ``Z``. It comes
    from ``date``: this server runs on real time, but what it starts reads the env's clock, as the model's commands
    do."""
    def unread(reason: str) -> Refusal:
        return Refusal(f"the env's time couldn't be read for the model: {reason}")

    try:
        code, out, err = await _read_date()
    except TimeoutError:
        raise unread(f"date gave no answer within {_DATE_TIMEOUT_S} s") from None
    except OSError as exc:
        raise unread(f"{type(exc).__name__}: {exc}") from exc
    if code != 0:
        raise unread(f"date exited {code}: {err.decode(errors='replace').strip()[:_SHOWN_CHARS]}")
    stamp = out.decode(errors="replace").strip()
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{2}:[0-9]{2}", stamp):
        raise unread(f"date printed {stamp[:_SHOWN_CHARS]!r}")
    return stamp[:-6] + "Z" if stamp.endswith(("+00:00", "-00:00")) else stamp


async def _read_date() -> tuple[int, bytes, bytes]:
    """``date``'s exit code and output. Past the timeout its process group is killed, which reaches anything it left
    holding its output, even after it exited itself."""
    environ = {name: value for name, value in os.environ.items() if name != "AGENTENV_REAL_TIME"}
    proc = await asyncio.create_subprocess_exec("date", "+%Y-%m-%dT%H:%M:%S%:z", stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE, env=environ, start_new_session=True)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=_DATE_TIMEOUT_S)
    except BaseException:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), _DATE_TIMEOUT_S)
        raise
    return proc.returncode, out, err
