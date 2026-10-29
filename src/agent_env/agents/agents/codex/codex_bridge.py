"""Subprocess wrapper around the OpenAI Codex CLI.

Runs `codex exec` (or `codex exec resume <thread_id>`) once and parses the
--json NDJSON output into a structured result. The caller is responsible for
threading sessions: pass thread_id=None on the first turn of a context, then
feed back result.session_id on subsequent turns.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from codex_config import MODEL_HEADERS_ENV, model_headers

logger = logging.getLogger(__name__)
WORKSPACE_DIR = os.environ.get("CODEX_WORKSPACE_DIR", "/app")

# `todo_list` is excluded: it's a plan update, not a tool invocation.
_TOOL_ITEM_TYPES = {"command_execution", "mcp_tool_call", "file_change", "web_search"}

# Item-level `error` items and top-level `error` events are excluded on purpose:
# codex uses both for advisory notices and transient retries on turns that then
# succeed. A real failure emits `turn.failed` and exits non-zero.
_FATAL_EVENT_TYPES = {"turn.failed", "thread.failed"}

# Codex CLI's per-thread rollout JSONL lives at
#   $CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ISO-ts>-<thread_uuid>.jsonl
# The filename's *suffix* is the thread_id; the prefix is a creation timestamp
# we don't know up-front. To find a thread's file we scan recursively for the
# uuid suffix. `codex exec resume <thread_id>` does the same (filesystem scan
# falls back when the SQLite index at $CODEX_HOME/state.db is stale), so this
# is consistent with the CLI's own lookup path.
SESSIONS_DIR = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex")) / "sessions"


def find_rollout_path(thread_id: str) -> Path | None:
    """Locate the JSONL rollout file for a given codex thread_id, or None."""
    if not SESSIONS_DIR.exists():
        return None
    matches = list(SESSIONS_DIR.rglob(f"rollout-*-{thread_id}.jsonl"))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            f"Multiple rollout files matched thread_id={thread_id}: {matches}; using newest"
        )
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


# LiteLLM attribution-probe state. The codex CLI's config.toml maps
# LITELLM_CUSTOMER_ID/LITELLM_TAGS env vars onto x-litellm-customer-id /
# x-litellm-tags headers at request time (via env_http_headers). We
# record the *semantic* project_id/task_id values per subprocess so
# /ext/attribution-probe matches the agent-env runtime verifier shape
# (wire-format header keys are an implementation detail of the call).
_last_attribution: dict[str, str | None] = {
    "project_id": None,
    "task_id": None,
    "last_seen_at_utc": None,
}
_attribution_lock = threading.Lock()


def _record_attribution(project_id: str | None, task_id: str | None) -> None:
    if not project_id and not task_id:
        return
    with _attribution_lock:
        _last_attribution["project_id"] = project_id or None
        _last_attribution["task_id"] = task_id or None
        _last_attribution["last_seen_at_utc"] = datetime.now(timezone.utc).isoformat()


def get_last_attribution_snapshot() -> dict[str, str | None]:
    with _attribution_lock:
        return dict(_last_attribution)


def cli_env(environ: Mapping[str, str], *, project_id: str | None = None,
            task_id: str | None = None) -> dict[str, str]:
    """The CLI's environment: this process's, with the endpoint's key also under the name codex's own precheck
    reads, and the run's attribution, which config.toml's env_http_headers sends as headers on each model call. The
    run's tags join the x-litellm-tags [model.headers] names."""
    env = dict(environ)
    if env.get("LITELLM_API_KEY") and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = env["LITELLM_API_KEY"]
    tags: list[str] = []
    if project_id:
        env["LITELLM_CUSTOMER_ID"] = project_id
        tags.append(f"projectId:{project_id}")
    if task_id:
        tags.append(f"taskId:{task_id}")
    configured = next((value for name, value in model_headers(env.get(MODEL_HEADERS_ENV)).items()
                       if name.lower() == _TAGS_HEADER), "")
    if joined := _join_tags(configured, ",".join(tags)):
        env["LITELLM_TAGS"] = joined
    return env


@dataclass
class CodexResult:
    response: str
    tool_call_count: int
    session_id: str | None
    trajectory: list[dict] = field(default_factory=list)
    error: str | None = None
    # Codex's own account of a failure (its turn.failed event, else stderr).
    # run_codex folds it into `error`.
    error_detail: str | None = None
    warnings: list[str] = field(default_factory=list)
    attempts: int = 1
    usage: dict = field(default_factory=dict)
    token_usage: dict | None = None


# Inactivity watchdog: if the CLI emits no output for this many seconds (e.g. a
# hung first model call through the LiteLLM proxy), kill it instead of waiting
# out the full timeout_s. Legit work streams events continuously, so this won't
# fire on normal tool use; tune via env for tasks with long silent tool calls.
_INACTIVITY_TIMEOUT_S = int(os.environ.get("AGENT_CLI_INACTIVITY_TIMEOUT_S", "300"))
# A deep-reasoning turn is silent for as long as the model thinks, and at the
# top efforts that is routinely longer than five minutes before the first
# token, so such turns were killed here as "stalled" mid-thought. The watchdog
# is for a hung proxy call, not a slow one, so those efforts get a longer leash. An explicit
# AGENT_CLI_INACTIVITY_TIMEOUT_S still wins: an operator who pinned the value
# for a task container knows their workload better than this table does.
_DEEP_EFFORT_INACTIVITY_TIMEOUT_S = 900
_DEEP_EFFORTS = frozenset({"xhigh", "max"})
_PROCESS_CLEANUP_TIMEOUT_S = 5
_TERMINAL_FRAGMENT_DRAIN_S = 0.1
_MAX_STDERR_BYTES = int(
    os.environ.get("CODEX_MAX_STDERR_BYTES", str(4 * 1024**2))
)
_MAX_EVENT_BYTES = int(
    os.environ.get("CODEX_MAX_EVENT_BYTES", str(16 * 1024**2))
)
_RETRY_BACKOFF_S = (5, 15)
_MIN_RETRY_BUDGET_S = 120


def inactivity_timeout_s(effort: str | None) -> int:
    """Seconds of CLI silence tolerated before a turn is declared stalled."""
    if os.environ.get("AGENT_CLI_INACTIVITY_TIMEOUT_S"):
        return _INACTIVITY_TIMEOUT_S
    if effort and effort.strip().lower() in _DEEP_EFFORTS:
        return max(_INACTIVITY_TIMEOUT_S, _DEEP_EFFORT_INACTIVITY_TIMEOUT_S)
    return _INACTIVITY_TIMEOUT_S
_DEADLINE_HEADROOM_S = 30
_RESUME_PROMPT = (
    "Your previous turn was interrupted by a transient API error before it "
    "finished. The conversation above is intact. Continue the original request "
    "from where you stopped, without redoing work that is already complete, "
    "and finish it."
)
_RETRYABLE_ERROR_MARKERS = (
    "request timed out",
    "operation timed out",
    "stream disconnected",
    "error while reading the server response",
    "connection reset",
    "connection closed during streaming",
    "empty or malformed response",
    "overloaded",
    "high demand",
    "service unavailable",
    "internal server error",
    "http 408",
    "http 425",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "http 529",
)
_NON_RETRYABLE_ERROR_MARKERS = (
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "quota",
    "credits",
    "spend cap",
    "usage limit",
    "prompt is too long",
    "context length",
    "context window",
    "invalid request",
    "bad request",
    "model not found",
    "thread not found",
    "budget exhausted",
    "unsupported",
    "configuration",
    "sandbox",
    "connection refused",
    "failedtoopensocket",
    "http 400",
    "http 401",
    "http 403",
    "http 404",
    "http 409",
    "http 422",
)
_USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


_RUNNING: set = set()


def kill_running_clis() -> int:
    """Kill every CLI still running a turn, with its process group; how many. The agent's shutdown does: the CLIs, in
    sessions of their own, would outlive its server, as when a reinstall stops it mid-turn."""
    running = [proc for proc in _RUNNING if proc.returncode is None]
    for proc in running:
        _kill_process_group(proc)
    return len(running)


def _kill_process_group(proc) -> None:
    pid = getattr(proc, "pid", None)
    if pid is not None:
        try:
            os.killpg(pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


def read_rollout_usage(thread_id: str | None) -> dict[str, int] | None:
    """Read Codex's latest cumulative token total from a native rollout."""
    if not thread_id:
        return None
    path = find_rollout_path(thread_id)
    if path is None:
        return None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != "event_msg":
            continue
        payload = record.get("payload") or {}
        if payload.get("type") != "token_count":
            continue
        info = payload.get("info") or {}
        total = info.get("total_token_usage") or {}
        if not isinstance(total, dict):
            continue
        usage = {
            key: int(total[key])
            for key in _USAGE_KEYS
            if isinstance(total.get(key), (int, float))
            and not isinstance(total.get(key), bool)
        }
        if usage:
            return usage
    return None


def _usage_delta(
    current: dict, baseline: dict[str, int] | None
) -> dict[str, int] | None:
    baseline = baseline or {}
    delta: dict[str, int] = {}
    for key in _USAGE_KEYS:
        value = current.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        prior = baseline.get(key, 0)
        if value < prior:
            return None
        delta[key] = int(value - prior)
    return delta or None


async def _read_with_watchdog(
    proc,
    timeout_s: float,
    stdin_data: bytes,
    event_callback: Callable[[dict], None] | None = None,
    inactivity_s: float | None = None,
) -> tuple[list[dict], bytes, bool, bool]:
    """Drain a subprocess under deadline and inactivity limits.

    `inactivity_s` defaults to the module-wide watchdog; callers that know the
    turn's effort pass `inactivity_timeout_s(effort)` instead.
    """
    if inactivity_s is None:
        inactivity_s = _INACTIVITY_TIMEOUT_S
    async def feed_stdin() -> None:
        try:
            proc.stdin.write(stdin_data or b"\n")
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # Early config rejection can close stdin.
            pass
        finally:
            proc.stdin.close()

    async def read_stderr() -> bytes:
        chunks: list[bytes] = []
        retained = 0
        truncated = False
        while True:
            chunk = await proc.stderr.read(65536)
            if not chunk:
                break
            available = _MAX_STDERR_BYTES - retained
            if available > 0:
                chunks.append(chunk[:available])
                retained += min(len(chunk), available)
            truncated = truncated or len(chunk) > max(available, 0)
        if truncated:
            chunks.append(b"\n[stderr truncated]")
        return b"".join(chunks)

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    stdin_task = asyncio.create_task(feed_stdin())
    stderr_task = asyncio.create_task(read_stderr())
    events: list[dict] = []
    pending = b""
    timed_out = stalled = False
    terminal_seen = False

    def emit_line(raw_line: bytes) -> None:
        nonlocal terminal_seen
        if not raw_line.strip():
            return
        try:
            event = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning(
                "codex CLI emitted non-JSON line: %s",
                raw_line[:200].decode(errors="replace"),
            )
            return
        if not isinstance(event, dict):
            return
        events.append(event)
        terminal_seen = terminal_seen or event.get("type") in {
            "turn.completed",
            "turn.failed",
        }
        if event_callback is not None:
            event_callback(event)

    def consume_chunk(chunk: bytes) -> None:
        nonlocal pending
        pending += chunk
        lines = pending.split(b"\n")
        pending = lines.pop()
        for raw_line in lines:
            if len(raw_line) > _MAX_EVENT_BYTES:
                raise RuntimeError("codex CLI event limit exceeded")
            emit_line(raw_line)
        if len(pending) > _MAX_EVENT_BYTES:
            raise RuntimeError("codex CLI event limit exceeded")

    async def drain_terminal_fragment() -> None:
        """Finish an already-started final NDJSON record, without a long wait."""
        nonlocal pending
        grace_deadline = time.monotonic() + _TERMINAL_FRAGMENT_DRAIN_S
        while pending and not terminal_seen:
            remaining = grace_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                chunk = await asyncio.wait_for(
                    proc.stdout.read(65536), timeout=remaining
                )
            except asyncio.TimeoutError:
                break
            if not chunk:
                break
            consume_chunk(chunk)
        if pending:
            final_record, pending = pending, b""
            emit_line(final_record)

    async def stop_process() -> None:
        _kill_process_group(proc)
        stdin_task.cancel()
        stderr_task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                proc.wait(), timeout=_PROCESS_CLEANUP_TIMEOUT_S
            )
        await asyncio.gather(
            stdin_task, stderr_task, return_exceptions=True
        )

    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                await drain_terminal_fragment()
                timed_out = not terminal_seen
                break
            try:
                chunk = await asyncio.wait_for(
                    proc.stdout.read(65536),
                    timeout=min(inactivity_s, remaining),
                )
            except asyncio.TimeoutError:
                if loop.time() >= deadline:
                    await drain_terminal_fragment()
                    timed_out = not terminal_seen
                else:
                    await drain_terminal_fragment()
                    stalled = not terminal_seen
                break
            if not chunk:
                break
            consume_chunk(chunk)
            if terminal_seen:
                break

        # Codex does not promise a trailing newline. Parse the final buffered
        # record even when the deadline/inactivity watchdog fired immediately
        # after reading it; a terminal event is authoritative over drain time.
        if pending:
            emit_line(pending)

        if terminal_seen and (timed_out or stalled):
            await stop_process()
            return events, b"", False, False

        if not timed_out and not stalled:
            remaining = deadline - loop.time()
            if remaining <= 0:
                if terminal_seen:
                    await stop_process()
                    return events, b"", False, False
                timed_out = True
            else:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(proc.wait(), stdin_task),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    if terminal_seen:
                        await stop_process()
                        return events, b"", False, False
                    timed_out = True

        if timed_out or stalled:
            await stop_process()
            return events, b"", timed_out, stalled

        remaining = deadline - loop.time()
        try:
            stderr = await asyncio.wait_for(
                stderr_task, timeout=max(0.001, remaining)
            )
        except asyncio.TimeoutError:
            await stop_process()
            return events, b"", not terminal_seen, False
        return events, stderr, False, False
    except BaseException:
        await stop_process()
        raise


def _configured_api_error_retries() -> int:
    try:
        return max(0, int(os.environ.get("CODEX_API_ERROR_RETRIES", "2")))
    except ValueError:
        return 2


def _error_status(value: object) -> int | None:
    """Find an HTTP-like status in Codex/LiteLLM's nested error payloads."""
    if isinstance(value, dict):
        for key in ("status", "status_code", "code"):
            raw = value.get(key)
            try:
                code = int(raw)
            except (TypeError, ValueError):
                continue
            if 400 <= code <= 599:
                return code
        for nested in value.values():
            code = _error_status(nested)
            if code is not None:
                return code
    elif isinstance(value, str):
        try:
            nested = json.loads(value)
        except json.JSONDecodeError:
            nested = None
        if nested is not None:
            code = _error_status(nested)
            if code is not None:
                return code
        match = re.search(
            r'(?:http|status|status_code|code)[^0-9]{0,12}([45]\d\d)',
            value,
            re.IGNORECASE,
        )
        if match:
            return int(match.group(1))
    return None


def _resumable_api_failure(result: CodexResult) -> str | None:
    """Return a transient terminal failure that can safely resume its thread."""
    if not result.error or not result.session_id:
        return None
    if find_rollout_path(result.session_id) is None:
        return None
    terminal = next(
        (
            event
            for event in reversed(result.trajectory)
            if event.get("type") == "turn.failed"
        ),
        None,
    )
    if terminal is None:
        return None
    terminal_error = terminal.get("error")
    message = (
        terminal_error.get("message")
        if isinstance(terminal_error, dict)
        else terminal_error
    )
    if not isinstance(message, str):
        return None
    lowered = message.lower()
    if any(marker in lowered for marker in _NON_RETRYABLE_ERROR_MARKERS):
        return None
    status = _error_status(terminal_error)
    if status is not None:
        if status in {408, 425, 429} or 500 <= status <= 599:
            return message
        return None
    if any(marker in lowered for marker in _RETRYABLE_ERROR_MARKERS):
        return message
    return None


# Bounded so the whole message, attempt-1 suffix included, stays well under
# a2a_server's 4000-character cap.
_ERROR_DETAIL_LIMIT = 600
_FIRST_FAILURE_LIMIT = 200
# A misconfigured key can surface verbatim in the CLI's error text: a LiteLLM
# (`sk-`) or Google (`AIza`) key, or a bearer token, which ends where RFC 6750's
# token characters do so the JSON around it survives. LiteLLM's own masked form
# (`sk-...abcd`) does not match.
_SECRET_PATTERN = re.compile(
    r"\bsk-[A-Za-z0-9_-]{8,}|\bAIza[0-9A-Za-z_-]{35}|Bearer\s+[A-Za-z0-9._~+/=-]+"
)
# Node's HTTP client echoes a rejected header's whole value, whatever its shape;
# it runs to the last quote (or the end), since the value may hold quotes itself.
_HEADER_VALUE_PATTERN = re.compile(r"(Header '[^']*' has invalid value: ')(?:.*'|.*)", re.DOTALL)
_LINE_BREAK_PATTERN = re.compile(r"\s*\n\s*")
# codex's tracing log records tool calls the model recovered from, never what
# ended the turn.
_STDERR_NOISE = re.compile(r"\d{4}-\d\d-\d\dT\S+Z\s+(?:TRACE|DEBUG|INFO|WARN|ERROR)\s")
# Codex appends a fresh id to every request's error, so attempts that failed the
# same way differ only here.
_REQUEST_ID_PATTERN = re.compile(r", (?:request id|cf-ray): \S+")


def _redact(text: str) -> str:
    return _SECRET_PATTERN.sub("[redacted]", _HEADER_VALUE_PATTERN.sub(r"\1[redacted]'", text))


def _one_line(text: str) -> str:
    return _LINE_BREAK_PATTERN.sub(" ", text).strip()


def _redacted_notice(notice: dict) -> str:
    return json.dumps(
        {key: _redact(value) if isinstance(value, str) else value
         for key, value in notice.items()}
    )


def _unwrap_error_body(message: str) -> str:
    """LiteLLM's own message out of a 4xx body, which codex forwards verbatim."""
    if not message.lstrip().startswith("{"):
        return message
    try:
        error = json.loads(message).get("error")
    except json.JSONDecodeError:
        return message
    if not isinstance(error, dict) or not isinstance(error.get("message"), str):
        return message
    inner = error["message"]
    # LiteLLM sends an absent type as the string "None".
    tags = [
        str(value) for value in (error.get("code"), error.get("type"))
        if value not in (None, "None") and str(value) not in inner
    ]
    return f"{inner} [{', '.join(tags)}]" if tags else inner


def _cli_error_text(trajectory: list[dict]) -> str | None:
    """Codex's own account of a failed turn, from its last turn.failed / thread.failed event."""
    for event in reversed(trajectory):
        if event.get("type") not in _FATAL_EVENT_TYPES:
            continue
        error = event.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        if isinstance(message, str) and message.strip():
            return _one_line(_unwrap_error_body(message))
        return None
    return None


def _stderr_tail(stderr: str) -> str | None:
    """stderr minus tracing, for a CLI that failed before emitting a fatal event."""
    lines = [
        line.strip() for line in _redact(stderr).splitlines()
        if line.strip() and not _STDERR_NOISE.match(line)
    ]
    return " | ".join(lines)[-_ERROR_DETAIL_LIMIT:] or None


def _trailing_notice(trajectory: list[dict]) -> str | None:
    """A top-level `error` notice that is the attempt's last event: the reconnect
    in flight when a stall or kill ended the turn. Item-level `error` items are
    advisories printed on every run, so they never count."""
    last = trajectory[-1] if trajectory else {}
    message = last.get("message") if last.get("type") == "error" else None
    return _one_line(message) if isinstance(message, str) and message.strip() else None


def _failure_message(
    head: str, detail: str | None, attempts: int = 1, first_failure: str | None = None
) -> str:
    """What ended the turn, then codex's own words, then how a resumed turn began.

    In that order so what matters most survives a caller's truncation.
    """
    message = head if attempts == 1 else f"{head} on attempt {attempts}"
    if detail:
        message += f": {_redact(detail)[:_ERROR_DETAIL_LIMIT]}"
    if first_failure and (
        _REQUEST_ID_PATTERN.sub("", first_failure) != _REQUEST_ID_PATTERN.sub("", detail or "")
    ):
        message += f" (attempt 1: {_redact(first_failure)[:_FIRST_FAILURE_LIMIT]})"
    return message


async def _run_codex_once(
    prompt: str,
    thread_id: str | None,
    model: str | None = None,
    effort: str | None = None,
    system_prompt: str | None = None,
    output_format: dict | None = None,
    timeout_s: float = 600,
    project_id: str | None = None,
    task_id: str | None = None,
    event_callback: Callable[[dict], None] | None = None,
    config_overrides: Sequence[str] = (),
) -> CodexResult:
    # JSON Schema: codex's `--output-schema` flag takes a FILE PATH (unlike
    # claude-code's inline-JSON `--json-schema`). Write the inner schema to a
    # per-invocation temp file. Canonical agent-env envelope is
    # {"type":"json_schema", "schema": {...}}; mirror its strict validation here.
    # Known caveat: codex silently ignores --output-schema when MCP servers or
    # tool calls are active (openai/codex#15451). For tool-using judges the
    # schema enforcement may not actually fire — the model still typically
    # produces JSON-shaped output because the prompt asks for it.
    schema_file: Path | None = None
    if output_format:
        if output_format.get("type") != "json_schema" or "schema" not in output_format:
            raise ValueError(
                f'output_format must be {{"type": "json_schema", "schema": {{...}}}}; '
                f'got keys={list(output_format.keys())}'
            )
        schema_file = Path(f"/tmp/codex_schema_{uuid.uuid4().hex}.json")
        schema_file.write_text(json.dumps(output_format["schema"]))

    cmd = ["codex", "exec", "--cd", WORKSPACE_DIR]
    if thread_id is None:
        cmd += ["-"]
    else:
        cmd += ["resume", thread_id, "-"]
    cmd += ["--json", "--dangerously-bypass-approvals-and-sandbox"]
    if model:
        cmd += ["-c", f'model="{model}"']
    if effort:
        cmd += ["-c", f'model_reasoning_effort="{effort}"']
    if system_prompt:
        cmd += [
            "-c",
            (
                "developer_instructions="
                f"{json.dumps(system_prompt, ensure_ascii=False)}"
            ),
        ]
    if schema_file is not None:
        cmd += ["--output-schema", str(schema_file)]
    for override in config_overrides:
        cmd += ["-c", override]

    inactivity_s = inactivity_timeout_s(effort)
    logger.info(
        f"Running codex CLI (thread_id={thread_id}, model={model}, effort={effort}, "
        f"inactivity_watchdog={inactivity_s}s)"
    )

    subprocess_env = cli_env(os.environ, project_id=project_id, task_id=task_id)
    if project_id or task_id:
        _record_attribution(project_id=project_id, task_id=task_id)
        logger.info(
            "codex CLI attribution: LITELLM_CUSTOMER_ID=%s LITELLM_TAGS=%s",
            subprocess_env.get("LITELLM_CUSTOMER_ID", "(unset)"),
            subprocess_env.get("LITELLM_TAGS", "(unset)"),
        )

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=subprocess_env,
            start_new_session=True,
        )
        _RUNNING.add(proc)
        trajectory, stderr, timed_out, stalled = await _read_with_watchdog(
            proc, timeout_s, prompt.encode(), event_callback, inactivity_s
        )

        final_message = ""
        fatal_seen = False
        warnings: list[str] = []
        tool_count = 0
        turn_completed = False
        usage: dict = {}
        captured_thread_id: str | None = thread_id

        for event in trajectory:
            etype = event.get("type")
            if etype == "thread.started":
                captured_thread_id = event.get("thread_id", captured_thread_id)
            elif etype == "turn.completed":
                turn_completed = True
                if isinstance(event.get("usage"), dict):
                    usage = dict(event["usage"])
            elif etype == "item.completed":
                item = event.get("item") or {}
                item_type = item.get("type")
                if item_type == "agent_message":
                    # Last message wins.
                    final_message = item.get("text", "")
                elif item_type == "error":
                    warnings.append(_redacted_notice(item))
                elif item_type in _TOOL_ITEM_TYPES:
                    tool_count += 1
            elif etype in _FATAL_EVENT_TYPES:
                fatal_seen = True
            elif etype == "error":
                warnings.append(_redacted_notice(event))

        err: str | None = None
        if timed_out or stalled:
            err = (
                f"codex CLI timed out after {timeout_s:.0f}s" if timed_out
                else f"codex CLI produced no output for {inactivity_s}s "
                     "(stalled — likely a hung model call through the proxy)"
            )
        elif fatal_seen or (
            proc.returncode != 0 and not turn_completed
        ):
            if proc.returncode is not None and proc.returncode > 0:
                err = f"codex CLI exited with code {proc.returncode}"
            elif proc.returncode is not None and proc.returncode < 0 and not fatal_seen:
                err = f"codex CLI was killed by signal {-proc.returncode}"
            else:
                # After turn.failed, a signal is the watchdog stopping a CLI
                # that lingered, not what ended the turn.
                err = "codex CLI turn failed"
        elif not turn_completed:
            err = "codex CLI exited without turn.completed"
        detail = err and (
            _cli_error_text(trajectory)
            or _stderr_tail(stderr.decode(errors="replace"))
            or _trailing_notice(trajectory)
        )
        if err:
            logger.error(f"codex CLI failed: {_failure_message(err, detail)}")
        if warnings:
            logger.info(
                "codex CLI emitted %d non-fatal notice(s) (turn_completed=%s): %s",
                len(warnings), turn_completed, "; ".join(warnings)[:500],
            )

        return CodexResult(
            response=final_message,
            tool_call_count=tool_count,
            session_id=captured_thread_id,
            trajectory=trajectory,
            error=err,
            error_detail=detail,
            warnings=warnings,
            usage=usage,
        )
    finally:
        _RUNNING.discard(proc)
        if schema_file is not None and schema_file.exists():
            try:
                schema_file.unlink()
            except OSError as e:
                logger.warning(f"Failed to clean up {schema_file}: {e}")


async def run_codex(
    prompt: str,
    thread_id: str | None,
    model: str | None = None,
    effort: str | None = None,
    system_prompt: str | None = None,
    output_format: dict | None = None,
    timeout_s: int = 600,
    project_id: str | None = None,
    task_id: str | None = None,
    event_callback: Callable[[dict], None] | None = None,
    config_overrides: Sequence[str] = (),
) -> CodexResult:
    """Run one logical turn with bounded transient recovery."""
    deadline = time.monotonic() + timeout_s
    usage_baseline = read_rollout_usage(thread_id)
    retries_left = _configured_api_error_retries()
    attempt_prompt = prompt
    attempt_thread = thread_id
    attempts = 0
    trajectory: list[dict] = []
    warnings: list[str] = []
    tool_call_count = 0
    first_failure: str | None = None

    while True:
        attempts += 1
        available = deadline - time.monotonic()
        remaining = (
            available - _DEADLINE_HEADROOM_S
            if attempts > 1 and available > _DEADLINE_HEADROOM_S + 1
            else available
        )
        result = await _run_codex_once(
            prompt=attempt_prompt,
            thread_id=attempt_thread,
            model=model,
            effort=effort,
            system_prompt=system_prompt,
            output_format=output_format,
            timeout_s=max(0.001, remaining),
            project_id=project_id,
            task_id=task_id,
            event_callback=event_callback,
            config_overrides=config_overrides,
        )
        failure = _resumable_api_failure(result)
        budget_left = deadline - time.monotonic()
        backoff = _RETRY_BACKOFF_S[
            min(attempts - 1, len(_RETRY_BACKOFF_S) - 1)
        ]
        can_retry = (
            failure is not None
            and retries_left > 0
            and budget_left
            >= _MIN_RETRY_BUDGET_S + _DEADLINE_HEADROOM_S + backoff
        )

        warnings.extend(result.warnings)
        tool_call_count += result.tool_call_count

        if not can_retry:
            trajectory.extend(result.trajectory)
            error = result.error and _failure_message(
                result.error, result.error_detail, attempts, first_failure
            )
            if error:
                result.response = _redact(result.response)
            result.trajectory = trajectory
            result.warnings = warnings
            result.tool_call_count = tool_call_count
            result.attempts = attempts
            result.error = error
            cumulative_usage = result.usage or (
                read_rollout_usage(result.session_id) or {}
            )
            if cumulative_usage:
                result.token_usage = _usage_delta(
                    cumulative_usage, usage_baseline
                )
                if result.token_usage is None:
                    warning = (
                        "codex cumulative token counters regressed; "
                        "incremental usage is unavailable"
                    )
                    result.warnings.append(warning)
            return result

        first_failure = first_failure or result.error_detail
        trajectory.extend(result.trajectory)
        retries_left -= 1
        attempt_prompt = _RESUME_PROMPT
        attempt_thread = result.session_id
        logger.warning(
            "codex CLI turn failed transiently; resuming thread %s in %ss "
            "with %.1fs remaining",
            attempt_thread,
            backoff,
            budget_left,
        )
        await asyncio.sleep(min(backoff, max(0, budget_left)))


_TAGS_HEADER = "x-litellm-tags"


def _join_tags(configured: str, run: str) -> str:
    """x-litellm-tags as configured, then each of the run's tags whose key (the part before ``:``) they don't name."""
    tags = [tag.strip() for tag in configured.split(",") if tag.strip()]
    keys = {tag.partition(":")[0] for tag in tags}
    return ",".join([*tags, *(tag for tag in run.split(",") if tag and tag.partition(":")[0] not in keys)])
