"""One turn of the Claude Code CLI: a ``claude -p`` run, its stream-json output parsed into a result, resumed when a
transient API failure cuts it short. Session threading (new vs resume) is the caller's."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

from kit import in_thread_to_its_end

logger = logging.getLogger(__name__)

# The CLI's own state: settings, skills and session transcripts. The image fixes it, so it doesn't follow the uid.
CLAUDE_STATE_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def _workspace() -> Path:
    """Where the CLI runs and the model's files land: AGENT_WORKSPACE_DIR, else this process's working directory (the
    image's /app, or an installed agent's container's), except the filesystem root, which is never one."""
    if configured := os.environ.get("AGENT_WORKSPACE_DIR"):
        return Path(configured)
    cwd = Path.cwd()
    return Path("/app") if cwd == Path(cwd.anchor) else cwd


def project_dir(workspace: Path) -> Path:
    """Where the CLI keeps the transcripts of sessions run in ``workspace``: a directory named for its real path, as
    the CLI's working directory has it, each character but a letter or digit as -."""
    return CLAUDE_STATE_DIR / "projects" / re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(workspace))


WORKSPACE_DIR = _workspace()
SESSION_JSONL_DIR = project_dir(WORKSPACE_DIR)

# The model aliases the CLI resolves for itself, pinned to the run's model on a gateway that may serve no other.
_MODEL_ALIASES = (
    "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL",
)
# What agent-env gives an agent for [model.headers]: a JSON object of header names to values.
MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"


def session_jsonl_path(session_id: str) -> Path:
    return SESSION_JSONL_DIR / f"{session_id}.jsonl"


def _session_record_to_stream_event(record: dict) -> dict | None:
    """One persisted session record in stream-json shape: assistant messages and tool results, which a trajectory
    needs, and nothing of the session's bookkeeping or compaction summaries, which would repeat its history."""
    if record.get("isCompactSummary") or record.get("isVisibleInTranscriptOnly"):
        return None
    record_type = record.get("type")
    message = record.get("message")
    if record_type == "assistant":
        if not isinstance(message, dict):
            return None
    elif record_type == "user":
        # stream-json replays no user prompt, only the user-shaped tool results.
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list) or not any(
                isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
            return None
    else:
        return None
    event: dict = {"type": record_type, "message": message}
    field_map = {"sessionId": "session_id", "requestId": "request_id", "toolUseResult": "tool_use_result",
                 "uuid": "uuid", "timestamp": "timestamp"}
    for source, target in field_map.items():
        if source in record:
            event[target] = record[source]
    return event


def read_partial_session_trajectory(session_id: str) -> tuple[list[dict], int]:
    """The session's trajectory so far, and how many records it has decoded, which only grows. A torn last line,
    as the CLI appends, is skipped until the next read. Raises FileNotFoundError until the CLI writes the file."""
    data = session_jsonl_path(session_id).read_bytes()
    trajectory: list[dict] = []
    session_lines = 0
    for raw_line in data.splitlines():
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        session_lines += 1
        event = _session_record_to_stream_event(_redact_values(record) if _is_api_error(record) else record)
        if event is not None:
            trajectory.append(event)
    return trajectory, session_lines


@dataclass
class ClaudeResult:
    response: str
    tool_call_count: int
    session_id: str | None
    trajectory: list[dict] = field(default_factory=list)
    error: str | None = None
    # The CLI's own account of a failure, from its terminal `result` event or else stderr.
    error_detail: str | None = None
    # The terminal `result` event's output, validated against a requested JSON Schema.
    structured_output: dict | None = None
    # How many `claude -p` runs the turn took; more than one means it was resumed.
    attempts: int = 1


def _attribution_headers(project_id: str | None, task_id: str | None) -> dict[str, str]:
    """LiteLLM's spend attribution headers for a run's project and task."""
    if project_id:
        tags = [f"projectId:{project_id}", *([f"taskId:{task_id}"] if task_id else [])]
        return {"x-litellm-customer-id": project_id, "x-litellm-tags": ",".join(tags)}
    return {"x-litellm-tags": f"taskId:{task_id}"} if task_id else {}


def _model_headers(raw: str | None) -> dict[str, str]:
    """[model.headers], from agent-env's JSON object; none when it isn't one."""
    if not raw:
        return {}
    try:
        headers = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"ignoring {MODEL_HEADERS_ENV}, which isn't JSON")
        return {}
    if not isinstance(headers, dict):
        logger.warning(f"ignoring {MODEL_HEADERS_ENV}, which isn't a JSON object")
        return {}
    return {str(name): str(value) for name, value in headers.items()}


def cli_env(environ: Mapping[str, str], *, model: str | None = None, project_id: str | None = None,
            task_id: str | None = None, max_thinking_tokens: int | None = None) -> dict[str, str]:
    """The CLI's environment: this process's, pointed at the endpoint agent-env configured (its key as a bearer
    token, so an API key too would conflict), with the endpoint's headers and the run's attribution, and the model's
    aliases pinned to ``model`` on that endpoint. A variable already set is kept, LITELLM_* too, so a script the model
    runs can call the endpoint."""
    env = dict(environ)
    if not env.get("ANTHROPIC_AUTH_TOKEN") and env.get("LITELLM_API_KEY"):
        env["ANTHROPIC_AUTH_TOKEN"] = env["LITELLM_API_KEY"]
    if not env.get("ANTHROPIC_BASE_URL") and env.get("LITELLM_BASE_URL"):
        env["ANTHROPIC_BASE_URL"] = env["LITELLM_BASE_URL"]
    if env.get("ANTHROPIC_AUTH_TOKEN"):
        env["ANTHROPIC_API_KEY"] = ""
    if model and env.get("ANTHROPIC_BASE_URL"):
        for alias in _MODEL_ALIASES:
            env.setdefault(alias, model)
    headers = _with_attribution(_model_headers(env.get(MODEL_HEADERS_ENV)), _attribution_headers(project_id, task_id))
    if headers:
        env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(f"{name}: {value}" for name, value in headers.items())
    if max_thinking_tokens:
        env["MAX_THINKING_TOKENS"] = str(max_thinking_tokens)
    # Older CLIs prepend a changing attribution block to the system prompt on custom base URLs, which defeats the
    # provider's prefix cache.
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
    return env


_DEFAULT_API_ERROR_RETRIES = 2
_MIN_RETRY_BUDGET_S = 120
# agent-env polls with the timeout it posts, so a resume that spent the budget to zero would leave it no answer.
_DEADLINE_HEADROOM_S = 30
_RETRY_BACKOFF_S = (5, 15)

# For a CLI that reports no `terminal_reason`, or one this doesn't know: transient API failures, by their text.
_RESUMABLE_API_ERROR_MARKERS = (
    "operation timed out",
    "socket connection was closed",
    "overloaded",
    "internal server error",
    "bad gateway",
)

# A real user turn, kept in the session transcript; the stream-json trajectory never shows it.
_RESUME_PROMPT = (
    "Your previous turn was interrupted by a transient API error before it finished. "
    "The conversation above is intact. Continue the original request from where you "
    "stopped, without redoing work that is already complete, and finish it."
)


def _configured_api_error_retries() -> int:
    raw = os.environ.get("CLAUDE_API_ERROR_RETRIES", "").strip()
    if not raw:
        return _DEFAULT_API_ERROR_RETRIES
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(f"ignoring non-integer CLAUDE_API_ERROR_RETRIES={raw!r}")
        return _DEFAULT_API_ERROR_RETRIES


# The CLI has filed a stalled stream under both; an unknown reason falls through to the markers.
_RESUMABLE_TERMINAL_REASONS = ("completed", "api_error")
# The 4xx a resume may clear: LiteLLM answers its own timeout with 408.
_RETRYABLE_CLIENT_ERRORS = (408, 425)


def _status(value: object) -> int | None:
    """An HTTP status the CLI reports, as a number or a string."""
    if isinstance(value, str):
        value = value.strip()[:3]
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _is_client_error(status: object) -> bool:
    """Whether the CLI saw a 4xx that a resume would hit again. 429 counts: the CLI has already backed off."""
    code = _status(status)
    return code is not None and 400 <= code < 500 and code not in _RETRYABLE_CLIENT_ERRORS


def _terminal_result(trajectory: list[dict]) -> dict | None:
    return next((event for event in reversed(trajectory) if event.get("type") == "result"), None)


def _resumable_api_failure(result: ClaudeResult) -> str | None:
    """The CLI's error text when the turn died on an upstream failure a resume may clear, else None."""
    event = _terminal_result(result.trajectory)
    if result.error is None or event is None or not event.get("is_error"):
        return None
    text = event.get("result")
    if not isinstance(text, str) or _is_client_error(event.get("api_error_status")):
        return None
    if event.get("terminal_reason") in _RESUMABLE_TERMINAL_REASONS:
        return text
    return text if any(marker in text.lower() for marker in _RESUMABLE_API_ERROR_MARKERS) else None


def _endpoint_hint(trajectory: list[dict]) -> str | None:
    """What a 404 from the model endpoint likely means."""
    event = _terminal_result(trajectory)
    if event is None or _status(event.get("api_error_status")) != 404:
        return None
    base = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("LITELLM_BASE_URL") or "the model endpoint"
    return (f"{base} answered 404: Claude Code needs an endpoint serving Anthropic's Messages API "
            "(<base>/v1/messages), such as a LiteLLM gateway, that knows the model")


def _ends_in_result(trajectory: list[dict]) -> bool:
    return bool(trajectory) and trajectory[-1].get("type") == "result"


def _without_terminal_result(trajectory: list[dict]) -> list[dict]:
    """The attempt's events less its last `result`: a resumed turn is one turn, and readers take the first `result`
    as its answer."""
    for index in range(len(trajectory) - 1, -1, -1):
        if trajectory[index].get("type") == "result":
            return trajectory[:index] + trajectory[index + 1:]
    return list(trajectory)


# Bounded so the whole message fits the agent's 1000-character error message.
_ERROR_DETAIL_LIMIT = 600
_FIRST_FAILURE_LIMIT = 200
# A key in the CLI's error text: a LiteLLM or Google key, or a bearer token. LiteLLM's masked `sk-...abcd` is kept.
_SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}|\bAIza[0-9A-Za-z_-]{35}|Bearer\s+[A-Za-z0-9._~+/=-]+")
# Node echoes a rejected header's whole value, quotes and all.
_HEADER_VALUE_PATTERN = re.compile(r"(Header '[^']*' has invalid value: ')(?:.*'|.*)", re.DOTALL)
_LINE_BREAK_PATTERN = re.compile(r"\s*\n\s*")
_STDERR_NOISE = re.compile(r"\s+at ")


def _redact(text: str) -> str:
    return _SECRET_PATTERN.sub("[redacted]", _HEADER_VALUE_PATTERN.sub(r"\1[redacted]'", text))


def _is_api_error(record: dict) -> bool:
    """A record of a failed call to the endpoint, whose text can echo its key: the stream's error `result` and
    `assistant` events, and the transcript's error messages."""
    return bool(record.get("is_error") or record.get("is_api_error_message") or record.get("isApiErrorMessage"))


def _redact_values(value: object) -> object:
    """``value`` with every string in it redacted."""
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, list):
        return [_redact_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_values(item) for key, item in value.items()}
    return value


def redact_session_records(jsonl: bytes) -> bytes:
    """A session transcript with its API error records redacted, and every other line as it was."""
    lines = jsonl.split(b"\n")
    for index, line in enumerate(lines):
        if b"isApiErrorMessage" not in line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(record, dict) and _is_api_error(record):
            lines[index] = json.dumps(_redact_values(record), ensure_ascii=False, separators=(",", ":")).encode()
    return b"\n".join(lines)


def scrub_session_transcript(session_id: str | None) -> None:
    """Redact the API error records an exited CLI left in the session's transcript, which trajectories and snapshots
    read."""
    if not session_id:
        return
    path = session_jsonl_path(session_id)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return
    scrubbed = redact_session_records(data)
    if scrubbed == data:
        return
    staged = path.with_name(f".{path.name}.scrubbed")
    staged.write_bytes(scrubbed)
    os.chmod(staged, stat.S_IMODE(path.stat().st_mode))
    os.replace(staged, path)


def _one_line(text: str) -> str:
    return _LINE_BREAK_PATTERN.sub(" ", text).strip()


def _cli_error_text(trajectory: list[dict]) -> str | None:
    """The CLI's own account of a failed turn: its terminal `result` event's text, or its `errors`."""
    event = _terminal_result(trajectory)
    if event is None or not event.get("is_error"):
        return None
    text = event.get("result")
    if isinstance(text, str) and text.strip():
        return _one_line(text)
    errors = event.get("errors")
    if isinstance(errors, list):
        return _one_line("; ".join(e for e in errors if isinstance(e, str) and e)) or None
    return None


def _stderr_tail(stderr: str) -> str | None:
    """stderr less stack frames, for a CLI that emitted no `result` event."""
    lines = [line.strip() for line in _redact(stderr).splitlines() if line.strip() and not _STDERR_NOISE.match(line)]
    return " | ".join(lines)[-_ERROR_DETAIL_LIMIT:] or None


def _failure_message(head: str, detail: str | None, attempts: int = 1, first_failure: str | None = None) -> str:
    """What ended the turn, the CLI's own words, then how a resumed turn began: what matters most survives a
    caller's truncation."""
    message = head if attempts == 1 else f"{head} on attempt {attempts}"
    if detail:
        message += f": {_redact(detail)[:_ERROR_DETAIL_LIMIT]}"
    if first_failure and first_failure != detail:
        message += f" (attempt 1: {_redact(first_failure)[:_FIRST_FAILURE_LIMIT]})"
    return message


async def run_claude(
    prompt: str,
    session_id: str,
    resume: bool,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    mcp_servers: dict[str, dict] | None = None,
    system_prompt: str | None = None,
    output_format: dict | None = None,
    timeout_s: int = 600,
    project_id: str | None = None,
    task_id: str | None = None,
    role: str | None = None,
    effort: str | None = None,
    max_turns: int | None = None,
    max_thinking_tokens: int | None = None,
    event_callback: Callable[[dict], None] | None = None,
) -> ClaudeResult:
    """Run one turn, resuming the CLI session when a transient upstream failure cuts it short; a resume spends what
    is left of ``timeout_s``, so the turn never outlives its caller's deadline. ``CLAUDE_API_ERROR_RETRIES=0`` turns
    resumes off. ``event_callback`` gets each stdout event as the CLI prints it; an attempt's trailing `result` is
    held until it's known not to be resumed."""
    deadline = time.monotonic() + timeout_s
    retries_left = _configured_api_error_retries()
    spawn = partial(
        _spawn_claude, model=model, allowed_tools=allowed_tools, mcp_servers=mcp_servers,
        system_prompt=system_prompt, output_format=output_format, project_id=project_id, task_id=task_id,
        role=role, effort=effort, max_turns=max_turns, max_thinking_tokens=max_thinking_tokens,
        event_callback=event_callback,
    )
    attempt_prompt, attempt_resume, attempt_session = prompt, resume, session_id
    trajectory: list[dict] = []
    tool_calls = 0
    attempts = 0
    first_failure: str | None = None

    while True:
        attempts += 1
        attempt_timeout_s = (timeout_s if attempts == 1
                             else max(int(deadline - time.monotonic()) - _DEADLINE_HEADROOM_S, 1))
        # Each scrub ends before the turn does, cancelled or not: a later turn on the context mustn't start while it
        # can still replace the transcript that turn appends to.
        try:
            result = await spawn(prompt=attempt_prompt, session_id=attempt_session, resume=attempt_resume,
                                 timeout_s=attempt_timeout_s)
        finally:
            await in_thread_to_its_end(scrub_session_transcript, attempt_session)
        if result.session_id != attempt_session:
            await in_thread_to_its_end(scrub_session_transcript, result.session_id)
        tool_calls += result.tool_call_count

        failure = _resumable_api_failure(result)
        budget_left = deadline - time.monotonic()
        can_retry = (failure is not None and retries_left > 0 and budget_left >= _MIN_RETRY_BUDGET_S
                     and session_jsonl_path(result.session_id).exists())
        if not can_retry:
            trajectory.extend(result.trajectory)
            if event_callback is not None and _ends_in_result(result.trajectory):
                event_callback(result.trajectory[-1])
            if failure is not None:
                logger.warning(f"claude CLI turn cut short ({_redact(failure)!r}) and not resumed "
                               f"(retries_left={retries_left}, budget_left={int(budget_left)}s)")
            # The first attempt's `result` was dropped from the trajectory, so the error says how the turn began.
            error = result.error and _failure_message(result.error, result.error_detail, attempts, first_failure)
            return replace(result, response=_redact(result.response) if error else result.response,
                           tool_call_count=tool_calls, trajectory=trajectory, attempts=attempts, error=error)

        trajectory.extend(_without_terminal_result(result.trajectory))
        first_failure = first_failure or result.error_detail
        retries_left -= 1
        attempt_prompt, attempt_resume, attempt_session = _RESUME_PROMPT, True, result.session_id
        backoff = _RETRY_BACKOFF_S[min(attempts - 1, len(_RETRY_BACKOFF_S) - 1)]
        logger.warning(f"claude CLI turn cut short ({_redact(failure)!r}); resuming session {attempt_session} "
                       f"in {backoff}s with {int(budget_left)}s of the original {timeout_s}s budget left")
        await asyncio.sleep(backoff)


_PROCESS_CLEANUP_TIMEOUT_S = 5


def _kill_cli(proc) -> None:
    """Kill the CLI and everything under it: its process group, and, by parent, the tools it starts in sessions of
    their own (its shells), which no group kill reaches."""
    pid = getattr(proc, "pid", None)
    if pid is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return
    if proc.returncode is not None:
        return  # gone, and its pid may be another process's by now
    tree = [pid, *_descendants(pid)]
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    for member in tree:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(member, signal.SIGKILL)


async def _reap(proc) -> None:
    """Wait for the killed CLI, but not for a process that left its tree and still holds its pipes open."""
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), _PROCESS_CLEANUP_TIMEOUT_S)


def _descendants(pid: int) -> list[int]:
    """The processes under ``pid``, by parent, from /proc; none where there is no /proc."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc") if os.path.isdir("/proc") else ():
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as stat:
                parent = int(stat.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(parent, []).append(int(entry))
    found, pending = [], [pid]
    while pending:
        for child in children.get(pending.pop(), ()):
            found.append(child)
            pending.append(child)
    return found


# The CLIs running a turn now. Each runs in a session of its own, so the agent's shutdown kills them itself.
_RUNNING: set = set()


def kill_running_clis() -> int:
    """Kill every CLI running a turn, and what each started; how many. The agent's shutdown does: the CLIs, in sessions
    of their own, would outlive its server, as when a reinstall stops it mid-turn."""
    running = list(_RUNNING)
    for proc in running:
        _kill_cli(proc)
    return len(running)


async def _started(spawn):
    """The CLI ``spawn`` starts. A turn cancelled while it starts lets it finish starting, then kills it and what it
    started: asyncio would kill the CLI alone, and leave its tools running."""
    start = asyncio.ensure_future(spawn)
    try:
        return await asyncio.shield(start)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            proc = await start
            _kill_cli(proc)
            await _reap(proc)
        raise


_STDOUT_READ_BYTES = 64 * 1024
# How much of a line the CLI shouldn't have printed is logged.
_LOGGED_LINE_CHARS = 200


async def _stream_cli(proc, stdin_data: bytes, on_line: Callable[[str], None]) -> bytes:
    """``proc.communicate(stdin_data)``, handing ``on_line`` each stdout line as it arrives, split as
    ``splitlines`` splits the whole output; stderr once the CLI has exited."""

    async def feed_stdin() -> None:
        try:
            proc.stdin.write(stdin_data)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            proc.stdin.close()

    stdin_task = asyncio.create_task(feed_stdin())
    stderr_task = asyncio.create_task(proc.stderr.read())
    try:
        # A line's pieces, joined once its newline arrives, so a long line costs its length, not its square.
        pieces: list[bytes] = []
        while chunk := await proc.stdout.read(_STDOUT_READ_BYTES):
            *ends, rest = chunk.split(b"\n")
            for end in ends:
                pieces.append(end)
                for line in b"".join(pieces).decode(errors="replace").splitlines():
                    on_line(line)
                pieces = []
            pieces.append(rest)
        for line in b"".join(pieces).decode(errors="replace").splitlines():
            on_line(line)
        await stdin_task
        stderr = await stderr_task
        await proc.wait()
        return stderr
    finally:
        stdin_task.cancel()
        stderr_task.cancel()


# Behind a gateway the CLI gives a bare id of these a 200K window, and envs whose tool definitions alone pass that
# make it exit with "Autocompact is thrashing". Add a model only once the pinned CLI maps its [1m] form.
_NATIVE_1M_MODELS = frozenset({"claude-opus-5", "claude-opus-5-5", "claude-fable-5-1", "claude-opus-4-8",
                               "claude-opus-4-7"})


def _with_1m_window(model: str) -> str:
    return f"{model}[1m]" if model in _NATIVE_1M_MODELS else model


def _command(session_id: str, resume: bool, *, model: str | None, allowed_tools: list[str] | None,
             mcp_servers: dict[str, dict] | None, system_prompt: str | None, output_format: dict | None,
             role: str | None, effort: str | None, max_turns: int | None) -> list[str]:
    """The CLI's argv. The prompt goes on stdin, since one argument over 128 KiB fails exec."""
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions",
           "--setting-sources", "user"]
    cmd += ["--resume", session_id] if resume else ["--session-id", session_id]
    if model:
        cmd += ["--model", _with_1m_window(model)]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]
    if effort:
        cmd += ["--effort", effort]
    if max_turns:
        cmd += ["--max-turns", str(max_turns)]
    if output_format:
        # agent-env's {"type": "json_schema", "schema": {...}}; the flag takes the schema alone.
        if output_format.get("type") != "json_schema" or "schema" not in output_format:
            raise ValueError(f'output_format must be {{"type": "json_schema", "schema": {{...}}}}; '
                             f'got keys={list(output_format.keys())}')
        cmd += ["--json-schema", json.dumps(output_format["schema"])]
    if allowed_tools:
        cmd += ["--allowedTools", ",".join(allowed_tools)]
    if mcp_servers:
        def entry(info: dict) -> dict:
            server: dict = {"type": "http", "url": info["url"]}
            # The gateway filters an env's tools by the role header, which wins over a stored one.
            headers = {**(info.get("headers") or {}), **({"AgentEnv-Role": role} if role else {})}
            if headers:
                server["headers"] = headers
            return server

        cmd += ["--mcp-config", json.dumps({"mcpServers": {name: entry(info) for name, info in mcp_servers.items()}})]
    return cmd


async def _spawn_claude(
    prompt: str,
    session_id: str,
    resume: bool,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    mcp_servers: dict[str, dict] | None = None,
    system_prompt: str | None = None,
    output_format: dict | None = None,
    timeout_s: int = 600,
    project_id: str | None = None,
    task_id: str | None = None,
    role: str | None = None,
    effort: str | None = None,
    max_turns: int | None = None,
    max_thinking_tokens: int | None = None,
    event_callback: Callable[[dict], None] | None = None,
) -> ClaudeResult:
    cmd = _command(session_id, resume, model=model, allowed_tools=allowed_tools, mcp_servers=mcp_servers,
                   system_prompt=system_prompt, output_format=output_format, role=role, effort=effort,
                   max_turns=max_turns)
    logger.info(f"Running claude CLI (resume={resume}, session_id={session_id}, model={model})")
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    proc = await _started(asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=cli_env(os.environ, model=model, project_id=project_id, task_id=task_id,
                    max_thinking_tokens=max_thinking_tokens),
        cwd=WORKSPACE_DIR,
        start_new_session=True,
    ))
    trajectory: list[dict] = []
    tool_count = 0
    cli_session_id: str | None = None
    final_response: str | None = None
    structured_output: dict | None = None
    # A trailing `result` is run_claude's to report, since it knows whether the attempt is resumed.
    held_result: dict | None = None

    def on_line(line: str) -> None:
        nonlocal tool_count, cli_session_id, final_response, structured_output, held_result
        if not line.strip():
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"claude CLI emitted non-JSON line: {line[:_LOGGED_LINE_CHARS]}")
            return
        if not isinstance(event, dict):
            logger.warning(f"claude CLI emitted a non-object JSON line: {line[:_LOGGED_LINE_CHARS]}")
            return
        if _is_api_error(event):
            # A failure's text can echo the endpoint's key, and the event is kept and streamed like any other.
            event = _redact_values(event)
        trajectory.append(event)
        etype = event.get("type")
        if etype == "system" and event.get("subtype") == "init":
            cli_session_id = event.get("session_id", cli_session_id)
        elif etype == "assistant":
            # Only tool calls count: the answer is the terminal `result`, not the narration between calls.
            for block in (event.get("message") or {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_count += 1
        elif etype == "result":
            cli_session_id = event.get("session_id", cli_session_id)
            if event.get("subtype") == "success" and isinstance(event.get("result"), str):
                final_response = event["result"]
            if isinstance(event.get("structured_output"), dict):
                structured_output = event["structured_output"]
        if event_callback is None:
            return
        if held_result is not None:
            event_callback(held_result)
            held_result = None
        if etype == "result":
            held_result = event
        else:
            event_callback(event)

    _RUNNING.add(proc)
    try:
        stderr = await asyncio.wait_for(_stream_cli(proc, prompt.encode(), on_line), timeout=timeout_s)
    except asyncio.CancelledError:
        _kill_cli(proc)
        await _reap(proc)
        raise
    except asyncio.TimeoutError:
        _kill_cli(proc)
        await _reap(proc)
        return ClaudeResult(response="", tool_call_count=tool_count, session_id=session_id,
                            error=f"claude CLI timed out after {timeout_s}s")
    except Exception:
        # Reading failed, as when an event_callback raises: the CLI mustn't keep running.
        _kill_cli(proc)
        await _reap(proc)
        raise
    finally:
        _RUNNING.discard(proc)

    err: str | None = None
    detail: str | None = None
    if proc.returncode != 0:
        err = (f"claude CLI was killed by signal {-proc.returncode}" if proc.returncode < 0
               else f"claude CLI exited with code {proc.returncode}")
        detail = _cli_error_text(trajectory) or _stderr_tail(stderr.decode(errors="replace"))
        if hint := _endpoint_hint(trajectory):
            detail = f"{detail}; {hint}" if detail else hint
        logger.error(f"claude CLI failed: {_failure_message(err, detail)}")

    # A requested JSON Schema's output is the answer, as JSON text the caller loads.
    if structured_output is not None:
        final_response = json.dumps(structured_output)

    return ClaudeResult(response=final_response or "", tool_call_count=tool_count,
                        session_id=cli_session_id or session_id, trajectory=trajectory, error=err,
                        error_detail=detail, structured_output=structured_output)


_TAGS_HEADER = "x-litellm-tags"


def _with_attribution(headers: dict[str, str], attribution: dict[str, str]) -> dict[str, str]:
    """``headers`` with the run's ``attribution`` over them, in any case, except that the run's x-litellm-tags join
    theirs."""
    merged = dict(headers)
    for name, value in attribution.items():
        configured = next((merged.pop(own) for own in list(merged) if own.lower() == name), None)
        merged[name] = _join_tags(configured, value) if name == _TAGS_HEADER and configured else value
    return merged


def _join_tags(configured: str, run: str) -> str:
    """x-litellm-tags as configured, then each of the run's tags whose key (the part before ``:``) they don't name."""
    tags = [tag.strip() for tag in configured.split(",") if tag.strip()]
    keys = {tag.partition(":")[0] for tag in tags}
    return ",".join([*tags, *(tag for tag in run.split(",") if tag and tag.partition(":")[0] not in keys)])
