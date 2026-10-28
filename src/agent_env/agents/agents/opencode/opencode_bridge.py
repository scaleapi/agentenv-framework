"""Subprocess wrapper around the OpenCode CLI (`opencode run`).

Runs one `opencode run --format json` invocation, parses its JSONL event
stream, and returns a structured result.

  - Headless `opencode run` per turn.
  - Per-context isolation via a per-context XDG_DATA_HOME, so a session id
    resumes only *that* conversation.
  - Config is passed per-invocation via OPENCODE_CONFIG_CONTENT (inline JSON
    env var), no config file on disk, so concurrent turns never race, and the
    endpoint's headers, LiteLLM attribution and the current MCP server set are
    baked in per call. OpenCode talks to the LiteLLM endpoint directly through
    the @ai-sdk/openai-compatible provider it bundles.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import signal
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, SchemaError, ValidationError

logger = logging.getLogger(__name__)

# The model a run uses when neither the task nor the agent-config names one; the
# @ai-sdk/openai-compatible provider forwards it verbatim as the chat model.
DEFAULT_MODEL = os.environ.get("OPENCODE_MODEL", "anthropic/claude-sonnet-4-6")
ERROR_MESSAGE_LIMIT = 4000
ROLE_HEADER = "AgentEnv-Role"
# What agent-env gives an agent for [model.headers]: a JSON object of header names to values.
MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"

# Root under which each A2A context_id gets its own OpenCode data dir; lives
# under $HOME so host-mode instances don't share session state.
SESSIONS_ROOT = os.environ.get("OPENCODE_SESSIONS_ROOT") or os.path.join(os.environ.get("HOME", "/data"), "sessions")
_CLI_TIMEOUT_S = 120


def _base_url(environ: Mapping[str, str]) -> str:
    """The endpoint's chat-completions root: agent-env's LITELLM_BASE_URL has no /v1 suffix."""
    base = (environ.get("LITELLM_BASE_URL") or "").rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def context_data_dir(context_id: str) -> Path:
    """Per-context OpenCode data dir. A stable short hash keeps the path clean
    and filesystem-safe regardless of the context_id's charset."""
    ctx_hash = hashlib.sha1(context_id.encode()).hexdigest()[:16]
    return Path(SESSIONS_ROOT) / ctx_hash


def _context_xdg(context_id: str) -> dict[str, str]:
    """Per-context XDG data/state dirs so each conversation's session store is
    isolated. cache/config (binary + skills) stay global (inherited from env)."""
    data_dir = context_data_dir(context_id)
    (data_dir / "share").mkdir(parents=True, exist_ok=True)
    (data_dir / "state").mkdir(parents=True, exist_ok=True)
    return {"XDG_DATA_HOME": str(data_dir / "share"), "XDG_STATE_HOME": str(data_dir / "state")}


async def _communicate(*cmd: str, env: Mapping[str, str], timeout_s: int) -> tuple[int, bytes, bytes]:
    """Run a short CLI command to its end; one that outlives ``timeout_s`` is killed."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=dict(env),
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise
    return proc.returncode, out, err


async def export_session(context_id: str, session_id: str, timeout_s: int = _CLI_TIMEOUT_S) -> bytes:
    """`opencode export <session_id>` for the given context: the transcript JSON bytes."""
    env = {**os.environ, **_context_xdg(context_id)}
    code, out, err = await _communicate("opencode", "export", session_id, env=env, timeout_s=timeout_s)
    if code != 0:
        raise RuntimeError(f"opencode export failed: {err.decode(errors='replace')[:300]}")
    return out


async def import_session(context_id: str, json_path: str, timeout_s: int = _CLI_TIMEOUT_S) -> str | None:
    """`opencode import <json_path>` into the given context's data dir: the imported session id, from
    'Imported session: <id>'."""
    env = {**os.environ, **_context_xdg(context_id)}
    code, out, err = await _communicate("opencode", "import", json_path, env=env, timeout_s=timeout_s)
    if code != 0:
        raise RuntimeError(f"opencode import failed: {err.decode(errors='replace')[:300]}")
    # Match the session-id token directly: the "Imported session: <id>" line is
    # ANSI-colorized, so a greedy \S+ would capture escape codes with the id.
    match = re.search(r"ses_[A-Za-z0-9]+", out.decode(errors="replace"))
    return match.group(0) if match else None


@dataclass
class OpencodeResult:
    response: str
    tool_call_count: int
    session_id: str | None
    trajectory: list[dict] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    structured_output: object | None = None


def validate_output_format(output_format: dict) -> dict:
    if output_format.get("type") != "json_schema" or not isinstance(output_format.get("schema"), dict):
        raise ValueError("output_format must contain type=json_schema and a schema object")
    try:
        Draft202012Validator.check_schema(output_format["schema"])
    except SchemaError as exc:
        raise ValueError("output_format contains an invalid JSON schema") from exc
    return output_format["schema"]


# The attribution the last run sent, which /ext/attribution-probe reads back so the
# validator can confirm project_id/task_id were wired through.
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


def _attribution_headers(project_id: str | None, task_id: str | None) -> dict[str, str]:
    """LiteLLM cost-attribution headers, x-litellm-customer-id + x-litellm-tags; {} when nothing to attribute."""
    headers: dict[str, str] = {}
    if project_id:
        headers["x-litellm-customer-id"] = project_id
        tags = [f"projectId:{project_id}"]
        if task_id:
            tags.append(f"taskId:{task_id}")
        headers["x-litellm-tags"] = ",".join(tags)
    elif task_id:
        headers["x-litellm-tags"] = f"taskId:{task_id}"
    return headers


def model_headers(raw: str | None) -> dict[str, str]:
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


def provider_headers(environ: Mapping[str, str], attribution: Mapping[str, str]) -> dict[str, str]:
    """The endpoint's [model.headers], then the run's attribution, each replacing a header of the same name in any
    case, except that the run's x-litellm-tags join the ones [model.headers] name."""
    configured = {name.lower(): value for name, value in model_headers(environ.get(MODEL_HEADERS_ENV)).items()}
    if configured.get(_TAGS_HEADER) and _TAGS_HEADER in attribution:
        attribution = {**attribution, _TAGS_HEADER: _join_tags(configured[_TAGS_HEADER], attribution[_TAGS_HEADER])}
    return {name.lower(): value for layer in (configured, attribution) for name, value in layer.items()}


def _mcp_headers(headers: Mapping[str, str] | None, role: str | None) -> dict[str, str]:
    """A server's registered headers; the run's role, when it has one, replaces a role header they name."""
    merged = dict(headers or {})
    if role:
        merged = {name: value for name, value in merged.items() if name.lower() != ROLE_HEADER.lower()}
        merged[ROLE_HEADER] = role
    return merged


def _build_config(
    model: str,
    mcp_servers: dict[str, dict] | None,
    headers: dict[str, str],
    role: str | None,
    effort: str | None = None,
    environ: Mapping[str, str] = os.environ,
) -> dict:
    """Assemble the inline OpenCode config passed via OPENCODE_CONFIG_CONTENT.

    Everything dynamic (model, MCP servers, headers) is rebuilt per turn so there
    is no on-disk config to race on."""
    options: dict = {"baseURL": _base_url(environ), "apiKey": "{env:LITELLM_API_KEY}"}
    if headers:
        options["headers"] = dict(headers)

    config: dict = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "litellm": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "LiteLLM",
                "options": options,
                # Declaring image input is required for OpenCode to forward
                # image content to a custom openai-compatible provider: without
                # it OpenCode treats the model as text-only and strips
                # attachments (the model then falsely claims no image support).
                # Model-entry `options` are provider options. Two mapping rules
                # (both verified against opencode-ai 1.17.18):
                #   * reasoningEffort (camelCase) is a KNOWN option the provider
                #     maps to the wire `reasoning_effort` param; the snake_case
                #     spelling is not recognized as that option.
                #   * UNKNOWN keys pass through into the request body verbatim,
                #     which is how additional_drop_params (below) reaches LiteLLM.
                #
                # additional_drop_params: LiteLLM adapters for some upstreams
                # hard-reject standard OpenAI params instead of dropping them:
                # fireworks_ai 400s on `tool_choice` (litellm.UnsupportedParamsError),
                # which fails every call on such models since opencode always sends
                # its tool set. Dropping `tool_choice` is semantics-free: opencode
                # only ever sends the default ("auto"), and providers that do
                # support it behave identically without it.
                "models": {model: {
                    "name": model,
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                    "options": {
                        "additional_drop_params": ["tool_choice"],
                        **({"reasoningEffort": effort} if effort else {}),
                    },
                }},
            }
        },
        "model": f"litellm/{model}",
        # Headless: allow every tool, hard-deny the "ask the human" tool so a
        # run can never block waiting for input.
        "permission": {"*": "allow", "question": "deny"},
        "autoupdate": False,
    }

    if mcp_servers:
        mcp: dict[str, dict] = {}
        for name, info in mcp_servers.items():
            entry: dict = {"type": "remote", "url": info["url"], "enabled": True}
            if hdrs := _mcp_headers(info.get("headers"), role):
                entry["headers"] = hdrs
            mcp[name] = entry
        config["mcp"] = mcp

    return config


_PROCESS_CLEANUP_TIMEOUT_S = 5
_RUNNING: set = set()


def kill_running_clis() -> int:
    """Kill every CLI still running a turn, and what each started; how many. The agent's shutdown does: the CLIs, in
    sessions of their own, would outlive its server, as when a reinstall stops it mid-turn."""
    running = [proc for proc in _RUNNING if proc.returncode is None]
    for proc in running:
        _kill_cli(proc)
    return len(running)


def _kill_cli(proc) -> None:
    """Kill the CLI and everything under it: its process group, and, found by parent, the tools a CLI starts in a
    session of their own, which no group or session kill reaches."""
    pid = getattr(proc, "pid", None)
    if pid is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return
    if proc.returncode is not None:
        return  # the CLI is gone, and its pid may be another process's by now
    tree = [pid, *_descendants(pid)]
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    for member in tree:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(member, signal.SIGKILL)


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


async def _reap(proc) -> None:
    """Wait for the killed CLI, but not for a process that left its tree and still holds its pipes open."""
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), _PROCESS_CLEANUP_TIMEOUT_S)


def _descendants(pid: int) -> list[int]:
    """The processes under ``pid``, by parent, read from /proc; none where there is no /proc."""
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


async def run_opencode(
    prompt: str,
    context_id: str,
    session_id: str | None = None,
    model: str | None = None,
    mcp_servers: dict[str, dict] | None = None,
    system_prompt: str | None = None,
    output_format: dict | None = None,
    timeout_s: int = 600,
    project_id: str | None = None,
    task_id: str | None = None,
    role: str | None = None,
    attachment_paths: list[str] | None = None,
    effort: str | None = None,
    event_callback: Callable[[dict], None] | None = None,
    workspace: str = "/app",
) -> OpencodeResult:
    """Run one `opencode run` turn in ``workspace``. ``event_callback`` gets each JSON event as the CLI prints it:
    exactly the events of the returned trajectory, in order."""
    model = model or DEFAULT_MODEL
    attribution = _attribution_headers(project_id, task_id)
    if attribution:
        _record_attribution(project_id, task_id)

    # OpenCode's `run` has no system-prompt flag and its config `instructions`
    # field takes file paths (not inline text), so we prepend the caller's
    # system prompt to the message. The model follows it either way.
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
    if output_format is not None:
        schema = validate_output_format(output_format)
        prompt += (
            "\n\nReturn only a JSON object matching this schema, with no Markdown or surrounding text: "
            + json.dumps(schema, separators=(",", ":"))
        )

    config = _build_config(model, mcp_servers, provider_headers(os.environ, attribution), role, effort)

    # Resume a specific session by id (multi-turn continuity). `--continue`
    # does NOT replay history to the model; `-s <session_id>` does. First turn
    # has no session_id, so OpenCode mints one, which the caller captures from the
    # result and passes back on the next turn.
    #
    # The message is a variadic positional and --file is an array flag; if
    # --file comes before the message, yargs greedily swallows the message into
    # the file array ("File not found: <prompt>"). So: message first, files last.
    cmd = ["opencode", "run", "--format", "json", "--model", f"litellm/{model}", "--dir", workspace]
    if session_id:
        cmd += ["-s", session_id]
    cmd.append(prompt)
    for path in (attachment_paths or []):
        cmd += ["--file", path]

    env = {
        **os.environ,
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        **_context_xdg(context_id),
    }

    logger.info(f"Running opencode (session={session_id}, model={model}, effort={effort}, ctx={context_id[:12]}, mcp={list((mcp_servers or {}).keys())})")
    if attribution:
        logger.info(f"opencode attribution headers: {attribution}")

    proc = await _started(asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        start_new_session=True,
    ))
    _RUNNING.add(proc)
    parser = _EventParser(event_callback)
    try:
        stderr = await asyncio.wait_for(_stream_cli(proc, parser.on_line), timeout=timeout_s)
    except asyncio.CancelledError:
        _kill_cli(proc)
        await _reap(proc)
        raise
    except asyncio.TimeoutError:
        _kill_cli(proc)
        await _reap(proc)
        return OpencodeResult(
            response="", tool_call_count=0, session_id=parser.session_id,
            error=f"opencode timed out after {timeout_s}s",
        )
    except Exception:
        # A failure while reading, such as an event_callback that raises, must
        # not leave the CLI running.
        _kill_cli(proc)
        await _reap(proc)
        raise
    finally:
        _RUNNING.discard(proc)

    result = parser.result(stderr, proc.returncode)
    if output_format is not None and result.error is None:
        try:
            structured_output = json.loads(result.response)
            Draft202012Validator(schema).validate(structured_output)
        except (json.JSONDecodeError, ValidationError) as exc:
            result.error = f"OpenCode returned a response that does not match output_format: {exc}"
            result.error_code = "invalid_output_format"
        else:
            result.structured_output = structured_output
    return result


# How much of the CLI's stdout one read takes; a line may span several reads.
_STDOUT_READ_BYTES = 64 * 1024


async def _stream_cli(proc, on_line: Callable[[str], None]) -> bytes:
    """``proc.communicate()``, handing ``on_line`` each stdout line as it arrives.

    Lines are split as ``stdout.decode(errors="replace").splitlines()`` splits
    the whole output, so they are the same lines; a line has no length limit,
    as with ``communicate``. Returns stderr once the CLI has exited."""
    stderr_task = asyncio.create_task(proc.stderr.read())
    try:
        # A line's pieces, joined once when its newline arrives: re-joining them on every read
        # would cost a long line the square of its length.
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
        stderr = await stderr_task
        await proc.wait()
        return stderr
    finally:
        stderr_task.cancel()


def event_session_id(event: dict) -> str | None:
    """The session an event names, on the event or its part."""
    part = event.get("part") if isinstance(event.get("part"), dict) else {}
    return event.get("sessionID") or event.get("session_id") or part.get("sessionID") or part.get("session_id")


def _parse_events(stdout: bytes, stderr: bytes, returncode: int | None) -> OpencodeResult:
    """Parse a whole `--format json` JSONL event stream."""
    parser = _EventParser()
    for line in stdout.decode(errors="replace").splitlines():
        parser.on_line(line)
    return parser.result(stderr, returncode)


class _EventParser:
    """Parse the `--format json` JSONL event stream one line at a time.

    Events carry a `type` and (for content) a `part`. We accumulate assistant
    `text` parts as the response, count `tool` parts as tool calls, and note
    the session id wherever it appears. The whole event list is the trajectory,
    and each event is handed to ``event_callback`` as it is parsed.
    """

    def __init__(self, event_callback: Callable[[dict], None] | None = None) -> None:
        self._event_callback = event_callback
        self.trajectory: list[dict] = []
        self.text_chunks: list[str] = []
        self.tool_calls: set[str] = set()
        self.session_id: str | None = None
        self.completed = False
        self.event_errors: list[str] = []
        self.error_code: str | None = None

    def on_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.debug(f"opencode emitted non-JSON line: {line[:200]}")
            return
        if not isinstance(event, dict):
            logger.debug(f"opencode emitted a non-object JSON line: {line[:200]}")
            return
        self.trajectory.append(event)

        self.session_id = event_session_id(event) or self.session_id
        part = event.get("part") if isinstance(event.get("part"), dict) else None

        etype = event.get("type")
        ptype = part.get("type") if part else None

        if etype == "error":
            # JSON mode emits provider/session failures on stdout and commonly
            # leaves stderr empty. Extract only useful bounded fields; the full
            # payload can contain response headers and remains in trajectory.
            payload = event.get("error")
            message: str | None = None
            name: str | None = None
            code: object = None
            if isinstance(payload, dict):
                name_value = payload.get("name")
                if isinstance(name_value, str) and name_value:
                    name = name_value
                data = payload.get("data")
                if isinstance(data, dict):
                    message_value = data.get("message")
                    if isinstance(message_value, str) and message_value:
                        message = message_value
                    code = (
                        data.get("statusCode")
                        or data.get("status_code")
                        or data.get("code")
                    )
                if message is None:
                    message_value = payload.get("message")
                    if isinstance(message_value, str) and message_value:
                        message = message_value
                if code is None:
                    code = (
                        payload.get("statusCode")
                        or payload.get("status_code")
                        or payload.get("code")
                    )
            elif isinstance(payload, str) and payload:
                message = payload

            if message:
                rendered = (
                    f"{name}: {message}"
                    if name and not message.startswith(f"{name}:")
                    else message
                )
                if rendered not in self.event_errors:
                    self.event_errors.append(rendered)
            elif name and name not in self.event_errors:
                self.event_errors.append(name)
            if code is not None:
                self.error_code = str(code)
        elif etype == "text" or ptype == "text":
            text = (part or event).get("text")
            if isinstance(text, str) and text.strip():
                self.text_chunks.append(text)
        elif etype in ("tool", "tool_use") or ptype in ("tool", "tool_use"):
            # Dedupe by the tool call's id so a tool part seen in multiple
            # states (pending -> running -> completed) counts once.
            call_id = (part or event).get("callID") or (part or event).get("id") or f"tool-{len(self.tool_calls)}"
            self.tool_calls.add(str(call_id))
        elif etype == "step_finish" or ptype == "step-finish":
            if (part or event).get("reason") == "stop":
                self.completed = True

        if self._event_callback is not None:
            self._event_callback(event)

    def result(self, stderr: bytes, returncode: int | None) -> OpencodeResult:
        # A turn is successful once OpenCode emits a terminal step_finish{reason:stop}.
        # OpenCode can exit non-zero *after* a completed turn (e.g. when an attached
        # file's modality is unsupported but the model still answered), so we only
        # surface an error when no terminal stop was emitted.
        err: str | None = None
        if not self.completed:
            stderr_text = stderr.decode(errors="replace").strip()
            details = "\n".join(self.event_errors)
            err = (
                details
                or stderr_text
                or f"opencode exited with code {returncode} without completing"
            )[:ERROR_MESSAGE_LIMIT]
            logger.error(f"opencode failed (no terminal stop): {err}")

        return OpencodeResult(
            response="\n".join(self.text_chunks).strip(),
            tool_call_count=len(self.tool_calls),
            session_id=self.session_id,
            trajectory=self.trajectory,
            error=err,
            error_code=self.error_code if err else None,
        )


_TAGS_HEADER = "x-litellm-tags"


def _join_tags(configured: str, run: str) -> str:
    """x-litellm-tags as configured, then each of the run's tags whose key (the part before ``:``) they don't name."""
    tags = [tag.strip() for tag in configured.split(",") if tag.strip()]
    keys = {tag.partition(":")[0] for tag in tags}
    return ",".join([*tags, *(tag for tag in run.split(",") if tag and tag.partition(":")[0] not in keys)])
