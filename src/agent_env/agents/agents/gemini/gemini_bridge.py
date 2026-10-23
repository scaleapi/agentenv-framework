"""Subprocess wrapper around the Google Gemini CLI.

Runs `gemini -p "$prompt" --output-format stream-json --approval-mode=yolo`
(with `-r <uuid>` for resume) and parses the NDJSON output into a structured
result. The caller threads sessions by passing None on the first turn, then
feeding back result.session_id (server-assigned UUID captured from the init
event) on subsequent turns.
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
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from gemini_config import SYSTEM_SETTINGS_ENV, gemini_home

logger = logging.getLogger(__name__)

# Gemini CLI persists per-session JSONL transcripts at
#   <GEMINI_HOME>/tmp/<projectIdentifier>/chats/session-<ISO-ts>-<sessionId[:8]>.jsonl
# `<projectIdentifier>` is a per-cwd slug allocated on first run and persisted
# in `<GEMINI_HOME>/projects.json`. Resume (`gemini -r <uuid>`) scans the chats
# dir and matches by the `sessionId` field inside the first JSONL line — the
# filename's uuid prefix is just a search hint.
#
# The CLI keeps it under $GEMINI_CLI_HOME, else $HOME, as gemini_config reads it, so
# save/load find the files the CLI wrote whatever uid the runtime chose.
GEMINI_HOME = gemini_home()
GEMINI_TMP = GEMINI_HOME / "tmp"
GEMINI_PROJECTS_JSON = GEMINI_HOME / "projects.json"


def get_project_slug(cwd: str = "/workspace") -> str | None:
    """Return the project-identifier slug for `cwd` from gemini's projects.json,
    or None if the registry hasn't been initialized yet (no gemini run on this
    cwd has happened). The slug names the chats dir under `<GEMINI_HOME>/tmp/`.

    File shape is `{"projects": {"<absolute-cwd>": "<slug>", ...}}` (verified
    empirically on gemini-cli 0.42.0). Older versions may have stored the map
    at the top level — we handle both for safety.
    """
    if not GEMINI_PROJECTS_JSON.exists():
        return None
    try:
        registry = json.loads(GEMINI_PROJECTS_JSON.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read {GEMINI_PROJECTS_JSON}: {e}")
        return None
    projects = registry.get("projects") if isinstance(registry, dict) else None
    if not isinstance(projects, dict):
        projects = registry  # legacy top-level layout
    entry = projects.get(cwd) if isinstance(projects, dict) else None
    if isinstance(entry, dict):
        return entry.get("id") or entry.get("slug") or entry.get("shortId")
    if isinstance(entry, str):
        return entry
    return None


def chats_dir_for_cwd(cwd: str = "/workspace") -> Path | None:
    slug = get_project_slug(cwd)
    if slug is None:
        return None
    return GEMINI_TMP / slug / "chats"


def find_session_file(session_id: str, cwd: str = "/workspace") -> Path | None:
    """Locate the session transcript for `session_id`. Scans the cwd's chats
    dir for `session-*-<session_id[:8]>.{json,jsonl}` and verifies the
    sessionId inside matches.

    Gemini writes the first turn as legacy `.json` (single JSON object with a
    `messages` array) and auto-migrates to `.jsonl` on resume — so single-turn
    captures land on `.json`. Both formats are supported here.
    """
    chats = chats_dir_for_cwd(cwd)
    if chats is None or not chats.exists():
        return None
    short = session_id[:8]
    candidates = list(chats.glob(f"session-*-{short}.json")) \
        + list(chats.glob(f"session-*-{short}.jsonl"))
    for path in candidates:
        try:
            raw = path.read_text()
        except OSError:
            continue
        sid = _peek_session_id(raw)
        if sid == session_id:
            return path
    return None


def transcript_session_id(path: Path) -> str | None:
    """The sessionId a transcript file records; None when it can't be read or names none."""
    try:
        return _peek_session_id(path.read_text())
    except OSError:
        return None


def _peek_session_id(raw: str) -> str | None:
    """Pull `sessionId` from either a JSONL (first line) or single-object JSON
    transcript. Returns None on parse failure."""
    text = raw.strip()
    if not text:
        return None
    # JSONL first-line attempt. Gemini's pretty-printed legacy `.json` starts
    # with a bare `{` on its own line, which fails to parse — that's the
    # signal to fall through to whole-file parsing.
    first_line = text.splitlines()[0]
    try:
        meta = json.loads(first_line)
        if isinstance(meta, dict) and "sessionId" in meta:
            return meta["sessionId"]
    except json.JSONDecodeError:
        pass
    # Legacy single-object JSON — parse whole file.
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(whole, dict) and "sessionId" in whole:
        return whole["sessionId"]
    return None


# A file's modification time comes from a coarser clock than time.time(), so one written just after a run started can
# read as a little before it.
_MTIME_SLACK_S = 1.0


def _unclaimed_session_id(since: float, claimed: Collection[str], cwd: str) -> str | None:
    """The sessionId of the one transcript written since ``since`` that isn't ``claimed`` by another turn or
    context: a run whose stream carried no ``init`` event (a CLI that died early) still wrote its transcript to
    ``<GEMINI_HOME>/tmp/<slug>/chats/session-*.json{,l}``. None when several are, since any could be another
    turn's."""
    since -= _MTIME_SLACK_S
    chats = chats_dir_for_cwd(cwd)
    if chats is None or not chats.exists():
        return None
    found: set[str] = set()
    for path in [*chats.glob("session-*.json"), *chats.glob("session-*.jsonl")]:
        try:
            session = _peek_session_id(path.read_text()) if path.stat().st_mtime >= since else None
        except OSError:
            continue
        if session and session not in claimed:
            found.add(session)
    if len(found) > 1:
        logger.warning(f"not recovering a session: {len(found)} unclaimed transcripts were written during the run")
        return None
    return next(iter(found), None)


def detect_format(raw: bytes) -> str:
    """Classify a transcript blob as 'json' (legacy single-object) or 'jsonl'.
    Used by snapshot save/load to preserve the source format byte-for-byte
    rather than risk a lossy normalization (gemini's JSONL contains fields
    that a one-way `.json -> .jsonl` conversion can't reconstruct correctly,
    so we plant the exact source format and let gemini's loader handle it).
    """
    text = raw.decode().strip()
    if not text:
        return "jsonl"
    try:
        whole = json.loads(text)
        if isinstance(whole, dict) and "messages" in whole:
            return "json"
    except json.JSONDecodeError:
        pass
    return "jsonl"


def rewrite_session_id(raw: bytes, new_session_id: str) -> bytes:
    """Replace the sessionId in a transcript while preserving its on-disk
    format. For legacy `.json` single-object files, mutate `.sessionId` at
    the top level. For JSONL, parse the first metadata line, rewrite its
    `sessionId` field, and re-emit. Other lines are passed through unchanged.
    """
    fmt = detect_format(raw)
    text = raw.decode()
    if fmt == "json":
        obj = json.loads(text)
        obj["sessionId"] = new_session_id
        return json.dumps(obj).encode()
    # JSONL: rewrite the first line only.
    lines = text.splitlines()
    if not lines:
        return raw
    first = json.loads(lines[0])
    if not isinstance(first, dict) or "sessionId" not in first:
        raise ValueError("first JSONL line missing 'sessionId' — unexpected gemini chat format")
    first["sessionId"] = new_session_id
    lines[0] = json.dumps(first)
    return ("\n".join(lines) + "\n").encode()

# ---------------------------------------------------------------------------
# LiteLLM cost attribution.
#
# The gemini CLI / @google/genai SDK exposes no hook to attach custom HTTP headers
# to its model calls. So attribution is injected by an in-process reverse proxy in
# a2a_server.py: each run's GOOGLE_GEMINI_BASE_URL names the proxy and the run, and
# the proxy calls build_attribution_headers() + record_attribution() per forwarded
# request. The wire format (x-litellm-customer-id / x-litellm-tags) and the probe
# contract (semantic project_id/task_id) match the other agents'. `projectId:` /
# `taskId:` in x-litellm-tags are LiteLLM spend tags: a row per request in its spend
# records, which agent-env's attribution validator parses back out.
# ---------------------------------------------------------------------------
_last_attribution: dict[str, str | None] = {
    "project_id": None,
    "task_id": None,
    "last_seen_at_utc": None,
}
_attribution_lock = threading.Lock()


def record_attribution(project_id: str | None, task_id: str | None) -> None:
    """Snapshot the attribution the proxy just applied to an outbound model call."""
    if not project_id and not task_id:
        return
    with _attribution_lock:
        _last_attribution["project_id"] = project_id or None
        _last_attribution["task_id"] = task_id or None
        _last_attribution["last_seen_at_utc"] = datetime.now(timezone.utc).isoformat()


def get_last_attribution_snapshot() -> dict[str, str | None]:
    """Most recent attribution the proxy emitted. Read by /ext/attribution-probe;
    None values mean nothing has fired yet."""
    with _attribution_lock:
        return dict(_last_attribution)


def build_attribution_headers(project_id: str | None, task_id: str | None) -> dict[str, str]:
    """LiteLLM cost-attribution headers, same wire format as the sibling agents:
    x-litellm-customer-id + x-litellm-tags. Returns {} when nothing to attribute."""
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


# What agent-env gives an agent for [model.headers]: a JSON object of header names to values.
MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"


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


def cli_env(environ: Mapping[str, str], *, system_settings: Path, base_url: str | None) -> dict[str, str]:
    """The CLI's environment for one run: this process's, with the endpoint's key under the name the CLI reads, its
    model calls sent to ``base_url`` (the attribution proxy, for this run) unless the environment names its own
    endpoint, and the run's own system settings."""
    env = dict(environ)
    if env.get("LITELLM_API_KEY") and not env.get("GEMINI_API_KEY"):
        env["GEMINI_API_KEY"] = env["LITELLM_API_KEY"]
    if base_url and not environ.get("GOOGLE_GEMINI_BASE_URL"):
        env["GOOGLE_GEMINI_BASE_URL"] = base_url
    env[SYSTEM_SETTINGS_ENV] = str(system_settings)
    return env


@dataclass
class GeminiResult:
    response: str
    tool_call_count: int
    session_id: str | None
    trajectory: list[dict] = field(default_factory=list)
    error: str | None = None


# Bounded so the whole message stays under a2a_server's 1000-character cap.
_ERROR_DETAIL_LIMIT = 600
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
_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
# Appended to a 429 for a human at the terminal; it misleads anyone reading an
# agent's error.
_RATE_LIMIT_ADVICE = (
    "\nPlease wait and try again later. To increase your limits, request a quota "
    "increase through AI Studio, or switch to another /auth method"
)
# The yolo banner opens every run, an environment that turns off Node's TLS checks
# adds its warning, and a logged API error trails a stack trace and `{ status: N }`.
_STDERR_NOISE = re.compile(
    r"YOLO mode is enabled\."
    r"|\(node:\d+\) Warning: Setting the NODE_TLS_REJECT_UNAUTHORIZED"
    r"|\(Use `node --trace-warnings"
    r"|\s+at "
    r"|\s*[{}]\s*$"
    r"|\s+status: \d+\s*$"
)
_ERROR_REPORT_PREFIX = re.compile(r"^Error when talking to Gemini API Full report available at: \S+\s*")


def _redact(text: str) -> str:
    return _SECRET_PATTERN.sub("[redacted]", _HEADER_VALUE_PATTERN.sub(r"\1[redacted]'", text))


def _one_line(text: str) -> str:
    return _LINE_BREAK_PATTERN.sub(" ", text).strip()


def _cli_error_text(trajectory: list[dict]) -> str | None:
    """The CLI's own account of a failed turn, from its stream.

    A failed turn ends with a `result` event of status "error". Its type is
    "unknown" for every API error and a bare "Error" for an unnamed one; only
    the CLI's own fatal errors name a class worth keeping.
    """
    for event in reversed(trajectory):
        if event.get("type") != "result" or event.get("status") != "error":
            continue
        error = event.get("error")
        if not isinstance(error, dict) or not isinstance(error.get("message"), str):
            return None
        text = _one_line(_ANSI_PATTERN.sub("", error["message"].removesuffix(_RATE_LIMIT_ADVICE)))
        if error.get("type") not in ("unknown", "Error", None):
            text = f"{error['type']}: {text}"
        return text
    return None


def _stderr_tail(stderr: str) -> str | None:
    """stderr minus what every run prints, for a CLI whose stream carries no
    cause (a startup failure or a kill)."""
    lines = [
        _ERROR_REPORT_PREFIX.sub("", line).strip()
        for line in _ANSI_PATTERN.sub("", _redact(stderr)).splitlines()
        if line.strip() and not _STDERR_NOISE.match(line)
    ]
    return " | ".join(line for line in lines if line)[-_ERROR_DETAIL_LIMIT:] or None


def _failure_message(head: str, detail: str | None) -> str:
    """What ended the turn, then the CLI's own words.

    In that order so what matters most survives a caller's truncation.
    """
    message = head
    if detail:
        message += f": {_redact(detail)[:_ERROR_DETAIL_LIMIT]}"
    return message


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


def _parse_event(line: str) -> dict | None:
    """One stream-json line's event. The CLI prints some notices, such as an MCP server that failed to connect, on
    stdout with no newline, so the event follows them on the same line."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        start = line.find("{")
        if start <= 0:
            raise
        logger.warning(f"gemini CLI printed {line[:start].strip()!r} before an event")
        return json.loads(line[start:])


async def run_gemini(
    prompt: str,
    session_id: str | None,
    model: str | None = None,
    attachment_paths: list[str] | None = None,
    timeout_s: int = 600,
    event_callback: Callable[[dict], None] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    other_sessions: Callable[[], Collection[str]] = tuple,
) -> GeminiResult:
    """Run one `gemini -p` turn in ``env``, with ``cwd`` (else this process's) the CLI's workspace. ``event_callback``
    gets each stream-json event as the CLI prints it: exactly the events of the returned trajectory, in order.
    ``other_sessions`` answers, when it is asked, the sessions other turns and contexts claim, which a first turn's own
    is never taken for."""
    # Multimodal attachments are injected as `@<absolute_path>` references in
    # the prompt — that's the Gemini CLI's documented mechanism for headless
    # mode (there's no `--image` flag). The CLI's `read_many_files` tool
    # handles image / PDF / audio / video files transparently.
    if attachment_paths:
        refs = " ".join(f"@{p}" for p in attachment_paths)
        prompt = f"{prompt}\n\n{refs}" if prompt else refs

    cmd = [
        "gemini", "-p", prompt,
        "--output-format", "stream-json",
        "--approval-mode", "yolo",
    ]
    if session_id is not None:
        cmd += ["-r", session_id]
    if model:
        cmd += ["--model", model]

    logger.info(
        f"Running gemini CLI (session_id={session_id}, model={model}, "
        f"attachments={len(attachment_paths or [])})"
    )
    started = time.time()
    proc = await _started(asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env) if env is not None else None,
        cwd=cwd,
        start_new_session=True,
    ))
    _RUNNING.add(proc)
    trajectory: list[dict] = []
    response_parts: list[str] = []
    tool_count = 0
    captured_session_id: str | None = session_id

    def on_line(line: str) -> None:
        nonlocal tool_count, captured_session_id
        if not line.strip():
            return
        try:
            event = _parse_event(line)
        except json.JSONDecodeError:
            logger.warning(f"gemini CLI emitted non-JSON line: {line[:200]}")
            return
        if not isinstance(event, dict):
            logger.warning(f"gemini CLI emitted a non-object JSON line: {line[:200]}")
            return
        trajectory.append(event)
        etype = event.get("type")
        if etype == "init":
            captured_session_id = event.get("session_id", captured_session_id)
        elif etype == "message" and event.get("role") == "assistant":
            content = event.get("content")
            if isinstance(content, str):
                response_parts.append(content)
        elif etype in ("tool_use", "tool_result"):
            # Anything streamed before this tool event was interstitial reasoning
            # ("I'll check the list..."), not the final answer. Reset so only
            # post-last-tool assistant text survives: gemini's stream-json has no
            # final result text of its own.
            response_parts.clear()
            if etype == "tool_use":
                tool_count += 1
        elif etype == "result":
            # Final summary; prefer authoritative tool count if present.
            stats = event.get("stats") or {}
            if "tool_calls" in stats:
                tool_count = stats["tool_calls"]
        if event_callback is not None:
            event_callback(event)

    try:
        stderr = await asyncio.wait_for(_stream_cli(proc, on_line), timeout=timeout_s)
    except asyncio.CancelledError:
        _kill_cli(proc)
        await _reap(proc)
        raise
    except asyncio.TimeoutError:
        _kill_cli(proc)
        await _reap(proc)
        return GeminiResult(
            response="", tool_call_count=0, session_id=captured_session_id,
            error=f"gemini CLI timed out after {timeout_s}s",
        )
    except Exception:
        # A failure while reading, such as an event_callback that raises, must
        # not leave the CLI running.
        _kill_cli(proc)
        await _reap(proc)
        raise
    finally:
        _RUNNING.discard(proc)

    response = "".join(response_parts)
    err: str | None = None
    if proc.returncode != 0:
        head = (
            f"gemini CLI was killed by signal {-proc.returncode}" if proc.returncode < 0
            else f"gemini CLI exited with code {proc.returncode}"
        )
        detail = _cli_error_text(trajectory) or _stderr_tail(stderr.decode(errors="replace"))
        err = _failure_message(head, detail)
        response = _redact(response)
        logger.error(f"gemini CLI failed: {err}")

    # No session id in the stream (a CLI that died before its `init` event) → recover
    # it from the on-disk transcript so the session is threadable (multi-turn
    # resume + /ext/snapshot). Only on a fresh turn; a resume keeps its id.
    if not captured_session_id:
        recovered = _unclaimed_session_id(started, other_sessions(), cwd or os.getcwd())
        if recovered:
            logger.info(f"Recovered session id from transcript (no stream init event): {recovered}")
            captured_session_id = recovered

    return GeminiResult(
        response=response,
        tool_call_count=tool_count,
        session_id=captured_session_id,
        trajectory=trajectory,
        error=err,
    )


# ---------------------------------------------------------------------------
# Vertex / OpenAI-compat routing shim.
#
# gemini-cli embeds --model in the native Gemini API URL path
# (/v1beta/models/{model}:streamGenerateContent). A provider-prefixed LiteLLM
# model such as `vertex_ai/global/gemini-3.1-pro-preview` therefore (a) crashes
# @google/genai client-side on the slashes, and (b) is unreachable via LiteLLM's
# /gemini passthrough (AI-Studio-bound). To run gemini-cli against the Vertex
# route we hand the CLI a bare Gemini id and let the a2a_server proxy translate
# the CLI's native-Gemini request into an OpenAI /v1/chat/completions call whose
# `model` is the real (slash-bearing) route string — the one endpoint of a
# LiteLLM gateway that reaches Vertex.
# ---------------------------------------------------------------------------

_GEMINI_GENERATE_ACTIONS = (":generateContent", ":streamGenerateContent")


def needs_openai_route(model: str | None) -> bool:
    """A provider-prefixed model (contains '/') can't go through the native
    Gemini passthrough — route it via the OpenAI-compatible endpoint."""
    return bool(model) and "/" in model


def cli_model(model: str | None) -> str | None:
    """The model id handed to gemini-cli's --model. For OpenAI-routed models,
    strip to the bare trailing segment so @google/genai builds a valid URL
    (no client-side slash crash); the proxy overrides the real routing."""
    if needs_openai_route(model):
        return model.rsplit("/", 1)[-1] or model
    return model


def is_generate_path(rest: str) -> bool:
    """True if `rest` (the proxied sub-path) is a content-generation call."""
    path = rest.split("?", 1)[0]
    return any(path.endswith(a) for a in _GEMINI_GENERATE_ACTIONS)


def wants_sse(rest: str, query_alt: str | None = None) -> bool:
    path = rest.split("?", 1)[0]
    return path.endswith(":streamGenerateContent") or (query_alt == "sse")


def _parts_to_text(parts: list[dict]) -> str:
    return "\n".join(p.get("text", "") for p in (parts or []) if p.get("text"))


def _gemini_tools_to_openai(gtools: list[dict] | None) -> list[dict] | None:
    if not gtools:
        return None
    out: list[dict] = []
    for t in gtools:
        decls = t.get("functionDeclarations") or t.get("function_declarations") or []
        for fd in decls:
            params = fd.get("parameters") or fd.get("parametersJsonSchema") \
                or {"type": "object", "properties": {}}
            out.append({
                "type": "function",
                "function": {
                    "name": fd.get("name"),
                    "description": fd.get("description", ""),
                    "parameters": params,
                },
            })
    return out or None


def _media_part_to_openai(part: dict) -> dict | None:
    """Convert a Gemini inlineData/fileData part into an OpenAI multimodal
    content item so images / audio / PDF / video survive the Vertex hop.
    Inline media -> an ``image_url`` data URL carrying the original mimeType;
    fileData -> an ``image_url`` pointing at the file uri. LiteLLM maps both back
    to Gemini ``inline_data`` / ``file_data`` for Vertex (verified end-to-end for
    image/png), so multimodal reaches the model instead of being dropped."""
    inline = part.get("inlineData") or part.get("inline_data")
    if isinstance(inline, dict) and inline.get("data"):
        mime = inline.get("mimeType") or inline.get("mime_type") or "application/octet-stream"
        return {"type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{inline['data']}"}}
    fdata = part.get("fileData") or part.get("file_data")
    if isinstance(fdata, dict) and (fdata.get("fileUri") or fdata.get("file_uri")):
        return {"type": "image_url",
                "image_url": {"url": fdata.get("fileUri") or fdata.get("file_uri")}}
    return None


def gemini_request_to_openai(greq: dict, model: str) -> dict:
    """Translate a native Gemini generateContent request body into an OpenAI
    chat/completions body. Handles system instruction, multi-turn text, inline
    media (image/audio/PDF/video), function-call / function-response threading,
    tools, and generation config."""
    messages: list[dict] = []
    sys_inst = greq.get("systemInstruction") or greq.get("system_instruction")
    if isinstance(sys_inst, dict):
        stext = _parts_to_text(sys_inst.get("parts", []))
        if stext:
            messages.append({"role": "system", "content": stext})

    # Gemini functionResponse carries the tool name but no call id; synthesize
    # ids and match them positionally by name so multi-turn tool use threads.
    pending_ids: dict[str, list[str]] = {}
    counter = 0
    for content in greq.get("contents", []) or []:
        role = content.get("role", "user")
        parts = content.get("parts", []) or []
        if role == "model":
            text_bits: list[str] = []
            tool_calls: list[dict] = []
            for p in parts:
                if p.get("text"):
                    text_bits.append(p["text"])
                fc = p.get("functionCall") or p.get("function_call")
                if fc:
                    cid = f"call_{counter}"
                    counter += 1
                    pending_ids.setdefault(fc.get("name"), []).append(cid)
                    tool_calls.append({
                        "id": cid, "type": "function",
                        "function": {
                            "name": fc.get("name"),
                            "arguments": json.dumps(fc.get("args") or fc.get("arguments") or {}),
                        },
                    })
            msg: dict = {"role": "assistant",
                         "content": "\n".join(text_bits) if text_bits else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
        else:
            # A user turn carries tool results (functionResponse -> tool msgs)
            # and/or genuine user content (text + inline media). Collect content
            # items in order; emit a plain string when text-only (keeps the
            # common path byte-identical) or an OpenAI content-list when media is
            # present, so images/audio/PDF are not silently dropped on Vertex.
            content_items: list[dict] = []
            for p in parts:
                fr = p.get("functionResponse") or p.get("function_response")
                if fr:
                    name = fr.get("name")
                    ids = pending_ids.get(name)
                    cid = ids.pop(0) if ids else f"call_{name}"
                    resp = fr.get("response")
                    messages.append({
                        "role": "tool", "tool_call_id": cid,
                        "content": resp if isinstance(resp, str) else json.dumps(resp),
                    })
                    continue
                if p.get("text"):
                    content_items.append({"type": "text", "text": p["text"]})
                    continue
                media = _media_part_to_openai(p)
                if media is not None:
                    content_items.append(media)
                elif p:
                    # Never drop a non-empty part without a trace.
                    logger.warning(
                        "vertex bridge: unhandled Gemini part %s dropped from user turn",
                        list(p.keys()),
                    )
            if content_items:
                if all(it["type"] == "text" for it in content_items):
                    messages.append({"role": "user",
                                     "content": "\n".join(it["text"] for it in content_items)})
                else:
                    messages.append({"role": "user", "content": content_items})

    oai: dict = {"model": model, "messages": messages}
    gc = greq.get("generationConfig") or greq.get("generation_config") or {}
    if gc.get("maxOutputTokens") is not None:
        oai["max_tokens"] = gc["maxOutputTokens"]
    if gc.get("temperature") is not None:
        oai["temperature"] = gc["temperature"]
    if gc.get("topP") is not None:
        oai["top_p"] = gc["topP"]
    if gc.get("stopSequences"):
        oai["stop"] = gc["stopSequences"]
    tools = _gemini_tools_to_openai(greq.get("tools"))
    if tools:
        oai["tools"] = tools
    return oai


_FINISH_MAP = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "tool_calls": "STOP",
    "content_filter": "SAFETY",
    "function_call": "STOP",
}


def _thought_signature(msg: dict, tc: dict) -> str | None:
    for holder in (tc, msg):
        psf = holder.get("provider_specific_fields") or {}
        sig = psf.get("thought_signature") or psf.get("thoughtSignature")
        if isinstance(sig, str) and sig:
            return sig
        sigs = psf.get("thought_signatures") or psf.get("thoughtSignatures")
        if isinstance(sigs, list) and sigs and isinstance(sigs[0], str):
            return sigs[0]
    return None


def openai_response_to_gemini(oai: dict) -> dict:
    """Translate a (non-streaming) OpenAI chat/completions response into a single
    native-Gemini GenerateContentResponse the @google/genai client can parse."""
    choice = (oai.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    parts: list[dict] = []
    if msg.get("content"):
        parts.append({"text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        fcpart: dict = {"functionCall": {"name": fn.get("name"), "args": args}}
        sig = _thought_signature(msg, tc)
        if sig:
            fcpart["thoughtSignature"] = sig
        parts.append(fcpart)
    if not parts:
        parts.append({"text": ""})
    usage = oai.get("usage") or {}
    gresp: dict = {
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": _FINISH_MAP.get(choice.get("finish_reason"), "STOP"),
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
    }
    if oai.get("model"):
        gresp["modelVersion"] = oai["model"]
    return gresp
