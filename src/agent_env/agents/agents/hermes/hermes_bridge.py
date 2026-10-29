"""Subprocess wrapper around Hermes Agent: one `hermes agentenv turn` per turn.

  - Each A2A context is a Hermes profile of its own, `<HERMES_ROOT>/profiles/ctx-<hash>`, with its own config,
    sessions and memory, so a session id resumes only that conversation and concurrent turns never share a file.
  - The profile's config.yaml is written before each turn: the model endpoint as a named custom provider with its
    headers and the run's attribution, the turn's MCP servers, the shared skills folder, and the agentenv plugin
    enabled. The system prompt is the profile's SOUL.md, Hermes' identity slot.
  - The turn runs in the plugin's process, not in a Hermes gateway, so killing that process stops the turn. The
    plugin writes each hook event as a JSON line to a pipe this module reads as it goes: the turn's trajectory.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import signal
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, SchemaError, ValidationError

logger = logging.getLogger(__name__)

# The model a run uses when neither the task nor the agent-config names one, sent to the endpoint as named.
DEFAULT_MODEL = os.environ.get("HERMES_MODEL", "anthropic/claude-sonnet-4-6")
ERROR_MESSAGE_LIMIT = 4000
ROLE_HEADER = "AgentEnv-Role"
# What agent-env gives an agent for [model.headers]: a JSON object of header names to values.
MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"
# The custom provider each profile's config names for the endpoint.
PROVIDER = "litellm"
# The Hermes install's home; each context's profile is under its profiles/.
HERMES_ROOT = Path(os.environ.get("HERMES_ROOT") or Path(os.environ.get("HOME", "/data")) / ".hermes")
HERMES_BIN = os.environ.get("HERMES_BIN", "hermes")
PLUGIN_DIR = Path(__file__).resolve().parent / "agentenv_plugin"
_PLUGIN_NAME = "agentenv"
# Marks a SOUL.md this agent wrote, so a later turn with no system prompt gives Hermes its own back.
_SOUL_MARKER = ".agentenv-soul"
_EVENTS_FD_ENV = "AGENTENV_EVENTS_FD"
# How long the event pipe may stay open after the turn's process exits, held by something it left behind.
_EVENTS_DRAIN_S = 2


def _base_url(environ: Mapping[str, str]) -> str:
    """The endpoint's chat-completions root: agent-env's LITELLM_BASE_URL has no /v1 suffix."""
    base = (environ.get("LITELLM_BASE_URL") or "").rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def profile_dir(context_id: str) -> Path:
    """The context's Hermes profile home. A stable short hash keeps the path clean whatever the context id holds."""
    return HERMES_ROOT / "profiles" / f"ctx-{hashlib.sha1(context_id.encode()).hexdigest()[:16]}"


@dataclass
class HermesResult:
    response: str
    tool_call_count: int
    session_id: str | None
    trajectory: list[dict] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    structured_output: object | None = None
    usage: dict = field(default_factory=dict)


def validate_output_format(output_format: dict) -> dict:
    if output_format.get("type") != "json_schema" or not isinstance(output_format.get("schema"), dict):
        raise ValueError("output_format must contain type=json_schema and a schema object")
    try:
        Draft202012Validator.check_schema(output_format["schema"])
    except SchemaError as exc:
        raise ValueError("output_format contains an invalid JSON schema") from exc
    return output_format["schema"]


# The attribution the last run sent, which /ext/attribution-probe reads back so the validator can confirm
# project_id/task_id were wired through.
_last_attribution: dict[str, str | None] = {"project_id": None, "task_id": None, "last_seen_at_utc": None}
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


_TAGS_HEADER = "x-litellm-tags"


def _join_tags(configured: str, run: str) -> str:
    """x-litellm-tags as configured, then each of the run's tags whose key (the part before ``:``) they don't name."""
    tags = [tag.strip() for tag in configured.split(",") if tag.strip()]
    keys = {tag.partition(":")[0] for tag in tags}
    return ",".join([*tags, *(tag for tag in run.split(",") if tag and tag.partition(":")[0] not in keys)])


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


def build_config(
    model: str,
    mcp_servers: dict[str, dict] | None,
    headers: dict[str, str],
    role: str | None,
    skills_dir: Path,
    environ: Mapping[str, str] = os.environ,
) -> dict:
    """The profile's config.yaml for one turn, as JSON (which YAML reads). Everything dynamic is rebuilt each turn, and
    a context's turns run one at a time, so no turn reads another's."""
    provider: dict = {"api": _base_url(environ), "key_env": "LITELLM_API_KEY", "transport": "chat_completions"}
    if headers:
        provider["extra_headers"] = dict(headers)
    config: dict = {
        "model": {"provider": PROVIDER, "default": model},
        "providers": {PROVIDER: provider},
        "skills": {"external_dirs": [str(skills_dir)]},
        "plugins": {"enabled": [_PLUGIN_NAME]},
    }
    if mcp_servers:
        servers: dict[str, dict] = {}
        for name, info in mcp_servers.items():
            entry: dict = {"url": info["url"]}
            if hdrs := _mcp_headers(info.get("headers"), role):
                entry["headers"] = hdrs
            servers[name] = entry
        config["mcp_servers"] = servers
    return config


def prepare_profile(context_id: str, config: dict, system_prompt: str | None) -> Path:
    """Write the context's profile for a turn: its config, the plugin, and the system prompt as SOUL.md."""
    home = profile_dir(context_id)
    plugins = home / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(json.dumps(config, indent=2))
    plugin = plugins / _PLUGIN_NAME
    if not plugin.is_symlink():
        shutil.rmtree(plugin, ignore_errors=True)
        plugin.symlink_to(PLUGIN_DIR, target_is_directory=True)
    soul, marker = home / "SOUL.md", home / _SOUL_MARKER
    if system_prompt:
        soul.write_text(system_prompt)
        marker.touch()
    elif marker.exists():
        # Hermes writes its default identity back when SOUL.md is missing.
        soul.unlink(missing_ok=True)
        marker.unlink()
    return home


_PROCESS_CLEANUP_TIMEOUT_S = 5
_RUNNING: set = set()


def kill_running_clis() -> int:
    """Kill every turn's process still running, and what each started; how many. The agent's shutdown does: the
    processes, in sessions of their own, would outlive its server."""
    running = [proc for proc in _RUNNING if proc.returncode is None]
    for proc in running:
        _kill_cli(proc)
    return len(running)


def _kill_cli(proc) -> None:
    """Kill the turn's process and everything under it: its process group, and, found by parent, the tools it starts
    in a session of their own, which no group or session kill reaches."""
    pid = getattr(proc, "pid", None)
    if pid is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return
    if proc.returncode is not None:
        return  # the process is gone, and its pid may be another process's by now
    tree = [pid, *_descendants(pid)]
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    for member in tree:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(member, signal.SIGKILL)


async def _started(spawn):
    """The process ``spawn`` starts. A turn cancelled while it starts lets it finish starting, then kills it and what
    it started: asyncio would kill the process alone, and leave its tools running."""
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
    """Wait for the killed process, but not for one that left its tree and still holds its pipes open."""
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


async def run_hermes(
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
    effort: str | None = None,
    skills_dir: Path = HERMES_ROOT / "agentenv-skills",
    event_callback: Callable[[dict], None] | None = None,
    workspace: str = "/app",
) -> HermesResult:
    """Run one Hermes turn in ``workspace``. ``event_callback`` gets each event as the plugin writes it: exactly the
    events of the returned trajectory, in order."""
    model = model or DEFAULT_MODEL
    attribution = _attribution_headers(project_id, task_id)
    if attribution:
        _record_attribution(project_id, task_id)
    schema = None
    if output_format is not None:
        schema = validate_output_format(output_format)
        prompt += (
            "\n\nReturn only a JSON object matching this schema, with no Markdown or surrounding text: "
            + json.dumps(schema, separators=(",", ":"))
        )
    config = build_config(model, mcp_servers, provider_headers(os.environ, attribution), role, skills_dir)
    home = await asyncio.to_thread(prepare_profile, context_id, config, system_prompt)

    with tempfile.TemporaryDirectory(prefix="hermes-turn-") as scratch:
        prompt_file, usage_file = Path(scratch) / "prompt.txt", Path(scratch) / "usage.json"
        prompt_file.write_text(prompt)
        cmd = [HERMES_BIN, "agentenv", "turn", "--prompt-file", str(prompt_file), "--provider", PROVIDER,
               "--model", model, "--usage-file", str(usage_file)]
        if session_id:
            cmd += ["--resume", session_id]
        if effort:
            cmd += ["--reasoning", effort]
        logger.info(f"Running hermes (session={session_id}, model={model}, effort={effort}, ctx={context_id[:12]}, "
                    f"mcp={list((mcp_servers or {}).keys())})")
        if attribution:
            logger.info(f"hermes attribution headers: {attribution}")

        read_end, write_end = os.pipe()
        try:
            proc = await _started(asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
                env={**os.environ, "HERMES_HOME": str(home), _EVENTS_FD_ENV: str(write_end)},
                pass_fds=(write_end,),
                start_new_session=True,
            ))
        except BaseException:
            os.close(read_end)
            raise
        finally:
            os.close(write_end)
        _RUNNING.add(proc)
        events = os.fdopen(read_end, "rb", buffering=0)
        parser = _EventParser(event_callback)
        try:
            stdout, stderr = await asyncio.wait_for(_run_turn(proc, events, parser.on_line), timeout=timeout_s)
        except asyncio.CancelledError:
            _kill_cli(proc)
            await _reap(proc)
            raise
        except asyncio.TimeoutError:
            _kill_cli(proc)
            await _reap(proc)
            return HermesResult(response="", tool_call_count=parser.tool_call_count, session_id=parser.session_id,
                                trajectory=parser.trajectory, error=f"hermes timed out after {timeout_s}s")
        except Exception:
            # A failure while reading, such as an event_callback that raises, must not leave the turn running.
            _kill_cli(proc)
            await _reap(proc)
            raise
        finally:
            _RUNNING.discard(proc)
            events.close()
        usage = _read_usage(usage_file)

    result = parser.result(stdout, stderr, proc.returncode, usage)
    if schema is not None and result.error is None:
        try:
            structured_output = json.loads(result.response)
            Draft202012Validator(schema).validate(structured_output)
        except (json.JSONDecodeError, ValidationError) as exc:
            result.error = f"Hermes returned a response that does not match output_format: {exc}"
            result.error_code = "invalid_output_format"
        else:
            result.structured_output = structured_output
    return result


def _read_usage(path: Path) -> dict:
    try:
        usage = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return usage if isinstance(usage, dict) else {}


async def _run_turn(proc, events, on_line: Callable[[str], None]) -> tuple[bytes, bytes]:
    """Read the event pipe line by line as the turn writes it, and the process's stdout and stderr, until it exits.
    The pipe is read on past the exit only briefly: a process the turn left behind may still hold it open."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=2**30)
    transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), events)

    async def events() -> None:
        while line := await reader.readline():
            on_line(line.decode(errors="replace"))

    reading = asyncio.create_task(events())
    try:
        stdout, stderr = await proc.communicate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(reading), _EVENTS_DRAIN_S)
        if reading.done():
            reading.result()
        return stdout, stderr
    finally:
        reading.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await reading
        transport.close()


class _EventParser:
    """Parse the plugin's JSON event lines one at a time.

    Every event carries its hook's name as ``type``. ``pre_tool_call`` events are the turn's tool calls,
    ``post_llm_call`` carries the final answer, and any event's ``session_id`` is the session. The whole event list is
    the trajectory, and each event is handed to ``event_callback`` as it is parsed.
    """

    def __init__(self, event_callback: Callable[[dict], None] | None = None) -> None:
        self._event_callback = event_callback
        self.trajectory: list[dict] = []
        self.tool_calls: set[str] = set()
        self.session_id: str | None = None
        self.final: str | None = None
        self.errors: list[str] = []
        self.error_code: str | None = None

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    def on_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.debug(f"hermes emitted a non-JSON event line: {line[:200]}")
            return
        if not isinstance(event, dict):
            return
        self.trajectory.append(event)
        if isinstance(event.get("session_id"), str) and event["session_id"]:
            self.session_id = event["session_id"]
        kind = event.get("type")
        if kind == "pre_tool_call":
            self.tool_calls.add(str(event.get("tool_call_id") or f"tool-{len(self.tool_calls)}"))
        elif kind == "post_llm_call" and isinstance(event.get("assistant_response"), str):
            self.final = event["assistant_response"]
        elif kind == "api_request_error":
            message = str(event.get("error") or event.get("reason") or "model request failed")
            if message not in self.errors:
                self.errors.append(message)
            if event.get("status_code") is not None:
                self.error_code = str(event["status_code"])
        if self._event_callback is not None:
            self._event_callback(event)

    def result(self, stdout: bytes, stderr: bytes, returncode: int | None, usage: dict) -> HermesResult:
        session_id = usage.get("session_id") or self.session_id
        response = (self.final if self.final is not None else stdout.decode(errors="replace")).strip()
        failed = returncode != 0 or usage.get("failed") or (usage and usage.get("completed") is False)
        err: str | None = None
        if failed:
            stderr_text = stderr.decode(errors="replace").strip()
            err = ("\n".join(self.errors) or usage.get("failure") or stderr_text
                   or (f"hermes stopped without completing the turn ({usage['turn_exit_reason']})"
                       if usage.get("turn_exit_reason") else f"hermes exited with code {returncode}")
                   )[:ERROR_MESSAGE_LIMIT]
            logger.error(f"hermes turn failed: {err}")
        return HermesResult(
            response=response,
            tool_call_count=self.tool_call_count,
            session_id=session_id,
            trajectory=self.trajectory,
            error=err,
            error_code=self.error_code if err else None,
            usage=usage,
        )
