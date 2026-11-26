"""One Antigravity turn: an ``Agent`` of the Antigravity SDK (``google-antigravity``), run in this process, its harness
pointed at a model relay of the turn's own.

The SDK's ``LocalOpenAIAgentConfig`` takes a chat-completions ``base_url`` and no key or headers, and its harness,
the Go ``localharness``, sends none, so the relay adds the endpoint's key, ``[model.headers]`` and the run's
attribution to every call. It also counts each call's usage, which the SDK never reports, and, while the agent
follows an armed clock, puts the env's time in the line the harness gives the model (``antigravity_date``). The
``agy`` CLI can't be used instead: it authenticates only by interactive OAuth and has no endpoint override.

The harness is a static binary, which agent-env's libfaketime can't reach, so it keeps real time and its conversation
store stays in order at any env date, while the commands it starts read the env's time. They go out through the
egress proxy while the clock is armed, as every clocked agent's do (``agent_egress``).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import google.antigravity as ag
from google.antigravity import types as agt
from google.antigravity.hooks import hooks as ag_hooks
from google.antigravity.hooks import policy as ag_policy

from agent_egress import clock_armed, egress_env
from agentenv_protocol.a2a_agent import BuiltinTools, Usage
from antigravity_date import Records, Tracked, env_time_rewrite, message_digest
from model_relay import Calls, model_relay

logger = logging.getLogger(__name__)

# The model a run uses when neither the task nor the agent-config names one, sent to the endpoint as named.
DEFAULT_MODEL = os.environ.get("ANTIGRAVITY_MODEL", "anthropic/claude-sonnet-5-5")
ERROR_MESSAGE_LIMIT = 4000
ROLE_HEADER = "AgentEnv-Role"
# What agent-env gives an agent for [model.headers]: a JSON object of header names to values.
MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"
_TAGS_HEADER = "x-litellm-tags"
# The SDK version AntigravityTool names the tools of; a test holds it to requirements.txt's pin.
SDK_VERSION = "0.1.24"

HOME_DIR = Path(os.environ.get("ANTIGRAVITY_HOME") or Path.home() / ".antigravity")
# The SDK's conversations, one SQLite store per conversation id, the env times their turns were given and the
# images their messages carry.
SAVE_DIR = HOME_DIR / "conversations"
SKILLS_DIR = HOME_DIR / "skills"
ENV_TIMES = Records(SAVE_DIR / "env-times.json")
TURN_IMAGES = Records(SAVE_DIR / "turn-images.json")
# Each conversation's keys in those two, its turns' own, so a snapshot of one carries no other's.
CONVERSATION_RECORDS = Records(SAVE_DIR / "conversation-records.json")
# Each turn's images as the turn was given them, by their content, so the agent editing an attachment never changes what
# the user showed the model.
IMAGES_DIR = SAVE_DIR / "images"
# The characters of a value the SDK can't serialize that the trajectory keeps.
_REPR_CHARS = 2000

# MCP calls into a large env can be slow; the SDK's own read timeout is shorter than a first tools/list there takes.
_MCP_TIMEOUT_S = 60.0
_MCP_SSE_READ_TIMEOUT_S = 300.0
# A model call the endpoint answers with a transient error is retried by the SDK itself: six times, from 5 s, doubling.
_API_RETRY = agt.ModelAPIRetryConfig(max_retries=6, initial_sleep_duration_ms=5000, exponential_multiplier=2.0)
# The reasoning efforts chat completions takes, and how agent-env's higher ones map onto them.
_EFFORTS = {"none": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high",
            "xhigh": "high", "max": "high"}

# The SDK's own tools, by the names a task selects them with. `ask_question` is off on every turn, so a headless run
# never waits for a human, and so is `generate_image`, which calls Google with an OAuth credential no run has.
# `finish` ends a turn, so it can't be turned off.
AntigravityTool = Literal["list_directory", "search_directory", "find_file", "view_file", "create_file", "edit_file",
                          "run_command", "invoke_subagent", "run_workflow", "search_web", "read_url_content",
                          "schedule"]
_ALWAYS_OFF = (agt.BuiltinTools.ASK_QUESTION, agt.BuiltinTools.GENERATE_IMAGE)
_ALWAYS_ON = (agt.BuiltinTools.FINISH,)


@dataclass
class AntigravityResult:
    response: str = ""
    session_id: str | None = None
    tool_call_count: int = 0
    error: str | None = None
    error_code: str | None = None
    infra: bool = False
    usage: Usage | None = None


def _api_root(url: str) -> str:
    """``LITELLM_BASE_URL`` without a trailing ``/`` and one trailing ``/v1``, which the relay serves the SDK's
    ``<root>/v1/chat/completions`` from. Core's ``api_root``, copied since this image can't import core; a unit test
    keeps the copies equal."""
    parts = urlsplit(url.strip())
    return urlunsplit(parts._replace(path=parts.path.rstrip("/").removesuffix("/v1").rstrip("/")))


# The attribution the last turn sent, which /ext/attribution-probe reads back.
_last_attribution: dict[str, str | None] = {"project_id": None, "task_id": None, "last_seen_at_utc": None}
_attribution_lock = threading.Lock()


def _record_attribution(project_id: str | None, task_id: str | None) -> None:
    if not project_id and not task_id:
        return
    with _attribution_lock:
        _last_attribution.update(project_id=project_id or None, task_id=task_id or None,
                                 last_seen_at_utc=datetime.now(timezone.utc).isoformat())


def get_last_attribution_snapshot() -> dict[str, str | None]:
    with _attribution_lock:
        return dict(_last_attribution)


def attribution_headers(project_id: str | None, task_id: str | None) -> dict[str, str]:
    """LiteLLM cost-attribution headers, x-litellm-customer-id + x-litellm-tags; {} when nothing to attribute."""
    headers: dict[str, str] = {}
    if project_id:
        headers["x-litellm-customer-id"] = project_id
        headers[_TAGS_HEADER] = ",".join([f"projectId:{project_id}", *([f"taskId:{task_id}"] if task_id else [])])
    elif task_id:
        headers[_TAGS_HEADER] = f"taskId:{task_id}"
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


def endpoint_headers(environ: Mapping[str, str], attribution: Mapping[str, str]) -> dict[str, str]:
    """What every model call the relay forwards carries, the harness sending none: the endpoint's key, then its
    [model.headers], then the run's attribution, each replacing a header of the same name in any case, except that
    the run's x-litellm-tags join the ones [model.headers] name."""
    configured = {name.lower(): value for name, value in model_headers(environ.get(MODEL_HEADERS_ENV)).items()}
    if configured.get(_TAGS_HEADER) and _TAGS_HEADER in attribution:
        tags = [tag for tag in configured[_TAGS_HEADER].split(",") if tag]
        attribution = {**attribution, _TAGS_HEADER: ",".join(dict.fromkeys(
            [*tags, *attribution[_TAGS_HEADER].split(",")]))}
    key = environ.get("LITELLM_API_KEY") or ""
    return {**({"authorization": f"Bearer {key}"} if key else {}),
            **{name.lower(): value for layer in (configured, attribution) for name, value in layer.items()}}


def mcp_headers(headers: Mapping[str, str] | None, role: str | None) -> dict[str, str] | None:
    """A server's registered headers; the run's role, when it has one, replaces a role header they name."""
    merged = dict(headers or {})
    if role:
        merged = {name: value for name, value in merged.items() if name.lower() != ROLE_HEADER.lower()}
        merged[ROLE_HEADER] = role
    return merged or None


def capabilities(builtin_tools: BuiltinTools[AntigravityTool] | None, *, web_tools: bool) -> Any:
    """The SDK's capabilities for a turn: autonomous, with `ask_question` and `generate_image` off on every turn, and
    the task's selection of the rest; `finish` stays on whatever the selection, and ``web_tools`` False turns the
    SDK's web tools off whatever it keeps."""
    always_off = list(_ALWAYS_OFF)
    named = {tool.value: tool for tool in agt.BuiltinTools}
    web_off = [] if web_tools else [named[name] for name in _WEB_TOOLS]
    if builtin_tools is None:
        return ag.CapabilitiesConfig(agent_behavior=agt.AgentBehavior.AUTONOMOUS, disabled_tools=always_off + web_off)
    selectable = [named[name] for name in AntigravityTool.__args__]
    if builtin_tools.only is not None:
        enabled = [named[name] for name in builtin_tools.only if named[name] not in web_off] + list(_ALWAYS_ON)
        return ag.CapabilitiesConfig(agent_behavior=agt.AgentBehavior.AUTONOMOUS, enabled_tools=enabled)
    disabled = selectable if builtin_tools.disable == "*" else [named[name] for name in builtin_tools.disable]
    disabled += [tool for tool in web_off if tool not in disabled]
    return ag.CapabilitiesConfig(agent_behavior=agt.AgentBehavior.AUTONOMOUS, disabled_tools=always_off + disabled)


def conversation_store_exists(conversation_id: str | None) -> bool:
    """Whether the SDK's store for ``conversation_id`` is on disk: resuming one that isn't fails the harness's start
    rather than starting afresh."""
    return bool(conversation_id) and (SAVE_DIR / f"{conversation_id}.db").is_file()


def is_image(path: str) -> bool:
    return (mimetypes.guess_type(path)[0] or "").startswith("image/")


def media_part(path: str | Path) -> dict | None:
    """An image the model sees inline, as a chat-completions image part with a data URL; None for any other file, which
    the prompt names for the model's own tools to read, or one that is gone."""
    if not is_image(str(path)):
        return None
    try:
        data = base64.b64encode(Path(path).read_bytes()).decode()
    except OSError:
        logger.warning(f"image {path} is gone: the model's message goes without it")
        return None
    return {"type": "image_url", "image_url": {"url": f"data:{mimetypes.guess_type(str(path))[0]};base64,{data}"}}


def kept_image(path: str) -> str | None:
    """A copy of the image at ``path`` in ``IMAGES_DIR``, named by its content, which no edit of the original reaches;
    its name, or None when the image is gone."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        logger.warning(f"image {path} is gone: the model's message goes without it")
        return None
    name = hashlib.sha256(data).hexdigest() + Path(path).suffix.lower()
    if not (IMAGES_DIR / name).is_file():
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        staged = IMAGES_DIR / f".{name}.tmp"
        staged.write_bytes(data)
        staged.replace(IMAGES_DIR / name)
    return name


def _with_images(request: dict, images: list[str], recorded: Records | Tracked) -> bool:
    """Every user message's images, which the SDK's chat-completions backend drops: the turn's own go with its message,
    the last user message, kept as they were (``kept_image``) and recorded by that message's text, so the turns after
    it, which send it again from a store that never saw them, carry them too. Whether it changed the request."""
    users = [message for message in request.get("messages") or [] if isinstance(message, dict)
             and message.get("role") == "user"]
    if not users:
        return False
    if images:
        key = message_digest(_text_of(users[-1]))
        if recorded.get(key) is None:
            recorded.put(key, [name for name in map(kept_image, images) if name is not None])
    changed = False
    for message in users:
        names = recorded.get(message_digest(_text_of(message)))
        if not names:
            continue
        content = message.get("content")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
        present = {json.dumps(part, sort_keys=True) for part in parts if isinstance(part, dict)}
        added = [part for part in (media_part(IMAGES_DIR / name) for name in names)
                 if part is not None and json.dumps(part, sort_keys=True) not in present]
        if added:
            message["content"] = [*parts, *added]
            changed = True
    return changed


def _text_of(message: dict) -> str:
    """A message's text as the harness wrote it, by which its images are recorded."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(part.get("text") or "" for part in content or [] if isinstance(part, dict)
                   and part.get("type") == "text")


def turn_rewrite(effort: str | None, images: list[str], times: Records | Tracked | None = None,
                 turn_images: Records | Tracked | None = None) -> Callable[[bytes], Awaitable[bytes]]:
    """The relay's rewrite for a turn's model calls: its messages' images, the env's time (``antigravity_date``), then
    the turn's reasoning effort, which the SDK takes no setting for. ``times`` and ``turn_images`` are the records it
    reads and writes, ``ENV_TIMES`` and ``TURN_IMAGES`` by default."""
    turn_images = TURN_IMAGES if turn_images is None else turn_images
    date_rewrite = env_time_rewrite(ENV_TIMES if times is None else times)
    reasoning = _EFFORTS.get(effort or "")
    if effort and reasoning is None:
        logger.warning(f"effort {effort!r} isn't one chat completions takes; the turn runs without one")

    async def rewrite(body: bytes) -> bytes:
        try:
            request = json.loads(body)
        except ValueError:
            request = None
        if isinstance(request, dict) and _with_images(request, images, turn_images):
            body = json.dumps(request, separators=(",", ":")).encode()
        body = await date_rewrite(body)
        if reasoning is None:
            return body
        try:
            request = json.loads(body)
        except ValueError:
            return body
        if not isinstance(request, dict) or request.get("reasoning_effort") == reasoning:
            return body
        request["reasoning_effort"] = reasoning
        return json.dumps(request, separators=(",", ":")).encode()

    return rewrite


class TrajectoryRecorder:
    """A turn's native trajectory, ``antigravity-sdk/1``, appended to ``emit`` as the SDK streams it: each step's
    text and thought as one record, each tool call as the SDK streams it, and its result, which the SDK gives only to
    its hooks, after it as ``{"kind": "tool_result", ..., "call_ref": <the streamed call's id>}``. The hooks know a call
    by the model's id and the stream by a step's, so a call the SDK lets its hooks decide binds to the next streamed
    call of its name, which the SDK streams once the hook allows it; a result waits for its call to stream. The
    harness's own tools, such as its skill lookup, stream and give their result to the hooks undecided: such a result
    is tied to the earliest of those calls not yet tied to one."""

    def __init__(self, emit: Callable[[dict], None]) -> None:
        self._emit = emit
        self._buffer: dict | None = None
        self._deciding: list[tuple[str, str | None]] = []
        self._bound: dict[str, str | None] = {}
        self._undecided: list[str | None] = []
        self._waiting: dict[str, dict] = {}
        self.segments: list[list[tuple[Any, str]]] = [[]]
        self.tool_calls = 0

    def chunk(self, chunk: Any) -> None:
        kind = type(chunk).__name__
        record = {"type": kind, **_plain(chunk)}
        if kind in ("Text", "Thought"):
            if kind == "Text":
                self.segments[-1].append((record.get("step_index"), record.get("text") or ""))
            buffered = self._buffer
            same_step = buffered is not None and buffered.get("step_index") == record.get("step_index")
            if same_step and buffered["type"] == kind:
                buffered["text"] = (buffered.get("text") or "") + (record.get("text") or "")
                return
            self.flush()
            self._buffer = record
            return
        self.flush()
        self._emit(record)
        if kind == "ToolCall":
            self.tool_calls += 1
            self.segments.append([])
            streamed = record.get("id")
            decided = next((index for index, (_, name) in enumerate(self._deciding) if name == record.get("name")),
                           None)
            if decided is None:
                self._undecided.append(streamed)
                return
            call_id, _ = self._deciding.pop(decided)
            self._bound[call_id] = streamed
            if (waiting := self._waiting.pop(call_id, None)) is not None:
                self._emit({**waiting, "call_ref": streamed})

    def hooked_call(self, call: Any) -> None:
        if (call_id := getattr(call, "id", None)) is not None:
            name = getattr(call, "name", None)
            self._deciding.append((str(call_id), None if name is None else str(name)))

    def result(self, record: dict) -> None:
        call_id = str(record.get("id"))
        if call_id in self._bound:
            call_ref = self._bound.pop(call_id)
        elif any(deciding == call_id for deciding, _ in self._deciding):
            self._waiting[call_id] = record
            return
        else:
            call_ref = self._undecided.pop(0) if self._undecided else None
        self.flush()
        self._emit({**record, "call_ref": call_ref})

    def flush(self) -> None:
        if self._buffer is not None:
            self._emit(self._buffer)
            self._buffer = None

    def close(self) -> None:
        self.flush()
        for call_id in list(self._waiting):
            self._emit({**self._waiting.pop(call_id), "call_ref": None})

    def response(self) -> str:
        """The turn's answer: its last text after a tool call, else the last text it has, a paragraph per step. The
        model narrates between calls, and a verifier would read that as the outcome."""
        for segment in reversed(self.segments):
            steps: dict[Any, str] = {}
            for step, text in segment:
                steps[step] = steps.get(step, "") + text
            if answer := "\n\n".join(text.strip() for text in steps.values() if text.strip()):
                return answer
        return ""


def _plain(value: Any) -> dict:
    """A chunk or hook payload as JSON: pydantic's dump, with what it can't serialize as text."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return json.loads(json.dumps(dump(mode="json", exclude={"exception"}), default=str))
        except Exception:  # noqa: BLE001 - a result's value may be anything
            return {key: str(item) for key, item in dump(warnings=False).items() if key != "exception"}
    return {"repr": str(value)[:_REPR_CHARS]}


def _hooks(recorder: TrajectoryRecorder) -> list:
    """The turn's hooks, which the SDK runs only when its config names them: every call allowed and recorded, and its
    result, or its error, appended after it."""

    @ag_hooks.pre_tool_call_decide
    async def record_call(call: Any) -> Any:
        recorder.hooked_call(call)
        return agt.HookResult(allow=True)

    @ag_hooks.post_tool_call
    async def record_result(result: Any) -> None:
        recorder.result({"kind": "tool_result", **_plain(result)})

    @ag_hooks.on_tool_error
    async def record_error(error: Any) -> None:
        recorder.result({"kind": "tool_result", "id": getattr(error, "call_id", None),
                         "name": str(getattr(error, "tool_name", None)),
                         "server_name": getattr(error, "server_name", None), "result": None,
                         "error": str(error)[:ERROR_MESSAGE_LIMIT]})

    return [record_call, record_result, record_error]


async def run_antigravity(prompt: str, *, model: str | None, project_id: str | None = None,
                          task_id: str | None = None, effort: str | None = None,
                          attachment_paths: list[str] | None = None, **kwargs: Any) -> AntigravityResult:
    """Run one turn, its model calls through a relay of the turn's own that adds the endpoint's key and the run's
    attribution, counts every call the harness makes, its subagents' included, and tells the model the env's time on
    an armed clock. The turn's usage is theirs, a call answered without usage counted as unreported."""
    _record_attribution(project_id, task_id)
    images = [path for path in attachment_paths or [] if is_image(path)]
    if attachment_paths:
        prompt = f"{prompt}\n\nAttached files, also on disk for your tools:\n" + "\n".join(
            f"- {path}" for path in attachment_paths)
    calls = Calls()
    headers = endpoint_headers(os.environ, attribution_headers(project_id, task_id))
    times, turn_images = Tracked(ENV_TIMES), Tracked(TURN_IMAGES)
    started: list[Any] = []
    result = None
    try:
        async with model_relay(_api_root(os.environ.get("LITELLM_BASE_URL") or ""), calls, headers=headers,
                               rewrite=turn_rewrite(effort, images, times, turn_images)) as relay:
            result = await _turn(prompt, base_url=f"{relay}/v1", model=model or DEFAULT_MODEL, started=started,
                                 **kwargs)
    finally:
        # A cancelled turn's records go with its conversation too, for a snapshot taken next to carry.
        conversation = result.session_id if result is not None else _kept(started, kwargs.get("session_id"))
        if conversation:
            file_records(conversation, times.keys, turn_images.keys)
    result.usage = calls.usage(result.tool_call_count, model or DEFAULT_MODEL)
    return result


def _kept(started: list[Any], session_id: str | None) -> str | None:
    """The conversation a turn that didn't finish leaves for the next: the one the SDK started, which it names with the
    turn's first streamed step, once its store is on disk, else the one the turn resumed."""
    current = getattr(started[0], "conversation_id", None) if started else None
    return current if conversation_store_exists(current) else session_id


def file_records(conversation_id: str, env_times: set[str], turn_images: set[str]) -> None:
    """File a turn's record keys under its conversation, beside the ones its earlier turns filed."""
    filed = CONVERSATION_RECORDS.get(conversation_id) or {}
    CONVERSATION_RECORDS.put(conversation_id, {
        "env_times": sorted(set(filed.get("env_times") or []) | env_times),
        "turn_images": sorted(set(filed.get("turn_images") or []) | turn_images)})


async def _turn(prompt: str, *, base_url: str, model: str, session_id: str | None,
                mcp_servers: dict[str, dict] | None, system_prompt: str | None, timeout_s: int, role: str | None,
                builtin_tools: BuiltinTools[AntigravityTool] | None, max_turns: int | None, workspace: str,
                event_callback: Callable[[dict], None], started: list[Any], web_tools: bool) -> AntigravityResult:
    """One turn of the SDK; ``started`` gets the turn's ``Agent`` once it starts, so whoever awaits the turn knows its
    conversation however it ends."""
    recorder = TrajectoryRecorder(event_callback)
    servers = mcp_servers or {}
    urls = [os.environ.get("LITELLM_BASE_URL", ""), *(info["url"] for info in servers.values())]
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    config = dict(
        base_url=base_url, model=model, save_dir=str(SAVE_DIR), workspaces=[workspace],
        capabilities=capabilities(builtin_tools, web_tools=web_tools), policies=[ag_policy.allow_all()],
        hooks=_hooks(recorder),
        retry_config=agt.RetryConfig(api_retry=_API_RETRY),
        # Merged into the harness's environment, which the model's commands inherit: on an armed clock, the egress
        # proxy, which keeps their TLS working at the env's date.
        env=egress_env(os.environ, urls, armed=clock_armed()) or None,
        mcp_servers=[agt.McpStreamableHttpServer(name=name, url=info["url"],
                                                 headers=mcp_headers(info.get("headers"), role),
                                                 timeout=_MCP_TIMEOUT_S, sse_read_timeout=_MCP_SSE_READ_TIMEOUT_S)
                     for name, info in servers.items()],
        skills_paths=[str(SKILLS_DIR)] if SKILLS_DIR.is_dir() else [],
    )
    if system_prompt:
        # Appended to the harness's own instructions as a section, never in their place.
        config["system_instructions"] = system_prompt
    if max_turns:
        config["budget_config"] = agt.BudgetConfig(max_model_calls=max_turns, scope=agt.BudgetScope.FORWARD_LOOKING)
    if session_id:
        if conversation_store_exists(session_id):
            config["conversation_id"] = session_id
        else:
            logger.warning(f"conversation {session_id} has no store in {SAVE_DIR}: the turn starts a new one")
            session_id = None

    async def converse() -> tuple[str | None, Any]:
        async with ag.Agent(ag.LocalOpenAIAgentConfig(**config)) as agent:
            started.append(agent)
            response = await agent.chat(prompt)
            async for chunk in response.chunks:
                recorder.chunk(chunk)
            return agent.conversation_id, response.stop_reason

    try:
        conversation_id, stop = await asyncio.wait_for(converse(), timeout=timeout_s)
    except TimeoutError:
        recorder.close()
        return AntigravityResult(session_id=_kept(started, session_id), tool_call_count=recorder.tool_calls,
                                 error_code="timeout",
                                 error=f"the turn ran past its {timeout_s} s")
    except agt.AntigravityExecutionError as exc:
        recorder.close()
        text = str(exc)[:ERROR_MESSAGE_LIMIT]
        # A task env's MCP server that fails its initialize fails the harness's start: the run's tools aren't there.
        mcp = "mcp load failed" in text.lower()
        return AntigravityResult(session_id=_kept(started, session_id), tool_call_count=recorder.tool_calls, error=text,
                                 error_code="mcp_load_failed" if mcp else "antigravity_error", infra=mcp)
    recorder.close()
    result = AntigravityResult(response=recorder.response(), session_id=conversation_id or session_id,
                               tool_call_count=recorder.tool_calls)
    if stop == agt.StopReason.MAX_MODEL_CALLS_EXCEEDED:
        result.error_code, result.error = "max_turns_exceeded", f"the turn stopped at its max_turns, {max_turns}"
    elif stop not in (None, agt.StopReason.UNSPECIFIED):
        result.error_code, result.error = "stopped", f"the SDK stopped the turn: {getattr(stop, 'name', stop)}"
    return result


# The SDK's own web tools: what a deploy_agent web_tools False turns off.
_WEB_TOOLS = ("search_web", "read_url_content")
