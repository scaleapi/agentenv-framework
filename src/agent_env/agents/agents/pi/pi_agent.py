"""The pi coding agent (https://github.com/earendil-works/pi) as an AgentEnv A2A agent.

Each task runs ``pi --mode json`` once, in the workspace, against a per-task pi config dir that points pi
at the configured LiteLLM endpoint and the registered MCP servers. A context maps to one pi session id,
so later tasks in the context resume the same conversation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import signal
import struct
import tarfile
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
import ipaddress
import socket
from agentenv_protocol.a2a_agent import (
    INSTALL_V1,
    MCP_CONFIG_V1,
    PEER_AGENTS_V1,
    SKILL_CONFIG_V1,
    SNAPSHOT_V1,
    TRAJECTORY_V1,
    TRIGGERS_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    BundleSkillRequest,
    DataPart,
    FilePart,
    InlineSkillRequest,
    McpAddRequest,
    NamespaceChangelogEnableRequest,
    ObjectChangelogApplyRequest,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    PeerAgentsSetRequest,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    WriteOnly,
    a2a_agent,
    download,
    enable,
    extension,
    upload,
)
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.routing import Route

from changelog import ChangelogCapture, apply as apply_increment
from peers import Peers

PI_VERSION = "1.1.0"
PROVIDER = "agentenv"
TRAJECTORY_FORMAT = "pi-json-events/v1"
# JSON object of request-body fields every model call carries, set per registration (``put --env-var``)
# for fields agent-env will not forward as ``model_params``, such as ``user``.
MODEL_PARAMS_ENV = "PI_A2A_MODEL_PARAMS"
# The model when agent-config sets none, as for a peer that other agents message directly.
DEFAULT_MODEL_ENV = "PI_A2A_MODEL"
# Colon-separated directories changelog capture watches and replay may write; a replay agent trusts only these.
CHANGELOG_ROOTS_ENV = "PI_A2A_CHANGELOG_ROOTS"
A2A_PORT = int(os.environ.get("A2A_PORT", "8000"))
PEER_MCP_NAME = "peers"
PEER_MCP_PATH = "/mcp"
PEER_MCP_URL = f"http://127.0.0.1:{A2A_PORT}{PEER_MCP_PATH}"
INSTALL_DIR = "/opt/pi-a2a"
ENV_FILE = "agent.env"
STREAM_LIMIT_BYTES = 64 * 1024 * 1024
STDERR_TAIL_BYTES = 8 * 1024
FETCH_TIMEOUT_SECONDS = 120
MAX_ATTACHMENT_BYTES = 256 * 1024 * 1024
# The peer MCP server's header naming the run, whose value pi reads from its environment, never from disk.
PEER_RUN_HEADER = "X-Pi-A2A-Run"
PEER_RUN_ENV = "AGENTENV_MCP_PEER_RUN"
# Deltas are reconstructible from the message_end records, so they are left out of the trajectory.
_UNRECORDED_EVENTS = frozenset({"message_update", "tool_execution_update"})
# A context id passes through as its session id only when it has none of pi's ``.`` and ``_``; every derived id
# contains ``_``, so the two never meet.
_SESSION_ID_PLAIN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?")
_SESSION_ID_UNSAFE = re.compile(r"[^A-Za-z0-9-]")
# pi attaches images as images and inlines text; any other file is left on disk for its tools.
_TEXT_MEDIA_TYPES = frozenset({"application/json", "application/xml", "application/yaml", "application/x-yaml"})

# sudo only where it needs no password: VM sandboxes have it, a local sandbox runs docker as the user.
_DOCKER = "$(sudo -n true 2>/dev/null && echo sudo) docker"
INSTALL_COMMANDS = [
    f"{_DOCKER} cp {{agent_ctx_tar}} {{container}}:/tmp/pi-a2a-ctx.tar.gz",
    f"{_DOCKER} exec {{container}} sh -c 'mkdir -p {INSTALL_DIR} && tar -xf /tmp/pi-a2a-ctx.tar.gz -C {INSTALL_DIR}"
    " && rm /tmp/pi-a2a-ctx.tar.gz'",
    f"{_DOCKER} exec {{container}} sh {INSTALL_DIR}/install.sh",
    # The credentials travel on a heredoc into a 0600 file: the installer logs only a command's first line,
    # and neither a command line nor the container's config ever holds them.
    f"{_DOCKER} exec -i {{container}} sh -c 'umask 077 && cat > {INSTALL_DIR}/{ENV_FILE}' <<'PI_A2A_ENV'\n"
    "LITELLM_API_KEY={litellm_api_key}\n"
    "LITELLM_BASE_URL={litellm_base_url}\n"
    "PI_A2A_ENV",
    f"{_DOCKER} exec -d -e A2A_PORT={{a2a_port}} {{container}} sh {INSTALL_DIR}/start.sh",
]


class PiConfig(AgentConfig):
    model: str | None = None
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    effort: Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    tools: str | None = None
    context_window: int = 200_000
    max_tokens: int = 32_000
    model_params: WriteOnly[dict[str, Any] | None] = None
    timeout_seconds: int = 1800


@dataclass
class _Run:
    session_id: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    assistant_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0


def _session_id(context_id: str) -> str:
    """A pi session id (``[A-Za-z0-9._-]``, alphanumeric at both ends), one-to-one with the context: a plain
    context id as-is, anything else a readable prefix and a hash of the context id."""
    if _SESSION_ID_PLAIN.fullmatch(context_id):
        return context_id
    prefix = _SESSION_ID_UNSAFE.sub("-", context_id).strip("-")[:48].rstrip("-") or "context"
    return f"{prefix}_{hashlib.sha256(context_id.encode()).hexdigest()[:32]}"


def _models_json(config: PiConfig, base_url: str, default_params: Mapping[str, Any]) -> dict[str, Any]:
    reasoning = config.effort not in (None, "off")
    model: dict[str, Any] = {
        "id": config.model,
        "reasoning": reasoning,
        "input": ["text", "image"],
        "contextWindow": config.context_window,
        "maxTokens": config.max_tokens,
    }
    params = {**default_params, **(config.model_params or {})}
    if params:
        # pi merges samplingParams into the request body, so deployment params such as attribution reach the proxy.
        model["samplingParams"] = params
    return {
        "providers": {
            PROVIDER: {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": "${LITELLM_API_KEY}",
                "models": [model],
            }
        }
    }


def _mcp_json(
    mcp_servers: Mapping[str, Any], timeout_seconds: int, peer_run: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """pi's ``mcp.json`` and the env vars carrying its header values.

    Header values are passed as ``${VAR}`` references so they never reach disk and a value starting
    with ``!`` is not run as a command by pi's config resolver. ``direct`` exposure declares the tools to
    the model; pi's default ``codemode`` would hide them behind a JS sandbox."""
    servers: dict[str, Any] = {}
    env: dict[str, str] = {}
    for index, (name, registration) in enumerate(mcp_servers.items()):
        headers = {}
        for header_index, (header, value) in enumerate((registration.get("headers") or {}).items()):
            variable = f"AGENTENV_MCP_{index}_HEADER_{header_index}"
            env[variable] = value
            headers[header] = "${" + variable + "}"
        if registration["url"] == PEER_MCP_URL:
            env[PEER_RUN_ENV] = peer_run
            headers[PEER_RUN_HEADER] = "${" + PEER_RUN_ENV + "}"
        servers[name] = {"type": "http", "url": registration["url"], "exposure": "direct", "timeout": timeout_seconds}
        if headers:
            servers[name]["headers"] = headers
    return {"mcpServers": servers}, env


#: Set by the image (root.conf) where the egress proxy runs: the CLI and the model's commands go out through it
#: while the agent follows an armed clock.
EGRESS_PROXY_ENV = "AGENTENV_EGRESS_PROXY"
#: The clock page agent-env's libfaketime reads the CLI's time from (clock/clockpage.py, version 2): it keeps the last
#: armed clock while the follower reconnects or restarts, which the follower's status doesn't say.
CLOCK_PAGE = "/dev/shm/agentenv-clock"
_PAGE = struct.Struct("<IIQqqdIIQQqq")  # magic version seq universe_ns anchor_ns rate armed reserved generation check ...
_PAGE_MAGIC, _PAGE_VERSION, _MASK = 0x4B434541, 2, 0xFFFFFFFFFFFFFFFF
#: The constant every page's check is salted with (clockpage.check).
_PAGE_CHECK_SALT = 0x9E3779B97F4A7C15
#: How often, and how far apart, a read retries a page a write is under way on: 50 ms in all, far longer than a write.
_PAGE_READ_TRIES, _PAGE_RETRY_S = 50, 0.001
_DIRECT_HOSTS = ("localhost", "127.0.0.1", "::1", "host.docker.internal")


def clock_armed(page_path: str | None = None) -> bool:
    """Whether the clock page holds an armed clock, so the CLI started now reads the env's time: the page itself,
    as libfaketime reads it, retried while a write is under way (seq odd) or its check doesn't match."""
    try:
        with open(page_path or CLOCK_PAGE, "rb") as page:
            for _ in range(_PAGE_READ_TRIES):
                page.seek(0)
                raw = page.read(_PAGE.size)
                if len(raw) < _PAGE.size:
                    return False
                magic, version, seq, universe, anchor, rate, armed, _, generation, checked, mono_u, mono_a = \
                    _PAGE.unpack(raw)
                if magic != _PAGE_MAGIC or version != _PAGE_VERSION:
                    return False
                rate_bits = struct.unpack("<Q", struct.pack("<d", rate))[0]
                expected = (seq ^ (universe & _MASK) ^ (anchor & _MASK) ^ rate_bits ^ armed ^ generation
                            ^ (mono_u & _MASK) ^ (mono_a & _MASK) ^ _PAGE_CHECK_SALT) & _MASK
                if seq % 2 == 0 and checked == expected:
                    return bool(armed)
                time.sleep(_PAGE_RETRY_S)
    except OSError:
        return False
    return bool(armed)  # a writer that died mid-write: the next write takes the page up; its armed flag says the most


#: How long a CLI's start waits for its model host's addresses, once per host per process (``_GLOBAL_HOSTS``): the
#: lookup runs on a thread of its own, so a slow resolver holds the server's loop this long at most, once.
_RESOLVE_TIMEOUT_S = 2.0
_GLOBAL_HOSTS: dict[str, bool] = {}
_RESOLVER = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-host")


def _global_host(host: str) -> bool:
    """Whether ``host`` is a public address, or a name every address of resolves to one: a host the egress proxy
    reaches. A name that doesn't resolve here, or not within ``_RESOLVE_TIMEOUT_S``, is taken as public, for the proxy
    to resolve; each name's answer is kept."""
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        pass
    if host not in _GLOBAL_HOSTS:
        try:
            infos = _RESOLVER.submit(socket.getaddrinfo, host, None).result(timeout=_RESOLVE_TIMEOUT_S)
            addresses = {info[4][0] for info in infos}
            _GLOBAL_HOSTS[host] = all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addresses)
        except (OSError, concurrent.futures.TimeoutError):
            _GLOBAL_HOSTS[host] = True
    return _GLOBAL_HOSTS[host]


def model_direct_hosts(base_url: str | None) -> list[str]:
    """The model endpoint's host when the CLI reaches it directly on an armed clock: over plain HTTP, which has no
    certificate to check at the env's date, or at an address that isn't public, which the egress proxy refuses."""
    endpoint = urlsplit(base_url or "")
    host = endpoint.hostname or ""
    return [host] if host and (endpoint.scheme == "http" or not _global_host(host)) else []


def egress_env(environ: Mapping[str, str], direct: Iterable[str] = (), *, armed: bool = False) -> dict[str, str]:
    """The egress proxy for the CLI and what it starts, where the image runs one (FD-5916) and the clock is ``armed``:
    they read the env's time, so a client checking a live site's certificate against it would refuse the site once
    the clock is months off, and the proxy, on real time, hands it a certificate that holds then. The env's MCP servers
    (``direct``) are reached directly: the proxy refuses an address that isn't global, as a local or private gateway's
    is. Unarmed, the CLI reads real time and goes out directly, as an image without the proxy does, so a model endpoint
    or mirror at a private address keeps working. Hosts the agent was deployed to reach directly (``NO_PROXY``,
    ``no_proxy``) stay direct."""
    proxy = environ.get(EGRESS_PROXY_ENV)
    if not proxy or not armed:
        return {}
    configured = [host.strip() for name in ("NO_PROXY", "no_proxy") for host in (environ.get(name) or "").split(",")]
    hosts = ",".join(dict.fromkeys([*_DIRECT_HOSTS, *filter(None, configured), *filter(None, direct)]))
    return {**{name: proxy for base in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") for name in (base, base.lower())},
            "NO_PROXY": hosts, "no_proxy": hosts, "NODE_USE_ENV_PROXY": "1"}


def cli_env(environ: Mapping[str, str], base_url: str, mcp_servers: Mapping[str, Any] | None = None, *,
            clock_armed: bool = False) -> dict[str, str]:
    """The pi CLI's environment, before its run's own variables: this process's, and on an armed clock
    (``clock_armed``) the egress proxy (``egress_env``). The env's MCP servers, which the CLI calls itself over HTTP
    (mcp.json), are reached directly, and so is a model endpoint over plain HTTP, which has no certificate to check at
    the env's date and, at a private address, the proxy would refuse."""
    model_host = model_direct_hosts(base_url) if clock_armed else []
    mcp_hosts = [urlsplit(str(entry.get("url") or "")).hostname or "" for entry in (mcp_servers or {}).values()]
    return {**environ, **egress_env(environ, [*mcp_hosts, *model_host], armed=clock_armed)}


def _prompt_text(parts: Sequence[Any], unattached: Sequence[tuple[Path, str]]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part, TextPart):
            chunks.append(part.text)
        elif isinstance(part, DataPart):
            chunks.append(json.dumps(part.data, indent=2, default=str))
    chunks += [f"[Attached file ({media_type}) saved at {path}]" for path, media_type in unattached]
    return "\n\n".join(chunks)


def _media_type(part: FilePart) -> str:
    return part.mime_type or mimetypes.guess_type(part.name or "")[0] or "application/octet-stream"


def _attachable(media_type: str) -> bool:
    return media_type.startswith(("image/", "text/")) or media_type in _TEXT_MEDIA_TYPES


async def _materialize(part: FilePart, index: int, directory: Path) -> Path:
    """Write a file part to disk; inline bytes and HTTP(S) URLs only."""
    name = Path(part.name).name if part.name else ""
    destination = directory / f"{index}-{name or 'file'}"
    if part.bytes is not None:
        destination.write_bytes(base64.b64decode(part.bytes))
        return destination
    if urlsplit(part.uri).scheme not in {"http", "https"}:
        raise ValueError(f"unsupported file URI scheme for {name or 'file'}")
    received = 0
    async with (
        httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client,
        client.stream("GET", part.uri) as response,
    ):
        response.raise_for_status()
        with destination.open("wb") as handle:
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > MAX_ATTACHMENT_BYTES:
                    raise ValueError(f"{name or 'file'} is larger than {MAX_ATTACHMENT_BYTES} bytes")
                handle.write(chunk)
    return destination


def _record(run: _Run, event: dict[str, Any]) -> None:
    kind = event.get("type")
    if kind == "session":
        run.session_id = event.get("id")
    elif kind == "message_end" and (event.get("message") or {}).get("role") == "assistant":
        run.assistant_messages.append(event["message"])
    elif kind == "tool_execution_end":
        run.tool_calls += 1
    if kind not in _UNRECORDED_EVENTS:
        run.events.append(event)


def _usage(run: _Run) -> Usage:
    totals = {"input": 0, "output": 0, "totalTokens": 0, "cacheRead": 0, "cacheWrite": 0}
    cost = 0.0
    for message in run.assistant_messages:
        usage = message.get("usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
        cost += float((usage.get("cost") or {}).get("total") or 0)
    return Usage(
        tool_call_count=run.tool_calls,
        input_tokens=totals["input"],
        output_tokens=totals["output"],
        total_tokens=totals["totalTokens"],
        cost_usd=cost or None,
        provider_details={"cache_read_tokens": totals["cacheRead"], "cache_write_tokens": totals["cacheWrite"]},
    )


def _text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content or () if block.get("type") == "text")


def _result(run: _Run, returncode: int, stderr: str, session_id: str) -> TaskResult:
    last = run.assistant_messages[-1] if run.assistant_messages else None
    builder = (
        TaskResult.builder()
        .session_ref(run.session_id or session_id)
        .usage(_usage(run))
        .native_trajectory(format=TRAJECTORY_FORMAT, payload=run.events)
    )
    stop_reason = last.get("stopReason") if last else None
    if last is not None and returncode == 0 and stop_reason not in ("error", "aborted"):
        return builder.succeeded().add_text(_text(last)).build()
    if stop_reason == "aborted":
        message = last.get("errorMessage") or "pi aborted the run"
        builder.failed("pi.aborted", message)
    elif stop_reason == "error":
        message = last.get("errorMessage") or stderr or "pi reported a model error"
        builder.failed("pi.model_error", message, error_type="infra_error")
    else:
        message = stderr or f"pi exited with status {returncode} without an assistant message"
        builder.failed("pi.exited", message, error_type="infra_error")
    return builder.add_text(message).build()


async def _tail(stream: asyncio.StreamReader, limit: int) -> bytes:
    """Drain ``stream``, keeping only its last ``limit`` bytes."""
    kept = bytearray()
    while chunk := await stream.read(64 * 1024):
        kept += chunk
        del kept[:-limit]
    return bytes(kept)


def _kill(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _session_header(path: Path) -> dict[str, Any] | None:
    with path.open("rb") as handle:
        line = handle.readline()
    try:
        header = json.loads(line)
    except json.JSONDecodeError:
        return None
    return header if isinstance(header, dict) and header.get("type") == "session" else None


def _pack(directory: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        for child in sorted(directory.iterdir()):
            tar.add(child, arcname=child.name)


def _unpack(archive: Path, directory: Path) -> None:
    """Replace ``directory``'s contents with the archive's. The archive is extracted into a staging directory
    inside it first, so one that fails to extract leaves the contents as they were."""
    staging = directory / f".pi-a2a-restore-{uuid.uuid4().hex}"
    try:
        # An empty workspace's archive has no members, so extraction creates nothing.
        staging.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(staging, filter="data")
        for child in directory.iterdir():
            if child == staging:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        for child in staging.iterdir():
            child.rename(directory / child.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@a2a_agent(
    identity=AgentIdentity(
        name="pi",
        description="The pi coding agent: reads, edits and runs code in its workspace with MCP tools and skills.",
        version=PI_VERSION,
        input_modes=("text", "image/png", "image/jpeg", "image/gif", "application/pdf"),
    ),
    config=PiConfig,
    config_description="Configure the pi runtime: model, effort, system prompt and tool allowlist.",
    extensions=(
        MCP_CONFIG_V1,
        enable(TRAJECTORY_V1, description="The pi JSON-mode event stream of a completed task."),
        enable(
            SNAPSHOT_V1,
            description=(
                "The trajectory object is the context's pi session (JSONL), the workspace object a gzipped tar. "
                "Changelog increments are tars of changed files plus the session, one per tool call."
            ),
        ),
        enable(
            PEER_AGENTS_V1,
            description="Peers are offered to pi as the peer_list and peer_send_message tools of a loopback MCP server.",
        ),
        enable(
            INSTALL_V1,
            install_commands=INSTALL_COMMANDS,
            required_params=["container", "agent_ctx_tar", "a2a_port", "litellm_api_key", "litellm_base_url"],
            a2a_port=8000,
        ),
        TRIGGERS_V1,
    ),
)
class PiAgent(AgentEnvAgent):
    def __init__(
        self,
        *,
        home: Path | None = None,
        workspace: Path | None = None,
        pi_command: Sequence[str] = ("pi",),
    ) -> None:
        self.home = home or Path.home() / ".pi-a2a"
        self.workspace = workspace or Path("/workspace")
        self.pi_command = tuple(pi_command)
        self.skills_dir = self.home / "skills"
        self.sessions_dir = self.home / "sessions"
        self.peers = Peers()
        self.changelog: ChangelogCapture | None = None
        # A changelog is one sequence of tool calls, so while one is captured, runs take turns.
        self._changelog_turn = asyncio.Lock()
        # Restoring the workspace waits for no task to be running in it; tasks and saves wait for a restore to finish.
        self._workspace_lock = asyncio.Lock()
        self._running = 0

    def create_app(self) -> Starlette:
        app = super().create_app()
        if not any(getattr(route, "path", None) == PEER_MCP_PATH for route in app.router.routes):
            app.router.routes.append(Route(PEER_MCP_PATH, self.peers.mcp, methods=["GET", "POST", "DELETE"]))
        return app

    async def run(self, request: TaskRequest[PiConfig]) -> TaskResult:
        config = request.config
        if not config.model and (fallback := os.environ.get(DEFAULT_MODEL_ENV)):
            config = config.model_copy(update={"model": fallback})
        if not config.model:
            return TaskResult.failure("pi.no_model", "no model is configured; set 'model' through agent-config")
        base_url = os.environ.get("LITELLM_BASE_URL")
        if not base_url:
            return TaskResult.failure("pi.no_endpoint", "LITELLM_BASE_URL is not set", error_type="infra_error")

        run_dir = self.home / "runs" / request.task_id
        agent_dir = run_dir / "agent"
        files_dir = run_dir / "files"
        for directory in (agent_dir, files_dir, self.sessions_dir, self.skills_dir):
            directory.mkdir(parents=True, exist_ok=True)
        cwd = request.workspace or self.workspace
        cwd.mkdir(parents=True, exist_ok=True)
        async with self._workspace_lock:
            self._running += 1
        try:
            with self.peers.run(request.context_id) as peer_run:
                if self.changelog is None:
                    return await self._run_pi(request, config, base_url, agent_dir, files_dir, cwd, peer_run)
                async with self._changelog_turn:
                    return await self._run_pi(request, config, base_url, agent_dir, files_dir, cwd, peer_run)
        finally:
            self._running -= 1
            shutil.rmtree(run_dir, ignore_errors=True)

    async def _run_pi(
        self,
        request: TaskRequest[PiConfig],
        config: PiConfig,
        base_url: str,
        agent_dir: Path,
        files_dir: Path,
        cwd: Path,
        peer_run: str,
    ) -> TaskResult:
        mcp_config, mcp_env = _mcp_json(request.mcp_servers, config.timeout_seconds, peer_run)
        default_params = json.loads(os.environ.get(MODEL_PARAMS_ENV) or "{}")
        (agent_dir / "models.json").write_text(json.dumps(_models_json(config, base_url, default_params)))
        (agent_dir / "mcp.json").write_text(json.dumps(mcp_config))

        attached: list[Path] = []
        unattached: list[tuple[Path, str]] = []
        try:
            for index, part in enumerate(request.parts):
                if isinstance(part, FilePart):
                    path = await _materialize(part, index, files_dir)
                    media_type = _media_type(part)
                    if _attachable(media_type):
                        attached.append(path)
                    else:
                        unattached.append((path, media_type))
        except (ValueError, httpx.HTTPError) as exc:
            return TaskResult.failure("pi.attachment", f"could not read a file part: {exc}")

        session_id = request.session_ref or _session_id(request.context_id)
        argv = [
            *self.pi_command,
            "--mode", "json",
            "--session-id", session_id,
            "--session-dir", str(self.sessions_dir),
            "--provider", PROVIDER,
            "--model", config.model,
            "--no-approve",
        ]
        if config.effort:
            argv += ["--thinking", config.effort]
        if config.system_prompt:
            argv += ["--system-prompt", config.system_prompt]
        if config.append_system_prompt:
            argv += ["--append-system-prompt", config.append_system_prompt]
        if config.tools:
            argv += ["--tools", config.tools]
        for skill in request.skills:
            skill_dir = self.skills_dir / str(skill["name"])
            if skill_dir.is_dir():
                argv += ["--skill", str(skill_dir)]
        argv += [f"@{path}" for path in attached]

        env = {
            **cli_env(os.environ, base_url, request.mcp_servers, clock_armed=clock_armed()),
            **mcp_env,
            "PI_CODING_AGENT_DIR": str(agent_dir),
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
        }
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=STREAM_LIMIT_BYTES,
        )
        run = _Run()
        stderr_task = asyncio.create_task(_tail(process.stderr, STDERR_TAIL_BYTES))
        try:
            async with asyncio.timeout(config.timeout_seconds):
                process.stdin.write(_prompt_text(request.parts, unattached).encode())
                await process.stdin.drain()
                process.stdin.close()
                async for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    _record(run, event)
                    if self.changelog is not None and _is_tool_result(event):
                        await self.changelog.capture(
                            self._session_file(session_id), context_id=request.context_id, session_id=session_id
                        )
                returncode = await process.wait()
                stderr = (await stderr_task).decode(errors="replace").strip()
                if self.changelog is not None and not await self.changelog.flush():
                    return (
                        TaskResult.builder()
                        .failed(
                            "pi.changelog_incomplete",
                            "a changelog increment could not be uploaded",
                            error_type="infra_error",
                        )
                        .add_text("a changelog increment could not be uploaded")
                        .session_ref(run.session_id or session_id)
                        .usage(_usage(run))
                        .native_trajectory(format=TRAJECTORY_FORMAT, payload=run.events)
                        .build()
                    )
        except TimeoutError:
            _kill(process)
            stderr_task.cancel()
            await process.wait()
            return (
                TaskResult.builder()
                .failed("pi.timeout", f"pi did not finish within {config.timeout_seconds}s")
                .add_text(f"pi did not finish within {config.timeout_seconds}s")
                .session_ref(run.session_id or session_id)
                .usage(_usage(run))
                .native_trajectory(format=TRAJECTORY_FORMAT, payload=run.events)
                .build()
            )
        except BaseException:
            _kill(process)
            stderr_task.cancel()
            raise
        return _result(run, returncode, stderr, session_id)

    def _changelog_roots(self) -> list[str]:
        """The registration's changelog roots, absolute and canonical: all a capture may watch and a replay write."""
        configured = os.environ.get(CHANGELOG_ROOTS_ENV)
        roots = configured.split(":") if configured else [str(self.workspace), "/app"]
        return [os.path.realpath(root) for root in roots]

    def _require_idle(self) -> None:
        if self._running:
            raise HTTPException(status_code=409, detail="a task is running; restore the workspace once it finishes")

    def _session_file(self, session_id: str) -> Path | None:
        if not self.sessions_dir.is_dir():
            return None
        for path in self.sessions_dir.glob("*.jsonl"):
            header = _session_header(path)
            if header is not None and header.get("id") == session_id:
                return path
        return None

    def _install_session(self, body: bytes, context_id: str) -> None:
        """Make ``body`` (a pi session) the conversation of ``context_id``, re-identified for this agent."""
        lines = body.decode().splitlines()
        header = json.loads(lines[0])
        session_id = _session_id(context_id)
        header.update(id=session_id, cwd=str(self.workspace))
        existing = self._session_file(session_id)
        if existing is not None:
            existing.unlink()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
        (self.sessions_dir / f"{stamp}_{session_id}.jsonl").write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
        self.set_session_ref_for_context(context_id, session_id)

    @extension(SNAPSHOT_V1.save)
    async def save_snapshot(self, request: ObjectSnapshotSaveRequest) -> dict[str, Any]:
        session_id = self.session_ref_for_context(request.context_id) or _session_id(request.context_id)
        with tempfile.TemporaryDirectory() as scratch:
            # The conversation and the workspace are copied together, so a restore cannot fall between them.
            copy = Path(scratch) / "session.jsonl"
            archive = Path(scratch) / "workspace.tar.gz"
            async with self._workspace_lock:
                session = self._session_file(session_id)
                if session is None:
                    raise HTTPException(status_code=404, detail=f"No conversation for context '{request.context_id}'")
                shutil.copyfile(session, copy)
                if request.objects.workspace is not None:
                    self.workspace.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(_pack, self.workspace, archive)
            objects = {"trajectory": (await upload(request.objects.trajectory, copy)).model_dump(mode="json")}
            if request.objects.workspace is not None:
                objects["workspace"] = (await upload(request.objects.workspace, archive)).model_dump(mode="json")
        return {"context_id": request.context_id, "objects": objects}

    @extension(SNAPSHOT_V1.load)
    async def load_snapshot(self, request: ObjectSnapshotLoadRequest) -> dict[str, Any]:
        context_id = request.target_context_id or uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as scratch:
            session = Path(scratch) / "session.jsonl"
            await download(request.objects.trajectory, session)
            if request.objects.workspace is not None:
                archive = Path(scratch) / "workspace.tar.gz"
                await download(request.objects.workspace, archive)
            async with self._workspace_lock:
                self._require_idle()
                if request.objects.workspace is not None:
                    self.workspace.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(_unpack, archive, self.workspace)
                self._install_session(session.read_bytes(), context_id)
        return {"context_id": context_id}

    @extension(SNAPSHOT_V1.changelog.enable)
    async def enable_changelog(self, request: NamespaceChangelogEnableRequest) -> dict[str, Any]:
        allowed = self._changelog_roots()
        if request.roots is None:
            roots = allowed
        else:
            if not all(os.path.isabs(root) for root in request.roots):
                raise HTTPException(status_code=400, detail="changelog roots must be absolute paths")
            roots = [os.path.realpath(root) for root in request.roots]
            outside = [root for root in roots if not any(Path(root).is_relative_to(base) for base in allowed)]
            if outside:
                raise HTTPException(status_code=400, detail=f"changelog roots outside this agent's roots: {outside}")
        capture = ChangelogCapture(request.write_namespace, roots)
        await capture.start()
        self.changelog = capture
        return {"roots": roots}

    @extension(SNAPSHOT_V1.changelog.apply)
    async def apply_changelog(self, request: ObjectChangelogApplyRequest) -> dict[str, Any]:
        session = None
        with tempfile.TemporaryDirectory() as scratch:
            paths = []
            for item in request.increments:
                paths.append(Path(scratch) / f"{item.sequence:06d}.tar")
                await download(item.object, paths[-1])
            async with self._workspace_lock:
                self._require_idle()
                for path in paths:
                    applied = await asyncio.to_thread(apply_increment, path, self._changelog_roots())
                    session = applied.session or session
        if not request.resume_conversation:
            return {"count": len(request.increments)}
        context_id = request.target_context_id or uuid.uuid4().hex
        if session is not None:
            self._install_session(session, context_id)
        return {"count": len(request.increments), "context_id": context_id}

    @extension(PEER_AGENTS_V1.set)
    async def set_peers(self, request: PeerAgentsSetRequest) -> dict[str, Any]:
        self.peers.set(request.peers)
        listed = await self.default_handlers.call(MCP_CONFIG_V1.list)
        if request.peers and PEER_MCP_NAME not in listed["mcp_servers"]:
            await self.default_handlers.call(MCP_CONFIG_V1.add, McpAddRequest(name=PEER_MCP_NAME, url=PEER_MCP_URL))
        return {"status": "updated", "peers": [peer.name for peer in request.peers]}

    @extension(PEER_AGENTS_V1.list)
    async def list_peers(self) -> dict[str, Any]:
        return {"peers": self.peers.listing()}

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> dict[str, Any]:
        skill_dir = self.skills_dir / request.name
        skill_dir.mkdir(parents=True, exist_ok=False)
        (skill_dir / "SKILL.md").write_text(request.skill_md)
        return {"name": request.name}

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> dict[str, Any]:
        staging = self.skills_dir / f".{request.name}.{uuid.uuid4().hex}"
        try:
            for item in request.skill_bundle.files:
                destination = (staging / item.path).resolve()
                if not destination.is_relative_to(staging.resolve()):
                    raise ValueError(f"skill file {item.path} escapes the skill directory")
                await download(item.object, destination)
            staging.rename(self.skills_dir / request.name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return {"name": request.name}


def _is_tool_result(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "message_end" and (event.get("message") or {}).get("role") == "toolResult"
