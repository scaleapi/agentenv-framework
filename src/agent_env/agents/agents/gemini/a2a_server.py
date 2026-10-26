"""A2A agent wrapping the Google Gemini CLI.

Built on agentenv-protocol's `AgentEnvAgent`: the framework serves the Agent Card,
the A2A task lifecycle and the SDK-owned extensions (agent-config, mcp-config,
skill listing, task trajectories, peer agents, triggers). Maps each A2A `context_id`
to a Gemini session UUID captured from the first turn's `init` event; subsequent
turns use `gemini -r <uuid>` to resume.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from a2a.types import AgentSkill
from agentenv_protocol.a2a_agent import (
    ATTRIBUTION_PROBE_V1,
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
    AttributionProbeResponse,
    BundleSkillRequest,
    FilePart,
    InlineSkillRequest,
    McpAddRequest,
    NamespaceChangelogEnableRequest,
    NamespaceChangelogEnableResponse,
    ObjectChangelogApplyRequest,
    ObjectChangelogApplyResponse,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotLoadResponse,
    ObjectSnapshotSaveRequest,
    ObjectSnapshotSaveResponse,
    SkillAddResponse,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    a2a_agent,
    enable,
    extension,
)
from agentenv_protocol.transfers import Uploaded, upload
from gemini_bridge import (
    GEMINI_HOME,
    GEMINI_PROJECTS_JSON,
    GEMINI_TMP,
    MODEL_HEADERS_ENV,
    build_attribution_headers,
    chats_dir_for_cwd,
    cli_env,
    cli_model,
    detect_format,
    find_session_file,
    gemini_request_to_openai,
    get_last_attribution_snapshot,
    get_project_slug,
    is_generate_path,
    kill_running_clis,
    model_headers,
    needs_openai_route,
    openai_response_to_gemini,
    record_attribution,
    rewrite_session_id,
    run_gemini,
    transcript_session_id,
    wants_sse,
)
from gemini_changelog import (
    HOOK_TIMEOUT_S,
    Changelog,
    changelog_dir,
    gnu_tar,
    is_missing_capture_command,
    merge_settings_hook,
    outermost,
    remove_settings_hooks,
)
from gemini_config import gemini_home, run_settings, user_settings_path
from gemini_kit import (
    DurableSessionRefs,
    FilePartMaterializeError,
    install_skill,
    json_error_bodies,
    materialize_file_part,
    read_object_bytes,
)
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The versioned format of the stream-json events this agent appends to its trajectory.
TRAJECTORY_FORMAT = "gemini-stream-json/1"

# The wrapper's cwd doubles as the gemini CLI's workspace (its file tools
# refuse paths outside it). /workspace in the image and an install.
GEMINI_CWD = os.environ.get("GEMINI_CWD", "/workspace")

# Gemini auto-loads Agent Skills from ~/.gemini/skills/<name>/SKILL.md.
SKILLS_DIR = gemini_home() / "skills"

# Where FilePart payloads get materialized before being passed to the CLI as
# `@<absolute_path>` references, one directory per task. They must live inside
# the CLI's workspace, since its `read_file` tool refuses paths outside it.
INPUTS_DIR = Path(GEMINI_CWD) / "inputs"

_DEFAULT_TIMEOUT_S = 600
# What agent-env keeps of a failed task's error_message.
_ERROR_MESSAGE_LIMIT = 1000
# A snapshot save waits this long for the context's turn to finish before refusing.
_SNAPSHOT_LOCK_TIMEOUT_S = 0.05

# gemini-cli / @google/genai has no custom-header hook, so each run's
# GOOGLE_GEMINI_BASE_URL names this route and the run. It injects
# x-litellm-customer-id/-tags from the run's own config and forwards to LiteLLM's
# /gemini passthrough. Responses are buffered (headless `gemini -p` is single-shot):
# the CLI prints each model call's events once that call returns, so the live
# trajectory advances per call.
_LITELLM_PROXY_PATH = "/litellm-gemini"
_LITELLM_PROXY_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
_PROXY_DROP_REQ_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding"}
_PROXY_DROP_RESP_HEADERS = {"content-length", "transfer-encoding", "content-encoding", "connection", "keep-alive"}


class GeminiConfig(AgentConfig):
    timeout_seconds: int = _DEFAULT_TIMEOUT_S
    model: str | None = None
    # Accepted and read back, but gemini-cli has no per-run tool allowlist, so no turn applies it.
    allowed_tools: list[str] | None = None
    system_prompt: str | None = None
    # LiteLLM cost attribution — the /litellm-gemini proxy converts these into
    # x-litellm-customer-id / x-litellm-tags on every forwarded model call.
    project_id: str | None = None
    task_id: str | None = None


@dataclass(frozen=True)
class _Route:
    """What the proxy forwards one run's model calls with."""

    model: str | None
    project_id: str | None
    task_id: str | None


# install.sh, from the build context install_agent unpacks, brings Node, the Gemini CLI and a Python of the agent's
# own; the endpoint reaches it as environment, never as arguments.
_UNPACK = "mkdir -p {work_dir}/agent && tar -xzf {agent_ctx_tar} -C {work_dir}/agent"
_ENDPOINT = ("LITELLM_API_KEY={litellm_api_key} LITELLM_BASE_URL={litellm_base_url} "
             "LITELLM_EXTRA_HEADERS={litellm_extra_headers} AGENT_TRUST_ENV={trust_env} ")
_ENDPOINT_PARAMS = ["litellm_api_key", "litellm_base_url", "litellm_extra_headers", "trust_env"]
_INSTALL_COMMANDS = [_UNPACK, _ENDPOINT + "bash {work_dir}/agent/install.sh container {container} {a2a_port}"]
_HOST_WORKDIR = "/workspace"
_INSTALL_COMMANDS_HOST = [_UNPACK, _ENDPOINT + f"bash {{work_dir}}/agent/install.sh host {{agent_name}} {{a2a_port}} {_HOST_WORKDIR}"]


def _generate_mcp_name() -> str:
    # Gemini's policy engine splits tool names on the first `_`, so avoid
    # underscores in server aliases — use a hex name instead.
    return f"mcp{uuid.uuid4().hex[:8]}"


def _bootstrap_gemini_project_slug() -> None:
    """Pre-allocate the project-identifier slug for the CLI's cwd so /ext/snapshot
    load can find the chats dir before any gemini turn has run.

    Gemini normally allocates the slug on its first invocation, populating
    `~/.gemini/projects.json` with `{"projects": {"<cwd>": "<slug>"}}`. For a
    fresh target agent receiving a load request (the validator's path), we
    need that allocation in place before any prompt has been processed:
      ~/.gemini/projects.json  → {"projects": {"/workspace": "workspace"}}
      ~/.gemini/tmp/workspace/.project_root  → empty marker file
      ~/.gemini/tmp/workspace/chats/         → empty directory

    All three pieces are written idempotently — if gemini has already run
    and populated them, we leave the existing state alone.
    """
    slug = Path(GEMINI_CWD).name  # gemini uses basename(cwd) as the slug
    try:
        GEMINI_HOME.mkdir(parents=True, exist_ok=True)
        if not GEMINI_PROJECTS_JSON.exists():
            GEMINI_PROJECTS_JSON.write_text(json.dumps({"projects": {GEMINI_CWD: slug}}))
            logger.info(f"Bootstrapped {GEMINI_PROJECTS_JSON} with slug '{slug}' for {GEMINI_CWD!r}")
        slug_dir = GEMINI_TMP / slug
        (slug_dir / "chats").mkdir(parents=True, exist_ok=True)
        marker = slug_dir / ".project_root"
        if not marker.exists():
            marker.touch()
        logger.info(f"Bootstrapped slug dir {slug_dir} (chats/ + .project_root marker)")
    except OSError as e:
        logger.warning(f"Failed to bootstrap gemini project slug: {e}")


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    """Bootstrap the CLI's project slug as the server starts, so a snapshot load works on a fresh agent before any
    turn; as it stops, kill the CLIs of turns still running, which would otherwise go on in the workspace after it."""
    _bootstrap_gemini_project_slug()
    try:
        yield
    finally:
        if stopped := kill_running_clis():
            logger.info(f"Killed {stopped} gemini CLI(s) still running a turn as the agent shut down")


def _changelog_roots() -> list[str]:
    """The changelog's default capture roots: the writable layer a fresh agent of the image reconstructs, the CLI's
    state (and its transcripts) included."""
    return outermost([str(Path.home()), str(gemini_home()), GEMINI_CWD, "/app", "/tmp/data", "/usr/local", "/etc"])


def _install_changelog_hook(command: str) -> None:
    """The capture as an AfterTool hook in the CLI's user settings: matcher ".*" matches every tool, and the CLI's hook
    timeouts are in milliseconds."""
    merge_settings_hook(user_settings_path(), event="AfterTool", matcher=".*", command=command,
                        timeout=HOOK_TIMEOUT_S * 1000)


def _remove_restored_capture_hooks() -> None:
    """Drop capture hooks restored with the CLI's settings, whose scripts stayed on the agent that captured: the CLI
    runs AfterTool hooks synchronously, so a missing script would break every tool call."""
    remove_settings_hooks(user_settings_path(), event="AfterTool", matches=is_missing_capture_command)


_DIAGNOSTIC_LISTING_LIMIT = 20


def _session_diagnostics(session_id: str) -> dict[str, Any]:
    """What the snapshot save saw when it found no transcript, so the failure is
    debuggable without re-deploying the agent: the chats dir contents, the gemini
    home contents and projects.json."""
    slug = get_project_slug(GEMINI_CWD)
    chats = chats_dir_for_cwd(GEMINI_CWD)
    diag: dict[str, Any] = {
        "session_id": session_id,
        "cwd": GEMINI_CWD,
        "slug": slug,
        "gemini_home": str(GEMINI_HOME),
        "gemini_home_exists": GEMINI_HOME.exists(),
        "projects_json_exists": GEMINI_PROJECTS_JSON.exists(),
        "chats_dir": str(chats) if chats else None,
        "chats_dir_exists": chats.exists() if chats else False,
    }
    if GEMINI_PROJECTS_JSON.exists():
        try:
            diag["projects_json_content"] = json.loads(GEMINI_PROJECTS_JSON.read_text())
        except Exception as e:
            diag["projects_json_error"] = str(e)
    if chats and chats.exists():
        try:
            diag["chats_listing"] = sorted(p.name for p in chats.iterdir())[:_DIAGNOSTIC_LISTING_LIMIT]
        except OSError as e:
            diag["chats_listing_error"] = str(e)
    if GEMINI_TMP.exists():
        try:
            diag["gemini_tmp_listing"] = sorted(p.name for p in GEMINI_TMP.iterdir())[:_DIAGNOSTIC_LISTING_LIMIT]
        except OSError as e:
            diag["gemini_tmp_listing_error"] = str(e)
    # If we resolved a slug, list its dir to see where sessions live
    if slug:
        slug_dir = GEMINI_TMP / slug
        if slug_dir.exists():
            try:
                diag["slug_dir_listing"] = sorted(p.name for p in slug_dir.iterdir())[:_DIAGNOSTIC_LISTING_LIMIT]
            except OSError as e:
                diag["slug_dir_listing_error"] = str(e)
    return diag


def _plant_transcript(chats: Path, session_id: str, source_format: str, data: bytes) -> Path:
    """Write a transcript under this agent's chats dir as the session `session_id`."""
    chats.mkdir(parents=True, exist_ok=True)
    # Match gemini's filename format exactly so resume's filename-based
    # short-circuit (sessionExists) can hit it: session-YYYY-MM-DDTHH-MM-<sid[:8]>.<ext>
    filename_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M")
    ext = "json" if source_format == "json" else "jsonl"
    target_path = chats / f"session-{filename_ts}-{session_id[:8]}.{ext}"
    target_path.write_bytes(data)
    return target_path


def _restored_transcripts() -> list[Path]:
    return [*GEMINI_TMP.glob("*/chats/session-*.json"), *GEMINI_TMP.glob("*/chats/session-*.jsonl")]


def _upstream_headers(headers: dict[str, str], project_id: str | None, task_id: str | None) -> dict[str, str]:
    """``headers``, then the endpoint's [model.headers], then the run's attribution, each replacing a header of the
    same name in any case."""
    attribution = build_attribution_headers(project_id, task_id)
    if attribution:
        record_attribution(project_id, task_id)
    layers = (headers, model_headers(os.environ.get(MODEL_HEADERS_ENV)), attribution)
    return {name.lower(): value for layer in layers for name, value in layer.items()}


def _answer(uploaded: Uploaded) -> dict[str, Any]:
    """An upload, as agent-env reads it back."""
    return uploaded.model_dump(mode="json")


async def _vertex_openai_bridge(
    rest: str, request: Request, base: str, route_model: str, project_id: str | None, task_id: str | None,
) -> Response:
    """Translate the CLI's native-Gemini generate call into an OpenAI
    /v1/chat/completions call whose `model` is the provider-prefixed route
    string (e.g. vertex_ai/global/gemini-3.1-pro-preview) — the only endpoint
    that reaches Vertex — then translate the response back to a single native
    Gemini GenerateContentResponse (SSE-framed when the CLI asked for
    streamGenerateContent). Attribution headers are injected identically to the
    passthrough path so cost tagging is preserved."""
    body = await request.body()
    try:
        greq = json.loads(body) if body else {}
    except json.JSONDecodeError:
        greq = {}
    oai_req = gemini_request_to_openai(greq, route_model)
    headers = _upstream_headers({
        "Authorization": f"Bearer {os.environ.get('LITELLM_API_KEY', '')}",
        "Content-Type": "application/json",
    }, project_id, task_id)
    try:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S) as client:
            upstream = await client.post(
                f"{base}/v1/chat/completions",
                headers=headers, content=json.dumps(oai_req).encode(),
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"vertex bridge upstream failed: {e}")
    if upstream.status_code >= 400:
        logger.warning("vertex bridge upstream %s: %s", upstream.status_code, upstream.text[:500])
        return Response(
            content=upstream.content, status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )
    try:
        oai_resp = upstream.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="vertex bridge: non-JSON upstream response")
    gresp = openai_response_to_gemini(oai_resp)
    if wants_sse(rest, request.query_params.get("alt")):
        payload = f"data: {json.dumps(gresp)}\n\n".encode()
        return Response(content=payload, media_type="text/event-stream")
    return Response(content=json.dumps(gresp).encode(), media_type="application/json")


@a2a_agent(
    identity=AgentIdentity(
        name="Gemini",
        description="A2A-compliant wrapper around the Google Gemini CLI.",
        version="1.0.0",
        # The wrapper materializes any FilePart to disk and injects `@<path>` into
        # the prompt; gemini-cli auto-attaches the bytes as inline data to the
        # Gemini API call. Modes below are functionally validated end-to-end by
        # the A2A modality probe (see `verify_a2a_modalities.py`).
        input_modes=(
            "text",
            "image/png", "image/jpeg", "image/gif", "image/webp",
            "audio/wav", "audio/mpeg", "audio/mp4", "audio/ogg",
            "application/pdf",
            "video/mp4",
        ),
        output_modes=("text",),
        skills=(
            AgentSkill(
                id="gemini",
                name="gemini",
                description="Runs the Google Gemini CLI with full tool access (shell, file edits, web).",
                tags=["coding", "agent"],
            ),
        ),
    ),
    config=GeminiConfig,
    config_description="Set or read agent configuration such as model, role, and allowed tools.",
    extensions=(
        enable(
            INSTALL_V1,
            description=(
                "Commands to install this agent into a running task container, "
                "or directly onto a task VM (host mode)."
            ),
            install_commands=_INSTALL_COMMANDS,
            required_params=["container", "work_dir", "agent_ctx_tar", "a2a_port", *_ENDPOINT_PARAMS],
            install_commands_host=_INSTALL_COMMANDS_HOST,
            a2a_port=8000,
            required_params_host=["work_dir", "agent_ctx_tar", "a2a_port", "agent_name", *_ENDPOINT_PARAMS],
            workdir_host=_HOST_WORKDIR,
            user_host="root",
        ),
        enable(
            ATTRIBUTION_PROBE_V1,
            description=(
                "Read back the LiteLLM attribution headers the agent most recently emitted on an "
                "outbound LLM call. Used by the agent-env runtime attribution validator."
            ),
        ),
        enable(
            TRAJECTORY_V1,
            live=True,
            description="Retrieve a task's Gemini stream-json trajectory, also while it runs.",
        ),
        enable(MCP_CONFIG_V1, description="Register MCP servers the Gemini CLI should connect to."),
        enable(SKILL_CONFIG_V1, description="Register a skill loaded by the Gemini CLI from ~/.gemini/skills/."),
        enable(
            PEER_AGENTS_V1,
            description="Delegate to the task's other agents: the CLI reaches them through the SDK's peer_agents MCP tools.",
        ),
        enable(
            SNAPSHOT_V1,
            description=(
                "Capture and restore Gemini CLI session transcripts, and, where the host has GNU tar, "
                "each tool call's writes as a changelog to replay onto a fresh agent, via agent-env's "
                "object-transfer grants."
            ),
        ),
        enable(
            TRIGGERS_V1,
            description=(
                "Register a typed trigger script and decide the stakeholder reply per turn "
                "(step / env_trigger / conversational / all / any conditions -> say / end actions)."
            ),
        ),
    ),
    lifespan=lifespan,
)
class GeminiAgent(DurableSessionRefs, AgentEnvAgent):
    """Bridges A2A tasks onto the `gemini` CLI."""

    def __init__(self) -> None:
        # Per-context_id locks so a snapshot save sees an at-rest session transcript
        # while a turn is mid-flight. Held around the run_gemini call; save tries to
        # acquire with a 50ms timeout -> 409 if a turn is running.
        self._context_locks: dict[str, asyncio.Lock] = {}
        self._context_locks_guard = asyncio.Lock()
        self._active_contexts: set[str] = set()
        # Each running turn's route, by the token its CLI's model calls name.
        self._routes: dict[str, _Route] = {}
        # Every session a turn's init event named, bound or not: a first turn that saw no init of its own never takes
        # one of these for its session.
        self._claimed_sessions: set[str] = set()

    async def _context_lock(self, context_id: str) -> asyncio.Lock:
        async with self._context_locks_guard:
            lock = self._context_locks.get(context_id)
            if lock is None:
                lock = asyncio.Lock()
                self._context_locks[context_id] = lock
            return lock

    async def run(self, request: TaskRequest[GeminiConfig]) -> TaskResult:
        # The CLI's stream-json events, appended as it prints them, are the turn's
        # trajectory: readable while it runs, and its final record once it ends.
        request.trajectory.set_format(TRAJECTORY_FORMAT)
        # TextPart -> prompt text; FilePart -> materialized path that the bridge
        # will reference via `@<path>` in the prompt. A caller's DataPart is ignored.
        text_chunks: list[str] = []
        attachment_paths: list[str] = []
        try:
            for part in request.parts:
                if isinstance(part, TextPart):
                    text_chunks.append(part.text or "")
                elif isinstance(part, FilePart):
                    attachment_paths.append(await materialize_file_part(part, INPUTS_DIR / request.task_id))
        except FilePartMaterializeError as e:
            logger.error(f"FilePart materialization failed: {e}", exc_info=True)
            return (
                TaskResult.builder()
                .failed("filepart_error", str(e)[:_ERROR_MESSAGE_LIMIT])
                .add_text(str(e))
                .build()
            )

        user_text = "\n".join(c for c in text_chunks if c)
        cfg = request.config
        context_id = request.context_id
        ctx_lock = await self._context_lock(context_id)
        async with ctx_lock:
            # Under the guard a changelog apply holds throughout, so no turn starts on a workspace being restored.
            async with self._context_locks_guard:
                self._active_contexts.add(context_id)
            token = uuid.uuid4().hex
            self._routes[token] = _Route(cfg.model, cfg.project_id, cfg.task_id)
            own = self.session_ref_for_context(context_id)
            try:
                with run_settings(request.mcp_servers, cfg.role, cfg.system_prompt) as settings:
                    base_url = None
                    if os.environ.get("LITELLM_BASE_URL"):
                        port = os.environ.get("A2A_PORT") or "8000"
                        base_url = f"http://127.0.0.1:{port}{_LITELLM_PROXY_PATH}/{token}"
                    # The prompt travels on argv, so one past Linux's 128 KiB per-argument
                    # limit fails the spawn with E2BIG, answered as an infra_error.
                    result = await run_gemini(
                        prompt=user_text,
                        session_id=own,
                        # gemini-cli embeds --model in the native Gemini API URL path;
                        # a provider-prefixed route model (vertex_ai/...) crashes the
                        # @google/genai client on the slashes. Hand the CLI a bare id
                        # and let the /litellm-gemini proxy route the real model via the
                        # OpenAI-compat endpoint (the only path that reaches Vertex).
                        model=cli_model(cfg.model),
                        attachment_paths=attachment_paths or None,
                        timeout_s=cfg.timeout_seconds,
                        event_callback=lambda event: self._on_event(event, request),
                        env=cli_env(os.environ, system_settings=settings, base_url=base_url),
                        cwd=GEMINI_CWD,
                        other_sessions=lambda: {*self.bound_session_refs(), *self._claimed_sessions} - {own},
                    )
            except Exception as e:
                logger.error(f"gemini CLI invocation raised: {e}", exc_info=True)
                return (
                    TaskResult.builder()
                    .failed("infra_error", str(e)[:_ERROR_MESSAGE_LIMIT], error_type="infra_error")
                    .add_text(str(e))
                    .build()
                )
            finally:
                self._routes.pop(token, None)
                self._active_contexts.discard(context_id)

        if result.session_id:
            self.set_session_ref_for_context(context_id, result.session_id)
        builder = TaskResult.builder()
        if result.error:
            builder.failed("cli_error", result.error[:_ERROR_MESSAGE_LIMIT])
        else:
            builder.succeeded()
        return (
            builder
            # A turn that fails before any assistant text has an empty response;
            # fall back to the error so the TextPart is not blank.
            .add_text(result.response or (result.error or ""))
            .add_data({"tool_call_count": result.tool_call_count})
            .session_ref(result.session_id or None)
            .usage(Usage(tool_call_count=result.tool_call_count))
            .build()
        )

    def _on_event(self, event: dict, request: TaskRequest[GeminiConfig]) -> None:
        if event.get("type") == "init" and event.get("session_id"):
            self._claimed_sessions.add(event["session_id"])
        request.trajectory.append(event)

    @extension(MCP_CONFIG_V1.add)
    async def add_mcp_server(self, request: McpAddRequest) -> dict[str, Any]:
        # Gemini splits mcp_<server>_<tool> on the first "_", so an alias keeps none.
        name = (request.name or "").replace("_", "") or _generate_mcp_name()
        return await self.default_handlers.call(MCP_CONFIG_V1.add, request.model_copy(update={"name": name}))

    @extension(MCP_CONFIG_V1.list)
    async def list_mcp_servers(self) -> dict[str, Any]:
        return await self.default_handlers.call(MCP_CONFIG_V1.list)

    @extension(ATTRIBUTION_PROBE_V1.probe)
    async def probe_attribution(self) -> AttributionProbeResponse:
        """The most recent LiteLLM attribution headers the proxy emitted.

        Used by agent-env's runtime attribution validator to confirm that
        project_id/task_id from /ext/agent-config actually reached an outbound model
        call."""
        snapshot = get_last_attribution_snapshot()
        return AttributionProbeResponse(
            last_seen_attribution={
                key: value for key, value in snapshot.items() if key != "last_seen_at_utc" and isinstance(value, str)
            },
            last_seen_at_utc=snapshot.get("last_seen_at_utc"),
        )

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(SNAPSHOT_V1.save)
    async def save_snapshot(self, request: ObjectSnapshotSaveRequest) -> ObjectSnapshotSaveResponse:
        """Capture the Gemini session transcript for `context_id` through agent-env's
        write grants.

        Gemini writes one transcript per session at
        <GEMINI_HOME>/tmp/<projectIdentifier>/chats/session-<ISO-ts>-<sid[:8]>.json{,l}.
        The projectIdentifier slug is per-cwd. We find the file by matching
        `session-*-<sid[:8]>.*` and verifying its `sessionId`. The bytes go to the
        "trajectory" object as-is. A gemini snapshot is the transcript alone, so
        "workspace" is written empty: agent-env requires both objects to exist after
        the answer.
        """
        context_id = request.context_id
        if not context_id:
            raise HTTPException(status_code=400, detail="context_id is required")
        session_id = self.session_ref_for_context(context_id)
        if session_id is None:
            raise HTTPException(status_code=404, detail=f"No active session for context_id '{context_id}'")

        ctx_lock = await self._context_lock(context_id)
        try:
            await asyncio.wait_for(ctx_lock.acquire(), timeout=_SNAPSHOT_LOCK_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=409, detail="session is mid-turn; retry after current turn completes")
        try:
            jsonl_path = find_session_file(session_id, cwd=GEMINI_CWD)
            if jsonl_path is None:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"no session JSONL found for session_id={session_id} — "
                        f"diag: {json.dumps(_session_diagnostics(session_id))}"
                    ),
                )
            # Byte-faithful capture — gemini's JSONL contains fields (e.g.
            # `directories`) we can't always reconstruct from a `.json -> .jsonl`
            # conversion, so we upload the source bytes as-is and let load
            # preserve the same on-disk format when planting on the target.
            jsonl_bytes = jsonl_path.read_bytes()
        finally:
            ctx_lock.release()

        uploaded = {"trajectory": _answer(await upload(request.objects.trajectory, jsonl_bytes))}
        if request.objects.workspace is not None:
            uploaded["workspace"] = _answer(await upload(request.objects.workspace, b""))
        logger.info(f"Snapshot saved: context_id={context_id} session_id={session_id} -> object-transfer grants")
        return ObjectSnapshotSaveResponse.model_validate({"context_id": context_id, "objects": uploaded})

    @extension(SNAPSHOT_V1.load)
    async def load_snapshot(self, request: ObjectSnapshotLoadRequest) -> ObjectSnapshotLoadResponse:
        """Restore a Gemini session transcript under a (possibly caller-specified) context_id.

        Reads the "trajectory" object, mints a fresh session_id, rewrites the
        transcript's `sessionId`, writes it under
        <GEMINI_HOME>/tmp/<slug>/chats/session-<now-ts>-<new_sid[:8]>.<ext>, and binds
        target_context_id -> new_session_id. Next message/send for that context will pass
        `-r <new_session_id>` and gemini finds it by scanning the chats dir for a
        matching sessionId. The "workspace" object a gemini save writes is always
        empty, so it is not fetched.

        Requires a project slug for the CLI's cwd (the lifespan bootstraps one, and
        gemini allocates one on its first invocation; if missing here we 503 and the
        caller can retry after a warm-up prompt).
        """
        target_context_id = request.target_context_id
        if target_context_id is not None:
            if not target_context_id:
                raise HTTPException(status_code=400, detail="target_context_id must be a non-empty string")
            new_context_id = target_context_id
        else:
            new_context_id = uuid.uuid4().hex
        new_session_id = str(uuid.uuid4())

        chats = chats_dir_for_cwd(GEMINI_CWD)
        if chats is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"gemini project slug not yet allocated for cwd={GEMINI_CWD!r}; "
                    "send any message to this agent first to bootstrap projects.json"
                ),
            )

        raw_bytes = await read_object_bytes(request.objects.trajectory)
        # Detect source format ('.json' single object or '.jsonl' multi-line) and
        # rewrite sessionId in-place. The source may be either depending on
        # whether the original session had been resumed before snapshotting —
        # gemini writes `.json` on first turn and auto-migrates to `.jsonl` on
        # resume.
        try:
            source_format = detect_format(raw_bytes)
            rewritten_bytes = rewrite_session_id(raw_bytes, new_session_id)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"failed to rewrite sessionId: {e}")

        target_path = _plant_transcript(chats, new_session_id, source_format, rewritten_bytes)
        async with self._context_locks_guard:
            if self.session_ref_for_context(new_context_id) is not None:
                target_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=409,
                    detail=f"context_id '{new_context_id}' is already bound — retry with a different target_context_id",
                )
            self.set_session_ref_for_context(new_context_id, new_session_id)

        logger.info(
            f"Snapshot loaded: object-transfer grants -> context_id={new_context_id} "
            f"session_id={new_session_id} (file={target_path})"
        )
        return ObjectSnapshotLoadResponse(context_id=new_context_id)

    async def litellm_gemini_proxy(self, request: Request) -> Response:
        """Forward one run's model call to LiteLLM with that run's attribution headers."""
        route = self._routes.get(request.path_params["token"])
        if route is None:
            raise HTTPException(status_code=404, detail="no running turn has this proxy route")
        rest = request.path_params["rest"]
        base = (os.environ.get("LITELLM_BASE_URL") or "").rstrip("/")
        if not base:
            raise HTTPException(status_code=503, detail="LITELLM_BASE_URL not set; cannot forward gemini model calls")
        # Provider-prefixed route models (vertex_ai/...) can't traverse the /gemini
        # passthrough (AI-Studio-bound) — bridge their generate calls to the
        # OpenAI-compat endpoint instead. Everything else (countTokens, the native
        # AI-Studio models) keeps the verbatim passthrough below.
        route_model = (route.model or "").strip()
        if needs_openai_route(route_model) and is_generate_path(rest):
            return await _vertex_openai_bridge(rest, request, base, route_model, route.project_id, route.task_id)
        fwd_headers = _upstream_headers(
            {k: v for k, v in request.headers.items() if k.lower() not in _PROXY_DROP_REQ_HEADERS},
            route.project_id, route.task_id,
        )
        body = await request.body()
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S) as client:
                upstream = await client.request(
                    request.method, f"{base}/gemini/{rest}",
                    params=request.query_params, headers=fwd_headers, content=body,
                )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"gemini upstream request failed: {e}")
        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _PROXY_DROP_RESP_HEADERS}
        return Response(content=upstream.content, status_code=upstream.status_code, headers=resp_headers)


class GeminiChangelogAgent(GeminiAgent):
    """With snapshot/v1's changelog, which needs GNU tar's incremental archives."""

    def __init__(self) -> None:
        super().__init__()
        self._changelog = Changelog(default_roots=_changelog_roots(), install_hook=_install_changelog_hook,
                                    resume=self._resume_conversation, directory=changelog_dir(os.environ),
                                    after_apply=_remove_restored_capture_hooks)
        # A changelog is one sequence of tool calls, so while one is captured, turns take turns. A turn reads whether
        # it's captured, and an uncaptured one is counted, under the admission lock an enable holds throughout.
        self._admission = asyncio.Lock()
        self._capture_turn = asyncio.Lock()
        self._capturing = False
        self._uncaptured = 0

    async def run(self, request: TaskRequest[GeminiConfig]) -> TaskResult:
        async with self._admission:
            captured = self._capturing
            if not captured:
                self._uncaptured += 1
        if not captured:
            try:
                return await super().run(request)
            finally:
                self._uncaptured -= 1
        async with self._capture_turn:
            return await super().run(request)

    @extension(SNAPSHOT_V1.changelog.enable)
    async def enable_changelog(self, request: NamespaceChangelogEnableRequest) -> NamespaceChangelogEnableResponse:
        """Capture each tool call's writes, uploaded as <root_path>/<position:06d>.tgz under the grant's limits."""
        async with self._admission:
            if self._uncaptured or self._active_contexts:
                raise HTTPException(status_code=409, detail="cannot enable a changelog while an agent turn is running")
            answer = await self._changelog.enable(request)
            self._capturing = True
        return answer

    @extension(SNAPSHOT_V1.changelog.apply)
    async def apply_changelog(self, request: ObjectChangelogApplyRequest) -> ObjectChangelogApplyResponse:
        """Reconstruct the writable layer from the granted increments, and with `resume_conversation` make the
        restored transcript resumable."""
        if request.target_context_id is not None and not request.target_context_id:
            raise HTTPException(status_code=400, detail="target_context_id must be a non-empty string")
        # Held across the restore, as a snapshot load is: no turn runs on a workspace being rewritten.
        async with self._context_locks_guard:
            if self._active_contexts:
                raise HTTPException(status_code=409, detail="cannot apply a changelog while an agent turn is running")
            return await self._changelog.apply(request)

    async def _resume_conversation(self, target_context_id: str | None, session: str | None) -> str | None:
        """Bind a context to the restored transcript of ``session``, the last applied tool call's, under a fresh
        session id, so the next turn there resumes the rewound conversation; None when no transcript was restored, or
        when it can't be resumed, which leaves the restore itself standing."""
        try:
            return self._bind_restored_transcript(target_context_id, session)
        except Exception as e:
            logger.warning(f"agent-changelog resume failed, the workspace stays restored: {e}")
            return None

    def _bind_restored_transcript(self, target_context_id: str | None, session: str | None) -> str | None:
        candidates = _restored_transcripts()
        # Increments that recorded no session, or whose transcript wasn't restored: the newest is the likeliest.
        source = next((path for path in candidates if session and transcript_session_id(path) == session),
                      max(candidates, key=lambda p: p.stat().st_mtime, default=None))
        if source is None:
            logger.info("agent-changelog resume: no restored transcript")
            return None
        chats = chats_dir_for_cwd(GEMINI_CWD)
        if chats is None:
            logger.info(f"agent-changelog resume: no chats dir for cwd={GEMINI_CWD!r}")
            return None
        raw = source.read_bytes()
        new_session_id = str(uuid.uuid4())
        _plant_transcript(chats, new_session_id, detect_format(raw), rewrite_session_id(raw, new_session_id))
        new_context_id = target_context_id or uuid.uuid4().hex
        self.set_session_ref_for_context(new_context_id, new_session_id)
        logger.info(f"agent-changelog resume: bound context {new_context_id} -> session {new_session_id} "
                    f"(from {source.name})")
        return new_context_id


def agent_for_this_host() -> GeminiAgent:
    """The agent, with the changelog where this host's tar can make its archives."""
    return GeminiChangelogAgent() if gnu_tar() else GeminiAgent()


def build_app(agent: GeminiAgent) -> Starlette:
    """The agent's Starlette app, answering errors with FastAPI's `{"detail": ...}` bodies,
    plus the LiteLLM attribution proxy."""
    app = json_error_bodies(agent.create_app())
    app.add_route(f"{_LITELLM_PROXY_PATH}/{{token}}/{{rest:path}}", agent.litellm_gemini_proxy,
                  methods=_LITELLM_PROXY_METHODS)
    return app


app = build_app(agent_for_this_host())
# agent-env's install/v1 deploys find the install parameters by importing this name from the image.
AGENT_CARD = app.state.agentenv_a2a.card
