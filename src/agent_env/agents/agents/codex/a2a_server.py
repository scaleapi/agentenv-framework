"""A2A agent wrapping the OpenAI Codex CLI.

Built on agentenv-protocol's `AgentEnvAgent`: the framework serves the Agent Card,
the A2A task lifecycle and the SDK-owned extensions (agent-config, mcp-config,
skill listing, task trajectories, triggers). Each Codex `--json` event is appended to
the framework's trajectory log as the CLI prints it, so a task's trajectory is
readable while it runs and is its final trajectory once it ends. Maps each A2A `context_id`
to a Codex `thread_id` captured from the first turn's `thread.started` event;
subsequent turns use `codex exec resume <thread_id>`.
"""
from __future__ import annotations

import asyncio
import gzip
import inspect
import json
import logging
import os
import shutil
import tarfile
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from a2a.types import AgentSkill
from agentenv_protocol.a2a_agent import (
    AGENT_CONFIG_V1,
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
    ContextEventsTrajectoryRequest,
    ContextObjectTrajectoryRequest,
    ContextTrajectoryRequest,
    InlineSkillRequest,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    SkillAddResponse,
    TaskRequest,
    TaskResult,
    TextPart,
    TrajectoryState,
    Usage,
    a2a_agent,
    enable,
    extension,
)
from agentenv_protocol.transfers import ReadObject, TransferError, Uploaded, download, upload
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

import codex_config
from codex_bridge import (
    SESSIONS_DIR,
    find_rollout_path,
    get_last_attribution_snapshot,
    kill_running_clis,
    run_codex,
)
from codex_kit import DurableSessionRefs, in_thread_to_its_end, install_skill, json_error_bodies, read_object_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HARNESS = "codex-cli"
# The versioned format of the `codex exec --json` events this agent appends to its trajectory.
TRAJECTORY_FORMAT = "codex-exec-json/1"
WORKSPACE_DIR = Path(os.environ.get("CODEX_WORKSPACE_DIR", "/app"))
_MAX_SNAPSHOT_COMPRESSED_BYTES = int(
    os.environ.get("CODEX_SNAPSHOT_MAX_COMPRESSED_BYTES", str(2 * 1024**3))
)
_MAX_SNAPSHOT_UNCOMPRESSED_BYTES = int(
    os.environ.get("CODEX_SNAPSHOT_MAX_UNCOMPRESSED_BYTES", str(4 * 1024**3))
)
_MAX_SNAPSHOT_MEMBERS = int(
    os.environ.get("CODEX_SNAPSHOT_MAX_MEMBERS", "100000")
)
_MAX_TAR_METADATA_BYTES = 1024 * 1024
_TRANSFER_CHUNK_BYTES = 1024 * 1024
_LOCK_TIMEOUT_S = 0.05
_SNAPSHOT_QUIESCE_TIMEOUT_S = float(
    os.environ.get("CODEX_SNAPSHOT_QUIESCE_TIMEOUT_S", "90")
)
_MAX_TRANSCRIPT_BYTES = int(
    os.environ.get("CODEX_SNAPSHOT_MAX_TRANSCRIPT_BYTES", str(64 * 1024**2))
)
# What agent-env keeps of a failed task's error_message.
_ERROR_MESSAGE_LIMIT = 4000
# How many of a turn's warnings the result carries, and how much of each.
_WARNINGS_KEPT = 10
_WARNING_CHARS = 500

# Codex discovers skills (Agent Skills spec) under ~/.agents/skills/<name>/SKILL.md.
SKILLS_DIR = Path.home() / ".agents" / "skills"

_active_execution_tasks: dict[str, set[asyncio.Task]] = {}

# Per-context_id locks so /ext/snapshot save sees an at-rest rollout JSONL
# while a turn is mid-flight. Codex appends to the rollout file during a
# turn, so reading it concurrently could capture a partial line. Acquired
# around the run_codex call in run(); save tries to acquire with a
# 50ms timeout -> 409 if a turn is running.
_context_locks: dict[str, asyncio.Lock] = {}
_context_locks_guard = asyncio.Lock()


async def _acquire_within(
    acquire: Callable[[], Awaitable[object]], release: Callable[[], object], seconds: float
) -> None:
    """``await acquire()`` with a deadline, raising ``asyncio.TimeoutError`` when it passes.

    Not ``wait_for``, which can swallow a cancel that lands as the acquire completes, so the
    snapshot would carry on for a gone caller. A gate the abandoned acquire already won is given back."""
    waiter = asyncio.ensure_future(acquire())
    try:
        done, _ = await asyncio.wait({waiter}, timeout=seconds)
    except asyncio.CancelledError:
        await _abandon(waiter, release)
        raise
    if not done:
        await _abandon(waiter, release)
        raise asyncio.TimeoutError
    waiter.result()


async def _abandon(waiter: asyncio.Future, release: Callable[[], object]) -> None:
    if not waiter.done():
        waiter.cancel()
        return
    if not waiter.cancelled() and waiter.exception() is None:
        released = release()
        if inspect.isawaitable(released):
            await asyncio.shield(released)


class _WorkspaceBarrier:
    """Allow concurrent turns while giving snapshots exclusive workspace access.

    A process-wide mutex around every turn made independent validator prompts
    queue behind one another until their 120-second polling deadlines expired.
    Turns therefore take shared access; snapshot save/load take exclusive
    access so their transcript + workspace state still comes from a quiescent
    point. Waiting writers block new readers to avoid snapshot starvation.
    """

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    async def acquire_shared(self) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._writer and self._waiting_writers == 0
            )
            self._readers += 1

    async def release_shared(self) -> None:
        async with self._condition:
            if self._readers <= 0:
                raise RuntimeError("workspace shared lock is not held")
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    async def acquire_exclusive(self) -> None:
        async with self._condition:
            self._waiting_writers += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._writer and self._readers == 0
                )
                self._writer = True
            finally:
                self._waiting_writers -= 1
                if not self._writer:
                    self._condition.notify_all()

    async def release_exclusive(self) -> None:
        async with self._condition:
            if not self._writer:
                raise RuntimeError("workspace exclusive lock is not held")
            self._writer = False
            self._condition.notify_all()

    @asynccontextmanager
    async def shared(self):
        await self.acquire_shared()
        try:
            yield
        finally:
            await self.release_shared()


_workspace_barrier = _WorkspaceBarrier()


async def _get_context_lock(context_id: str) -> asyncio.Lock:
    async with _context_locks_guard:
        lock = _context_locks.get(context_id)
        if lock is None:
            lock = asyncio.Lock()
            _context_locks[context_id] = lock
        return lock


def _discard_execution_task(context_id: str, task: asyncio.Task) -> None:
    tasks = _active_execution_tasks.get(context_id)
    if tasks is None:
        return
    tasks.discard(task)
    if not tasks:
        _active_execution_tasks.pop(context_id, None)


class CodexConfig(AgentConfig):
    model: str | None = None
    effort: str | None = None
    system_prompt: str | None = None
    output_format: dict[str, Any] | None = None
    project_id: str | None = None
    task_id: str | None = None
    timeout_seconds: int = 1800


@asynccontextmanager
async def lifespan(app: Starlette):
    """The model endpoint into config.toml as the server starts; as it stops, the CLIs of turns still running are
    killed, which would otherwise go on in the workspace after it."""
    codex_config.write_base_config(os.environ)
    try:
        yield
    finally:
        if stopped := kill_running_clis():
            logger.info(f"Killed {stopped} codex CLI(s) still running a turn as the agent shut down")


def _read_complete_jsonl(path: Path) -> bytes:
    """Return complete JSONL records, ignoring a concurrently torn final line."""
    with path.open("rb") as stream:
        data = stream.read(_MAX_TRANSCRIPT_BYTES + 1)
    if len(data) > _MAX_TRANSCRIPT_BYTES:
        raise ValueError("session transcript exceeds the size limit")
    raw_lines = data.splitlines()
    trailing_newline = data.endswith(b"\n")
    complete: list[bytes] = []
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        try:
            json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(raw_lines) - 1 and not trailing_newline:
                break
            raise ValueError(
                f"session transcript contains malformed complete record {index + 1}"
            )
        complete.append(raw_line)
    if not complete:
        raise ValueError("session transcript has no complete JSONL record")
    return b"\n".join(complete) + b"\n"


class _LimitedWriter:
    def __init__(self, stream, limit: int):
        self.stream = stream
        self.limit = limit
        self.written = 0

    def write(self, data: bytes):
        if self.written + len(data) > self.limit:
            raise ValueError("workspace archive exceeds the compressed size limit")
        written = self.stream.write(data)
        self.written += written
        return written

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.written


def _build_workspace_archive(path: Path) -> None:
    if not WORKSPACE_DIR.is_dir():
        raise FileNotFoundError(f"workspace directory does not exist: {WORKSPACE_DIR}")

    member_count = 0
    uncompressed_bytes = 0
    workspace_root = WORKSPACE_DIR.resolve()

    def safe_member(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        nonlocal member_count, uncompressed_bytes
        if not (member.isfile() or member.isdir() or member.issym()):
            raise ValueError(
                f"workspace contains unsupported entry: {member.name}"
            )
        if member.issym():
            link = Path(member.linkname)
            link_target = (
                link
                if link.is_absolute()
                else workspace_root / Path(member.name).parent / link
            ).resolve()
            if (
                link.is_absolute()
                or (
                    link_target != workspace_root
                    and workspace_root not in link_target.parents
                )
            ):
                raise ValueError(
                    f"workspace contains unsafe link: {member.name}"
                )
        member_count += 1
        uncompressed_bytes += member.size
        if member_count > _MAX_SNAPSHOT_MEMBERS:
            raise ValueError("workspace contains too many entries")
        if uncompressed_bytes > _MAX_SNAPSHOT_UNCOMPRESSED_BYTES:
            raise ValueError("workspace exceeds the uncompressed size limit")
        return member

    with path.open("wb") as raw:
        writer = _LimitedWriter(raw, _MAX_SNAPSHOT_COMPRESSED_BYTES)
        with tarfile.open(
            fileobj=writer, mode="w|gz", dereference=False
        ) as archive:
            for child in WORKSPACE_DIR.iterdir():
                archive.add(
                    child,
                    arcname=child.name,
                    recursive=True,
                    filter=safe_member,
                )


def _validate_tar_headers(path: Path) -> int:
    metadata_bytes = 0
    payload_bytes = 0
    record_count = 0
    with gzip.open(path, "rb") as stream:
        while True:
            header = stream.read(512)
            if not header or header == b"\0" * 512:
                return payload_bytes
            if len(header) != 512:
                raise ValueError("workspace archive has a truncated header")
            record_count += 1
            if record_count > _MAX_SNAPSHOT_MEMBERS * 2:
                raise ValueError("workspace archive contains too many records")
            try:
                size = tarfile.nti(header[124:136])
            except ValueError as e:
                raise ValueError("workspace archive has an invalid size") from e
            if not isinstance(size, int) or size < 0:
                raise ValueError("workspace archive has an invalid size")
            if header[156:157] in {b"x", b"g", b"L", b"K"}:
                metadata_bytes += size
                if metadata_bytes > _MAX_TAR_METADATA_BYTES:
                    raise ValueError("workspace archive metadata is too large")
            payload_bytes += size
            if (
                payload_bytes
                > _MAX_SNAPSHOT_UNCOMPRESSED_BYTES
                + _MAX_TAR_METADATA_BYTES
            ):
                raise ValueError(
                    "workspace archive exceeds the uncompressed size limit"
                )
            remaining = ((size + 511) // 512) * 512
            while remaining:
                chunk = stream.read(min(_TRANSFER_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ValueError("workspace archive is truncated")
                remaining -= len(chunk)


def _restore_workspace_archive(path: Path) -> None:
    """Transactionally replace workspace contents without parent write access."""
    WORKSPACE_DIR.parent.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    payload_bytes = _validate_tar_headers(path)
    if (
        shutil.disk_usage(WORKSPACE_DIR).free
        < payload_bytes + 64 * 1024**2
    ):
        raise ValueError("insufficient disk space to restore workspace")
    stage = Path(
        tempfile.mkdtemp(
            prefix=".snapshot-stage-",
            dir=WORKSPACE_DIR,
        )
    )
    backup: Path | None = None
    try:
        root = stage.resolve()
        member_count = 0
        uncompressed_bytes = 0
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                member_count += 1
                uncompressed_bytes += member.size
                if member_count > _MAX_SNAPSHOT_MEMBERS:
                    raise ValueError("workspace archive contains too many entries")
                if uncompressed_bytes > _MAX_SNAPSHOT_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        "workspace archive exceeds the uncompressed size limit"
                    )
                target = (root / member.name).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(
                        f"workspace archive path escapes its root: {member.name}"
                    )
                if not (
                    member.isfile() or member.isdir() or member.issym()
                ):
                    raise ValueError(
                        f"workspace archive contains unsupported entry: {member.name}"
                    )
                target = stage / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    target.chmod(0o755)
                    continue
                if member.issym():
                    if Path(member.linkname).is_absolute():
                        raise ValueError(
                            f"workspace archive contains unsafe link: {member.name}"
                        )
                    link_target = (target.parent / member.linkname).resolve()
                    if link_target != root and root not in link_target.parents:
                        raise ValueError(
                            f"workspace archive contains unsafe link: {member.name}"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(member.linkname)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(
                        f"workspace archive file is unreadable: {member.name}"
                    )
                with source, target.open("wb") as output:
                    shutil.copyfileobj(
                        source, output, length=_TRANSFER_CHUNK_BYTES
                    )
                target.chmod(0o755 if member.mode & 0o111 else 0o644)

        backup = Path(
            tempfile.mkdtemp(
                prefix=".snapshot-backup-",
                dir=WORKSPACE_DIR,
            )
        )
        reserved_names = {stage.name, backup.name}
        staged_entries = list(stage.iterdir())
        if any(entry.name in reserved_names for entry in staged_entries):
            raise ValueError(
                "workspace archive collides with internal restore paths"
            )
        existing_entries = [
            entry
            for entry in WORKSPACE_DIR.iterdir()
            if entry.name not in reserved_names
        ]
        installed: list[Path] = []
        try:
            for entry in existing_entries:
                os.replace(entry, backup / entry.name)
            for entry in staged_entries:
                target = WORKSPACE_DIR / entry.name
                os.replace(entry, target)
                installed.append(target)
        except BaseException:
            for target in reversed(installed):
                os.replace(target, stage / target.name)
            for entry in list(backup.iterdir()):
                os.replace(entry, WORKSPACE_DIR / entry.name)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        backup = None
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if backup is not None and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


async def _restore_workspace_before_cancelling(path: Path) -> None:
    task = asyncio.create_task(
        asyncio.to_thread(_restore_workspace_archive, path)
    )
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        with suppress(Exception):
            task.result()
        raise


def _bounded(source: ReadObject, limit: int) -> ReadObject:
    """``source`` held to this agent's own size limit, which is below agent-env's grant."""
    if source.size_bytes is not None and source.size_bytes > limit:
        raise TransferError("transfer_too_large", "The snapshot object exceeds the agent's size limit.")
    return source.model_copy(update={"max_bytes": min(source.max_bytes, limit)})


def _answer(uploaded: Uploaded) -> dict[str, Any]:
    """An upload, as agent-env reads it back."""
    return uploaded.model_dump(mode="json")


async def _stage_granted_workspace(source: ReadObject | None) -> Path | None:
    """The workspace archive behind a read grant, in a temp file; None when the capture has
    no workspace (no object, or the empty one a save writes without a workspace directory)."""
    if source is None:
        return None
    fd, raw_path = tempfile.mkstemp(prefix="codex-workspace-download-", suffix=".tar.gz")
    os.close(fd)
    path = Path(raw_path)
    try:
        await download(source, path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if path.stat().st_size == 0:
        path.unlink()
        return None
    return path


_INSTALL_DESCRIPTION = (
    "Commands to install this agent into an already-running task container "
    "(in place; no new image)."
)
# install.sh, from the build context install_agent unpacks, brings Node, the Codex CLI and a Python of the agent's
# own; the endpoint reaches it as environment, never as arguments.
_UNPACK = "mkdir -p {work_dir}/agent && tar -xzf {agent_ctx_tar} -C {work_dir}/agent"
_ENDPOINT = ("LITELLM_API_KEY={litellm_api_key} LITELLM_BASE_URL={litellm_base_url} "
             "LITELLM_EXTRA_HEADERS={litellm_extra_headers} AGENT_TRUST_ENV={trust_env} ")
_ENDPOINT_PARAMS = ["litellm_api_key", "litellm_base_url", "litellm_extra_headers", "trust_env"]
_INSTALL_COMMANDS = [_UNPACK, _ENDPOINT + "bash {work_dir}/agent/install.sh container {container} {a2a_port}"]
_HOST_WORKDIR = "/app"
_INSTALL_COMMANDS_HOST = [_UNPACK, _ENDPOINT + f"bash {{work_dir}}/agent/install.sh host {{agent_name}} {{a2a_port}} {_HOST_WORKDIR}"]


@a2a_agent(
    identity=AgentIdentity(
        name="Codex",
        description="A2A-compliant wrapper around the OpenAI Codex CLI.",
        version="1.0.0",
        input_modes=("text",),
        output_modes=("text",),
        skills=(
            AgentSkill(
                id="codex",
                name="codex",
                description="Runs the OpenAI Codex CLI with full tool access (shell, file edits, web).",
                tags=["coding", "agent"],
            ),
        ),
    ),
    config=CodexConfig,
    config_description="Set or read agent configuration such as model and role.",
    extensions=(
        enable(
            TRAJECTORY_V1,
            live=True,
            description="Retrieve a task's Codex --json trajectory, also while it runs.",
        ),
        enable(
            ATTRIBUTION_PROBE_V1,
            description=(
                "Read back the LiteLLM attribution headers the agent most recently emitted on an "
                "outbound LLM call. Used by the agent-env runtime attribution validator."
            ),
        ),
        enable(MCP_CONFIG_V1, description="Register MCP servers the Codex CLI should connect to."),
        enable(
            SKILL_CONFIG_V1,
            description="Register a skill loaded by the Codex CLI from ~/.agents/skills/.",
        ),
        enable(
            INSTALL_V1,
            description=_INSTALL_DESCRIPTION,
            install_commands=_INSTALL_COMMANDS,
            # Port the agent's A2A server listens on inside the container.
            a2a_port=8000,
            # Every placeholder referenced in install_commands must be in this list.
            required_params=["container", "work_dir", "agent_ctx_tar", "a2a_port", *_ENDPOINT_PARAMS],
            install_commands_host=_INSTALL_COMMANDS_HOST,
            required_params_host=["work_dir", "agent_ctx_tar", "a2a_port", "agent_name", *_ENDPOINT_PARAMS],
            workdir_host=_HOST_WORKDIR,
            user_host="root",
        ),
        enable(
            PEER_AGENTS_V1,
            description="Delegate to the task's other agents: the CLI reaches them through the SDK's peer_agents MCP tools.",
        ),
        enable(
            SNAPSHOT_V1,
            description=(
                "Capture and restore Codex threads and workspace via agent-env's "
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
class CodexAgent(DurableSessionRefs, AgentEnvAgent):
    """Bridges A2A tasks onto the `codex exec` CLI."""

    def __init__(self) -> None:
        # `effort` is per-turn and agent-env omits it for a step that declares none, so a
        # set without it clears the previous step's; readback then omits it, as it did.
        self._effort_cleared = False

    async def run(self, request: TaskRequest[CodexConfig]) -> TaskResult:
        request.trajectory.set_format(TRAJECTORY_FORMAT)
        prompt = "\n".join(part.text for part in request.parts if isinstance(part, TextPart))
        cfg = request.config
        context_id = request.context_id
        # Hold the per-context lock around the CLI invocation so /ext/snapshot
        # save sees an at-rest rollout JSONL.
        ctx_lock = await _get_context_lock(context_id)
        execution_task = asyncio.current_task()
        if execution_task is not None:
            _active_execution_tasks.setdefault(context_id, set()).add(execution_task)
        try:
            async with ctx_lock:
                # Read under the lock rather than from the request: a snapshot load that
                # held it may have just bound this context to a restored thread.
                thread_id = self.session_ref_for_context(context_id)

                def _record_event(event: dict) -> None:
                    request.trajectory.append(event)
                    if event.get("type") == "thread.started" and event.get("thread_id"):
                        self.set_session_ref_for_context(context_id, event["thread_id"])

                try:
                    await codex_config.write_mcp_servers(request.mcp_servers)
                    async with _workspace_barrier.shared():
                        result = await run_codex(
                            prompt=prompt,
                            thread_id=thread_id,
                            model=cfg.model,
                            effort=cfg.effort,
                            system_prompt=cfg.system_prompt,
                            output_format=cfg.output_format,
                            timeout_s=cfg.timeout_seconds,
                            # Forward LiteLLM attribution per subprocess.
                            project_id=cfg.project_id,
                            task_id=cfg.task_id,
                            event_callback=_record_event,
                            config_overrides=codex_config.role_overrides(request.mcp_servers, cfg.role),
                        )
                except Exception as e:
                    logger.error(f"codex CLI invocation raised: {e}", exc_info=True)
                    return (
                        TaskResult.builder()
                        .failed("infra_error", str(e)[:_ERROR_MESSAGE_LIMIT], error_type="infra_error")
                        .add_text(str(e))
                        .build()
                    )

                if result.session_id:
                    self.set_session_ref_for_context(context_id, result.session_id)
        finally:
            if execution_task is not None:
                _discard_execution_task(context_id, execution_task)

        data: dict[str, Any] = {"tool_call_count": result.tool_call_count}
        if result.attempts > 1:
            data["cli_attempts"] = result.attempts
        if result.token_usage:
            data["token_usage"] = result.token_usage
        # Reported for debuggability only; must not flip the task to `failed`.
        if result.warnings:
            data["warnings"] = [w[:_WARNING_CHARS] for w in result.warnings[:_WARNINGS_KEPT]]

        builder = TaskResult.builder()
        if result.error:
            builder.failed("cli_error", result.error[:_ERROR_MESSAGE_LIMIT])
        else:
            builder.succeeded()
        return (
            builder
            .add_text(result.response or (result.error or ""))
            .add_data(data)
            .session_ref(result.session_id)
            .usage(Usage(tool_call_count=result.tool_call_count))
            .build()
        )

    @extension(AGENT_CONFIG_V1.set)
    async def set_config(self, request: CodexConfig) -> dict[str, Any]:
        provided = request.model_fields_set
        if "effort" not in provided:
            request = request.model_copy(update={"effort": None})
        response = await self.default_handlers.call(AGENT_CONFIG_V1.set, request)
        self._effort_cleared = "effort" not in provided
        return response

    @extension(AGENT_CONFIG_V1.get)
    async def get_config(self) -> dict[str, Any]:
        """The set config fields; validators read them back to confirm a set was accepted."""
        response = await self.default_handlers.call(AGENT_CONFIG_V1.get)
        if self._effort_cleared:
            response["config"].pop("effort", None)
        return response

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(ATTRIBUTION_PROBE_V1.probe)
    async def probe_attribution(self) -> AttributionProbeResponse:
        """The most recent LiteLLM attribution the bridge handed to the CLI.

        Used by agent-env's `VerifyA2ALitellmAttributionRuntimeStep`. Records
        the LITELLM_CUSTOMER_ID / LITELLM_TAGS env-var values set on the most
        recent codex subprocess — these map to x-litellm-customer-id /
        x-litellm-tags HTTP headers via the env_http_headers block in
        ~/.codex/config.toml (written by codex_config.write_base_config)."""
        snapshot = get_last_attribution_snapshot()
        return AttributionProbeResponse(
            last_seen_attribution={
                key: value for key, value in snapshot.items() if key != "last_seen_at_utc" and isinstance(value, str)
            },
            last_seen_at_utc=snapshot.get("last_seen_at_utc"),
        )

    async def _context_trajectory(self, context_id: str) -> dict[str, Any]:
        """The context's events from the framework's log, in this agent's own response shape."""
        trajectory: list[dict] = []
        after = 0
        while True:
            page = await self.default_handlers.call(
                TRAJECTORY_V1.get, ContextEventsTrajectoryRequest(context_id=context_id, after=after)
            )
            trajectory.extend(page.events)
            after = page.next
            if not page.has_more:
                break
        is_live = page.state in (TrajectoryState.PENDING, TrajectoryState.RUNNING)
        if is_live and not trajectory:
            raise HTTPException(
                status_code=404,
                detail=f"Trajectory has not started for context '{context_id}'",
            )
        return {
            "trajectory": trajectory,
            "is_live": is_live,
            "session_id": self.session_ref_for_context(context_id),
            "harness": HARNESS,
            "session_lines": len(trajectory),
        }

    @extension(TRAJECTORY_V1.get.context)
    async def get_context_trajectory(self, request: ContextTrajectoryRequest) -> dict[str, Any]:
        """The context's trajectory so far: every turn's events, the running turn's included."""
        return await self._context_trajectory(request.context_id)

    @extension(TRAJECTORY_V1.get.context_objects)
    async def upload_context_trajectory(self, request: ContextObjectTrajectoryRequest) -> dict[str, Any]:
        trajectory = (await self._context_trajectory(request.context_id))["trajectory"]
        body = json.dumps(trajectory, separators=(",", ":"), default=str).encode()
        return {"objects": {"trajectory": _answer(await upload(request.objects.trajectory, body))}}

    @extension(SNAPSHOT_V1.save)
    async def save_snapshot(self, request: ObjectSnapshotSaveRequest) -> dict[str, Any]:
        """Capture the Codex rollout and workspace."""
        # agent-env's write grants name the two objects, "trajectory" (the rollout JSONL)
        # and "workspace" (the workspace archive).
        context_id = request.context_id
        if not context_id:
            raise HTTPException(status_code=400, detail="context_id is required")
        trajectory_target = request.objects.trajectory
        workspace_target = request.objects.workspace

        thread_id = self.session_ref_for_context(context_id)
        if thread_id is None:
            raise HTTPException(status_code=404, detail=f"No active thread for context_id '{context_id}'")

        ctx_lock = await _get_context_lock(context_id)
        try:
            await _acquire_within(ctx_lock.acquire, ctx_lock.release, _LOCK_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=409, detail="session is mid-turn; retry after current turn completes")
        workspace_acquired = False
        archive_path: Path | None = None
        try:
            try:
                await _acquire_within(
                    _workspace_barrier.acquire_exclusive, _workspace_barrier.release_exclusive, _SNAPSHOT_QUIESCE_TIMEOUT_S
                )
                workspace_acquired = True
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=409,
                    detail="another context is mid-turn; retry after it completes",
                )
            rollout_path = find_rollout_path(thread_id)
            if rollout_path is None:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"no rollout JSONL found for thread_id={thread_id} under {SESSIONS_DIR} — "
                        "session must process at least one turn before snapshotting"
                    ),
                )
            try:
                jsonl_bytes = _read_complete_jsonl(rollout_path)
            except ValueError as e:
                raise HTTPException(status_code=500, detail=str(e))
            fd, raw_archive_path = tempfile.mkstemp(
                prefix="codex-workspace-", suffix=".tar.gz"
            )
            os.close(fd)
            archive_path = Path(raw_archive_path)
            # Without a workspace directory this uploads the empty file: agent-env
            # requires both objects, and a load reads an empty workspace as "conversation only".
            if workspace_target is not None and WORKSPACE_DIR.is_dir():
                await in_thread_to_its_end(_build_workspace_archive, archive_path)
        except BaseException:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)
            raise
        finally:
            if workspace_acquired:
                await _workspace_barrier.release_exclusive()
            ctx_lock.release()

        # Workspace first: agent-env reads a capture holding "trajectory" as a portable
        # snapshot, so an interrupted save never leaves one without its workspace.
        uploaded: dict[str, Any] = {}
        try:
            if workspace_target is not None:
                uploaded["workspace"] = _answer(await upload(workspace_target, archive_path))
            uploaded["trajectory"] = _answer(await upload(trajectory_target, jsonl_bytes))
        finally:
            archive_path.unlink(missing_ok=True)
        logger.info(f"Snapshot saved: context_id={context_id} thread_id={thread_id} -> object-transfer grants")
        return {"context_id": context_id, "objects": uploaded}

    def _discard_context_lock_if_unused(self, context_id: str, lock: asyncio.Lock) -> None:
        if (
            self.session_ref_for_context(context_id) is None
            and context_id not in _active_execution_tasks
            and _context_locks.get(context_id) is lock
            and not lock.locked()
        ):
            _context_locks.pop(context_id, None)

    @extension(SNAPSHOT_V1.load)
    async def load_snapshot(self, request: ObjectSnapshotLoadRequest) -> dict[str, Any]:
        """Restore a rollout and workspace under a fresh context/thread."""
        # agent-env's read grants for the objects the save wrote. agent-env registers a
        # capture only once both objects exist, and a capture without a (non-empty)
        # workspace is conversation-only.
        transcript_source = _bounded(request.objects.trajectory, _MAX_TRANSCRIPT_BYTES)
        workspace_source = (
            _bounded(request.objects.workspace, _MAX_SNAPSHOT_COMPRESSED_BYTES)
            if request.objects.workspace is not None
            else None
        )

        target_context_id = request.target_context_id
        if target_context_id is not None:
            if not target_context_id:
                raise HTTPException(status_code=400, detail="target_context_id must be a non-empty string")
            new_context_id = target_context_id
        else:
            new_context_id = uuid.uuid4().hex
        new_thread_id = str(uuid.uuid4())

        jsonl_bytes = await read_object_bytes(transcript_source)
        try:
            jsonl_bytes.decode()
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=400, detail=f"invalid snapshot trajectory: {e}")
        workspace_path: Path | None = None
        # Rewrite the thread_id in the first JSONL line's SessionMeta record.
        # The wire shape is:
        #   {"timestamp": ..., "type": "session_meta", "payload": {"id": <uuid>, ...}}
        # The Rust source uses #[serde(rename = "id")] on the SessionMeta.thread_id
        # field, so it serializes as `payload.id`. `codex exec resume <thread_id>`
        # validates both the filename uuid suffix AND read_session_meta_line's
        # returned id — both must match new_thread_id.
        first_line_end = jsonl_bytes.find(b"\n")
        if first_line_end < 0:
            raise HTTPException(status_code=400, detail="snapshot JSONL is empty")
        try:
            first_entry = json.loads(jsonl_bytes[:first_line_end])
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"first JSONL line is not valid JSON: {e}")
        if first_entry.get("type") != "session_meta":
            raise HTTPException(
                status_code=400,
                detail=f"first JSONL line type is '{first_entry.get('type')}', expected 'session_meta'",
            )
        payload = first_entry.get("payload")
        if not isinstance(payload, dict) or "id" not in payload:
            raise HTTPException(
                status_code=400,
                detail="session_meta payload missing 'id' field — unexpected codex JSONL format",
            )
        payload["id"] = new_thread_id
        rewritten_jsonl = (
            json.dumps(first_entry).encode() + jsonl_bytes[first_line_end:]
        )

        now = datetime.now(timezone.utc)
        target_dir = SESSIONS_DIR / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        # Codex's filename timestamp uses RFC3339 with `-` instead of `:` in the
        # time portion (see codex-rs/rollout/src/recorder.rs filename builder).
        ts = now.strftime("%Y-%m-%dT%H-%M-%S")
        target_path = target_dir / f"rollout-{ts}-{new_thread_id}.jsonl"
        target_path.write_bytes(rewritten_jsonl)

        target_lock = await _get_context_lock(new_context_id)
        try:
            await _acquire_within(target_lock.acquire, target_lock.release, _LOCK_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._discard_context_lock_if_unused(new_context_id, target_lock)
            target_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=409,
                detail="target context is active; retry snapshot load later",
            )
        except BaseException:
            self._discard_context_lock_if_unused(new_context_id, target_lock)
            target_path.unlink(missing_ok=True)
            raise
        try:
            await _acquire_within(
                _workspace_barrier.acquire_exclusive, _workspace_barrier.release_exclusive, _SNAPSHOT_QUIESCE_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            target_lock.release()
            self._discard_context_lock_if_unused(new_context_id, target_lock)
            target_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=409,
                detail="an agent turn is active; retry snapshot load after it completes",
            )
        except BaseException:
            target_lock.release()
            self._discard_context_lock_if_unused(new_context_id, target_lock)
            target_path.unlink(missing_ok=True)
            raise
        try:
            async with _context_locks_guard:
                if self.session_ref_for_context(new_context_id) is not None:
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"context_id '{new_context_id}' is already bound — "
                            "retry with a different target_context_id"
                        ),
                    )
            try:
                workspace_path = await _stage_granted_workspace(workspace_source)
            except BaseException:
                target_path.unlink(missing_ok=True)
                raise
            if workspace_path is not None:
                try:
                    await _restore_workspace_before_cancelling(workspace_path)
                except (OSError, tarfile.TarError, ValueError) as e:
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=400, detail=f"invalid workspace snapshot: {e}"
                    )
            async with _context_locks_guard:
                self.set_session_ref_for_context(new_context_id, new_thread_id)
        finally:
            await _workspace_barrier.release_exclusive()
            target_lock.release()
            self._discard_context_lock_if_unused(new_context_id, target_lock)
            if workspace_path is not None:
                workspace_path.unlink(missing_ok=True)

        logger.info(
            f"Snapshot loaded: object-transfer grants -> "
            f"context_id={new_context_id} thread_id={new_thread_id} "
            f"(file={target_path}, rewrote session_meta.payload.id)"
        )
        return {"context_id": new_context_id}


def build_app(agent: CodexAgent) -> Starlette:
    """The agent's Starlette app, answering errors with FastAPI's `{"detail": ...}` bodies."""
    return json_error_bodies(agent.create_app())


app = build_app(CodexAgent())
# agent-env's install/v1 deploys find the install parameters by importing this name from the image.
AGENT_CARD = app.state.agentenv_a2a.card
