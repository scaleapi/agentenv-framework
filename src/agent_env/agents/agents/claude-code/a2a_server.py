"""An A2A agent running the Claude Code CLI.

``AgentEnvAgent`` serves the Agent Card, the A2A task lifecycle and the SDK-owned extensions (agent-config,
mcp-config, skill listing, completed-task trajectories, triggers). Each context maps to a CLI session, so a
multi-turn conversation threads through ``claude --resume``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import posixpath
import shutil
import stat
import tarfile
import tempfile
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from a2a.types import AgentSkill
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
    ContextObjectTrajectoryRequest,
    ContextTrajectoryRequest,
    FilePart,
    InlineSkillRequest,
    NamespaceChangelogEnableRequest,
    NamespaceChangelogEnableResponse,
    ObjectChangelogApplyRequest,
    ObjectChangelogApplyResponse,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotLoadResponse,
    ObjectSnapshotSaveRequest,
    ObjectSnapshotSaveResponse,
    SkillAddResponse,
    SnapshotReadObjects,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    a2a_agent,
    enable,
    extension,
)
from agentenv_protocol.transfers import Uploaded, download, upload
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from tar_changelog import (
    HOOK_TIMEOUT_S,
    Changelog,
    changelog_dir,
    gnu_tar,
    is_missing_capture_command,
    merge_settings_hook,
    outermost,
    remove_settings_hooks,
)
from claude_bridge import (
    CLAUDE_STATE_DIR,
    SESSION_JSONL_DIR,
    WORKSPACE_DIR,
    kill_running_clis,
    read_partial_session_trajectory,
    redact_session_records,
    run_claude,
    session_jsonl_path,
)
from kit import (
    DurableSessionRefs,
    FilePartMaterializeError,
    install_skill,
    json_error_bodies,
    materialize_file_part,
    read_object_bytes,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HARNESS = "claude-code"
# The versioned format of the stream-json events a turn's trajectory holds.
TRAJECTORY_FORMAT = "claude-code-stream-json/1"
# The CLI loads skills from <state>/skills/<name>/SKILL.md.
SKILLS_DIR = CLAUDE_STATE_DIR / "skills"
# FileParts are written here, one folder per task, and their paths appended to the prompt: the CLI reads a path in
# the prompt and attaches an image or PDF as multimodal input.
INPUTS_DIR = WORKSPACE_DIR / "inputs"
_SKIPPABLE_WORKSPACE_ERRORS = (FileNotFoundError, PermissionError, NotADirectoryError, IsADirectoryError)

# What agent-env keeps of a failed task's error_message.
_ERROR_MESSAGE_LIMIT = 1000
# How much of a changing workspace file one read stages.
_STAGE_CHUNK_BYTES = 1024 * 1024

# install/v1: the folder, unpacked on the sandbox's host, installs itself (install.sh); install_agent fills each {param}
# in, quoted. The endpoint and CA go in the environment, not on the agent's command line.
_A2A_PORT = 8000
_UNPACK = "mkdir -p {work_dir}/agent && tar -xzf {agent_ctx_tar} -C {work_dir}/agent"
_ENDPOINT = ("LITELLM_API_KEY={litellm_api_key} LITELLM_BASE_URL={litellm_base_url} "
             "LITELLM_EXTRA_HEADERS={litellm_extra_headers} AGENT_TRUST_ENV={trust_env} ")
_ENDPOINT_PARAMS = ["litellm_api_key", "litellm_base_url", "litellm_extra_headers", "trust_env"]
_INSTALL_COMMANDS = [_UNPACK, _ENDPOINT + "bash {work_dir}/agent/install.sh container {container} {a2a_port}"]
# On a host there's no container working directory: the agent works in /app, as in its image.
_HOST_WORKDIR = "/app"
_INSTALL_COMMANDS_HOST = [_UNPACK, _ENDPOINT + f"bash {{work_dir}}/agent/install.sh host {{agent_name}} {{a2a_port}} {_HOST_WORKDIR}"]


class ClaudeCodeConfig(AgentConfig):
    timeout_seconds: int = 1800
    model: str | None = None
    allowed_tools: list[str] | None = None
    system_prompt: str | None = None
    output_format: dict[str, Any] | None = None
    # LiteLLM spend attribution, sent as headers on each model call.
    project_id: str | None = None
    task_id: str | None = None
    # The CLI's --effort: low, medium, high, xhigh, max, ultracode or auto.
    effort: str | None = None
    max_turns: int | None = None
    max_thinking_tokens: int | None = None


def _rewrite_session_ids(jsonl_text: str, session_id: str) -> tuple[list[str], int]:
    """The transcript's lines with every record's `sessionId` set to `session_id`, and how many records carried
    one. Blank lines are dropped and unparseable ones kept verbatim."""
    rewritten: list[str] = []
    rewrite_count = 0
    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            rewritten.append(line)
            continue
        if isinstance(entry, dict) and "sessionId" in entry:
            entry["sessionId"] = session_id
            rewrite_count += 1
        rewritten.append(json.dumps(entry))
    return rewritten, rewrite_count


def _restorable_link(path: Path, arcname: str) -> bool:
    """Whether the link at ``path`` restores where ``arcname`` puts it: relative, and inside the archive, as the load's
    data filter requires."""
    target = os.readlink(path)
    if posixpath.isabs(target):
        return False
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(arcname), target))
    return resolved != ".." and not resolved.startswith("../")


def _add_workspace_entry(tar: tarfile.TarFile, path: Path, arcname: str, *, stage_source: bool) -> bool:
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) and not _restorable_link(path, arcname):
            return False
    except OSError:
        return False
    # The load's data filter refuses pipes, sockets and devices, so leaving them out keeps the snapshot loadable.
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
        return False
    if not stage_source:
        try:
            tar.add(path, arcname=arcname, recursive=False)
        except _SKIPPABLE_WORKSPACE_ERRORS:
            return False
        return True
    try:
        tarinfo = tar.gettarinfo(path, arcname=arcname)
    except OSError:
        return False
    if not tarinfo.isreg():
        tar.addfile(tarinfo)
        return True
    try:
        source = path.open("rb")
    except OSError:
        return False
    with source, tempfile.TemporaryFile() as staged:
        remaining = tarinfo.size
        while remaining:
            try:
                chunk = source.read(min(_STAGE_CHUNK_BYTES, remaining))
            except OSError:
                return False
            if not chunk:
                break
            staged.write(chunk)
            remaining -= len(chunk)
        tarinfo.size = staged.tell()
        staged.seek(0)
        tar.addfile(tarinfo, staged)
    return True


def _write_workspace_archive(archive_path: Path, *, stage_sources: bool = False) -> tuple[int, int]:
    def add_tree(tar: tarfile.TarFile, source_root: Path, *, archive_prefix: str = "",
                 excluded_top_level: frozenset[str] = frozenset()) -> int:
        count = 0
        for dirpath, dirnames, filenames in os.walk(source_root, followlinks=False):
            dirnames.sort()
            filenames.sort()
            root = Path(dirpath)
            if root == source_root:
                dirnames[:] = [d for d in dirnames if d not in excluded_top_level]
                filenames = [f for f in filenames if f not in excluded_top_level]
            relative_root = root.relative_to(source_root)
            if relative_root.parts:
                arcname = "/".join(p for p in (archive_prefix, relative_root.as_posix()) if p)
                if not _add_workspace_entry(tar, root, arcname, stage_source=stage_sources):
                    continue
            # A link to a directory is listed here and never walked: it goes in as a link, like a file's.
            filenames = sorted([*filenames, *(d for d in dirnames if (root / d).is_symlink())])
            for filename in filenames:
                path = root / filename
                arcname = "/".join(p for p in (archive_prefix, path.relative_to(source_root).as_posix()) if p)
                if _add_workspace_entry(tar, path, arcname, stage_source=stage_sources):
                    count += 1
        return count

    with tarfile.open(archive_path, mode="w:gz", dereference=False) as tar:
        # The CLI's state goes in as `.claude` from wherever it lives, so a `/app/.claude` isn't added twice.
        file_count = add_tree(tar, WORKSPACE_DIR, excluded_top_level=frozenset({".claude"}))
        if CLAUDE_STATE_DIR.is_dir():
            file_count += add_tree(tar, CLAUDE_STATE_DIR, archive_prefix=".claude")
    return file_count, archive_path.stat().st_size


def _build_workspace_archive(archive_path: Path) -> tuple[int, int]:
    """``/app`` and the CLI's state as a gzip tar on disk, out of the agent's heap; its file count and size. A file
    that changes while it's read rebuilds the archive with each file staged first, so it can be skipped. Symlinked
    directories aren't followed."""
    if not WORKSPACE_DIR.is_dir():
        raise FileNotFoundError(f"workspace directory does not exist: {WORKSPACE_DIR}")
    try:
        return _write_workspace_archive(archive_path)
    except OSError:
        logger.warning("Workspace changed while archiving; rebuilding with staged files")
        return _write_workspace_archive(archive_path, stage_sources=True)


def _read_complete_session_jsonl(path: Path) -> bytes:
    """The transcript's newline-complete records, while the CLI may still be appending, its API error records
    redacted."""
    data = path.read_bytes()
    if not data:
        raise ValueError("session transcript is empty")
    if not data.endswith(b"\n"):
        complete, separator, tail = data.rpartition(b"\n")
        if not separator:
            try:
                json.loads(tail)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("session transcript has no complete JSONL record") from exc
        data = (complete if separator else tail) + b"\n"
    return redact_session_records(data)


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


def _copy_directory_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_symlink():
            target.symlink_to(os.readlink(child), target_is_directory=child.is_dir())
        elif child.is_dir():
            shutil.copytree(child, target, symlinks=True)
        else:
            shutil.copy2(child, target)


def _restore_workspace_archive(archive_path: Path, keep: Iterable[Path] = ()) -> None:
    """Restore the workspace's files, and the archive's `.claude` to the CLI's state directory, keeping the current
    copy of each file in ``keep``: the transcripts of the conversations this agent resumes."""
    with tempfile.TemporaryDirectory(prefix="claude-snapshot-load-") as raw_stage:
        stage, kept = Path(raw_stage) / "archive", Path(raw_stage) / "kept"
        kept.mkdir()
        held = [(path, kept / str(index)) for index, path in enumerate(keep) if path.is_file()]
        for path, copy in held:
            shutil.copy2(path, copy)
        with tarfile.open(archive_path, mode="r:gz") as tar:
            # The data filter refuses absolute paths, traversal, devices and unsafe links before anything is written.
            tar.extractall(stage, filter="data")
        staged_state = stage / ".claude"
        if not staged_state.is_dir() or staged_state.is_symlink():
            raise ValueError("workspace archive is missing Claude internal state")
        _clear_directory(WORKSPACE_DIR)
        for child in stage.iterdir():
            if child.name == ".claude":
                continue
            target = WORKSPACE_DIR / child.name
            if child.is_symlink():
                target.symlink_to(os.readlink(child), target_is_directory=child.is_dir())
            elif child.is_dir():
                shutil.copytree(child, target, symlinks=True)
            else:
                shutil.copy2(child, target)
        _clear_directory(CLAUDE_STATE_DIR)
        _copy_directory_contents(staged_state, CLAUDE_STATE_DIR)
        for path, copy in held:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(copy, path)


def _answer(uploaded: Uploaded) -> dict[str, Any]:
    """An upload, as agent-env reads it back."""
    return uploaded.model_dump(mode="json")


async def _fetch_granted_snapshot(objects: SnapshotReadObjects, archive_path: Path) -> tuple[str, bool]:
    """The conversation JSONL behind the "trajectory" read grant, and whether a "workspace" archive was sent and
    written to ``archive_path``; a conversation-only snapshot has none."""
    if objects.workspace is not None:
        await download(objects.workspace, archive_path)
    jsonl = await read_object_bytes(objects.trajectory)
    return jsonl.decode(), objects.workspace is not None


def _changelog_roots() -> list[str]:
    """The changelog's default capture roots: the writable layer a fresh agent of the image reconstructs, the CLI's
    state (and its transcripts) included."""
    return outermost([str(Path.home()), str(CLAUDE_STATE_DIR), str(WORKSPACE_DIR), "/tmp/data", "/usr/local", "/etc"])


# After every tool call, whether it succeeded or failed, so each call takes its position and its writes are captured.
_CAPTURE_EVENTS = ("PostToolUse", "PostToolUseFailure")


def _install_changelog_hook(command: str) -> None:
    """The capture as a hook in the CLI's user settings, which `--setting-sources user` reads."""
    for event in _CAPTURE_EVENTS:
        merge_settings_hook(CLAUDE_STATE_DIR / "settings.json", event=event, matcher="", command=command,
                            timeout=HOOK_TIMEOUT_S)


def _remove_restored_capture_hooks() -> None:
    """Drop capture hooks restored with the CLI's settings, whose scripts stayed on the agent that captured."""
    for event in _CAPTURE_EVENTS:
        remove_settings_hooks(CLAUDE_STATE_DIR / "settings.json", event=event, matches=is_missing_capture_command)


@asynccontextmanager
async def _turns_stop_with_the_server(app: Starlette) -> AsyncIterator[None]:
    """The server's lifespan: as it shuts down, the CLIs of turns still running are killed with their tools, which
    would otherwise go on in the workspace after it."""
    try:
        yield
    finally:
        if stopped := kill_running_clis():
            logger.info(f"Killed {stopped} claude CLI(s) still running a turn as the agent shut down")


@a2a_agent(
    identity=AgentIdentity(
        name="Claude Code",
        description="An A2A agent running the Claude Code CLI.",
        version="1.0.0",
        input_modes=("text", "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"),
        output_modes=("text",),
        skills=(
            AgentSkill(
                id="claude-code",
                name="claude-code",
                description="Runs the Claude Code CLI with full tool access (bash, read, edit, glob, grep).",
                tags=["coding", "agent"],
            ),
        ),
    ),
    config=ClaudeCodeConfig,
    config_description="Set or read agent configuration such as model, role, and allowed tools.",
    lifespan=_turns_stop_with_the_server,
    extensions=(
        enable(TRAJECTORY_V1, live=True, description=(
            "Retrieve a task's Claude Code stream-json trajectory, also while it runs, "
            "or a context's cumulative trajectory including a live run.")),
        enable(MCP_CONFIG_V1, description="Register MCP servers the Claude Code CLI should connect to."),
        enable(PEER_AGENTS_V1, description=(
            "Delegate to the task's other agents: the CLI reaches them through the SDK's peer_agents MCP tools.")),
        enable(SKILL_CONFIG_V1, description="Register a skill the Claude Code CLI loads from its skills directory."),
        enable(SNAPSHOT_V1, description=(
            "Capture and restore the Claude Code conversation, its workspace and the CLI's state, "
            "or, where the host has GNU tar, each tool call's writes as a changelog to replay onto a fresh agent, "
            "through agent-env's object-transfer grants.")),
        enable(
            INSTALL_V1,
            description="Commands to install this agent into a running task container, or onto its sandbox's host.",
            install_commands=_INSTALL_COMMANDS,
            a2a_port=_A2A_PORT,
            required_params=["container", "work_dir", "agent_ctx_tar", "a2a_port", *_ENDPOINT_PARAMS],
            install_commands_host=_INSTALL_COMMANDS_HOST,
            required_params_host=["work_dir", "agent_ctx_tar", "a2a_port", "agent_name", *_ENDPOINT_PARAMS],
            workdir_host=_HOST_WORKDIR,
            user_host="root",
        ),
        enable(TRIGGERS_V1, description=(
            "Register a typed trigger script and decide the stakeholder reply per turn "
            "(step / env_trigger / conversational / all / any conditions -> say / end actions).")),
    ),
)
class ClaudeCodeAgent(DurableSessionRefs, AgentEnvAgent):
    """Bridges A2A tasks onto the `claude` CLI."""

    def __init__(self) -> None:
        # Contexts whose CLI is running. Trajectory reads and snapshot saves take no lock, so this reports liveness
        # without waiting on the turn; a snapshot load holds the lock across its restore.
        self._active_contexts: set[str] = set()
        self._contexts_lock = asyncio.Lock()

    async def run(self, request: TaskRequest[ClaudeCodeConfig]) -> TaskResult:
        # The CLI's stdout events, appended as it prints them, are the turn's trajectory.
        request.trajectory.set_format(TRAJECTORY_FORMAT)
        cfg = request.config
        context_id = request.context_id
        async with self._contexts_lock:
            # Active from here, so a snapshot load can't clear the attachments written below.
            self._active_contexts.add(context_id)
            # Bound before the CLI starts, so a live trajectory read or a snapshot save finds this turn's session.
            session_id = self.session_ref_for_context(context_id)
            if session_id is None:
                session_id = str(uuid.uuid4())
                self.set_session_ref_for_context(context_id, session_id)
        try:
            return await self._turn(request, cfg, context_id, session_id)
        finally:
            self._active_contexts.discard(context_id)

    async def _turn(self, request: TaskRequest[ClaudeCodeConfig], cfg: ClaudeCodeConfig, context_id: str,
                    session_id: str) -> TaskResult:
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
            return (TaskResult.builder().failed("filepart_error", str(e)[:_ERROR_MESSAGE_LIMIT])
                    .add_text(str(e)).build())

        user_text = "\n".join(c for c in text_chunks if c)
        if attachment_paths:
            paths_block = "\n".join(attachment_paths)
            user_text = f"{user_text}\n\n{paths_block}" if user_text else paths_block

        # A turn that failed before the CLI wrote its transcript left no session to resume.
        resume = session_jsonl_path(session_id).exists()
        try:
            result = await run_claude(
                prompt=user_text,
                session_id=session_id,
                resume=resume,
                model=cfg.model,
                allowed_tools=cfg.allowed_tools,
                mcp_servers=dict(request.mcp_servers),
                system_prompt=cfg.system_prompt,
                output_format=cfg.output_format,
                timeout_s=cfg.timeout_seconds,
                project_id=cfg.project_id,
                task_id=cfg.task_id,
                role=cfg.role,
                effort=cfg.effort,
                max_turns=cfg.max_turns,
                max_thinking_tokens=cfg.max_thinking_tokens,
                event_callback=request.trajectory.append,
            )
        except Exception as e:
            logger.error(f"claude CLI invocation raised: {e}", exc_info=True)
            return (TaskResult.builder().failed("infra_error", str(e)[:_ERROR_MESSAGE_LIMIT], error_type="infra_error")
                    .add_text(str(e)).build())
        # The CLI keeps the session id it's given; if it ever reports another, the next turn resumes that one.
        if result.session_id and result.session_id != session_id:
            session_id = result.session_id
            self.set_session_ref_for_context(context_id, session_id)

        data: dict[str, Any] = {"tool_call_count": result.tool_call_count}
        if result.attempts > 1:
            data["cli_attempts"] = result.attempts
        builder = TaskResult.builder()
        if result.error:
            builder.failed("cli_error", result.error[:_ERROR_MESSAGE_LIMIT])
        else:
            builder.succeeded()
        return (
            builder
            # A CLI that failed before its `result` event has no response; the error is the diagnostic.
            .add_text(result.response or (result.error or ""))
            .add_data(data)
            .session_ref(session_id)
            .usage(Usage(tool_call_count=result.tool_call_count))
            .build()
        )

    async def _context_trajectory(self, context_id: str) -> dict[str, Any]:
        session_id = self.session_ref_for_context(context_id)
        if session_id is None:
            raise HTTPException(status_code=404, detail=f"No session available for context '{context_id}'")
        try:
            trajectory, session_lines = await asyncio.to_thread(read_partial_session_trajectory, session_id)
        except FileNotFoundError:
            # The session is bound just before the CLI starts, so a first read can beat its transcript.
            raise HTTPException(status_code=404, detail=f"No transcript yet for context_id '{context_id}'")
        return {
            "trajectory": trajectory,
            "session_lines": session_lines,
            "is_live": context_id in self._active_contexts,
            "session_id": session_id,
            "harness": HARNESS,
        }

    @extension(TRAJECTORY_V1.get.context)
    async def get_context_trajectory(self, request: ContextTrajectoryRequest) -> dict[str, Any]:
        """A context's cumulative trajectory, including a live run."""
        return await self._context_trajectory(request.context_id)

    @extension(TRAJECTORY_V1.get.context_objects)
    async def upload_context_trajectory(self, request: ContextObjectTrajectoryRequest) -> dict[str, Any]:
        trajectory = (await self._context_trajectory(request.context_id))["trajectory"]
        body = json.dumps(trajectory, separators=(",", ":"), default=str).encode()
        return {"objects": {"trajectory": _answer(await upload(request.objects.trajectory, body))}}

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(SNAPSHOT_V1.save)
    async def save_snapshot(self, request: ObjectSnapshotSaveRequest) -> ObjectSnapshotSaveResponse:
        """Capture a loadable bundle: "trajectory" is the conversation JSONL, "workspace" the /app and CLI state
        archive."""
        context_id = request.context_id
        if not context_id:
            raise HTTPException(status_code=400, detail="context_id is required")
        session_id = self.session_ref_for_context(context_id)
        if session_id is None:
            raise HTTPException(status_code=404, detail=f"No session available for context_id '{context_id}'")
        try:
            jsonl_bytes = await asyncio.to_thread(_read_complete_session_jsonl, session_jsonl_path(session_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"No transcript yet for context_id '{context_id}'") from exc
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        archive_fd, raw_archive_path = tempfile.mkstemp(prefix="claude-workspace-", suffix=".tar.gz")
        jsonl_fd, raw_jsonl_path = tempfile.mkstemp(prefix="claude-conversation-", suffix=".jsonl")
        os.close(archive_fd)
        os.close(jsonl_fd)
        archive_path, snapshot_jsonl_path = Path(raw_archive_path), Path(raw_jsonl_path)
        try:
            file_count, bytes_written = await asyncio.to_thread(_build_workspace_archive, archive_path)
            await asyncio.to_thread(snapshot_jsonl_path.write_bytes, jsonl_bytes)
            uploaded = {"trajectory": _answer(await upload(request.objects.trajectory, snapshot_jsonl_path))}
            if request.objects.workspace is not None:
                uploaded["workspace"] = _answer(await upload(request.objects.workspace, archive_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            archive_path.unlink(missing_ok=True)
            snapshot_jsonl_path.unlink(missing_ok=True)
        logger.info(f"Snapshot saved: context_id={context_id} session_id={session_id} files={file_count} "
                    f"bytes={bytes_written}")
        return ObjectSnapshotSaveResponse.model_validate({"context_id": context_id, "objects": uploaded})

    @extension(SNAPSHOT_V1.load)
    async def load_snapshot(self, request: ObjectSnapshotLoadRequest) -> ObjectSnapshotLoadResponse:
        """Restore a full snapshot, or a conversation-only one."""
        target_context_id = request.target_context_id
        if target_context_id is not None and not target_context_id:
            raise HTTPException(status_code=400, detail="target_context_id must be a non-empty string")
        new_context_id = target_context_id or uuid.uuid4().hex
        new_session_id = str(uuid.uuid4())

        archive_fd, raw_archive_path = tempfile.mkstemp(prefix="claude-workspace-load-", suffix=".tar.gz")
        os.close(archive_fd)
        archive_path = Path(raw_archive_path)
        try:
            jsonl_text, has_workspace_archive = await _fetch_granted_snapshot(request.objects, archive_path)
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise

        try:
            rewritten, rewrite_count = _rewrite_session_ids(jsonl_text, new_session_id)
            if rewrite_count == 0:
                raise HTTPException(status_code=400, detail="no sessionId fields found in JSONL — unexpected format")
            async with self._contexts_lock:
                if self.session_ref_for_context(new_context_id) is not None:
                    raise HTTPException(status_code=409, detail=(
                        f"context_id '{new_context_id}' is already bound — retry with a different target_context_id"))
                if self._active_contexts:
                    raise HTTPException(status_code=409, detail="cannot load a workspace while an agent turn is running")
                if has_workspace_archive:
                    keep = [session_jsonl_path(session) for session in self.bound_session_refs()]
                    try:
                        await asyncio.to_thread(_restore_workspace_archive, archive_path, keep)
                    except (tarfile.TarError, OSError, ValueError) as exc:
                        raise HTTPException(status_code=400, detail=f"invalid workspace archive: {exc}") from exc
                    _remove_restored_capture_hooks()
                SESSION_JSONL_DIR.mkdir(parents=True, exist_ok=True)
                session_jsonl_path(new_session_id).write_text("\n".join(rewritten) + "\n")
                self.set_session_ref_for_context(new_context_id, new_session_id)
        finally:
            archive_path.unlink(missing_ok=True)
        logger.info(f"Snapshot loaded: context_id={new_context_id} session_id={new_session_id} "
                    f"(workspace={'restored' if has_workspace_archive else 'not present'}; "
                    f"rewrote {rewrite_count} sessionId fields)")
        return ObjectSnapshotLoadResponse(context_id=new_context_id)


class ClaudeCodeChangelogAgent(ClaudeCodeAgent):
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

    async def run(self, request: TaskRequest[ClaudeCodeConfig]) -> TaskResult:
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
        async with self._contexts_lock:
            if self._active_contexts:
                raise HTTPException(status_code=409, detail="cannot apply a changelog while an agent turn is running")
            return await self._changelog.apply(request)

    async def _resume_conversation(self, target_context_id: str | None, session: str | None) -> str | None:
        """Bind a context to the restored transcript of ``session``, the last applied tool call's, under a fresh
        session id, so the next turn there resumes the rewound conversation; None when no transcript was restored."""
        recorded = SESSION_JSONL_DIR / f"{session}.jsonl" if session and Path(session).name == session else None
        if recorded is not None and recorded.is_file():
            source = recorded
        else:
            # Increments that recorded no session, or whose transcript wasn't restored: the largest is the likeliest.
            candidates = list(SESSION_JSONL_DIR.glob("*.jsonl")) if SESSION_JSONL_DIR.is_dir() else []
            if not candidates:
                logger.info("agent-changelog resume: no restored transcript")
                return None
            source = max(candidates, key=lambda p: p.stat().st_size)
        new_session_id = str(uuid.uuid4())
        rewritten, _ = _rewrite_session_ids(source.read_text(), new_session_id)
        session_jsonl_path(new_session_id).write_text("\n".join(rewritten) + "\n")
        new_context_id = target_context_id or uuid.uuid4().hex
        self.set_session_ref_for_context(new_context_id, new_session_id)
        logger.info(f"agent-changelog resume: bound context {new_context_id} -> session {new_session_id} "
                    f"(from {source.name})")
        return new_context_id


def agent_for_this_host() -> ClaudeCodeAgent:
    """The agent, with the changelog where this host's tar can make its archives."""
    return ClaudeCodeChangelogAgent() if gnu_tar() else ClaudeCodeAgent()


def build_app(agent: ClaudeCodeAgent) -> Starlette:
    """The agent's Starlette app, answering errors with ``{"detail": ...}`` bodies."""
    return json_error_bodies(agent.create_app())


app = build_app(agent_for_this_host())
# install_agent reads the agent's card by importing this name from the image.
AGENT_CARD = app.state.agentenv_a2a.card
