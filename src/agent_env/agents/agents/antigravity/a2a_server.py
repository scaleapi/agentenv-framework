"""A2A agent running Google's Antigravity SDK.

Built on agentenv-protocol's `AgentEnvAgent`: the framework serves the Agent Card, the A2A task lifecycle and the
SDK-owned extensions (agent-config, mcp-config, skill listing, task trajectories, peer agents, triggers). Each turn is
one `Agent.chat()` of the Antigravity SDK in this process (`antigravity_bridge`), whose streamed chunks are appended to
the framework's trajectory log as they come. A context's first turn starts an SDK conversation, which later turns
resume by its id.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from weakref import WeakValueDictionary

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
    BuiltinTools,
    BundleSkillRequest,
    DataPart,
    FilePart,
    InlineSkillRequest,
    NamespaceChangelogEnableRequest,
    NamespaceUploader,
    ObjectChangelogApplyRequest,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    SkillAddResponse,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    a2a_agent,
    download,
    enable,
    extension,
    upload,
)
from pydantic import Field
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from antigravity_bridge import (
    CONVERSATION_RECORDS,
    ENV_TIMES,
    ERROR_MESSAGE_LIMIT,
    IMAGES_DIR,
    SAVE_DIR,
    SKILLS_DIR,
    TURN_IMAGES,
    AntigravityTool,
    file_records,
    get_last_attribution_snapshot,
    run_antigravity,
)
from antigravity_kit import DurableSessionRefs, install_skill, json_error_bodies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WORKSPACE_DIR = Path(os.environ.get("ANTIGRAVITY_WORKSPACE_DIR", "/app"))
INPUTS_DIR = WORKSPACE_DIR / "inputs"
# One per port, since agents installed on one host listen on their own.
_CHANGELOG_DIR = Path(tempfile.gettempdir()) / f"antigravity-changelog-{os.environ.get('A2A_PORT') or 'default'}"
_CHANGELOG_SNAR = _CHANGELOG_DIR / "state.snar"
# Each increment's record of the roots it saw removed, in a directory of its own beside the changelog, which captures
# exclude, and outside its roots: tar dumps a directory it hasn't seen whole, so the record goes in however soon after
# the last capture it is written.
_META_DIR_SUFFIX = ".changelog-meta"
_META_NAME = "antigravity-changelog-meta.json"
_CHANGELOG_META_DIR = _CHANGELOG_DIR.parent / f"{_CHANGELOG_DIR.name}{_META_DIR_SUFFIX}"
_META_MEMBER = re.compile(rf"(?:^|/)[^/]+{re.escape(_META_DIR_SUFFIX)}/\d+/{re.escape(_META_NAME)}\Z")
# Never removed on replay, whatever an increment's record says.
_SYSTEM_DIRS = frozenset({"/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64", "/libx32",
                          "/media", "/mnt", "/opt", "/proc", "/root", "/run", "/sbin", "/srv", "/sys", "/tmp", "/usr",
                          "/var"})
_TAR_VERSION_TIMEOUT_S = 10
_FILE_DOWNLOAD_TIMEOUT_S = 60
_PDF_RENDER_TIMEOUT_S = 60
_PDF_RENDER_DPI = 150
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")
# The versioned format of the SDK chunks and hook results this agent appends to its trajectory.
TRAJECTORY_FORMAT = "antigravity-sdk/1"
_MANIFEST = "manifest.json"
_ENV_TIMES_MEMBER = "env-times.json"
_TURN_IMAGES_MEMBER = "turn-images.json"
_IMAGES_MEMBER = "images"


class AntigravityConfig(AgentConfig):
    model: str | None = None
    effort: str | None = None
    system_prompt: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    timeout_seconds: int = Field(default=1800, gt=0)
    # The model calls one turn may make; the SDK stops the turn at the cap.
    max_turns: int | None = Field(default=None, gt=0)
    # Which of the SDK's own tools it may use. With any selection, the tools of the MCP servers agent-env gives it stay.
    builtin_tools: BuiltinTools[AntigravityTool] | None = None
    # False: the SDK's own web tools, search_web and read_url_content, are off whatever builtin_tools selects.
    web_tools: bool = True


class FilePartMaterializeError(Exception):
    """A supplied file cannot be passed to the agent."""


def _safe_name(candidate: str | None, fallback_ext: str = "") -> str:
    if candidate:
        candidate = _SAFE_NAME_RE.sub("_", Path(candidate).name)
    return candidate or f"{uuid.uuid4().hex}{fallback_ext}"


def _download_uri_to(uri: str, target: Path) -> None:
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        raise FilePartMaterializeError(f"unsupported file URI scheme {parsed.scheme!r}")
    try:
        with httpx.Client(timeout=_FILE_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
            response = client.get(uri)
            response.raise_for_status()
            target.write_bytes(response.content)
    except httpx.HTTPError as exc:
        raise FilePartMaterializeError("HTTP file download failed") from exc


def _materialize_and_render(part: FilePart, task_id: str) -> list[str]:
    """The files a part gives the turn, under the workspace's inputs: the file, or a PDF's pages as images, which the
    model sees inline where it can't read the PDF itself."""
    directory = INPUTS_DIR / task_id
    directory.mkdir(parents=True, exist_ok=True)
    name = _safe_name(part.name)
    target = directory / name
    if target.exists():
        target = directory / f"{uuid.uuid4().hex[:8]}-{name}"
    if part.bytes is not None:
        try:
            target.write_bytes(base64.b64decode(part.bytes, validate=True))
        except (binascii.Error, ValueError, OSError) as exc:
            raise FilePartMaterializeError(f"invalid file attachment {name!r}") from exc
    else:
        _download_uri_to(part.uri, target)
    if (part.mime_type or "").lower() != "application/pdf" and target.suffix.lower() != ".pdf":
        return [str(target)]
    prefix = str(target.with_suffix("")) + "_page"
    try:
        subprocess.run(["pdftoppm", "-png", "-r", str(_PDF_RENDER_DPI), str(target), prefix],
                       check=True, capture_output=True, timeout=_PDF_RENDER_TIMEOUT_S)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("PDF rendering failed for %s; passing the PDF directly", target)
        return [str(target)]
    pages = sorted(directory.glob(f"{Path(prefix).name}*.png"))
    return [str(target), *(str(page) for page in pages)]


class _WorkspaceBarrier:
    """Turns share the workspace and snapshots take it alone, so a snapshot's conversation and files come from a point
    no turn is writing at, while turns of different contexts run at once. A waiting snapshot holds new turns back, so
    it is never starved."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @asynccontextmanager
    async def shared(self):
        async with self._condition:
            await self._condition.wait_for(lambda: not self._writer and self._waiting_writers == 0)
            self._readers += 1
        try:
            yield
        finally:
            async with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @asynccontextmanager
    async def exclusive(self):
        async with self._condition:
            self._waiting_writers += 1
            try:
                await self._condition.wait_for(lambda: not self._writer and self._readers == 0)
                self._writer = True
            finally:
                self._waiting_writers -= 1
                if not self._writer:
                    self._condition.notify_all()
        try:
            yield
        finally:
            async with self._condition:
                self._writer = False
                self._condition.notify_all()


# The turns running now, cancelled as the server stops: each one's SDK agent then ends its harness, which would
# otherwise go on in the workspace after it.
_RUNNING: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: Starlette):
    try:
        yield
    finally:
        running = [task for task in _RUNNING if not task.done()]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
            logger.info(f"Cancelled {len(running)} Antigravity turn(s) still running as the agent shut down")


# install.sh, from the build context install_agent unpacks, brings uv and a Python of the agent's own with the SDK;
# the endpoint reaches it as environment, never as arguments.
_UNPACK = "mkdir -p {work_dir}/agent && tar -xzf {agent_ctx_tar} -C {work_dir}/agent"
_ENDPOINT = ("LITELLM_API_KEY={litellm_api_key} LITELLM_BASE_URL={litellm_base_url} "
             "LITELLM_EXTRA_HEADERS={litellm_extra_headers} AGENT_TRUST_ENV={trust_env} ")
_ENDPOINT_PARAMS = ["litellm_api_key", "litellm_base_url", "litellm_extra_headers", "trust_env"]
_INSTALL_COMMANDS = [_UNPACK, _ENDPOINT + "bash {work_dir}/agent/install.sh container {container} {a2a_port}"]
_HOST_WORKDIR = "/app"
_INSTALL_COMMANDS_HOST = [_UNPACK, _ENDPOINT + f"bash {{work_dir}}/agent/install.sh host {{agent_name}} {{a2a_port}} {_HOST_WORKDIR}"]


@a2a_agent(
    identity=AgentIdentity(
        name="Antigravity",
        description="A2A-compliant agent running Google's Antigravity SDK.",
        version="1.0.0",
        input_modes=("text", "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"),
        output_modes=("text",),
        skills=(AgentSkill(
            id="antigravity", name="antigravity",
            description="Runs Google's Antigravity agent with its own tools (shell, files, subagents).",
            tags=["coding", "agent"],
        ),),
    ),
    config=AntigravityConfig,
    extensions=(
        MCP_CONFIG_V1,
        enable(TRAJECTORY_V1, live=True, native=TRAJECTORY_FORMAT),
        enable(ATTRIBUTION_PROBE_V1),
        enable(
            INSTALL_V1,
            install_commands=_INSTALL_COMMANDS,
            # Port the agent's A2A server listens on inside the container.
            a2a_port=8000,
            # Every placeholder referenced in install_commands must be in this list.
            optional_params=["control_token"],
            required_params=["container", "work_dir", "agent_ctx_tar", "a2a_port", *_ENDPOINT_PARAMS],
            install_commands_host=_INSTALL_COMMANDS_HOST,
            optional_params_host=["control_token"],
            required_params_host=["work_dir", "agent_ctx_tar", "a2a_port", "agent_name", *_ENDPOINT_PARAMS],
            workdir_host=_HOST_WORKDIR,
            user_host="root",
        ),
        # SDK default: the agent reaches the task's other agents through the SDK's peer_agents MCP tools.
        PEER_AGENTS_V1,
        enable(TRIGGERS_V1, description=(
            "Register a typed trigger script and decide the stakeholder reply per turn "
            "(step / env_trigger / conversational / all / any conditions -> say / end actions).")),
    ),
    lifespan=lifespan,
    workspace=str(WORKSPACE_DIR),
)
class AntigravityAgent(DurableSessionRefs, AgentEnvAgent):
    """Bridges A2A tasks onto the Antigravity SDK."""

    def __init__(self) -> None:
        self._turn_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._changelog: dict[str, Any] | None = None
        self._changelog_lock = asyncio.Lock()
        self._workspace = _WorkspaceBarrier()

    def _turn_lock(self, context_id: str) -> asyncio.Lock:
        return self._turn_locks.setdefault(context_id, asyncio.Lock())

    async def run(self, request: TaskRequest[AntigravityConfig]) -> TaskResult:
        request.trajectory.set_format(TRAJECTORY_FORMAT)
        prompt = "\n".join(part.text for part in request.parts if isinstance(part, TextPart) and part.text)
        context_id = request.context_id
        task = asyncio.current_task()
        if task is not None:
            _RUNNING.add(task)
        try:
            async with self._workspace.shared(), self._turn_lock(context_id):
                # Read under the lock rather than from the request: a snapshot load that held it may have just bound
                # this context to a restored conversation.
                session_id = self.session_ref_for_context(context_id)
                try:
                    attachments = []
                    for part in request.parts:
                        if isinstance(part, FilePart):
                            attachments.extend(await asyncio.to_thread(_materialize_and_render, part, request.task_id))
                except FilePartMaterializeError as exc:
                    return TaskResult.failure("filepart_error", str(exc), error_type="infra_error")
                try:
                    result = await run_antigravity(
                        prompt,
                        model=request.config.model,
                        project_id=request.config.project_id,
                        task_id=request.config.task_id,
                        effort=request.config.effort,
                        attachment_paths=attachments,
                        session_id=session_id,
                        mcp_servers=dict(request.mcp_servers),
                        system_prompt=request.config.system_prompt,
                        timeout_s=request.config.timeout_seconds,
                        role=request.config.role,
                        builtin_tools=request.config.builtin_tools,
                        web_tools=request.config.web_tools,
                        max_turns=request.config.max_turns,
                        workspace=str(WORKSPACE_DIR),
                        event_callback=request.trajectory.append,
                    )
                except Exception:
                    logger.exception("Antigravity invocation failed")
                    return TaskResult.failure("antigravity_invocation_error", "Antigravity invocation failed",
                                              error_type="infra_error")
                if result.session_id:
                    self.set_session_ref_for_context(context_id, result.session_id)
                await self._capture_changelog()
        finally:
            if task is not None:
                _RUNNING.discard(task)

        usage = result.usage or Usage(tool_call_count=result.tool_call_count)
        parts = [TextPart(text=result.response or result.error or "")]
        if result.error:
            message = result.error[:ERROR_MESSAGE_LIMIT]
            parts.append(DataPart(data={"error_type": "antigravity_error", "error_code": result.error_code,
                                        "error_message": message}))
            return (
                TaskResult.builder()
                .failed(result.error_code or "antigravity_error", message,
                        error_type="infra_error" if result.infra else "agent_error")
                .parts(parts)
                .session_ref(result.session_id)
                .usage(usage)
                .build()
            )
        return TaskResult.builder().succeeded().parts(parts).session_ref(result.session_id).usage(usage).build()

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(ATTRIBUTION_PROBE_V1.probe)
    async def probe_attribution(self) -> AttributionProbeResponse:
        """The LiteLLM attribution the last turn's model calls carried."""
        snapshot = get_last_attribution_snapshot()
        return AttributionProbeResponse(
            last_seen_attribution={
                key: value for key, value in snapshot.items() if key != "last_seen_at_utc" and isinstance(value, str)
            },
            last_seen_at_utc=snapshot.get("last_seen_at_utc"),
        )

    @extension(SNAPSHOT_V1.save)
    async def save_snapshot(self, request: ObjectSnapshotSaveRequest) -> dict[str, Any]:
        """The context's conversation, its SDK store whole with the env times its turns were given and the images their
        messages carry, as they were given, and the workspace."""
        with tempfile.TemporaryDirectory(prefix="antigravity-snapshot-") as directory:
            conversation = Path(directory) / "conversation.tgz"
            workspace = Path(directory) / "workspace.tgz"
            async with self._workspace.exclusive(), self._turn_lock(request.context_id):
                conversation_id = self.session_ref_for_context(request.context_id)
                if not conversation_id or not (SAVE_DIR / f"{conversation_id}.db").is_file():
                    raise HTTPException(status_code=404, detail="No active conversation for context_id")
                await asyncio.to_thread(_archive_conversation, conversation_id, conversation)
                if request.objects.workspace is not None:
                    await asyncio.to_thread(
                        subprocess.run,
                        ["tar", *_backup_exclusions(), "-czf", str(workspace), "-C", str(WORKSPACE_DIR), "."],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    )
            objects = {"trajectory": (await upload(request.objects.trajectory, conversation)).model_dump(mode="json")}
            if request.objects.workspace is not None:
                objects["workspace"] = (await upload(request.objects.workspace, workspace)).model_dump(mode="json")
        return {"context_id": request.context_id, "objects": objects}

    @extension(SNAPSHOT_V1.load)
    async def load_snapshot(self, request: ObjectSnapshotLoadRequest) -> dict[str, Any]:
        """The conversation under its own id, which the SDK keys its store by and holds inside it, bound to the target
        context. A conversation another context of this agent is bound to is refused: the two would share one history,
        and the load would roll the other's back."""
        target_context_id = request.target_context_id or uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="antigravity-restore-") as directory:
            archive = Path(directory) / "conversation.tgz"
            await download(request.objects.trajectory, archive)
            staged_conversation = Path(directory) / "conversation"
            staged_conversation.mkdir()
            conversation_id = await asyncio.to_thread(_unpack_conversation, archive, staged_conversation)
            staged = None
            if request.objects.workspace is not None:
                workspace_archive = Path(directory) / "workspace.tgz"
                staged = Path(directory) / "workspace"
                staged.mkdir()
                await download(request.objects.workspace, workspace_archive)
                await asyncio.to_thread(
                    subprocess.run, ["tar", "-xzf", str(workspace_archive), "-C", str(staged)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
            async with self._workspace.exclusive(), self._turn_lock(target_context_id):
                if [context for context in self.contexts_bound_to(conversation_id) if context != target_context_id]:
                    raise HTTPException(status_code=409, detail="the snapshot's conversation is another context's on "
                                        "this agent: load it into another agent")
                await asyncio.to_thread(_restore_conversation, staged_conversation, conversation_id)
                if staged is not None:
                    await asyncio.to_thread(_replace_workspace, staged)
                self.set_session_ref_for_context(target_context_id, conversation_id)
        return {"context_id": target_context_id}

    async def _capture_changelog(self) -> None:
        """One incremental tar of the changelog's roots, uploaded after each turn, with the roots removed since the
        last; the baseline advances only once the upload succeeds, so a failed capture's changes, and its removals, go
        with the next."""
        async with self._changelog_lock:
            config = self._changelog
            if config is None:
                return
            roots = [root for root in config["roots"] if Path(root).exists()]
            removed = [root for root in config["present"] if root not in roots]
            if not roots and not removed:
                return
            sequence = config["sequence"]
            _CHANGELOG_DIR.mkdir(parents=True, exist_ok=True)
            archive = _CHANGELOG_DIR / f"{sequence:06d}.tgz"
            pending_snar = _CHANGELOG_DIR / "pending.snar"
            record = _CHANGELOG_META_DIR / f"{sequence:06d}"
            try:
                await asyncio.to_thread(shutil.rmtree, record, True)
                record.mkdir(parents=True)
                (record / _META_NAME).write_text(json.dumps({"sequence": sequence, "removed": removed}))
                await asyncio.to_thread(shutil.copyfile, _CHANGELOG_SNAR, pending_snar)
                await asyncio.to_thread(
                    subprocess.run,
                    ["tar", "--ignore-failed-read", f"--listed-incremental={pending_snar}",
                     *_changelog_backup_exclusions(), f"--exclude={_CHANGELOG_DIR}", "-czf", str(archive), *roots,
                     str(record)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=_real_time_env(),
                )
                await config["uploader"].upload(f"{sequence:06d}.tgz", archive)
                pending_snar.replace(_CHANGELOG_SNAR)
                config["present"] = roots
                config["sequence"] += 1
            except Exception:
                logger.exception("Antigravity changelog capture failed")
            finally:
                archive.unlink(missing_ok=True)
                pending_snar.unlink(missing_ok=True)
                await asyncio.to_thread(shutil.rmtree, record, True)


class AntigravityChangelogAgent(AntigravityAgent):
    """With snapshot/v1's changelog, which needs GNU tar's incremental archives. The SDK runs its tools inside its
    harness with no hook between them this agent can hold a turn at, so an increment is a completed turn's changes;
    conversation state lives outside the captured roots, so apply restores files only."""

    @extension(SNAPSHOT_V1.changelog.enable)
    async def enable_changelog(self, request: NamespaceChangelogEnableRequest) -> dict[str, Any]:
        roots = request.roots or [str(WORKSPACE_DIR)]
        async with self._workspace.exclusive(), self._changelog_lock:
            await asyncio.to_thread(_baseline_changelog, roots)
            present = [root for root in roots if Path(root).exists()]
            self._changelog = {"roots": roots, "sequence": 0, "present": present,
                               "uploader": NamespaceUploader(request.write_namespace)}
        return {"roots": roots}

    @extension(SNAPSHOT_V1.changelog.apply)
    async def apply_changelog(self, request: ObjectChangelogApplyRequest) -> dict[str, Any]:
        async with self._workspace.exclusive():
            with tempfile.TemporaryDirectory(prefix="antigravity-changelog-apply-") as directory:
                for increment in request.increments:
                    archive = Path(directory) / f"{increment.sequence:06d}.tgz"
                    await download(increment.object, archive)
                    await asyncio.to_thread(_apply_increment, archive, increment.sequence)
        return {"count": len(request.increments), "context_id": request.target_context_id}


def _conversation_files(conversation_id: str) -> list[Path]:
    """The SDK's store for a conversation: its SQLite database and the journal files beside it."""
    return sorted(path for path in SAVE_DIR.glob(f"{conversation_id}.db*") if path.is_file())


def _archive_conversation(conversation_id: str, archive: Path) -> None:
    """The conversation's store, and its own records and images alone: the agent's other contexts' never go with it."""
    filed = CONVERSATION_RECORDS.get(conversation_id) or {}
    times = {key: value for key in filed.get("env_times") or [] if (value := ENV_TIMES.get(key)) is not None}
    images = {key: value for key in filed.get("turn_images") or [] if (value := TURN_IMAGES.get(key)) is not None}
    with tempfile.TemporaryDirectory(prefix="antigravity-conversation-") as directory:
        (Path(directory) / _MANIFEST).write_text(json.dumps({"conversation_id": conversation_id}))
        (Path(directory) / _ENV_TIMES_MEMBER).write_text(json.dumps(times))
        (Path(directory) / _TURN_IMAGES_MEMBER).write_text(json.dumps(images))
        with tarfile.open(archive, "w:gz") as tar:
            for member in (_MANIFEST, _ENV_TIMES_MEMBER, _TURN_IMAGES_MEMBER):
                tar.add(Path(directory) / member, arcname=member)
            kept = {name for names in images.values() if isinstance(names, list) for name in names}
            for name in sorted(kept):
                if isinstance(name, str) and (IMAGES_DIR / name).is_file():
                    tar.add(IMAGES_DIR / name, arcname=f"{_IMAGES_MEMBER}/{name}")
            for path in _conversation_files(conversation_id):
                tar.add(path, arcname=f"store/{path.name}")


def _unpack_conversation(archive: Path, staged: Path) -> str:
    """Extract a snapshot's conversation into ``staged``, regular files only and none outside it; its conversation
    id."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staged, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise HTTPException(status_code=400, detail="snapshot's conversation isn't an archive this agent reads") \
            from exc
    try:
        conversation_id = json.loads((staged / _MANIFEST).read_text())["conversation_id"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="snapshot has no conversation manifest") from exc
    if not isinstance(conversation_id, str) or Path(conversation_id).name != conversation_id or not conversation_id:
        raise HTTPException(status_code=400, detail="snapshot names no valid conversation id")
    if not (staged / "store" / f"{conversation_id}.db").is_file():
        raise HTTPException(status_code=400, detail="snapshot has no store for its conversation")
    return conversation_id


def _restore_conversation(staged: Path, conversation_id: str) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for path in _conversation_files(conversation_id):
        path.unlink()
    for path in sorted((staged / "store").iterdir()):
        if path.is_file() and path.name.startswith(f"{conversation_id}.db"):
            shutil.copyfile(path, SAVE_DIR / path.name)
    keys: dict[str, set[str]] = {}
    for member, records in ((_ENV_TIMES_MEMBER, ENV_TIMES), (_TURN_IMAGES_MEMBER, TURN_IMAGES)):
        try:
            restored = json.loads((staged / member).read_text())
        except (OSError, ValueError):
            restored = {}
        if isinstance(restored, dict):
            records.merge({str(key): value for key, value in restored.items()})
            keys[member] = {str(key) for key in restored}
    file_records(conversation_id, keys.get(_ENV_TIMES_MEMBER, set()), keys.get(_TURN_IMAGES_MEMBER, set()))
    if (staged / _IMAGES_MEMBER).is_dir():
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for path in (staged / _IMAGES_MEMBER).iterdir():
            if path.is_file() and not (IMAGES_DIR / path.name).exists():
                shutil.copyfile(path, IMAGES_DIR / path.name)


def _backup_exclusions() -> list[str]:
    if not WORKSPACE_DIR.exists():
        return []
    return [f"--exclude=./{path.name}" for path in WORKSPACE_DIR.iterdir()
            if path.is_dir() and path.name.startswith(".antigravity-backup-") and (path / "saved").is_dir()]


def _changelog_backup_exclusions() -> list[str]:
    root = WORKSPACE_DIR.relative_to("/")
    exclusions = _backup_exclusions()
    return exclusions + [f"--exclude={root / option.removeprefix('--exclude=./')}" for option in exclusions]


def _replace_workspace(staged: Path) -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=".antigravity-backup-", dir=WORKSPACE_DIR))
    saved = backup / "saved"
    saved.mkdir()
    committed = False
    try:
        try:
            for child in WORKSPACE_DIR.iterdir():
                if child != backup:
                    child.rename(saved / child.name)
        except Exception:
            for child in saved.iterdir():
                child.rename(WORKSPACE_DIR / child.name)
            raise
        try:
            for child in staged.iterdir():
                shutil.move(str(child), str(WORKSPACE_DIR / child.name))
            committed = True
        except Exception:
            for child in WORKSPACE_DIR.iterdir():
                if child == backup:
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            for child in saved.iterdir():
                child.rename(WORKSPACE_DIR / child.name)
            raise
    finally:
        if committed:
            try:
                shutil.rmtree(backup)
            except OSError:
                logger.warning("Could not remove snapshot backup %s", backup, exc_info=True)
        elif not any(saved.iterdir()):
            saved.rmdir()
            backup.rmdir()


def _apply_increment(archive: Path, sequence: int) -> None:
    """Extract an increment over /, all but its record, then remove the roots its record names."""
    listing = subprocess.run(["tar", "--incremental", "-tzf", str(archive)], check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE).stdout.decode(errors="replace").splitlines()
    records = [member for member in listing if _META_MEMBER.search(member)]
    excluded = {member.rsplit("/", 1)[0] for member in records}
    subprocess.run(
        ["tar", "--incremental", "--no-wildcards", *(f"--exclude={member}" for member in sorted(excluded)),
         "-xzf", str(archive), "-C", "/"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    for member in records:
        record = subprocess.run(["tar", "-xzOf", str(archive), member], check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).stdout
        try:
            removed = json.loads(record).get("removed")
        except (ValueError, AttributeError):
            logger.warning(f"changelog increment {sequence:06d}: unreadable record {member}")
            continue
        for path in removed if isinstance(removed, list) else []:
            if not _removable(path):
                logger.warning(f"changelog increment {sequence:06d}: not removing {path!r}")
                continue
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
            logger.info(f"changelog increment {sequence:06d}: removed root {path}")


def _removable(path: object) -> bool:
    """Whether a removed root a record names may be removed on replay: an absolute, normalized path that exists and
    isn't / or a top-level system directory."""
    return (isinstance(path, str) and os.path.isabs(path) and os.path.normpath(path) == path
            and path not in _SYSTEM_DIRS and os.path.lexists(path))


def _baseline_changelog(roots: list[str]) -> None:
    _CHANGELOG_DIR.mkdir(parents=True, exist_ok=True)
    _CHANGELOG_SNAR.unlink(missing_ok=True)
    present = [root for root in roots if Path(root).exists()]
    subprocess.run(
        ["tar", "--ignore-failed-read", f"--listed-incremental={_CHANGELOG_SNAR}", *_changelog_backup_exclusions(),
         "-cf", "/dev/null", *present],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=_real_time_env(),
    )


def _real_time_env() -> dict[str, str]:
    """This server's environment for the changelog's tar, which keeps the time it starts at as the next capture's
    baseline and archives only the files changed since, by their real mtimes, so it runs on real time: the server
    does, but libfaketime took the marker out of the server's environment, so what it starts doesn't."""
    return {**os.environ, "AGENTENV_REAL_TIME": "1"}


def gnu_tar() -> bool:
    """Whether this host's tar is GNU tar, which the changelog's incremental archives need."""
    try:
        version = subprocess.run(["tar", "--version"], capture_output=True, text=True,
                                 timeout=_TAR_VERSION_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "GNU tar" in version


def agent_for_this_host() -> AntigravityAgent:
    """The agent, with the changelog where this host's tar can make its archives."""
    return AntigravityChangelogAgent() if gnu_tar() else AntigravityAgent()


def build_app(agent: AntigravityAgent) -> Starlette:
    """The agent's Starlette app, answering errors with FastAPI's `{"detail": ...}` bodies."""
    return json_error_bodies(agent.create_app())


app = build_app(agent_for_this_host())
# agent-env's install/v1 deploys find the install parameters by importing this name from the image.
AGENT_CARD = app.state.agentenv_a2a.card
