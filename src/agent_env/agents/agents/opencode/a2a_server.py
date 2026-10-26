"""A2A agent wrapping the OpenCode CLI.

Built on agentenv-protocol's `AgentEnvAgent`: the framework serves the Agent Card,
the A2A task lifecycle and the SDK-owned extensions (agent-config, mcp-config,
skill listing, task trajectories, peer agents). Each turn is one
`opencode run --format json`, whose events are appended to the framework's
trajectory log as the CLI prints them. A context's first turn starts an OpenCode
session, which later turns resume with `-s <session_id>`.
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
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    AttributionProbeResponse,
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
from pydantic import Field, field_validator
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from opencode_bridge import (
    ERROR_MESSAGE_LIMIT,
    event_session_id,
    export_session,
    get_last_attribution_snapshot,
    import_session,
    kill_running_clis,
    run_opencode,
    validate_output_format,
)
from opencode_kit import DurableSessionRefs, install_skill, json_error_bodies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_XDG_CONFIG = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
SKILLS_DIR = Path(_XDG_CONFIG) / "opencode" / "skills"
WORKSPACE_DIR = Path(os.environ.get("OPENCODE_WORKSPACE_DIR", "/app"))
INPUTS_DIR = WORKSPACE_DIR / "inputs"
# One per port, since agents installed on one host listen on their own.
_CHANGELOG_DIR = Path(tempfile.gettempdir()) / f"opencode-changelog-{os.environ.get('A2A_PORT') or 'default'}"
_CHANGELOG_SNAR = _CHANGELOG_DIR / "state.snar"
# Each increment's record of the roots it saw removed, in a directory of its own beside the changelog, which captures
# exclude, and outside its roots: tar dumps a directory it hasn't seen whole, so the record goes in however soon after
# the last capture it is written.
_META_DIR_SUFFIX = ".changelog-meta"
_META_NAME = "opencode-changelog-meta.json"
_CHANGELOG_META_DIR = _CHANGELOG_DIR.parent / f"{_CHANGELOG_DIR.name}{_META_DIR_SUFFIX}"
# A record in an increment: _META_NAME in a sequence's directory under a directory named with _META_DIR_SUFFIX,
# wherever the capturing agent kept it; a file of that name anywhere else is a file of the layer.
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
# The versioned format of the `opencode run --format json` events this agent appends to its trajectory.
TRAJECTORY_FORMAT = "opencode-events/1"


def _backup_exclusions() -> list[str]:
    if not WORKSPACE_DIR.exists():
        return []
    return [
        f"--exclude=./{path.name}"
        for path in WORKSPACE_DIR.iterdir()
        if path.is_dir()
        and path.name.startswith(".opencode-backup-")
        and (path / "saved").is_dir()
    ]


def _changelog_backup_exclusions() -> list[str]:
    root = WORKSPACE_DIR.relative_to("/")
    exclusions = _backup_exclusions()
    return exclusions + [
        f"--exclude={root / option.removeprefix('--exclude=./')}"
        for option in exclusions
    ]


class OpenCodeConfig(AgentConfig):
    model: str | None = None
    effort: str | None = None
    system_prompt: str | None = None
    output_format: dict[str, Any] | None = None
    project_id: str | None = None
    task_id: str | None = None
    timeout_seconds: int = Field(default=1800, gt=0)

    @field_validator("output_format")
    @classmethod
    def check_output_format(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None:
            validate_output_format(value)
        return value


class FilePartMaterializeError(Exception):
    """A supplied file cannot be passed to OpenCode."""


def _safe_name(candidate: str | None, fallback_ext: str = "") -> str:
    if candidate:
        candidate = _SAFE_NAME_RE.sub("_", Path(candidate).name)
    return candidate or f"{uuid.uuid4().hex}{fallback_ext}"


def _download_uri_to(uri: str, target: Path) -> None:
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        try:
            with httpx.Client(timeout=_FILE_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
                response = client.get(uri)
                response.raise_for_status()
                target.write_bytes(response.content)
        except httpx.HTTPError as exc:
            raise FilePartMaterializeError("HTTP file download failed") from exc
    else:
        raise FilePartMaterializeError(f"unsupported file URI scheme {parsed.scheme!r}")


def _materialize_and_render(part: FilePart, task_id: str) -> list[str]:
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
    # The openai-compatible provider can't forward PDF document blocks, so a PDF goes as its pages' images.
    prefix = str(target.with_suffix("")) + "_page"
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(_PDF_RENDER_DPI), str(target), prefix],
            check=True, capture_output=True, timeout=_PDF_RENDER_TIMEOUT_S,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("PDF rendering failed for %s; passing the PDF directly", target)
        return [str(target)]
    pages = sorted(directory.glob(f"{Path(prefix).name}*.png"))
    return [str(page) for page in pages] or [str(target)]


class _WorkspaceBarrier:
    """Turns share the workspace and snapshots take it alone, so a snapshot's session and files come from a point no
    turn is writing at, while turns of different contexts run at once. A waiting snapshot holds new turns back, so
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


@asynccontextmanager
async def lifespan(app: Starlette):
    """As the server stops, the CLIs of turns still running are killed, which would otherwise go on in the workspace
    after it."""
    try:
        yield
    finally:
        if stopped := kill_running_clis():
            logger.info(f"Killed {stopped} opencode CLI(s) still running a turn as the agent shut down")


# install.sh, from the build context install_agent unpacks, brings Node, the OpenCode CLI and a Python of the agent's
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
        name="OpenCode",
        description="A2A-compliant wrapper around the OpenCode CLI.",
        version="1.0.0",
        input_modes=("text", "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"),
        output_modes=("text",),
        skills=(AgentSkill(
            id="opencode", name="opencode",
            description="Runs the OpenCode CLI with full tool access (bash, read, edit, glob, grep).",
            tags=["coding", "agent"],
        ),),
    ),
    config=OpenCodeConfig,
    extensions=(
        MCP_CONFIG_V1,
        enable(TRAJECTORY_V1, live=True),
        enable(ATTRIBUTION_PROBE_V1),
        enable(
            INSTALL_V1,
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
        # SDK default: the CLI reaches the task's other agents through the SDK's peer_agents MCP tools.
        PEER_AGENTS_V1,
    ),
    lifespan=lifespan,
    workspace=str(WORKSPACE_DIR),
)
class OpenCodeAgent(DurableSessionRefs, AgentEnvAgent):
    """Bridges A2A tasks onto `opencode run`."""

    def __init__(self) -> None:
        self._turn_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._changelog: dict[str, Any] | None = None
        self._changelog_lock = asyncio.Lock()
        self._workspace = _WorkspaceBarrier()

    def _turn_lock(self, context_id: str) -> asyncio.Lock:
        return self._turn_locks.setdefault(context_id, asyncio.Lock())

    async def run(self, request: TaskRequest[OpenCodeConfig]) -> TaskResult:
        # The CLI's JSON events, appended as it prints them, are the turn's
        # trajectory: readable while it runs, and its final record once it ends.
        request.trajectory.set_format(TRAJECTORY_FORMAT)
        prompt = "\n".join(part.text for part in request.parts if isinstance(part, TextPart) and part.text)
        context_id = request.context_id
        async with self._workspace.shared(), self._turn_lock(context_id):
            # Read under the lock rather than from the request: a snapshot load that held it may have just bound this
            # context to an imported session.
            session_id = self.session_ref_for_context(context_id)

            def record(event: dict) -> None:
                request.trajectory.append(event)
                # A first turn's session is bound as soon as the CLI names it, so a cancel can't lose it.
                if self.session_ref_for_context(context_id) is None and (started := event_session_id(event)):
                    self.set_session_ref_for_context(context_id, started)
            try:
                attachments = []
                for part in request.parts:
                    if isinstance(part, FilePart):
                        attachments.extend(await asyncio.to_thread(_materialize_and_render, part, request.task_id))
            except FilePartMaterializeError as exc:
                return TaskResult.failure("filepart_error", str(exc), error_type="infra_error")
            try:
                result = await run_opencode(
                    prompt=prompt,
                    context_id=context_id,
                    session_id=session_id,
                    model=request.config.model,
                    effort=request.config.effort,
                    mcp_servers=dict(request.mcp_servers),
                    system_prompt=request.config.system_prompt,
                    output_format=request.config.output_format,
                    timeout_s=request.config.timeout_seconds,
                    project_id=request.config.project_id,
                    task_id=request.config.task_id,
                    role=request.config.role,
                    attachment_paths=attachments,
                    event_callback=record,
                    workspace=str(WORKSPACE_DIR),
                )
            except Exception:
                logger.exception("OpenCode invocation failed")
                return TaskResult.failure(
                    "opencode_invocation_error", "OpenCode invocation failed", error_type="infra_error"
                )
            if result.session_id:
                self.set_session_ref_for_context(context_id, result.session_id)
            await self._capture_changelog()

        parts = [TextPart(text=result.response or result.error or "")]
        usage = Usage(tool_call_count=result.tool_call_count)
        if result.error:
            details = {"error_type": "cli_error", "error_message": result.error[:ERROR_MESSAGE_LIMIT]}
            if result.error_code:
                details["error_code"] = result.error_code
            parts.append(DataPart(data=details))
            return (
                TaskResult.builder()
                .failed(result.error_code or "opencode_cli_error", result.error[:ERROR_MESSAGE_LIMIT])
                .parts(parts)
                .session_ref(result.session_id)
                .usage(usage)
                .build()
            )
        response = (
            TaskResult.builder()
            .succeeded()
            .parts(parts)
            .session_ref(result.session_id)
            .usage(usage)
        )
        if request.config.output_format is not None:
            response.add_structured_output(getattr(result, "structured_output", None))
        return response.build()

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> SkillAddResponse:
        return await install_skill(request, SKILLS_DIR)

    @extension(ATTRIBUTION_PROBE_V1.probe)
    async def probe_attribution(self) -> AttributionProbeResponse:
        """The LiteLLM attribution the last CLI run sent with its model calls."""
        snapshot = get_last_attribution_snapshot()
        return AttributionProbeResponse(
            last_seen_attribution={
                key: value for key, value in snapshot.items() if key != "last_seen_at_utc" and isinstance(value, str)
            },
            last_seen_at_utc=snapshot.get("last_seen_at_utc"),
        )

    @extension(SNAPSHOT_V1.save)
    async def save_snapshot(self, request: ObjectSnapshotSaveRequest) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="opencode-snapshot-") as directory:
            workspace = Path(directory) / "workspace.tgz"
            async with self._workspace.exclusive(), self._turn_lock(request.context_id):
                # Read once no turn runs: a context's first turn, or a load into it, may be about to bind it.
                session_id = self.session_ref_for_context(request.context_id)
                if not session_id:
                    raise HTTPException(status_code=404, detail="No active session for context_id")
                session_json = await export_session(request.context_id, session_id)
                if request.objects.workspace is not None:
                    await asyncio.to_thread(
                        subprocess.run,
                        ["tar", *_backup_exclusions(), "-czf", str(workspace),
                         "-C", str(WORKSPACE_DIR), "."],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    )
            objects = {"trajectory": (await upload(request.objects.trajectory, session_json)).model_dump(mode="json")}
            if request.objects.workspace is not None:
                objects["workspace"] = (await upload(request.objects.workspace, workspace)).model_dump(mode="json")
        return {"context_id": request.context_id, "objects": objects}

    @extension(SNAPSHOT_V1.load)
    async def load_snapshot(self, request: ObjectSnapshotLoadRequest) -> dict[str, Any]:
        target_context_id = request.target_context_id or uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="opencode-restore-") as directory:
            session_file = Path(directory) / "session.json"
            await download(request.objects.trajectory, session_file)
            staged = None
            if request.objects.workspace is not None:
                archive = Path(directory) / "workspace.tgz"
                staged = Path(directory) / "workspace"
                staged.mkdir()
                await download(request.objects.workspace, archive)
                await asyncio.to_thread(
                    subprocess.run, ["tar", "-xzf", str(archive), "-C", str(staged)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
            async with self._workspace.exclusive(), self._turn_lock(target_context_id):
                imported_session_id = await import_session(target_context_id, str(session_file))
                if not imported_session_id:
                    raise HTTPException(status_code=500, detail="OpenCode import returned no session id")
                if staged is not None:
                    await asyncio.to_thread(_replace_workspace, staged)
                self.set_session_ref_for_context(target_context_id, imported_session_id)
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
                    ["tar", f"--listed-incremental={pending_snar}",
                     *_changelog_backup_exclusions(),
                     f"--exclude={_CHANGELOG_DIR}", "-czf", str(archive), *roots, str(record)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                await config["uploader"].upload(f"{sequence:06d}.tgz", archive)
                pending_snar.replace(_CHANGELOG_SNAR)
                config["present"] = roots
                config["sequence"] += 1
            except Exception:
                logger.exception("OpenCode changelog capture failed")
            finally:
                archive.unlink(missing_ok=True)
                pending_snar.unlink(missing_ok=True)
                await asyncio.to_thread(shutil.rmtree, record, True)


class OpenCodeChangelogAgent(OpenCodeAgent):
    """With snapshot/v1's changelog, which needs GNU tar's incremental archives. OpenCode has no hook after each tool
    call, so an increment is a completed turn's changes; conversation state lives outside the captured roots, so
    apply restores files only."""

    @extension(SNAPSHOT_V1.changelog.enable)
    async def enable_changelog(self, request: NamespaceChangelogEnableRequest) -> dict[str, Any]:
        roots = request.roots or [str(WORKSPACE_DIR)]
        async with self._workspace.exclusive(), self._changelog_lock:
            await asyncio.to_thread(_baseline_changelog, roots)
            self._changelog = {"roots": roots, "sequence": 0, "present": [root for root in roots if Path(root).exists()],
                               "uploader": NamespaceUploader(request.write_namespace)}
        return {"roots": roots}

    @extension(SNAPSHOT_V1.changelog.apply)
    async def apply_changelog(self, request: ObjectChangelogApplyRequest) -> dict[str, Any]:
        async with self._workspace.exclusive():
            with tempfile.TemporaryDirectory(prefix="opencode-changelog-apply-") as directory:
                for increment in request.increments:
                    archive = Path(directory) / f"{increment.sequence:06d}.tgz"
                    await download(increment.object, archive)
                    await asyncio.to_thread(_apply_increment, archive, increment.sequence)
        return {"count": len(request.increments), "context_id": request.target_context_id}


def _replace_workspace(staged: Path) -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=".opencode-backup-", dir=WORKSPACE_DIR))
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
    # The record's directory is the capture's, not part of the layer it restores.
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
        ["tar", f"--listed-incremental={_CHANGELOG_SNAR}",
         *_changelog_backup_exclusions(),
         "-cf", "/dev/null", *present],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def gnu_tar() -> bool:
    """Whether this host's tar is GNU tar, which the changelog's incremental archives need."""
    try:
        version = subprocess.run(["tar", "--version"], capture_output=True, text=True,
                                 timeout=_TAR_VERSION_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "GNU tar" in version


def agent_for_this_host() -> OpenCodeAgent:
    """The agent, with the changelog where this host's tar can make its archives."""
    return OpenCodeChangelogAgent() if gnu_tar() else OpenCodeAgent()


def build_app(agent: OpenCodeAgent) -> Starlette:
    """The agent's Starlette app, answering errors with FastAPI's `{"detail": ...}` bodies."""
    return json_error_bodies(agent.create_app())


app = build_app(agent_for_this_host())
# agent-env's install/v1 deploys find the install parameters by importing this name from the image.
AGENT_CARD = app.state.agentenv_a2a.card
